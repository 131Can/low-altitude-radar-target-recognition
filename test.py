import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import xgboost as xgboost
import pandas as pd
import scipy.stats as stats
import numpy as np
import os
import struct
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# 常量定义
FRAME_HEAD = 0xFA55FA55
FRAME_END = 0x55FA55FA
c = 3e8  # 光速
###################################原始回波数据处理############################
def parse_radar_dat(file_path):
    #解析雷达原始回波.dat文件
    frames = []
     
    with open(file_path, 'rb') as fid:
        while True:
            try:
                head_find = struct.unpack('<I', fid.read(4))[0]
                while head_find != FRAME_HEAD:
                    fid.seek(-3, os.SEEK_CUR)  # 回退3字节
                    head_find = struct.unpack('<I', fid.read(4))[0]
                
                frame_data_length = struct.unpack('<I', fid.read(4))[0] * 4
                
                fid.seek(frame_data_length - 12, os.SEEK_CUR)
                end_find = struct.unpack('<I', fid.read(4))[0]
                
                if end_find != FRAME_END:
                    fid.seek(-frame_data_length + 1, os.SEEK_CUR)
                    continue
                
                fid.seek(-frame_data_length + 4, os.SEEK_CUR)
                
                data_temp1 = struct.unpack('<III', fid.read(12))
                para = {
                    'E_scan_Az': data_temp1[1] * 0.01,
                    'pointNum_in_bowei': data_temp1[2]
                }
                
                # 读取航迹信息
                track_info_size = para['pointNum_in_bowei'] * 4
                track_info_data = fid.read(track_info_size * 4)
                para['Track_No_info'] = np.frombuffer(track_info_data, dtype='<I')
                
                # 读取雷达参数
                radar_params = struct.unpack('<IIIII', fid.read(20))
                para['Freq'] = radar_params[0] * 1e6
                para['CPIcount'] = radar_params[1]
                para['PRTnum'] = radar_params[2]
                para['PRT'] = radar_params[3] * 0.0125e-6
                para['data_length'] = radar_params[4]
                
                # 读取IQ数据
                iq_size = para['PRTnum'] * 31 * 8  # 31×PRTnum×2×4字节
                iq_data = np.frombuffer(fid.read(iq_size), dtype='<f4')
                
                data_out_real = iq_data[0::2]
                data_out_imag = iq_data[1::2]
                data_out_complex = data_out_real + 1j * data_out_imag
                data_out = data_out_complex.reshape(31, para['PRTnum'])
        
                fid.read(4)
                frames.append({
                    'para': para,
                    'data_out': data_out
                })
                
            except struct.error:
                break
    
    return frames

def extract_features(frame, track_id):
    #提取结构化特征
    para = frame['para']
    data_out = frame['data_out']
    features = {}
    
    #基础参数特征
    features['id'] = track_id
    features['PRT'] = para['PRT']
    features['PRTnum'] = para['PRTnum']
    features['Freq'] = para['Freq']
    
    #目标信息特征
    if len(para['Track_No_info']) >= 4:
        features['number'] = para['Track_No_info'][1]
        features['target_bin'] = para['Track_No_info'][2]
        features['doppler_bin'] = para['Track_No_info'][3]
    else:
        features['target_bin'] = 15
        features['doppler_bin'] = 0
    
    center_row = 15  # 中心行索引
    target_radius = 1  # 目标区域半径
    background_radius = 3  # 背景区域半径
    
    #目标区域
    target_rows = range(max(0, center_row-target_radius), 
                      min(31, center_row+target_radius+1))
    
    #背景区域
    background_rows = list(range(max(0, center_row-background_radius), 
                             center_row-target_radius)) + \
                    list(range(center_row+target_radius+1, 
                             min(31, center_row+background_radius+1)))
    
    #目标区域特征
    target_signals = data_out[target_rows, :]
    target_amps = np.abs(target_signals)
    
    features['target_amp_mean'] = np.mean(target_amps)
    features['target_amp_std'] = np.std(target_amps)
    features['target_amp_max'] = np.max(target_amps)
    features['target_amp_min'] = np.min(target_amps)
    
    target_phases = np.angle(target_signals)
    features['target_phase_var'] = np.var(np.diff(target_phases, axis=1))
    
    #背景区域特征
    if background_rows:
        bg_signals = data_out[background_rows, :]
        bg_amps = np.abs(bg_signals)
        
        features['bg_amp_mean'] = np.mean(bg_amps)
        features['bg_amp_std'] = np.std(bg_amps)
        features['bg_amp_max'] = np.max(bg_amps)
        features['bg_amp_min'] = np.min(bg_amps)
        
        features['target_bg_ratio'] = features['target_amp_mean'] / (features['bg_amp_mean'] + 1e-9)
        features['target_bg_diff'] = features['target_amp_mean'] - features['bg_amp_mean']
    else:
       
        features.update({
            'bg_amp_mean': 0,
            'bg_amp_std': 0,
            'bg_amp_max': 0,
            'bg_amp_min': 0,
            'target_bg_ratio': 1,
            'target_bg_diff': 0
        })
    
    #微多普勒特征
    center_signal = data_out[center_row, :]
    fft_spec = np.fft.fftshift(np.fft.fft(center_signal))
    peaks, _ = find_peaks(np.abs(fft_spec), prominence=0.1)
    
    features['harmonic_count'] = len(peaks)
    features['dominant_freq'] = np.argmax(np.abs(fft_spec)) - len(fft_spec)//2
    
    #空间分布特征
    full_amp = np.abs(data_out)
    features['spatial_var'] = np.var(full_amp)  
    features['spatial_gradient'] = np.mean(np.abs(np.gradient(full_amp, axis=0)))  
    
    #目标聚焦度特征
    center_amp = np.abs(data_out[center_row, :])
    features['focus_index'] = np.max(center_amp) / (np.mean(full_amp) + 1e-9)
    
    #背景杂波特征
    if background_rows:
        bg_amp_flat = np.abs(data_out[background_rows, :]).flatten()
        features['bg_skewness'] = stats.skew(bg_amp_flat)
        features['bg_kurtosis'] = stats.kurtosis(bg_amp_flat)
    else:
        features['bg_skewness'] = 0
        features['bg_kurtosis'] = 0
        
    #距离单元特征
    for i in range(0, 31, 5):  
        cell_signal = data_out[i, :]
        features[f'cell{i}_mean'] = np.mean(np.abs(cell_signal))
        features[f'cell{i}_kurt'] = stats.kurtosis(np.abs(cell_signal))
    
    return features


def aggregate_features(features_df):
    #对提取的特征进行时间维度上的聚合
    grouped = features_df.groupby(['id', 'number'])
    
    aggregation_functions = {
    #基础特征
    'PRT': 'first',          
    'PRTnum': 'first',       
    'Freq': 'first',         
    'target_bin': 'first',  
    'doppler_bin': 'first',  

    #目标区域特征
    'target_amp_mean': ['mean', 'std', 'max', 'min'],  
    'target_amp_std': ['mean', 'std', 'max'],  
    'target_amp_max': ['mean', 'std', 'max'],  
    'target_amp_min': ['mean', 'std'],        
    'target_phase_var': ['mean', 'std', 'max'],  

    #背景区域特征
    'bg_amp_mean': ['mean', 'std', 'max'],  
    'bg_amp_std': ['mean', 'std', 'max'],     
    'bg_amp_max': ['mean', 'max'],             
    'bg_amp_min': ['mean'],                    
    
    #目标-背景对比特征
    'target_bg_ratio': ['mean', 'std', 'max', 'min'],  
    'target_bg_diff': ['mean', 'std', 'max'], 
    
    #空间分布特征
    'spatial_var': ['mean', 'std', 'max'],    
    'spatial_gradient': ['mean', 'std'],       
    
    #微多普勒特征
    'harmonic_count': ['mean', 'max', 'min'], 
    'dominant_freq': ['mean', 'std'],    
    
    #杂波统计特征
    'bg_skewness': ['mean', 'std'],           
    'bg_kurtosis': ['mean', 'std', 'max'],   
    
    #高级复合特征
    'focus_index': ['mean', 'std', 'max']
}
    

    for col in features_df.columns:
        if col.startswith('cell'):
            if col.endswith('_mean'):
                aggregation_functions[col] = ['mean', 'std']
            elif col.endswith('_kurt'):
                aggregation_functions[col] = ['mean', 'std', 'max']
    

    aggregated_df = grouped.agg(aggregation_functions)
    
    aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]
    
    frame_counts = grouped.size().rename('frame_count')
    aggregated_df = aggregated_df.join(frame_counts)
    
    aggregated_df['target_range'] = aggregated_df['target_bin_first'] * 7.5  
    
    if 'dominant_freq_mean' in aggregated_df.columns:
        aggregated_df['freq_stability'] = aggregated_df['dominant_freq_std'] / aggregated_df['frame_count']
    
    return aggregated_df.reset_index()


def process_radar_data(dat_folder, output_csv=None):
    all_features = []
    
    # 遍历文件夹
    for filename in os.listdir(dat_folder):
        if filename.endswith('.dat'):
            file_path = os.path.join(dat_folder, filename)
            try:
                track_id = filename.split('.')[0]
                label = "unknown" 
            except:
                track_id = "error"
                label = "error"
            
            print(f"处理文件: {filename}, 航迹批号: {track_id}")
            frames = parse_radar_dat(file_path)
            print(f"  共解析 {len(frames)} 帧数据")
            for i, frame in enumerate(frames):
                try:
                    features = extract_features(frame, track_id)
                    all_features.append(features)
                except Exception as e:
                    print(f"  处理第 {i} 帧时出错: {str(e)}")
    
    if not all_features:
        print("没有提取到任何特征数据")
        return None
    
    feature_df = pd.DataFrame(all_features)
    feature_df=aggregate_features(feature_df)
    
    if output_csv:
        feature_df.to_csv(output_csv, index=False)
        print(f"所有特征已保存至 {output_csv}")
    
    return feature_df

################################数据写回处理###############################
def update_track_files(tracks_folder, result_df):
    # 遍历航迹文件夹
    for filename in os.listdir(tracks_folder):
        if filename.startswith('Tracks_') and filename.endswith('.txt'):
            try:
                parts = filename.split('_')
                track_id = int(parts[1])  
                
                track_results = result_df[result_df['id'] == track_id]
                if track_results.empty:
                    print(f'跳过文件 {filename}，未找到批号 {track_id} 对应的识别结果')
                    continue
                
                filepath = os.path.join(tracks_folder, filename)
                
                with open(filepath, 'r', encoding='gbk') as f:
                    lines = f.readlines()
                
                updated_lines = []
                headers = lines[0].strip().split(',') 
                headers.append('识别结果')  
                updated_lines.append(','.join(headers) + '\n')  
                
                if len(lines[1:]) != len(track_results):
                    print(f'警告: 文件 {filename} 的数据行数({len(lines[1:])})与批号 {track_id} 的结果数({len(track_results)})不匹配')
                
                for i, line in enumerate(lines[1:]):
                    line = line.strip()
                    if line:  
                        try:                          
                            predicted_class = track_results.iloc[i]['predicted_class']
                            updated_line = line + f',{predicted_class}\n'  
                            updated_lines.append(updated_line)
                        except IndexError:
                            print(f'警告: 文件 {filename} 的第 {i+1} 行没有对应的识别结果')
                            updated_line = line + ',\n'  
                            updated_lines.append(updated_line)

                with open(filepath, 'w', encoding='gbk') as f:
                    f.writelines(updated_lines)
                
                print(f'已更新文件: {filename}')
                
            except (IndexError, ValueError) as e:
                print(f'跳过文件 {filename}，文件名格式不正确: {e}')
                
################################训练数据预处理###############################
def transcolname(df, column_mapping):
    df.rename(columns=column_mapping, inplace=True)
    return df


def prepare(data, fit_scaler=None, transform_only=False):
    column_mapping = {
    # 基础探测信息 
    "点时间": "timestamp",
    "批号": "id",
    "距离": "range",
    "方位": "azimuth",
    "俯仰": "elevation",
    
    # 信号特征 
    "多普勒速度": "v_doppler",
    "和幅度": "amplitude",
    "信噪比": "SNR",
    "原始点数量": "raw_points",
    
    # 滤波后数据
    "滤波距离": "filtered_range",
    "滤波方位": "filtered_azimuth",
    "滤波俯仰": "filtered_elevation",
    
    # 运动状态 
    "全速度": "v_total",
    "X向速度": "v_x",
    "Y向速度": "v_y",
    "Z向速度": "v_z",
    "航向": "heading",
    
    
}
    data = transcolname(data, column_mapping)

    columns_to_drop = ['number']
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    
    return data

def Screening(data):
    #时间非递增筛选
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%H:%M:%S.%f')
    def is_increasing(group):
        diffs = group['timestamp'].diff().dropna()  
        return (diffs >= pd.Timedelta(0)).all()

    valid_ids = data.groupby('id').filter(is_increasing)['id'].unique()
    invalid_ids = set(data['id'].unique()) - set(valid_ids)
    data = data[data['id'].isin(valid_ids)]
    if not invalid_ids :
        print("无时间非递增异常数据")
    else:
        print("剔除时间非递增异常数据:",invalid_ids)
        
        
    #无原始回波筛选
    to_drop_ids = data.groupby('id')['PRTnum_first'].apply(lambda x: x.isna().all())
    to_drop_ids = to_drop_ids[to_drop_ids].index.tolist()
    data = data[~data['id'].isin(to_drop_ids)]
    if not to_drop_ids:
        print("无回波缺失异常数据")
    else:
        print("剔除回波缺失异常数据:",to_drop_ids)
           
    #NaN值筛选
    to_drop_ids = set()

    for id_, group in data.groupby('id'):
        last_two_rows = group.tail(2)  
        target_cols = last_two_rows.iloc[:, 10:18] 
    
        if (target_cols.isna().all(axis=1)).any():  
            to_drop_ids.add(id_)
    
    data = data[~data['id'].isin(to_drop_ids)]
    if not to_drop_ids:
        print("无出现NaN异常数据")
    else:
        print("剔除出现NaN异常数据:",to_drop_ids)

    return data
    

###############################特征工程#################################
def dot_trace_features(df, windows=[5, 10]):
    
    grouped = df.groupby("id")
    for window in windows:
        ########################速度###############################
        df[f'v_doppler_mean_{window}d'] = grouped['v_doppler'].transform(lambda x: x.rolling(window).mean())
        df[f'v_doppler_std_{window}d'] = grouped['v_doppler'].transform(lambda x: x.rolling(window).std())
        df[f'v_doppler_max_{window}d'] = grouped['v_doppler'].transform(lambda x: x.rolling(window).max())
        df[f'v_doppler_min_{window}d'] = grouped['v_doppler'].transform(lambda x: x.rolling(window).min())
        #速度变异系数
        df[f'v_doppler_cv_{window}d'] = np.where(df[f'v_doppler_mean_{window}d'] != 0
        ,df[f'v_doppler_std_{window}d'] / df[f'v_doppler_mean_{window}d'],np.nan)
        #加速度std
        df[f'v_doppler_accel_std_{window}d'] = grouped['v_doppler'].transform(
            lambda x: x.rolling(window).apply(lambda s: np.std(np.diff(s)) if len(s) >= 2 else np.nan ))
       
        
        df[f'v_total_mean_{window}d'] = grouped['v_total'].transform(lambda x: x.rolling(window).mean())
        df[f'v_total_std_{window}d'] = grouped['v_total'].transform(lambda x: x.rolling(window).std())
        df[f'v_total_max_{window}d'] = grouped['v_total'].transform(lambda x: x.rolling(window).max())
        df[f'v_total_min_{window}d'] = grouped['v_total'].transform(lambda x: x.rolling(window).min())
        #速度变异系数
        df[f'v_total_cv_{window}d'] = np.where(df[f'v_total_mean_{window}d'] != 0
        ,df[f'v_total_std_{window}d'] / df[f'v_total_mean_{window}d'],np.nan)
        #加速度std
        df[f'v_total_accel_std_{window}d'] = grouped['v_total'].transform(
            lambda x: x.rolling(window).apply(lambda s: np.std(np.diff(s)) if len(s) >= 2 else np.nan ))
       
        
        
        ########################SNR####################################
        # 分布偏斜度
        df[f'SNR_skew_{window}d'] = grouped['SNR'].transform(lambda x: x.rolling(window).apply(stats.skew))
        df[f'SNR_range_{window}d'] = grouped['SNR'].transform(lambda x: x.rolling(window).apply(np.ptp))

        #########################距离##################################
        def calc_r_trend(r_window):
            return np.polyfit(range(len(r_window)), r_window, 1)[0]
        # 距离变化趋势
        df[f'range_trend_{window}d'] = grouped['range'].transform(lambda x: x.rolling(window).apply(calc_r_trend))
    
    return df
    

def flight_trace_features(df, windows=[5, 10]):
    
    grouped = df.groupby("id")
    for window in windows:
        ##############################航向################################
        #平均航向变化
        df[f'heading_change_{window}d'] = grouped['heading'].transform(lambda x: x.rolling(window).apply(lambda h: np.mean(np.abs(
            np.where(np.diff(h) > 180, np.diff(h) - 360,
            np.where(np.diff(h) < -180, np.diff(h) + 360,np.diff(h))))) if len(h) >= 2 else np.nan))
        
        # 路径效率
        df[f'path_efficiency_{window}d'] = grouped.apply(lambda g: g['azimuth'].rolling(window).apply(
                lambda az: (np.sqrt(np.diff(az)**2 + np.diff(g['range'].loc[az.index])**2).sum() / 
                np.sqrt(np.diff(az).sum()**2 + np.diff(g['range'].loc[az.index]).sum()**2)) 
                if len(az) >= 2 else np.nan)).reset_index(level=0, drop=True)   
        df[f'path_efficiency_{window}d'] = df[f'path_efficiency_{window}d'].replace([np.inf, -np.inf], np.nan)
        
    return df





def getdataset(df):
    
    df=dot_trace_features(df, windows=[5])
    df=flight_trace_features(df,windows=[5,10])
    
    return df

def ensemble_models(test, n_classes=4):
    # 获取各模型的预测概率
    lgbm_probs = LGBM(test).drop(['id', 'timestamp'], axis=1)
    xgb_probs = XGB(test).drop(['id', 'timestamp'], axis=1)
    cat_probs = Cat(test).drop(['id', 'timestamp'], axis=1)
    
    weights = {'lgbm': 0.377, 'xgb': 0.301, 'cat': 0.322}

    # 加权平均
    avg_probs = (weights['lgbm']*lgbm_probs + 
                 weights['xgb']*xgb_probs + 
                 weights['cat']*cat_probs)
    
    predicted_class = np.argmax(avg_probs.values, axis=1) + 1  
    
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp'],
        'predicted_class': predicted_class
    })
    
    return result


#################################LGBM#################################
def LGBM(test):
    model = lgb.Booster(model_file='./models/lgbm_model.txt')
    test_data = test.drop(['id', 'timestamp'], axis=1)
    predict_proba = model.predict(test_data)
    
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp']
    })
    
    for i in range(predict_proba.shape[1]):
        result[f'class_{i+1}_prob'] = predict_proba[:, i]  
    
    return result



#################################XGB######################################
def XGB(test):
    model = xgb.Booster()
    model.load_model('./models/xgb_model.txt') 
    test_data = test.drop(['id', 'timestamp'], axis=1)
    dtest = xgb.DMatrix(test_data)
    predict_proba = model.predict(dtest)
    
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp']
    })
    
    for i in range(predict_proba.shape[1]):
        result[f'class_{i+1}_prob'] = predict_proba[:, i]  
    
    return result



###############################Cat##############################
def Cat(test):
    model = CatBoostClassifier()
    model.load_model('./models/cat_model.cbm')
    test_data = test.drop(['id', 'timestamp'], axis=1)
    
    predict_proba = model.predict_proba(test_data)

    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp']
    })
    
    for i in range(predict_proba.shape[1]):
        result[f'class_{i+1}_prob'] = predict_proba[:, i]  
    
    return result
    
    

#####################################主程序#########################################

#############################数据处理#########################

############点迹数据############
folder_path = "./test_data/点迹"
all_data = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        parts = filename.split('_')
        track_id = parts[1]  
        track_length = parts[2].split('.')[0]  

        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, encoding='GBK')
        
        data['批号'] = track_id
        
        cols = ['批号'] + [col for col in data.columns if col not in ['批号']]
        data = data[cols]
        
        all_data = pd.concat([all_data, data], ignore_index=True)

all_data.to_csv('dataset/dot_trace_test.csv', index=False, encoding='utf_8_sig')

print("点迹数据处理完成")


#############航迹数据###########
folder_path = "./test_data/航迹"
all_data = pd.DataFrame()

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        parts = filename.split('_')
        track_id = parts[1]  
        track_length = parts[2].split('.')[0]  
        
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, encoding='GBK')

        data['批号'] = track_id

        cols = ['批号'] + [col for col in data.columns if col not in ['批号']]
        data = data[cols]
        
        all_data = pd.concat([all_data, data], ignore_index=True)

all_data.to_csv('dataset/flight_trace_test.csv', index=False, encoding='utf_8_sig')
print("航迹数据处理完成")


df1 = pd.read_csv('dataset/dot_trace_test.csv')
df2 = pd.read_csv('dataset/flight_trace_test.csv')

merged_df = pd.concat([df1, df2], axis=1)


merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]


merged_df.to_csv('dataset/test_data.csv', index=False)


#############原始回波数据###########
DAT_FOLDER = "./test_data/原始回波"
OUTPUT_CSV = "dataset/Original_echo_test.csv"

feature_df = process_radar_data(DAT_FOLDER, OUTPUT_CSV)


#############################模型训练#########################

df = pd.read_csv('dataset/test_data.csv')
df['number'] = df.groupby('批号').cumcount() + 1
df = df.rename(columns={'批号': 'id'})
df2=pd.read_csv('dataset/Original_echo_test.csv')  
df = df.merge(df2, on=['id', 'number'], how='left')
df = df.rename(columns={'id':'批号'})

test=prepare(df)
test=Screening(test) #异常数据剔除
test=getdataset(test)

result=ensemble_models(test)

##############################结果写回###########################
tracks_folder = './test_data/航迹/'
update_track_files(tracks_folder, result)



