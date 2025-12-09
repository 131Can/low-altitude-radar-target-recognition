import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgboost
import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import struct
from sklearn.metrics import accuracy_score
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

def extract_features(frame, track_id, label):
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
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1] == "Label":
                    track_id = parts[0]
                    label = parts[2].split('.')[0]  
                    
                    if label in ['5', '6']:
                        print(f"跳过文件: {filename}, 因为label={label}")
                        continue
                else:
                    track_id = "unknown"
                    label = "unknown"
            except:
                track_id = "error"
                label = "error"
            
            print(f"处理文件: {filename}, 航迹批号: {track_id}, 标签: {label}")
            frames = parse_radar_dat(file_path)
            print(f"  共解析 {len(frames)} 帧数据")
            for i, frame in enumerate(frames):
                try:
                    features = extract_features(frame, track_id, label)
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
    
    # 分类标签
    "目标类型标签": "label"
}
    data = transcolname(data, column_mapping)
    data['label']=data['label']-1
    '''
    columns_to_drop = ['number', 'label_first',"range","azimuth","elevation","v_doppler","amplitude",
                       "SNR","raw_points","filtered_range","filtered_azimuth","filtered_elevation",
                       "v_total","v_x","v_z","v_y","heading"]
    '''
    columns_to_drop = ['number']
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    
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

def ensemble_models(train, test, n_classes=4):
    # 获取各模型的预测概率
    lgbm_probs = LGBM(train, test, test).drop(['id', 'timestamp'], axis=1)
    xgb_probs = XGB(train, test, test).drop(['id', 'timestamp'], axis=1)
    cat_probs = Cat(train, test, test).drop(['id', 'timestamp'], axis=1)
    
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
def LGBM(train, val, test):
    params = {
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 4,
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'max_depth': 5,
        'min_child_weight': 1,
        'lambda_l2': 1,
        'colsample_bytree': 0.7,
        'subsample': 0.9,
        'scale_pos_weight': 1,
        'verbose': -1
    }
    
    train_data = train.drop(['id', 'timestamp', 'label'], axis=1)
    train_label = train['label']
    val_data = val.drop(['id', 'timestamp', 'label'], axis=1)
    val_label = val['label']
    test_data = test.drop(['id', 'timestamp', 'label'], axis=1)
    
    dtrain = lgb.Dataset(train_data, label=train_label)
    dval = lgb.Dataset(val_data, label=val_label) 
    
    eval_results = {}
    callbacks = [
        lgb.record_evaluation(eval_results),
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=100)  
    ]
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],          
        valid_names=['train', 'val'],        
        callbacks=callbacks
    )
    
    final_train_loss = eval_results['train']['multi_logloss'][-1]
    final_val_loss = eval_results['val']['multi_logloss'][-1]
    #print(f"\nFinal Training Loss: {final_train_loss:.6f}")
    print(f"Final Validation Loss: {final_val_loss:.6f}")

    predict_proba = model.predict(test_data)
    predict_class = np.argmax(predict_proba, axis=1)
    
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp']
    })
    
    for i in range(predict_proba.shape[1]):
        result[f'class_{i+1}_prob'] = predict_proba[:, i]  
    
    return result


#################################XGB######################################
def XGB(train, val, test):
    params = {
        'objective': 'multi:softprob',
        'num_class': 4,
        'eval_metric': 'mlogloss',
        'booster': 'gbtree',
        'learning_rate': 0.01,
        'max_depth': 5,
        'min_child_weight': 1,
        'reg_lambda': 1,
        'colsample_bytree': 0.7,
        'subsample': 0.9,
        'verbosity': 0
    }
    
    train_data = train.drop(['id', 'timestamp', 'label'], axis=1)
    train_label = train['label']
    val_data = val.drop(['id', 'timestamp', 'label'], axis=1)
    val_label = val['label']
    test_data = test.drop(['id', 'timestamp', 'label'], axis=1)
    
    dtrain = xgb.DMatrix(train_data, label=train_label)
    dval = xgb.DMatrix(val_data, label=val_label)
    dtest = xgb.DMatrix(test_data)
    
    eval_list = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=eval_list,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    best_iter = model.best_iteration
    val_loss = model.best_score
    print(f"\nBest Validation Loss (iter={best_iter}): {val_loss:.6f}")
    
    
    predict_proba = model.predict(dtest, iteration_range=(0, best_iter))
    predict_class = np.argmax(predict_proba, axis=1)
    
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp']
    })
    
    for i in range(predict_proba.shape[1]):
        result[f'class_{i+1}_prob'] = predict_proba[:, i] 
    
    return result


###############################Cat##############################
def Cat(train, val, test):
    params = {
        'loss_function': 'MultiClass',       
        'eval_metric': 'MultiClass',         
        'classes_count': 4,                  
        'learning_rate': 0.01,
        'depth': 5,                         
        'l2_leaf_reg': 1,                    
        'colsample_bylevel': 0.7,           
        'bootstrap_type': 'Bernoulli', 
        'subsample': 0.9,                    
        'random_seed': 42,                   
        'verbose': 100,                      
        'early_stopping_rounds': 100,         
        'iterations': 5000                  
    }
    
    train_data = train.drop(['id', 'timestamp', 'label'], axis=1)
    train_label = train['label']
    val_data = val.drop(['id', 'timestamp', 'label'], axis=1)
    val_label = val['label']
    test_data = test.drop(['id', 'timestamp', 'label'], axis=1)

    train_pool = Pool(train_data, label=train_label)
    val_pool = Pool(val_data, label=val_label)  

    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,                   
        use_best_model=True,                
        plot=False                           
    )

    best_iter = model.get_best_iteration()
    val_metrics = model.get_evals_result()['validation']['MultiClass']
    final_val_loss = val_metrics[best_iter - 1] 
    print(f"Best Iteration: {best_iter}")
    print(f"Final Validation Loss: {final_val_loss:.6f}")
    
    predict_proba = model.predict_proba(test_data)
    predict_class = np.argmax(predict_proba, axis=1)

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
folder_path = "./Original_data/点迹"

all_data = pd.DataFrame()


for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        parts = filename.split('_')
        track_id = parts[1]  # 航迹批号
        target_type = parts[2].split('.')[0] # 目标类型标签
        if target_type in ['5', '6']:
            continue

        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, encoding='GBK')
        
        #data['航迹批号'] = track_id
        data['目标类型标签'] = target_type
        
        cols = data.columns.tolist()
        cols = ['批号'] + [col for col in cols if col != '批号']
        data = data[cols]
        
        all_data = pd.concat([all_data, data], ignore_index=True)


all_data.to_csv('dataset/dot_trace.csv', index=False, encoding='utf_8_sig')

print("点迹数据处理完成")


#############航迹数据###########
folder_path = "./Original_data/航迹"

all_data = pd.DataFrame()


for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        parts = filename.split('_')
        track_id = parts[1]  # 航迹批号
        target_type = parts[2].split('.')[0]  # 目标类型标签
        if target_type in ['5', '6']:
            continue
        
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, encoding='GBK')
        
        #data['航迹批号'] = track_id
        data['目标类型标签'] = target_type
        
        cols = data.columns.tolist()
        cols = ['批号'] + [col for col in cols if col != '批号']
        data = data[cols]
        
        all_data = pd.concat([all_data, data], ignore_index=True)


all_data.to_csv('dataset/flight_trace.csv', index=False, encoding='utf_8_sig')

print("航迹数据处理完成")

df1 = pd.read_csv('dataset/dot_trace.csv')
df2 = pd.read_csv('dataset/flight_trace.csv')

merged_df = pd.concat([df1, df2], axis=1)


merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]


cols = [col for col in merged_df.columns if col != '目标类型标签'] + ['目标类型标签']
merged_df = merged_df[cols]

merged_df.to_csv('dataset/train_data.csv', index=False)


#############原始回波数据###########
DAT_FOLDER = "./Original_data/原始回波"
OUTPUT_CSV = "dataset/Original_echo.csv"

feature_df = process_radar_data(DAT_FOLDER, OUTPUT_CSV)


#############################模型训练#########################
df = pd.read_csv('dataset/train_data.csv')
df['number'] = df.groupby('批号').cumcount() + 1
df = df.rename(columns={'批号': 'id'})
df2=pd.read_csv('dataset/Original_echo.csv')  
df = df.merge(df2, on=['id', 'number'], how='left')
df = df.rename(columns={'id':'批号'})

#merged_df.to_csv('data2.csv', index=False)
# 按照目标类型标签分组
grouped = df.groupby('目标类型标签')

###########################5折CV#############################
folds = {i: [] for i in range(5)}

# 对每个标签类别进行分层抽样
for label, group in grouped:
    batch_numbers = group['批号'].unique()
    np.random.seed(30)  # 设置随机种子以保证可重复性
    shuffled_batches = np.random.permutation(batch_numbers)
    
    n_batches = len(shuffled_batches)
    fold_size = n_batches // 5
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else n_batches
        folds[i].extend(shuffled_batches[start:end])

Sum_accuracy=0
for i in range(5):
    test_batch_numbers = folds[i]
    train_batch_numbers = []
    for j in range(5):
        if j != i:
            train_batch_numbers.extend(folds[j])
    
    test = df[df['批号'].isin(test_batch_numbers)]
    train = df[df['批号'].isin(train_batch_numbers)]

    train=prepare(train)
    test=prepare(test)
    
    train=getdataset(train)
    test=getdataset(test)
####################################软投票##################################

    result=ensemble_models(train, test)
    
    result.to_csv('result/result.csv', index=False)
    
    true_labels = test['label']+1  
    predicted_labels = result['predicted_class'] 
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"第{i+1}折准确率: {accuracy:.4f}")
    Sum_accuracy+=accuracy
Sum_accuracy=Sum_accuracy/5
print(f"5折平均准确率: {Sum_accuracy:.4f}")

###################################硬投票###################################
'''
    # 获取三个模型的预测结果
    lgbm_pred = LGBM(train, test, test)['predicted_class']
    xgb_pred = XGB(train, test, test)['predicted_class']
    cat_pred = Cat(train, test, test)['predicted_class']
    
    final_pred = []
    
    # 对每个样本进行投票
    for l, x, c in zip(lgbm_pred, xgb_pred, cat_pred):
        votes = Counter([l, x, c])
        final_pred.append(votes.most_common(1)[0][0])
    
    true_labels = test['label'] + 1
    accuracy = accuracy_score(true_labels, final_pred)
    print(f"第{i+1}折投票法准确率: {accuracy:.4f}")
    Sum_accuracy += accuracy
Sum_accuracy=Sum_accuracy/5

print(f"5折平均准确率: {Sum_accuracy:.4f}")
'''



'''
result = pd.DataFrame({
    'id': test['id'],
    'timestamp': test['timestamp'],
    'true_label': test['label']+1,       # 真实标签
    'predicted_class': result['predicted_class']  # 预测标签
})

result['is_correct'] = result['true_label'] == result['predicted_class']


wrong_samples = result[~result['is_correct']]  
print(wrong_samples)
'''





