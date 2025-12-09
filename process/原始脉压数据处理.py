import numpy as np
import struct
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy import stats
# 常量定义
FRAME_HEAD = 0xFA55FA55
FRAME_END = 0x55FA55FA
c = 3e8  # 光速

def parse_radar_dat(file_path):
    """解析雷达原始回波.dat文件，返回包含多帧数据的列表"""
    frames = []
    
    with open(file_path, 'rb') as fid:
        while True:
            try:
                # 搜索帧头
                head_find = struct.unpack('<I', fid.read(4))[0]
                while head_find != FRAME_HEAD:
                    fid.seek(-3, os.SEEK_CUR)  # 回退3字节
                    head_find = struct.unpack('<I', fid.read(4))[0]
                
                # 读取帧长度
                frame_data_length = struct.unpack('<I', fid.read(4))[0] * 4
                
                # 跳转到帧尾验证位置
                fid.seek(frame_data_length - 12, os.SEEK_CUR)
                end_find = struct.unpack('<I', fid.read(4))[0]
                
                # 验证帧完整性
                if end_find != FRAME_END:
                    fid.seek(-frame_data_length + 1, os.SEEK_CUR)
                    continue
                
                # 回退到帧开始位置
                fid.seek(-frame_data_length + 4, os.SEEK_CUR)
                
                # 读取基础参数
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
                
                # 重组为复数矩阵
                data_out_real = iq_data[0::2]
                data_out_imag = iq_data[1::2]
                data_out_complex = data_out_real + 1j * data_out_imag
                data_out = data_out_complex.reshape(31, para['PRTnum'])
                
                # 跳过帧尾
                fid.read(4)
                
                # 保存帧数据
                frames.append({
                    'para': para,
                    'data_out': data_out
                })
                
            except struct.error:
                # 文件结束或数据不完整
                break
    
    return frames

def extract_features(frame, track_id, label):
    """从单帧数据中提取结构化特征，包含目标中心及周边背景信息"""
    para = frame['para']
    data_out = frame['data_out']
    features = {}
    
    # 1. 基础参数特征
    features['id'] = track_id
    features['PRT'] = para['PRT']
    features['PRTnum'] = para['PRTnum']
    features['Freq'] = para['Freq']
    
    # 2. 目标信息特征
    if len(para['Track_No_info']) >= 4:
        features['number'] = para['Track_No_info'][1]
        features['target_bin'] = para['Track_No_info'][2]
        features['doppler_bin'] = para['Track_No_info'][3]
    else:
        features['target_bin'] = 15
        features['doppler_bin'] = 0
    
    # 3. 定义目标区域和背景区域
    center_row = 15  # 中心行索引
    target_radius = 1  # 目标区域半径
    background_radius = 3  # 背景区域半径
    
    # 目标区域 (中心行±2)
    target_rows = range(max(0, center_row-target_radius), 
                      min(31, center_row+target_radius+1))
    
    # 背景区域 (中心行±5，排除目标区域)
    background_rows = list(range(max(0, center_row-background_radius), 
                             center_row-target_radius)) + \
                    list(range(center_row+target_radius+1, 
                             min(31, center_row+background_radius+1)))
    
    # 4. 目标区域特征提取 (中心±2行)
    target_signals = data_out[target_rows, :]
    target_amps = np.abs(target_signals)
    
    # 目标区域统计特征
    features['target_amp_mean'] = np.mean(target_amps)
    features['target_amp_std'] = np.std(target_amps)
    features['target_amp_max'] = np.max(target_amps)
    features['target_amp_min'] = np.min(target_amps)
    
    # 目标区域相位特征
    target_phases = np.angle(target_signals)
    features['target_phase_var'] = np.var(np.diff(target_phases, axis=1))
    
    # 5. 背景区域特征提取 (中心±5行，排除目标区域)
    if background_rows:
        bg_signals = data_out[background_rows, :]
        bg_amps = np.abs(bg_signals)
        
        # 背景统计特征
        features['bg_amp_mean'] = np.mean(bg_amps)
        features['bg_amp_std'] = np.std(bg_amps)
        features['bg_amp_max'] = np.max(bg_amps)
        features['bg_amp_min'] = np.min(bg_amps)
        
        # 目标-背景对比特征
        features['target_bg_ratio'] = features['target_amp_mean'] / (features['bg_amp_mean'] + 1e-9)
        features['target_bg_diff'] = features['target_amp_mean'] - features['bg_amp_mean']
    else:
        # 没有背景区域时的默认值
        features.update({
            'bg_amp_mean': 0,
            'bg_amp_std': 0,
            'bg_amp_max': 0,
            'bg_amp_min': 0,
            'target_bg_ratio': 1,
            'target_bg_diff': 0
        })
    
    # 6. 微多普勒特征 (仅使用中心行)
    center_signal = data_out[center_row, :]
    fft_spec = np.fft.fftshift(np.fft.fft(center_signal))
    peaks, _ = find_peaks(np.abs(fft_spec), prominence=0.1)
    
    features['harmonic_count'] = len(peaks)
    features['dominant_freq'] = np.argmax(np.abs(fft_spec)) - len(fft_spec)//2
    
    # 7. 空间分布特征 (全矩阵)
    full_amp = np.abs(data_out)
    features['spatial_var'] = np.var(full_amp)  # 空间方差
    features['spatial_gradient'] = np.mean(np.abs(np.gradient(full_amp, axis=0)))  # 垂直梯度
    
    # 8. 目标聚焦度特征
    center_amp = np.abs(data_out[center_row, :])
    features['focus_index'] = np.max(center_amp) / (np.mean(full_amp) + 1e-9)
    
    # 9. 背景杂波特征
    if background_rows:
        bg_amp_flat = np.abs(data_out[background_rows, :]).flatten()
        features['bg_skewness'] = stats.skew(bg_amp_flat)
        features['bg_kurtosis'] = stats.kurtosis(bg_amp_flat)
    else:
        features['bg_skewness'] = 0
        features['bg_kurtosis'] = 0
        
    # 10. 距离单元特征
    for i in range(0, 31, 5):  # 每5个单元采一个样
        cell_signal = data_out[i, :]
        features[f'cell{i}_mean'] = np.mean(np.abs(cell_signal))
        features[f'cell{i}_kurt'] = stats.kurtosis(np.abs(cell_signal))
    
    return features


def aggregate_features(features_df):
    """对提取的特征进行时间维度上的聚合"""
    grouped = features_df.groupby(['id', 'number'])
    
    aggregation_functions = {
    # ============ 基础静态特征 ============ 
    'PRT': 'first',          # 脉冲重复周期（物理设备参数不变）
    'PRTnum': 'first',       # 脉冲数量（采集参数不变）
    'Freq': 'first',         # 载频（雷达硬件参数不变）
    'target_bin': 'first',   # 目标距离单元（物理位置）
    'doppler_bin': 'first',  # 目标多普勒单元（物理速度）

    # ============ 目标区域特征 ============
    'target_amp_mean': ['mean', 'std', 'max', 'min'],  # 目标区域平均幅度（稳定性）
    'target_amp_std': ['mean', 'std', 'max'],  # 目标波动强度
    'target_amp_max': ['mean', 'std', 'max'],  # 最大回波强度
    'target_amp_min': ['mean', 'std'],         # 最小回波强度
    'target_phase_var': ['mean', 'std', 'max'],  # 相位变化（表面复杂度）

    # ============ 背景区域特征 ============
    'bg_amp_mean': ['mean', 'std', 'max'],  # 背景平均强度
    'bg_amp_std': ['mean', 'std', 'max'],      # 背景波动性
    'bg_amp_max': ['mean', 'max'],             # 背景最大干扰
    'bg_amp_min': ['mean'],                    # 背景基底噪声
    
    # ============ 目标-背景对比特征 ============
    'target_bg_ratio': ['mean', 'std', 'max', 'min'],  # 信杂比动态范围
    'target_bg_diff': ['mean', 'std', 'max'],  # 绝对强度差
    
    # ============ 空间分布特征 ============
    'spatial_var': ['mean', 'std', 'max'],     # 空间方差（目标扩展性）
    'spatial_gradient': ['mean', 'std'],       # 垂直梯度（边界锐度）
    
    # ============ 微多普勒特征 ============
    'harmonic_count': ['mean', 'max', 'min'],  # 谐波数量变化
    'dominant_freq': ['mean', 'std'],    # 主频变化范围
    
    # ============ 杂波统计特征 ============
    'bg_skewness': ['mean', 'std'],           # 背景分布偏度
    'bg_kurtosis': ['mean', 'std', 'max'],    # 背景峰度（脉冲性）
    
    # ============ 高级复合特征 ============
    'focus_index': ['mean', 'std', 'max']
}
    
    # 为所有距离单元特征添加聚合函数
    for col in features_df.columns:
        if col.startswith('cell'):
            if col.endswith('_mean'):
                aggregation_functions[col] = ['mean', 'std']
            elif col.endswith('_kurt'):
                aggregation_functions[col] = ['mean', 'std', 'max']
    
    # 执行聚合
    aggregated_df = grouped.agg(aggregation_functions)
    
    # 扁平化多级列索引
    aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]
    
    # 添加帧数统计特征
    frame_counts = grouped.size().rename('frame_count')
    aggregated_df = aggregated_df.join(frame_counts)
    
    # 添加目标距离特征（基于第一个目标单元）
    aggregated_df['target_range'] = aggregated_df['target_bin_first'] * 7.5  # 假设7.5m/单元
    
    # 添加目标运动稳定性特征
    if 'dominant_freq_mean' in aggregated_df.columns:
        aggregated_df['freq_stability'] = aggregated_df['dominant_freq_std'] / aggregated_df['frame_count']
    
    # 重置索引
    return aggregated_df.reset_index()

def process_radar_data(dat_folder, output_csv=None):
    """处理整个文件夹的雷达数据并保存特征到一个CSV文件"""
    all_features = []
    
    # 遍历文件夹中的所有.dat文件
    for filename in os.listdir(dat_folder):
        if filename.endswith('.dat'):
            file_path = os.path.join(dat_folder, filename)
            
            # 从文件名中提取航迹批号和标签
            try:
                parts = filename.split('_')
                if len(parts) >= 3 and parts[1] == "Label":
                    track_id = parts[0]
                    label = parts[2].split('.')[0]  # 去掉文件扩展名
                    
                    # 跳过label为5或6的情况
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
            
            # 1. 解析原始数据
            frames = parse_radar_dat(file_path)
            print(f"  共解析 {len(frames)} 帧数据")
            
            # 2. 提取特征
            for i, frame in enumerate(frames):
                try:
                    features = extract_features(frame, track_id, label)
                    all_features.append(features)
                except Exception as e:
                    print(f"  处理第 {i} 帧时出错: {str(e)}")
    
    # 3. 转换为DataFrame
    if not all_features:
        print("没有提取到任何特征数据")
        return None
    
    feature_df = pd.DataFrame(all_features)
    feature_df=aggregate_features(feature_df)
    
    # 4. 保存到CSV
    if output_csv:
        feature_df.to_csv(output_csv, index=False)
        print(f"所有特征已保存至 {output_csv}")
    
    return feature_df

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 指定原始回波文件夹路径
    DAT_FOLDER = r"D:\Competition\挑战杯-揭榜挂帅2025\CQ-08-低空监视雷达目标智能识别技术研究比赛方案\数据集\数据集\原始回波"
    OUTPUT_CSV = "all_radar_features.csv"
    
    # 处理文件夹中的所有数据并保存特征
    feature_df = process_radar_data(DAT_FOLDER, OUTPUT_CSV)
    
    