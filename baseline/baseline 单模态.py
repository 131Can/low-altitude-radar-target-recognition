import lightgbm as lgb
import xgboost as xgb
import xgboost as xgboost
import pandas as pd
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

################################预处理###############################
def transcolname(df, column_mapping):
    df.rename(columns=column_mapping, inplace=True)
    return df


def prepare(data):
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
    return data






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
        # 峰峰值
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

    return df



def getdataset(df):
    
    df=dot_trace_features(df, windows=[5, 10])
    df=flight_trace_features(df,windows=[5, 10])
    
    return df


def LGBM(train, test):
    # LightGBM 参数
    params = {
        'objective': 'multiclass',  # 或多分类任务
        'metric': 'multi_logloss',  # 多分类评估指标
        'num_class': 4,   # 必须指定类别数量
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
    
    # 准备数据
    train_data = train.drop(['id', 'timestamp', 'label'], axis=1)
    train_label = train['label']
    test_data = test.drop(['id', 'timestamp','label'], axis=1)
    
    # 转换为 LightGBM 的数据格式
    dtrain = lgb.Dataset(train_data, label=train_label)
    
    # 训练模型
    model = lgb.train(params, dtrain, num_boost_round=2000)
    
    # 预测（返回每个样本属于每个类别的概率）
    predict_proba = model.predict(test_data)
    
    # 获取预测类别（概率最大的类别）
    predict_class = np.argmax(predict_proba, axis=1)
    
    # 处理结果
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp':test['timestamp'],
        'predicted_class': predict_class+1
    })
    
    
    return result





train=pd.read_csv('数据集/train.csv')
test=pd.read_csv('数据集/test.csv')


train=prepare(train)
test=prepare(test)


train=getdataset(train)
test=getdataset(test)

train.to_csv('result/train_feature.csv', index=False)

result=LGBM(train,test)

result.to_csv('result/result.csv', index=False)

true_labels = test['label']+1  
predicted_labels = result['predicted_class'] 

accuracy = accuracy_score(true_labels, predicted_labels)
print(f"线下准确率: {accuracy:.4f}")


'''
result = pd.DataFrame({
    'id': test['id'],
    'timestamp': test['timestamp'],
    'true_label': test['label']+1,       # 真实标签
    'predicted_class': result['predicted_class']  # 预测标签
})

# 标记错误样本
result['is_correct'] = result['true_label'] == result['predicted_class']

# 筛选所有识别错误的行
wrong_samples = result[~result['is_correct']]  # 取反操作，选择错误样本
print(wrong_samples)
'''





