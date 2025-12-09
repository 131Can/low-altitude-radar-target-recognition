import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
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
from collections import Counter
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

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
        df[f'path_efficiency_{window}d'] = df[f'path_efficiency_{window}d'].replace([np.inf, -np.inf], np.nan)
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
    
    # 获取最终预测类别
    predicted_class = np.argmax(avg_probs.values, axis=1) + 1  # 假设类别从1开始
    
    # 构建结果DataFrame
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp'],
        'predicted_class': predicted_class
    })
    
    
    
    return result


def getdataset(df):
    
    df=dot_trace_features(df, windows=[5,10])
    df=flight_trace_features(df,windows=[5,10])
    
    return df


#################################LGBM#################################
def LGBM(train, val, test):
    # LightGBM 参数
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
    
    # 准备数据
    train_data = train.drop(['id', 'timestamp', 'label'], axis=1)
    train_label = train['label']
    val_data = val.drop(['id', 'timestamp', 'label'], axis=1)
    val_label = val['label']
    test_data = test.drop(['id', 'timestamp', 'label'], axis=1)
    
    # 转换为 LightGBM 数据格式
    dtrain = lgb.Dataset(train_data, label=train_label)
    dval = lgb.Dataset(val_data, label=val_label)  # 验证集
    
    # 回调函数记录评估结果
    eval_results = {}
    callbacks = [
        lgb.record_evaluation(eval_results),
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=100)  # 可选：早停机制
    ]
    
    # 训练模型（监控训练集和验证集）
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        valid_sets=[dtrain, dval],           # 同时监控训练集和验证集
        valid_names=['train', 'val'],        # 命名用于结果区分
        callbacks=callbacks
    )
    
    # 打印最终loss
    final_train_loss = eval_results['train']['multi_logloss'][-1]
    final_val_loss = eval_results['val']['multi_logloss'][-1]
    #print(f"\nFinal Training Loss: {final_train_loss:.6f}")
    print(f"Final Validation Loss: {final_val_loss:.6f}")
    
    # 预测概率（已经得到的是概率）
    predict_proba = model.predict(test_data)
    
    # 返回概率结果
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp']
    })
    
    # 为每个类别添加概率列
    for i in range(predict_proba.shape[1]):
        result[f'class_{i+1}_prob'] = predict_proba[:, i]  # 假设类别从1开始编号
    
    return result


#################################XGB######################################
def XGB(train, val, test):
    # XGBoost 参数
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
    
    # 准备数据
    train_data = train.drop(['id', 'timestamp', 'label'], axis=1)
    train_label = train['label']
    val_data = val.drop(['id', 'timestamp', 'label'], axis=1)
    val_label = val['label']
    test_data = test.drop(['id', 'timestamp', 'label'], axis=1)
    
    # 转换为 DMatrix 格式
    dtrain = xgb.DMatrix(train_data, label=train_label)
    dval = xgb.DMatrix(val_data, label=val_label)
    dtest = xgb.DMatrix(test_data)
    
    # 训练模型（带验证集监控）
    eval_list = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=eval_list,
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    # 获取最佳迭代结果
    best_iter = model.best_iteration
    val_loss = model.best_score
    print(f"\nBest Validation Loss (iter={best_iter}): {val_loss:.6f}")
    
    
    # 预测概率（已经得到的是概率）
    predict_proba = model.predict(dtest, iteration_range=(0, best_iter))
    
    # 返回概率结果
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp']
    })
    
    # 为每个类别添加概率列
    for i in range(predict_proba.shape[1]):
        result[f'class_{i+1}_prob'] = predict_proba[:, i]  # 假设类别从1开始编号
    
    return result


###############################Cat##############################
def Cat(train, val, test):
    # CatBoost 参数
    params = {
        'loss_function': 'MultiClass',       # 多分类任务
        'eval_metric': 'MultiClass',         # 评估指标（也支持'MultiLogloss'）
        'classes_count': 4,                  # 类别数
        'learning_rate': 0.01,
        'depth': 5,                          # 树的最大深度
        'l2_leaf_reg': 1,                    # L2正则化（类似lambda_l2）
        'colsample_bylevel': 0.7,            # 特征采样（类似colsample_bytree）
        'bootstrap_type': 'Bernoulli', 
        'subsample': 0.9,                    # 样本采样
        'random_seed': 42,                   # 随机种子
        'verbose': 100,                      # 每100轮打印日志
        'early_stopping_rounds': 100,         # 早停轮数
        'iterations': 5000                  # 明确设置最大轮数
    }

    # 准备数据（移除无关列）
    train_data = train.drop(['id', 'timestamp', 'label'], axis=1)
    train_label = train['label']
    val_data = val.drop(['id', 'timestamp', 'label'], axis=1)
    val_label = val['label']
    test_data = test.drop(['id', 'timestamp', 'label'], axis=1)

    # 转换为 CatBoost 的 Pool 格式（高效数据容器）
    train_pool = Pool(train_data, label=train_label)
    val_pool = Pool(val_data, label=val_label)  # 验证集

    # 训练模型（监控训练集和验证集）
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,                   # 监控验证集
        use_best_model=True,                 # 启用早停后保留最佳模型
        plot=False                           # 可选：关闭实时训练曲线
    )

    # 打印最佳迭代的验证损失
    best_iter = model.get_best_iteration()
    val_metrics = model.get_evals_result()['validation']['MultiClass']
    final_val_loss = val_metrics[best_iter - 1]  # CatBoost索引从0开始
    print(f"Best Iteration: {best_iter}")
    print(f"Final Validation Loss: {final_val_loss:.6f}")
    

    # 预测概率（已经得到的是概率）
    predict_proba = model.predict_proba(test_data)
    
    # 返回概率结果
    result = pd.DataFrame({
        'id': test['id'],
        'timestamp': test['timestamp']
    })
    
    # 为每个类别添加概率列
    for i in range(predict_proba.shape[1]):
        result[f'class_{i+1}_prob'] = predict_proba[:, i]  # 假设类别从1开始编号
    
    return result


############################5折CV###############################
df = pd.read_csv('数据集/data.csv')  

# 按照目标类型标签分组
grouped = df.groupby('目标类型标签')

# 初始化5个fold的批号字典
folds = {i: [] for i in range(5)}

# 对每个标签类别进行分层抽样
for label, group in grouped:
    # 获取该标签下的所有唯一批号
    batch_numbers = group['批号'].unique()
    
    # 随机打乱批号
    np.random.seed(30)  # 设置随机种子以保证可重复性
    shuffled_batches = np.random.permutation(batch_numbers)
    
    # 将批号分成5个fold
    n_batches = len(shuffled_batches)
    fold_size = n_batches // 5
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else n_batches
        folds[i].extend(shuffled_batches[start:end])

Sum_accuracy=0
# 生成5个fold的训练集和测试集
for i in range(5):
    # 当前fold的测试集批号
    test_batch_numbers = folds[i]
    
    # 其他fold的批号合并为训练集
    train_batch_numbers = []
    for j in range(5):
        if j != i:
            train_batch_numbers.extend(folds[j])
    
    # 划分训练集和测试集
    test = df[df['批号'].isin(test_batch_numbers)]
    train = df[df['批号'].isin(train_batch_numbers)]


    train=prepare(train)
    test=prepare(test)
    
    
    train=getdataset(train)
    test=getdataset(test)
    
####################################单模##################################

    result=ensemble_models(train, test)
    
    result.to_csv('result/result.csv', index=False)
    
    true_labels = test['label']+1  
    predicted_labels = result['predicted_class'] 
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"第{i+1}折准确率: {accuracy:.4f}")
    Sum_accuracy+=accuracy
Sum_accuracy=Sum_accuracy/5
print(f"5折平均准确率: {Sum_accuracy:.4f}")



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
