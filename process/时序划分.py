import pandas as pd

# 读取数据
df = pd.read_csv('数据集/data.csv')

# 获取每个批号的最早点时间
batch_min_time = df.groupby(['目标类型标签', '批号'])['点时间'].min().reset_index()
batch_min_time.rename(columns={'点时间': '批号最早点时间'}, inplace=True)

# 合并回原始数据
df = pd.merge(df, batch_min_time, on=['目标类型标签', '批号'])

# 对每个目标类型标签的批号按最早点时间排序
df_sorted = df.sort_values(['目标类型标签', '批号最早点时间'])

# 划分训练集和验证集
train_dfs = []
val_dfs = []

for label in df['目标类型标签'].unique():
    label_df = df_sorted[df_sorted['目标类型标签'] == label]
    batches = label_df['批号'].unique()
    
    # 计算划分点
    split_idx = int(len(batches) * 0.8)
    train_batches = batches[:split_idx]
    val_batches = batches[split_idx:]
    
    # 筛选数据并删除临时列
    train_dfs.append(label_df[label_df['批号'].isin(train_batches)].drop(columns=['批号最早点时间']))
    val_dfs.append(label_df[label_df['批号'].isin(val_batches)].drop(columns=['批号最早点时间']))

# 合并所有目标类型标签的结果
train_df = pd.concat(train_dfs)
val_df = pd.concat(val_dfs)

# 保存结果
train_df.to_csv('数据集/train.csv', index=False)
val_df.to_csv('数据集/test.csv', index=False)