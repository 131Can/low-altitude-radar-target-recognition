import os
import pandas as pd

####################################点迹数据###################################
folder_path = r"D:\Competition\挑战杯-揭榜挂帅2025\CQ-08-低空监视雷达目标智能识别技术研究比赛方案\数据集\数据集\点迹"

all_data = pd.DataFrame()


for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        # 解析文件名获取航迹批号和目标类型标签
        parts = filename.split('_')
        track_id = parts[1]  # 航迹批号
        target_type = parts[2].split('.')[0]  # 目标类型标签
        if target_type in ['5', '6']:
            continue
        
        # 读取txt文件
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, encoding='GBK')
        
        # 添加航迹批号和目标类型标签列
        #data['航迹批号'] = track_id
        data['目标类型标签'] = target_type
        
        # 调整列顺序，将'航迹批号'放在第一列
        cols = data.columns.tolist()
        cols = ['批号'] + [col for col in cols if col != '批号']
        data = data[cols]
        
        # 将当前文件数据添加到总数据中
        all_data = pd.concat([all_data, data], ignore_index=True)


all_data.to_csv('数据集/点迹数据.csv', index=False, encoding='utf_8_sig')

print("点迹数据处理完成")


####################################航迹数据###################################
folder_path = r"D:\Competition\挑战杯-揭榜挂帅2025\CQ-08-低空监视雷达目标智能识别技术研究比赛方案\数据集\数据集\航迹"

all_data = pd.DataFrame()


for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        # 解析文件名获取航迹批号和目标类型标签
        parts = filename.split('_')
        track_id = parts[1]  # 航迹批号
        target_type = parts[2].split('.')[0]  # 目标类型标签
        if target_type in ['5', '6']:
            continue
        
        # 读取txt文件
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path, encoding='GBK')
        
        # 添加航迹批号和目标类型标签列
        #data['航迹批号'] = track_id
        data['目标类型标签'] = target_type
        
        # 调整列顺序，将'航迹批号'放在第一列
        cols = data.columns.tolist()
        cols = ['批号'] + [col for col in cols if col != '批号']
        data = data[cols]
        
        # 将当前文件数据添加到总数据中
        all_data = pd.concat([all_data, data], ignore_index=True)


all_data.to_csv('数据集/航迹数据.csv', index=False, encoding='utf_8_sig')

print("航迹数据处理完成")

df1 = pd.read_csv('数据集/点迹数据.csv')
df2 = pd.read_csv('数据集/航迹数据.csv')

merged_df = pd.concat([df1, df2], axis=1)


merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]


cols = [col for col in merged_df.columns if col != '目标类型标签'] + ['目标类型标签']
merged_df = merged_df[cols]

merged_df.to_csv('数据集/data.csv', index=False)
'''
# 读取CSV文件
df = pd.read_csv('数据集/data.csv')  # 替换为你的文件路径

# 按照目标类型标签分组
grouped = df.groupby('目标类型标签')

# 初始化测试集的批号列表
test_batch_numbers = []

# 对每个标签类别抽取50个批号
for label, group in grouped:
    # 获取该标签下的所有唯一批号
    batch_numbers = group['批号'].unique()
    
    # 如果该标签的批号数量不足50个，则全部选取
    n_samples = min(50, len(batch_numbers))
    
    # 随机选择50个批号（如果可用）
    selected_batches = pd.Series(batch_numbers).sample(n=n_samples, random_state=42)
    test_batch_numbers.extend(selected_batches)

# 划分训练集和测试集
test_df = df[df['批号'].isin(test_batch_numbers)]
train_df = df[~df['批号'].isin(test_batch_numbers)]

# 保存结果
train_df.to_csv('数据集/train.csv', index=False)
test_df.to_csv('数据集/test.csv', index=False)

# 打印结果信息
print(f"训练集大小: {len(train_df)}")
print(f"测试集大小: {len(test_df)}")
print("测试集中各标签的批号数量分布:")
print(test_df.groupby('目标类型标签')['批号'].nunique())
'''