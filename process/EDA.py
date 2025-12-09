import pandas as pd
import matplotlib.pyplot as plt

# 1. 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 2. 加载数据并取多普勒速度绝对值
df = pd.read_csv('数据集/点迹数据.csv')  # 替换为实际路径
df['多普勒速度'] = df['多普勒速度'].abs()  # 关键修改：速度转为绝对值

# 3. 定义目标类型标签映射字典
label_map = {
    '1': '1: 轻型旋翼无人机',
    '2': '2: 小型旋翼无人机',
    '3': '3: 鸟类',
    '4': '4: 空飘球'
}
df['目标类型标签'] = df['目标类型标签'].astype(str).map(label_map)  # 应用标签映射

# 4. 计算平均速度（按映射后的标签分组）
mean_speed = df.groupby('目标类型标签')['多普勒速度'].mean().sort_values()
#mean_speed = df.groupby('目标类型标签')['多普勒速度'].std().sort_values()

# 5. 创建图表
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(mean_speed.index, mean_speed.values, 
              color='#1f77b4',  # 定制颜色
              width=0.7, edgecolor='white', linewidth=1)

# 6. 添加数值标签和注释
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height,  # 数值显示在柱顶上方
            f'{height:.1f}m/s',                           # 保留1位小数+单位
            ha='center', va='bottom', fontsize=10, color='black')

# 7. 设置标题和轴标签
ax.set_title('目标类型多普勒速度', fontsize=14, pad=20)
ax.set_xlabel('目标类型', fontsize=12, labelpad=10)
ax.set_ylabel('速度', fontsize=12, labelpad=10)

plt.tight_layout()
plt.show()