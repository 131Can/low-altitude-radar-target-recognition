# 低空监视雷达目标智能识别技术研究
### 2025挑战杯揭榜挂帅CQ-08赛题落榜方案

## 📄 项目简介（Project Overview）
本项目基于 **低空监视雷达观测数据**，针对无人机、鸟类、空飘球等典型低空目标，构建了一个 **轻量级、高实时性、可部署** 的目标识别模型方案。

相比端到端深度学习模型，本方案采用 **特征工程驱动 + GBDT 模型集成** 的方式，在保证识别精度的同时显著降低了算力需求，适用于嵌入式设备与雷达前端系统的实时处理场景。

本仓库包含：
- 数据预处理脚本  
- 点迹/航迹/回波特征工程实现  
- LGBM、XGBoost、CatBoost 模型训练与集成  
- 交叉验证及测试评估流程  
- 完整的技术方案文档（比赛方案）

## 📌 项目背景（Background）
随着低空经济迅速发展，低空飞行架次呈指数级增长，空域安全压力持续提升。低空监视雷达必须在楼宇遮挡、气象干扰等复杂环境下实现对“低慢小”目标的高精度识别。

核心挑战包括：
- **弱散射、弱回波**  
- **样本规模有限**  
- **算力受限**

本项目旨在设计一套 **低算力、高鲁棒、多源融合** 的智能识别算法体系。

## 🧭 方法总览（Solution Overview）
方案采用“**特征工程驱动 + 轻量级模型集成**”。

### ✨ 方案优势
- 🚀 推理速度快  
- 🖥 算力要求低  
- 🧱 鲁棒性强  
- 🧪 多模态融合  

## 🔧 数据预处理（Data Preprocessing）
处理三类数据：点迹、航迹、回波  
- TXT → CSV  
- 回波复数矩阵解析  
- 异常数据剔除  
- 数据源统一合并  

## 🧩 特征工程（Feature Engineering）
### 点迹数据  
- 时间窗口（window=5）  
- 速度统计量、加速度、SNR偏度、距离趋势等  

### 航迹数据  
- 窗口 5 & 10  
- 航向变化、路径效率等  

### 回波数据  
- 复数矩阵重组  
- 多通道特征提取（能量、纹理、频域结构）  

## 🧠 模型体系（Models）
采用三类 GBDT 模型：
- LightGBM  
- XGBoost  
- CatBoost  

最终使用 **软投票（soft voting）** 进行集成。

## 🧪 训练与验证（Training & Evaluation）
- 5 折交叉验证  
- 分层采样（按批号）  
- Early Stopping（100 次）  
- 指标：Accuracy  


## 🚀 快速开始（Quick Start）

### 1. 克隆项目
```bash
git clone https://github.com/yourname/yourrepo.git
cd yourrepo
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行train
```bash
python train.py
```

### 4. 运行test
```bash
python test.py
```


## 📜 许可证（License）
建议使用 MIT 或 Apache-2.0。

## 📬 联系方式（Contact）
如有建议或问题欢迎提出 Issue。
