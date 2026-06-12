# Higgs Event Discovery Dashboard
## 基于机器学习的希格斯玻色子信号识别与可解释分析系统

---

# 一、项目简介

## 项目背景

大型强子对撞机（LHC）每天会产生海量粒子碰撞事件，在这些事件中：
- 大部分为 Background（背景事件）
- 少部分为 Signal（希格斯玻色子相关信号）

传统分析方法依赖大量人工规则。本项目尝试利用机器学习方法，从结构化碰撞事件特征中自动学习模式，识别潜在的希格斯信号事件。

---
## 项目目标
构建一个完整的机器学习系统，实现：

### 分类任务
输入：一次粒子碰撞事件的特征向量

输出：

- Signal（信号事件）
- Background（背景事件）

### 可解释分析
解释：
- 模型为什么认为该事件是 Signal
- 哪些特征最重要

### 科学发现优化
不仅优化 Accuracy / AUC，同时优化：AMS（Approximate Median Significance），模拟真实高能物理发现任务。

---

# 二、数据集

## 数据来源

CERN Open Data Portal
ATLAS Higgs Machine Learning Challenge Dataset

数据规模：
- 总样本数：818,238
- 总字段数：35
- 输入特征：30
- Signal：279,560（34.17%）
- Background：538,678（65.83%）

---

## 字段说明

### 不参与训练
- EventId
- Weight
- Label
- KaggleSet
- KaggleWeight

### 参与训练
30个物理特征：

#### DER 特征（高阶物理特征）

- DER_mass_MMC
- DER_mass_transverse_met_lep
- DER_mass_vis
- DER_pt_h
- DER_deltaeta_jet_jet
- DER_mass_jet_jet
- DER_prodeta_jet_jet
- DER_deltar_tau_lep
- DER_pt_tot
- DER_sum_pt
- DER_pt_ratio_lep_tau
- DER_met_phi_centrality
- DER_lep_eta_centrality

#### PRI 特征（重建特征）

- PRI_tau_pt
- PRI_tau_eta
- PRI_tau_phi
- PRI_lep_pt
- PRI_lep_eta
- PRI_lep_phi
- PRI_met
- PRI_met_phi
- PRI_met_sumet
- PRI_jet_num
- PRI_jet_leading_pt
- PRI_jet_leading_eta
- PRI_jet_leading_phi
- PRI_jet_subleading_pt
- PRI_jet_subleading_eta
- PRI_jet_subleading_phi
- PRI_jet_all_pt

---

## 数据划分

采用官方 KaggleSet 划分：

| 数据集            | KaggleSet | 数量    |
| ----------------- | --------- | ------- |
| Training          | t         | 250,000 |
| Public Validation | b         | 100,000 |
| Private Test      | v         | 450,000 |

---

# 三、数据探索（EDA）

## 数据质量检查

已完成：
- 总样本数统计
- 数据类型统计
- 重复值检测
- 标签分布分析
- Weight分布分析
- PRI_jet_num分布分析

---

## 缺失值分析

数据集中使用-999表示缺失值。

### 缺失率最高特征

| 特征                   | 缺失率 |
| ---------------------- | ------ |
| DER_deltaeta_jet_jet   | 70.91% |
| DER_mass_jet_jet       | 70.91% |
| DER_prodeta_jet_jet    | 70.91% |
| DER_lep_eta_centrality | 70.91% |
| PRI_jet_subleading_*   | 70.91% |
| PRI_jet_leading_*      | 40.01% |
| DER_mass_MMC           | 15.23% |

---

## Jet数量分析

### 喷注数量分布

| PRI_jet_num | 数量    | 占比   |
| ----------- | ------- | ------ |
| 0           | 327,371 | 40.01% |
| 1           | 252,882 | 30.91% |
| 2           | 165,027 | 20.17% |
| 3           | 72,958  | 8.92%  |

---

### 各组 Signal 比例

| PRI_jet_num | Signal占比 |
| ----------- | ---------- |
| 0           | 25.3%      |
| 1           | 35.7%      |
| 2           | 51.0%      |
| 3           | 30.5%      |

结论：不同喷注数量对应明显不同的数据分布。

后续考虑：
- 全量建模
- Jet分组建模
进行对比实验。

---

## PCA降维分析

### 目标

利用主成分分析（PCA）探索数据在低维空间中的分布特征。

本实验的目的并非提升分类性能，而是帮助理解数据结构，并分析为什么需要非线性模型。

### 方法

```python
StandardScaler
↓
PCA(n_components=2)
↓
二维可视化
```

同时尝试：

```python
PCA(n_components=3)
```

观察三维空间中的分布情况。

### 输出

- PCA二维散点图
- PCA三维散点图
- Explained Variance Ratio

### 预期现象

由于粒子物理特征之间存在复杂的非线性关系：
- Signal与Background可能在低维空间高度重叠
- PCA结果未必表现出明显可分性

### 实验意义

若PCA可视化结果呈现明显混合状态，说明问题具有较强非线性特征，这将为后续采用：

- Random Forest
- XGBoost
- MLP

等非线性模型提供依据。

### 讨论

比较：
- PCA可视化结果
- XGBoost最终分类结果

分析：为什么简单线性模型难以完成该任务。

---

## 特征相关性分析

注意，以下字段不参与分析：

```python
[
    "EventId",
    "Weight",
    "Label",
    "KaggleSet",
    "KaggleWeight"
]
```
分析内容：
- Pearson相关系数
- Spearman相关系数
- Correlation Heatmap

---

# 四、数据预处理

## 缺失值处理

数据集使用-999表示缺失值。
根据数据分析结果，不同缺失值具有不同来源：

### 类型1：结构性缺失

例如：

- PRI_jet_leading_pt
- PRI_jet_subleading_pt
- DER_deltaeta_jet_jet

当`PRI_jet_num < 2`时对应物理量本身不存在。这类缺失并非测量失败，而是事件中不存在对应喷注（Jet）。

---

### 类型2：重建失败缺失

例如：

```python
DER_mass_MMC
```

缺失率约15.23%，表示物理量重建失败。

---

## Tree Models预处理方案

适用于：
- Random Forest
- XGBoost

处理方式：保留 -999，因为树模型能够天然利用缺失模式进行分裂。

---

## Linear / Neural Models预处理方案

适用于：
- Logistic Regression
- MLP

### Step1：中位数填充

使用训练集统计量进行填充。

---

### Step2：Missing Indicator

为存在缺失的特征增加指示变量。
例如：
```python
PRI_jet_leading_pt
```

新增：
```python
missing_PRI_jet_leading_pt
```

1 = 原始值缺失
0 = 原始值存在

---

### Step3：PRI_jet_num One-Hot编码

将：

```python
PRI_jet_num
```

转换为：

```text
jet_0
jet_1
jet_2
jet_3
```

帮助模型理解不同喷注拓扑结构。

---

### Step4：StandardScaler

对数值特征进行标准化。

---

## 对比实验

比较：

### 方案A

```text
仅中位数填充
```

### 方案B

```text
中位数填充
+
Missing Indicator
```

观察：

- Accuracy
- AUC
- AMS

变化情况。

### 研究问题

缺失信息本身是否携带分类信息？

---

## 特征标准化

用于：

- Logistic Regression
- SVM
- MLP

方法：

```python
StandardScaler
```

---

## 类别变量处理

针对：

```python
PRI_jet_num
```

尝试：

- 原始数值
- One-Hot Encoding

比较模型效果。

---

# 五、基线模型

## Logistic Regression

作为Baseline。

评估指标：
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- AMS

---

# 六、超参数优化与交叉验证

## 调参目标

为了避免使用默认参数导致模型性能受限，本项目采用系统化超参数优化策略，而非人工经验调参。目标：
- 提高模型泛化能力
- 降低过拟合风险
- 获得最优ROC-AUC
- 为后续AMS优化提供基础
## 交叉验证

对于重点模型 XGBoost、MLP，采用：
```python
5-Fold Cross Validation
```
评估模型稳定性。记录：
- Mean ROC-AUC
- Std ROC-AUC
- 
用于评价模型泛化能力。

---

## Randomized Search

首先进行随机搜索：

```python
RandomizedSearchCV
```

搜索空间：
```python
{
    "max_depth": [3,5,7,9],
    "learning_rate": [0.01,0.05,0.1,0.2],
    "n_estimators": [100,300,500],
    "subsample": [0.6,0.8,1.0],
    "colsample_bytree": [0.6,0.8,1.0]
}
```

评价指标：

```python
ROC-AUC
```

目的：

快速定位高性能区域。

---

## Grid Search

在随机搜索结果附近进行局部网格搜索：

```python
GridSearchCV
```

进一步寻找最优参数组合。

---

## 调参结果分析

记录：

- 最优参数组合
- 最优CV得分
- ROC-AUC变化趋势

生成：

- Hyperparameter Importance
- Learning Curve
- Validation Curve

分析：

不同参数对模型性能的影响。

---

## 与AMS优化的关系

超参数搜索阶段：

```python
ROC-AUC
```

作为主要优化目标。

模型确定后：

进行：

```python
Threshold Search
```

寻找：

```python
Maximum AMS
```

最终获得用于物理发现任务的最优决策阈值。

# 六、模型比较实验

## Random Forest
特点：

- 非线性建模
- 解释性较强
- 对缺失值友好

---

## XGBoost（主模型）

项目核心模型。

调参内容：

```python
max_depth
learning_rate
n_estimators
subsample
colsample_bytree
```

目标：最大化：

- ROC-AUC
- AMS

---

## MLP

结构：
```text
Input
 ↓
Dense
 ↓
Dense
 ↓
Output
```
用于与树模型对比。

---

## 模型排行榜

统一比较：
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- AMS

---

# 七、Jet分组建模（创新实验）

## 实验动机

EDA发现不同 PRI_jet_num 组具有明显不同的Signal比例。

因此尝试：

### 方案A
全量训练

### 方案B

按Jet数量分别训练：
- Jet=0
- Jet=1
- Jet=2
- Jet=3

比较：
- AUC
- AMS

提升情况。

---

# 八、AMS优化

## AMS指标

Approximate Median Significance 用于衡量 Signal 发现显著性。

---

## 阈值优化

遍历：
```python
0.10 ~ 0.99
```

寻找：
AMS最大的分类阈值。

输出：
- Threshold-AMS曲线
- 最优阈值

---

# 九、模型可解释性分析

## 全局解释

使用：

```python
SHAP
```

生成：
- Feature Importance
- SHAP Summary Plot
- SHAP Beeswarm Plot

---

## 局部解释

随机抽取事件，解释：

- 为什么是Signal
- 为什么是Background

---

# 十、错误案例分析

## False Positive

预测：
Signal

实际：
Background

分析：
- 特征分布
- SHAP贡献

---

## False Negative

预测：
Background

实际：
Signal

分析：
- 特征分布
- SHAP贡献

---

# 十一、可信机器学习

## 鲁棒性实验

模拟探测器测量误差，添加噪声：
- ±5%
- ±10%
- ±20%

观察：
- Accuracy变化
- AUC变化
- AMS变化

---

## 稳定性实验

使用多个随机种子：

```python
42
2025
3407
114514
```
统计：平均值；标准差。

---

# 十二、系统Demo

系统名称：

# Higgs Event Discovery Dashboard

---

## 页面1：项目介绍

展示：
- 项目背景
- 数据集信息
- 模型信息

---

## 页面2：Event Display（MADAI Inspired）

随机抽取一个事件，根据：
- PRI_tau_phi
- PRI_lep_phi
- PRI_met_phi
- PRI_jet_num

生成碰撞事件可视化。展示：
- Tau
- Lepton
- Jet
- Missing Energy

说明：
该图为基于特征生成的解释性可视化，
并非真实探测器事件重建。

---

## 页面3：实时预测

显示：
- Signal Probability
- Prediction Label
- Ground Truth

---

## 页面4：SHAP解释

展示：当前事件，哪些特征推动其成为Signal。

---

## 页面5：AMS实验室

用户拖动阈值，实时更新：
- Precision
- Recall
- F1
- AMS

---

## 页面6：模型比较

展示：
- ROC Curve
- PR Curve
- Confusion Matrix
- Model Leaderboard

---

# 十三、项目创新点

## 创新点1
AMS优化，而非单纯Accuracy优化。

---

## 创新点2
Jet分组建模，利用物理拓扑结构信息。

---

## 创新点3
SHAP可解释分析，实现可信预测。

---

## 创新点4
MADAI风格Event Display，提升系统交互体验。

---

## 创新点5

可信机器学习，鲁棒性实验。

---

## 创新点6

PCA降维分析。

---

# 十四、课程知识覆盖

本项目覆盖课程核心内容：
- 线性分类（Logistic Regression）
- 主成分分析（PCA）
- 集成学习（Random Forest、XGBoost）
- 神经网络（MLP）
- 模型评估
- 可解释机器学习（SHAP）
- 可信机器学习（鲁棒性分析）
- 数据可视化
- 机器学习系统开发

---

# 十五、预期成果

## 学术成果

- 完整机器学习分类流程
- AMS优化策略
- SHAP解释分析
- 鲁棒性评估

## 系统成果

- Streamlit交互平台
- Event Display
- 实时预测系统
- AMS实验室