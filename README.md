# Higgs-Discovery-Dashboard
## ML Final Project

### 一、项目简介

2014 年，CERN 在 Kaggle 平台发起 HiggsML 挑战赛，旨在区分希格斯玻色子衰变信号与背景噪声，吸引了全球1785支队伍参赛，并催生了 XGBoost 等工业级工具。本项目基于该挑战赛，使用 ATLAS 实验的真实模拟数据，构建了一个希格斯玻色子信号识别与可解释分析系统，在复现经典方案的基础上实现了按喷注数量分组建模、SHAP 可解释性分析、物理权重训练 + CV-Bagging 等创新。主模型 XGBoost 在 45 万事件的 private 集达到 **ROC-AUC 0.913** 和 **AMS 3.60**，进一步通过物理权重训练和 CV-Bagging 将 **AMS** 在 private 集上提升至 **3.69**，对应挑战赛中前 10% 的水平。  
在合作之前，请移步下面的链接下载文件或阅读文档：  
数据集：http://opendata.cern.ch/record/328/files/atlas-higgs-challenge-2014-v2.csv.gz  
赛题文档：http://opendata.cern.ch/record/329/files/atlas-higgs-challenge-2014.pdf  
参考论文：https://proceedings.mlr.press/v42/

### 二、项目结构
需本地存放的已用“*”标明。
```text
Higgs-Discovery-Dashboard/
├── code/
│   ├── config.py               # 全局配置：路径、特征列表、随机种子、AMS 常量
│   ├── src/                    # 核心模块
│   │   ├── data.py             # 数据加载、按 KaggleSet 划分、AMS 权重
│   │   ├── preprocessing.py    # 两套预处理（树模型 / 线性模型 + Missing Indicator）
│   │   ├── metrics.py          # AMS、阈值搜索、统一评估
│   │   ├── models.py           # 模型工厂 + Pipeline + 训练评估
│   │   ├── explain.py          # XGBoost 原生 TreeSHAP（兼容 xgboost 3.x）
│   │   ├── plotting.py         # 统一绘图风格与曲线工具
│   │   └── utils.py            # UTF-8 输出、计时器等
│   ├── scripts/                # 顺序执行的实验脚本
│   │   ├── 01_eda.py                   # 探索性分析（缺失/喷注/PCA/相关性）
│   │   ├── 02_baseline.py              # 基准模型（Dummy + 逻辑回归）
│   │   ├── 03_model_comparison.py      # 5 模型比较 + 排行榜（保存模型）
│   │   ├── 04_hyperparameter_tuning.py # 随机搜索 + 网格搜索 + CV + 学习曲线
│   │   ├── 05_jet_grouped.py           # Jet 分组建模对照实验
│   │   ├── 06_ams_optimization.py      # AMS 阈值优化（生成 Dashboard 数据）
│   │   ├── 07_shap_analysis.py         # SHAP 全局/局部 + 错误案例
│   │   ├── 08_robustness.py            # 噪声鲁棒性 + 随机种子稳定性
│   │   └── 09_ams_boost.py             # AMS 提升实验（物理权重 + CV-Bagging）
│   ├── app/
│   │   ├── dashboard.py        # Streamlit 交互式系统
│   │   └── loaders.py          # Dashboard 数据、模型加载
│   └── outputs/
│       ├── figures/            # 所有图表，已按主题划分目录
│       ├── *models/            # .joblib 已训练模型
│       └── reports/            # .json/.csv 指标、.npz 概率缓存
├── data/
│   ├── *atlas-higgs-challenge-2014-v2.csv # 包含 818238 个事件的数据集
│   └── check_dataset.py                   # 数据集探测
├── .gitignore
├── LICENSE
├── requirements.txt
├── README.md
└── REPORT.md
```

### 三、快速复现
1\. 在 Anaconda Prompt 中完成下面的操作以配置环境：
```bash
conda create -n higgs python=3.10 -y
conda activate higgs
pip install -r requirements.txt
```
2\. 在 `code/` 目录下，按顺序执行：
```bash
python scripts/01_eda.py                   # EDA 图表
python scripts/02_baseline.py              # 基准
python scripts/03_model_comparison.py      # 训练并保存 5 个模型
python scripts/04_hyperparameter_tuning.py # 调优主模型
python scripts/05_jet_grouped.py           # Jet 分组实验
python scripts/06_ams_optimization.py      # AMS 阈值优化
python scripts/07_shap_analysis.py         # SHAP 可解释性
python scripts/08_robustness.py            # 鲁棒性与稳定性
python scripts/09_ams_boost.py             # AMS 提升实验，确定最终模型
```
脚本之间存在依赖关系：03 产出模型供 04、06、07 和 08 使用；09 复用 03 的随机森林、MLP 与 04 的最优超参数，并将最终的集成模型保存供 Dashboard 的“实时预测 / SHAP 解释”页面加载。  
3\. 启动交互式系统：
```bash
streamlit run app/dashboard.py
```  
由于部分交互功能需要读取的 `atlas-higgs-challenge-2014-v2.csv` 与模型权重太大，因此本项目不部署 Streamlit Cloud，仅在本地运行。

### 四、补充说明
1\. 训练流程：Training 上训练 + Training 内 CV 调参 → Public 选阈值/比模型 → Private 一次性评分。   
2\. AMS 权重：使用官方 `KaggleWeight`，即每个集合分别归一化。在子集上评估时按 `N_full / N_subset` 重新缩放（`src/metrics.rescale_weights`）。  
3\. 缺失值：-999 是占位符；树模型保留（XGBoost `missing=-999`），线性模型中位数填充 + 指示列。  
4\. SHAP 兼容性：因 shap 0.49 与 xgboost 3.x 的 `base_score` 解析不兼容，改用 XGBoost 原生 `pred_contribs` 计算精确 TreeSHAPs。  
5\. 交互面板中只有“项目介绍”展示 private 集的最终成绩，其余交互功能一律建立在 public 集上。因为这些功能本质是查看、调参、选阈值，如果在 private 上反复进行这些操作，相当于不断窥探测试集。