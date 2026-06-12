# Higgs Event Discovery Dashboard — 代码说明

基于机器学习的希格斯玻色子信号识别与可解释分析系统。

主模型 XGBoost 在官方 private 集（45 万事件）达到 **ROC-AUC 0.913 / AMS 3.60**。

---

## 环境准备

```bash
conda create -n higgs python=3.10 -y
conda activate higgs
pip install -r requirements.txt
```

依赖：numpy, pandas, scipy, scikit-learn, xgboost, shap, matplotlib, seaborn, plotly, streamlit, joblib, tqdm。

---

## 目录结构

```text
code/
├── config.py               # 全局配置：路径、特征列表、随机种子、AMS 常量
├── requirements.txt
├── REPORT.md               # 完整项目报告（背景-方法-结果-结论）
├── src/                    # 核心模块
│   ├── data.py             # 数据加载、按 KaggleSet 划分、AMS 权重
│   ├── preprocessing.py    # 两套预处理（树模型 / 线性模型 + Missing Indicator）
│   ├── metrics.py          # AMS、阈值搜索、统一评估
│   ├── models.py           # 模型工厂 + Pipeline + 训练评估
│   ├── explain.py          # XGBoost 原生 TreeSHAP（兼容 xgboost 3.x）
│   ├── plotting.py         # 统一绘图风格与曲线工具
│   └── utils.py            # UTF-8 输出、计时器等
├── scripts/                # 顺序执行的实验脚本
│   ├── 01_eda.py                  # 探索性分析（缺失/喷注/PCA/相关性）
│   ├── 02_baseline.py             # 基准模型（Dummy + 逻辑回归）
│   ├── 03_model_comparison.py     # 5 模型比较 + 排行榜（保存模型）
│   ├── 04_hyperparameter_tuning.py# 随机搜索 + 网格搜索 + CV + 学习曲线
│   ├── 05_jet_grouped.py          # Jet 分组建模对照实验
│   ├── 06_ams_optimization.py     # AMS 阈值优化（生成 Dashboard 数据）
│   ├── 07_shap_analysis.py        # SHAP 全局/局部 + 错误案例
│   └── 08_robustness.py           # 噪声鲁棒性 + 随机种子稳定性
├── app/
│   ├── dashboard.py        # Streamlit 交互式系统（6 页面）
│   └── loaders.py          # Dashboard 数据/模型加载（带缓存）
└── outputs/
    ├── figures/            # 所有图表（按主题分子目录）
    ├── models/             # 已训练模型 (.joblib)
    └── reports/            # 指标 JSON/CSV、概率缓存 (.npz)
```

---

## 复现流程

> 在 `code/` 目录下，按顺序运行（脚本之间有依赖：03 产出模型供 04/06/07/08 使用）。

```bash
python scripts/01_eda.py                   # EDA 图表
python scripts/02_baseline.py              # 基准
python scripts/03_model_comparison.py      # 训练并保存 5 个模型
python scripts/04_hyperparameter_tuning.py # 调优主模型（耗时较长，~15 min）
python scripts/05_jet_grouped.py           # Jet 分组实验
python scripts/06_ams_optimization.py      # AMS 阈值优化
python scripts/07_shap_analysis.py         # SHAP 可解释性
python scripts/08_robustness.py            # 鲁棒性与稳定性
```

启动交互式系统：

```bash
streamlit run app/dashboard.py
```

---

## 关键设计说明

- **数据划分**：采用官方 `KaggleSet`（训练 25 万 / public 10 万 / private 45 万），
  阈值在 public 上选定，private 仅用于最终评估一次，避免信息泄漏。
- **AMS 权重**：使用官方 `KaggleWeight`（每个集合分别归一化）。在子集上评估时按
  `N_full / N_subset` 重新缩放（`src/metrics.rescale_weights`）。
- **缺失值**：-999 是占位符；树模型保留（XGBoost `missing=-999`），线性模型中位数填充 + 指示列。
- **SHAP 兼容性**：因 shap 0.49 与 xgboost 3.x 的 `base_score` 解析不兼容，
  改用 XGBoost 原生 `pred_contribs` 计算精确 TreeSHAP（`src/explain.py`）。
```
