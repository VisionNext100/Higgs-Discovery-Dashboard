"""
统一建模模块：模型工厂 + sklearn Pipeline + 训练/评估封装。

设计目标：让基线、模型比较、调参、Jet 分组、AMS、SHAP 等脚本共享同一套
「构建 -> 训练 -> 评估」接口，保证一致性与可复现。

每个模型声明其所需的预处理类型：
- "linear"：使用 HiggsLinearPreprocessor（中位数填充 + 缺失指示 + One-Hot + 标准化）
- "tree"  ：保留 -999（XGBoost 用 missing=-999，RandomForest 直接当数值）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.data import DataSplit  # noqa: E402
from src.metrics import best_ams_threshold, evaluate_classification  # noqa: E402
from src.preprocessing import build_linear_preprocessor  # noqa: E402

# 模型 -> 预处理类型
PREPROCESS_TYPE: dict[str, str] = {
    "dummy": "tree",
    "logreg": "linear",
    "random_forest": "tree",
    "xgboost": "tree",
    "mlp": "linear",
}

# 模型中文显示名
DISPLAY_NAME: dict[str, str] = {
    "dummy": "基准(多数类)",
    "logreg": "逻辑回归",
    "random_forest": "随机森林",
    "xgboost": "XGBoost",
    "mlp": "MLP神经网络",
}


def make_estimator(name: str, **overrides):
    """根据名称构造一个未拟合的分类器。overrides 用于覆盖默认超参数。"""
    seed = C.RANDOM_SEED
    if name == "dummy":
        params = dict(strategy="prior")
        params.update(overrides)
        return DummyClassifier(**params)

    if name == "logreg":
        params = dict(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=-1)
        params.update(overrides)
        return LogisticRegression(**params)

    if name == "random_forest":
        params = dict(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=seed,
        )
        params.update(overrides)
        return RandomForestClassifier(**params)

    if name == "xgboost":
        params = dict(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            missing=C.MISSING_VALUE,
            n_jobs=-1,
            random_state=seed,
        )
        params.update(overrides)
        return XGBClassifier(**params)

    if name == "mlp":
        params = dict(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=1e-3,
            max_iter=60,
            early_stopping=True,
            n_iter_no_change=8,
            random_state=seed,
        )
        params.update(overrides)
        return MLPClassifier(**params)

    raise ValueError(f"未知模型: {name}")


def build_pipeline(
    name: str,
    add_missing_indicator: bool = True,
    onehot_jet: bool = True,
    **estimator_overrides,
) -> Pipeline:
    """构建「预处理 + 分类器」管道，所有模型统一为 Pipeline 接口。"""
    ptype = PREPROCESS_TYPE[name]
    est = make_estimator(name, **estimator_overrides)

    if ptype == "linear":
        pre = build_linear_preprocessor(
            add_missing_indicator=add_missing_indicator, onehot_jet=onehot_jet
        )
        return Pipeline([("pre", pre), ("clf", est)])
    # tree：保留 -999，不做预处理
    return Pipeline([("pre", "passthrough"), ("clf", est)])


@dataclass
class FitResult:
    name: str
    pipeline: Pipeline
    proba: np.ndarray  # 在评估集上的 signal 概率
    metrics_default: dict  # 阈值 0.5 的指标
    best_threshold: float  # 最大化 AMS 的阈值
    best_ams: float
    metrics_best: dict  # 最优阈值下的指标


def fit_and_evaluate(
    name: str,
    train: DataSplit,
    eval_split: DataSplit,
    pipeline: Pipeline | None = None,
    **build_kwargs,
) -> FitResult:
    """
    训练并在评估集上计算指标（含默认阈值与 AMS 最优阈值两套）。
    """
    if pipeline is None:
        pipeline = build_pipeline(name, **build_kwargs)
    pipeline.fit(train.X, train.y)
    proba = pipeline.predict_proba(eval_split.X)[:, 1]

    metrics_default = evaluate_classification(
        eval_split.y, proba, weights=eval_split.w, threshold=0.5
    )
    best_t, best_ams, _, _ = best_ams_threshold(eval_split.y, proba, eval_split.w)
    metrics_best = evaluate_classification(
        eval_split.y, proba, weights=eval_split.w, threshold=best_t
    )

    return FitResult(
        name=name,
        pipeline=pipeline,
        proba=proba,
        metrics_default=metrics_default,
        best_threshold=best_t,
        best_ams=best_ams,
        metrics_best=metrics_best,
    )
