"""
评估指标模块。

核心是 AMS（Approximate Median Significance），这是 ATLAS Higgs Challenge 的官方
评分指标，用于衡量发现 signal 的统计显著性：

    AMS = sqrt( 2 * ( (s + b + b_reg) * ln(1 + s / (b + b_reg)) - s ) )

其中：
- s     : 被判为 signal 的 *真实 signal* 的权重之和（加权 TP）
- b     : 被判为 signal 的 *真实 background* 的权重之和（加权 FP）
- b_reg : 正则常数，官方设为 10

重要细节：权重必须保持「整体归一化」。当我们只在一个子集（如 CV 验证折）上评估时，
权重需要按 N_full / N_subset 重新缩放，否则 s、b 数值偏小，AMS 失真。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


def ams(s: float, b: float, b_reg: float = C.AMS_B_REG) -> float:
    """根据加权 signal/background 计数计算 AMS。"""
    if s < 0 or b < 0:
        return 0.0
    radicand = 2.0 * ((s + b + b_reg) * np.log(1.0 + s / (b + b_reg)) - s)
    if radicand < 0:
        return 0.0
    return float(np.sqrt(radicand))


def rescale_weights(weights: np.ndarray, n_full: int, n_subset: int) -> np.ndarray:
    """
    把子集权重按 N_full / N_subset 缩放，使其与整体归一化一致。

    例：从 25 万训练集中切出 5 万做验证，应乘以 250000 / 50000 = 5。
    """
    if n_subset == 0:
        return weights
    return weights * (float(n_full) / float(n_subset))


def ams_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    b_reg: float = C.AMS_B_REG,
) -> float:
    """给定阈值，计算 AMS。"""
    pred_pos = y_proba >= threshold
    s = float(weights[pred_pos & (y_true == 1)].sum())
    b = float(weights[pred_pos & (y_true == 0)].sum())
    return ams(s, b, b_reg=b_reg)


def best_ams_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray | None = None,
    b_reg: float = C.AMS_B_REG,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    扫描阈值寻找最大 AMS。

    返回: (best_threshold, best_ams, thresholds, ams_curve)
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.99, 90)

    pos_mask = y_true == 1
    neg_mask = y_true == 0
    ams_curve = np.empty_like(thresholds, dtype=float)
    for i, t in enumerate(thresholds):
        pred_pos = y_proba >= t
        s = float(weights[pred_pos & pos_mask].sum())
        b = float(weights[pred_pos & neg_mask].sum())
        ams_curve[i] = ams(s, b, b_reg=b_reg)

    best_idx = int(np.argmax(ams_curve))
    return float(thresholds[best_idx]), float(ams_curve[best_idx]), thresholds, ams_curve


def evaluate_classification(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    weights: np.ndarray | None = None,
    threshold: float = 0.5,
    b_reg: float = C.AMS_B_REG,
) -> dict[str, float]:
    """
    统一的分类评估，返回常用指标 + AMS。

    AMS 仅在提供 weights 时计算。
    """
    y_pred = (y_proba >= threshold).astype(int)
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "threshold": threshold,
    }
    if weights is not None:
        metrics["ams"] = ams_at_threshold(y_true, y_proba, weights, threshold, b_reg=b_reg)
    return metrics
