"""
可解释性工具：用 XGBoost 原生 TreeSHAP 计算 SHAP 值。

背景：shap 库的 XGBoost 解析器在 xgboost 3.x 上存在 base_score 解析不兼容问题。
XGBoost 自身实现了 TreeSHAP，可通过 ``booster.predict(..., pred_contribs=True)``
直接得到精确的 SHAP 贡献（margin / log-odds 空间），无需依赖 shap 的解析器。

我们把结果包装成 ``shap.Explanation``，从而仍可使用 shap 的 beeswarm / bar / waterfall 等可视化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


def _booster_and_missing(model):
    """从 XGBClassifier 或 Booster 取出 booster 与 missing 值。"""
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        missing = model.get_params().get("missing", np.nan)
    else:
        booster = model
        missing = np.nan
    return booster, missing


def tree_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """返回 (n_samples, n_features) 的 SHAP 值（不含 bias 列）。"""
    booster, missing = _booster_and_missing(model)
    dm = xgb.DMatrix(X, missing=missing, feature_names=list(X.columns))
    contribs = booster.predict(dm, pred_contribs=True)
    return contribs[:, :-1]


def tree_shap_explanation(model, X: pd.DataFrame) -> shap.Explanation:
    """返回 shap.Explanation，可直接喂给 shap.plots.*。"""
    booster, missing = _booster_and_missing(model)
    dm = xgb.DMatrix(X, missing=missing, feature_names=list(X.columns))
    contribs = booster.predict(dm, pred_contribs=True)
    values = contribs[:, :-1]
    base = contribs[:, -1]
    return shap.Explanation(
        values=values,
        base_values=base,
        data=X.to_numpy(),
        feature_names=list(X.columns),
    )
