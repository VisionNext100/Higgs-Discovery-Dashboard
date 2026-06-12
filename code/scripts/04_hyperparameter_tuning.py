"""
04 - 超参数优化与交叉验证（针对主模型 XGBoost）

策略（系统化调优，而非人工试参）：
1. RandomizedSearchCV  在大搜索空间中快速定位高性能区域（scoring=ROC-AUC, 3 折）
2. GridSearchCV         在随机搜索最优点附近做局部精细网格（5 折）
3. 5-Fold CV            报告最终模型的 Mean / Std ROC-AUC，评估稳定性
4. 学习曲线              诊断偏差/方差（是否过拟合、是否需要更多数据）
5. 验证曲线              观察单个超参数（max_depth）对性能的影响
6. 超参数重要性          基于随机搜索结果，分析各参数与 CV 得分的相关性

产出：
- outputs/reports/xgb_best_params.json
- outputs/models/xgboost_tuned.joblib（在全量训练集上重训）
- 图：随机搜索分布、验证曲线、学习曲线、超参数重要性
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve,
    validation_curve,
)

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import joblib  # noqa: E402

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import models as M  # noqa: E402
from src import plotting as P  # noqa: E402
from src.metrics import best_ams_threshold, evaluate_classification  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402

TUNE_DIR = "tuning"
# XGBoost 单模型并行 4，CV 并行 8 -> 共用满 32 核且不过度超额订阅
XGB_NJOBS = 4
SEARCH_NJOBS = 8


def base_xgb():
    return M.make_estimator("xgboost", n_jobs=XGB_NJOBS)


def run_randomized_search(X, y):
    param_dist = {
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.29),
        "n_estimators": randint(200, 600),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
        "reg_lambda": uniform(0.0, 5.0),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=base_xgb(),
        param_distributions=param_dist,
        n_iter=30,
        scoring="roc_auc",
        cv=cv,
        n_jobs=SEARCH_NJOBS,
        random_state=C.RANDOM_SEED,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X, y)
    return search


def run_grid_search(X, y, center: dict):
    """在随机搜索最优点附近做局部网格。"""
    md = int(center["max_depth"])
    lr = float(center["learning_rate"])
    ne = int(center["n_estimators"])
    grid = {
        "max_depth": sorted({max(3, md - 1), md, md + 1}),
        "learning_rate": sorted({round(max(0.01, lr * 0.5), 4), round(lr, 4), round(lr * 1.5, 4)}),
        "n_estimators": sorted({max(100, ne - 100), ne, ne + 100}),
    }
    fixed = {
        "subsample": float(center["subsample"]),
        "colsample_bytree": float(center["colsample_bytree"]),
        "min_child_weight": int(center["min_child_weight"]),
        "reg_lambda": float(center["reg_lambda"]),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
    search = GridSearchCV(
        estimator=base_xgb().set_params(**fixed),
        param_grid=grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=SEARCH_NJOBS,
        verbose=1,
    )
    search.fit(X, y)
    return search, grid


def plot_hyperparam_importance(cv_results: dict):
    df = pd.DataFrame(cv_results)
    param_cols = [c for c in df.columns if c.startswith("param_")]
    score = df["mean_test_score"].astype(float)
    importances = {}
    for col in param_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > 2 and vals.std() > 0:
            importances[col.replace("param_", "")] = abs(np.corrcoef(vals, score)[0, 1])
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    fig, ax = plt_bar(importances)
    return fig, importances


def plt_bar(importances: dict):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(importances.keys())[::-1]
    vals = [importances[n] for n in names]
    ax.barh(names, vals, color="#937860")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.set_xlabel("|与 CV ROC-AUC 的相关系数|")
    ax.set_title("超参数重要性（基于随机搜索结果）")
    return fig, ax


def plot_validation_curve(X, y):
    param_range = [3, 4, 5, 6, 7, 8, 9]
    train_scores, val_scores = validation_curve(
        base_xgb(),
        X, y,
        param_name="max_depth",
        param_range=param_range,
        scoring="roc_auc",
        cv=3,
        n_jobs=SEARCH_NJOBS,
    )
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(param_range, train_scores.mean(1), "o-", color="#4c72b0", label="训练集")
    ax.fill_between(param_range, train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#4c72b0")
    ax.plot(param_range, val_scores.mean(1), "o-", color=P.COLOR_SIGNAL, label="验证集(CV)")
    ax.fill_between(param_range, val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15, color=P.COLOR_SIGNAL)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("验证曲线：树深度对性能的影响")
    ax.legend()
    return fig


def plot_learning_curve(estimator, X, y):
    sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        train_sizes=np.linspace(0.1, 1.0, 6),
        scoring="roc_auc",
        cv=3,
        n_jobs=SEARCH_NJOBS,
        random_state=C.RANDOM_SEED,
    )
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_scores.mean(1), "o-", color="#4c72b0", label="训练集")
    ax.fill_between(sizes, train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#4c72b0")
    ax.plot(sizes, val_scores.mean(1), "o-", color=P.COLOR_SIGNAL, label="验证集(CV)")
    ax.fill_between(sizes, val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15, color=P.COLOR_SIGNAL)
    ax.set_xlabel("训练样本数")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("学习曲线（XGBoost 调优后）")
    ax.legend()
    return fig


def main() -> None:
    enable_utf8_stdout()
    P.setup_style()
    banner("04 - 超参数优化与交叉验证 (XGBoost)")

    splits = D.get_official_splits()
    train, public = splits["train"], splits["public"]
    X, y = train.X, train.y

    with timer("RandomizedSearchCV (30 组 x 3 折)"):
        rsearch = run_randomized_search(X, y)
    print(f"随机搜索最优 CV ROC-AUC = {rsearch.best_score_:.4f}")
    print(f"随机搜索最优参数: {rsearch.best_params_}")

    with timer("GridSearchCV (局部精细, 5 折)"):
        gsearch, grid = run_grid_search(X, y, rsearch.best_params_)
    print(f"网格搜索空间: {grid}")
    print(f"网格搜索最优 CV ROC-AUC = {gsearch.best_score_:.4f}")
    print(f"网格搜索最优参数: {gsearch.best_params_}")

    # 合并最优参数
    best_params = dict(rsearch.best_params_)
    best_params.update(gsearch.best_params_)
    best_params = {k: (int(v) if isinstance(v, (np.integer,)) else
                       float(v) if isinstance(v, (np.floating,)) else v)
                   for k, v in best_params.items()}

    # 用最优参数构建最终模型并做 5 折 CV 报告稳定性
    final = M.make_estimator("xgboost", n_jobs=XGB_NJOBS, **best_params)
    from sklearn.model_selection import cross_val_score

    with timer("最终模型 5 折交叉验证"):
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
        scores = cross_val_score(final, X, y, scoring="roc_auc", cv=cv5, n_jobs=SEARCH_NJOBS)
    print(f"5 折 CV ROC-AUC: 均值={scores.mean():.4f}, 标准差={scores.std():.4f}")

    # 在全量训练集重训，评估 public
    with timer("全量训练集重训最终模型"):
        final.fit(X, y)
    proba = final.predict_proba(public.X)[:, 1]
    m = evaluate_classification(public.y, proba, weights=public.w, threshold=0.5)
    best_t, best_ams, _, _ = best_ams_threshold(public.y, proba, public.w)
    print(f"\n调优后 XGBoost @public: ROC-AUC={m['roc_auc']:.4f}, "
          f"AMS@0.5={m['ams']:.4f}, AMS(best)={best_ams:.4f} @thr={best_t:.2f}")

    # 保存模型与参数
    joblib.dump(final, C.MODEL_DIR / "xgboost_tuned.joblib")
    out = {
        "best_params": best_params,
        "cv5_roc_auc_mean": float(scores.mean()),
        "cv5_roc_auc_std": float(scores.std()),
        "public_roc_auc": float(m["roc_auc"]),
        "public_ams_best": float(best_ams),
        "public_best_threshold": float(best_t),
        "randomized_best_score": float(rsearch.best_score_),
        "grid_best_score": float(gsearch.best_score_),
    }
    (C.REPORT_DIR / "xgb_best_params.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 图
    with timer("绘制超参数重要性 / 验证曲线 / 学习曲线"):
        fig_imp, importances = plot_hyperparam_importance(rsearch.cv_results_)
        P.savefig(fig_imp, "01_hyperparam_importance.png", TUNE_DIR)
        P.savefig(plot_validation_curve(X, y), "02_validation_curve_maxdepth.png", TUNE_DIR)
        P.savefig(plot_learning_curve(final, X, y), "03_learning_curve.png", TUNE_DIR)

    banner("超参数优化完成", char="-")
    print(f"超参数重要性: {importances}")
    print(f"最优参数已保存: {C.REPORT_DIR / 'xgb_best_params.json'}")
    print(f"调优模型已保存: {C.MODEL_DIR / 'xgboost_tuned.joblib'}")


if __name__ == "__main__":
    main()
