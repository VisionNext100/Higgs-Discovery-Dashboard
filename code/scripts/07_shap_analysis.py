"""
07 - 模型可解释性（SHAP）与错误案例分析

主模型 XGBoost 用 TreeExplainer 计算 SHAP 值（精确、快速）。

全局解释：
- SHAP 特征重要性条形图（mean |SHAP|）
- SHAP Beeswarm（蜂群图，展示特征取值如何推动预测）

局部解释：
- 对「正确 Signal / 正确 Background / 假正例 FP / 假负例 FN」各取一个典型事件，
  用 waterfall 图解释单次预测。

错误案例分析（在 AMS 最优阈值下定义错误）：
- FP（把 Background 误判为 Signal）与 FN（把 Signal 漏判为 Background）的
  平均 SHAP 贡献，分析模型犯错的主要驱动特征。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import shap
from matplotlib import pyplot as plt

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import plotting as P  # noqa: E402
from src.explain import tree_shap_explanation  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402

SHAP_DIR = "shap"
N_SAMPLE = 5000


def load_threshold() -> float:
    path = C.REPORT_DIR / "ams_optimization.json"
    if path.exists():
        return float(json.loads(path.read_text(encoding="utf-8"))["public_best_threshold"])
    return 0.5


def waterfall_for(explanation, idx: int, title: str, fname: str) -> None:
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(explanation[idx], max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontweight="bold", y=1.02)
    P.savefig(fig, fname, SHAP_DIR)


def main() -> None:
    enable_utf8_stdout()
    P.setup_style()
    banner("07 - SHAP 可解释性与错误案例分析")

    model = joblib.load(C.MODEL_DIR / "xgboost_tuned.joblib")
    threshold = load_threshold()
    print(f"AMS 最优阈值 = {threshold:.3f}")

    splits = D.get_official_splits()
    public = splits["public"]

    rng = np.random.default_rng(C.RANDOM_SEED)
    idx = rng.choice(len(public.y), size=min(N_SAMPLE, len(public.y)), replace=False)
    X = public.X.iloc[idx].reset_index(drop=True)
    y = public.y[idx]
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    with timer("计算 SHAP 值 (XGBoost 原生 TreeSHAP)"):
        sv = tree_shap_explanation(model, X)  # Explanation 对象

    # 全局：bar
    plt.figure(figsize=(9, 7))
    shap.plots.bar(sv, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP 全局特征重要性（mean |SHAP|）", fontweight="bold", y=1.02)
    P.savefig(fig, "01_shap_bar.png", SHAP_DIR)

    # 全局：beeswarm
    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(sv, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Beeswarm（特征取值对 signal 预测的推动方向）", fontweight="bold", y=1.02)
    P.savefig(fig, "02_shap_beeswarm.png", SHAP_DIR)

    # 局部：四类典型事件
    def pick(cond):
        cand = np.where(cond)[0]
        return int(cand[0]) if len(cand) else None

    tp = pick((y == 1) & (pred == 1))
    tn = pick((y == 0) & (pred == 0))
    fp = pick((y == 0) & (pred == 1))
    fn = pick((y == 1) & (pred == 0))

    if tp is not None:
        waterfall_for(sv, tp, f"正确判定 Signal (proba={proba[tp]:.3f})", "03_local_TP.png")
    if tn is not None:
        waterfall_for(sv, tn, f"正确判定 Background (proba={proba[tn]:.3f})", "04_local_TN.png")
    if fp is not None:
        waterfall_for(sv, fp, f"假正例 FP：Background 被误判 (proba={proba[fp]:.3f})", "05_local_FP.png")
    if fn is not None:
        waterfall_for(sv, fn, f"假负例 FN：Signal 被漏判 (proba={proba[fn]:.3f})", "06_local_FN.png")

    # 错误案例：FP / FN 的平均 |SHAP| 驱动特征
    feat_names = list(X.columns)
    abs_sv = np.abs(sv.values)
    fp_mask = (y == 0) & (pred == 1)
    fn_mask = (y == 1) & (pred == 0)

    def top_drivers(mask, k=8):
        if mask.sum() == 0:
            return []
        mean_abs = abs_sv[mask].mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:k]
        return [(feat_names[i], float(mean_abs[i])) for i in order]

    fp_drivers = top_drivers(fp_mask)
    fn_drivers = top_drivers(fn_mask)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, drivers, title, color in [
        (axes[0], fp_drivers, f"FP 错误驱动特征 (n={int(fp_mask.sum())})", "#c44e52"),
        (axes[1], fn_drivers, f"FN 错误驱动特征 (n={int(fn_mask.sum())})", "#4c72b0"),
    ]:
        names = [d[0] for d in drivers][::-1]
        vals = [d[1] for d in drivers][::-1]
        ax.barh(names, vals, color=color)
        ax.set_xlabel("平均 |SHAP|")
        ax.set_title(title)
    fig.suptitle("错误案例的 SHAP 贡献分析", fontweight="bold")
    P.savefig(fig, "07_error_drivers.png", SHAP_DIR)

    # 保存全局重要性
    mean_abs_all = np.abs(sv.values).mean(axis=0)
    importance = sorted(zip(feat_names, mean_abs_all), key=lambda x: x[1], reverse=True)
    (C.REPORT_DIR / "shap_importance.json").write_text(
        json.dumps({
            "global_importance": {k: float(v) for k, v in importance},
            "fp_drivers": fp_drivers,
            "fn_drivers": fn_drivers,
            "n_fp": int(fp_mask.sum()),
            "n_fn": int(fn_mask.sum()),
            "threshold": threshold,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    banner("SHAP 分析完成", char="-")
    print("Top 5 全局重要特征:")
    for name, val in importance[:5]:
        print(f"  {name}: {val:.4f}")
    print(f"图表已保存至: {C.FIGURE_DIR / SHAP_DIR}")


if __name__ == "__main__":
    main()
