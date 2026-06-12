"""
06 - AMS 阈值优化

AMS（Approximate Median Significance）是物理发现任务的核心指标。模型给出的是
signal 概率，需要选择一个决策阈值把概率转成「是否纳入 signal 区域」。

本脚本针对调优后的主模型 XGBoost：
1. 在 public 集上遍历阈值 0.10~0.99，绘制 Threshold-AMS 曲线，定位最大 AMS。
2. 同时绘制 Precision / Recall / F1 随阈值的变化，解释「为什么 AMS 最优阈值远高于 0.5」。
3. 在 private 集（最终评估，只用一次）上用该最优阈值汇报最终 AMS。
4. 保存阈值扫描数据，供 Dashboard「AMS 实验室」页面实时使用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import plotting as P  # noqa: E402
from src.metrics import ams_at_threshold, best_ams_threshold, evaluate_classification  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402

AMS_DIR = "ams"


def main() -> None:
    enable_utf8_stdout()
    P.setup_style()
    banner("06 - AMS 阈值优化 (XGBoost 主模型)")

    model_path = C.MODEL_DIR / "xgboost_tuned.joblib"
    if not model_path.exists():
        raise FileNotFoundError("请先运行 04_hyperparameter_tuning.py 生成 xgboost_tuned.joblib")
    model = joblib.load(model_path)

    splits = D.get_official_splits()
    public, private = splits["public"], splits["private"]

    with timer("预测 public / private 概率"):
        proba_pub = model.predict_proba(public.X)[:, 1]
        proba_pri = model.predict_proba(private.X)[:, 1]

    thresholds = np.linspace(0.10, 0.99, 90)
    best_t, best_ams, _, ams_curve = best_ams_threshold(
        public.y, proba_pub, public.w, thresholds=thresholds
    )

    # Precision / Recall / F1 随阈值变化
    prec, rec, f1 = [], [], []
    for t in thresholds:
        yp = (proba_pub >= t).astype(int)
        prec.append(precision_score(public.y, yp, zero_division=0))
        rec.append(recall_score(public.y, yp, zero_division=0))
        f1.append(f1_score(public.y, yp, zero_division=0))
    prec, rec, f1 = map(np.array, (prec, rec, f1))

    # private 集最终评估（仅一次）
    ams_pri_at_best = ams_at_threshold(private.y, proba_pri, private.w, best_t)
    bt_pri, ams_pri_best, _, _ = best_ams_threshold(private.y, proba_pri, private.w)
    m_pri = evaluate_classification(private.y, proba_pri, weights=private.w, threshold=best_t)

    print(f"\npublic 最优阈值 = {best_t:.3f}, 对应 AMS = {best_ams:.4f}")
    print(f"private 用同一阈值({best_t:.3f}): AMS = {ams_pri_at_best:.4f}, "
          f"ROC-AUC = {m_pri['roc_auc']:.4f}")
    print(f"private 自身最优阈值 {bt_pri:.3f} 的 AMS = {ams_pri_best:.4f}（上界参考）")

    # 图1：Threshold-AMS
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresholds, ams_curve, color=P.COLOR_SIGNAL, lw=2, label="AMS (public)")
    ax.axvline(best_t, ls="--", color="gray")
    ax.scatter([best_t], [best_ams], color="black", zorder=5)
    ax.annotate(f"最优阈值={best_t:.2f}\nAMS={best_ams:.3f}",
                xy=(best_t, best_ams), xytext=(best_t - 0.28, best_ams - 0.4),
                arrowprops=dict(arrowstyle="->"))
    ax.axhline(0.5, color="lightgray", lw=0.8)
    ax.set_xlabel("决策阈值")
    ax.set_ylabel("AMS")
    ax.set_title("Threshold–AMS 曲线（XGBoost, public）")
    ax.legend()
    P.savefig(fig, "01_threshold_ams.png", AMS_DIR)

    # 图2：Precision/Recall/F1/AMS(归一化) 对比
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thresholds, prec, label="Precision", color="#4c72b0")
    ax.plot(thresholds, rec, label="Recall", color="#55a868")
    ax.plot(thresholds, f1, label="F1", color="#937860")
    ax2 = ax.twinx()
    ax2.plot(thresholds, ams_curve, label="AMS", color=P.COLOR_SIGNAL, lw=2, ls="-")
    ax2.set_ylabel("AMS", color=P.COLOR_SIGNAL)
    ax.axvline(best_t, ls="--", color="gray")
    ax.axvline(0.5, ls=":", color="lightgray")
    ax.set_xlabel("决策阈值")
    ax.set_ylabel("Precision / Recall / F1")
    ax.set_title("阈值对各指标的影响：AMS 偏好高纯度（高阈值）")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    P.savefig(fig, "02_threshold_metrics.png", AMS_DIR)

    # 保存数据（供 Dashboard 复用）
    np.savez_compressed(
        C.REPORT_DIR / "ams_scan.npz",
        thresholds=thresholds, ams=ams_curve,
        precision=prec, recall=rec, f1=f1,
        proba_public=proba_pub, y_public=public.y, w_public=public.w,
    )
    (C.REPORT_DIR / "ams_optimization.json").write_text(
        json.dumps({
            "public_best_threshold": float(best_t),
            "public_best_ams": float(best_ams),
            "private_ams_at_public_threshold": float(ams_pri_at_best),
            "private_best_ams": float(ams_pri_best),
            "private_roc_auc": float(m_pri["roc_auc"]),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    banner("AMS 优化完成", char="-")
    print(f"最终（private）AMS = {ams_pri_at_best:.4f} @ 阈值 {best_t:.3f}")


if __name__ == "__main__":
    main()
