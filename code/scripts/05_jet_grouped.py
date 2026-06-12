"""
05 - Jet 分组建模实验（创新点）

动机：EDA 显示不同 PRI_jet_num 组的 Signal 比例差异显著（25.3% ~ 51.0%），
且缺失模式由喷注数量决定（结构性缺失）。因此对比两种建模范式：

- 方案 A（全量）：单个 XGBoost 在全部训练数据上训练。
- 方案 B（分组）：按 jet 数量 {0,1,2,3} 分别训练 4 个 XGBoost 专家模型，
  在评估时各自预测自己的子集，再合并为完整预测。

公平比较：两方案使用相同的调优超参数；在合并后的 public 全集上计算 AUC 与 AMS(best)，
AMS 阈值在合并概率上全局搜索。额外报告各分组的单独 AUC。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import models as M  # noqa: E402
from src import plotting as P  # noqa: E402
from src.metrics import best_ams_threshold, evaluate_classification  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402

JET_DIR = "jet"


def load_best_params() -> dict:
    path = C.REPORT_DIR / "xgb_best_params.json"
    if path.exists():
        params = json.loads(path.read_text(encoding="utf-8"))["best_params"]
        print(f"使用调优超参数: {params}")
        return params
    print("未找到调优参数，使用默认 XGBoost 参数")
    return {}


def main() -> None:
    enable_utf8_stdout()
    P.setup_style()
    banner("05 - Jet 分组建模实验")

    splits = D.get_official_splits()
    train, public = splits["train"], splits["public"]
    best_params = load_best_params()

    jet_train = train.X[C.JET_NUM_COL].to_numpy().astype(int)
    jet_public = public.X[C.JET_NUM_COL].to_numpy().astype(int)
    # jet>=3 合并到组 3（数据中最大为 3）
    jet_train = np.clip(jet_train, 0, 3)
    jet_public = np.clip(jet_public, 0, 3)

    # ---- 方案 A：全量模型 ----
    with timer("方案A：全量 XGBoost"):
        full = M.make_estimator("xgboost", **best_params)
        full.fit(train.X, train.y)
        proba_full = full.predict_proba(public.X)[:, 1]
    auc_a = roc_auc_score(public.y, proba_full)
    bt_a, ams_a, _, _ = best_ams_threshold(public.y, proba_full, public.w)

    # ---- 方案 B：分组专家模型 ----
    proba_grouped = np.zeros(len(public.y), dtype=float)
    group_rows = []
    group_models = {}
    for g in [0, 1, 2, 3]:
        tr_mask = jet_train == g
        pb_mask = jet_public == g
        with timer(f"方案B：jet={g} 专家模型 (训练{tr_mask.sum():,} / 评估{pb_mask.sum():,})"):
            clf = M.make_estimator("xgboost", **best_params)
            clf.fit(train.X[tr_mask], train.y[tr_mask])
            p = clf.predict_proba(public.X[pb_mask])[:, 1]
        proba_grouped[pb_mask] = p
        group_models[g] = clf

        # 各组单独 AUC（组内 signal 全为同类时跳过）
        yg = public.y[pb_mask]
        auc_g = roc_auc_score(yg, p) if len(np.unique(yg)) > 1 else float("nan")
        group_rows.append({
            "jet组": g,
            "训练样本": int(tr_mask.sum()),
            "评估样本": int(pb_mask.sum()),
            "组内Signal占比": float(yg.mean()),
            "组内AUC": auc_g,
        })

    auc_b = roc_auc_score(public.y, proba_grouped)
    bt_b, ams_b, _, _ = best_ams_threshold(public.y, proba_grouped, public.w)

    # ---- 结果汇总 ----
    cmp = pd.DataFrame([
        {"方案": "A 全量模型", "ROC-AUC": auc_a, "AMS(best)": ams_a, "best_thr": bt_a},
        {"方案": "B 分组专家模型", "ROC-AUC": auc_b, "AMS(best)": ams_b, "best_thr": bt_b},
    ])
    group_df = pd.DataFrame(group_rows)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + "=" * 70)
    print("方案 A vs 方案 B（评估集 = public）")
    print("=" * 70)
    print(cmp.to_string(index=False))
    print("\n各 jet 组细节：")
    print(group_df.to_string(index=False))

    # 保存
    cmp.to_csv(C.REPORT_DIR / "jet_grouped_comparison.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(C.REPORT_DIR / "jet_grouped_per_group.csv", index=False, encoding="utf-8-sig")
    for g, clf in group_models.items():
        joblib.dump(clf, C.MODEL_DIR / f"xgboost_jet{g}.joblib")
    (C.REPORT_DIR / "jet_grouped_summary.json").write_text(
        json.dumps({
            "approach_A": {"auc": float(auc_a), "ams_best": float(ams_a), "threshold": float(bt_a)},
            "approach_B": {"auc": float(auc_b), "ams_best": float(ams_b), "threshold": float(bt_b)},
            "ams_gain": float(ams_b - ams_a),
            "auc_gain": float(auc_b - auc_a),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 图1：AMS / AUC 对比
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(["A 全量", "B 分组"], [ams_a, ams_b], color=["#4c72b0", P.COLOR_SIGNAL])
    for i, v in enumerate([ams_a, ams_b]):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom")
    axes[0].set_title("AMS(best) 对比")
    axes[0].set_ylabel("AMS")
    axes[1].bar(["A 全量", "B 分组"], [auc_a, auc_b], color=["#4c72b0", P.COLOR_SIGNAL])
    for i, v in enumerate([auc_a, auc_b]):
        axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom")
    axes[1].set_title("ROC-AUC 对比")
    axes[1].set_ylim(0.88, 0.92)
    fig.suptitle("Jet 分组建模 vs 全量建模", fontweight="bold")
    P.savefig(fig, "01_approach_comparison.png", JET_DIR)

    # 图2：各组 AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(group_df["jet组"].astype(str), group_df["组内AUC"], color="#55a868")
    for i, v in enumerate(group_df["组内AUC"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_xlabel("PRI_jet_num 组")
    ax.set_ylabel("组内 ROC-AUC")
    ax.set_title("各 jet 专家模型的组内 AUC")
    ax.set_ylim(0.7, 0.95)
    P.savefig(fig, "02_per_group_auc.png", JET_DIR)

    banner("Jet 分组实验完成", char="-")
    gain = ams_b - ams_a
    print(f"AMS 提升: {ams_a:.4f} -> {ams_b:.4f}  (Δ={gain:+.4f})")
    print(f"AUC 提升: {auc_a:.4f} -> {auc_b:.4f}  (Δ={auc_b - auc_a:+.4f})")


if __name__ == "__main__":
    main()
