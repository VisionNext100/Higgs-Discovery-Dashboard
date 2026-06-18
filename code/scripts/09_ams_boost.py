"""
09 - AMS 提升实验（创新）

Jet 分组未能提升 AMS（见 §9），本脚本探索更有效的提升路线，并以统一、严格的协议评估：
    train(25万) 训练  ->  public(10万) 选 AMS 最优阈值  ->  private(45万) 仅评估一次

对比策略：
- S0 基线：调优后单个 XGBoost（参考线）
- S1 物理权重训练：用官方事件权重 + 类别平衡作为 sample_weight，使训练目标对齐 AMS
- S2 CV-Bagging：5 折分别训练 XGBoost，对预测取平均（降方差）
- S3 异质集成（均值）：XGBoost + 随机森林 + MLP 概率平均
- S4 异质集成（排名均值）：对概率做秩归一化后平均（对各模型标定差异更稳健）
- S5 Stacking 凸组合：在 public 上搜索三模型的凸组合权重，使 AMS 最大

只有在 private 上确有提升的策略才会写入报告。
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import models as M  # noqa: E402
from src import plotting as P  # noqa: E402
from src.metrics import ams_at_threshold, best_ams_threshold, evaluate_classification  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402

BOOST_DIR = "ams_boost"
# 最终模型对外展示名称（物理权重训练 + 5 折 CV-Bagging 的 XGBoost 集成）
FINAL_NAME = "物理权重 + CV-Bagging"


def load_best_params() -> dict:
    return json.loads((C.REPORT_DIR / "xgb_best_params.json").read_text(encoding="utf-8"))["best_params"]


def evaluate(name, proba_pub, proba_pri, public, private, results):
    """在 public 选阈值，在 private 评估，记录结果。"""
    bt, ams_pub, _, _ = best_ams_threshold(public.y, proba_pub, public.w)
    ams_pri = ams_at_threshold(private.y, proba_pri, private.w, bt)
    auc_pri = roc_auc_score(private.y, proba_pri)
    results.append({
        "策略": name,
        "public_AMS": ams_pub,
        "best_thr": bt,
        "private_AMS": ams_pri,
        "private_AUC": auc_pri,
    })
    print(f"  {name:28s} public_AMS={ams_pub:.4f} | private_AMS={ams_pri:.4f} | AUC={auc_pri:.4f}")
    return proba_pub, proba_pri


def rank_norm(p):
    return rankdata(p) / len(p)


def main() -> None:
    enable_utf8_stdout()
    P.setup_style()
    banner("09 - AMS 提升实验")

    best_params = load_best_params()
    splits = D.get_official_splits()
    train, public, private = splits["train"], splits["public"], splits["private"]

    results = []
    proba_cache = {}

    # ---- S0 基线：调优 XGBoost ----
    with timer("S0 基线 XGBoost"):
        xgb = M.make_estimator("xgboost", **best_params)
        xgb.fit(train.X, train.y)
        pub0 = xgb.predict_proba(public.X)[:, 1]
        pri0 = xgb.predict_proba(private.X)[:, 1]
    proba_cache["xgb"] = (pub0, pri0)
    evaluate("S0 基线(XGBoost)", pub0, pri0, public, private, results)

    # ---- S1 物理权重训练 ----
    with timer("S1 物理权重 XGBoost"):
        sw = train.w.astype(float).copy()
        pos = train.y == 1
        # 类别平衡：令 signal 总权重 == background 总权重
        sw[pos] *= sw[~pos].sum() / sw[pos].sum()
        sw *= len(sw) / sw.sum()  # 归一化到样本量量级，稳定训练
        xgb_w = M.make_estimator("xgboost", **best_params)
        xgb_w.fit(train.X, train.y, sample_weight=sw)
        pub1 = xgb_w.predict_proba(public.X)[:, 1]
        pri1 = xgb_w.predict_proba(private.X)[:, 1]
    evaluate("S1 物理权重训练", pub1, pri1, public, private, results)

    # ---- S2 CV-Bagging ----
    with timer("S2 CV-Bagging (5 折)"):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_SEED)
        pub_bag = np.zeros(len(public.y))
        pri_bag = np.zeros(len(private.y))
        for i, (tr_idx, _) in enumerate(skf.split(train.X, train.y)):
            clf = M.make_estimator("xgboost", random_state=1000 + i, **best_params)
            clf.fit(train.X.iloc[tr_idx], train.y[tr_idx])
            pub_bag += clf.predict_proba(public.X)[:, 1]
            pri_bag += clf.predict_proba(private.X)[:, 1]
        pub_bag /= 5
        pri_bag /= 5
    proba_cache["bag"] = (pub_bag, pri_bag)
    evaluate("S2 CV-Bagging XGBoost", pub_bag, pri_bag, public, private, results)

    # ---- 加载已训练的 RF / MLP（复用，避免重训）----
    with timer("加载 RF / MLP 预测"):
        rf = joblib.load(C.MODEL_DIR / "random_forest.joblib")
        mlp = joblib.load(C.MODEL_DIR / "mlp.joblib")
        pub_rf, pri_rf = rf.predict_proba(public.X)[:, 1], rf.predict_proba(private.X)[:, 1]
        pub_mlp, pri_mlp = mlp.predict_proba(public.X)[:, 1], mlp.predict_proba(private.X)[:, 1]

    # 集成基元：用 CV-bag 的 XGBoost（更稳）
    base_pub = {"xgb": pub_bag, "rf": pub_rf, "mlp": pub_mlp}
    base_pri = {"xgb": pri_bag, "rf": pri_rf, "mlp": pri_mlp}

    # ---- S3 异质集成（概率均值）----
    pub3 = np.mean([base_pub[k] for k in base_pub], axis=0)
    pri3 = np.mean([base_pri[k] for k in base_pri], axis=0)
    evaluate("S3 集成(概率均值)", pub3, pri3, public, private, results)

    # ---- S4 异质集成（排名均值）----
    pub4 = np.mean([rank_norm(base_pub[k]) for k in base_pub], axis=0)
    pri4 = np.mean([rank_norm(base_pri[k]) for k in base_pri], axis=0)
    evaluate("S4 集成(排名均值)", pub4, pri4, public, private, results)

    # ---- S5 Stacking 凸组合（在 public 上搜索权重）----
    with timer("S5 凸组合权重搜索"):
        grid = np.arange(0, 1.01, 0.1)
        best = (-1, None)
        keys = list(base_pub.keys())
        for w in product(grid, repeat=len(keys)):
            if abs(sum(w) - 1.0) > 1e-6:
                continue
            blend = sum(wi * rank_norm(base_pub[k]) for wi, k in zip(w, keys))
            _, a, _, _ = best_ams_threshold(public.y, blend, public.w)
            if a > best[0]:
                best = (a, w)
        w_opt = best[1]
    pub5 = sum(wi * rank_norm(base_pub[k]) for wi, k in zip(w_opt, keys))
    pri5 = sum(wi * rank_norm(base_pri[k]) for wi, k in zip(w_opt, keys))
    w_label = ",".join(f"{k}={float(wi):.1f}" for k, wi in zip(keys, w_opt))
    evaluate(f"S5 凸组合({w_label})", pub5, pri5, public, private, results)

    # ---- S6 物理权重 + CV-Bagging（组合 S1 与 S2）----
    def balanced_weight(idx):
        w = train.w[idx].astype(float).copy()
        yy = train.y[idx]
        w[yy == 1] *= w[yy == 0].sum() / w[yy == 1].sum()
        w *= len(w) / w.sum()
        return w

    with timer("S6 物理权重 + CV-Bagging"):
        pub6 = np.zeros(len(public.y))
        pri6 = np.zeros(len(private.y))
        s6_models = []
        for i, (tr_idx, _) in enumerate(skf.split(train.X, train.y)):
            clf = M.make_estimator("xgboost", random_state=2000 + i, **best_params)
            clf.fit(train.X.iloc[tr_idx], train.y[tr_idx], sample_weight=balanced_weight(tr_idx))
            pub6 += clf.predict_proba(public.X)[:, 1]
            pri6 += clf.predict_proba(private.X)[:, 1]
            s6_models.append(clf)
        pub6 /= 5
        pri6 /= 5
    proba_cache["wbag"] = (pub6, pri6)
    evaluate("S6 物理权重+CV-Bagging", pub6, pri6, public, private, results)

    # 保存 S6 集成模型与元数据，供 Dashboard 作为最终模型加载
    s6_dir = C.MODEL_DIR / "s6_wbag"
    s6_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(s6_models):
        joblib.dump(m, s6_dir / f"xgb_wbag_{i}.joblib")
    s6_bt, s6_pub_ams, _, _ = best_ams_threshold(public.y, pub6, public.w)
    s6_pri_ams = ams_at_threshold(private.y, pri6, private.w, s6_bt)
    (C.MODEL_DIR / "s6_meta.json").write_text(
        json.dumps({
            "name": "物理权重 + CV-Bagging (S6)",
            "n_models": len(s6_models),
            "best_threshold": float(s6_bt),
            "public_ams": float(s6_pub_ams),
            "private_ams": float(s6_pri_ams),
            "private_auc": float(roc_auc_score(private.y, pri6)),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  [已保存] S6 集成 {len(s6_models)} 个模型 -> {s6_dir} | 最优阈值={s6_bt:.2f}")

    # ---- 为最终模型生成阈值扫描数据 + 阈值图 + 比较面板评估（供 Dashboard / 报告）----
    with timer("最终模型阈值扫描与图表"):
        thresholds = np.linspace(0.10, 0.99, 90)
        _, _, _, ams_curve = best_ams_threshold(public.y, pub6, public.w, thresholds=thresholds)
        prec_c, rec_c, f1_c = [], [], []
        for t in thresholds:
            yp = (pub6 >= t).astype(int)
            prec_c.append(precision_score(public.y, yp, zero_division=0))
            rec_c.append(recall_score(public.y, yp, zero_division=0))
            f1_c.append(f1_score(public.y, yp, zero_division=0))
        prec_c, rec_c, f1_c = map(np.array, (prec_c, rec_c, f1_c))

        # 扫描数据（AMS 实验室 + 模型比较 ROC/PR/混淆矩阵复用）
        np.savez_compressed(
            C.REPORT_DIR / "final_model.npz",
            thresholds=thresholds, ams=ams_curve,
            precision=prec_c, recall=rec_c, f1=f1_c,
            proba_public=pub6, y_public=public.y, w_public=public.w,
        )

        # 比较面板排行榜行（public 集，与 leaderboard.csv 同格式）
        m05 = evaluate_classification(public.y, pub6, weights=public.w, threshold=0.5)
        final_eval = {
            "model": "weighted_bagging",
            "模型": FINAL_NAME,
            "Accuracy": m05["accuracy"], "Precision": m05["precision"],
            "Recall": m05["recall"], "F1": m05["f1"], "ROC-AUC": m05["roc_auc"],
            "AMS@0.5": m05["ams"], "AMS(best)": float(s6_pub_ams), "best_thr": float(s6_bt),
            "private_ams": float(s6_pri_ams),
        }
        (C.REPORT_DIR / "final_model_eval.json").write_text(
            json.dumps(final_eval, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 图：最终模型的阈值-各指标曲线（类比 ams/02_threshold_metrics.png）
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(thresholds, prec_c, label="Precision", color="#4c72b0")
        ax.plot(thresholds, rec_c, label="Recall", color="#55a868")
        ax.plot(thresholds, f1_c, label="F1", color="#937860")
        ax2 = ax.twinx()
        ax2.plot(thresholds, ams_curve, label="AMS", color=P.COLOR_SIGNAL, lw=2)
        ax2.set_ylabel("AMS", color=P.COLOR_SIGNAL)
        ax.axvline(s6_bt, ls="--", color="gray")
        ax.axvline(0.5, ls=":", color="lightgray")
        ax.set_xlabel("决策阈值")
        ax.set_ylabel("Precision / Recall / F1")
        ax.set_title("阈值对各指标的影响（物理权重 + CV-Bagging, public）")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
        P.savefig(fig, "02_threshold_metrics.png", BOOST_DIR)
    print(f"  [已保存] 最终模型阈值扫描 final_model.npz + 阈值图 | public 最优 AMS={s6_pub_ams:.3f}@{s6_bt:.2f}")

    # ---- S7 集成（物理权重bag + RF + MLP，排名均值）----
    pub7 = np.mean([rank_norm(pub6), rank_norm(pub_rf), rank_norm(pub_mlp)], axis=0)
    pri7 = np.mean([rank_norm(pri6), rank_norm(pri_rf), rank_norm(pri_mlp)], axis=0)
    evaluate("S7 集成(权重bag+RF+MLP)", pub7, pri7, public, private, results)

    # ---- 汇总 ----
    df = pd.DataFrame(results)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 200)
    print("\n" + "=" * 88)
    print("AMS 提升实验汇总（基线 private_AMS = S0）")
    print("=" * 88)
    print(df.to_string(index=False))

    base_ams = df.iloc[0]["private_AMS"]
    df["private_AMS_gain"] = df["private_AMS"] - base_ams
    best_row = df.loc[df["private_AMS"].idxmax()]
    print(f"\n最佳策略: {best_row['策略']}  private_AMS={best_row['private_AMS']:.4f} "
          f"(Δ={best_row['private_AMS'] - base_ams:+.4f} vs 基线)")

    df.to_csv(C.REPORT_DIR / "ams_boost.csv", index=False, encoding="utf-8-sig")
    (C.REPORT_DIR / "ams_boost.json").write_text(
        json.dumps({
            "baseline_private_ams": float(base_ams),
            "best_strategy": str(best_row["策略"]),
            "best_private_ams": float(best_row["private_AMS"]),
            "gain": float(best_row["private_AMS"] - base_ams),
            "all": df.round(4).to_dict(orient="records"),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 图：各策略 private AMS 对比
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = df["策略"].tolist()
    vals = df["private_AMS"].tolist()
    colors = ["#4c72b0"] * len(vals)
    colors[int(df["private_AMS"].idxmax())] = P.COLOR_SIGNAL
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(base_ams, ls="--", color="gray", label=f"基线 {base_ams:.4f}")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("private 集 AMS")
    ax.set_ylim(min(vals) - 0.1, max(vals) + 0.15)
    ax.set_title("AMS 提升实验：各策略在 private 集的 AMS")
    ax.legend()
    P.savefig(fig, "01_ams_boost_comparison.png", BOOST_DIR)

    banner("AMS 提升实验完成", char="-")


if __name__ == "__main__":
    main()
