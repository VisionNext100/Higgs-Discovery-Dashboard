"""
08 - 可信机器学习：鲁棒性与稳定性实验

实验 A：鲁棒性（对抗测量噪声）
  探测器测量存在误差。对评估集特征注入相对高斯噪声 x*(1+N(0,σ))，
  σ ∈ {0%, 5%, 10%, 20%}，每个等级重复多次，观察 Accuracy / AUC / AMS 的退化。
  说明：-999（结构性缺失）与 PRI_jet_num（离散）不加噪。

实验 B：稳定性（随机种子）
  用多个随机种子重训 XGBoost（best params），统计 AUC / AMS 的均值与标准差，
  评估模型对随机性的敏感度。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import models as M  # noqa: E402
from src import plotting as P  # noqa: E402
from src.metrics import best_ams_threshold, evaluate_classification  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402

ROB_DIR = "robustness"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
NOISE_REPEATS = 5


def load_best_params() -> dict:
    path = C.REPORT_DIR / "xgb_best_params.json"
    return json.loads(path.read_text(encoding="utf-8"))["best_params"] if path.exists() else {}


def add_noise(X: pd.DataFrame, sigma: float, rng) -> pd.DataFrame:
    """对连续特征注入相对高斯噪声，保留 -999 与 PRI_jet_num。"""
    if sigma <= 0:
        return X
    Xn = X.copy()
    for col in X.columns:
        if col == C.JET_NUM_COL:
            continue
        vals = Xn[col].to_numpy(dtype=float)
        valid = vals != C.MISSING_VALUE
        noise = rng.normal(1.0, sigma, size=valid.sum())
        vals[valid] = vals[valid] * noise
        Xn[col] = vals
    return Xn


def robustness_experiment(model, public, threshold: float) -> pd.DataFrame:
    rows = []
    for sigma in NOISE_LEVELS:
        accs, aucs, amss = [], [], []
        reps = 1 if sigma == 0 else NOISE_REPEATS
        for r in range(reps):
            rng = np.random.default_rng(C.RANDOM_SEED + r)
            Xn = add_noise(public.X, sigma, rng)
            proba = model.predict_proba(Xn)[:, 1]
            m = evaluate_classification(public.y, proba, weights=public.w, threshold=threshold)
            accs.append(m["accuracy"])
            aucs.append(m["roc_auc"])
            amss.append(m["ams"])
        rows.append({
            "noise": sigma,
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
            "ams_mean": np.mean(amss), "ams_std": np.std(amss),
        })
        print(f"  σ={sigma:.0%}: ACC={np.mean(accs):.4f}, AUC={np.mean(aucs):.4f}, "
              f"AMS={np.mean(amss):.4f} (±{np.std(amss):.4f})")
    return pd.DataFrame(rows)


def stability_experiment(train, public, best_params) -> pd.DataFrame:
    rows = []
    for seed in C.SEED_LIST:
        with timer(f"种子 {seed} 重训 XGBoost"):
            params = dict(best_params)
            params["random_state"] = seed
            clf = M.make_estimator("xgboost", **params)
            clf.fit(train.X, train.y)
            proba = clf.predict_proba(public.X)[:, 1]
        m = evaluate_classification(public.y, proba, weights=public.w, threshold=0.5)
        _, ams_best, _, _ = best_ams_threshold(public.y, proba, public.w)
        rows.append({"seed": seed, "roc_auc": m["roc_auc"], "ams_best": ams_best})
        print(f"  种子 {seed}: AUC={m['roc_auc']:.4f}, AMS(best)={ams_best:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    enable_utf8_stdout()
    P.setup_style()
    banner("08 - 可信机器学习：鲁棒性与稳定性")

    model = joblib.load(C.MODEL_DIR / "xgboost_tuned.joblib")
    best_params = load_best_params()
    th_path = C.REPORT_DIR / "ams_optimization.json"
    threshold = float(json.loads(th_path.read_text(encoding="utf-8"))["public_best_threshold"]) \
        if th_path.exists() else 0.5

    splits = D.get_official_splits()
    train, public = splits["train"], splits["public"]

    banner("实验 A：鲁棒性（测量噪声）", char="-")
    rob = robustness_experiment(model, public, threshold)
    rob.to_csv(C.REPORT_DIR / "robustness_noise.csv", index=False, encoding="utf-8-sig")

    banner("实验 B：稳定性（随机种子）", char="-")
    stab = stability_experiment(train, public, best_params)
    stab.to_csv(C.REPORT_DIR / "robustness_seeds.csv", index=False, encoding="utf-8-sig")

    # 图 A：噪声退化曲线
    from matplotlib import pyplot as plt

    x = [f"{s:.0%}" for s in rob["noise"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, std, title, color) in zip(axes, [
        ("auc_mean", "auc_std", "ROC-AUC vs 噪声", "#4c72b0"),
        ("ams_mean", "ams_std", "AMS vs 噪声", P.COLOR_SIGNAL),
        ("acc_mean", "acc_std", "Accuracy vs 噪声", "#55a868"),
    ]):
        ax.errorbar(x, rob[col], yerr=rob[std], marker="o", capsize=4, color=color)
        ax.set_xlabel("注入噪声水平 σ")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("鲁棒性实验：模型对测量噪声的敏感度", fontweight="bold")
    P.savefig(fig, "01_noise_robustness.png", ROB_DIR)

    # 图 B：种子稳定性
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(stab["seed"].astype(str), stab["roc_auc"], color="#4c72b0")
    axes[0].set_ylim(0.905, 0.915)
    axes[0].set_title(f"ROC-AUC 跨种子 (μ={stab['roc_auc'].mean():.4f}, σ={stab['roc_auc'].std():.5f})")
    axes[0].set_xlabel("随机种子")
    axes[1].bar(stab["seed"].astype(str), stab["ams_best"], color=P.COLOR_SIGNAL)
    axes[1].set_title(f"AMS(best) 跨种子 (μ={stab['ams_best'].mean():.4f}, σ={stab['ams_best'].std():.4f})")
    axes[1].set_xlabel("随机种子")
    fig.suptitle("稳定性实验：模型对随机种子的敏感度", fontweight="bold")
    P.savefig(fig, "02_seed_stability.png", ROB_DIR)

    # 汇总 JSON
    auc0 = rob.loc[rob["noise"] == 0.0, "auc_mean"].iloc[0]
    auc20 = rob.loc[rob["noise"] == 0.20, "auc_mean"].iloc[0]
    (C.REPORT_DIR / "robustness_summary.json").write_text(
        json.dumps({
            "noise_robustness": rob.round(4).to_dict(orient="records"),
            "auc_drop_at_20pct": float(auc0 - auc20),
            "seed_stability": {
                "auc_mean": float(stab["roc_auc"].mean()),
                "auc_std": float(stab["roc_auc"].std()),
                "ams_mean": float(stab["ams_best"].mean()),
                "ams_std": float(stab["ams_best"].std()),
            },
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    banner("可信机器学习实验完成", char="-")
    print(f"20% 噪声下 AUC 下降: {auc0:.4f} -> {auc20:.4f} (Δ={auc0 - auc20:.4f})")
    print(f"跨种子 AUC: μ={stab['roc_auc'].mean():.4f}, σ={stab['roc_auc'].std():.5f}")


if __name__ == "__main__":
    main()
