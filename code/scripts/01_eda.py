"""
01 - 探索性数据分析（EDA）

产出（保存到 outputs/figures/eda/）：
- 标签分布
- 喷注数量分布 & 各组 signal 比例
- 缺失率条形图
- 特征相关性热图（Pearson）
- 关键特征与标签的相关性排序（Pearson / Spearman）
- PCA 2D / 3D 可视化 + 解释方差

EDA 的核心结论：数据具有强非线性、结构性缺失与喷注拓扑分组特性，
为后续「非线性模型 + 缺失指示 + Jet 分组建模」提供依据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import plotting as P  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402

EDA_DIR = "eda"


def plot_label_distribution(df: pd.DataFrame) -> None:
    counts = df[C.LABEL_COL].value_counts()
    n_sig = int(counts.get("s", 0))
    n_bkg = int(counts.get("b", 0))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        ["Background (b)", "Signal (s)"],
        [n_bkg, n_sig],
        color=[P.COLOR_BACKGROUND, P.COLOR_SIGNAL],
    )
    total = n_sig + n_bkg
    for bar, n in zip(bars, [n_bkg, n_sig]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{n:,}\n({n / total:.1%})",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("样本数")
    ax.set_title("标签分布（全数据 818,238 条）")
    ax.set_ylim(0, max(n_bkg, n_sig) * 1.15)
    P.savefig(fig, "01_label_distribution.png", EDA_DIR)


def plot_jet_analysis(df: pd.DataFrame) -> None:
    jet_counts = df[C.JET_NUM_COL].value_counts().sort_index()
    cross = pd.crosstab(df[C.JET_NUM_COL], df[C.LABEL_COL], normalize="index")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    jets = [0, 1, 2, 3]
    counts = [int(jet_counts.get(j, 0)) for j in jets]
    axes[0].bar([str(j) for j in jets], counts, color="#4c72b0")
    total = sum(counts)
    for i, c in enumerate(counts):
        axes[0].text(i, c, f"{c:,}\n({c / total:.1%})", ha="center", va="bottom")
    axes[0].set_xlabel("PRI_jet_num（喷注数量）")
    axes[0].set_ylabel("样本数")
    axes[0].set_title("喷注数量分布")
    axes[0].set_ylim(0, max(counts) * 1.15)

    sig_ratio = [cross.loc[j, "s"] * 100 for j in jets]
    bars = axes[1].bar([str(j) for j in jets], sig_ratio, color=P.COLOR_SIGNAL)
    for i, r in enumerate(sig_ratio):
        axes[1].text(i, r, f"{r:.1f}%", ha="center", va="bottom")
    axes[1].axhline(34.17, ls="--", color="gray", label="整体 Signal 比例 34.17%")
    axes[1].set_xlabel("PRI_jet_num（喷注数量）")
    axes[1].set_ylabel("Signal 占比 (%)")
    axes[1].set_title("各喷注组的 Signal 比例（分组建模动机）")
    axes[1].legend()
    P.savefig(fig, "02_jet_analysis.png", EDA_DIR)


def plot_missing_rates(df: pd.DataFrame) -> dict:
    rates = {}
    for col in C.FEATURE_COLS:
        rate = (df[col] == C.MISSING_VALUE).mean()
        if rate > 0:
            rates[col] = rate
    rates = dict(sorted(rates.items(), key=lambda kv: kv[1], reverse=True))

    fig, ax = plt.subplots(figsize=(9, 6))
    cols = list(rates.keys())
    vals = [rates[c] * 100 for c in cols]
    ax.barh(cols[::-1], vals[::-1], color="#dd8452")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("缺失率 (%)（-999 占比）")
    ax.set_title("各特征缺失率（含缺失的特征）")
    ax.set_xlim(0, 100)
    P.savefig(fig, "03_missing_rates.png", EDA_DIR)
    return rates


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    X = df[C.FEATURE_COLS].replace(C.MISSING_VALUE, np.nan)
    corr = X.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.7, "label": "Pearson 相关系数"},
        ax=ax,
    )
    ax.set_title("30 个物理特征 Pearson 相关性热图（-999 视为缺失）")
    P.savefig(fig, "04_correlation_heatmap.png", EDA_DIR)


def plot_label_correlation(df: pd.DataFrame) -> pd.DataFrame:
    X = df[C.FEATURE_COLS].replace(C.MISSING_VALUE, np.nan)
    y = D.to_label(df[C.LABEL_COL])
    pearson = X.apply(lambda col: col.corr(pd.Series(y), method="pearson"))
    spearman = X.apply(lambda col: col.corr(pd.Series(y), method="spearman"))
    corr_df = pd.DataFrame({"pearson": pearson, "spearman": spearman})
    corr_df["abs_pearson"] = corr_df["pearson"].abs()
    corr_df = corr_df.sort_values("abs_pearson", ascending=False)

    top = corr_df.head(15)
    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["pearson"][::-1], color="#55a868", label="Pearson")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top.index[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("与标签 (signal=1) 的相关系数")
    ax.set_title("Top 15 与标签相关的特征（Pearson）")
    P.savefig(fig, "05_label_correlation.png", EDA_DIR)
    return corr_df


def plot_pca(df: pd.DataFrame, n_sample: int = 30000) -> dict:
    rng = np.random.default_rng(C.RANDOM_SEED)
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    sub = df.iloc[idx]

    X = sub[C.FEATURE_COLS].replace(C.MISSING_VALUE, np.nan)
    X = X.fillna(X.median())
    y = D.to_label(sub[C.LABEL_COL])

    Xs = StandardScaler().fit_transform(X.to_numpy())

    pca = PCA(n_components=3, random_state=C.RANDOM_SEED).fit(Xs)
    Z = pca.transform(Xs)
    evr = pca.explained_variance_ratio_

    # 2D
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=4, alpha=0.25, c=P.COLOR_BACKGROUND, label="Background")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=4, alpha=0.25, c=P.COLOR_SIGNAL, label="Signal")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_title("PCA 二维投影（Signal 与 Background 高度重叠）")
    ax.legend(markerscale=3)
    P.savefig(fig, "06_pca_2d.png", EDA_DIR)

    # 3D
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], Z[y == 0, 2], s=3, alpha=0.2, c=P.COLOR_BACKGROUND, label="Background")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], Z[y == 1, 2], s=3, alpha=0.2, c=P.COLOR_SIGNAL, label="Signal")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax.set_zlabel(f"PC3 ({evr[2]:.1%})")
    ax.set_title("PCA 三维投影")
    ax.legend(markerscale=4)
    P.savefig(fig, "07_pca_3d.png", EDA_DIR)

    # 解释方差曲线（前 10 个主成分）
    pca_full = PCA(n_components=10, random_state=C.RANDOM_SEED).fit(Xs)
    fig, ax = plt.subplots(figsize=(8, 5))
    comps = np.arange(1, 11)
    ax.bar(comps, pca_full.explained_variance_ratio_, color="#4c72b0", alpha=0.7, label="单个主成分")
    ax.plot(comps, np.cumsum(pca_full.explained_variance_ratio_), "o-", color=P.COLOR_SIGNAL, label="累计")
    ax.set_xlabel("主成分序号")
    ax.set_ylabel("解释方差比例")
    ax.set_title("PCA 解释方差（前 10 个主成分）")
    ax.legend()
    P.savefig(fig, "08_pca_explained_variance.png", EDA_DIR)

    return {
        "pc1_evr": float(evr[0]),
        "pc2_evr": float(evr[1]),
        "pc3_evr": float(evr[2]),
        "top3_cumulative": float(np.sum(evr)),
    }


def main() -> None:
    enable_utf8_stdout()
    P.setup_style()
    banner("01 - 探索性数据分析 (EDA)")

    with timer("加载数据"):
        df = D.load_raw()
    print(f"数据规模: {len(df):,} 行 x {len(df.columns)} 列")

    summary: dict = {}
    with timer("标签 / 喷注分布"):
        plot_label_distribution(df)
        plot_jet_analysis(df)

    with timer("缺失率分析"):
        rates = plot_missing_rates(df)
        summary["missing_rates"] = {k: round(v, 4) for k, v in rates.items()}

    with timer("相关性分析"):
        plot_correlation_heatmap(df)
        corr_df = plot_label_correlation(df)
        summary["top_label_corr"] = corr_df.head(10)["pearson"].round(4).to_dict()

    with timer("PCA 降维分析"):
        pca_info = plot_pca(df)
        summary["pca"] = pca_info

    out = C.REPORT_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    banner("EDA 完成", char="-")
    print(f"图像已保存至: {C.FIGURE_DIR / EDA_DIR}")
    print(f"摘要已保存至: {out}")
    print(f"\nPCA 前三主成分累计解释方差: {pca_info['top3_cumulative']:.1%}")
    print("结论：低维空间中 Signal 与 Background 高度重叠 -> 需要非线性模型。")


if __name__ == "__main__":
    main()
