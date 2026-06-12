"""绘图工具：统一风格、中文字体、保存助手。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # 无界面后端，脚本批量出图

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

# 颜色：signal / background 统一配色
COLOR_SIGNAL = "#d62728"
COLOR_BACKGROUND = "#1f77b4"
PALETTE = [COLOR_BACKGROUND, COLOR_SIGNAL]


def setup_style() -> None:
    """设置统一的 matplotlib 风格与中文字体。"""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.autolayout": False,
        }
    )
    # 中文字体：优先常见 Windows 字体，避免方块
    for font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
        try:
            plt.rcParams["font.sans-serif"] = [font]
            break
        except Exception:  # pragma: no cover
            continue
    plt.rcParams["axes.unicode_minus"] = False


def savefig(fig, name: str, subdir: str | None = None) -> Path:
    """保存图像到 outputs/figures，返回路径。"""
    out_dir = C.FIGURE_DIR if subdir is None else (C.FIGURE_DIR / subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------
# 评估曲线（被模型比较脚本与 Dashboard 复用）
# ----------------------------------------------------------------------------
def plot_roc_curves(curves: dict, title: str = "ROC 曲线对比"):
    """curves: {显示名: (fpr, tpr, auc)}。返回 fig。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="随机猜测")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_pr_curves(curves: dict, title: str = "Precision-Recall 曲线对比"):
    """curves: {显示名: (recall, precision, ap)}。返回 fig。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (recall, precision, ap) in curves.items():
        ax.plot(recall, precision, lw=1.8, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    return fig


def plot_confusion(cm, title: str = "混淆矩阵", labels=("Background", "Signal")):
    """cm: 2x2 numpy 数组。返回 fig。"""
    import numpy as np

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title(title)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(
                j, i, f"{cm[i, j]:,}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.grid(False)
    return fig


def plot_leaderboard(names, values, metric_name="AMS", title=None):
    """水平条形排行榜。"""
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(names) + 1.5))
    order = sorted(range(len(values)), key=lambda i: values[i])
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    bars = ax.barh(names, values, color="#4c72b0")
    bars[-1].set_color(COLOR_SIGNAL)  # 最优模型高亮
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=10)
    ax.set_xlabel(metric_name)
    ax.set_title(title or f"模型排行榜（按 {metric_name}）")
    return fig
