"""
数据加载与划分模块。

职责：
- 读取原始 CSV
- 把 Label 转成 0/1（s -> 1 signal, b -> 0 background）
- 按官方 KaggleSet 划分训练 / 公开测试 / 私有测试
- 返回特征矩阵 X、标签 y、AMS 权重 w

关于权重：
- ``Weight``        原始权重（全数据归一化）
- ``KaggleWeight``  官方按每个 KaggleSet 分别重新归一化后的权重，AMS 计算应使用它。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 允许以脚本方式直接运行时找到 config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


@dataclass
class DataSplit:
    """一个数据子集：特征、标签、AMS 权重。"""

    X: pd.DataFrame
    y: np.ndarray  # 0/1
    w: np.ndarray  # AMS 权重 (KaggleWeight)
    name: str

    def __repr__(self) -> str:  # pragma: no cover - 便于调试
        n = len(self.y)
        pos = int(self.y.sum())
        return (
            f"DataSplit(name={self.name!r}, n={n:,}, "
            f"signal={pos:,} ({pos / n:.2%}), background={n - pos:,})"
        )


def load_raw(path: Path | str = C.DATA_FILE) -> pd.DataFrame:
    """读取原始 CSV，返回 DataFrame。"""
    df = pd.read_csv(path)
    return df


def to_label(series: pd.Series) -> np.ndarray:
    """把 's'/'b' 标签转成 1/0。"""
    return (series == "s").astype(int).to_numpy()


def make_split(df: pd.DataFrame, name: str, features: list[str] | None = None) -> DataSplit:
    """从 DataFrame 构造一个 DataSplit。"""
    features = features or C.FEATURE_COLS
    X = df[features].copy()
    y = to_label(df[C.LABEL_COL])
    w = df[C.KAGGLE_WEIGHT_COL].to_numpy()
    return DataSplit(X=X, y=y, w=w, name=name)


def get_official_splits(
    df: pd.DataFrame | None = None,
    features: list[str] | None = None,
) -> dict[str, DataSplit]:
    """
    按官方 KaggleSet 返回三个划分。

    返回 dict: {"train": ..., "public": ..., "private": ...}
    """
    if df is None:
        df = load_raw()

    mapping = {
        "train": C.SET_TRAIN,
        "public": C.SET_PUBLIC,
        "private": C.SET_PRIVATE,
    }
    splits: dict[str, DataSplit] = {}
    for name, tag in mapping.items():
        sub = df[df[C.KAGGLE_SET_COL] == tag]
        splits[name] = make_split(sub, name=name, features=features)
    return splits


if __name__ == "__main__":
    # 快速自检：加载数据并打印三个划分的规模
    print("加载数据中 ...")
    df = load_raw()
    print(f"原始数据: {len(df):,} 行 x {len(df.columns)} 列")
    splits = get_official_splits(df)
    for s in splits.values():
        print("  ", s)
        # 检查权重归一化：每个集合内 signal/background 权重之和
        sig_w = s.w[s.y == 1].sum()
        bkg_w = s.w[s.y == 0].sum()
        print(f"       signal权重和={sig_w:.3f}, background权重和={bkg_w:.3f}")
