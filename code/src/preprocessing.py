"""
预处理模块：针对两类模型提供两套方案。

================================================================================
方案一：树模型（Random Forest / XGBoost）
--------------------------------------------------------------------------------
保留 -999 作为缺失标记。理由：
- XGBoost 原生支持缺失值，可学习每个分裂的最优默认方向（通过 missing=-999 告知）。
- 树模型对单调变换不敏感，-999 作为一个极端取值天然可被分裂利用，
  「缺失这件事本身」可能携带分类信息（结构性缺失 <-> jet 数量）。

================================================================================
方案二：线性 / 神经网络模型（Logistic Regression / MLP）
--------------------------------------------------------------------------------
-999 对线性模型是灾难性的异常值，必须处理。流水线：
  Step1 中位数填充（用训练集统计量）
  Step2 Missing Indicator（为含缺失的特征新增 0/1 指示列）—— 可开关，用于 A/B 实验
  Step3 PRI_jet_num One-Hot 编码（jet_0 / jet_1 / jet_2 / jet_3）—— 可开关
  Step4 StandardScaler（仅对连续数值特征）

A/B 实验对比「是否加入 Missing Indicator」，研究：缺失信息本身是否携带分类信息。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


# ============================================================================
# 方案一：树模型
# ============================================================================
def prepare_tree_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    树模型数据：原样保留 -999。

    XGBoost 训练时配合 ``missing=-999`` 使用；Random Forest 直接把 -999 当普通取值。
    这里返回拷贝以避免副作用。
    """
    return X.copy()


def tree_data_as_nan(X: pd.DataFrame) -> pd.DataFrame:
    """把 -999 替换为 np.nan（供需要 NaN 语义的 XGBoost 配置使用）。"""
    return X.replace(C.MISSING_VALUE, np.nan)


# ============================================================================
# 方案二：线性 / 神经网络模型
# ============================================================================
class HiggsLinearPreprocessor(BaseEstimator, TransformerMixin):
    """
    线性 / 神经网络模型的预处理器（sklearn 兼容）。

    参数
    ----
    add_missing_indicator : bool
        是否为含缺失特征添加 0/1 指示列（A/B 实验开关）。
    onehot_jet : bool
        是否对 PRI_jet_num 做 One-Hot；False 时保留原始数值并参与标准化。
    """

    def __init__(self, add_missing_indicator: bool = True, onehot_jet: bool = True):
        self.add_missing_indicator = add_missing_indicator
        self.onehot_jet = onehot_jet

    # ---- fit ----
    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        X = X.copy()
        self.feature_names_in_ = list(X.columns)

        # 连续数值特征 = 全部特征去掉 jet_num（若做 one-hot）
        if self.onehot_jet:
            self.numeric_features_ = [c for c in X.columns if c != C.JET_NUM_COL]
        else:
            self.numeric_features_ = list(X.columns)

        # 以 -999 -> nan 计算中位数
        X_nan = X.replace(C.MISSING_VALUE, np.nan)

        # 含缺失的特征（用于 missing indicator）
        self.missing_features_ = [
            c for c in self.numeric_features_ if X_nan[c].isna().any()
        ]

        # 训练集中位数
        self.medians_ = X_nan[self.numeric_features_].median()

        # 标准化器（在「填充后」的数据上拟合）
        X_imputed = X_nan[self.numeric_features_].fillna(self.medians_)
        self.scaler_ = StandardScaler().fit(X_imputed.to_numpy())

        # jet one-hot 取值固定 0/1/2/3，保证列稳定
        self.jet_categories_ = [0, 1, 2, 3]

        self._build_output_names()
        return self

    def _build_output_names(self) -> None:
        names = list(self.numeric_features_)
        if self.onehot_jet:
            names += [f"jet_{k}" for k in [0, 1, 2, 3]]
        if self.add_missing_indicator:
            names += [f"missing_{c}" for c in self.missing_features_]
        self.feature_names_out_ = names

    # ---- transform ----
    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        X = X.copy()
        X_nan = X.replace(C.MISSING_VALUE, np.nan)

        # 缺失指示（在填充之前计算）
        indicator_block = None
        if self.add_missing_indicator and self.missing_features_:
            indicator_block = (
                X_nan[self.missing_features_].isna().astype(float).to_numpy()
            )

        # 中位数填充 + 标准化
        X_imputed = X_nan[self.numeric_features_].fillna(self.medians_)
        numeric_block = self.scaler_.transform(X_imputed.to_numpy())

        blocks = [numeric_block]

        # jet one-hot
        if self.onehot_jet:
            jet_vals = X[C.JET_NUM_COL].to_numpy().astype(int)
            onehot = np.zeros((len(X), 4), dtype=float)
            for i, k in enumerate([0, 1, 2, 3]):
                onehot[:, i] = (jet_vals == k).astype(float)
            blocks.append(onehot)

        if indicator_block is not None:
            blocks.append(indicator_block)

        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None):  # noqa: D401
        return np.asarray(self.feature_names_out_, dtype=object)


def build_linear_preprocessor(
    add_missing_indicator: bool = True, onehot_jet: bool = True
) -> HiggsLinearPreprocessor:
    """工厂函数，便于脚本中按需构造。"""
    return HiggsLinearPreprocessor(
        add_missing_indicator=add_missing_indicator, onehot_jet=onehot_jet
    )


if __name__ == "__main__":
    # 自检：在训练集上拟合，检查输出维度
    sys.path.append(str(Path(__file__).resolve().parent))
    from utils import enable_utf8_stdout

    import data as D

    enable_utf8_stdout()
    splits = D.get_official_splits()
    Xtr = splits["train"].X

    for ind in (False, True):
        pre = build_linear_preprocessor(add_missing_indicator=ind, onehot_jet=True)
        Xt = pre.fit_transform(Xtr)
        print(
            f"add_missing_indicator={ind}: 输出维度={Xt.shape}, "
            f"含缺失特征数={len(pre.missing_features_)}, "
            f"总输出列数={len(pre.feature_names_out_)}"
        )
    print("缺失指示特征列表:", pre.missing_features_)
