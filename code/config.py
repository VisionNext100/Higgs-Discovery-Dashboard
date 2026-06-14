"""
全局配置：路径、特征列表、随机种子等常量。

所有脚本和模块统一从这里导入常量，保证「单一事实来源」，方便复现。
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# 路径
# ----------------------------------------------------------------------------
# config.py 位于 code/ 下，项目根目录为其父目录
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "atlas-higgs-challenge-2014-v2.csv"

OUTPUT_DIR = CODE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"

for _d in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# 随机种子
# ----------------------------------------------------------------------------
RANDOM_SEED = 42
SEED_LIST = [42, 2025, 3407, 111111]  # 用于稳定性实验

# ----------------------------------------------------------------------------
# 缺失值占位符
# ----------------------------------------------------------------------------
MISSING_VALUE = -999.0

# ----------------------------------------------------------------------------
# 字段定义
# ----------------------------------------------------------------------------
# 不参与训练的列
ID_COL = "EventId"
LABEL_COL = "Label"
WEIGHT_COL = "Weight"
KAGGLE_SET_COL = "KaggleSet"
KAGGLE_WEIGHT_COL = "KaggleWeight"

NON_FEATURE_COLS = [
    ID_COL,
    WEIGHT_COL,
    LABEL_COL,
    KAGGLE_SET_COL,
    KAGGLE_WEIGHT_COL,
]

# 高阶物理特征（DER）
DER_FEATURES = [
    "DER_mass_MMC",
    "DER_mass_transverse_met_lep",
    "DER_mass_vis",
    "DER_pt_h",
    "DER_deltaeta_jet_jet",
    "DER_mass_jet_jet",
    "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep",
    "DER_pt_tot",
    "DER_sum_pt",
    "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality",
    "DER_lep_eta_centrality",
]

# 原始重建特征（PRI）
PRI_FEATURES = [
    "PRI_tau_pt",
    "PRI_tau_eta",
    "PRI_tau_phi",
    "PRI_lep_pt",
    "PRI_lep_eta",
    "PRI_lep_phi",
    "PRI_met",
    "PRI_met_phi",
    "PRI_met_sumet",
    "PRI_jet_num",
    "PRI_jet_leading_pt",
    "PRI_jet_leading_eta",
    "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt",
    "PRI_jet_subleading_eta",
    "PRI_jet_subleading_phi",
    "PRI_jet_all_pt",
]

FEATURE_COLS = DER_FEATURES + PRI_FEATURES  # 30 个训练特征

# 类别型 / 离散特征
JET_NUM_COL = "PRI_jet_num"

# 官方数据划分标记
SET_TRAIN = "t"  # 250,000 训练集
SET_PUBLIC = "b"  # 100,000 公开测试集（调参 / 验证）
SET_PRIVATE = "v"  # 450,000 私有测试集（最终评估）

# ----------------------------------------------------------------------------
# AMS 常量
# ----------------------------------------------------------------------------
AMS_B_REG = 10.0  # 正则项 b_reg，官方挑战赛设定为 10
