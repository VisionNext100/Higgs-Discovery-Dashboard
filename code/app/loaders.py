"""Dashboard 数据/模型加载（带 Streamlit 缓存）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402


@st.cache_resource(show_spinner="加载主模型 XGBoost ...")
def load_model():
    return joblib.load(C.MODEL_DIR / "xgboost_tuned.joblib")


@st.cache_resource(show_spinner=False)
def load_all_models() -> dict:
    models = {}
    for name in ["logreg", "random_forest", "xgboost", "mlp"]:
        p = C.MODEL_DIR / f"{name}.joblib"
        if p.exists():
            models[name] = joblib.load(p)
    return models


@st.cache_data(show_spinner="加载评估数据集 (public) ...")
def load_public_df() -> pd.DataFrame:
    """返回 public 集（含特征 + 标签 + 权重），用于事件浏览与预测。"""
    df = D.load_raw()
    pub = df[df[C.KAGGLE_SET_COL] == C.SET_PUBLIC].reset_index(drop=True)
    return pub


@st.cache_data(show_spinner=False)
def load_public_eval() -> dict:
    path = C.REPORT_DIR / "public_eval.npz"
    if not path.exists():
        return {}
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


@st.cache_data(show_spinner=False)
def load_ams_scan() -> dict:
    path = C.REPORT_DIR / "ams_scan.npz"
    if not path.exists():
        return {}
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    path = C.REPORT_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_leaderboard() -> pd.DataFrame:
    path = C.REPORT_DIR / "leaderboard.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()
