"""
Higgs Event Discovery Dashboard
================================
基于机器学习的希格斯玻色子信号识别与可解释分析系统。

运行：
    streamlit run app/dashboard.py
（在 code/ 目录下执行）

页面：
1. 项目介绍       背景 / 数据集 / 模型信息 / AMS 指标
2. 碰撞事件可视化  基于特征的解释性示意（非真实重建）
3. 实时预测       Signal 概率 / 可调阈值 / 预测对照（最终模型）
4. SHAP 解释      当前事件的特征贡献（最终模型）
5. AMS 实验室     拖动阈值实时更新 Precision/Recall/F1/AMS
6. 模型比较       ROC / PR / 混淆矩阵 / 排行榜

最终模型 = 物理权重训练 + 5 折 CV-Bagging 的 XGBoost 集成。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from app import loaders as L  # noqa: E402
from src import data as D  # noqa: E402
from src import models as M  # noqa: E402
from src.metrics import ams as ams_fn  # noqa: E402
from src.metrics import ams_at_threshold  # noqa: E402

# ----------------------------------------------------------------------------
# 页面配置与样式
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Higgs Event Discovery Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    /* 指标卡片：白底深字，保证清晰可读 */
    .stMetric { background: #ffffff; border: 1px solid #d8dee9;
        border-radius: 12px; padding: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.25); }
    .stMetric [data-testid="stMetricLabel"],
    .stMetric [data-testid="stMetricLabel"] * { color: #44506b !important; }
    .stMetric [data-testid="stMetricValue"] { color: #0e1117 !important; }
    .stMetric [data-testid="stMetricDelta"] { color: #1c2333 !important; }
    h1, h2, h3 { color: #f0f3f9; }
    .signal-badge { background:#d62728; color:white; padding:6px 16px;
        border-radius:20px; font-weight:700; font-size:20px; }
    .bkg-badge { background:#1f77b4; color:white; padding:6px 16px;
        border-radius:20px; font-weight:700; font-size:20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

COLOR_SIGNAL = "#d62728"
COLOR_BKG = "#1f77b4"

# 最终模型对外展示名称（物理权重训练 + 5 折 CV-Bagging 的 XGBoost 集成）
FINAL_NAME = "物理权重 + CV-Bagging"


def final_display_name(key: str) -> str:
    return FINAL_NAME if key == "final" else M.DISPLAY_NAME.get(key, key)


def get_merged_leaderboard() -> pd.DataFrame:
    """在基础模型排行榜中并入最终模型，按 AMS(best) 降序。"""
    lb = L.load_leaderboard()
    fe = L.load_final_eval()
    if lb.empty or not fe:
        return lb
    if (lb["模型"] == FINAL_NAME).any():
        return lb
    row = {c: fe.get(c, np.nan) for c in lb.columns}
    row["模型"] = FINAL_NAME
    lb = pd.concat([lb, pd.DataFrame([row])], ignore_index=True)
    return lb.sort_values("AMS(best)", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 共享状态：当前事件索引
# ----------------------------------------------------------------------------
def ensure_event_index(n: int) -> int:
    if "event_idx" not in st.session_state:
        st.session_state.event_idx = int(np.random.default_rng(0).integers(0, n))
    return st.session_state.event_idx


def predict_proba_single(model, row: pd.Series) -> float:
    X = row[C.FEATURE_COLS].to_frame().T
    X = X.astype(float)
    return float(model.predict_proba(X)[:, 1][0])


def final_proba_single(models: list, row: pd.Series) -> float:
    """最终模型集成：对 5 个加权 Bagging 模型的 Signal 概率取平均。"""
    X = row[C.FEATURE_COLS].to_frame().T.astype(float)
    return float(np.mean([m.predict_proba(X)[:, 1][0] for m in models]))


# ============================================================================
# 页面 1：项目介绍
# ============================================================================
def page_intro():
    st.title("Higgs Event Discovery Dashboard")
    st.caption("基于机器学习的希格斯玻色子信号识别与可解释分析系统")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("数据总量", "818,238", "事件")
    col2.metric("物理特征", "30", "DER + PRI")
    col3.metric("最佳 ROC-AUC", "0.913", "private")
    col4.metric("最佳 AMS", "3.69", "private")

    st.markdown("---")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("项目背景")
        st.markdown(
            """
            大型强子对撞机每天产生海量粒子碰撞事件，其中绝大多数是
            **背景事件（Background）**，只有极少数与 **希格斯玻色子信号（Signal）** 相关。

            本项目用机器学习从 30 个结构化物理特征中自动识别潜在的希格斯信号事件，
            并优化高能物理领域的发现显著性指标 **AMS（Approximate Median Significance）**，
            而非单纯优化 Accuracy。
            """
        )
        st.latex(
            r"\mathrm{AMS} = \sqrt{2\left[(s+b+b_r)\,\ln\!\left(1+\frac{s}{b+b_r}\right) - s\right]}"
        )
        st.markdown(
            r"""
            其中 $s$、$b$ 分别为被判定为信号区域内加权 TP 与 FP 计数，
            $b_r=10$ 为正则项。直观地：

            - **奖励 TP**：增大 $s$ 会提升 AMS；
            - **惩罚 FP**：增大 $b$ 会拉低 AMS；
            - 因此 AMS 偏好高决策阈值——只在高纯度区域才宣称“发现”。
            """
        )
        st.subheader("方法概览")
        st.markdown(
            """
            - **任务类型**：二分类（Signal vs Background）
            - **数据划分**：官方 KaggleSet（训练 25 万 / 公开 10 万 / 私有 45 万）
            - **模型**：逻辑回归 → 随机森林 → XGBoost（主模型） → MLP
            - **最终模型**：物理权重训练 + 5 折 CV-Bagging 的 XGBoost 集成
            - **调优**：RandomizedSearch + GridSearch + 5 折交叉验证
            - **可解释性**：SHAP（全局 + 局部）
            - **可信性**：噪声鲁棒性 + 随机种子稳定性
            """
        )
    with c2:
        st.subheader("数据集信息")
        st.markdown(
            """
            **来源**：CERN Open Data Portal —
            ATLAS Higgs Machine Learning Challenge 2014

            | 项目 | 值 |
            |---|---|
            | 总样本 | 818,238 |
            | Signal | 279,560 (34.2%) |
            | Background | 538,678 (65.8%) |
            | 缺失标记 | -999（结构性 / 重建失败）|
            """
        )
        lb = get_merged_leaderboard()
        if not lb.empty:
            st.subheader("模型排行榜（public）")
            show = lb[["模型", "ROC-AUC", "AMS(best)"]].copy()
            st.dataframe(show, hide_index=True, use_container_width=True)

    st.info("说明：本系统的「碰撞事件可视化」为基于特征生成的解释性示意图，并非真实探测器事件重建。")


# ============================================================================
# 页面 2：碰撞事件可视化
# ============================================================================
def page_event_display(pub: pd.DataFrame):
    st.title("碰撞事件可视化")
    st.caption("横向平面（transverse plane）示意图：箭头方向为方位角 φ，长度正比于横动量 pT")

    idx = ensure_event_index(len(pub))
    cols = st.columns([1, 1, 3])
    if cols[0].button("随机抽取事件", use_container_width=True):
        st.session_state.event_idx = int(np.random.default_rng().integers(0, len(pub)))
        idx = st.session_state.event_idx
    idx = cols[1].number_input("事件索引", 0, len(pub) - 1, idx, key="event_idx")

    row = pub.iloc[idx]
    truth = "Signal" if row[C.LABEL_COL] == "s" else "Background"

    objects = []

    def add_obj(name, phi, pt, color, symbol):
        if pt != C.MISSING_VALUE and not np.isnan(pt):
            objects.append((name, np.degrees(phi), pt, color, symbol))

    add_obj("τ (tau)", row["PRI_tau_phi"], row["PRI_tau_pt"], "#e377c2", "circle")
    add_obj("Lepton", row["PRI_lep_phi"], row["PRI_lep_pt"], "#2ca02c", "diamond")
    add_obj("MET (缺失能量)", row["PRI_met_phi"], row["PRI_met"], "#ff7f0e", "x")
    if row["PRI_jet_num"] >= 1:
        add_obj("Leading Jet", row["PRI_jet_leading_phi"], row["PRI_jet_leading_pt"], "#1f77b4", "square")
    if row["PRI_jet_num"] >= 2:
        add_obj("Subleading Jet", row["PRI_jet_subleading_phi"], row["PRI_jet_subleading_pt"], "#17becf", "square")

    c1, c2 = st.columns([1.4, 1])
    with c1:
        fig = go.Figure()
        max_pt = max([o[2] for o in objects]) if objects else 1
        for name, theta, pt, color, symbol in objects:
            fig.add_trace(go.Scatterpolar(
                r=[0, pt], theta=[theta, theta], mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=[0, 14], color=color, symbol=symbol),
                name=f"{name} (pT={pt:.1f})",
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, max_pt * 1.1], showticklabels=True, ticksuffix=" GeV"),
                angularaxis=dict(rotation=0, direction="counterclockwise"),
                bgcolor="#0e1117",
            ),
            showlegend=True, height=520, template="plotly_dark",
            title=f"事件 #{idx} — 横向平面粒子分布",
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("事件信息")
        if truth == "Signal":
            st.markdown(f'真实标签：<span class="signal-badge">SIGNAL</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'真实标签：<span class="bkg-badge">BACKGROUND</span>', unsafe_allow_html=True)
        st.metric("喷注数量 PRI_jet_num", int(row["PRI_jet_num"]))
        st.metric("DER_mass_MMC（希格斯质量估计）",
                  "缺失" if row["DER_mass_MMC"] == C.MISSING_VALUE else f"{row['DER_mass_MMC']:.1f} GeV")
        st.metric("MET（缺失横向能量）", f"{row['PRI_met']:.1f} GeV")
        st.markdown("**检测到的物理对象：**")
        for name, _, pt, _, _ in objects:
            st.write(f"- {name}: pT = {pt:.1f} GeV")

    st.session_state.current_row = idx


# ============================================================================
# 页面 3：实时预测
# ============================================================================
def page_prediction(pub: pd.DataFrame, final_models: list, final_meta: dict):
    st.title("实时预测")
    st.caption("最终模型：物理权重 + CV-Bagging（5 个 XGBoost 概率平均）")
    idx = ensure_event_index(len(pub))
    cols = st.columns([1, 1, 3])
    if cols[0].button("随机事件", use_container_width=True):
        st.session_state.event_idx = int(np.random.default_rng().integers(0, len(pub)))
    idx = cols[1].number_input("事件索引", 0, len(pub) - 1, st.session_state.event_idx, key="event_idx")

    row = pub.iloc[idx]
    proba = final_proba_single(final_models, row)

    best_t = float(final_meta.get("best_threshold", 0.5))
    threshold = st.slider(
        "决策阈值（拖动可调，默认值为最终模型在 public 集上的 AMS 最优阈值）",
        0.10, 0.99, round(best_t, 2), 0.01,
    )
    pred = "Signal" if proba >= threshold else "Background"
    truth = "Signal" if row[C.LABEL_COL] == "s" else "Background"
    correct = pred == truth

    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            title={"text": "Signal 概率"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_SIGNAL if proba >= threshold else COLOR_BKG},
                "threshold": {"line": {"color": "white", "width": 3},
                              "value": threshold * 100},
                "steps": [{"range": [0, threshold * 100], "color": "#26344d"},
                          {"range": [threshold * 100, 100], "color": "#4d2630"}],
            },
        ))
        fig.update_layout(height=300, template="plotly_dark", margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### 预测结果")
        badge = "signal-badge" if pred == "Signal" else "bkg-badge"
        st.markdown(f'预测：<span class="{badge}">{pred.upper()}</span>', unsafe_allow_html=True)
        st.write(f"当前决策阈值：**{threshold:.3f}**")
        st.write(f"Signal 概率：**{proba:.4f}**")
    with c3:
        st.markdown("### 真实标签对照")
        badge = "signal-badge" if truth == "Signal" else "bkg-badge"
        st.markdown(f'真实：<span class="{badge}">{truth.upper()}</span>', unsafe_allow_html=True)
        if correct:
            st.success("✅ 预测正确")
        else:
            st.error("❌ 预测错误")

    with st.expander("查看该事件的原始特征向量"):
        feat = row[C.FEATURE_COLS].to_frame("取值")
        feat["缺失"] = (feat["取值"] == C.MISSING_VALUE)
        st.dataframe(feat, use_container_width=True, height=400)

# ============================================================================
# 页面 4：SHAP 解释
# ============================================================================
def page_shap(pub: pd.DataFrame, final_models: list):
    st.title("SHAP 可解释性")
    st.caption("展示当前事件中，哪些特征把预测推向 Signal（红）或 Background（蓝）。"
               "贡献为最终模型（物理权重 + CV-Bagging）5 个成员模型的平均 SHAP。")

    from src.explain import tree_shap_values

    idx = ensure_event_index(len(pub))
    idx = st.number_input("事件索引", 0, len(pub) - 1, st.session_state.event_idx, key="event_idx")
    row = pub.iloc[idx]
    X = row[C.FEATURE_COLS].to_frame().T.astype(float)

    proba = final_proba_single(final_models, row)
    sv = np.mean([tree_shap_values(m, X)[0] for m in final_models], axis=0)
    contrib = pd.DataFrame({"feature": C.FEATURE_COLS, "shap": sv, "value": X.iloc[0].values})
    contrib["abs"] = contrib["shap"].abs()
    top = contrib.sort_values("abs", ascending=False).head(15).iloc[::-1]

    st.metric("当前事件 Signal 概率", f"{proba:.4f}")
    fig = go.Figure(go.Bar(
        x=top["shap"], y=top["feature"], orientation="h",
        marker_color=[COLOR_SIGNAL if v > 0 else COLOR_BKG for v in top["shap"]],
        text=[f"{v:+.3f}" for v in top["shap"]], textposition="outside",
        customdata=top["value"],
        hovertemplate="%{y}<br>特征取值=%{customdata:.3f}<br>SHAP=%{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=560, template="plotly_dark",
        title="Top 15 特征 SHAP 贡献（>0 推向 Signal，<0 推向 Background）",
        xaxis_title="SHAP 值（log-odds）",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("注：SHAP 值在 log-odds 空间，此处为最终集成各成员模型贡献的平均值。")


# ============================================================================
# 页面 5：AMS 实验室
# ============================================================================
def page_ams_lab():
    st.title("AMS 实验室")
    st.caption("基于最终模型（物理权重 + CV-Bagging）。拖动决策阈值，实时观察 "
               "Precision / Recall / F1 / AMS 的变化")

    scan = L.load_final_scan()
    if not scan:
        scan = L.load_ams_scan()
    if not scan:
        st.warning("缺少 final_model.npz，请先运行 scripts/09_ams_boost.py")
        return

    proba = scan["proba_public"]
    y = scan["y_public"]
    w = scan["w_public"]
    best_idx = int(np.argmax(scan["ams"]))
    best_t = float(scan["thresholds"][best_idx])

    threshold = st.slider("决策阈值", 0.10, 0.99, round(best_t, 2), 0.01)

    yp = (proba >= threshold).astype(int)
    prec = precision_score(y, yp, zero_division=0)
    rec = recall_score(y, yp, zero_division=0)
    f1 = f1_score(y, yp, zero_division=0)
    cur_ams = ams_at_threshold(y, proba, w, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{prec:.4f}")
    c2.metric("Recall", f"{rec:.4f}")
    c3.metric("F1", f"{f1:.4f}")
    c4.metric("AMS", f"{cur_ams:.4f}", f"{cur_ams - scan['ams'][best_idx]:+.3f} vs 最优")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scan["thresholds"], y=scan["ams"], name="AMS",
                             line=dict(color=COLOR_SIGNAL, width=3)))
    fig.add_vline(x=threshold, line=dict(color="white", dash="dash"))
    fig.add_vline(x=best_t, line=dict(color="gold", dash="dot"),
                  annotation_text=f"最优 {best_t:.2f}")
    fig.add_trace(go.Scatter(x=[threshold], y=[cur_ams], mode="markers",
                             marker=dict(size=14, color="white"), name="当前阈值"))
    fig.update_layout(height=420, template="plotly_dark",
                      title="Threshold–AMS 曲线", xaxis_title="决策阈值", yaxis_title="AMS")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    for arr, name, color in [(scan["precision"], "Precision", "#4c72b0"),
                             (scan["recall"], "Recall", "#55a868"),
                             (scan["f1"], "F1", "#937860")]:
        fig2.add_trace(go.Scatter(x=scan["thresholds"], y=arr, name=name, line=dict(color=color)))
    fig2.add_vline(x=threshold, line=dict(color="white", dash="dash"))
    fig2.update_layout(height=360, template="plotly_dark",
                       title="Precision / Recall / F1 随阈值变化", xaxis_title="决策阈值")
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# 页面 6：模型比较
# ============================================================================
def page_comparison():
    st.title("模型比较")
    ev = L.load_public_eval()
    lb = get_merged_leaderboard()
    if not ev:
        st.warning("缺少 public_eval.npz，请先运行 scripts/03_model_comparison.py")
        return
    y = ev["y"]
    w = ev["w"]

    # 并入最终模型（物理权重 + CV-Bagging）的 public 概率
    final_scan = L.load_final_scan()
    final_eval = L.load_final_eval()
    ev = {k: ev[k] for k in ev.files} if hasattr(ev, "files") else dict(ev)
    if final_scan:
        ev["final"] = final_scan["proba_public"]
    model_keys = [k for k in ev.keys() if k not in ("y", "w", "dummy")]

    tab1, tab2, tab3 = st.tabs(["ROC / PR 曲线", "混淆矩阵", "排行榜"])

    with tab1:
        c1, c2 = st.columns(2)
        fig_roc = go.Figure()
        fig_pr = go.Figure()
        for k in model_keys:
            proba = ev[k]
            fpr, tpr, _ = roc_curve(y, proba)
            from sklearn.metrics import auc as auc_fn
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{final_display_name(k)} ({auc_fn(fpr, tpr):.3f})"))
            pr_p, pr_r, _ = precision_recall_curve(y, proba)
            fig_pr.add_trace(go.Scatter(x=pr_r, y=pr_p, name=final_display_name(k)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="随机"))
        fig_roc.update_layout(template="plotly_dark", height=480, title="ROC 曲线",
                              xaxis_title="FPR", yaxis_title="TPR")
        fig_pr.update_layout(template="plotly_dark", height=480, title="Precision-Recall 曲线",
                             xaxis_title="Recall", yaxis_title="Precision")
        c1.plotly_chart(fig_roc, use_container_width=True)
        c2.plotly_chart(fig_pr, use_container_width=True)

    with tab2:
        default_idx = model_keys.index("final") if "final" in model_keys else (
            model_keys.index("xgboost") if "xgboost" in model_keys else 0)
        sel = st.selectbox("选择模型", model_keys, index=default_idx, format_func=final_display_name)
        # 默认阈值随所选模型而定：最终模型 0.95（AMS 最优），基础模型 0.84
        default_thr = round(float(final_eval.get("best_thr", 0.95)), 2) if sel == "final" else 0.84
        thr = st.slider("决策阈值", 0.10, 0.99, default_thr, 0.01)
        yp = (ev[sel] >= thr).astype(int)
        cm = confusion_matrix(y, yp)
        labels = ["Background", "Signal"]
        fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues",
                                   text=cm, texttemplate="%{text:,}", showscale=True))
        fig.update_layout(template="plotly_dark", height=460,
                          title=f"{final_display_name(sel)} 混淆矩阵 (阈值={thr:.2f})",
                          xaxis_title="预测标签", yaxis_title="真实标签")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if not lb.empty:
            st.dataframe(
                lb[["模型", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "AMS(best)", "best_thr"]],
                hide_index=True, use_container_width=True,
            )
            fig = go.Figure(go.Bar(x=lb["AMS(best)"], y=lb["模型"], orientation="h",
                                   marker_color=COLOR_SIGNAL))
            fig.update_layout(template="plotly_dark", height=360, title="AMS(best) 排行榜")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 主入口
# ============================================================================
def main():
    st.sidebar.title("⚛️ Higgs Dashboard")
    st.sidebar.markdown("基于机器学习的希格斯信号识别系统")
    page = st.sidebar.radio(
        "导航",
        ["1 · 项目介绍", "2 · 碰撞事件可视化", "3 · 实时预测",
         "4 · SHAP 解释", "5 · AMS 实验室", "6 · 模型比较"],
    )
    st.sidebar.markdown("---")
    st.sidebar.info("模型：物理权重 + CV-Bagging\n\nprivate AMS ≈ 3.69")

    if page.startswith("1"):
        page_intro()
    elif page.startswith("5"):
        page_ams_lab()
    elif page.startswith("6"):
        page_comparison()
    else:
        pub = L.load_public_df()
        if page.startswith("2"):
            page_event_display(pub)
            return
        # 页面 3 / 4 使用最终模型集成；若产物缺失则回退到调优 XGBoost
        final_models = L.load_final_models()
        final_meta = L.load_final_meta()
        if not final_models:
            st.warning("未找到最终集成模型，回退至调优 XGBoost。请运行 scripts/09_ams_boost.py 生成最终模型。")
            final_models = [L.load_model()]
            final_meta = {"best_threshold": L.load_json("ams_optimization.json").get("public_best_threshold", 0.5)}
        if page.startswith("3"):
            page_prediction(pub, final_models, final_meta)
        elif page.startswith("4"):
            page_shap(pub, final_models)


if __name__ == "__main__":
    main()
