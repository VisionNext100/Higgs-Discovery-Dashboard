"""
03 - 模型比较实验

训练并比较 5 个模型（评估集 = official public）：
- 基准(多数类) Dummy
- 逻辑回归 Logistic Regression（线性基线）
- 随机森林 Random Forest（集成 / Bagging）
- XGBoost（集成 / Boosting，主模型）
- MLP 神经网络

为什么选这些算法（写入报告用）：
- 逻辑回归：可解释的线性基线，衡量问题的线性可分程度。
- 随机森林：Bagging 集成，天然处理非线性与缺失，方差低、稳健。
- XGBoost：梯度提升，业界处理结构化/表格数据的 SOTA，原生支持缺失值，本项目主模型。
- MLP：神经网络代表，验证「通用函数逼近器」在该任务上的表现并与树模型对比。

产出：
- outputs/models/{name}.joblib       已训练管道（供 Dashboard / 后续脚本复用）
- outputs/reports/public_eval.npz     各模型在 public 集的概率（+ y, w）
- outputs/reports/leaderboard.{csv,json}
- 图：ROC / PR 曲线、主模型混淆矩阵、AMS & AUC 排行榜
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import models as M  # noqa: E402
from src import plotting as P  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402

MODELS = ["dummy", "logreg", "random_forest", "xgboost", "mlp"]
CMP_DIR = "comparison"


def main() -> None:
    enable_utf8_stdout()
    P.setup_style()
    banner("03 - 模型比较实验")

    splits = D.get_official_splits()
    train, public = splits["train"], splits["public"]
    print(f"训练集: {train}\n评估集(public): {public}\n")

    rows = []
    roc_curves = {}
    pr_curves = {}
    proba_store = {"y": public.y, "w": public.w}
    results = {}

    for name in MODELS:
        with timer(f"训练 {M.DISPLAY_NAME[name]}"):
            res = M.fit_and_evaluate(name, train, public)
        results[name] = res

        # 持久化管道与概率
        joblib.dump(res.pipeline, C.MODEL_DIR / f"{name}.joblib")
        proba_store[name] = res.proba

        m05, mb = res.metrics_default, res.metrics_best
        rows.append({
            "model": name,
            "模型": M.DISPLAY_NAME[name],
            "Accuracy": m05["accuracy"],
            "Precision": m05["precision"],
            "Recall": m05["recall"],
            "F1": m05["f1"],
            "ROC-AUC": m05["roc_auc"],
            "AMS@0.5": m05["ams"],
            "AMS(best)": res.best_ams,
            "best_thr": res.best_threshold,
        })

        # 曲线（dummy 的 PR/ROC 退化，跳过画线以免干扰）
        if name != "dummy":
            fpr, tpr, _ = roc_curve(public.y, res.proba)
            roc_curves[M.DISPLAY_NAME[name]] = (fpr, tpr, m05["roc_auc"])
            prec, rec, _ = precision_recall_curve(public.y, res.proba)
            ap = average_precision_score(public.y, res.proba)
            pr_curves[M.DISPLAY_NAME[name]] = (rec, prec, ap)

    np.savez_compressed(C.REPORT_DIR / "public_eval.npz", **proba_store)

    # 排行榜
    table = pd.DataFrame(rows)
    table_sorted = table.sort_values("AMS(best)", ascending=False).reset_index(drop=True)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 220)
    print("\n" + "=" * 90)
    print("模型排行榜（评估集 = public，按 AMS(best) 降序）")
    print("=" * 90)
    print(table_sorted.drop(columns=["model"]).to_string(index=False))

    table_sorted.to_csv(C.REPORT_DIR / "leaderboard.csv", index=False, encoding="utf-8-sig")
    (C.REPORT_DIR / "leaderboard.json").write_text(
        table_sorted.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 图：ROC / PR
    P.savefig(P.plot_roc_curves(roc_curves), "01_roc_curves.png", CMP_DIR)
    P.savefig(P.plot_pr_curves(pr_curves), "02_pr_curves.png", CMP_DIR)

    # 图：排行榜
    disp = [r["模型"] for r in rows]
    P.savefig(
        P.plot_leaderboard(disp, [r["AMS(best)"] for r in rows], "AMS(best)"),
        "03_leaderboard_ams.png", CMP_DIR,
    )
    P.savefig(
        P.plot_leaderboard(disp, [r["ROC-AUC"] for r in rows], "ROC-AUC"),
        "04_leaderboard_auc.png", CMP_DIR,
    )

    # 图：主模型（AMS 最优者）的混淆矩阵（用其 AMS 最优阈值）
    best_name = table_sorted.iloc[0]["model"]
    best_res = results[best_name]
    y_pred = (best_res.proba >= best_res.best_threshold).astype(int)
    cm = confusion_matrix(public.y, y_pred)
    P.savefig(
        P.plot_confusion(cm, title=f"{M.DISPLAY_NAME[best_name]} 混淆矩阵 (阈值={best_res.best_threshold:.2f})"),
        "05_confusion_best.png", CMP_DIR,
    )

    banner("模型比较完成", char="-")
    print(f"最优模型: {M.DISPLAY_NAME[best_name]}  |  "
          f"ROC-AUC={best_res.metrics_default['roc_auc']:.4f}  |  "
          f"AMS(best)={best_res.best_ams:.4f}")
    print(f"模型已保存至: {C.MODEL_DIR}")
    print(f"图表已保存至: {C.FIGURE_DIR / CMP_DIR}")


if __name__ == "__main__":
    main()
