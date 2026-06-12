"""
02 - 基准模型

建立两条性能底线：
1. DummyClassifier(prior)  —— 最朴素的「多数类」基线，AUC≈0.5，作为绝对下限
2. Logistic Regression     —— 线性基线，衡量「线性可分性」上限

评估集：官方 public 集（KaggleSet='b'，10 万条），训练集：official train（25 万条）。
私有集 private 留作最终评估，避免信息泄漏。

指标：Accuracy / Precision / Recall / F1 / ROC-AUC / AMS（默认阈值 0.5 与 AMS 最优阈值）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))

import config as C  # noqa: E402
from src import data as D  # noqa: E402
from src import models as M  # noqa: E402
from src.utils import banner, enable_utf8_stdout, timer  # noqa: E402


def main() -> None:
    enable_utf8_stdout()
    banner("02 - 基准模型 (Dummy + Logistic Regression)")

    splits = D.get_official_splits()
    train, public = splits["train"], splits["public"]
    print(f"训练集: {train}\n评估集(public): {public}\n")

    rows = []
    results = {}
    for name in ["dummy", "logreg"]:
        with timer(f"训练 {M.DISPLAY_NAME[name]}"):
            res = M.fit_and_evaluate(name, train, public)
        results[name] = res
        m05, mb = res.metrics_default, res.metrics_best
        rows.append(
            {
                "模型": M.DISPLAY_NAME[name],
                "Accuracy": m05["accuracy"],
                "Precision": m05["precision"],
                "Recall": m05["recall"],
                "F1": m05["f1"],
                "ROC-AUC": m05["roc_auc"],
                "AMS@0.5": m05["ams"],
                "AMS(best)": res.best_ams,
                "best_thr": res.best_threshold,
            }
        )

    table = pd.DataFrame(rows).set_index("模型")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 200)
    print("\n" + "=" * 80)
    print("基准模型结果（评估集 = public）")
    print("=" * 80)
    print(table.to_string())

    out = C.REPORT_DIR / "baseline_metrics.json"
    out.write_text(
        json.dumps(
            {k: {**v.metrics_default, "best_threshold": v.best_threshold, "best_ams": v.best_ams}
             for k, v in results.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    table.to_csv(C.REPORT_DIR / "baseline_metrics.csv", encoding="utf-8-sig")

    banner("基准建立完成", char="-")
    print(f"逻辑回归 ROC-AUC = {results['logreg'].metrics_default['roc_auc']:.4f}，"
          f"AMS(best) = {results['logreg'].best_ams:.4f}")
    print("后续非线性模型需要显著超过这条线性基线才算有效。")


if __name__ == "__main__":
    main()
