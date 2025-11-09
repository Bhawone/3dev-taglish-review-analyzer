"""
Utility to consolidate evaluation metrics and confusion matrices for
baseline and transformer runs into the `exports` directory.

The script looks for the following files in the project root:
- baseline_predictions_test.csv
- transformer_predictions_test.csv (optional)
- runs_log.csv (optional experiment tracker)

Usage:
    python tools/summarize_metrics.py

Outputs:
- exports/metrics_summary.json  (with accuracy/F1 + confusion matrix)
- exports/metrics_summary.md    (human-readable report)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


EXPORT_DIR = Path("exports")


def ensure_export_dir() -> Path:
    EXPORT_DIR.mkdir(exist_ok=True)
    return EXPORT_DIR


def load_predictions(csv_path: Path) -> Tuple[pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    if {"gold", "pred"} - set(df.columns):
        raise ValueError(f"{csv_path} must contain 'gold' and 'pred' columns.")
    return df["gold"], df["pred"]


def summarize_run(name: str, csv_path: Path) -> Dict:
    y_true, y_pred = load_predictions(csv_path)
    metrics = {
        "model": name,
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    return metrics


def load_runs_log() -> List[Dict]:
    csv_path = Path("runs_log.csv")
    if not csv_path.exists():
        return []

    try:
        runs_df = pd.read_csv(csv_path)
    except Exception as exc:
        runs_df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        print(f"Warning: runs_log.csv parsed with python engine due to {exc!r}")

    return runs_df.fillna("").to_dict(orient="records")


def main() -> None:
    ensure_export_dir()
    summaries: List[Dict] = []

    baseline_csv = Path("baseline_predictions_test.csv")
    if baseline_csv.exists():
        summaries.append(summarize_run("baseline", baseline_csv))
    else:
        print("Warning: baseline_predictions_test.csv not found.")

    transformer_csv = Path("transformer_predictions_test.csv")
    if transformer_csv.exists():
        summaries.append(summarize_run("transformer_best", transformer_csv))
    else:
        print("Warning: transformer_predictions_test.csv not found.")

    runs_log = load_runs_log()

    output_json = {
        "summaries": summaries,
        "runs_log_entries": runs_log,
    }

    json_path = EXPORT_DIR / "metrics_summary.json"
    json_path.write_text(json.dumps(output_json, indent=2))

    lines = ["# Metrics Summary", ""]
    if not summaries:
        lines.append("No prediction CSV files were found.")
    else:
        for summary in summaries:
            lines.append(f"## {summary['model']}")
            lines.append(f"- samples: {summary['num_samples']}")
            lines.append(f"- accuracy: {summary['accuracy']:.4f}")
            lines.append(f"- macro-F1: {summary['f1_macro']:.4f}")
            lines.append("")
            lines.append("Confusion matrix (rows=true, cols=pred):")
            matrix = np.array(summary["confusion_matrix"])
            lines.append("```\n" + str(matrix) + "\n```")
            lines.append("")

    if runs_log:
        lines.append("## Logged Runs")
        for row in runs_log:
            member = row.get("member", "unknown")
            model = row.get("model", "")
            f1_macro = row.get("f1_macro", "")
            notes = row.get("notes", "")
            lines.append(f"- {member}/{model}: macro-F1={f1_macro} ({notes})")

    md_path = EXPORT_DIR / "metrics_summary.md"
    md_path.write_text("\n".join(lines))

    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()


