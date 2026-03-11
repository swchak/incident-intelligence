from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class EvalConfig:
    label_col: str = "root_cause_label"
    metrics_out: str = "artifacts/metrics/evaluation.json"
    summary_csv_out: Optional[str] = "artifacts/metrics/evaluation_summary.csv"
    plots_dir: str = "artifacts/plots"
    reports_dir: str = "artifacts/reports"


def load_df(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv or .parquet)")


def load_pipeline(path: str | Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(model)}")
    return model


def find_model_files(models_dir: str | Path) -> List[Path]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    return sorted(models_dir.glob("*.joblib"))


def split_xy(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Columns={list(df.columns)}")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def evaluate_one(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, Any]:
    y_pred = model.predict(X)

    report_dict = classification_report(
        y, y_pred, output_dict=True, zero_division=0
    )
    macro_avg = report_dict.get("macro avg", {})

    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision_macro": float(macro_avg.get("precision", 0.0)),
        "recall_macro": float(macro_avg.get("recall", 0.0)),
        "f1_macro": float(macro_avg.get("f1-score", 0.0)),
        "classification_report": report_dict,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "y_pred": y_pred,
    }

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
            else:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y, proba, multi_class="ovr", average="macro")
                )
        except Exception as e:
            out["roc_auc_error"] = str(e)

    return out


def get_final_estimator(model: Pipeline):
    if hasattr(model, "named_steps"):
        for step_name in ["clf", "model", "classifier"]:
            if step_name in model.named_steps:
                return model.named_steps[step_name]
        return list(model.named_steps.values())[-1]
    return model


def _safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str,
    plots_dir: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"confusion_matrix_{_safe_name(model_name)}.png"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(
    summary_rows: List[Dict[str, Any]],
    plots_dir: Path,
) -> None:
    df = pd.DataFrame(summary_rows).copy()

    metric_cols = [
        c
        for c in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc",
        ]
        if c in df.columns
    ]
    if not metric_cols:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "model_comparison.png"

    plot_df = df[["model_name"] + metric_cols].set_index("model_name")

    ax = plot_df.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model: Pipeline,
    feature_names: List[str],
    model_name: str,
    plots_dir: Path,
    top_n: int = 20,
) -> None:
    estimator = get_final_estimator(model)

    values = None
    value_col = None

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        value_col = "importance"
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        values = np.abs(coef).mean(axis=0) if np.ndim(coef) > 1 else np.abs(coef)
        value_col = "absolute_coefficient"
    else:
        print(f"Skipping feature importance for {model_name}: unsupported estimator.")
        return

    if len(values) != len(feature_names):
        print(
            f"Skipping feature importance for {model_name}: "
            f"{len(values)=} does not match {len(feature_names)=}"
        )
        return

    df = pd.DataFrame(
        {
            "feature": feature_names,
            value_col: values,
        }
    ).sort_values(value_col, ascending=False).head(top_n)

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"feature_importance_{_safe_name(model_name)}.png"

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x=value_col, y="feature")
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_classification_report(
    y_true,
    y_pred,
    model_name: str,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_path = reports_dir / f"{_safe_name(model_name)}_classification_report.md"

    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    ).T

    title = f"# Classification Report - {model_name}\n\n"
    table = report_df.to_markdown()
    out_path.write_text(title + table + "\n", encoding="utf-8")


def evaluate_models(
    model_paths: List[Path],
    df_eval: pd.DataFrame,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    X, y = split_xy(df_eval, cfg.label_col)

    plots_dir = Path(cfg.plots_dir)
    reports_dir = Path(cfg.reports_dir)

    results: Dict[str, Any] = {"label_col": cfg.label_col, "models": []}
    summary_rows: List[Dict[str, Any]] = []

    for mp in model_paths:
        model = load_pipeline(mp)
        metrics = evaluate_one(model, X, y)

        y_pred = metrics.pop("y_pred")

        model_result = {
            "model_path": str(mp),
            "model_name": mp.stem,
            "metrics": metrics,
        }
        results["models"].append(model_result)

        row = {
            "model_name": mp.stem,
            "model_path": str(mp),
            "accuracy": metrics.get("accuracy"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "f1_macro": metrics.get("f1_macro"),
            "roc_auc": metrics.get("roc_auc", metrics.get("roc_auc_ovr_macro")),
        }
        summary_rows.append(row)

        print(f"\n=== {mp.stem} ===")
        print(classification_report(y, y_pred, zero_division=0))

        plot_confusion_matrix(y, y_pred, mp.stem, plots_dir)
        plot_feature_importance(model, X.columns.tolist(), mp.stem, plots_dir)
        save_classification_report(y, y_pred, mp.stem, reports_dir)

    metrics_path = Path(cfg.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")

    if cfg.summary_csv_out:
        summary_path = Path(cfg.summary_csv_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame(summary_rows).sort_values("f1_macro", ascending=False)
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        plot_model_comparison(summary_rows, plots_dir)

    return results


def run_evaluation(
    *,
    data_path: str | Path,
    cfg: EvalConfig,
    model_path: str | Path | None = None,
    models_dir: str | Path = "artifacts/models",
) -> Dict[str, Any]:
    df_eval = load_df(data_path)

    if model_path:
        model_paths = [Path(model_path)]
    else:
        model_paths = find_model_files(models_dir)

    return evaluate_models(model_paths, df_eval, cfg)