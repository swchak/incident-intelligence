"""
Evaluate trained classification pipelines and export metrics, plots, and reports.

This module is intended to run after model training has completed and one or
more serialized sklearn `Pipeline` artifacts have been written to disk. It
provides utilities to:

- load evaluation data from CSV or Parquet
- load one or more trained pipeline artifacts
- compute standard classification metrics
- compute ROC-AUC when probability estimates are available
- generate confusion matrix, feature-importance, and model-comparison plots
- write markdown classification reports and JSON/CSV summaries

Typical usage:
    cfg = EvalConfig()
    results = run_evaluation(
        data_path="data/test.csv",
        cfg=cfg,
        models_dir="artifacts/models",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from incident_intelligence.modeling.predict import predict_outputs


@dataclass(frozen=True)
class EvalConfig:
    """
    Configuration for offline model evaluation.

    Attributes:
        label_col: Target column expected in the evaluation dataframe.
        metrics_out: Destination JSON file containing detailed per-model metrics.
        summary_csv_out: Optional CSV leaderboard containing a compact summary of
            the key metrics for each evaluated model. Set to None to disable CSV
            export.
        plots_dir: Directory where generated PNG plots are written.
        reports_dir: Directory where markdown classification reports are written.
    """

    label_col: str = "root_cause_label"
    metrics_out: str = "artifacts/metrics/evaluation.json"
    summary_csv_out: Optional[str] = "artifacts/metrics/evaluation_summary.csv"
    plots_dir: str = "artifacts/plots"
    reports_dir: str = "artifacts/reports"



def load_df(path: str | Path) -> pd.DataFrame:
    """
    Load a tabular dataset from CSV or Parquet.

    Args:
        path: Path to the input dataset.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        ValueError: If the file type is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv or .parquet)")



def load_pipeline(path: str | Path) -> Pipeline:
    """
    Load a serialized sklearn Pipeline artifact.

    Args:
        path: Path to a `.joblib` model artifact.

    Returns:
        Deserialized sklearn Pipeline.

    Raises:
        FileNotFoundError: If the model file is missing.
        TypeError: If the loaded object is not an sklearn Pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(model)}")
    return model



def find_model_files(models_dir: str | Path) -> List[Path]:
    """
    Find all serialized model artifacts inside a directory.

    Args:
        models_dir: Directory containing `.joblib` model files.

    Returns:
        Sorted list of model paths.

    Raises:
        FileNotFoundError: If the models directory does not exist.
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    return sorted(models_dir.glob("*.joblib"))



def split_xy(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: Optional[List[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into features and target.

    Args:
        df: Input dataframe containing both features and target.
        label_col: Name of the target column.

    Returns:
        Tuple of `(X, y)`.

    Raises:
        ValueError: If the target column is not present.
    """
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Columns={list(df.columns)}")
    drop_cols = drop_cols or []
    cols_to_drop = [label_col] + [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[label_col]
    return X, y


def load_eval_data(dataset_kind: str = "snapshot") -> pd.DataFrame:
    """
    Load the standard processed evaluation dataset for a given dataset kind.
    """
    if dataset_kind == "snapshot":
        eval_path = "data/processed/incident_snapshot_eval.csv"
    elif dataset_kind == "temporal":
        eval_path = "data/processed/incident_temporal_eval.csv"
    else:
        raise ValueError(
            f"Unsupported dataset_kind='{dataset_kind}'. "
            "Expected one of: ['snapshot', 'temporal']"
        )

    return load_df(eval_path)



def _json_safe(obj: Any) -> Any:
    """
    Recursively convert NumPy-heavy objects into JSON-serializable objects.

    This utility is used before writing metrics to disk because sklearn metric
    outputs often include NumPy arrays and scalar types that `json.dumps`
    cannot serialize directly.
    """
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
    pred_out: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Evaluate a single fitted classification pipeline on a labeled dataset.

    Args:
        model: Fitted sklearn Pipeline.
        X: Evaluation features.
        y: Evaluation labels.
        pred_out: Optional precomputed outputs from predict_outputs().

    Returns:
        Dictionary of metrics and derived outputs.
    """
    pred_out = pred_out or predict_outputs(model, X)

    y_pred = pred_out["y_pred"]
    proba = pred_out["y_proba"]

    report_dict = classification_report(
        y,
        y_pred,
        output_dict=True,
        zero_division=0,
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

    if proba is not None:
        try:
            if proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
            else:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y, proba, multi_class="ovr", average="macro")
                )
        except Exception as e:
            out["roc_auc_error"] = str(e)

    return out



def get_final_estimator(model: Pipeline) -> Any:
    """
    Return the final learned estimator from a pipeline-like object.

    The helper prefers common terminal step names used in this project such as
    `clf`, `model`, and `classifier`, but will fall back to the final named step
    when necessary.

    Args:
        model: sklearn Pipeline or estimator-like object.

    Returns:
        Final estimator object.
    """
    if hasattr(model, "named_steps"):
        for step_name in ["clf", "model", "classifier"]:
            if step_name in model.named_steps:
                return model.named_steps[step_name]
        return list(model.named_steps.values())[-1]
    return model



def _safe_name(name: str) -> str:
    """
    Convert a display name into a filesystem-safe stem.

    Example:
        "SVM (RBF)" -> "SVM_RBF"
    """
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )



def plot_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    plots_dir: Path,
) -> None:
    """
    Save a confusion matrix heatmap for one model.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Name used in the chart title and filename.
        plots_dir: Directory where the plot should be written.
    """
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
    """
    Save a grouped bar chart comparing core metrics across models.

    Only metrics present in the provided summary rows are plotted.

    Args:
        summary_rows: One compact metrics row per model.
        plots_dir: Directory where the comparison plot should be written.
    """
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
    """
    Save a feature-importance plot when the estimator exposes importance data.

    Supported cases:
    - tree-based estimators exposing `feature_importances_`
    - linear estimators exposing `coef_`

    For multiclass linear models, the mean absolute coefficient magnitude is
    used as a simple global importance proxy.

    Args:
        model: Fitted sklearn Pipeline.
        feature_names: Names of the input features used during evaluation.
        model_name: Name used in the plot title and filename.
        plots_dir: Directory where the plot should be written.
        top_n: Number of top-ranked features to plot.
    """
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
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    reports_dir: Path,
) -> None:
    """
    Save a markdown classification report for one model.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        model_name: Name used in the report title and filename.
        reports_dir: Directory where the report should be written.
    """
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
    """
    Evaluate one or more serialized pipelines on a labeled evaluation dataset.

    Side effects:
    - writes JSON metrics summary
    - optionally writes CSV leaderboard
    - writes plots and markdown reports for each model

    Args:
        model_paths: Paths to `.joblib` pipeline artifacts.
        df_eval: Evaluation dataframe containing features and labels.
        cfg: Evaluation configuration.

    Returns:
        Nested results dictionary that is also persisted to disk.
    """
    X, y = split_xy(df_eval, cfg.label_col, drop_cols=["incident_id"])

    plots_dir = Path(cfg.plots_dir)
    reports_dir = Path(cfg.reports_dir)

    results: Dict[str, Any] = {"label_col": cfg.label_col, "models": []}
    summary_rows: List[Dict[str, Any]] = []

    for mp in model_paths:
        model = load_pipeline(mp)
        metrics = evaluate_one(model, X, y)

        # `y_pred` is used for report generation but not persisted in the final
        # JSON payload to keep the artifact smaller and easier to inspect.
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
    """
    Entry point for evaluating either a single model or all models in a folder.

    Args:
        data_path: Path to the evaluation dataset.
        cfg: Evaluation configuration.
        model_path: Optional path to one specific model artifact to evaluate.
            When omitted, all `.joblib` files in `models_dir` are evaluated.
        models_dir: Directory searched when `model_path` is not supplied.

    Returns:
        The same nested results payload returned by `evaluate_models`.
    """
    df_eval = load_df(data_path)

    if model_path:
        model_paths = [Path(model_path)]
    else:
        model_paths = find_model_files(models_dir)

    return evaluate_models(model_paths, df_eval, cfg)


def run_evaluation_for_dataset_kind(
    *,
    dataset_kind: str,
    cfg: EvalConfig,
    model_path: str | Path | None = None,
    models_dir: str | Path = "artifacts/models",
) -> Dict[str, Any]:
    """
    Evaluate one or more models against the standard processed dataset for the
    requested dataset family.
    """
    df_eval = load_eval_data(dataset_kind=dataset_kind)

    if model_path:
        model_paths = [Path(model_path)]
    else:
        model_paths = find_model_files(models_dir)

    return evaluate_models(model_paths, df_eval, cfg)
