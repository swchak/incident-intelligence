from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
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
    # Makes numpy arrays/scalars JSON serializable
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

    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "classification_report": classification_report(y, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    # Optional ROC-AUC (only if predict_proba exists and labels support it)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # Binary
            if proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
            # Multiclass
            else:
                out["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y, proba, multi_class="ovr", average="macro")
                )
        except Exception as e:
            out["roc_auc_error"] = str(e)

    return out


def evaluate_models(
    model_paths: List[Path],
    df_eval: pd.DataFrame,
    cfg: EvalConfig,
) -> Dict[str, Any]:
    X, y = split_xy(df_eval, cfg.label_col)

    results: Dict[str, Any] = {"label_col": cfg.label_col, "models": []}
    summary_rows: List[Dict[str, Any]] = []

    for mp in model_paths:
        model = load_pipeline(mp)
        metrics = evaluate_one(model, X, y)

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
            "roc_auc": metrics.get("roc_auc", metrics.get("roc_auc_ovr_macro")),
        }
        summary_rows.append(row)

    # Write outputs
    metrics_path = Path(cfg.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_safe(results), indent=2))

    if cfg.summary_csv_out:
        summary_path = Path(cfg.summary_csv_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False).to_csv(summary_path, index=False)

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