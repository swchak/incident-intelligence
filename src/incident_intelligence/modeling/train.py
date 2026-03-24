"""
This module contains the main training loop for baseline model selection using train/validation splits.

The workflow is as follows:
1. Load train and validation datasets from disk
2. For each baseline model:
   a. Tune hyperparameters with GridSearchCV on the train split only
   b. Evaluate the tuned best estimator on the validation split
   c. Save the tuned pipeline and record validation metrics
3. Select the overall best model by validation macro F1 score
4. Write out a detailed JSON report and a leaderboard CSV summarizing all models' validation performance 
5. Save the selected best model pipeline to disk

The main entry point is the `main()` function, which can be invoked from the CLI. The core logic is in `train_and_validate()`, 
which is also callable directly for programmatic use.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from incident_intelligence.modeling.baseline import (
    BaselineTrainConfig,
    get_models_to_run,
    make_pipeline,
)
from incident_intelligence.modeling.evaluate import evaluate_one
from incident_intelligence.modeling.predict import predict_outputs


@dataclass(frozen=True)
class TrainValidateConfig:
    """
    Configuration for training and validating baseline models.

    Attributes:
    - label_col: Name of the target label column in the datasets.
    - models_out_dir: Directory to save trained model pipelines.
    - metrics_out_json: Path to save detailed training/validation metrics JSON.
    - leaderboard_out_csv: Path to save summary CSV leaderboard of models.
    - best_model_out: Path to save the selected best model pipeline.
    """
    label_col: str = "root_cause_label"
    models_out_dir: str = "artifacts/models"
    metrics_out_json: str = "artifacts/metrics/train_val_results.json"
    leaderboard_out_csv: str = "artifacts/metrics/leaderboard_val.csv"
    best_model_out: str = "artifacts/models/best_model.joblib"
    cv: int = 5
    n_jobs: int = -1
    verbose: int = 1
    scoring: str = "f1_macro"
    models: tuple[str, ...] | None = None
    fast_mode: bool = False

def with_dataset_suffix(path_str: str, dataset_kind: str) -> str:
    """
    Append dataset kind (e.g., 'snapshot' or 'temporal') to a file path.

    Example:
        artifacts/models/best_model.joblib
        -> artifacts/models/best_model_temporal.joblib
    """
    path = Path(path_str)
    return str(path.with_name(f"{path.stem}_{dataset_kind}{path.suffix}"))


def with_parent_dir_suffix(path_str: str, dataset_kind: str) -> str:
    """
    Append dataset kind to the parent directory of a path while preserving the
    original filename.

    Example:
        artifacts/metrics/train_val_results.json
        -> artifacts/metrics_temporal/train_val_results.json
    """
    path = Path(path_str)
    parent = path.parent
    suffixed_parent = parent.with_name(f"{parent.name}_{dataset_kind}")
    return str(suffixed_parent / path.name)

def _safe_model_name(name: str) -> str:
    """
    Convert a model name into a filesystem-safe filename stem by replacing or removing special characters.
    This is used to create output filenames for each model, where the filename is derived from the model name but 
    sanitized to avoid issues with special characters. 
    
    For example, "Random Forest (v1)" would become "Random_Forest_v1".
    """
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_df(path: str | Path) -> pd.DataFrame:
    """
    Load a dataset from a CSV or Parquet file into a pandas DataFrame.
    The file type is inferred from the extension. Supported formats are .csv and .parquet/.pq.

    Args:
        path: Path to the dataset file.

    Returns:
        Loaded dataframe.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv or .parquet)")

def load_training_data(
    dataset_kind: str = "snapshot",
    *,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and validation datasets based on dataset kind.

    Args:
        dataset_kind: Which processed dataset family to load.
            Supported:
            - "snapshot"
            - "temporal"

    Returns:
        (train_df, val_df)
    """
    if train_path is None or val_path is None:
        if dataset_kind == "snapshot":
            default_train_path = "data/processed/incident_snapshot_train.csv"
            default_val_path = "data/processed/incident_snapshot_val.csv"
        elif dataset_kind == "temporal":
            default_train_path = "data/processed/incident_temporal_train.csv"
            default_val_path = "data/processed/incident_temporal_val.csv"
        else:
            raise ValueError(
                f"Unsupported dataset_kind='{dataset_kind}'. "
                "Expected one of: ['snapshot', 'temporal']"
            )
        train_path = train_path or default_train_path
        val_path = val_path or default_val_path

    train_df = load_df(train_path)
    val_df = load_df(val_path)
    return train_df, val_df


def load_eval_data(dataset_kind: str = "snapshot") -> pd.DataFrame:
    """
    Load evaluation dataset based on dataset kind.

    Args:
        dataset_kind: Which processed dataset family to load.

    Returns:
        Evaluation dataframe.
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


def split_xy(
    df: pd.DataFrame,
    label_col: str,
    drop_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into features (X) and target labels (y).
    """
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Columns={list(df.columns)}")

    drop_cols = drop_cols or []
    cols_to_drop = [label_col] + [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df[label_col]
    return X, y

def _json_safe(obj: Any) -> Any:
    """
    Recursively convert an object into a JSON-serializable format by converting numpy types to native Python types.
    This is used to prepare the training/validation results payload for JSON serialization, ensuring that all
    values are compatible with JSON encoding. For example, numpy arrays are converted to lists, and numpy
    numeric types are converted to native Python int or float.
    
    Args:
        obj: The object to convert.
    Returns:
        A JSON-serializable version of the input object.
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


def fit_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    estimator: Any,
    param_grid: Dict[str, Any],
    base_cfg: BaselineTrainConfig,
) -> GridSearchCV:
    """
    Build a pipeline, run GridSearchCV, and return the fitted search object.

    Hyperparameters are selected using base_cfg.scoring when provided.
    Otherwise, GridSearchCV falls back to the estimator's default score method,
    which is typically accuracy for classifiers.

    Args:
        X_train: Training feature dataframe.
        y_train: Training target series.
        estimator: Estimator instance to wrap in a pipeline.
        param_grid: GridSearchCV parameter grid using pipeline step names.
        base_cfg: Shared baseline training configuration.

    Returns:
        Fitted GridSearchCV instance.
    """
    pipe = make_pipeline(estimator)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=base_cfg.cv,
        n_jobs=base_cfg.n_jobs,
        verbose=base_cfg.verbose,
        scoring=base_cfg.scoring,
        refit=True,
    )
    grid.fit(X_train, y_train)
    return grid


def save_pipeline(pipeline: Any, out_path: Path) -> Path:
    """
    Save a fitted pipeline artifact to disk.

    Args:
        pipeline: Fitted pipeline or estimator to serialize.
        out_path: Destination path.

    Returns:
        The output path used for saving.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    return out_path


def train_and_validate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    cfg: TrainValidateConfig,
    base_cfg: Optional[BaselineTrainConfig] = None,
) -> Dict[str, Any]:
    """
    Train all configured baseline models on the train split and evaluate on the validation split.

    Workflow:
    1. Split train and validation data into features and labels
    2. Tune each model with GridSearchCV on the train split only
    3. Evaluate the tuned best estimator on the validation split
    4. Save each tuned pipeline
    5. Select the overall best model by validation macro F1
    6. Write JSON and leaderboard artifacts

    Important:
        Hyperparameter selection inside GridSearchCV uses base_cfg.scoring
        (or estimator default scoring if None), while the final model selection
        across trained models uses validation `val_f1_macro`.

    Args:
        train_df: Training dataframe.
        val_df: Validation dataframe.
        cfg: Train/validation artifact configuration.
        base_cfg: Optional baseline training configuration controlling CV behavior.

    Returns:
        A payload containing the selected best model and all per-model validation results.
    """
    NON_FEATURE_COLUMNS = ["incident_id"]
    base_cfg = base_cfg or BaselineTrainConfig(
        label_col=cfg.label_col,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
        scoring=cfg.scoring,
        selected_models=cfg.models,
        fast_mode=cfg.fast_mode,
    )

    X_train, y_train = split_xy(train_df, cfg.label_col, drop_cols=NON_FEATURE_COLUMNS)
    X_val, y_val = split_xy(val_df, cfg.label_col, drop_cols=NON_FEATURE_COLUMNS)

    models_out_dir = Path(cfg.models_out_dir)
    models_out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for model_info in get_models_to_run(
        base_cfg.random_state,
        selected_models=base_cfg.selected_models,
        fast_mode=base_cfg.fast_mode,
    ):
        name = model_info["name"]
        est = model_info["estimator"]
        param_grid = model_info["param_grid"]
        started_at = time.perf_counter()

        grid = fit_grid(
            X_train,
            y_train,
            estimator=est,
            param_grid=param_grid,
            base_cfg=base_cfg,
        )

        best_pipe = grid.best_estimator_

        pred_out = predict_outputs(best_pipe, X_val)

        metrics = evaluate_one(
            best_pipe,
            X_val,
            y_val,
            pred_out=pred_out,
        )

        metrics["val_accuracy"] = metrics["accuracy"]
        metrics["val_f1_macro"] = metrics["f1_macro"]
        metrics["fit_seconds"] = float(time.perf_counter() - started_at)

        model_file = models_out_dir / f"{_safe_model_name(name)}_pipeline.joblib"
        save_pipeline(best_pipe, model_file)

        results.append(
            {
                "model_name": name,
                "model_path": str(model_file),
                "best_params": grid.best_params_,
                "best_cv_score": float(grid.best_score_),
                "cv_scoring": base_cfg.scoring or "estimator_default_score",
                "fit_seconds": metrics["fit_seconds"],
                "val_metrics": metrics,
            }
        )

        print(f"[OK] {name}: val_accuracy={metrics['val_accuracy']:.4f}  saved={model_file}")

    best = max(results, key=lambda r: r["val_metrics"]["val_f1_macro"])
    best_out_path = Path(cfg.best_model_out)
    best_out_path.parent.mkdir(parents=True, exist_ok=True)

    best_pipeline = joblib.load(best["model_path"])
    joblib.dump(best_pipeline, best_out_path)

    payload = {
        "label_col": cfg.label_col,
        "selection_metric": "val_f1_macro",
        "cv_scoring": base_cfg.scoring or "estimator_default_score",
        "best_model": {
            "model_name": best["model_name"],
            "model_path": str(best_out_path),
            "val_f1_macro": best["val_metrics"]["val_f1_macro"],
            "val_accuracy": best["val_metrics"]["val_accuracy"],
        },
        "all_models": results,
    }

    metrics_path = Path(cfg.metrics_out_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")

    leaderboard_rows = [
        {
            "model_name": r["model_name"],
            "model_path": r["model_path"],
            "val_accuracy": r["val_metrics"].get("val_accuracy"),
            "val_f1_macro": r["val_metrics"].get("val_f1_macro"),
            "fit_seconds": r.get("fit_seconds"),
        }
        for r in results
    ]
    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        "val_f1_macro",
        ascending=False,
    )

    leaderboard_path = Path(cfg.leaderboard_out_csv)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(leaderboard_path, index=False)

    print(
        f"\nBest model: {payload['best_model']['model_name']}  "
        f"(val_f1_macro={payload['best_model']['val_f1_macro']:.4f})"
    )
    print(f"Wrote metrics JSON: {metrics_path}")
    print(f"Wrote leaderboard:  {leaderboard_path}")
    print(f"Wrote best model:   {best_out_path}")

    return payload


def run_training(
    train_path: str | Path,
    val_path: str | Path,
    *,
    cfg: TrainValidateConfig,
    base_cfg: Optional[BaselineTrainConfig] = None,
) -> Dict[str, Any]:
    """
    Load train and validation datasets from disk, then run train/validation model selection.

    Args:
        train_path: Path to the training dataset.
        val_path: Path to the validation dataset.
        cfg: Train/validation artifact configuration.
        base_cfg: Optional baseline training configuration controlling CV behavior.

    Returns:
        The same payload returned by train_and_validate().
    """
    train_df = load_df(train_path)
    val_df = load_df(val_path)
    return train_and_validate(train_df, val_df, cfg=cfg, base_cfg=base_cfg)

def run_training_for_dataset_kind(
    dataset_kind: str,
    *,
    cfg: TrainValidateConfig,
    base_cfg: Optional[BaselineTrainConfig] = None,
    train_path: str | Path | None = None,
    val_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Load the standard processed train/validation datasets for a given dataset kind
    and run training.
    """
    train_df, val_df = load_training_data(
        dataset_kind=dataset_kind,
        train_path=train_path,
        val_path=val_path,
    )
    return train_and_validate(train_df, val_df, cfg=cfg, base_cfg=base_cfg)
