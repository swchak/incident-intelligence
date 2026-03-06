from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

from incident_intelligence.modeling.baseline import (
    BaselineTrainConfig,
    get_models_to_run,
    make_pipeline,
)
from incident_intelligence.modeling.evaluate import evaluate_one  # reuse your evaluator


@dataclass(frozen=True)
class TrainValidateConfig:
    label_col: str = "root_cause_label"
    models_out_dir: str = "artifacts/models"
    metrics_out_json: str = "artifacts/metrics/train_val_results.json"
    leaderboard_out_csv: str = "artifacts/metrics/leaderboard_val.csv"
    best_model_out: str = "artifacts/models/best_model.joblib"


def load_df(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv or .parquet)")


def split_xy(df: pd.DataFrame, label_col: str):
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


def fit_grid(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_name: str,
    estimator,
    param_grid: Dict[str, Any],
    base_cfg: BaselineTrainConfig,
) -> GridSearchCV:
    pipe = make_pipeline(estimator)
    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=base_cfg.cv,
        n_jobs=base_cfg.n_jobs,
        verbose=base_cfg.verbose,
    )
    grid.fit(X_train, y_train)
    return grid


def save_pipeline(pipeline, out_path: Path) -> Path:
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
    base_cfg = base_cfg or BaselineTrainConfig(label_col=cfg.label_col)

    X_train, y_train = split_xy(train_df, cfg.label_col)
    X_val, y_val = split_xy(val_df, cfg.label_col)

    models_out_dir = Path(cfg.models_out_dir)
    models_out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for model_info in get_models_to_run(base_cfg.random_state):
        name = model_info["name"]
        est = model_info["estimator"]
        param_grid = model_info["param_grid"]

        grid = fit_grid(
            X_train,
            y_train,
            model_name=name,
            estimator=est,
            param_grid=param_grid,
            base_cfg=base_cfg,
        )

        best_pipe = grid.best_estimator_

        # Evaluate on validation set
        metrics = evaluate_one(best_pipe, X_val, y_val)

        # Add a couple leaderboard-friendly numbers
        y_pred = best_pipe.predict(X_val)
        metrics["val_accuracy"] = float(accuracy_score(y_val, y_pred))
        metrics["val_f1_macro"] = float(f1_score(y_val, y_pred, average="macro"))

        # Save pipeline
        model_file = models_out_dir / f"{name.replace(' ', '_')}_pipeline.joblib"
        save_pipeline(best_pipe, model_file)

        results.append(
            {
                "model_name": name,
                "model_path": str(model_file),
                "best_params": grid.best_params_,
                "val_metrics": metrics,
            }
        )

        print(f"[OK] {name}: val_accuracy={metrics['val_accuracy']:.4f}  saved={model_file}")

    # pick best by macro F1 (usually better than accuracy with imbalance)
    best = max(results, key=lambda r: r["val_metrics"]["val_f1_macro"])
    best_model_path = Path(best["model_path"])
    best_out_path = Path(cfg.best_model_out)
    best_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(joblib.load(best_model_path), best_out_path)

    payload = {
        "label_col": cfg.label_col,
        "selection_metric": "val_f1_macro",
        "best_model": {
            "model_name": best["model_name"],
            "model_path": str(best_out_path),
            "val_f1_macro": best["val_metrics"]["val_f1_macro"],
            "val_accuracy": best["val_metrics"]["val_accuracy"],
        },
        "all_models": results,
    }

    # Write JSON + leaderboard CSV
    metrics_path = Path(cfg.metrics_out_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_safe(payload), indent=2))

    leaderboard_rows = [
        {
            "model_name": r["model_name"],
            "model_path": r["model_path"],
            "val_accuracy": r["val_metrics"].get("val_accuracy"),
            "val_f1_macro": r["val_metrics"].get("val_f1_macro"),
        }
        for r in results
    ]
    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("val_f1_macro", ascending=False)

    leaderboard_path = Path(cfg.leaderboard_out_csv)
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_df.to_csv(leaderboard_path, index=False)

    print(f"\nBest model: {payload['best_model']['model_name']}  "
          f"(val_f1_macro={payload['best_model']['val_f1_macro']:.4f})")
    print(f"Wrote metrics JSON: {metrics_path}")
    print(f"Wrote leaderboard:  {leaderboard_path}")
    print(f"Wrote best model:   {best_out_path}")

    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Train baseline models on train set and evaluate on validation set.")
    p.add_argument("--train", type=str, required=True, help="Train CSV/Parquet")
    p.add_argument("--val", type=str, required=True, help="Validation CSV/Parquet")
    p.add_argument("--label-col", type=str, default="root_cause_label")

    p.add_argument("--models-out-dir", type=str, default="artifacts/models")
    p.add_argument("--metrics-out-json", type=str, default="artifacts/metrics/train_val_results.json")
    p.add_argument("--leaderboard-out-csv", type=str, default="artifacts/metrics/leaderboard_val.csv")
    p.add_argument("--best-model-out", type=str, default="artifacts/models/best_model.joblib")
    args = p.parse_args()

    cfg = TrainValidateConfig(
        label_col=args.label_col,
        models_out_dir=args.models_out_dir,
        metrics_out_json=args.metrics_out_json,
        leaderboard_out_csv=args.leaderboard_out_csv,
        best_model_out=args.best_model_out,
    )

    train_df = load_df(args.train)
    val_df = load_df(args.val)

    train_and_validate(train_df, val_df, cfg=cfg)


if __name__ == "__main__":
    main()