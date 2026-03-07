from __future__ import annotations

import argparse

from incident_intelligence.modeling.train import (
    TrainValidateConfig,
    run_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline models on train set and evaluate on validation set."
    )
    parser.add_argument(
        "--train",
        type=str,
        default="data/processed/incident_root_cause_train.csv",
        help="Path to train CSV/Parquet",
    )
    parser.add_argument(
        "--val",
        type=str,
        default="data/processed/incident_root_cause_val.csv",
        help="Path to validation CSV/Parquet",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="root_cause_label",
        help="Target label column name",
    )
    parser.add_argument(
        "--models-out-dir",
        type=str,
        default="artifacts/models",
        help="Directory to save trained model pipelines",
    )
    parser.add_argument(
        "--metrics-out-json",
        type=str,
        default="artifacts/metrics/train_val_results.json",
        help="Path to save detailed training/validation metrics JSON",
    )
    parser.add_argument(
        "--leaderboard-out-csv",
        type=str,
        default="artifacts/metrics/leaderboard_val.csv",
        help="Path to save validation leaderboard CSV",
    )
    parser.add_argument(
        "--best-model-out",
        type=str,
        default="artifacts/models/best_model.joblib",
        help="Path to save the selected best model",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = TrainValidateConfig(
        label_col=args.label_col,
        models_out_dir=args.models_out_dir,
        metrics_out_json=args.metrics_out_json,
        leaderboard_out_csv=args.leaderboard_out_csv,
        best_model_out=args.best_model_out,
    )

    result = run_training(
        train_path=args.train,
        val_path=args.val,
        cfg=cfg,
    )

    best = result["best_model"]
    print(
        f"Training complete. Best model: {best['model_name']} "
        f"(val_f1_macro={best['val_f1_macro']:.4f}, "
        f"val_accuracy={best['val_accuracy']:.4f})"
    )


if __name__ == "__main__":
    main()