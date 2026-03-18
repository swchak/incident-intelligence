from __future__ import annotations

import argparse

from incident_intelligence.modeling.train import (
    TrainValidateConfig,
    run_training,
)
from incident_intelligence.config import (
    TrainCLIConfig,
    load_config,
    merge_cli_args,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline models on train set and evaluate on validation set."
    )
    parser.add_argument(
        "--train",
        type=str,
        default=None,
        help="Path to train CSV/Parquet",
    )
    parser.add_argument(
        "--val",
        type=str,
        default=None,
        help="Path to validation CSV/Parquet",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Target label column name",
    )
    parser.add_argument(
        "--models-out-dir",
        type=str,
        default=None,
        help="Directory to save trained model pipelines",
    )
    parser.add_argument(
        "--metrics-out-json",
        type=str,
        default=None,
        help="Path to save detailed training/validation metrics JSON",
    )
    parser.add_argument(
        "--leaderboard-out-csv",
        type=str,
        default=None,
        help="Path to save validation leaderboard CSV",
    )
    parser.add_argument(
        "--best-model-out",
        type=str,
        default=None,
        help="Path to save the selected best model",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = merge_cli_args(args, load_config(TrainCLIConfig, "train"))

    cfg = TrainValidateConfig(
        label_col=settings.label_col,
        models_out_dir=settings.models_out_dir,
        metrics_out_json=settings.metrics_out_json,
        leaderboard_out_csv=settings.leaderboard_out_csv,
        best_model_out=settings.best_model_out,
    )

    result = run_training(
        train_path=settings.train,
        val_path=settings.val,
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