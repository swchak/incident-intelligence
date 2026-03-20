"""CLI entrypoint for training and validating baseline incident models.

This module exposes a small command-line interface that:
1. parses optional file/output overrides,
2. merges them with the named ``train`` application config,
3. builds a :class:`TrainValidateConfig`, and
4. runs the training pipeline before printing a short summary.

The heavy lifting lives in ``incident_intelligence.modeling.train``; this file
primarily wires CLI arguments into the shared training/configuration helpers.
"""

from __future__ import annotations

import argparse

from incident_intelligence.modeling.train import (
    TrainValidateConfig,
    run_training,
    run_training_for_dataset_kind,
    with_dataset_suffix,
)
from incident_intelligence.config import (
    TrainCLIConfig,
    load_config,
    merge_cli_args,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI.

    All arguments are optional so callers can rely on values loaded from the
    ``train`` configuration section and only override the fields they need.

    Returns:
        Configured argument parser for the training command.
    """
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
    parser.add_argument(
    "--dataset-kind",
    choices=["snapshot", "temporal"],
    default="snapshot",
)
    return parser



def main() -> None:
    """Parse settings, execute training, and print the best-model summary.

    CLI-provided values take precedence over config-file values via
    ``merge_cli_args``. The resulting settings are translated into the runtime
    ``TrainValidateConfig`` expected by ``run_training``.
    """
    parser = build_parser()
    args = parser.parse_args()
    dataset_kind = args.dataset_kind

    # Load the named training config and let explicit CLI flags override it.
    settings = merge_cli_args(args, load_config(TrainCLIConfig, "train"))


    cfg = TrainValidateConfig(
        label_col=settings.label_col or "root_cause_label",
        models_out_dir=(
            settings.models_out_dir
            or (
                "artifacts/models"
                if dataset_kind == "snapshot"
                else with_dataset_suffix("artifacts/models", dataset_kind)
            )
        ),
        metrics_out_json=(
            settings.metrics_out_json
            or with_dataset_suffix(
                "artifacts/metrics/train_val_results.json",
                dataset_kind,
            )
        ),
        leaderboard_out_csv=(
            settings.leaderboard_out_csv
            or with_dataset_suffix(
                "artifacts/metrics/leaderboard_val.csv",
                dataset_kind,
            )
        ),
        best_model_out=(
            settings.best_model_out
            or with_dataset_suffix(
                "artifacts/models/best_model.joblib",
                dataset_kind,
            )
        ),
    )

    result = run_training_for_dataset_kind(
        dataset_kind=dataset_kind,
        cfg=cfg,
    )

    # Emit a concise terminal summary for humans while richer artifacts are
    # written to the configured output paths.
    best = result["best_model"]
    print(
        f"Training complete. Best model: {best['model_name']} "
        f"(val_f1_macro={best['val_f1_macro']:.4f}, "
        f"val_accuracy={best['val_accuracy']:.4f})"
    )


if __name__ == "__main__":
    main()
