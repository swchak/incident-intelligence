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
    with_parent_dir_suffix,
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
        "--cv",
        type=int,
        default=None,
        help="Number of cross-validation folds for GridSearchCV",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for GridSearchCV (-1 uses all cores)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=None,
        help="Verbosity for GridSearchCV",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default=None,
        help="Scoring metric for model selection during GridSearchCV",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model aliases, e.g. logistic,rf,gb,svm",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use smaller grids intended for faster iteration",
    )
    parser.add_argument(
    "--dataset-kind",
    choices=["snapshot", "temporal"],
    default="snapshot",
)
    return parser


def _resolve_train_val_paths(
    dataset_kind: str,
    settings: TrainCLIConfig,
    *,
    train_override: str | None = None,
    val_override: str | None = None,
) -> tuple[str, str]:
    train_path = train_override
    val_path = val_override

    if dataset_kind == "snapshot":
        train_path = train_path or settings.train_snapshot
        val_path = val_path or settings.val_snapshot
    elif dataset_kind == "temporal":
        train_path = train_path or settings.train_temporal
        val_path = val_path or settings.val_temporal
    else:
        raise ValueError(
            f"Unsupported dataset_kind='{dataset_kind}'. "
            "Expected one of: ['snapshot', 'temporal']"
        )

    return train_path, val_path


def main() -> None:
    """Parse settings, execute training, and print the best-model summary.

    CLI-provided values take precedence over config-file values via
    ``merge_cli_args``. The resulting settings are translated into the runtime
    ``TrainValidateConfig`` expected by ``run_training``.
    """
    parser = build_parser()
    args = parser.parse_args()
    dataset_kind = args.dataset_kind
    if args.models is not None:
        args.models = tuple(
            part.strip() for part in args.models.split(",") if part.strip()
        )

    # Load the named training config and let explicit CLI flags override it.
    settings = merge_cli_args(args, load_config(TrainCLIConfig, "train"))
    train_path, val_path = _resolve_train_val_paths(
        dataset_kind,
        settings,
        train_override=args.train,
        val_override=args.val,
    )


    cfg = TrainValidateConfig(
        label_col=settings.label_col or "root_cause_label",
        models_out_dir=(
            args.models_out_dir
            or (
                "artifacts/models"
                if dataset_kind == "snapshot"
                else with_dataset_suffix("artifacts/models", dataset_kind)
            )
        ),
        metrics_out_json=(
            args.metrics_out_json
            or with_parent_dir_suffix(
                "artifacts/metrics/train_val_results.json",
                dataset_kind,
            )
        ),
        leaderboard_out_csv=(
            args.leaderboard_out_csv
            or with_parent_dir_suffix(
                "artifacts/metrics/leaderboard_val.csv",
                dataset_kind,
            )
        ),
        best_model_out=(
            args.best_model_out
            or with_parent_dir_suffix(
                "artifacts/models/best_model.joblib",
                dataset_kind,
            )
        ),
        cv=settings.cv,
        n_jobs=settings.n_jobs,
        verbose=settings.verbose,
        scoring=settings.scoring,
        models=settings.models,
        fast_mode=settings.fast_mode,
    )

    result = run_training_for_dataset_kind(
        dataset_kind=dataset_kind,
        cfg=cfg,
        train_path=train_path,
        val_path=val_path,
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
