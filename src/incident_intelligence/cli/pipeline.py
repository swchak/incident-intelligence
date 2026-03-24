"""
pipeline.py

End-to-end pipeline orchestrator for the incident intelligence project.

Supported workflows:
1. snapshot: synthetic snapshot dataset -> train -> evaluate -> explain
2. temporal: sequence generation -> temporal features -> train -> evaluate -> explain
"""

from __future__ import annotations

import argparse
from pathlib import Path

from incident_intelligence.config import (
    EvaluateCLIConfig,
    ExplainCLIConfig,
    GeneratorCLIConfig,
    SequenceGeneratorCLIConfig,
    TemporalFeaturesCLIConfig,
    TrainCLIConfig,
    load_config,
)
from incident_intelligence.data.generate_sequence import generate_sequence_dataset
from incident_intelligence.data.generate_snapshot import (
    GeneratorConfig,
    generate_and_save_datasets,
)
from incident_intelligence.data.splitters import split_by_incident
from incident_intelligence.data.temporal_features import build_temporal_feature_dataset
from incident_intelligence.modeling.evaluate import (
    EvalConfig,
    run_evaluation,
    run_evaluation_for_dataset_kind,
)
from incident_intelligence.modeling.explain import (
    ExplainConfig,
    run_explainability,
    run_explainability_for_dataset_kind,
)
from incident_intelligence.modeling.train import (
    TrainValidateConfig,
    run_training,
    run_training_for_dataset_kind,
    with_parent_dir_suffix,
    with_dataset_suffix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end incident intelligence pipeline."
    )
    parser.add_argument(
        "--dataset-kind",
        choices=["snapshot", "temporal"],
        default="snapshot",
        help="Run the snapshot workflow or the sequence->temporal workflow",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=None,
        help="Number of cross-validation folds for training",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel jobs for training GridSearchCV (-1 uses all cores)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=None,
        help="Verbosity for training GridSearchCV",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default=None,
        help="Scoring metric for training GridSearchCV",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated training model aliases, e.g. logistic,rf,gb,svm",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use smaller training grids intended for faster iteration",
    )
    return parser


def _models_dir_for_dataset_kind(base_models_dir: str, dataset_kind: str) -> str:
    if dataset_kind == "snapshot":
        return base_models_dir
    return with_dataset_suffix(base_models_dir, dataset_kind)


def _path_for_dataset_kind(path_str: str, dataset_kind: str) -> str:
    if dataset_kind == "snapshot":
        return path_str
    return with_dataset_suffix(path_str, dataset_kind)


def _path_in_dataset_dir(path_str: str, dataset_kind: str) -> str:
    if dataset_kind == "snapshot":
        return path_str
    return with_parent_dir_suffix(path_str, dataset_kind)


def _train_val_paths_for_dataset_kind(settings: TrainCLIConfig, dataset_kind: str) -> tuple[str, str]:
    if dataset_kind == "snapshot":
        return settings.train_snapshot, settings.val_snapshot
    if dataset_kind == "temporal":
        return settings.train_temporal, settings.val_temporal
    raise ValueError(
        f"Unsupported dataset_kind='{dataset_kind}'. "
        "Expected one of: ['snapshot', 'temporal']"
    )


def _model_paths_from_training_result(train_result: dict) -> list[str]:
    model_paths = [m["model_path"] for m in train_result.get("all_models", [])]
    best_model_path = train_result.get("best_model", {}).get("model_path")
    if best_model_path:
        model_paths.append(best_model_path)

    # Preserve order while removing duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for path in model_paths:
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def _run_snapshot_generation() -> None:
    gen_settings = load_config(GeneratorCLIConfig, "generator")
    gen_cfg = GeneratorConfig(
        n_samples=gen_settings.n_samples,
        seed=gen_settings.seed,
        raw_out=gen_settings.raw_out,
        processed_dir=gen_settings.processed_dir,
        train_size=gen_settings.train_size,
        val_size=gen_settings.val_size,
        label_col=gen_settings.label_col,
    )

    gen_result = generate_and_save_datasets(gen_cfg)

    print(f"[generate] raw:   {gen_result['raw_path']}")
    print(f"[generate] train: {gen_result['train_path']}")
    print(f"[generate] val:   {gen_result['val_path']}")
    print(f"[generate] eval:  {gen_result['eval_path']}")


def _run_temporal_generation() -> None:
    seq_settings = load_config(SequenceGeneratorCLIConfig, "sequence_generator")
    temporal_settings = load_config(TemporalFeaturesCLIConfig, "temporal_features")

    seq_df = generate_sequence_dataset(
        n_incidents=seq_settings.n_incidents,
        sequence_length=seq_settings.sequence_length,
        random_seed=seq_settings.random_seed,
        label_probs=seq_settings.label_probs,
    )

    sequence_output = Path(seq_settings.output)
    sequence_output.parent.mkdir(parents=True, exist_ok=True)
    seq_df.to_csv(sequence_output, index=False)

    feature_df = build_temporal_feature_dataset(seq_df)
    train_df, val_df, eval_df = split_by_incident(feature_df)

    output_dir = Path(temporal_settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_all_path = output_dir / "incident_temporal_all.csv"
    temporal_train_path = output_dir / "incident_temporal_train.csv"
    temporal_val_path = output_dir / "incident_temporal_val.csv"
    temporal_eval_path = output_dir / "incident_temporal_eval.csv"

    feature_df.to_csv(temporal_all_path, index=False)
    train_df.to_csv(temporal_train_path, index=False)
    val_df.to_csv(temporal_val_path, index=False)
    eval_df.to_csv(temporal_eval_path, index=False)

    print(f"[sequence] raw:   {sequence_output}")
    print(f"[temporal] all:   {temporal_all_path}")
    print(f"[temporal] train: {temporal_train_path}")
    print(f"[temporal] val:   {temporal_val_path}")
    print(f"[temporal] eval:  {temporal_eval_path}")


def main() -> None:
    args = build_parser().parse_args()
    dataset_kind = args.dataset_kind
    selected_models = None
    if args.models is not None:
        selected_models = tuple(
            part.strip() for part in args.models.split(",") if part.strip()
        )

    print("Running pipeline...\n")

    if dataset_kind == "snapshot":
        _run_snapshot_generation()
    else:
        _run_temporal_generation()

    train_settings = load_config(TrainCLIConfig, "train")
    train_path, val_path = _train_val_paths_for_dataset_kind(
        train_settings,
        dataset_kind,
    )
    train_cfg = TrainValidateConfig(
        label_col=train_settings.label_col,
        models_out_dir=_models_dir_for_dataset_kind(
            train_settings.models_out_dir,
            dataset_kind,
        ),
        metrics_out_json=_path_in_dataset_dir(
            train_settings.metrics_out_json,
            dataset_kind,
        ),
        leaderboard_out_csv=_path_in_dataset_dir(
            train_settings.leaderboard_out_csv,
            dataset_kind,
        ),
        best_model_out=_path_in_dataset_dir(
            train_settings.best_model_out,
            dataset_kind,
        ),
        cv=args.cv if args.cv is not None else train_settings.cv,
        n_jobs=args.n_jobs if args.n_jobs is not None else train_settings.n_jobs,
        verbose=args.verbose if args.verbose is not None else train_settings.verbose,
        scoring=args.scoring if args.scoring is not None else train_settings.scoring,
        models=selected_models if selected_models is not None else train_settings.models,
        fast_mode=args.fast_mode or train_settings.fast_mode,
    )

    if dataset_kind == "snapshot":
        train_result = run_training(
            train_path=train_path,
            val_path=val_path,
            cfg=train_cfg,
        )
    else:
        train_result = run_training_for_dataset_kind(
            dataset_kind=dataset_kind,
            cfg=train_cfg,
            train_path=train_path,
            val_path=val_path,
        )

    print(f"[train] best model: {train_result['best_model']['model_name']}")
    print(f"[train] best model path: {train_result['best_model']['model_path']}")
    trained_model_paths = _model_paths_from_training_result(train_result)

    eval_settings = load_config(EvaluateCLIConfig, "evaluate")
    eval_cfg = EvalConfig(
        label_col=eval_settings.label_col,
        metrics_out=_path_in_dataset_dir(eval_settings.metrics_out, dataset_kind),
        summary_csv_out=_path_in_dataset_dir(
            eval_settings.summary_csv_out,
            dataset_kind,
        ),
        plots_dir=(
            "artifacts/plots"
            if dataset_kind == "snapshot"
            else with_dataset_suffix("artifacts/plots", dataset_kind)
        ),
        reports_dir=(
            "artifacts/reports"
            if dataset_kind == "snapshot"
            else with_dataset_suffix("artifacts/reports", dataset_kind)
        ),
    )

    eval_models_dir = _models_dir_for_dataset_kind(
        eval_settings.models_dir,
        dataset_kind,
    )
    if dataset_kind == "snapshot":
        eval_result = run_evaluation(
            data_path="data/processed/incident_snapshot_eval.csv",
            cfg=eval_cfg,
            model_path=eval_settings.model,
            models_dir=eval_models_dir,
            model_paths=None if eval_settings.model else trained_model_paths,
        )
    else:
        eval_result = run_evaluation_for_dataset_kind(
            dataset_kind=dataset_kind,
            cfg=eval_cfg,
            model_path=eval_settings.model,
            models_dir=eval_models_dir,
            model_paths=None if eval_settings.model else trained_model_paths,
        )

    print(f"[evaluate] evaluated {len(eval_result['models'])} model(s)")

    explain_settings = load_config(ExplainCLIConfig, "explain")
    explain_cfg = ExplainConfig(
        label_col=explain_settings.label_col,
        out_dir=(
            explain_settings.out_dir
            if dataset_kind == "snapshot"
            else with_dataset_suffix(explain_settings.out_dir, dataset_kind)
        ),
        background_n=explain_settings.background_n,
        explain_n=explain_settings.explain_n,
        kernel_bg=explain_settings.kernel_bg,
        kernel_nsamples=explain_settings.kernel_nsamples,
        perm_repeats=explain_settings.perm_repeats,
        random_state=explain_settings.random_state,
        top_k=explain_settings.top_k,
    )

    explain_models_dir = _models_dir_for_dataset_kind(
        explain_settings.models_dir,
        dataset_kind,
    )
    if dataset_kind == "snapshot":
        explain_result = run_explainability(
            data_path="data/processed/incident_snapshot_eval.csv",
            cfg=explain_cfg,
            model_path=explain_settings.model,
            models_dir=explain_models_dir,
            model_paths=None if explain_settings.model else trained_model_paths,
        )
    else:
        explain_result = run_explainability_for_dataset_kind(
            dataset_kind=dataset_kind,
            cfg=explain_cfg,
            model_path=explain_settings.model,
            models_dir=explain_models_dir,
            model_paths=None if explain_settings.model else trained_model_paths,
        )

    print(f"[explain] generated artifacts for {len(explain_result['models'])} model(s)")
    print(f"[explain] out dir: {explain_result['out_dir']}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
