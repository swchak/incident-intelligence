"""
explain.py

CLI for generating global explainability artifacts for trained models.

Produces model-level explanations such as:
- SHAP feature importance
- permutation importance
- summary plots
"""

from __future__ import annotations

import argparse

from incident_intelligence.config import (
    ExplainCLIConfig,
    load_config,
    merge_cli_args,
)
from incident_intelligence.modeling.explain import (
    ExplainConfig,
    run_explainability,
    run_explainability_for_dataset_kind,
)
from incident_intelligence.modeling.train import with_dataset_suffix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate explainability artifacts for trained models."
    )
    
    parser.add_argument("--data", type=str, default=None, help="Evaluation dataset")
    parser.add_argument("--label-col", type=str, default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model .joblib path (overrides models-dir)",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--background-n", type=int, default=None)
    parser.add_argument("--explain-n", type=int, default=None)
    parser.add_argument("--kernel-bg", type=int, default=None)
    parser.add_argument("--kernel-nsamples", type=int, default=None)
    parser.add_argument("--perm-repeats", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--dataset-kind",
        choices=["snapshot", "temporal"],
        default="snapshot",
        help="Use the standard processed eval dataset and dataset-specific artifact paths",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dataset_kind = args.dataset_kind

    settings = merge_cli_args(args, load_config(ExplainCLIConfig, "explain"))

    out_dir = args.out_dir or (
        "artifacts/explain"
        if dataset_kind == "snapshot"
        else with_dataset_suffix("artifacts/explain", dataset_kind)
    )
    models_dir = args.models_dir or (
        "artifacts/models"
        if dataset_kind == "snapshot"
        else with_dataset_suffix("artifacts/models", dataset_kind)
    )

    cfg = ExplainConfig(
        label_col=settings.label_col,
        out_dir=out_dir,
        background_n=settings.background_n,
        explain_n=settings.explain_n,
        kernel_bg=settings.kernel_bg,
        kernel_nsamples=settings.kernel_nsamples,
        perm_repeats=settings.perm_repeats,
        random_state=settings.random_state,
        top_k=settings.top_k,
    )

    if args.data is None:
        results = run_explainability_for_dataset_kind(
            dataset_kind=dataset_kind,
            cfg=cfg,
            model_path=settings.model,
            models_dir=models_dir,
        )
    else:
        results = run_explainability(
            data_path=settings.data,
            cfg=cfg,
            model_path=settings.model,
            models_dir=models_dir,
        )

    print(f"Generated explainability for {len(results['models'])} model(s).")
    print(f"Artifacts saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()
