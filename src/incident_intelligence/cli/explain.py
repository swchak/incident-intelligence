from __future__ import annotations

import argparse

from incident_intelligence.modeling.explain import (
    ExplainConfig,
    run_explainability,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate explainability artifacts for trained models."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/incident_root_cause_eval.csv",
        help="Evaluation dataset",
    )

    parser.add_argument(
        "--label-col",
        type=str,
        default="root_cause_label",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        default="artifacts/models",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model .joblib path (overrides models-dir)",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/explain",
    )

    parser.add_argument(
        "--background-n",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--explain-n",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--kernel-bg",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--kernel-nsamples",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--perm-repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = ExplainConfig(
        label_col=args.label_col,
        out_dir=args.out_dir,
        background_n=args.background_n,
        explain_n=args.explain_n,
        kernel_bg=args.kernel_bg,
        kernel_nsamples=args.kernel_nsamples,
        perm_repeats=args.perm_repeats,
        random_state=args.random_state,
        top_k=args.top_k,
    )

    results = run_explainability(
        data_path=args.data,
        cfg=cfg,
        model_path=args.model,
        models_dir=args.models_dir,
    )

    print(f"Generated explainability for {len(results['models'])} model(s).")
    print(f"Artifacts saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()