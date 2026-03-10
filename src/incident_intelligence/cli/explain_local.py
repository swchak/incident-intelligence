from __future__ import annotations

import argparse

from incident_intelligence.modeling.explain import ExplainConfig, run_local_explainability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local explainability artifacts for selected evaluation examples."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/incident_root_cause_eval.csv",
        help="Path to evaluation CSV/Parquet including label column",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a single saved .joblib model",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="root_cause_label",
        help="Target label column name",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="artifacts/explain",
        help="Base directory for explainability artifacts",
    )
    parser.add_argument(
        "--background-n",
        type=int,
        default=100,
        help="Number of background samples for SHAP explainers",
    )
    parser.add_argument(
        "--explain-n",
        type=int,
        default=200,
        help="Unused by local explainability today, kept for config consistency",
    )
    parser.add_argument(
        "--kernel-bg",
        type=int,
        default=40,
        help="Background size for kernel SHAP",
    )
    parser.add_argument(
        "--kernel-nsamples",
        type=int,
        default=80,
        help="Number of samples for kernel SHAP",
    )
    parser.add_argument(
        "--perm-repeats",
        type=int,
        default=10,
        help="Unused by local explainability today, kept for config consistency",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Maximum features shown in waterfall plot",
    )
    parser.add_argument(
        "--row-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit eval row indices to explain",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=3,
        help="Number of eval examples to explain when row indices are not provided",
    )
    parser.add_argument(
        "--top-k-classes",
        type=int,
        default=3,
        help="Number of candidate classes to include in RCA output",
    )
    parser.add_argument(
        "--top-features-per-class",
        type=int,
        default=8,
        help="Number of most influential features to show per class",
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

    result = run_local_explainability(
        data_path=args.data,
        model_path=args.model,
        cfg=cfg,
        row_indices=args.row_indices,
        n_examples=args.n_examples,
        top_k_classes=args.top_k_classes,
        top_features_per_class=args.top_features_per_class,
    )

    print(f"Generated local explainability for model: {result['model_name']}")
    print(f"Method: {result['method']}")
    print(f"Explained rows: {len(result['rows'])}")

    if result.get("html_reports"):
        index_pages = [
            p
            for p in result["html_reports"]
            if p.endswith("local_explainability_index.html")
        ]
        if index_pages:
            print(f"Open report index: {index_pages[0]}")

    for row in result["rows"]:
        print(
            f"  row={row['row_index']} "
            f"true={row['true_label']} "
            f"pred={row['predicted_label']} "
            f"waterfall={row['waterfall_png']}"
        )


if __name__ == "__main__":
    main()