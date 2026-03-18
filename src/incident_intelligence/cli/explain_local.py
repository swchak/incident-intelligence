from __future__ import annotations

import argparse

from incident_intelligence.config import (
    ExplainLocalCLIConfig,
    load_config,
    merge_cli_args,
)

from incident_intelligence.modeling.explain_local import (
    ExplainLocalConfig,
    run_local_explainability,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local explainability artifacts for selected evaluation examples."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to evaluation CSV/Parquet including label column",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a single saved .joblib model",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Target label column name",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Base directory for explainability artifacts",
    )
    parser.add_argument(
        "--background-n",
        type=int,
        default=None,
        help="Number of background samples for SHAP explainers",
    )
    parser.add_argument(
        "--explain-n",
        type=int,
        default=None,
        help="Unused by local explainability today, kept for config consistency",
    )
    parser.add_argument(
        "--kernel-bg",
        type=int,
        default=None,
        help="Background size for kernel SHAP",
    )
    parser.add_argument(
        "--kernel-nsamples",
        type=int,
        default=None,
        help="Number of samples for kernel SHAP",
    )
    parser.add_argument(
        "--perm-repeats",
        type=int,
        default=None,
        help="Unused by local explainability today, kept for config consistency",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
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
        default=None,
        help="Number of eval examples to explain when row indices are not provided",
    )
    parser.add_argument(
        "--top-k-classes",
        type=int,
        default=None,
        help="Number of candidate classes to include in RCA output",
    )
    parser.add_argument(
        "--top-features-per-class",
        type=int,
        default=None,
        help="Number of most influential features to show per class",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = merge_cli_args(args, load_config(ExplainLocalCLIConfig, "explain_local"))

    cfg = ExplainLocalConfig(
        label_col=settings.label_col,
        out_dir=settings.out_dir,
        background_n=settings.background_n,
        kernel_bg=settings.kernel_bg,
        kernel_nsamples=settings.kernel_nsamples,
        random_state=settings.random_state,
        top_k=settings.top_k,
    )

    result = run_local_explainability(
        data_path=settings.data,
        model_path=settings.model,
        cfg=cfg,
        row_indices=settings.row_indices,
        n_examples=settings.n_examples,
        top_k_classes=settings.top_k_classes,
        top_features_per_class=settings.top_features_per_class,
    )

    print(f"Generated local explainability for model: {result.get('model', 'unknown')}")
    print(
        f"Method: local SHAP explainability with "
        f"{cfg.kernel_nsamples} samples and background size {cfg.kernel_bg}"
    )
    print(f"Explained rows: {len(result.get('rows', []))}")
    print(f"Output directory: {result.get('out_dir', 'N/A')}")

    for row in result.get("rows", []):
        print(
            f"row={row['row_index']} "
            f"true={row['true_label']} "
            f"pred={row['predicted_label']} "
            f"json={row.get('json_path', 'N/A')} "
            f"markdown={row.get('markdown_path', 'N/A')}"
        )

        for cls in row.get("classes", []):
            print(
                f"  class={cls['class']} "
                f"waterfall={cls.get('waterfall_plot', 'N/A')}"
            )


if __name__ == "__main__":
    main()