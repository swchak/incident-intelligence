from __future__ import annotations

import argparse

from incident_intelligence.config import EvaluateCLIConfig, load_config, merge_cli_args

from incident_intelligence.modeling.evaluate import (
    EvalConfig,
    run_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved model pipelines on an evaluation dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to eval CSV/Parquet including the label column",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Target label column name",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing saved .joblib pipelines",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional single .joblib model path; overrides --models-dir",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Path to save detailed evaluation JSON",
    )
    parser.add_argument(
        "--summary-csv-out",
        type=str,
        default=None,
        help="Path to save evaluation summary CSV",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = merge_cli_args(args, load_config(EvaluateCLIConfig, "evaluate"))

    cfg = EvalConfig(
        label_col=settings.label_col,
        metrics_out=settings.metrics_out,
        summary_csv_out=settings.summary_csv_out,
    )

    results = run_evaluation(
        data_path=settings.data,
        cfg=cfg,
        model_path=settings.model,
        models_dir=settings.models_dir,
    )

    best = None
    for m in results["models"]:
        acc = m["metrics"].get("accuracy", -1)
        if best is None or acc > best["metrics"].get("accuracy", -1):
            best = m

    if best:
        print(f"Evaluated {len(results['models'])} model(s).")
        print(f"Best by accuracy: {best['model_name']} ({best['metrics']['accuracy']:.4f})")
        print(f"Metrics saved to: {cfg.metrics_out}")
        if cfg.summary_csv_out:
            print(f"Summary saved to: {cfg.summary_csv_out}")


if __name__ == "__main__":
    main()