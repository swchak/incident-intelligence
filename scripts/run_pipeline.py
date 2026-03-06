import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n" + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run end-to-end pipeline: generate -> train -> evaluate -> explain")

    # Data generation
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-size", type=float, default=0.70)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--label-col", type=str, default="root_cause_label")

    # Paths (match your conventions)
    p.add_argument("--train-path", type=str, default="data/processed/incident_root_cause_train.csv")
    p.add_argument("--val-path", type=str, default="data/processed/incident_root_cause_val.csv")
    p.add_argument("--eval-path", type=str, default="data/processed/incident_root_cause_eval.csv")

    p.add_argument("--models-dir", type=str, default="artifacts/models")
    p.add_argument("--metrics-dir", type=str, default="artifacts/metrics")
    p.add_argument("--explain-dir", type=str, default="artifacts/explain")

    args = p.parse_args()

    py = sys.executable  # ensures we use the current venv python

    # 1) Generate dataset (raw + splits)
    run([
        py, "scripts/generate_dataset.py",
        "--n-samples", str(args.n_samples),
        "--seed", str(args.seed),
        "--train-size", str(args.train_size),
        "--val-size", str(args.val_size),
        "--label-col", args.label_col,
        "--raw-out", "raw/incidents_raw.csv",
        "--processed-dir", "processed",
    ])

    # Sanity check files exist
    for fp in [args.train_path, args.val_path, args.eval_path]:
        if not Path(fp).exists():
            raise FileNotFoundError(f"Expected file not found after generation: {fp}")

    # 2) Train + validate (saves per-model pipelines + best_model.joblib + leaderboard)
    run([
        py, "scripts/train.py",
        "--train", args.train_path,
        "--val", args.val_path,
        "--label-col", args.label_col,
        "--models-out-dir", args.models_dir,
        "--metrics-out-json", str(Path(args.metrics_dir) / "train_val_results.json"),
        "--leaderboard-out-csv", str(Path(args.metrics_dir) / "leaderboard_val.csv"),
        "--best-model-out", str(Path(args.models_dir) / "best_model.joblib"),
    ])

    # 3) Evaluate on eval set (writes evaluation.json + summary csv)
    run([
        py, "scripts/evaluate.py",
        "--data", args.eval_path,
        "--label-col", args.label_col,
        "--models-dir", args.models_dir,
        "--metrics-out", str(Path(args.metrics_dir) / "evaluation.json"),
        "--summary-csv-out", str(Path(args.metrics_dir) / "evaluation_summary.csv"),
    ])

    # 4) Explain (writes per-model feature importance CSV/PNG + summary json)
    run([
        py, "scripts/explain.py",
        "--data", args.eval_path,
        "--label-col", args.label_col,
        "--models-dir", args.models_dir,
        "--out-dir", args.explain_dir,
    ])

    print("\n✅ Pipeline complete.")
    print("Outputs:")
    print(f"  data/raw/incidents_raw.csv")
    print(f"  {args.train_path}")
    print(f"  {args.val_path}")
    print(f"  {args.eval_path}")
    print(f"  {args.models_dir}/")
    print(f"  {args.metrics_dir}/")
    print(f"  {args.explain_dir}/")


if __name__ == "__main__":
    main()