import argparse
import pandas as pd

from incident_intelligence.modeling.explain import ExplainConfig, explain_models, find_models


def main():
    p = argparse.ArgumentParser(description="Generate explainability artifacts for saved models.")
    p.add_argument("--data", type=str, required=True, help="Eval dataset CSV/Parquet including label column.")
    p.add_argument("--label-col", type=str, default="root_cause_label")
    p.add_argument("--models-dir", type=str, default="artifacts/models")
    p.add_argument("--out-dir", type=str, default="artifacts/explain")

    p.add_argument("--background-n", type=int, default=100)
    p.add_argument("--explain-n", type=int, default=200)
    args = p.parse_args()

    df = pd.read_csv(args.data) if args.data.endswith(".csv") else pd.read_parquet(args.data)

    cfg = ExplainConfig(
        label_col=args.label_col,
        out_dir=args.out_dir,
        background_n=args.background_n,
        explain_n=args.explain_n,
    )

    models = find_models(args.models_dir)
    explain_models(models, df, cfg)


if __name__ == "__main__":
    main()