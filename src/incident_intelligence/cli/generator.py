import argparse

from incident_intelligence.data.generator import GeneratorConfig, generate_and_save_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic incident data and save train/val/eval splits."
    )
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-out", type=str, default="raw/incidents_raw.csv")
    parser.add_argument("--processed-dir", type=str, default="processed")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--label-col", type=str, default="root_cause_label")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = GeneratorConfig(
        n_samples=args.n_samples,
        seed=args.seed,
        raw_out=args.raw_out,
        processed_dir=args.processed_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        label_col=args.label_col,
    )

    result = generate_and_save_datasets(cfg)

    print(f"Wrote raw:   {result['raw_path']}")
    print(f"Wrote train: {result['train_path']}")
    print(f"Wrote val:   {result['val_path']}")
    print(f"Wrote eval:  {result['eval_path']}")