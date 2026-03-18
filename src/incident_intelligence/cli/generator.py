import argparse

from incident_intelligence.config import load_config, merge_cli_args, GeneratorCLIConfig
from incident_intelligence.data.generator import GeneratorConfig, generate_and_save_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic incident data and save train/val/eval splits."
    )

    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--raw-out", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--train-size", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--label-col", type=str, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = merge_cli_args(args, load_config(GeneratorCLIConfig, "generator"))

    cfg = GeneratorConfig(
        n_samples=settings.n_samples,
        seed=settings.seed,
        raw_out=settings.raw_out,
        processed_dir=settings.processed_dir,
        train_size=settings.train_size,
        val_size=settings.val_size,
        label_col=settings.label_col,
    )

    result = generate_and_save_datasets(cfg)

    print(f"Wrote raw:   {result['raw_path']}")
    print(f"Wrote train: {result['train_path']}")
    print(f"Wrote val:   {result['val_path']}")
    print(f"Wrote eval:  {result['eval_path']}")


if __name__ == "__main__":
    main()