"""
generator.py

Command-line interface for generating synthetic incident datasets.

This script loads configuration from a config file and CLI overrides,
constructs a GeneratorConfig, and runs the dataset generation pipeline.
The pipeline produces:

- a raw synthetic dataset
- processed train/validation/evaluation splits

Outputs are written to disk and the resulting file paths are printed.
"""

import argparse

from incident_intelligence.config import load_config, merge_cli_args, GeneratorCLIConfig
from incident_intelligence.data.generate_snapshot import GeneratorConfig, generate_and_save_datasets


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for dataset generation.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with dataset generation arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic incident data and save train/val/eval splits."
    )

    # Total number of synthetic samples to generate
    parser.add_argument("--n-samples", type=int, default=None)

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=None)

    # Path where the raw generated dataset will be saved
    parser.add_argument("--raw-out", type=str, default=None)

    # Directory where processed train/val/eval splits will be written
    parser.add_argument("--processed-dir", type=str, default=None)

    # Proportion of dataset used for training split
    parser.add_argument("--train-size", type=float, default=None)

    # Proportion of dataset used for validation split
    parser.add_argument("--val-size", type=float, default=None)

    # Name of the label column in the generated dataset
    parser.add_argument("--label-col", type=str, default=None)

    return parser


def main() -> None:
    """
    Entry point for the synthetic data generation CLI.

    Workflow
    --------
    1. Parse CLI arguments
    2. Load default configuration
    3. Merge CLI overrides into configuration
    4. Construct a GeneratorConfig object
    5. Generate datasets and save them to disk
    6. Print resulting file paths
    """

    # Parse command-line arguments
    parser = build_parser()
    args = parser.parse_args()

    # Load base configuration and merge CLI overrides
    settings = merge_cli_args(args, load_config(GeneratorCLIConfig, "generator"))

    # Convert CLI settings into the internal generation configuration
    cfg = GeneratorConfig(
        n_samples=settings.n_samples,
        seed=settings.seed,
        raw_out=settings.raw_out,
        processed_dir=settings.processed_dir,
        train_size=settings.train_size,
        val_size=settings.val_size,
        label_col=settings.label_col,
    )

    # Run dataset generation pipeline
    result = generate_and_save_datasets(cfg)

    # Report generated output locations
    print(f"Wrote raw:   {result['raw_path']}")
    print(f"Wrote train: {result['train_path']}")
    print(f"Wrote val:   {result['val_path']}")
    print(f"Wrote eval:  {result['eval_path']}")


if __name__ == "__main__":
    main()