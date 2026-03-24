"""
generate_sequence.py

Command-line interface for generating synthetic incident sequence datasets.

This script loads configuration from a config file and CLI overrides,
and runs the sequence dataset generation pipeline.

The pipeline produces:

- a raw synthetic incident sequence dataset

Output is written to disk and the resulting file path is printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from incident_intelligence.config import (
    SequenceGeneratorCLIConfig,
    load_config,
    merge_cli_args,
)
from incident_intelligence.data.generate_sequence import generate_sequence_dataset


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for sequence dataset generation.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with sequence dataset generation arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic incident sequence data and save it to disk."
    )

    # Total number of incident sequences to generate
    parser.add_argument("--n-incidents", type=int, default=None)

    # Number of timesteps per incident sequence
    parser.add_argument("--sequence-length", type=int, default=None)

    # Random seed for reproducibility
    parser.add_argument("--random-seed", type=int, default=None)

    # Path where the raw generated sequence dataset will be saved
    parser.add_argument("--output", type=str, default=None)

    return parser


def main() -> None:
    """
    Entry point for the synthetic incident sequence generation CLI.

    Workflow
    --------
    1. Parse CLI arguments
    2. Load default configuration
    3. Merge CLI overrides into configuration
    4. Generate the sequence dataset and save it to disk
    5. Print the resulting file path
    """
    # Parse command-line arguments
    parser = build_parser()
    args = parser.parse_args()

    # Load base configuration and merge CLI overrides
    settings = merge_cli_args(
        args,
        load_config(SequenceGeneratorCLIConfig, "sequence_generator"),
    )

    # Run sequence dataset generation pipeline
    df = generate_sequence_dataset(
        n_incidents=settings.n_incidents,
        sequence_length=settings.sequence_length,
        random_seed=settings.random_seed,
        label_probs=settings.label_probs,
    )

    output_path = Path(settings.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Report generated output location
    print(f"Wrote sequence dataset: {output_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
