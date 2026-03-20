"""
build_temporal_features.py

Command-line interface for building temporal feature datasets from incident sequences.

This script loads configuration from a config file and CLI overrides,
constructs a TemporalFeaturesCLIConfig, and runs the temporal feature
engineering pipeline.

The pipeline produces:

- a full temporal feature dataset
- train/validation/evaluation splits

Outputs are written to disk and the resulting file paths are printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from incident_intelligence.config import (
    TemporalFeaturesCLIConfig,
    load_config,
    merge_cli_args,
)
from incident_intelligence.data.temporal_features import build_temporal_feature_dataset
from incident_intelligence.data.splitters import split_by_incident


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser for temporal feature generation.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with temporal feature generation arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build temporal features from incident sequences."
    )

    # Input sequence dataset path
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to sequence CSV",
    )

    # Output directory for processed datasets
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for processed temporal-feature outputs",
    )

    return parser


def main() -> None:
    """
    Entry point for temporal feature generation CLI.

    Workflow
    --------
    1. Parse CLI arguments
    2. Load default configuration
    3. Merge CLI overrides into configuration
    4. Load input sequence dataset
    5. Build temporal feature dataset
    6. Split dataset into train/val/eval
    7. Save outputs to disk
    8. Print resulting file paths
    """
    # Parse CLI arguments
    parser = build_parser()
    args = parser.parse_args()

    # Load base configuration and merge CLI overrides
    settings = merge_cli_args(
        args,
        load_config(TemporalFeaturesCLIConfig, "temporal_features"),
    )

    input_path = Path(settings.input)
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sequence dataset
    sequence_df = pd.read_csv(input_path)

    # Build temporal features
    feature_df = build_temporal_feature_dataset(sequence_df)

    # Split into train/val/eval
    train_df, val_df, test_df = split_by_incident(feature_df)

    # Save outputs
    all_path = output_dir / "incident_temporal_all.csv"
    train_path = output_dir / "incident_temporal_train.csv"
    val_path = output_dir / "incident_temporal_val.csv"
    eval_path = output_dir / "incident_temporal_eval.csv"

    feature_df.to_csv(all_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(eval_path, index=False)

    # Report generated output locations
    print("Temporal feature datasets written successfully.")
    print(f"All:   {all_path}")
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Eval:  {eval_path}")


if __name__ == "__main__":
    main()