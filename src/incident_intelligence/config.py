"""
Central configuration utilities for loading settings from pyproject.toml and merging with CLI args.

This module defines dataclasses for configuration schemas, functions to load those configs from
a pyproject.toml file, and a helper to merge CLI arguments with the loaded configuration.

The CLIConfig dataclasses serve as the schema for both the configuration file and the expected
CLI arguments for each command.

The expected usage pattern in CLI scripts is:
    1. Build an argparse.ArgumentParser with optional arguments corresponding to the config fields.
    2. Parse CLI arguments.
    3. Load the base configuration from pyproject.toml for the relevant section.
    4. Merge CLI arguments into the loaded configuration, overriding any fields that were specified on the CLI
    5. Use the resulting configuration object to run the desired workflow.

The configuration sections currently defined are:
    - generator: For synthetic snapshot dataset generation settings.
    - sequence_generator: For synthetic incident sequence generation settings.
    - temporal_features: For temporal feature-building settings.
    - train: For model training settings.
    - evaluate: For model evaluation settings.
    - explain: For model explainability settings.
    - explain_local: For local explainability settings on specific examples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar
import tomllib


T = TypeVar("T")


def find_pyproject(start: Path | None = None) -> Path | None:
    """
    Search for a pyproject.toml file starting from the given directory and moving up the directory tree.

    Args:
        start: Optional starting directory for the search. If None, the search will start from the current working directory.

    Returns:
        The Path to the found pyproject.toml file, or None if no such file is found in the current directory or any of its parents.
    """
    current = (start or Path.cwd()).resolve()
    for path in [current, *current.parents]:
        candidate = path / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def load_tool_section(
    section: str,
    pyproject_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load a specific section from the [tool.incident_intelligence] configuration in pyproject.toml.

    Args:
        section: The specific config section to load (e.g. "train", "evaluate", etc.)
        pyproject_path: Optional path to a specific pyproject.toml file. If None, the function will search for one starting from the current directory.

    Returns:
        A dictionary containing the configuration values from the specified section, or an empty dictionary if the file or section is not found.
    """
    path = Path(pyproject_path) if pyproject_path else find_pyproject()
    if path is None or not path.exists():
        return {}

    with path.open("rb") as f:
        data = tomllib.load(f)

    return (
        data.get("tool", {})
        .get("incident_intelligence", {})
        .get(section, {})
    )


def load_config(
    cls: type[T],
    section: str,
    pyproject_path: str | Path | None = None,
) -> T:
    """
    Load a configuration dataclass instance of type cls from the specified section in pyproject.toml.
    Only fields defined in the dataclass will be loaded; extra fields in the config section will be ignored.
    Missing keys will fall back to the dataclass defaults.

    Args:
    cls: The dataclass type to instantiate with the loaded configuration values.
    section: The specific config section to load (e.g. "train", "evaluate", etc.)
    pyproject_path: Optional path to a specific pyproject.toml file. If None, the function will search for one starting from the current directory.

    Returns:
    An instance of cls populated with the configuration values from the specified section, merged with any defaults defined in the dataclass.
    """
    raw = load_tool_section(section, pyproject_path)
    valid_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return cls(**filtered)


def merge_cli_args(args: Any, config: T) -> T:
    """
    Merge CLI arguments into a configuration dataclass instance, overriding any fields that were specified on the CLI.
    Only fields that are not None in the CLI arguments will override the config values.

    Args:
        args: The parsed CLI arguments, typically from argparse.Namespace.
        config: The configuration dataclass instance to merge with the CLI arguments.

    Returns:
        A new instance of the same type as config, with values overridden by any non-None CLI arguments.
    """
    merged = asdict(config)
    for key, value in vars(args).items():
        if key in merged and value is not None:
            merged[key] = value
    return type(config)(**merged)


@dataclass
class GeneratorCLIConfig:
    """
    Configuration schema for the dataset generation CLI. All fields have defaults that can be overridden 
    by CLI arguments or pyproject.toml settings.
    """
    n_samples: int = 10000
    seed: int = 42
    raw_out: str = "raw/incidents_raw.csv"
    processed_dir: str = "processed"
    train_size: float = 0.70
    val_size: float = 0.15
    label_col: str = "root_cause_label"

@dataclass
class SequenceGeneratorCLIConfig:
    """
    Configuration schema for the incident sequence generation CLI.
    """
    n_incidents: int = 5000
    sequence_length: int = 20
    random_seed: int = 42
    output: str = "data/raw/incidents_sequence_raw.csv"
    label_probs: dict[str, float] = field(
        default_factory=lambda: {
            "memory_leak": 0.18,
            "bad_deployment": 0.15,
            "external_dependency_failure": 0.17,
            "cpu_saturation": 0.16,
            "traffic_spike": 0.14,
            "normal": 0.20,
        }
    )


@dataclass
class TemporalFeaturesCLIConfig:
    """
    Configuration schema for the temporal feature-building CLI.
    """
    input: str = "data/raw/incidents_sequence_raw.csv"
    output_dir: str = "data/processed"

@dataclass
class TrainCLIConfig:
    """
    Configuration schema for the training CLI. All fields have defaults that can be overridden by 
    CLI arguments or pyproject.toml settings.
    """
    train: str = "data/processed/incident_snapshot_train.csv"
    val: str = "data/processed/incident_snapshot_val.csv"
    label_col: str = "root_cause_label"
    models_out_dir: str = "artifacts/models"
    metrics_out_json: str = "artifacts/metrics/train_val_results.json"
    leaderboard_out_csv: str = "artifacts/metrics/leaderboard_val.csv"
    best_model_out: str = "artifacts/models/best_model.joblib"
    cv: int = 5
    n_jobs: int = -1
    verbose: int = 1
    scoring: str = "f1_macro"
    models: tuple[str, ...] | None = None
    fast_mode: bool = False


@dataclass
class EvaluateCLIConfig:
    """
    Configuration schema for the evaluation CLI. All fields have defaults that can be overridden 
    by CLI arguments or pyproject.toml settings.
    """
    data: str = "data/processed/incident_snapshot_eval.csv"
    label_col: str = "root_cause_label"
    models_dir: str = "artifacts/models"
    model: str | None = None
    metrics_out: str = "artifacts/metrics/evaluation.json"
    summary_csv_out: str = "artifacts/metrics/evaluation_summary.csv"


@dataclass
class ExplainCLIConfig:
    """
    Configuration schema for the explanation CLI. All fields have defaults that can be overridden 
    by CLI arguments or pyproject.toml settings.
    """
    data: str = "data/processed/incident_snapshot_eval.csv"
    label_col: str = "root_cause_label"
    models_dir: str = "artifacts/models"
    model: str | None = None
    out_dir: str = "artifacts/explain"
    background_n: int = 100
    explain_n: int = 200
    kernel_bg: int = 40
    kernel_nsamples: int = 80
    perm_repeats: int = 10
    random_state: int = 42
    top_k: int = 20


@dataclass
class ExplainLocalCLIConfig:
    """
    Configuration schema for the local explanation CLI. All fields have defaults that can be overridden 
    by CLI arguments or pyproject.toml settings.
    """
    data: str = "data/processed/incident_snapshot_eval.csv"
    label_col: str = "root_cause_label"
    out_dir: str = "artifacts/explain"
    background_n: int = 100
    explain_n: int = 200
    kernel_bg: int = 40
    kernel_nsamples: int = 80
    perm_repeats: int = 10
    random_state: int = 42
    top_k: int = 20
    row_indices: list[int] | None = None
    n_examples: int = 3
    top_k_classes: int = 3
    top_features_per_class: int = 8
    model: str = "artifacts/models/best_model.joblib"


"""
Registry of config sections to their corresponding dataclass types for easy loading in CLI scripts.
This mapping allows CLI scripts to load the appropriate configuration section from pyproject.toml into a 
strongly-typed dataclass instance, which can then be merged with CLI arguments for flexible configuration management.
"""
CONFIG_SECTIONS: dict[str, type] = {
    "generator": GeneratorCLIConfig,
    "sequence_generator": SequenceGeneratorCLIConfig,
    "temporal_features": TemporalFeaturesCLIConfig,
    "train": TrainCLIConfig,
    "evaluate": EvaluateCLIConfig,
    "explain": ExplainCLIConfig,
    "explain_local": ExplainLocalCLIConfig,
}


def load_named_config(
    section: str,
    pyproject_path: str | Path | None = None,
):
    """
    Load a configuration dataclass instance for the specified section from pyproject.toml using the CONFIG_SECTIONS registry.

    Args:
        section: The specific config section to load (e.g. "train", "evaluate", etc.)
        pyproject_path: Optional path to a specific pyproject.toml file. If None, the function will search for one starting from the current directory.

    Returns:    
        An instance of the dataclass corresponding to the specified section, populated with the configuration values from pyproject.toml.

    Raises:
        ValueError: If the specified section is not found in the CONFIG_SECTIONS registry.
    """
    try:
        cls = CONFIG_SECTIONS[section]
    except KeyError as e:
        raise ValueError(f"Unknown config section: {section}") from e

    return load_config(cls, section, pyproject_path)
