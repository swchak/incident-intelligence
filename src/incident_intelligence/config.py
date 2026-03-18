from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar
import tomllib


T = TypeVar("T")


def find_pyproject(start: Path | None = None) -> Path | None:
    """
    Search upward from the current working directory for pyproject.toml.
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
    Load one [tool.incident_intelligence.<section>] table from pyproject.toml.

    Returns an empty dict if pyproject.toml or the section is missing.
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
    Load config values from pyproject.toml into a dataclass.

    Unknown keys are ignored so the dataclass remains the schema.
    Missing keys fall back to the dataclass defaults.
    """
    raw = load_tool_section(section, pyproject_path)
    valid_fields = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return cls(**filtered)


def merge_cli_args(args: Any, config: T) -> T:
    """
    Override config values with CLI arguments when those args are not None.
    """
    merged = asdict(config)
    for key, value in vars(args).items():
        if key in merged and value is not None:
            merged[key] = value
    return type(config)(**merged)


@dataclass
class GeneratorCLIConfig:
    n_samples: int = 10000
    seed: int = 42
    raw_out: str = "raw/incidents_raw.csv"
    processed_dir: str = "processed"
    train_size: float = 0.70
    val_size: float = 0.15
    label_col: str = "root_cause_label"


@dataclass
class TrainCLIConfig:
    train: str = "data/processed/incident_root_cause_train.csv"
    val: str = "data/processed/incident_root_cause_val.csv"
    label_col: str = "root_cause_label"
    models_out_dir: str = "artifacts/models"
    metrics_out_json: str = "artifacts/metrics/train_val_results.json"
    leaderboard_out_csv: str = "artifacts/metrics/leaderboard_val.csv"
    best_model_out: str = "artifacts/models/best_model.joblib"


@dataclass
class EvaluateCLIConfig:
    data: str = "data/processed/incident_root_cause_eval.csv"
    label_col: str = "root_cause_label"
    models_dir: str = "artifacts/models"
    model: str | None = None
    metrics_out: str = "artifacts/metrics/evaluation.json"
    summary_csv_out: str = "artifacts/metrics/evaluation_summary.csv"


@dataclass
class ExplainCLIConfig:
    data: str = "data/processed/incident_root_cause_eval.csv"
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
    data: str = "data/processed/incident_root_cause_eval.csv"
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


CONFIG_SECTIONS: dict[str, type] = {
    "generator": GeneratorCLIConfig,
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
    Load one named config section using the registry above.
    """
    try:
        cls = CONFIG_SECTIONS[section]
    except KeyError as e:
        raise ValueError(f"Unknown config section: {section}") from e

    return load_config(cls, section, pyproject_path)