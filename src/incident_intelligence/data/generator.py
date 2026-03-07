from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from incident_intelligence.settings import SETTINGS, load_class_config


DEFAULT_ROOT_CAUSE_PROBS = {
    "bad_deployment": 0.2,
    "external_dependency_failure": 0.2,
    "traffic_spike": 0.15,
    "memory_leak": 0.15,
    "cpu_saturation": 0.15,
    "normal": 0.15,
}


@dataclass(frozen=True)
class GeneratorConfig:
    n_samples: int = 4000
    seed: int = 42
    label_col: str = "root_cause_label"
    train_size: float = 0.70
    val_size: float = 0.15
    raw_out: str = "raw/incidents_raw.csv"
    processed_dir: str = "processed"


def validate_configs(class_config):
    for root_cause, metric_mix in class_config.items():
        for metric, mixture in metric_mix.items():
            total = sum(p for p, _ in mixture)
            if not np.isclose(total, 1.0):
                raise ValueError(
                    f"config error: root_cause={root_cause} | metric={metric} | mixture probs sum to {total}"
                )


def apply_mixture(value, config):
    r = np.random.rand()
    cumulative = 0.0
    for prob, (mean, std) in config:
        cumulative += prob
        if r < cumulative:
            return value + np.random.normal(mean, std)
    return value


def generate_incident(root_cause, class_config):
    metrics = {
        "request_rate": np.random.normal(300, 50),
        "mem_growth": np.random.normal(0.2, 0.1),
        "error_rate": np.random.normal(0.5, 0.2),
        "upstream_error": np.random.normal(0.3, 0.2),
        "dependency_latency": np.random.normal(150, 40),
        "latency": np.random.normal(200, 50),
        "avg_cpu": np.random.normal(40, 10),
    }

    config = class_config[root_cause]

    for metric, mixture in config.items():
        if metric in metrics:
            metrics[metric] = apply_mixture(metrics[metric], mixture)

    metrics["avg_cpu"] += (0.05 * metrics["request_rate"] + np.random.normal(5, 3))
    metrics["avg_cpu"] = np.clip(metrics["avg_cpu"], 0, 100)
    metrics["mem_growth"] += 0.005 * metrics["avg_cpu"]
    metrics["dependency_latency"] += 0.1 * max(metrics["request_rate"] - 300, 0)

    metrics["latency"] += (
        0.5 * metrics["avg_cpu"]
        + 0.3 * metrics["dependency_latency"]
        + 10 * metrics["mem_growth"]
        + np.random.normal(50, 15)
    )
    metrics["latency"] = max(metrics["latency"], 50)

    from scipy.special import expit as sigmoid

    bias = -4.0
    system_stress_score = (
        bias
        + 0.02 * metrics["avg_cpu"]
        + 1.0 * metrics["mem_growth"]
        + 0.0015 * metrics["dependency_latency"]
        + 0.05 * metrics["upstream_error"]
        + 0.002 * metrics["request_rate"]
        + 0.01 * metrics["latency"]
    )
    system_stress_score += 2.0 * metrics["error_rate"]
    metrics["error_rate"] += sigmoid(system_stress_score)

    oom_logs = np.random.poisson(max(metrics["mem_growth"] * 2, 0.5))
    timeout_logs = np.random.poisson(max(metrics["dependency_latency"] / 100, 1))

    metrics["request_rate"] = max(metrics["request_rate"], 0)
    metrics["dependency_latency"] = max(metrics["dependency_latency"], 1)
    metrics["mem_growth"] = max(metrics["mem_growth"], 0)
    metrics["error_rate"] = max(metrics["error_rate"], 0)

    return {
        "avg_cpu_usage": metrics["avg_cpu"],
        "mem_growth": metrics["mem_growth"],
        "oom_log_count": oom_logs,
        "request_rate": metrics["request_rate"],
        "error_rate": metrics["error_rate"],
        "latency": metrics["latency"],
        "upstream_error_rate": metrics["upstream_error"],
        "dependency_latency": metrics["dependency_latency"],
        "timeout_log_count": timeout_logs,
        "root_cause_label": root_cause,
    }


def generate_dataset(n_samples, root_cause_probs, class_config, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data = []
    labels = list(root_cause_probs.keys())
    weights = list(root_cause_probs.values())

    for _ in range(n_samples):
        root_cause = random.choices(labels, weights=weights, k=1)[0]
        data.append(generate_incident(root_cause, class_config))

    return pd.DataFrame(data)


def stratified_splits(df: pd.DataFrame, label_col: str, seed: int, train_size: float, val_size: float):
    eval_size = 1.0 - train_size - val_size
    if eval_size <= 0:
        raise ValueError("train_size + val_size must be < 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df[label_col],
        random_state=seed,
    )

    val_share_of_temp = val_size / (val_size + eval_size)

    val_df, eval_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        stratify=temp_df[label_col],
        random_state=seed,
    )

    return train_df, val_df, eval_df


def generate_and_save_datasets(
    cfg: GeneratorConfig,
    root_cause_probs: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    root_cause_probs = root_cause_probs or DEFAULT_ROOT_CAUSE_PROBS

    class_config = load_class_config()
    validate_configs(class_config)

    df = generate_dataset(
        cfg.n_samples,
        root_cause_probs,
        class_config,
        seed=cfg.seed,
    )

    raw_path = SETTINGS.data_dir / cfg.raw_out
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)

    train_df, val_df, eval_df = stratified_splits(
        df,
        label_col=cfg.label_col,
        seed=cfg.seed,
        train_size=cfg.train_size,
        val_size=cfg.val_size,
    )

    processed_dir = SETTINGS.data_dir / cfg.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "incident_root_cause_train.csv"
    val_path = processed_dir / "incident_root_cause_val.csv"
    eval_path = processed_dir / "incident_root_cause_eval.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    return {
        "raw_path": raw_path,
        "train_path": train_path,
        "val_path": val_path,
        "eval_path": eval_path,
        "n_raw": len(df),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_eval": len(eval_df),
    }