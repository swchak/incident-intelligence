from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


TIME_SERIES_FEATURES = [
    "avg_cpu_usage",
    "mem_growth",
    "oom_log_count",
    "request_rate",
    "error_rate",
    "latency",
    "upstream_error_rate",
    "dependency_latency",
    "timeout_log_count",
]


def _safe_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    slope, _ = np.polyfit(x, values, 1)
    return float(slope)


def _spike_ratio(values: np.ndarray) -> float:
    mean_val = float(np.mean(values))
    if abs(mean_val) < 1e-8:
        return 0.0
    return float(np.max(values) / mean_val)


def _end_minus_start(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(values[-1] - values[0])


def _auc(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.trapezoid(values))


def _build_feature_dict(values: np.ndarray, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_start": float(values[0]),
        f"{prefix}_end": float(values[-1]),
        f"{prefix}_delta": _end_minus_start(values),
        f"{prefix}_slope": _safe_slope(values),
        f"{prefix}_spike_ratio": _spike_ratio(values),
        f"{prefix}_auc": _auc(values),
    }


def build_temporal_feature_dataset(sequence_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"incident_id", "timestep", "root_cause_label", *TIME_SERIES_FEATURES}
    missing = required_cols.difference(sequence_df.columns)
    if missing:
        raise ValueError(f"sequence_df is missing required columns: {sorted(missing)}")

    df = sequence_df.sort_values(["incident_id", "timestep"]).copy()

    rows: List[Dict] = []
    for incident_id, group in df.groupby("incident_id", sort=True):
        row: Dict[str, float | str | int] = {
            "incident_id": int(incident_id),
            "root_cause_label": str(group["root_cause_label"].iloc[0]),
        }

        for feature in TIME_SERIES_FEATURES:
            values = group[feature].to_numpy(dtype=float)
            row.update(_build_feature_dict(values, feature))

        # A few cross-metric relationships
        cpu = group["avg_cpu_usage"].to_numpy(dtype=float)
        latency = group["latency"].to_numpy(dtype=float)
        req = group["request_rate"].to_numpy(dtype=float)
        dep_lat = group["dependency_latency"].to_numpy(dtype=float)

        if len(cpu) > 1 and np.std(cpu) > 1e-8 and np.std(latency) > 1e-8:
            row["cpu_latency_corr"] = float(np.corrcoef(cpu, latency)[0, 1])
        else:
            row["cpu_latency_corr"] = 0.0

        if len(req) > 1 and np.std(req) > 1e-8 and np.std(latency) > 1e-8:
            row["request_latency_corr"] = float(np.corrcoef(req, latency)[0, 1])
        else:
            row["request_latency_corr"] = 0.0

        row["dependency_to_service_latency_ratio"] = float(
            np.mean(dep_lat) / max(np.mean(latency), 1e-8)
        )

        row["oom_total"] = int(group["oom_log_count"].sum())
        row["timeout_total"] = int(group["timeout_log_count"].sum())

        rows.append(row)

    return pd.DataFrame(rows)