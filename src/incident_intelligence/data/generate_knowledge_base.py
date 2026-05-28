from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

from incident_intelligence.settings import SETTINGS


ROOT_CAUSE_TEMPLATES: dict[str, dict[str, Any]] = {
    "memory_leak": {
        "summary": "Memory pressure built gradually before user-facing latency and errors increased.",
        "symptoms": [
            "Memory usage increased steadily throughout the incident window.",
            "OOM-related log activity appeared near the end of the incident.",
            "Latency degraded after sustained memory growth.",
        ],
        "recent_changes": [
            "Review recent cache, object-retention, or batch-processing changes.",
            "Check whether a shared dependency upgrade changed memory behavior.",
        ],
        "resolution": [
            "Recycle affected pods or instances to restore capacity.",
            "Inspect recent code paths for memory-retention regressions.",
        ],
        "lessons": [
            "Add alerting on memory slope, not just absolute thresholds.",
            "Record whether memory growth is traffic-dependent or time-dependent.",
        ],
        "runbook": {
            "when_to_use": "Use this runbook when memory rises over time, restarts help temporarily, and OOM signals appear near the end of the incident.",
            "investigation_steps": [
                "Confirm whether memory growth is monotonic or bursty.",
                "Review container restart history and OOM kill events.",
                "Compare the affected service against its normal traffic baseline.",
                "Inspect recent code changes involving caching or object retention.",
            ],
            "mitigation": [
                "Recycle unhealthy pods or instances.",
                "Reduce load or lower concurrency if memory pressure is compounding recovery.",
            ],
        },
    },
    "bad_deployment": {
        "summary": "Service health changed sharply after a release boundary, causing errors and elevated latency.",
        "symptoms": [
            "Error rate increased quickly after a version transition.",
            "Latency shifted materially after rollout.",
            "Rollback or traffic shift reduced symptoms.",
        ],
        "recent_changes": [
            "Inspect deployment timestamps and configuration changes near incident start.",
            "Compare the failing version against the last known healthy release.",
        ],
        "resolution": [
            "Roll back the deployment or shift traffic to the prior stable version.",
            "Inspect release diff and canary behavior before resuming rollout.",
        ],
        "lessons": [
            "Strengthen canary validation for request latency and error changes.",
            "Capture deployment metadata directly in incident timelines.",
        ],
        "runbook": {
            "when_to_use": "Use this runbook when symptoms line up closely with a release event or version rollout.",
            "investigation_steps": [
                "Compare the release timestamp with the first symptom spike.",
                "Check canary metrics and rollback history.",
                "Review application config, feature flags, and dependency changes in the release.",
                "Verify whether symptoms disappear on the previous version.",
            ],
            "mitigation": [
                "Pause or roll back the deployment.",
                "Limit blast radius by routing traffic away from the new version.",
            ],
        },
    },
    "external_dependency_failure": {
        "summary": "An external provider or upstream dependency degraded, amplifying latency and errors in dependency-backed request paths.",
        "symptoms": [
            "Upstream or provider-specific error rates rose sharply.",
            "Latency increased as retries accumulated.",
            "Core application CPU stayed closer to baseline than the user-facing failure severity suggested.",
        ],
        "recent_changes": [
            "Check provider status pages or incident channels.",
            "Review retry, timeout, and circuit-breaker settings for dependency-backed requests.",
        ],
        "resolution": [
            "Enable degraded-mode behavior or bypass flows where available.",
            "Tighten retries if retry storms are amplifying latency.",
        ],
        "lessons": [
            "Add provider-specific alerting before global checkout failures surface.",
            "Review whether retry policies worsen dependency outages.",
        ],
        "runbook": {
            "when_to_use": "Use this runbook when dependency latency and upstream errors increase while core service resources remain relatively stable.",
            "investigation_steps": [
                "Identify which dependency-backed routes are failing.",
                "Check external provider health and status pages.",
                "Review timeout, retry, and circuit-breaker behavior.",
                "Confirm whether fallback paths are available and working.",
            ],
            "mitigation": [
                "Switch to degraded mode or fallback flows.",
                "Reduce retry amplification during the outage window.",
            ],
        },
    },
    "cpu_saturation": {
        "summary": "Sustained compute pressure pushed CPU utilization toward saturation and caused broad performance degradation.",
        "symptoms": [
            "CPU usage remained near saturation for an extended window.",
            "Latency increased alongside CPU pressure.",
            "Throughput flattened while request pressure remained high.",
        ],
        "recent_changes": [
            "Review batch jobs, expensive queries, or concurrency changes.",
            "Check whether autoscaling lagged behind incoming demand.",
        ],
        "resolution": [
            "Scale out capacity and reduce expensive background work.",
            "Profile the hottest code paths involved in the incident.",
        ],
        "lessons": [
            "Alert on sustained CPU pressure and queue growth together.",
            "Separate background compute from latency-sensitive request traffic.",
        ],
        "runbook": {
            "when_to_use": "Use this runbook when CPU remains elevated for a sustained period and latency rises with it.",
            "investigation_steps": [
                "Confirm whether saturation is localized or fleet-wide.",
                "Review request concurrency and autoscaling behavior.",
                "Inspect recent workload changes and expensive execution paths.",
                "Check whether background jobs overlap with peak traffic.",
            ],
            "mitigation": [
                "Scale out or shift traffic away from the affected fleet.",
                "Throttle or disable expensive non-critical work.",
            ],
        },
    },
    "traffic_spike": {
        "summary": "A sharp increase in request volume drove latency higher and stressed the serving path.",
        "symptoms": [
            "Request rate increased rapidly over a short period.",
            "Latency climbed with incoming demand.",
            "Error rate rose only after sustained traffic pressure.",
        ],
        "recent_changes": [
            "Check marketing events, client retries, or bot traffic changes.",
            "Review whether caching or rate-limiting behaved as expected.",
        ],
        "resolution": [
            "Scale capacity and apply traffic-shaping or rate limits if needed.",
            "Confirm caches and edge protections are absorbing expected burst traffic.",
        ],
        "lessons": [
            "Correlate request spikes with upstream events or campaigns.",
            "Validate that scaling and cache warmup policies cover burst scenarios.",
        ],
        "runbook": {
            "when_to_use": "Use this runbook when request rate rises faster than normal baselines and latency follows demand.",
            "investigation_steps": [
                "Compare current request volume against historical baselines.",
                "Check whether the spike is localized by route, region, or caller.",
                "Review autoscaling behavior and cache hit rates.",
                "Look for retry storms or bot-driven traffic amplification.",
            ],
            "mitigation": [
                "Scale out capacity or shift traffic.",
                "Enable rate limiting or protective controls if traffic is abusive.",
            ],
        },
    },
    "normal": {
        "summary": "Observed telemetry remained within expected operating bounds and did not match a strong incident signature.",
        "symptoms": [
            "Latency and error rates stayed near baseline.",
            "Resource utilization remained within expected ranges.",
            "No single failure pattern dominated the telemetry.",
        ],
        "recent_changes": [
            "Review alert thresholds for noise or false positives.",
            "Compare the event against historical non-incident behavior.",
        ],
        "resolution": [
            "No major intervention required.",
            "Capture the event as a baseline reference for future comparisons.",
        ],
        "lessons": [
            "Refine alert thresholds to reduce noise.",
            "Preserve representative non-incident examples for retrieval and comparison.",
        ],
        "runbook": {
            "when_to_use": "Use this runbook when telemetry looks noisy but does not support a clear incident pattern.",
            "investigation_steps": [
                "Validate that the alert is not a transient blip or threshold issue.",
                "Compare metrics against historical healthy baselines.",
                "Check whether a downstream dependency or release event is actually involved.",
                "Document why the event was classified as normal for future reference.",
            ],
            "mitigation": [
                "Monitor the system without applying disruptive remediation.",
                "Tune thresholds or dashboards if this pattern creates repeated false positives.",
            ],
        },
    },
}


@dataclass(frozen=True)
class KnowledgeBaseGeneratorConfig:
    input_path: str = "data/raw/incidents_sequence_raw.csv"
    output_dir: str = "data/knowledge_base/generated"
    max_postmortems: int = 6
    random_seed: int = 42


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SETTINGS.project_root / path


def _humanize_root_cause(label: str) -> str:
    return label.replace("_", " ").title()


def _incident_doc_id(raw_id: Any) -> str:
    if isinstance(raw_id, (int, np.integer)):
        return f"INC-{int(raw_id):04d}"
    raw = str(raw_id)
    if raw.upper().startswith("INC-"):
        return raw.upper()
    digits = "".join(ch for ch in raw if ch.isdigit())
    return f"INC-{digits.zfill(4)}" if digits else f"INC-{raw.upper()}"


def _metric_summary_line(name: str, value: float) -> str:
    if name in {"avg_cpu_usage", "error_rate"}:
        return f"{name}: {value:.2f}"
    if name in {"latency", "dependency_latency", "request_rate"}:
        return f"{name}: {value:.1f}"
    return f"{name}: {value:.2f}"


def _score_severity(row: pd.Series) -> float:
    return float(
        row.get("avg_cpu_usage", 0) * 0.2
        + row.get("latency", 0) * 0.03
        + row.get("request_rate", 0) * 0.01
        + row.get("error_rate", 0) * 40
        + row.get("mem_growth", 0) * 30
        + row.get("dependency_latency", 0) * 0.02
        + row.get("oom_log_count", 0) * 6
        + row.get("timeout_log_count", 0) * 4
    )


def aggregate_incidents(df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
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
    available_metrics = [column for column in metric_columns if column in df.columns]

    if "incident_id" in df.columns and "timestep" in df.columns:
        sorted_df = df.sort_values(["incident_id", "timestep"]).copy()
        grouped = sorted_df.groupby("incident_id", sort=True)
        records: list[dict[str, Any]] = []
        for incident_id, group in grouped:
            label = str(group["root_cause_label"].iloc[0])
            record: dict[str, Any] = {
                "incident_id": incident_id,
                "root_cause_label": label,
                "timesteps": int(len(group)),
            }
            for column in available_metrics:
                record[column] = float(group[column].mean())
                record[f"{column}_max"] = float(group[column].max())
                record[f"{column}_start"] = float(group[column].iloc[0])
                record[f"{column}_end"] = float(group[column].iloc[-1])
            records.append(record)
        return pd.DataFrame.from_records(records)

    flat_df = df.copy().reset_index(drop=True)
    if "incident_id" not in flat_df.columns:
        flat_df["incident_id"] = np.arange(1, len(flat_df) + 1)
    flat_df["timesteps"] = 1
    for column in available_metrics:
        flat_df[f"{column}_max"] = flat_df[column].astype(float)
        flat_df[f"{column}_start"] = flat_df[column].astype(float)
        flat_df[f"{column}_end"] = flat_df[column].astype(float)
    return flat_df


def build_incident_markdown(row: pd.Series) -> str:
    label = str(row["root_cause_label"])
    template = ROOT_CAUSE_TEMPLATES[label]
    incident_id = _incident_doc_id(row["incident_id"])
    metric_lines = [
        _metric_summary_line("avg_cpu_usage", float(row.get("avg_cpu_usage", 0.0))),
        _metric_summary_line("latency", float(row.get("latency", 0.0))),
        _metric_summary_line("request_rate", float(row.get("request_rate", 0.0))),
        _metric_summary_line("error_rate", float(row.get("error_rate", 0.0))),
    ]

    observations = [
        f"Observed {int(row.get('timesteps', 1))} telemetry step(s) for this incident.",
        f"Average dependency latency was {float(row.get('dependency_latency', 0.0)):.1f}.",
        f"Maximum timeout log count reached {int(round(float(row.get('timeout_log_count_max', row.get('timeout_log_count', 0.0)))))}.",
    ]

    lines = [
        f"# Incident {incident_id}",
        "",
        "## Summary",
        template["summary"],
        "",
        "## Symptoms",
        *[f"- {item}" for item in template["symptoms"]],
        "",
        "## Recent Changes",
        *[f"- {item}" for item in template["recent_changes"]],
        "",
        "## Observations",
        *[f"- {item}" for item in observations],
        "",
        "## Key Metrics",
        *[f"- {item}" for item in metric_lines],
        "",
        "## Root Cause",
        label,
        "",
        "## Resolution",
        *[f"- {item}" for item in template["resolution"]],
        "",
        "## Lessons Learned",
        *[f"- {item}" for item in template["lessons"]],
        "",
    ]
    return "\n".join(lines)


def build_runbook_markdown(root_cause: str) -> str:
    template = ROOT_CAUSE_TEMPLATES[root_cause]
    runbook = template["runbook"]
    lines = [
        f"# Runbook: {_humanize_root_cause(root_cause)}",
        "",
        "## When To Use",
        runbook["when_to_use"],
        "",
        "## Investigation Steps",
        *[f"1. {step}" if idx == 0 else f"{idx + 1}. {step}" for idx, step in enumerate(runbook["investigation_steps"])],
        "",
        "## Immediate Mitigation",
        *[f"- {item}" for item in runbook["mitigation"]],
        "",
        "## Follow-Up Actions",
        *[f"- {item}" for item in template["lessons"]],
        "",
    ]
    return "\n".join(lines)


def build_postmortem_markdown(row: pd.Series) -> str:
    label = str(row["root_cause_label"])
    template = ROOT_CAUSE_TEMPLATES[label]
    incident_id = _incident_doc_id(row["incident_id"])
    severity_score = _score_severity(row)
    lines = [
        f"# Postmortem: {incident_id}",
        "",
        "## Incident Overview",
        template["summary"],
        "",
        "## Customer Impact",
        "- Elevated customer-facing latency or error rates were observed during the incident window.",
        "- The exact blast radius depends on the service path affected by this root-cause pattern.",
        "",
        "## Technical Summary",
        f"This incident was associated with the `{label}` pattern and a derived severity score of {severity_score:.1f}.",
        f"Average latency was {float(row.get('latency', 0.0)):.1f}, while average error rate was {float(row.get('error_rate', 0.0)):.2f}.",
        "",
        "## Root Cause",
        _humanize_root_cause(label),
        "",
        "## Contributing Factors",
        *[f"- {item}" for item in template["recent_changes"]],
        "",
        "## Resolution",
        *[f"- {item}" for item in template["resolution"]],
        "",
        "## Preventive Actions",
        *[f"- {item}" for item in template["lessons"]],
        "",
    ]
    return "\n".join(lines)


def generate_knowledge_base(cfg: KnowledgeBaseGeneratorConfig) -> dict[str, Any]:
    input_path = _resolve_project_path(cfg.input_path)
    output_dir = _resolve_project_path(cfg.output_dir)
    incidents_dir = output_dir / "incidents"
    runbooks_dir = output_dir / "runbooks"
    postmortems_dir = output_dir / "postmortems"
    incidents_dir.mkdir(parents=True, exist_ok=True)
    runbooks_dir.mkdir(parents=True, exist_ok=True)
    postmortems_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_path)
    incidents_df = aggregate_incidents(raw_df)

    incident_paths: list[Path] = []
    for _, row in incidents_df.iterrows():
        incident_id = _incident_doc_id(row["incident_id"])
        incident_path = incidents_dir / f"{incident_id}.md"
        incident_path.write_text(build_incident_markdown(row), encoding="utf-8")
        incident_paths.append(incident_path)

    runbook_paths: list[Path] = []
    for label in sorted(incidents_df["root_cause_label"].astype(str).unique()):
        runbook_path = runbooks_dir / f"{label.replace('_', '-')}.md"
        runbook_path.write_text(build_runbook_markdown(label), encoding="utf-8")
        runbook_paths.append(runbook_path)

    ranked = incidents_df.copy()
    ranked["severity_score"] = ranked.apply(_score_severity, axis=1)
    selected_postmortems = (
        ranked.sort_values(["root_cause_label", "severity_score"], ascending=[True, False])
        .groupby("root_cause_label", sort=True)
        .head(1)
        .sort_values("severity_score", ascending=False)
        .head(max(cfg.max_postmortems, 0))
    )

    postmortem_paths: list[Path] = []
    for _, row in selected_postmortems.iterrows():
        incident_id = _incident_doc_id(row["incident_id"])
        postmortem_path = postmortems_dir / f"{incident_id}-postmortem.md"
        postmortem_path.write_text(build_postmortem_markdown(row), encoding="utf-8")
        postmortem_paths.append(postmortem_path)

    return {
        "input_path": input_path,
        "output_dir": output_dir,
        "incidents_dir": incidents_dir,
        "runbooks_dir": runbooks_dir,
        "postmortems_dir": postmortems_dir,
        "incident_paths": incident_paths,
        "runbook_paths": runbook_paths,
        "postmortem_paths": postmortem_paths,
        "n_incident_docs": len(incident_paths),
        "n_runbooks": len(runbook_paths),
        "n_postmortems": len(postmortem_paths),
    }
