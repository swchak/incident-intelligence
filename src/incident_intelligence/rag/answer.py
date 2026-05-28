from __future__ import annotations

from typing import Any


ROOT_CAUSE_LABELS = [
    "memory_leak",
    "bad_deployment",
    "external_dependency_failure",
    "cpu_saturation",
    "traffic_spike",
    "normal",
]

ROOT_CAUSE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "memory_leak": ("memory leak", "oom", "memory growth", "heap", "garbage collection"),
    "bad_deployment": ("deployment", "rollback", "release", "version rollout", "config change"),
    "external_dependency_failure": (
        "external dependency",
        "dependency failure",
        "upstream",
        "provider outage",
        "third-party",
    ),
    "cpu_saturation": ("cpu saturation", "cpu spike", "high cpu", "compute bound", "thread pool"),
    "traffic_spike": ("traffic spike", "request surge", "high request rate", "flash crowd"),
    "normal": ("normal", "steady state", "baseline"),
}

NEXT_STEPS: dict[str, list[str]] = {
    "memory_leak": [
        "Check heap growth, OOM logs, and recent library or image-processing changes.",
        "Restart or reschedule affected pods while the root cause is isolated.",
        "Review canary and memory alert thresholds before the next rollout.",
    ],
    "bad_deployment": [
        "Inspect the most recent deployment, config diff, and rollout timeline.",
        "Compare canary metrics and rollback status against the error spike window.",
        "If needed, roll back the latest release and validate recovery.",
    ],
    "external_dependency_failure": [
        "Review upstream latency, timeout, and retry behavior for the dependency.",
        "Check provider status pages or recent dependency incident notices.",
        "Enable fallback behavior or rate-limit calls until the dependency stabilizes.",
    ],
    "cpu_saturation": [
        "Inspect CPU utilization, hot endpoints, and worker concurrency settings.",
        "Check whether a recent workload or batch job increased compute pressure.",
        "Scale affected services or shed non-critical traffic while investigating.",
    ],
    "traffic_spike": [
        "Review request rate anomalies, top endpoints, and client distribution.",
        "Validate autoscaling behavior and cache hit rates during the spike.",
        "Consider throttling or traffic-shaping if saturation risk remains high.",
    ],
    "normal": [
        "Compare the retrieval evidence with recent alerts to confirm the signal.",
        "Continue monitoring and gather more evidence before escalating.",
    ],
}


def build_grounded_context(results: list[dict[str, Any]], max_snippets: int = 4) -> str:
    if not results:
        return "No supporting knowledge-base documents were retrieved."

    snippets: list[str] = []
    for index, item in enumerate(results[:max_snippets], start=1):
        metadata = item.get("metadata") or {}
        title = metadata.get("title") or metadata.get("source_path") or f"Document {index}"
        source = metadata.get("source_path", "unknown")
        body = str(item.get("document", "")).strip().replace("\n", " ")
        preview = body[:260] + ("..." if len(body) > 260 else "")
        snippets.append(f"{index}. {title} ({source})\n{preview}")
    return "\n\n".join(snippets)


def _infer_root_cause(query: str, results: list[dict[str, Any]]) -> str:
    combined_text = " ".join(
        [
            query.lower(),
            *[
                " ".join(
                    str(value).lower()
                    for value in (
                        item.get("document", ""),
                        (item.get("metadata") or {}).get("title", ""),
                        (item.get("metadata") or {}).get("source_path", ""),
                    )
                )
                for item in results[:4]
            ],
        ]
    )
    scores: dict[str, int] = {}
    for label, keywords in ROOT_CAUSE_KEYWORDS.items():
        scores[label] = sum(1 for keyword in keywords if keyword in combined_text)
    best_label = max(scores, key=scores.get)
    return best_label if scores[best_label] > 0 else "normal"


def _confidence_from_matches(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.25
    best_distance = results[0].get("distance")
    if not isinstance(best_distance, (int, float)):
        return 0.5
    confidence = 1.0 - float(best_distance)
    return max(0.1, min(confidence, 0.99))


def _evidence_lines(results: list[dict[str, Any]], max_evidence: int) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for match in results[:max_evidence]:
        metadata = match.get("metadata") or {}
        body = str(match.get("document", "")).strip().replace("\n", " ")
        snippet = body[:220] + ("..." if len(body) > 220 else "")
        evidence.append(
            {
                "source_path": metadata.get("source_path", "unknown"),
                "doc_type": metadata.get("doc_type", "document"),
                "title": metadata.get("title") or metadata.get("source_path") or "Untitled document",
                "distance": match.get("distance"),
                "snippet": snippet,
            }
        )
    return evidence


def build_template_answer(
    query: str,
    results: list[dict[str, Any]],
    *,
    max_evidence: int = 3,
    mode: str = "template",
) -> dict[str, Any]:
    root_cause = _infer_root_cause(query, results)
    confidence = _confidence_from_matches(results)
    evidence = _evidence_lines(results, max_evidence=max_evidence)
    next_steps = NEXT_STEPS.get(root_cause, NEXT_STEPS["normal"])
    summary = (
        f"The retrieval layer suggests {root_cause} with confidence {confidence:.2f}. "
        f"Retrieved knowledge-base matches show similar patterns in prior incidents and runbooks."
    )
    return {
        "answer_mode": mode,
        "predicted_root_cause": root_cause,
        "confidence": round(confidence, 4),
        "diagnostic_summary": summary,
        "retrieved_evidence": evidence,
        "next_steps": next_steps,
    }
