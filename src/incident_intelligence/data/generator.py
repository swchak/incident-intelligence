import numpy as np
import pandas as pd
import random
from scipy.special import expit as sigmoid

from incident_intelligence.settings import load_class_config

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


# def generate_incident(root_cause, class_config):
#     # ... same as your current function ...
#     # (unchanged except class_config passed in)
#     pass


def generate_incident(root_cause, class_config):
    """Generate a single incident with noise."""
        
    # Baseline metrics
    metrics = {
        "request_rate": np.random.normal(300, 50),
        "mem_growth": np.random.normal(0.2, 0.1),
        "error_rate": np.random.normal(0.5, 0.2),
        "upstream_error": np.random.normal(0.3, 0.2),
        "dependency_latency": np.random.normal(150, 40),
        "latency": np.random.normal(200, 50),
        "avg_cpu": np.random.normal(40, 10),
    }

    # Root cause-specific patterns

    config = class_config[root_cause]

    for metric, mixture in config.items():
        if metric in metrics:
            metrics[metric] = apply_mixture(
                metrics[metric],
                mixture
            )

    metrics["avg_cpu"] += (0.05 * metrics["request_rate"] + np.random.normal(5, 3))
    metrics["avg_cpu"] = np.clip(metrics["avg_cpu"], 0, 100)
    metrics["mem_growth"] += 0.005 * metrics["avg_cpu"] 

    metrics["dependency_latency"] += 0.1 * max(metrics["request_rate"] - 300, 0)

    metrics["latency"] += (
        0.5 * metrics["avg_cpu"] + 
        0.3 * metrics["dependency_latency"] + 
        10 * metrics["mem_growth"] + 
        np.random.normal(50, 15))
    metrics["latency"] = max(metrics["latency"], 50)

    bias = -4.0
    system_stress_score = (
        bias
        +0.02 * metrics["avg_cpu"]
        +1.0 * metrics["mem_growth"] 
        +0.0015 * metrics["dependency_latency"]
        +0.05 * metrics["upstream_error"]
        +0.002 * metrics["request_rate"]
        +0.01 * metrics["latency"]
    )
    system_stress_score += 2.0 * metrics["error_rate"]
    metrics["error_rate"] += sigmoid(system_stress_score)

    # OOM depends on memory growth
    oom_logs = np.random.poisson(max(metrics["mem_growth"] * 2, 0.5))

    # Timeouts depend on latency
    timeout_logs = np.random.poisson(max(metrics["dependency_latency"] / 100, 1))

    metrics["request_rate"] = max(metrics["request_rate"], 0)

    metrics["dependency_latency"] = max(metrics["dependency_latency"], 1)
    metrics["mem_growth"] = max(metrics["mem_growth"], 0)
    metrics["error_rate"] = max(metrics["error_rate"], 0)

    return {
        "avg_cpu_usage": metrics["avg_cpu"],
        "mem_growth": metrics["mem_growth"] ,
        "oom_log_count": oom_logs,
        "request_rate": metrics["request_rate"],
        "error_rate": metrics["error_rate"],
        "latency": metrics["latency"],
        "upstream_error_rate": metrics["upstream_error"],
        "dependency_latency": metrics["dependency_latency"],
        "timeout_log_count": timeout_logs,
        "root_cause_label": root_cause
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