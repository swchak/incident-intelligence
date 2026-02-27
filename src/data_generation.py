import numpy as np
import pandas as pd
import random
from scipy.special import expit as sigmoid

np.random.seed(42)
random.seed(42)

N_SAMPLES = 4000

ROOT_CAUSES = [
    "traffic_spike",
    "memory_leak",
    "cpu_saturation",
    "bad_deployment",
    "external_dependency_failure",
    "normal"
]

ROOT_CAUSE_PROBS = {
    "bad_deployment": 0.2,
    "external_dependency_failure": 0.2,
    "traffic_spike": 0.15,
    "memory_leak": 0.15,
    "cpu_saturation": 0.15,
    "normal": 0.15,
}

CLASS_CONFIG = {
    "traffic_spike": {
        "request_rate": [
            (0.4, (400, 100)),
            (0.2, (300, 100)),
            (0.2, (200, 100)),
            (0.2, (100, 50))
        ],
        "avg_cpu": [
            (0.2, (30, 15)),
            (0.3, (20, 10)),
            (0.5, (10, 5))
        ],
        "latency": [
            (0.4, (150, 100)),
            (0.4, (100, 50)),
            (0.2, (50, 20))
        ],
        "mem_growth": [
            (0.2, (1.2, 0.5)),
            (0.2, (1.0, 0.5)),
            (0.4, (0.8, 0.4)),
            (0.2, (0.5, 0.2)),
        ],
        "dependency_latency": [
            (0.1, (300, 100)),
            (0.2, (200, 100)),
            (0.3, (100, 50)),
            (0.4, (50, 20))
        ],
        "error_rate": [
            (0.1, (2.0, 1.0)),
            (0.1, (1.5, 0.5)),
            (0.3, (0.8, 0.4)),
            (0.5, (0.5, 0.2))
        ],
        "upstream_error": [
            (0.1, (2.0, 1.0)),
            (0.2, (1.5, 0.5)),
            (0.3, (1.0, 0.5)),
            (0.4, (0.5, 0.2)),
        ],
    },
    "memory_leak": {
        "mem_growth": [
            (0.3, (1.5, 0.5)),
            (0.4, (1.2, 0.5)),
            (0.2, (1.0, 0.5)),
            (0.1, (0.8, 0.5))
        ],
        "latency": [
            (0.1, (150, 100)),
            (0.2, (120, 60)),
            (0.4, (80, 40)),
            (0.3, (50, 20))
        ],
        "dependency_latency": [
            (0.1, (300, 100)),
            (0.1, (200, 100)),
            (0.3, (100, 50)),
            (0.5, (50, 20))
        ],
        "request_rate": [
            (0.1, (300, 100)),
            (0.1, (200, 100)),
            (0.1, (100, 50)),
            (0.3, (50, 20)),
            (0.2, (-50, 20)),
            (0.2, (-20, 5))
        ],
        "avg_cpu": [
            (0.2, (30, 15)),
            (0.2, (20, 10)),
            (0.6, (10, 5))
        ],
        "error_rate": [
            (0.1, (1.0, 0.3)),
            (0.3, (0.8, 0.4)),
            (0.6, (0.5, 0.2))
        ],
        "upstream_error": [
            (0.1, (2.0, 1.0)),
            (0.1, (1.5, 0.5)),
            (0.1, (1.0, 0.5)),
            (0.7, (0.5, 0.2)),
        ],

    },
    "cpu_saturation": {
        "avg_cpu": [
            (0.5, (35, 20)),
            (0.3, (25, 10)),
            (0.2, (15, 5))
        ],
        "error_rate": [
            (0.4, (1.5, 1.0)),
            (0.4, (1.0, 0.5)),
            (0.2, (0.5, 0.2))
        ],
        "latency": [
            (0.1, (150, 100)),
            (0.3, (120, 60)),
            (0.3, (100, 50)),
            (0.3, (80, 40))
        ],
        "dependency_latency": [
            (0.1, (300, 100)),
            (0.1, (200, 100)),
            (0.3, (100, 50)),
            (0.5, (50, 20))
        ],
        "request_rate": [
            (0.1, (300, 100)),
            (0.1, (200, 100)),
            (0.1, (100, 50)),
            (0.4, (-50, 20)),
            (0.3, (-20, 5))
        ],
        "mem_growth": [
            (0.3, (1.2, 0.5)),
            (0.2, (1.0, 0.5)),
            (0.4, (0.8, 0.4)),
            (0.1, (0.5, 0.2))
        ],
        "upstream_error": [
            (0.1, (2.0, 1.0)),
            (0.1, (1.5, 0.5)),
            (0.2, (1.0, 0.5)),
            (0.6, (0.5, 0.2)),
        ],
    },
    "bad_deployment": {
        "error_rate": [
            (0.3, (3.0, 1.5)),
            (0.4, (2.0, 1.0)),
            (0.3, (1.5, 1.0))
        ],
        "latency": [
            (0.2, (120, 60)),
            (0.3, (100, 50)),
            (0.5, (80, 40))
        ],
        "dependency_latency": [
            (0.1, (300, 100)),
            (0.1, (200, 100)),
            (0.4, (100, 50)),
            (0.4, (50, 20))
        ],
        "avg_cpu": [
            (0.1, (30, 15)),
            (0.2, (20, 10)),
            (0.7, (10, 5))
        ],
        "request_rate": [
            (0.1, (300, 150)),
            (0.1, (100, 50)),
            (0.4, (-50, 20)),
            (0.4, (-20, 5))
        ],
        # "mem_growth": [
        #     (0.4, (1.0, 0.5)),
        #     (0.4, (0.8, 0.4)),
        #     (0.2, (0.5, 0.2)),
        # ],
        "upstream_error": [
            (0.1, (2.0, 1.0)),
            (0.1, (1.5, 0.5)),
            (0.2, (1.0, 0.5)),
            (0.6, (0.5, 0.2)),
        ],
        
    },
    "external_dependency_failure": {
        "dependency_latency": [
            (0.4, (300, 50)),
            (0.3, (200, 50)),
            (0.3, (100, 50))
        ],
        "upstream_error": [
            (0.3, (3.0, 1.5)),
            (0.4, (2.0, 1.0)),
            (0.3, (1.0, 0.5))
        ],
        "avg_cpu": [
            (0.1, (30, 15)),
            (0.2, (20, 10)),
            (0.7, (10, 5))
        ],
        "request_rate": [
            (0.1, (300, 150)),
            (0.1, (100, 50)),
            (0.5, (-50, 20)),
            (0.3, (-20, 5))
        ],
        # "mem_growth": [
        #     (0.4, (1.0, 0.5)),
        #     (0.4, (0.8, 0.4)),
        #     (0.2, (0.5, 0.2)),
        # ],
        "latency": [
            (0.1, (150, 100)),
            (0.3, (100, 50)),
            (0.6, (80, 40))
        ],
        "error_rate": [
            (0.1, (3.0, 2.0)),
            (0.2, (2.0, 1.0)),
            (0.7, (1.0, 0.5))
        ],
    },
    "normal": {}
}


def validate_configs():
    for root_cause, metric_mix in CLASS_CONFIG.items():
        for metric, mixture in metric_mix.items():
            total = sum(p for p, _ in mixture)
            if not np.isclose(total, 1.0):
                raise ValueError(f"config error: root_cause={root_cause} | metric={metric} | mixture probs sum to {total}")


def apply_mixture(value, config): 
    r = np.random.rand()
    cumulative = 0
    for prob, (mean, std) in config:
        cumulative += prob
        if r < cumulative:
            return value + np.random.normal(mean, std)
    return value


def generate_incident(root_cause):
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

    config = CLASS_CONFIG[root_cause]

    for metric, mixture in config.items():
        if metric in metrics:
            metrics[metric] = apply_mixture(
                metrics[metric],
                mixture
            )

    metrics["avg_cpu"] += (0.05 * metrics["request_rate"] + np.random.normal(5, 3))
    metrics["avg_cpu"] = np.clip(metrics["avg_cpu"], 0, 100)

    # mem_growth += 0.005 * avg_cpu
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


def generate_dataset(n_samples):
    data = []
    for _ in range(n_samples):
        root_cause = random.choices(
            list(ROOT_CAUSE_PROBS.keys()),
            weights=ROOT_CAUSE_PROBS.values(),
            k=1
        )[0]
        data.append(generate_incident(root_cause))
    return pd.DataFrame(data)


if __name__ == "__main__":
    validate_configs()
    df = generate_dataset(N_SAMPLES)
    df.to_csv("./data/incident_root_cause_data.csv", index=False)
    print(df.head())
    print("\nClass distribution:\n", df["root_cause_label"].value_counts(normalize=True))
