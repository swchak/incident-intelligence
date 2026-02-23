import numpy as np
import pandas as pd
import random

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
    "normal": 0.30,
    "traffic_spike": 0.14,
    "memory_leak": 0.14,
    "cpu_saturation": 0.14,
    "bad_deployment": 0.14,
    "external_dependency_failure": 0.14
}

CLASS_CONFIG = {
    "traffic_spike": {
        "request_rate": [(0.2, (400, 50)),
                         (0.4, (300, 50)),
                         (0.2, (200, 50)),
                         (0.1, (100, 50)),
                         (0.1, (80, 20)),
                         ],
        "avg_cpu": [
                    (0.1, (30, 5)),
                    (0.4, (25, 5)),
                    (0.2, (20, 5)),
                    (0.2, (15, 5)),
                    (0.1, (10, 5))
                ],
        "latency": [
                    (0.3, (150, 50)),
                    (0.5, (125, 50)),
                    (0.1, (100, 50)),
                    (0.1, (50, 20))
            ],
        "mem_growth": [
                        (0.2, (1.0, 0.2)),
                        (0.7, (0.5, 0.2))
        ],
        "error_rate": [
              (0.2, (0.5, 0.2))
        ],
        "upstream_error": [
                        (0.3, (1.0, 0.5)),
                        (0.5, (0.3, 0.2))
        ],
        "dependency_latency": [
                               (0.1, (250, 50)),
                                (0.3, (150, 50)),
                                (0.3, (100, 50)),
                                (0.3, (50, 10))
        ]
    },
    "memory_leak": {
        # add request_rate noise
        "request_rate": [
                         (0.1, (400, 50)),
                         (0.1, (300, 50)),
                         (0.1, (200, 50)),
                         (0.1, (100, 50)),
                         (0.6, (60, 20))],
        # add avg_cpu noise
        "avg_cpu": [
                    (0.1, (30, 5)),
                    (0.1, (25, 5)),
                    (0.3, (20, 5)),
                    (0.4, (15, 5)),
                    (0.1, (10, 5)),
        ],
        "latency": [
                    (0.2, (120, 20)),
                    (0.3, (100, 20)),
                    (0.4, (80, 20)),
                    (0.1, (40, 10))
                ],
        "mem_growth": [
                        (0.4, (1.5, 0.4)),
                        (0.3, (1.2, 0.3)),
                        (0.2, (1.0, 0.2)),
                        (0.1, (0.8, 0.3)),
                    ],
        "error_rate": [
                        (0.2, (0.5, 0.2))
        ],
        "upstream_error": [
                            # (0.1, (2.0, 0.5)),
                            # (0.1, (1.0, 0.5)),
                            # (0.1, (0.3, 0.2)),
        ],
        "dependency_latency": [
                                # (0.1, (150, 50)),
                                # (0.1, (100, 50)),
                                # (0.1, (50, 10))
        ]
    },
    "cpu_saturation": {
        # add request_rate noise
        "request_rate": [(0.2, (400, 50)),
                         (0.2, (300, 50)),
                         (0.5, (200, 50)),
                         (0.1, (60, 20))],
        "avg_cpu": 
                    [(0.4, (35, 5)),
                    (0.3, (30, 5)),
                    (0.2, (25, 5)),
                    (0.1, (10, 5))],
        "latency": [
                    (0.1, (150, 30)),
                    (0.6, (120, 30)),
                    (0.2, (80, 20)),
                    (0.1, (20, 5))
                    ],
        "mem_growth": [
                        (0.3, (1.0, 0.2)),
                        (0.5, (0.5, 0.2))
        ],
        "error_rate": [
                        (0.05, (2.0, 0.5)),
                        (0.55, (1.5, 0.5)),
                        (0.3, (1.0, 0.2)),
                    ],
        "upstream_error": [
                            (0.3, (1.0, 0.5)),
                            (0.4, (0.3, 0.2))
        ],
        "dependency_latency": [
                                (0.3, (100, 50)),
                                (0.4, (50, 10))
        ]
    },
    "bad_deployment": {
       # add request_rate noise
       "request_rate": [
                         (0.3, (100, 20)),
                         (0.4, (80, 20)),
                         (0.3, (30, 10))],
                         
        "avg_cpu": [
                    (0.1, (20, 5)),
                    (0.1, (15, 5)),
                    (0.3, (10, 5))
        ],
        "latency": [
                    (0.1, (100, 20)),
                    (0.2, (80, 20)),
                    (0.2, (60, 10))
                ],
        "mem_growth": [
                        (0.1, (0.5, 0.2))
        ],
        "error_rate": [
                        (0.2, (3.0, 0.5)),
                        (0.2, (2.0, 0.5)),
                        (0.4, (1.0, 0.5)),
                        (0.2, (0.8, 0.3))
                    ],
        "upstream_error": [
                            (0.1, (1.0, 0.5)),
                            (0.2, (0.3, 0.2))
        ],
        "dependency_latency": [
                                (0.1, (100, 50)),
                                (0.2, (50, 10))
        ]
    },
    "external_dependency_failure": {
        # add request_rate noise
        "request_rate": [
                        (0.1, (400, 50)),
                        (0.2, (300, 50)),
                        (0.4, (100, 50)),
                        (0.2, (80, 20))
        ],
        "avg_cpu": [
                    (0.2, (20, 5)),
                    (0.2, (15, 5)),
                    (0.3, (10, 5)),
        ],
        "latency": [
                    (0.1, (125, 50)),
                    (0.3, (100, 20)),
                    (0.3, (80, 20)),
                    (0.3, (60, 10))
        ],
        "mem_growth": [
                        # (0.3, (1.0, 0.2)),
                        # (0.4, (0.5, 0.2))
        ],
        "error_rate": [
                        # (0.2, (0.8, 0.2))
        ],
        "upstream_error": [
                            (0.3, (4.0, 0.5)),
                            (0.3, (3.0, 0.5)),
                            (0.2, (2.0, 0.5)),
                            (0.1, (1.0, 0.5)),
                            (0.1, (0.3, 0.2))
        ],
        "dependency_latency": [     
                                (0.4, (300, 50)),
                                (0.3, (250, 50)),
                                (0.25, (150, 50)),
                                (0.05, (50, 10))
        ]
    },
    "normal": {}
}


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
    # avg_cpu = np.random.normal(40, 10)
    mem_growth = np.random.normal(0.2, 0.1)
    # req_rate = np.random.normal(300, 50)
    error_rate = np.random.normal(0.5, 0.2)
    # latency = np.random.normal(200, 50)

    req_rate = np.random.normal(300, 50)

    avg_cpu = 0.1 * req_rate + np.random.normal(10, 5)
    latency = 0.5 * avg_cpu + np.random.normal(100, 20)

    upstream_error = np.random.normal(0.3, 0.2)
    dependency_latency = np.random.normal(150, 40)


    # Root cause-specific patterns
    config = CLASS_CONFIG[root_cause]
    if root_cause == "traffic_spike":
        req_rate = apply_mixture(req_rate, config["request_rate"])
        avg_cpu = apply_mixture(avg_cpu, config["avg_cpu"])
        latency = apply_mixture(latency, config["latency"])

        mem_growth = apply_mixture(mem_growth, config["mem_growth"])
        error_rate = apply_mixture(error_rate, config["error_rate"])
        dependency_latency = apply_mixture(dependency_latency, config["dependency_latency"])
        upstream_error = apply_mixture(upstream_error, config["upstream_error"])
        # if np.random.rand() < 0.3:
        #     oom_logs += np.random.poisson(1)

    elif root_cause == "memory_leak":
        latency = apply_mixture(latency, config["latency"])
        mem_growth = apply_mixture(mem_growth, config["mem_growth"])

        # noise 
        req_rate = apply_mixture(req_rate, config["request_rate"])
        avg_cpu = apply_mixture(avg_cpu, config["avg_cpu"])
        error_rate = apply_mixture(error_rate, config["error_rate"])
        # dependency_latency = apply_mixture(dependency_latency, config["dependency_latency"])
        # upstream_error = apply_mixture(upstream_error, config["upstream_error"])

    elif root_cause == "cpu_saturation":
        avg_cpu = apply_mixture(avg_cpu, config["avg_cpu"])
        latency = apply_mixture(latency, config["latency"])
        error_rate = apply_mixture(error_rate, config["error_rate"])

        # noise
        req_rate = apply_mixture(req_rate, config["request_rate"])
        mem_growth = apply_mixture(mem_growth, config["mem_growth"])
        dependency_latency = apply_mixture(dependency_latency, config["dependency_latency"])
        upstream_error = apply_mixture(upstream_error, config["upstream_error"])


    elif root_cause == "bad_deployment":
        error_rate = apply_mixture(error_rate, config["error_rate"])
        # timeout_logs += np.random.poisson(10)

        # noise
        req_rate = apply_mixture(req_rate, config["request_rate"])
        avg_cpu = apply_mixture(avg_cpu, config["avg_cpu"])
        latency = apply_mixture(latency, config["latency"])
        dependency_latency = apply_mixture(dependency_latency, config["dependency_latency"])
        upstream_error = apply_mixture(upstream_error, config["upstream_error"])

    elif root_cause == "external_dependency_failure":
        dependency_latency = apply_mixture(dependency_latency, config["dependency_latency"])
        upstream_error = apply_mixture(upstream_error, config["upstream_error"])
        # timeout_logs += np.random.poisson(7)

        # noise
        req_rate = apply_mixture(req_rate, config["request_rate"])
        avg_cpu = apply_mixture(avg_cpu, config["avg_cpu"])
        # mem_growth = apply_mixture(mem_growth, config["mem_growth"])
        # error_rate = apply_mixture(error_rate, config["error_rate"])
  
    # Add small random noise for realism
    # time_since_deploy += np.random.normal(0, 5)
    # time_since_deploy = max(time_since_deploy, 0)


    # OOM depends on memory growth
    oom_logs = np.random.poisson(max(mem_growth * 2, 0.5))

    # Timeouts depend on latency
    timeout_logs = np.random.poisson(max(dependency_latency / 100, 1))

    # Add noise and clip values
    avg_cpu = np.clip(avg_cpu, 0, 100)
    error_rate = max(error_rate, 0)
    latency = max(0.5 * avg_cpu + np.random.normal(100, 20), 50)

    return {
        "avg_cpu_usage": avg_cpu,
        "memory_growth_rate": mem_growth,
        "oom_log_count": oom_logs,
        "request_rate": req_rate,
        "error_rate": error_rate,
        "latency": latency,
        "upstream_error_rate": upstream_error,
        "dependency_latency": dependency_latency,
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
    df = generate_dataset(N_SAMPLES)
    df.to_csv("./data/incident_root_cause_data.csv", index=False)
    print(df.head())
    print("\nClass distribution:\n", df["root_cause_label"].value_counts(normalize=True))
