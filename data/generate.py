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


def generate_incident(root_cause):
    """Generate a single incident with noise."""
    
    # Baseline metrics
    avg_cpu = np.random.normal(40, 10)
    max_cpu = avg_cpu + np.random.normal(10, 5)
    avg_mem = np.random.normal(50, 10)
    mem_growth = np.random.normal(0.2, 0.1)
    req_rate = np.random.normal(300, 50)
    error_rate = np.random.normal(0.5, 0.3)
    latency = np.random.normal(200, 50)

    time_since_deploy = np.random.uniform(60, 1000)
    recent_deploy = 0

    upstream_error = np.random.normal(0.3, 0.2)
    dependency_latency = np.random.normal(150, 40)

    error_logs = np.random.poisson(5)
    timeout_logs = np.random.poisson(3)
    oom_logs = np.random.poisson(1)
    db_error_logs = np.random.poisson(2)

    # Root cause-specific patterns
    if root_cause == "traffic_spike":
        req_rate += np.random.normal(400, 100)
        avg_cpu += np.random.normal(20, 5)
        latency += np.random.normal(150, 50)

    elif root_cause == "memory_leak":
        avg_mem += np.random.normal(30, 5)
        mem_growth += np.random.normal(1.5, 0.3)
        oom_logs += np.random.poisson(5)
        latency += np.random.normal(80, 30)

    elif root_cause == "cpu_saturation":
        avg_cpu += np.random.normal(35, 5)
        max_cpu += np.random.normal(25, 5)
        error_rate += np.random.normal(1.5, 0.4)
        latency += np.random.normal(120, 40)

    elif root_cause == "bad_deployment":
        recent_deploy = 1
        time_since_deploy = np.random.uniform(0, 60)
        error_rate += np.random.normal(3.0, 0.6)
        timeout_logs += np.random.poisson(10)

    elif root_cause == "external_dependency_failure":
        upstream_error += np.random.normal(4.0, 1.0)
        dependency_latency += np.random.normal(300, 80)
        timeout_logs += np.random.poisson(7)

    # Add noise and clip values
    avg_cpu = np.clip(avg_cpu, 0, 100)
    max_cpu = np.clip(max_cpu, 0, 100)
    avg_mem = np.clip(avg_mem, 0, 100)
    error_rate = max(error_rate, 0)
    latency = max(latency, 50)

    return {
        "avg_cpu_usage": avg_cpu,
        "max_cpu_usage": max_cpu,
        "avg_memory_usage": avg_mem,
        "memory_growth_rate": mem_growth,
        "request_rate": req_rate,
        "error_rate": error_rate,
        "p95_latency": latency,
        "time_since_last_deploy": time_since_deploy,
        "deploy_happened_recently": recent_deploy,
        "upstream_error_rate": upstream_error,
        "dependency_latency": dependency_latency,
        "error_log_count": error_logs,
        "timeout_log_count": timeout_logs,
        "oom_log_count": oom_logs,
        "db_error_log_count": db_error_logs,
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
    df.to_csv("incident_root_cause_data.csv", index=False)
    print(df.head())
    print("\nClass distribution:\n", df["root_cause_label"].value_counts(normalize=True))
