from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


ROOT_CAUSES = [
    "memory_leak",
    "bad_deployment",
    "external_dependency_failure",
    "cpu_saturation",
    "traffic_spike",
    "normal",
]


@dataclass
class SequenceGeneratorConfig:
    n_incidents: int = 5000
    sequence_length: int = 20
    random_seed: int = 42


class IncidentSequenceGenerator:
    def __init__(self, config: SequenceGeneratorConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def generate(self) -> pd.DataFrame:
        rows: List[Dict] = []

        for incident_id in range(self.config.n_incidents):
            label = self._sample_label()
            incident_rows = self._generate_incident_sequence(
                incident_id=incident_id,
                label=label,
                sequence_length=self.config.sequence_length,
            )
            rows.extend(incident_rows)

        df = pd.DataFrame(rows)
        return df

    def _sample_label(self) -> str:
        # Replace with your class-config-driven probabilities later
        probs = np.array([0.18, 0.15, 0.17, 0.16, 0.14, 0.20])
        return self.rng.choice(ROOT_CAUSES, p=probs)

    def _generate_incident_sequence(
        self,
        incident_id: int,
        label: str,
        sequence_length: int,
    ) -> List[Dict]:
        baseline = self._make_baseline()
        t = np.arange(sequence_length)

        if label == "memory_leak":
            metrics = self._memory_leak_pattern(t, baseline)
        elif label == "bad_deployment":
            metrics = self._bad_deployment_pattern(t, baseline)
        elif label == "external_dependency_failure":
            metrics = self._dependency_failure_pattern(t, baseline)
        elif label == "cpu_saturation":
            metrics = self._cpu_saturation_pattern(t, baseline)
        elif label == "traffic_spike":
            metrics = self._traffic_spike_pattern(t, baseline)
        else:
            metrics = self._normal_pattern(t, baseline)

        rows: List[Dict] = []
        for i in range(sequence_length):
            rows.append(
                {
                    "incident_id": incident_id,
                    "timestep": int(i),
                    "root_cause_label": label,
                    "avg_cpu_usage": float(metrics["avg_cpu_usage"][i]),
                    "mem_growth": float(metrics["mem_growth"][i]),
                    "oom_log_count": int(metrics["oom_log_count"][i]),
                    "request_rate": float(metrics["request_rate"][i]),
                    "error_rate": float(metrics["error_rate"][i]),
                    "latency": float(metrics["latency"][i]),
                    "upstream_error_rate": float(metrics["upstream_error_rate"][i]),
                    "dependency_latency": float(metrics["dependency_latency"][i]),
                    "timeout_log_count": int(metrics["timeout_log_count"][i]),
                }
            )

        return rows

    def _make_baseline(self) -> Dict[str, float]:
        return {
            "avg_cpu_usage": self.rng.uniform(25, 55),
            "mem_growth": self.rng.uniform(0.0, 0.03),
            "request_rate": self.rng.uniform(80, 140),
            "error_rate": self.rng.uniform(0.0, 0.02),
            "latency": self.rng.uniform(80, 140),
            "upstream_error_rate": self.rng.uniform(0.0, 0.02),
            "dependency_latency": self.rng.uniform(40, 90),
        }

    def _noise(self, n: int, scale: float) -> np.ndarray:
        return self.rng.normal(loc=0.0, scale=scale, size=n)

    def _clip_nonnegative(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, a_min=0, a_max=None)

    def _memory_leak_pattern(self, t: np.ndarray, b: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(t)
        mem_growth = b["mem_growth"] + 0.02 * t + self._noise(n, 0.01)
        latency = b["latency"] + 2.5 * t + 20 * mem_growth + self._noise(n, 4.0)
        avg_cpu_usage = b["avg_cpu_usage"] + 0.4 * t + self._noise(n, 2.0)
        request_rate = b["request_rate"] + self._noise(n, 3.0)
        error_rate = b["error_rate"] + 0.003 * t + self._noise(n, 0.003)
        dependency_latency = b["dependency_latency"] + self._noise(n, 2.0)
        upstream_error_rate = b["upstream_error_rate"] + self._noise(n, 0.002)
        oom_log_count = (mem_growth > np.percentile(mem_growth, 80)).astype(int)
        timeout_log_count = (latency > np.percentile(latency, 75)).astype(int)

        return self._finalize(
            avg_cpu_usage=avg_cpu_usage,
            mem_growth=mem_growth,
            oom_log_count=oom_log_count,
            request_rate=request_rate,
            error_rate=error_rate,
            latency=latency,
            upstream_error_rate=upstream_error_rate,
            dependency_latency=dependency_latency,
            timeout_log_count=timeout_log_count,
        )

    def _bad_deployment_pattern(self, t: np.ndarray, b: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(t)
        change_point = max(2, n // 3)

        avg_cpu_usage = b["avg_cpu_usage"] + self._noise(n, 3.0)
        mem_growth = b["mem_growth"] + self._noise(n, 0.01)
        request_rate = b["request_rate"] + self._noise(n, 4.0)
        error_rate = b["error_rate"] + self._noise(n, 0.003)
        latency = b["latency"] + self._noise(n, 5.0)
        upstream_error_rate = b["upstream_error_rate"] + self._noise(n, 0.003)
        dependency_latency = b["dependency_latency"] + self._noise(n, 3.0)

        error_rate[change_point:] += 0.12
        latency[change_point:] += 80
        avg_cpu_usage[change_point:] += 10
        timeout_log_count = (latency > np.percentile(latency, 70)).astype(int)
        oom_log_count = np.zeros(n, dtype=int)

        return self._finalize(
            avg_cpu_usage=avg_cpu_usage,
            mem_growth=mem_growth,
            oom_log_count=oom_log_count,
            request_rate=request_rate,
            error_rate=error_rate,
            latency=latency,
            upstream_error_rate=upstream_error_rate,
            dependency_latency=dependency_latency,
            timeout_log_count=timeout_log_count,
        )

    def _dependency_failure_pattern(self, t: np.ndarray, b: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(t)
        change_point = max(2, n // 4)

        dependency_latency = b["dependency_latency"] + self._noise(n, 3.0)
        upstream_error_rate = b["upstream_error_rate"] + self._noise(n, 0.004)
        latency = b["latency"] + self._noise(n, 5.0)
        error_rate = b["error_rate"] + self._noise(n, 0.004)
        avg_cpu_usage = b["avg_cpu_usage"] + self._noise(n, 2.5)
        mem_growth = b["mem_growth"] + self._noise(n, 0.01)
        request_rate = b["request_rate"] + self._noise(n, 3.0)

        dependency_latency[change_point:] += np.linspace(30, 120, n - change_point)
        upstream_error_rate[change_point:] += np.linspace(0.03, 0.18, n - change_point)
        latency[change_point:] += 0.6 * (dependency_latency[change_point:] - b["dependency_latency"])
        error_rate[change_point:] += 0.5 * upstream_error_rate[change_point:]

        timeout_log_count = (latency > np.percentile(latency, 70)).astype(int)
        oom_log_count = np.zeros(n, dtype=int)

        return self._finalize(
            avg_cpu_usage=avg_cpu_usage,
            mem_growth=mem_growth,
            oom_log_count=oom_log_count,
            request_rate=request_rate,
            error_rate=error_rate,
            latency=latency,
            upstream_error_rate=upstream_error_rate,
            dependency_latency=dependency_latency,
            timeout_log_count=timeout_log_count,
        )

    def _cpu_saturation_pattern(self, t: np.ndarray, b: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(t)
        avg_cpu_usage = b["avg_cpu_usage"] + 2.8 * t + self._noise(n, 3.5)
        request_rate = b["request_rate"] + self._noise(n, 4.0)
        latency = b["latency"] + 1.8 * np.maximum(avg_cpu_usage - 70, 0) + self._noise(n, 4.0)
        error_rate = b["error_rate"] + 0.002 * np.maximum(avg_cpu_usage - 75, 0) + self._noise(n, 0.003)
        mem_growth = b["mem_growth"] + self._noise(n, 0.01)
        dependency_latency = b["dependency_latency"] + self._noise(n, 2.0)
        upstream_error_rate = b["upstream_error_rate"] + self._noise(n, 0.002)
        timeout_log_count = (latency > np.percentile(latency, 75)).astype(int)
        oom_log_count = np.zeros(n, dtype=int)

        return self._finalize(
            avg_cpu_usage=avg_cpu_usage,
            mem_growth=mem_growth,
            oom_log_count=oom_log_count,
            request_rate=request_rate,
            error_rate=error_rate,
            latency=latency,
            upstream_error_rate=upstream_error_rate,
            dependency_latency=dependency_latency,
            timeout_log_count=timeout_log_count,
        )

    def _traffic_spike_pattern(self, t: np.ndarray, b: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(t)
        center = n // 2
        spike = np.exp(-0.5 * ((t - center) / 2.0) ** 2) * 180

        request_rate = b["request_rate"] + spike + self._noise(n, 6.0)
        avg_cpu_usage = b["avg_cpu_usage"] + 0.22 * spike + self._noise(n, 3.0)
        latency = b["latency"] + 0.35 * spike + self._noise(n, 6.0)
        error_rate = b["error_rate"] + 0.0008 * spike + self._noise(n, 0.003)
        mem_growth = b["mem_growth"] + self._noise(n, 0.01)
        dependency_latency = b["dependency_latency"] + self._noise(n, 2.0)
        upstream_error_rate = b["upstream_error_rate"] + self._noise(n, 0.003)
        timeout_log_count = (latency > np.percentile(latency, 75)).astype(int)
        oom_log_count = np.zeros(n, dtype=int)

        return self._finalize(
            avg_cpu_usage=avg_cpu_usage,
            mem_growth=mem_growth,
            oom_log_count=oom_log_count,
            request_rate=request_rate,
            error_rate=error_rate,
            latency=latency,
            upstream_error_rate=upstream_error_rate,
            dependency_latency=dependency_latency,
            timeout_log_count=timeout_log_count,
        )

    def _normal_pattern(self, t: np.ndarray, b: Dict[str, float]) -> Dict[str, np.ndarray]:
        n = len(t)
        avg_cpu_usage = b["avg_cpu_usage"] + self._noise(n, 2.0)
        mem_growth = b["mem_growth"] + self._noise(n, 0.005)
        request_rate = b["request_rate"] + self._noise(n, 3.0)
        error_rate = b["error_rate"] + self._noise(n, 0.002)
        latency = b["latency"] + self._noise(n, 3.0)
        upstream_error_rate = b["upstream_error_rate"] + self._noise(n, 0.002)
        dependency_latency = b["dependency_latency"] + self._noise(n, 2.0)
        oom_log_count = np.zeros(n, dtype=int)
        timeout_log_count = np.zeros(n, dtype=int)

        return self._finalize(
            avg_cpu_usage=avg_cpu_usage,
            mem_growth=mem_growth,
            oom_log_count=oom_log_count,
            request_rate=request_rate,
            error_rate=error_rate,
            latency=latency,
            upstream_error_rate=upstream_error_rate,
            dependency_latency=dependency_latency,
            timeout_log_count=timeout_log_count,
        )

    def _finalize(self, **metrics: np.ndarray) -> Dict[str, np.ndarray]:
        metrics["avg_cpu_usage"] = np.clip(metrics["avg_cpu_usage"], 0, 100)
        metrics["mem_growth"] = self._clip_nonnegative(metrics["mem_growth"])
        metrics["request_rate"] = self._clip_nonnegative(metrics["request_rate"])
        metrics["error_rate"] = np.clip(metrics["error_rate"], 0, 1)
        metrics["latency"] = self._clip_nonnegative(metrics["latency"])
        metrics["upstream_error_rate"] = np.clip(metrics["upstream_error_rate"], 0, 1)
        metrics["dependency_latency"] = self._clip_nonnegative(metrics["dependency_latency"])
        metrics["oom_log_count"] = np.asarray(metrics["oom_log_count"], dtype=int)
        metrics["timeout_log_count"] = np.asarray(metrics["timeout_log_count"], dtype=int)
        return metrics


def generate_sequence_dataset(
    n_incidents: int = 5000,
    sequence_length: int = 20,
    random_seed: int = 42,
) -> pd.DataFrame:
    config = SequenceGeneratorConfig(
        n_incidents=n_incidents,
        sequence_length=sequence_length,
        random_seed=random_seed,
    )
    generator = IncidentSequenceGenerator(config)
    return generator.generate()