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

    def _make_incident_profile(self, label: str, sequence_length: int) -> Dict[str, float | int | str]:
        lower = max(2, int(sequence_length * 0.2))
        upper = max(lower + 1, int(sequence_length * 0.7))
        secondary_choices = [cause for cause in ROOT_CAUSES if cause != label]
        severity = float(self.rng.uniform(0.65, 1.20))
        if label != "normal" and self.rng.random() < 0.35:
            severity *= float(self.rng.uniform(0.35, 0.7))

        primary_blend = float(self.rng.uniform(0.75, 1.0))
        if label != "normal" and self.rng.random() < 0.40:
            primary_blend *= float(self.rng.uniform(0.45, 0.8))

        return {
            "severity": severity,
            "primary_blend": primary_blend,
            "noise_scale": float(self.rng.uniform(1.1, 1.9)),
            "change_point": int(self.rng.integers(lower, upper)),
            "spike_center": int(self.rng.integers(max(2, sequence_length // 4), max(3, (3 * sequence_length) // 4))),
            "spike_width": float(self.rng.uniform(2.0, 4.8)),
            "secondary_label": str(self.rng.choice(secondary_choices)),
            "secondary_weight": float(self.rng.uniform(0.15, 0.42)),
            "normal_noise_weight": float(self.rng.uniform(0.05, 0.22)),
        }

    def _generate_incident_sequence(
        self,
        incident_id: int,
        label: str,
        sequence_length: int,
    ) -> List[Dict]:
        baseline = self._make_baseline()
        profile = self._make_incident_profile(label, sequence_length)
        t = np.arange(sequence_length)

        if label == "memory_leak":
            metrics = self._memory_leak_pattern(t, baseline, profile)
        elif label == "bad_deployment":
            metrics = self._bad_deployment_pattern(t, baseline, profile)
        elif label == "external_dependency_failure":
            metrics = self._dependency_failure_pattern(t, baseline, profile)
        elif label == "cpu_saturation":
            metrics = self._cpu_saturation_pattern(t, baseline, profile)
        elif label == "traffic_spike":
            metrics = self._traffic_spike_pattern(t, baseline, profile)
        else:
            metrics = self._normal_pattern(t, baseline, profile)

        metrics = self._apply_secondary_symptoms(
            metrics=metrics,
            t=t,
            baseline=baseline,
            primary_label=label,
            profile=profile,
        )

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

    def _scaled_noise(self, n: int, scale: float, profile: Dict[str, float | int | str]) -> np.ndarray:
        return self._noise(n, scale * float(profile["noise_scale"]))

    def _secondary_overlay(
        self,
        label: str,
        t: np.ndarray,
        weight: float,
        profile: Dict[str, float | int | str],
    ) -> Dict[str, np.ndarray]:
        n = len(t)
        change_point = int(profile["change_point"])
        center = int(profile["spike_center"])
        width = float(profile["spike_width"])
        ramp = np.clip((t - change_point) / max(n - change_point, 1), 0, 1)
        spike = np.exp(-0.5 * ((t - center) / width) ** 2)

        zeros = np.zeros(n, dtype=float)
        overlay = {
            "avg_cpu_usage": zeros.copy(),
            "mem_growth": zeros.copy(),
            "oom_log_count": zeros.copy(),
            "request_rate": zeros.copy(),
            "error_rate": zeros.copy(),
            "latency": zeros.copy(),
            "upstream_error_rate": zeros.copy(),
            "dependency_latency": zeros.copy(),
            "timeout_log_count": zeros.copy(),
        }

        if label == "memory_leak":
            overlay["mem_growth"] += weight * (0.012 * ramp + self._noise(n, 0.003))
            overlay["latency"] += weight * (20 * ramp + self._noise(n, 3.0))
            overlay["avg_cpu_usage"] += weight * (4 * ramp + self._noise(n, 1.5))
        elif label == "bad_deployment":
            overlay["error_rate"] += weight * (0.08 * ramp + self._noise(n, 0.004))
            overlay["latency"] += weight * (30 * ramp + self._noise(n, 4.0))
            overlay["avg_cpu_usage"] += weight * (5 * ramp + self._noise(n, 1.5))
        elif label == "external_dependency_failure":
            overlay["dependency_latency"] += weight * (40 * ramp + self._noise(n, 3.0))
            overlay["upstream_error_rate"] += weight * (0.07 * ramp + self._noise(n, 0.003))
            overlay["latency"] += weight * (18 * ramp + self._noise(n, 3.0))
        elif label == "cpu_saturation":
            overlay["avg_cpu_usage"] += weight * (12 * ramp + self._noise(n, 2.0))
            overlay["latency"] += weight * (16 * ramp + self._noise(n, 3.0))
            overlay["error_rate"] += weight * (0.03 * ramp + self._noise(n, 0.002))
        elif label == "traffic_spike":
            overlay["request_rate"] += weight * (90 * spike + self._noise(n, 4.0))
            overlay["avg_cpu_usage"] += weight * (10 * spike + self._noise(n, 2.0))
            overlay["latency"] += weight * (14 * spike + self._noise(n, 3.0))
        else:
            overlay["latency"] += weight * self._noise(n, 2.0)
            overlay["error_rate"] += weight * self._noise(n, 0.002)

        return overlay

    def _apply_secondary_symptoms(
        self,
        *,
        metrics: Dict[str, np.ndarray],
        t: np.ndarray,
        baseline: Dict[str, float],
        primary_label: str,
        profile: Dict[str, float | int | str],
    ) -> Dict[str, np.ndarray]:
        weight = float(profile["secondary_weight"])
        if primary_label == "normal":
            weight *= 0.75
        overlay = self._secondary_overlay(
            str(profile["secondary_label"]),
            t,
            weight,
            profile,
        )
        normal_overlay = self._secondary_overlay(
            "normal",
            t,
            float(profile["normal_noise_weight"]),
            profile,
        )

        blended = {}
        for key, values in metrics.items():
            primary_scale = float(profile["primary_blend"])
            blended[key] = (
                primary_scale * values
                + overlay.get(key, 0.0)
                + normal_overlay.get(key, 0.0)
            )

        blended["latency"] += 0.15 * np.maximum(blended["dependency_latency"] - baseline["dependency_latency"], 0)
        blended["avg_cpu_usage"] += 0.06 * np.maximum(blended["request_rate"] - baseline["request_rate"], 0)
        blended["error_rate"] += 0.0015 * np.maximum(blended["latency"] - baseline["latency"], 0)
        blended["mem_growth"] += 0.0004 * np.maximum(blended["avg_cpu_usage"] - baseline["avg_cpu_usage"], 0)
        return blended

    def _memory_leak_pattern(
        self,
        t: np.ndarray,
        b: Dict[str, float],
        profile: Dict[str, float | int | str],
    ) -> Dict[str, np.ndarray]:
        n = len(t)
        severity = float(profile["severity"])
        mem_growth = b["mem_growth"] + (0.005 + 0.005 * severity) * t + self._scaled_noise(n, 0.016, profile)
        latency = b["latency"] + (0.7 + 0.8 * severity) * t + 10 * mem_growth + self._scaled_noise(n, 10.0, profile)
        avg_cpu_usage = b["avg_cpu_usage"] + (0.05 + 0.16 * severity) * t + self._scaled_noise(n, 4.0, profile)
        request_rate = b["request_rate"] + self._scaled_noise(n, 8.0, profile)
        error_rate = b["error_rate"] + 0.0007 * t + self._scaled_noise(n, 0.006, profile)
        dependency_latency = b["dependency_latency"] + self._scaled_noise(n, 4.0, profile)
        upstream_error_rate = b["upstream_error_rate"] + self._scaled_noise(n, 0.004, profile)
        oom_log_count = (mem_growth > np.percentile(mem_growth, 88)).astype(int)
        timeout_log_count = (latency > np.percentile(latency, 82)).astype(int)

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

    def _bad_deployment_pattern(
        self,
        t: np.ndarray,
        b: Dict[str, float],
        profile: Dict[str, float | int | str],
    ) -> Dict[str, np.ndarray]:
        n = len(t)
        change_point = int(profile["change_point"])
        severity = float(profile["severity"])

        avg_cpu_usage = b["avg_cpu_usage"] + self._scaled_noise(n, 4.0, profile)
        mem_growth = b["mem_growth"] + self._scaled_noise(n, 0.012, profile)
        request_rate = b["request_rate"] + self._scaled_noise(n, 6.0, profile)
        error_rate = b["error_rate"] + self._scaled_noise(n, 0.004, profile)
        latency = b["latency"] + self._scaled_noise(n, 8.0, profile)
        upstream_error_rate = b["upstream_error_rate"] + self._scaled_noise(n, 0.004, profile)
        dependency_latency = b["dependency_latency"] + self._scaled_noise(n, 4.0, profile)

        error_rate[change_point:] += 0.025 + 0.035 * severity
        latency[change_point:] += 16 + 18 * severity
        avg_cpu_usage[change_point:] += 2.5 + 3.5 * severity
        upstream_error_rate[change_point:] += 0.006 + 0.016 * severity
        timeout_log_count = (latency > np.percentile(latency, 76)).astype(int)
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

    def _dependency_failure_pattern(
        self,
        t: np.ndarray,
        b: Dict[str, float],
        profile: Dict[str, float | int | str],
    ) -> Dict[str, np.ndarray]:
        n = len(t)
        change_point = int(profile["change_point"])
        severity = float(profile["severity"])

        dependency_latency = b["dependency_latency"] + self._scaled_noise(n, 4.0, profile)
        upstream_error_rate = b["upstream_error_rate"] + self._scaled_noise(n, 0.005, profile)
        latency = b["latency"] + self._scaled_noise(n, 7.0, profile)
        error_rate = b["error_rate"] + self._scaled_noise(n, 0.005, profile)
        avg_cpu_usage = b["avg_cpu_usage"] + self._scaled_noise(n, 3.5, profile)
        mem_growth = b["mem_growth"] + self._scaled_noise(n, 0.012, profile)
        request_rate = b["request_rate"] + self._scaled_noise(n, 5.0, profile)

        dependency_latency[change_point:] += np.linspace(10, 42 + 18 * severity, n - change_point)
        upstream_error_rate[change_point:] += np.linspace(0.008, 0.045 + 0.03 * severity, n - change_point)
        latency[change_point:] += 0.42 * (dependency_latency[change_point:] - b["dependency_latency"])
        error_rate[change_point:] += 0.22 * upstream_error_rate[change_point:]

        timeout_log_count = (latency > np.percentile(latency, 76)).astype(int)
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

    def _cpu_saturation_pattern(
        self,
        t: np.ndarray,
        b: Dict[str, float],
        profile: Dict[str, float | int | str],
    ) -> Dict[str, np.ndarray]:
        n = len(t)
        severity = float(profile["severity"])
        avg_cpu_usage = b["avg_cpu_usage"] + (0.7 + 0.8 * severity) * t + self._scaled_noise(n, 5.5, profile)
        request_rate = b["request_rate"] + self._scaled_noise(n, 7.0, profile)
        latency = b["latency"] + 0.8 * np.maximum(avg_cpu_usage - 74, 0) + self._scaled_noise(n, 8.0, profile)
        error_rate = b["error_rate"] + 0.0008 * np.maximum(avg_cpu_usage - 80, 0) + self._scaled_noise(n, 0.005, profile)
        mem_growth = b["mem_growth"] + self._scaled_noise(n, 0.014, profile)
        dependency_latency = b["dependency_latency"] + self._scaled_noise(n, 4.0, profile)
        upstream_error_rate = b["upstream_error_rate"] + self._scaled_noise(n, 0.004, profile)
        timeout_log_count = (latency > np.percentile(latency, 80)).astype(int)
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

    def _traffic_spike_pattern(
        self,
        t: np.ndarray,
        b: Dict[str, float],
        profile: Dict[str, float | int | str],
    ) -> Dict[str, np.ndarray]:
        n = len(t)
        center = int(profile["spike_center"])
        width = float(profile["spike_width"])
        severity = float(profile["severity"])
        spike = np.exp(-0.5 * ((t - center) / width) ** 2) * (55 + 45 * severity)

        request_rate = b["request_rate"] + spike + self._scaled_noise(n, 10.0, profile)
        avg_cpu_usage = b["avg_cpu_usage"] + 0.14 * spike + self._scaled_noise(n, 5.0, profile)
        latency = b["latency"] + 0.15 * spike + self._scaled_noise(n, 10.0, profile)
        error_rate = b["error_rate"] + 0.00025 * spike + self._scaled_noise(n, 0.005, profile)
        mem_growth = b["mem_growth"] + self._scaled_noise(n, 0.014, profile)
        dependency_latency = b["dependency_latency"] + 0.03 * spike + self._scaled_noise(n, 4.0, profile)
        upstream_error_rate = b["upstream_error_rate"] + 0.0001 * spike + self._scaled_noise(n, 0.005, profile)
        timeout_log_count = (latency > np.percentile(latency, 80)).astype(int)
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

    def _normal_pattern(
        self,
        t: np.ndarray,
        b: Dict[str, float],
        profile: Dict[str, float | int | str],
    ) -> Dict[str, np.ndarray]:
        n = len(t)
        avg_cpu_usage = b["avg_cpu_usage"] + self._scaled_noise(n, 4.5, profile)
        mem_growth = b["mem_growth"] + self._scaled_noise(n, 0.012, profile)
        request_rate = b["request_rate"] + self._scaled_noise(n, 7.0, profile)
        error_rate = b["error_rate"] + self._scaled_noise(n, 0.004, profile)
        latency = b["latency"] + self._scaled_noise(n, 8.0, profile)
        upstream_error_rate = b["upstream_error_rate"] + self._scaled_noise(n, 0.004, profile)
        dependency_latency = b["dependency_latency"] + self._scaled_noise(n, 4.0, profile)
        if self.rng.random() < 0.45:
            mild_overlay = self._secondary_overlay(
                str(profile["secondary_label"]),
                t,
                float(profile["secondary_weight"]) * float(self.rng.uniform(0.35, 0.7)),
                profile,
            )
            avg_cpu_usage += mild_overlay["avg_cpu_usage"]
            mem_growth += mild_overlay["mem_growth"]
            request_rate += mild_overlay["request_rate"]
            error_rate += mild_overlay["error_rate"]
            latency += mild_overlay["latency"]
            upstream_error_rate += mild_overlay["upstream_error_rate"]
            dependency_latency += mild_overlay["dependency_latency"]
        timeout_log_count = (latency > np.percentile(latency, 92)).astype(int)
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
