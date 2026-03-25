from __future__ import annotations

import unittest

from incident_intelligence.data.generate_sequence import (
    IncidentSequenceGenerator,
    SequenceGeneratorConfig,
)


class SequenceGenerationTests(unittest.TestCase):
    def test_generate_respects_label_probabilities_and_shape(self) -> None:
        generator = IncidentSequenceGenerator(
            SequenceGeneratorConfig(
                n_incidents=3,
                sequence_length=4,
                random_seed=7,
                label_probs={
                    "memory_leak": 0.0,
                    "bad_deployment": 0.0,
                    "external_dependency_failure": 0.0,
                    "cpu_saturation": 0.0,
                    "traffic_spike": 0.0,
                    "normal": 1.0,
                },
            )
        )

        df = generator.generate()

        self.assertEqual(len(df), 12)
        self.assertEqual(df["incident_id"].nunique(), 3)
        self.assertEqual(set(df["root_cause_label"]), {"normal"})
        self.assertEqual(df.groupby("incident_id")["timestep"].count().tolist(), [4, 4, 4])
        self.assertTrue(
            {
                "avg_cpu_usage",
                "mem_growth",
                "request_rate",
                "error_rate",
                "latency",
                "dependency_latency",
            }.issubset(df.columns)
        )

    def test_missing_root_cause_probability_raises(self) -> None:
        with self.assertRaises(ValueError) as exc_info:
            IncidentSequenceGenerator(
                SequenceGeneratorConfig(
                    label_probs={
                        "memory_leak": 0.2,
                        "bad_deployment": 0.2,
                        "external_dependency_failure": 0.2,
                        "cpu_saturation": 0.2,
                        "traffic_spike": 0.2,
                    }
                )
            )

        self.assertIn("missing required root causes", str(exc_info.exception))

    def test_unknown_root_cause_probability_raises(self) -> None:
        with self.assertRaises(ValueError) as exc_info:
            IncidentSequenceGenerator(
                SequenceGeneratorConfig(
                    label_probs={
                        "memory_leak": 0.18,
                        "bad_deployment": 0.15,
                        "external_dependency_failure": 0.17,
                        "cpu_saturation": 0.16,
                        "traffic_spike": 0.14,
                        "normal": 0.18,
                        "cache_miss": 0.02,
                    }
                )
            )

        self.assertIn("unknown root causes", str(exc_info.exception))
