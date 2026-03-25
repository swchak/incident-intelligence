from __future__ import annotations

import unittest

import pandas as pd

from incident_intelligence.data.temporal_features import build_temporal_feature_dataset


class TemporalFeatureEngineeringTests(unittest.TestCase):
    def test_build_temporal_feature_dataset_creates_expected_aggregates(self) -> None:
        sequence_df = pd.DataFrame(
            [
                {
                    "incident_id": 10,
                    "timestep": 0,
                    "root_cause_label": "memory_leak",
                    "avg_cpu_usage": 10.0,
                    "mem_growth": 0.10,
                    "oom_log_count": 0,
                    "request_rate": 100.0,
                    "error_rate": 0.01,
                    "latency": 50.0,
                    "upstream_error_rate": 0.02,
                    "dependency_latency": 20.0,
                    "timeout_log_count": 1,
                },
                {
                    "incident_id": 10,
                    "timestep": 1,
                    "root_cause_label": "memory_leak",
                    "avg_cpu_usage": 20.0,
                    "mem_growth": 0.20,
                    "oom_log_count": 2,
                    "request_rate": 120.0,
                    "error_rate": 0.03,
                    "latency": 70.0,
                    "upstream_error_rate": 0.04,
                    "dependency_latency": 30.0,
                    "timeout_log_count": 2,
                },
            ]
        )

        features = build_temporal_feature_dataset(sequence_df)

        self.assertEqual(features.shape[0], 1)
        row = features.iloc[0]
        self.assertEqual(row["incident_id"], 10)
        self.assertEqual(row["root_cause_label"], "memory_leak")
        self.assertAlmostEqual(row["avg_cpu_usage_start"], 10.0)
        self.assertAlmostEqual(row["avg_cpu_usage_end"], 20.0)
        self.assertAlmostEqual(row["avg_cpu_usage_delta"], 10.0)
        self.assertAlmostEqual(row["avg_cpu_usage_mean"], 15.0)
        self.assertAlmostEqual(row["dependency_to_service_latency_ratio"], 25.0 / 60.0)
        self.assertEqual(row["oom_total"], 2)
        self.assertEqual(row["timeout_total"], 3)

    def test_missing_required_columns_raise(self) -> None:
        bad_df = pd.DataFrame([{"incident_id": 1, "timestep": 0}])

        with self.assertRaises(ValueError) as exc_info:
            build_temporal_feature_dataset(bad_df)

        self.assertIn("missing required columns", str(exc_info.exception))
