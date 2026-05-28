from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from incident_intelligence.data.generate_knowledge_base import (
    KnowledgeBaseGeneratorConfig,
    aggregate_incidents,
    generate_knowledge_base,
)


class KnowledgeBaseGenerationTests(unittest.TestCase):
    def test_aggregate_incidents_handles_sequence_rows(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "incident_id": 1,
                    "timestep": 0,
                    "root_cause_label": "memory_leak",
                    "avg_cpu_usage": 50.0,
                    "mem_growth": 0.3,
                    "oom_log_count": 0,
                    "request_rate": 100.0,
                    "error_rate": 0.1,
                    "latency": 200.0,
                    "upstream_error_rate": 0.02,
                    "dependency_latency": 80.0,
                    "timeout_log_count": 0,
                },
                {
                    "incident_id": 1,
                    "timestep": 1,
                    "root_cause_label": "memory_leak",
                    "avg_cpu_usage": 60.0,
                    "mem_growth": 0.5,
                    "oom_log_count": 1,
                    "request_rate": 110.0,
                    "error_rate": 0.2,
                    "latency": 240.0,
                    "upstream_error_rate": 0.03,
                    "dependency_latency": 90.0,
                    "timeout_log_count": 1,
                },
            ]
        )

        incidents_df = aggregate_incidents(df)

        self.assertEqual(len(incidents_df), 1)
        self.assertEqual(int(incidents_df.loc[0, "timesteps"]), 2)
        self.assertAlmostEqual(float(incidents_df.loc[0, "latency"]), 220.0)
        self.assertAlmostEqual(float(incidents_df.loc[0, "latency_max"]), 240.0)

    def test_generate_knowledge_base_writes_incidents_runbooks_and_postmortems(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "incident_id": 7,
                    "root_cause_label": "memory_leak",
                    "avg_cpu_usage": 62.0,
                    "mem_growth": 0.9,
                    "oom_log_count": 3,
                    "request_rate": 122.0,
                    "error_rate": 0.24,
                    "latency": 415.0,
                    "upstream_error_rate": 0.03,
                    "dependency_latency": 85.0,
                    "timeout_log_count": 1,
                },
                {
                    "incident_id": 8,
                    "root_cause_label": "bad_deployment",
                    "avg_cpu_usage": 51.0,
                    "mem_growth": 0.1,
                    "oom_log_count": 0,
                    "request_rate": 118.0,
                    "error_rate": 0.31,
                    "latency": 310.0,
                    "upstream_error_rate": 0.04,
                    "dependency_latency": 95.0,
                    "timeout_log_count": 2,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "incidents.csv"
            output_dir = Path(temp_dir) / "knowledge_base"
            df.to_csv(input_path, index=False)

            result = generate_knowledge_base(
                KnowledgeBaseGeneratorConfig(
                    input_path=str(input_path),
                    output_dir=str(output_dir),
                    max_postmortems=2,
                )
            )

            self.assertEqual(result["n_incident_docs"], 2)
            self.assertEqual(result["n_runbooks"], 2)
            self.assertEqual(result["n_postmortems"], 2)

            incident_text = (output_dir / "incidents" / "INC-0007.md").read_text(encoding="utf-8")
            self.assertIn("## Summary", incident_text)
            self.assertIn("## Root Cause", incident_text)
            self.assertIn("memory_leak", incident_text)

            runbook_text = (output_dir / "runbooks" / "memory-leak.md").read_text(encoding="utf-8")
            self.assertIn("## Investigation Steps", runbook_text)

            postmortem_text = (output_dir / "postmortems" / "INC-0007-postmortem.md").read_text(encoding="utf-8")
            self.assertIn("## Preventive Actions", postmortem_text)
