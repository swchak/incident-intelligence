from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from incident_intelligence.rag.evaluate import RagEvaluationConfig, evaluate_rag


def _write_incident(path: Path, *, incident_id: str, summary: str, symptoms: list[str], root_cause: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# Incident {incident_id}",
                "",
                "## Summary",
                summary,
                "",
                "## Symptoms",
                *[f"- {symptom}" for symptom in symptoms],
                "",
                "## Root Cause",
                root_cause,
                "",
                "## Resolution",
                "- Synthetic fix",
                "",
            ]
        ),
        encoding="utf-8",
    )


class RagEvaluateTests(unittest.TestCase):
    def test_evaluate_rag_scores_retrieval_against_incident_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            incidents_dir = base_dir / "incidents"
            incidents_dir.mkdir()
            _write_incident(
                incidents_dir / "INC-1001.md",
                incident_id="INC-1001",
                summary="Memory usage increased steadily before OOM events.",
                symptoms=["Latency increased gradually", "OOM logs appeared late"],
                root_cause="memory_leak",
            )
            _write_incident(
                incidents_dir / "INC-1002.md",
                incident_id="INC-1002",
                summary="Error rate jumped right after deployment.",
                symptoms=["Rollback reduced errors", "Latency shifted after rollout"],
                root_cause="bad_deployment",
            )

            def fake_retrieve(query: str, cfg, n_results: int = 5):
                if "memory" in query.lower():
                    return [
                        {
                            "document": "# Runbook: Memory Leak\n\n## Root Cause\nmemory_leak\n",
                            "metadata": {
                                "source_path": "runbooks/memory-leak.md",
                                "doc_type": "runbooks",
                                "title": "Runbook: Memory Leak",
                            },
                            "distance": 0.08,
                        }
                    ]
                return [
                    {
                        "document": "# Incident INC-1002\n\n## Root Cause\nbad_deployment\n",
                        "metadata": {
                            "source_path": "incidents/INC-1002.md",
                            "doc_type": "incidents",
                            "title": "Incident INC-1002",
                        },
                        "distance": 0.11,
                    }
                ]

            result = evaluate_rag(
                RagEvaluationConfig(
                    input_dir=str(base_dir),
                    output_dir=str(base_dir / "artifacts"),
                    top_k=3,
                    max_incidents=10,
                ),
                retrieve_fn=fake_retrieve,
            )

        self.assertEqual(result["n_incidents"], 2)
        self.assertEqual(result["retrieval_hit_rate"], 1.0)
        self.assertEqual(result["answer_accuracy"], 1.0)
        self.assertEqual(len(result["per_incident"]), 2)
