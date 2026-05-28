from __future__ import annotations

import unittest
from pathlib import Path

from incident_intelligence.rag.answer import build_grounded_context, build_template_answer
from incident_intelligence.rag.index import RagIndexConfig
from incident_intelligence.rag.retrieve import retrieve_similar_documents


class _FakeQueryCollection:
    def query(self, **kwargs):
        return {
            "documents": [[
                "# Incident INC-1001\n\nMemory usage increased steadily.",
                "# Runbook: Memory Leak\n\nRecycle unhealthy pods.",
            ]],
            "metadatas": [[
                {"title": "Incident INC-1001", "source_path": "incidents/INC-1001.md"},
                {"title": "Runbook: Memory Leak", "source_path": "runbooks/memory-leak.md"},
            ]],
            "distances": [[0.12, 0.19]],
        }


class _FakeQueryClient:
    def get_collection(self, name: str):
        self.collection_name = name
        return _FakeQueryCollection()


class _FakeQueryModel:
    def encode_query(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return [[0.8, 0.2]]


class RagRetrieveTests(unittest.TestCase):
    def test_retrieve_similar_documents_returns_ranked_rows(self) -> None:
        results = retrieve_similar_documents(
            query="memory leak symptoms",
            cfg=RagIndexConfig(output_dir="artifacts/rag"),
            n_results=2,
            client_factory=lambda chroma_dir: _FakeQueryClient(),
            model_factory=lambda model_name: _FakeQueryModel(),
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["metadata"]["title"], "Incident INC-1001")
        self.assertAlmostEqual(results[0]["distance"], 0.12)

    def test_build_grounded_context_formats_sources(self) -> None:
        context = build_grounded_context(
            [
                {
                    "document": "Memory usage increased steadily before OOM logs appeared.",
                    "metadata": {
                        "title": "Incident INC-1001",
                        "source_path": "incidents/INC-1001.md",
                    },
                    "distance": 0.12,
                }
            ]
        )

        self.assertIn("Incident INC-1001", context)
        self.assertIn("incidents/INC-1001.md", context)
        self.assertIn("Memory usage increased steadily", context)

    def test_build_template_answer_returns_deterministic_summary(self) -> None:
        answer = build_template_answer(
            "memory leak symptoms",
            [
                {
                    "document": "Memory usage increased steadily before OOM logs appeared.",
                    "metadata": {
                        "title": "Incident INC-1001",
                        "source_path": "incidents/INC-1001.md",
                        "doc_type": "incidents",
                    },
                    "distance": 0.12,
                }
            ],
            max_evidence=2,
        )

        self.assertEqual(answer["answer_mode"], "template")
        self.assertEqual(answer["predicted_root_cause"], "memory_leak")
        self.assertGreater(answer["confidence"], 0.8)
        self.assertEqual(len(answer["retrieved_evidence"]), 1)
        self.assertIn("OOM logs appeared", answer["retrieved_evidence"][0]["snippet"])
        self.assertTrue(answer["next_steps"])
