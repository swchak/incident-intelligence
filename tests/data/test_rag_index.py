from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from incident_intelligence.rag.index import RagIndexConfig, build_rag_index


class _FakeCollection:
    def __init__(self) -> None:
        self.rows = []

    def upsert(self, **kwargs) -> None:
        self.rows.append(kwargs)


class _FakeClient:
    def __init__(self) -> None:
        self.collection = _FakeCollection()

    def get_or_create_collection(self, name: str):
        self.collection_name = name
        return self.collection


class _FakeModel:
    def encode_document(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return [[float(index + 1), 0.5] for index, _ in enumerate(texts)]


class RagIndexTests(unittest.TestCase):
    def test_build_rag_index_writes_manifest_and_upserts_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            input_dir = base_dir / "kb"
            output_dir = base_dir / "rag"
            (input_dir / "incidents").mkdir(parents=True)
            (input_dir / "incidents" / "INC-1001.md").write_text(
                "# Incident INC-1001\n\n## Summary\nSynthetic incident summary.\n",
                encoding="utf-8",
            )

            fake_client = _FakeClient()
            result = build_rag_index(
                RagIndexConfig(
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    collection_name="kb_test",
                    chunk_size=200,
                    chunk_overlap=20,
                ),
                client_factory=lambda chroma_dir: fake_client,
                model_factory=lambda model_name: _FakeModel(),
            )

            self.assertEqual(result["n_documents"], 1)
            self.assertTrue((output_dir / "documents_manifest.json").exists())

            manifest = json.loads((output_dir / "documents_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["collection_name"], "kb_test")
            self.assertEqual(manifest["n_documents"], 1)
            self.assertEqual(len(fake_client.collection.rows), 1)

    def test_build_rag_index_upserts_in_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            input_dir = base_dir / "kb"
            output_dir = base_dir / "rag"
            (input_dir / "incidents").mkdir(parents=True)

            for index in range(5):
                (input_dir / "incidents" / f"INC-{index:04d}.md").write_text(
                    f"# Incident INC-{index:04d}\n\n## Summary\nSynthetic incident summary {index}.\n",
                    encoding="utf-8",
                )

            fake_client = _FakeClient()
            result = build_rag_index(
                RagIndexConfig(
                    input_dir=str(input_dir),
                    output_dir=str(output_dir),
                    collection_name="kb_test",
                    chunk_size=200,
                    chunk_overlap=20,
                    upsert_batch_size=2,
                ),
                client_factory=lambda chroma_dir: fake_client,
                model_factory=lambda model_name: _FakeModel(),
            )

            self.assertEqual(result["n_documents"], 5)
            self.assertEqual(len(fake_client.collection.rows), 3)
            self.assertEqual(len(fake_client.collection.rows[0]["ids"]), 2)
            self.assertEqual(len(fake_client.collection.rows[1]["ids"]), 2)
            self.assertEqual(len(fake_client.collection.rows[2]["ids"]), 1)
