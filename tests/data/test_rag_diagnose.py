from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from incident_intelligence.rag.diagnose import diagnose_rag_index
from incident_intelligence.rag.index import RagIndexConfig


class RagDiagnoseTests(unittest.TestCase):
    def test_diagnose_rag_index_reads_manifest_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "chroma").mkdir()
            (output_dir / "documents_manifest.json").write_text(
                '{"n_documents": 12, "collection_name": "incident_knowledge_base"}',
                encoding="utf-8",
            )

            result = diagnose_rag_index(
                RagIndexConfig(
                    input_dir=str(output_dir / "kb"),
                    output_dir=str(output_dir),
                    collection_name="incident_knowledge_base",
                )
            )

        self.assertTrue(result["index_exists"])
        self.assertEqual(result["rag"]["n_documents"], 12)
        self.assertTrue(result["rag"]["chroma_exists"])
        self.assertTrue(result["rag"]["manifest_exists"])
