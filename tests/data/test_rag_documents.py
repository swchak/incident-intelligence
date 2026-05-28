from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from incident_intelligence.rag.documents import build_chunked_documents, split_text


class RagDocumentTests(unittest.TestCase):
    def test_split_text_creates_multiple_chunks(self) -> None:
        text = "A" * 1200
        chunks = split_text(text, chunk_size=500, chunk_overlap=100)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(chunks))

    def test_build_chunked_documents_loads_markdown_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "incidents").mkdir()
            (base_dir / "incidents" / "INC-1001.md").write_text(
                "# Incident INC-1001\n\n## Summary\nSynthetic incident summary.\n",
                encoding="utf-8",
            )

            documents = build_chunked_documents(base_dir, chunk_size=200, chunk_overlap=20)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].metadata["doc_type"], "incidents")
            self.assertEqual(documents[0].metadata["title"], "Incident INC-1001")
