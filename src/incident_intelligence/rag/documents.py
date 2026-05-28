from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from incident_intelligence.settings import SETTINGS


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    text: str
    metadata: dict[str, str | int]


def _resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SETTINGS.project_root / path


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def load_markdown_documents(input_dir: str | Path) -> list[KnowledgeDocument]:
    base_dir = _resolve_project_path(input_dir)
    documents: list[KnowledgeDocument] = []

    for path in sorted(base_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        relative_path = path.relative_to(base_dir).as_posix()
        parts = relative_path.split("/")
        doc_type = parts[0] if parts else "unknown"
        title = _extract_title(text, path.stem.replace("-", " ").replace("_", " "))
        documents.append(
            KnowledgeDocument(
                id=_slugify(relative_path),
                text=text,
                metadata={
                    "source_path": relative_path,
                    "doc_type": doc_type,
                    "title": title,
                },
            )
        )
    return documents


def split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    if len(normalized) <= chunk_size:
        return [normalized] if normalized else []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start += step
    return chunks


def build_chunked_documents(
    input_dir: str | Path,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[KnowledgeDocument]:
    source_documents = load_markdown_documents(input_dir)
    chunked_documents: list[KnowledgeDocument] = []

    for document in source_documents:
        chunks = split_text(document.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for index, chunk in enumerate(chunks):
            chunked_documents.append(
                KnowledgeDocument(
                    id=f"{document.id}-chunk-{index}",
                    text=chunk,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                        "chunk_count": len(chunks),
                    },
                )
            )
    return chunked_documents
