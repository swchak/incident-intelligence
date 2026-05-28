from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from incident_intelligence.rag.index import RagIndexConfig
from incident_intelligence.settings import SETTINGS


def _resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SETTINGS.project_root / path


def diagnose_rag_index(cfg: RagIndexConfig) -> dict[str, Any]:
    rag_root = _resolve_project_path(cfg.output_dir)
    chroma_dir = rag_root / "chroma"
    manifest_path = rag_root / "documents_manifest.json"

    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "index_exists": chroma_dir.exists() and manifest_path.exists(),
        "rag": {
            "input_dir": str(_resolve_project_path(cfg.input_dir)),
            "output_dir": str(rag_root),
            "collection_name": cfg.collection_name,
            "model_name": cfg.model_name,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "chroma_dir": str(chroma_dir),
            "manifest_path": str(manifest_path),
            "chroma_exists": chroma_dir.exists(),
            "manifest_exists": manifest_path.exists(),
            "n_documents": manifest.get("n_documents", 0) if isinstance(manifest, dict) else 0,
            "manifest": manifest or {},
        },
    }
