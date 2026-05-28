from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from incident_intelligence.rag.documents import build_chunked_documents
from incident_intelligence.settings import SETTINGS


@dataclass(frozen=True)
class RagIndexConfig:
    input_dir: str = "data/knowledge_base"
    output_dir: str = "artifacts/rag"
    collection_name: str = "incident_knowledge_base"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 120
    upsert_batch_size: int = 5000


def _resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SETTINGS.project_root / path


def _create_persistent_client(chroma_dir: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir))


def _create_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    if batch_size <= 0:
        raise ValueError("upsert_batch_size must be greater than 0")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def build_rag_index(
    cfg: RagIndexConfig,
    *,
    client_factory=None,
    model_factory=None,
) -> dict[str, Any]:
    input_dir = _resolve_project_path(cfg.input_dir)
    output_dir = _resolve_project_path(cfg.output_dir)
    chroma_dir = output_dir / "chroma"
    manifest_path = output_dir / "documents_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    documents = build_chunked_documents(
        input_dir=input_dir,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    if not documents:
        raise ValueError(f"No markdown documents found under {input_dir}")

    client = (client_factory or _create_persistent_client)(chroma_dir)
    collection = client.get_or_create_collection(name=cfg.collection_name)
    model = (model_factory or _create_embedding_model)(cfg.model_name)

    texts = [doc.text for doc in documents]
    ids = [doc.id for doc in documents]
    metadatas = [dict(doc.metadata) for doc in documents]
    if hasattr(model, "encode_document"):
        embeddings = model.encode_document(texts, convert_to_numpy=True, normalize_embeddings=True)
    else:
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    serializable_embeddings = [
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        for embedding in embeddings
    ]

    id_batches = _batched(ids, cfg.upsert_batch_size)
    text_batches = _batched(texts, cfg.upsert_batch_size)
    metadata_batches = _batched(metadatas, cfg.upsert_batch_size)
    embedding_batches = _batched(serializable_embeddings, cfg.upsert_batch_size)

    for batch_index in range(len(id_batches)):
        collection.upsert(
            ids=id_batches[batch_index],
            documents=text_batches[batch_index],
            metadatas=metadata_batches[batch_index],
            embeddings=embedding_batches[batch_index],
        )

    manifest = {
        "collection_name": cfg.collection_name,
        "model_name": cfg.model_name,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_documents": len(documents),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "upsert_batch_size": cfg.upsert_batch_size,
        "documents": [
            {
                "id": doc.id,
                "metadata": doc.metadata,
                "preview": doc.text[:200],
            }
            for doc in documents
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "collection_name": cfg.collection_name,
        "model_name": cfg.model_name,
        "n_documents": len(documents),
        "chroma_dir": chroma_dir,
        "manifest_path": manifest_path,
        "upsert_batch_size": cfg.upsert_batch_size,
    }
