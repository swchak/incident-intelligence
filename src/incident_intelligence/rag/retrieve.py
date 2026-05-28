from __future__ import annotations

from pathlib import Path
from typing import Any

from incident_intelligence.rag.index import RagIndexConfig, _create_embedding_model
from incident_intelligence.settings import SETTINGS


def _resolve_project_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return SETTINGS.project_root / path


def _create_persistent_client(chroma_dir: Path):
    import chromadb

    return chromadb.PersistentClient(path=str(chroma_dir))


def retrieve_similar_documents(
    query: str,
    cfg: RagIndexConfig,
    *,
    n_results: int = 5,
    client_factory=None,
    model_factory=None,
) -> list[dict[str, Any]]:
    output_dir = _resolve_project_path(cfg.output_dir)
    chroma_dir = output_dir / "chroma"
    client = (client_factory or _create_persistent_client)(chroma_dir)
    collection = client.get_collection(name=cfg.collection_name)
    model = (model_factory or _create_embedding_model)(cfg.model_name)

    if hasattr(model, "encode_query"):
        query_embedding = model.encode_query([query], convert_to_numpy=True, normalize_embeddings=True)
    else:
        query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    results = collection.query(
        query_embeddings=[
            query_embedding[0].tolist() if hasattr(query_embedding[0], "tolist") else list(query_embedding[0])
        ],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "document": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]
