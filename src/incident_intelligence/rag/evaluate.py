from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable

from incident_intelligence.rag.answer import build_template_answer
from incident_intelligence.rag.documents import KnowledgeDocument, load_markdown_documents
from incident_intelligence.rag.index import RagIndexConfig
from incident_intelligence.rag.retrieve import retrieve_similar_documents


def _extract_section(text: str, heading: str) -> str:
    lines = text.splitlines()
    current_heading = f"## {heading}".strip()
    capture = False
    captured: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == current_heading:
            capture = True
            continue
        if capture and stripped.startswith("## "):
            break
        if capture:
            captured.append(line.rstrip())
    return "\n".join(captured).strip()


def _normalize_root_cause(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized


def _extract_expected_root_cause(document: KnowledgeDocument) -> str:
    root_cause = _extract_section(document.text, "Root Cause")
    if not root_cause:
        raise ValueError(f"Incident document {document.metadata.get('source_path', document.id)} has no Root Cause section")
    first_line = root_cause.splitlines()[0].strip()
    return _normalize_root_cause(first_line)


def _build_query(document: KnowledgeDocument) -> str:
    summary = _extract_section(document.text, "Summary")
    symptoms = _extract_section(document.text, "Symptoms")
    symptom_lines = [
        line.lstrip("-").strip()
        for line in symptoms.splitlines()
        if line.strip().startswith("-")
    ][:2]
    parts = [summary.strip(), *symptom_lines]
    query = " ".join(part for part in parts if part)
    return query or document.metadata.get("title", document.id)


def _root_cause_from_match(match: dict[str, Any]) -> str | None:
    metadata = match.get("metadata") or {}
    source_path = str(metadata.get("source_path", ""))
    doc_type = str(metadata.get("doc_type", ""))
    document = str(match.get("document", ""))

    if doc_type == "runbooks" and source_path:
        return _normalize_root_cause(Path(source_path).stem.replace("-", "_"))

    root_cause = _extract_section(document, "Root Cause")
    if root_cause:
        return _normalize_root_cause(root_cause.splitlines()[0].strip())
    return None


@dataclass(frozen=True)
class RagEvaluationConfig:
    input_dir: str = "data/knowledge_base"
    output_dir: str = "artifacts/rag"
    collection_name: str = "incident_knowledge_base"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 5
    max_incidents: int = 100


def evaluate_rag(
    cfg: RagEvaluationConfig,
    *,
    answer_mode: str = "template",
    max_evidence: int = 3,
    retrieve_fn: Callable[..., list[dict[str, Any]]] = retrieve_similar_documents,
) -> dict[str, Any]:
    documents = [
        document
        for document in load_markdown_documents(cfg.input_dir)
        if document.metadata.get("doc_type") == "incidents"
    ]
    if cfg.max_incidents > 0:
        documents = documents[: cfg.max_incidents]
    if not documents:
        raise ValueError("No incident documents found to evaluate in the knowledge base")

    rag_cfg = RagIndexConfig(
        input_dir=cfg.input_dir,
        output_dir=cfg.output_dir,
        collection_name=cfg.collection_name,
        model_name=cfg.model_name,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    per_incident: list[dict[str, Any]] = []
    retrieval_hits = 0
    answer_hits = 0

    for document in documents:
        expected_root_cause = _extract_expected_root_cause(document)
        query = _build_query(document)
        matches = retrieve_fn(query=query, cfg=rag_cfg, n_results=cfg.top_k)
        answer = build_template_answer(
            query,
            matches,
            max_evidence=max_evidence,
            mode=answer_mode,
        )
        predicted_root_cause = str(answer["predicted_root_cause"])
        retrieval_hit = any(
            _root_cause_from_match(match) == expected_root_cause
            for match in matches
        )
        answer_hit = predicted_root_cause == expected_root_cause
        retrieval_hits += int(retrieval_hit)
        answer_hits += int(answer_hit)
        top_match = matches[0] if matches else None
        top_metadata = (top_match or {}).get("metadata") or {}

        per_incident.append(
            {
                "incident_id": document.metadata.get("title", document.id),
                "source_path": document.metadata.get("source_path", ""),
                "query": query,
                "expected_root_cause": expected_root_cause,
                "predicted_root_cause": predicted_root_cause,
                "retrieval_hit": retrieval_hit,
                "answer_hit": answer_hit,
                "top_match_source_path": top_metadata.get("source_path"),
                "top_match_doc_type": top_metadata.get("doc_type"),
                "confidence": answer["confidence"],
            }
        )

    n_incidents = len(per_incident)
    return {
        "n_incidents": n_incidents,
        "top_k": cfg.top_k,
        "retrieval_hit_rate": retrieval_hits / n_incidents,
        "answer_accuracy": answer_hits / n_incidents,
        "per_incident": per_incident,
    }
