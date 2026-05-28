from __future__ import annotations

import argparse
import json

from incident_intelligence.config import (
    RagAnswerCLIConfig,
    RagIndexCLIConfig,
    load_config,
    merge_cli_args,
)
from incident_intelligence.rag.answer import build_grounded_context, build_template_answer
from incident_intelligence.rag.index import RagIndexConfig
from incident_intelligence.rag.retrieve import retrieve_similar_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query the local incident knowledge-base vector index."
    )
    parser.add_argument("query", type=str, help="Natural-language search query")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--answer-mode", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = merge_cli_args(args, load_config(RagIndexCLIConfig, "rag_index"))
    answer_settings = load_config(RagAnswerCLIConfig, "rag_answer")

    results = retrieve_similar_documents(
        query=args.query,
        cfg=RagIndexConfig(
            input_dir=settings.input_dir,
            output_dir=settings.output_dir,
            collection_name=settings.collection_name,
            model_name=settings.model_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        n_results=args.n_results,
    )
    answer = build_template_answer(
        args.query,
        results,
        max_evidence=answer_settings.max_evidence,
        mode=args.answer_mode or answer_settings.mode,
    )

    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print("")
    print("Template Answer")
    print(json.dumps(answer, indent=2))
    print("")
    print("Grounded Context")
    print(build_grounded_context(results, max_snippets=args.n_results))


if __name__ == "__main__":
    main()
