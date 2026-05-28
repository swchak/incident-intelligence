from __future__ import annotations

import argparse

from incident_intelligence.config import RagIndexCLIConfig, load_config, merge_cli_args
from incident_intelligence.rag.index import RagIndexConfig, build_rag_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Chroma vector index from the synthetic incident knowledge base."
    )
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = merge_cli_args(args, load_config(RagIndexCLIConfig, "rag_index"))
    result = build_rag_index(
        RagIndexConfig(
            input_dir=settings.input_dir,
            output_dir=settings.output_dir,
            collection_name=settings.collection_name,
            model_name=settings.model_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    )
    print(f"Indexed documents: {result['n_documents']}")
    print(f"Chroma dir:         {result['chroma_dir']}")
    print(f"Manifest:           {result['manifest_path']}")


if __name__ == "__main__":
    main()
