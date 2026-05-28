from __future__ import annotations

import argparse
import json

from incident_intelligence.config import RagIndexCLIConfig, load_config, merge_cli_args
from incident_intelligence.rag.diagnose import diagnose_rag_index
from incident_intelligence.rag.index import RagIndexConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the local incident knowledge-base vector index."
    )
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = merge_cli_args(args, load_config(RagIndexCLIConfig, "rag_index"))
    result = diagnose_rag_index(
        RagIndexConfig(
            input_dir=settings.input_dir,
            output_dir=settings.output_dir,
            collection_name=settings.collection_name,
            model_name=settings.model_name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    )

    if args.as_json:
        print(json.dumps(result, indent=2))
        return

    rag = result["rag"]
    print(f"Index exists:      {result['index_exists']}")
    print(f"Input dir:         {rag['input_dir']}")
    print(f"Output dir:        {rag['output_dir']}")
    print(f"Collection:        {rag['collection_name']}")
    print(f"Embedding model:   {rag['model_name']}")
    print(f"Chunk size:        {rag['chunk_size']}")
    print(f"Chunk overlap:     {rag['chunk_overlap']}")
    print(f"Chroma dir:        {rag['chroma_dir']}")
    print(f"Manifest:          {rag['manifest_path']}")
    print(f"Chroma exists:     {rag['chroma_exists']}")
    print(f"Manifest exists:   {rag['manifest_exists']}")
    print(f"Indexed documents: {rag['n_documents']}")
