from __future__ import annotations

import argparse
import json

from incident_intelligence.config import (
    RagAnswerCLIConfig,
    RagIndexCLIConfig,
    load_config,
    merge_cli_args,
)
from incident_intelligence.rag.evaluate import RagEvaluationConfig, evaluate_rag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality over incident knowledge-base documents."
    )
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-incidents", type=int, default=None)
    parser.add_argument("--answer-mode", type=str, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    index_settings = merge_cli_args(args, load_config(RagIndexCLIConfig, "rag_index"))
    answer_settings = load_config(RagAnswerCLIConfig, "rag_answer")

    result = evaluate_rag(
        RagEvaluationConfig(
            input_dir=index_settings.input_dir,
            output_dir=index_settings.output_dir,
            collection_name=index_settings.collection_name,
            model_name=index_settings.model_name,
            chunk_size=index_settings.chunk_size,
            chunk_overlap=index_settings.chunk_overlap,
            top_k=args.top_k,
            max_incidents=args.max_incidents if args.max_incidents is not None else 100,
        ),
        answer_mode=args.answer_mode or answer_settings.mode,
        max_evidence=answer_settings.max_evidence,
    )

    if args.as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"Evaluated incidents: {result['n_incidents']}")
    print(f"Top-k:               {result['top_k']}")
    print(f"Retrieval hit rate:  {result['retrieval_hit_rate']:.3f}")
    print(f"Answer accuracy:     {result['answer_accuracy']:.3f}")
    print("")
    print("Sample results")
    for row in result["per_incident"][:5]:
        print(
            f"- {row['incident_id']}: expected={row['expected_root_cause']} "
            f"predicted={row['predicted_root_cause']} retrieval_hit={row['retrieval_hit']}"
        )


if __name__ == "__main__":
    main()
