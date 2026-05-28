from __future__ import annotations

import argparse

from incident_intelligence.config import (
    KnowledgeBaseCLIConfig,
    load_config,
    merge_cli_args,
)
from incident_intelligence.data.generate_knowledge_base import (
    KnowledgeBaseGeneratorConfig,
    generate_knowledge_base,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic knowledge-base markdown files from existing incident labels and telemetry."
    )
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-postmortems", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = merge_cli_args(
        args,
        load_config(KnowledgeBaseCLIConfig, "knowledge_base"),
    )

    result = generate_knowledge_base(
        KnowledgeBaseGeneratorConfig(
            input_path=settings.input_path,
            output_dir=settings.output_dir,
            max_postmortems=settings.max_postmortems,
            random_seed=settings.random_seed,
        )
    )

    print(f"Wrote incident docs: {result['n_incident_docs']} -> {result['incidents_dir']}")
    print(f"Wrote runbooks:      {result['n_runbooks']} -> {result['runbooks_dir']}")
    print(f"Wrote postmortems:   {result['n_postmortems']} -> {result['postmortems_dir']}")


if __name__ == "__main__":
    main()
