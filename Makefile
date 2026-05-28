.PHONY: help install \
	generate generate-sequence build-temporal-features \
	knowledge-base rag-index rag-query rag-diagnose evaluate-rag \
	train train-temporal \
	evaluate evaluate-temporal \
	explain explain-temporal \
	explain-local explain-local-temporal \
	test test-backend test-frontend \
	api web-install web-dev \
	docker-build docker-up docker-down \
	pipeline pipeline-temporal pipeline-temporal-fast \
	clean

PY ?= ./venv/bin/python
NPM ?= npm
TRAIN_ARGS ?=
EVAL_ARGS ?=
EXPLAIN_ARGS ?=
EXPLAIN_LOCAL_ARGS ?=
PIPELINE_ARGS ?=
KB_ARGS ?=
RAG_ARGS ?=
RAG_QUERY ?=
RAG_QUERY_ARGS ?=
RAG_DIAGNOSE_ARGS ?=
RAG_EVAL_ARGS ?=

help:
	@echo "Targets:"
	@echo "  install                 - install package editable (pip install -e .)"
	@echo "  generate                - generate snapshot raw + train/val/eval splits"
	@echo "  generate-sequence       - generate temporal raw sequence data"
	@echo "  build-temporal-features - build temporal feature datasets from sequences"
	@echo "  knowledge-base          - generate synthetic incident markdown, runbooks, and postmortems"
	@echo "  rag-index               - build a local Chroma index over knowledge-base markdown"
	@echo "  rag-query               - query the local knowledge-base vector index"
	@echo "  rag-diagnose            - inspect local Chroma index health and manifest details"
	@echo "  evaluate-rag            - score retrieval quality over incident knowledge-base documents"
	@echo "  train                   - train snapshot models"
	@echo "  train-temporal          - train temporal models"
	@echo "  evaluate                - evaluate snapshot models"
	@echo "  evaluate-temporal       - evaluate temporal models"
	@echo "  explain                 - generate snapshot global explainability artifacts"
	@echo "  explain-temporal        - generate temporal global explainability artifacts"
	@echo "  explain-local           - generate snapshot local explainability artifacts"
	@echo "  explain-local-temporal  - generate temporal local explainability artifacts"
	@echo "  test-backend            - run backend API tests"
	@echo "  test-frontend           - run frontend dashboard tests"
	@echo "  test                    - run backend and frontend tests"
	@echo "  api                     - run the FastAPI dashboard backend"
	@echo "  web-install             - install frontend dependencies"
	@echo "  web-dev                 - run the Vite dashboard frontend"
	@echo "  docker-build            - build API and frontend deployment images"
	@echo "  docker-up               - start the full stack with docker compose"
	@echo "  docker-down             - stop the docker compose stack"
	@echo "  pipeline                - run the full snapshot pipeline"
	@echo "  pipeline-temporal       - run the full temporal pipeline"
	@echo "  pipeline-temporal-fast  - run the temporal pipeline with faster training defaults"
	@echo "  clean                   - remove artifacts and data"
	@echo ""
	@echo "Optional overrides:"
	@echo "  TRAIN_ARGS='--fast-mode --models logistic,rf --cv 3 --n-jobs 1'"
	@echo "  EVAL_ARGS='...'"
	@echo "  EXPLAIN_ARGS='...'"
	@echo "  EXPLAIN_LOCAL_ARGS='...'"
	@echo "  PIPELINE_ARGS='...'"
	@echo "  KB_ARGS='--input-path data/raw/incidents_sequence_raw.csv --max-postmortems 6'"
	@echo "  RAG_ARGS='--input-dir data/knowledge_base --collection-name incident_knowledge_base'"
	@echo "  RAG_QUERY='memory leak symptoms' RAG_QUERY_ARGS='--n-results 3'"
	@echo "  RAG_DIAGNOSE_ARGS='--json'"
	@echo "  RAG_EVAL_ARGS='--top-k 5 --max-incidents 50'"

install:
	$(PY) -m pip install -e .

generate:
	$(PY) -m incident_intelligence.cli.generator

generate-sequence:
	$(PY) -m incident_intelligence.cli.generate_sequence

build-temporal-features:
	$(PY) -m incident_intelligence.cli.build_temporal_features

knowledge-base:
	$(PY) -m incident_intelligence.cli.generate_knowledge_base $(KB_ARGS)

rag-index:
	$(PY) -m incident_intelligence.cli.build_rag_index $(RAG_ARGS)

rag-query:
	$(PY) -m incident_intelligence.cli.query_rag "$(RAG_QUERY)" $(RAG_QUERY_ARGS)

rag-diagnose:
	$(PY) -m incident_intelligence.cli.diagnose_rag $(RAG_DIAGNOSE_ARGS)

evaluate-rag:
	$(PY) -m incident_intelligence.cli.evaluate_rag $(RAG_EVAL_ARGS)

train:
	$(PY) -m incident_intelligence.cli.train $(TRAIN_ARGS)

train-temporal:
	$(PY) -m incident_intelligence.cli.train --dataset-kind temporal $(TRAIN_ARGS)

evaluate:
	$(PY) -m incident_intelligence.cli.evaluate $(EVAL_ARGS)

evaluate-temporal:
	$(PY) -m incident_intelligence.cli.evaluate --dataset-kind temporal $(EVAL_ARGS)

explain:
	$(PY) -m incident_intelligence.cli.explain $(EXPLAIN_ARGS)

explain-temporal:
	$(PY) -m incident_intelligence.cli.explain --dataset-kind temporal $(EXPLAIN_ARGS)

explain-local:
	$(PY) -m incident_intelligence.cli.explain_local $(EXPLAIN_LOCAL_ARGS)

explain-local-temporal:
	$(PY) -m incident_intelligence.cli.explain_local --dataset-kind temporal $(EXPLAIN_LOCAL_ARGS)

test-backend:
	PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp $(PY) -m unittest discover -s tests

test-frontend:
	cd web && $(NPM) test

test: test-backend test-frontend

api:
	PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp $(PY) -m incident_intelligence.api.app

web-install:
	cd web && $(NPM) install

web-dev:
	cd web && $(NPM) run dev

docker-build:
	docker compose build

docker-up:
	docker compose up --build

docker-down:
	docker compose down

pipeline:
	$(PY) -m incident_intelligence.cli.pipeline $(PIPELINE_ARGS)

pipeline-temporal:
	$(PY) -m incident_intelligence.cli.pipeline --dataset-kind temporal $(PIPELINE_ARGS)

pipeline-temporal-fast:
	MPLBACKEND=Agg $(PY) -m incident_intelligence.cli.pipeline --dataset-kind temporal --fast-mode --models logistic,rf --n-jobs 1 --cv 3 --verbose 0 $(PIPELINE_ARGS)

clean:
	rm -rf artifacts data
	@echo "Removed artifacts/ and data/"
