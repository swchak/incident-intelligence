.PHONY: help install \
	generate generate-sequence build-temporal-features \
	train train-temporal \
	evaluate evaluate-temporal \
	explain explain-temporal \
	explain-local explain-local-temporal \
	api web-install web-dev \
	pipeline pipeline-temporal pipeline-temporal-fast \
	clean

PY ?= python
NPM ?= npm
TRAIN_ARGS ?=
EVAL_ARGS ?=
EXPLAIN_ARGS ?=
EXPLAIN_LOCAL_ARGS ?=
PIPELINE_ARGS ?=

help:
	@echo "Targets:"
	@echo "  install                 - install package editable (pip install -e .)"
	@echo "  generate                - generate snapshot raw + train/val/eval splits"
	@echo "  generate-sequence       - generate temporal raw sequence data"
	@echo "  build-temporal-features - build temporal feature datasets from sequences"
	@echo "  train                   - train snapshot models"
	@echo "  train-temporal          - train temporal models"
	@echo "  evaluate                - evaluate snapshot models"
	@echo "  evaluate-temporal       - evaluate temporal models"
	@echo "  explain                 - generate snapshot global explainability artifacts"
	@echo "  explain-temporal        - generate temporal global explainability artifacts"
	@echo "  explain-local           - generate snapshot local explainability artifacts"
	@echo "  explain-local-temporal  - generate temporal local explainability artifacts"
	@echo "  api                     - run the FastAPI dashboard backend"
	@echo "  web-install             - install frontend dependencies"
	@echo "  web-dev                 - run the Vite dashboard frontend"
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

install:
	$(PY) -m pip install -e .

generate:
	$(PY) -m incident_intelligence.cli.generator

generate-sequence:
	$(PY) -m incident_intelligence.cli.generate_sequence

build-temporal-features:
	$(PY) -m incident_intelligence.cli.build_temporal_features

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

api:
	PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp $(PY) -m incident_intelligence.api.app

web-install:
	cd web && $(NPM) install

web-dev:
	cd web && $(NPM) run dev

pipeline:
	$(PY) -m incident_intelligence.cli.pipeline $(PIPELINE_ARGS)

pipeline-temporal:
	$(PY) -m incident_intelligence.cli.pipeline --dataset-kind temporal $(PIPELINE_ARGS)

pipeline-temporal-fast:
	MPLBACKEND=Agg $(PY) -m incident_intelligence.cli.pipeline --dataset-kind temporal --fast-mode --models logistic,rf --n-jobs 1 --cv 3 --verbose 0 $(PIPELINE_ARGS)

clean:
	rm -rf artifacts data
	@echo "Removed artifacts/ and data/"
