.PHONY: help install generate train evaluate explain explain-local pipeline clean

PY ?= python

help:
	@echo "Targets:"
	@echo "  install       - install package editable (pip install -e .)"
	@echo "  generate      - generate raw + train/val/eval splits"
	@echo "  train         - train on train, select best on val"
	@echo "  evaluate      - evaluate saved models on eval"
	@echo "  explain       - generate global explainability artifacts on eval"
	@echo "  explain-local - generate local explainability artifacts"
	@echo "  pipeline      - run generate -> train -> evaluate -> explain"
	@echo "  clean         - remove artifacts (keeps data)"

install:
	$(PY) -m pip install -e .

generate:
	$(PY) -m incident_intelligence.cli.generator

train:
	$(PY) -m incident_intelligence.cli.train

evaluate:
	$(PY) -m incident_intelligence.cli.evaluate

explain:
	$(PY) -m incident_intelligence.cli.explain

explain-local:
	$(PY) -m incident_intelligence.cli.explain_local

pipeline:
	$(PY) -m incident_intelligence.cli.pipeline

clean:
	rm -rf artifacts data
	@echo "Removed artifacts/"