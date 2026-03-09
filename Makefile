.PHONY: help install generate train evaluate explain pipeline clean

PY ?= python

help:
	@echo "Targets:"
	@echo "  install    - install package editable (pip install -e .)"
	@echo "  generate   - generate raw + train/val/eval splits"
	@echo "  train      - train on train, select best on val"
	@echo "  evaluate   - evaluate saved models on eval"
	@echo "  explain    - generate explainability artifacts on eval"
	@echo "  pipeline   - run generate -> train -> evaluate -> explain"
	@echo "  clean      - remove artifacts (keeps data)"

install:
	$(PY) -m pip install -e .

generate:
	$(PY) scripts/generate_dataset.py

train:
	$(PY) scripts/train.py

evaluate:
	$(PY) scripts/evaluate.py

explain:
	$(PY) scripts/explain.py

pipeline:
	$(PY) scripts/run_pipeline.py

clean:
	rm -rf artifacts
	@echo "Removed artifacts/"