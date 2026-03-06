# --------
# Defaults (override like: make pipeline N_SAMPLES=20000 SEED=7)
# --------
PY ?= python
N_SAMPLES ?= 4000
SEED ?= 42
TRAIN_SIZE ?= 0.70
VAL_SIZE ?= 0.15
LABEL_COL ?= root_cause_label

TRAIN_PATH ?= data/processed/incident_root_cause_train.csv
VAL_PATH ?= data/processed/incident_root_cause_val.csv
EVAL_PATH ?= data/processed/incident_root_cause_eval.csv

MODELS_DIR ?= artifacts/models
METRICS_DIR ?= artifacts/metrics
EXPLAIN_DIR ?= artifacts/explain


.PHONY: help install generate train evaluate explain pipeline clean

help:
	@echo "Targets:"
	@echo "  install    - install package editable (pip install -e .)"
	@echo "  generate   - generate raw + train/val/eval splits"
	@echo "  train      - train on train, select best on val"
	@echo "  evaluate   - evaluate saved models on eval"
	@echo "  explain    - generate explainability artifacts on eval"
	@echo "  pipeline   - run generate -> train -> evaluate -> explain"
	@echo "  clean      - remove artifacts (keeps data)"
	@echo ""
	@echo "Overrides:"
	@echo "  N_SAMPLES SEED TRAIN_SIZE VAL_SIZE LABEL_COL"
	@echo "  TRAIN_PATH VAL_PATH EVAL_PATH MODELS_DIR METRICS_DIR EXPLAIN_DIR"
	@echo ""
	@echo "Example:"
	@echo "  make pipeline N_SAMPLES=20000 SEED=7"


install:
	$(PY) -m pip install -e .


generate:
	$(PY) scripts/generate_dataset.py \
		--n-samples $(N_SAMPLES) \
		--seed $(SEED) \
		--train-size $(TRAIN_SIZE) \
		--val-size $(VAL_SIZE) \
		--label-col $(LABEL_COL) \
		--raw-out raw/incidents_raw.csv \
		--processed-dir processed


train:
	$(PY) scripts/train.py \
		--train $(TRAIN_PATH) \
		--val $(VAL_PATH) \
		--label-col $(LABEL_COL) \
		--models-out-dir $(MODELS_DIR) \
		--metrics-out-json $(METRICS_DIR)/train_val_results.json \
		--leaderboard-out-csv $(METRICS_DIR)/leaderboard_val.csv \
		--best-model-out $(MODELS_DIR)/best_model.joblib


evaluate:
	$(PY) scripts/evaluate.py \
		--data $(EVAL_PATH) \
		--label-col $(LABEL_COL) \
		--models-dir $(MODELS_DIR) \
		--metrics-out $(METRICS_DIR)/evaluation.json \
		--summary-csv-out $(METRICS_DIR)/evaluation_summary.csv


explain:
	$(PY) scripts/explain.py \
		--data $(EVAL_PATH) \
		--label-col $(LABEL_COL) \
		--models-dir $(MODELS_DIR) \
		--out-dir $(EXPLAIN_DIR)


pipeline: generate train evaluate explain
	@echo ""
	@echo "✅ Pipeline complete"
	@echo "Data:"
	@echo "  data/raw/incidents_raw.csv"
	@echo "  $(TRAIN_PATH)"
	@echo "  $(VAL_PATH)"
	@echo "  $(EVAL_PATH)"
	@echo "Artifacts:"
	@echo "  $(MODELS_DIR)/"
	@echo "  $(METRICS_DIR)/"
	@echo "  $(EXPLAIN_DIR)/"


clean:
	rm -rf artifacts
	@echo "Removed artifacts/"