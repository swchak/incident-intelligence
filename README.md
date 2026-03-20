# Incident Intelligence

Incident Intelligence is an end-to-end machine learning project for classifying incident root causes from synthetic telemetry data.

The repository now supports two workflows:

- `snapshot`: one row per incident with aggregate telemetry features
- `temporal`: multi-step incident sequences that are transformed into temporal features before training

The project covers dataset generation, model training, evaluation, global explainability, and local per-incident explainability through a CLI-first workflow.

## What The Project Does

The pipeline simulates production-style incidents using signals such as:

- CPU usage
- memory growth
- request rate
- latency
- upstream/dependency failures
- error and timeout signals

It then trains baseline classifiers to predict a synthetic `root_cause_label` such as:

- `memory_leak`
- `bad_deployment`
- `external_dependency_failure`
- `cpu_saturation`
- `traffic_spike`
- `normal`

## Current Workflows

### Snapshot Workflow

The snapshot workflow generates a flat incident dataset and writes:

```text
data/raw/incidents_raw.csv
data/processed/incident_snapshot_train.csv
data/processed/incident_snapshot_val.csv
data/processed/incident_snapshot_eval.csv
```

This path is driven by:

- [src/incident_intelligence/cli/generator.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/cli/generator.py)
- [src/incident_intelligence/data/generate_snapshot.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/data/generate_snapshot.py)

### Temporal Workflow

The temporal workflow generates incident sequences first, then builds aggregate temporal features from each sequence.

It writes:

```text
data/raw/incidents_sequence_raw.csv
data/processed/incident_temporal_all.csv
data/processed/incident_temporal_train.csv
data/processed/incident_temporal_val.csv
data/processed/incident_temporal_eval.csv
```

This path is driven by:

- [src/incident_intelligence/cli/generate_sequence.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/cli/generate_sequence.py)
- [src/incident_intelligence/cli/build_temporal_features.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/cli/build_temporal_features.py)
- [src/incident_intelligence/data/generate_sequence.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/data/generate_sequence.py)
- [src/incident_intelligence/data/temporal_features.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/data/temporal_features.py)
- [src/incident_intelligence/data/splitters.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/data/splitters.py)

## CLI Commands

The installed CLI entrypoints are defined in [pyproject.toml](/Users/swethachakravarthy/Projects/incident-intelligence/pyproject.toml).

| Command | Description |
| --- | --- |
| `incident-generate` | Generate snapshot data and train/val/eval splits |
| `incident-generate-sequence` | Generate raw incident sequences |
| `incident-build-temporal-features` | Convert raw sequences into temporal feature datasets |
| `incident-train` | Train baseline models |
| `incident-evaluate` | Evaluate saved models |
| `incident-explain` | Generate global explainability artifacts |
| `incident-explain-local` | Generate local explainability artifacts for selected incidents |
| `incident-pipeline` | Run the full snapshot or temporal workflow |

## Quickstart

### 1. Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Run the snapshot pipeline

```bash
incident-pipeline
```

### 3. Run the temporal pipeline

```bash
incident-pipeline --dataset-kind temporal
```

For headless environments, this is often the safest form:

```bash
MPLBACKEND=Agg incident-pipeline --dataset-kind temporal
```

## Recommended Commands

### Snapshot End To End

```bash
incident-pipeline
```

### Temporal End To End

```bash
incident-pipeline --dataset-kind temporal
```

### Temporal End To End With Faster Training

The temporal workflow can be slower because it runs cross-validated grid search across multiple models on a wider feature set. For quicker iteration:

```bash
MPLBACKEND=Agg incident-pipeline \
  --dataset-kind temporal \
  --fast-mode \
  --models logistic,rf \
  --n-jobs 1 \
  --cv 3 \
  --verbose 0
```

### Run Individual Temporal Stages

```bash
incident-generate-sequence
incident-build-temporal-features
incident-train --dataset-kind temporal
incident-evaluate --dataset-kind temporal
incident-explain --dataset-kind temporal
incident-explain-local --dataset-kind temporal
```

## Training Controls

Training supports configurable search behavior through [src/incident_intelligence/cli/train.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/cli/train.py) and [src/incident_intelligence/cli/pipeline.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/cli/pipeline.py).

Available options include:

- `--dataset-kind {snapshot,temporal}`
- `--cv`
- `--n-jobs`
- `--verbose`
- `--scoring`
- `--models`
- `--fast-mode`

Example:

```bash
incident-train \
  --dataset-kind temporal \
  --fast-mode \
  --models logistic,rf \
  --n-jobs 1 \
  --cv 3
```

Supported model aliases are:

- `logistic`
- `rf`
- `gb`
- `svm`

## Artifact Layout

Snapshot and temporal runs now write to separate default directories.

### Snapshot Defaults

```text
artifacts/models/
artifacts/metrics/
artifacts/plots/
artifacts/reports/
artifacts/explain/
```

### Temporal Defaults

```text
artifacts/models_temporal/
artifacts/metrics_temporal/
artifacts/plots_temporal/
artifacts/reports_temporal/
artifacts/explain_temporal/
```

Important temporal defaults include:

```text
artifacts/models_temporal/best_model.joblib
artifacts/metrics_temporal/train_val_results.json
artifacts/metrics_temporal/leaderboard_val.csv
artifacts/metrics_temporal/evaluation.json
artifacts/metrics_temporal/evaluation_summary.csv
```

This separation prevents snapshot and temporal runs from overwriting each other when you use the standard CLI defaults.

## Explainability

### Global Explainability

Global explainability writes model-level artifacts such as:

- SHAP-based summaries when supported
- permutation importance fallback outputs
- feature ranking summaries

Run it with:

```bash
incident-explain
incident-explain --dataset-kind temporal
```

### Local Explainability

Local explainability focuses on individual evaluation examples and can generate:

- waterfall plots
- JSON artifacts
- markdown RCA-style summaries

Run it with:

```bash
incident-explain-local --model artifacts/models/best_model.joblib
incident-explain-local --dataset-kind temporal
```

Temporal local explainability defaults to the temporal best model and temporal explain directory automatically.

## Configuration

Default CLI settings are stored in [pyproject.toml](/Users/swethachakravarthy/Projects/incident-intelligence/pyproject.toml) under:

- `[tool.incident_intelligence.generator]`
- `[tool.incident_intelligence.sequence_generator]`
- `[tool.incident_intelligence.temporal_features]`
- `[tool.incident_intelligence.train]`
- `[tool.incident_intelligence.evaluate]`
- `[tool.incident_intelligence.explain]`
- `[tool.incident_intelligence.explain_local]`

Config loading and CLI override behavior live in [src/incident_intelligence/config.py](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/config.py).

## Project Structure

```text
incident-intelligence/
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── docs/images/
├── generator_spec/
│   └── class_config.json
├── notebooks/
├── src/incident_intelligence/
│   ├── cli/
│   ├── data/
│   ├── modeling/
│   ├── config.py
│   ├── settings.py
│   └── __init__.py
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Main Code Areas

- CLI orchestration: [src/incident_intelligence/cli](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/cli)
- Data generation and temporal feature engineering: [src/incident_intelligence/data](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/data)
- Model training, evaluation, and explainability: [src/incident_intelligence/modeling](/Users/swethachakravarthy/Projects/incident-intelligence/src/incident_intelligence/modeling)

## Makefile Notes

The Makefile still provides convenience targets for the common workflow:

```bash
make install
make generate
make train
make evaluate
make explain
make explain-local
make pipeline
```

For the most up-to-date feature set, especially the temporal workflow and training controls, prefer the CLI commands directly.

Also note that `make clean` removes both `artifacts/` and `data/`.

## Reproducibility

The project uses fixed seeds in the default configs for repeatable dataset generation and model runs. You can override those values from the CLI or in [pyproject.toml](/Users/swethachakravarthy/Projects/incident-intelligence/pyproject.toml).

## Notebooks

The notebooks directory is intended for exploration and analysis rather than the canonical production workflow:

- [notebooks/01_data_generation.ipynb](/Users/swethachakravarthy/Projects/incident-intelligence/notebooks/01_data_generation.ipynb)
- [notebooks/02_eda.ipynb](/Users/swethachakravarthy/Projects/incident-intelligence/notebooks/02_eda.ipynb)
- [notebooks/03_baseline_model.ipynb](/Users/swethachakravarthy/Projects/incident-intelligence/notebooks/03_baseline_model.ipynb)
- [notebooks/04_model_explainability.ipynb](/Users/swethachakravarthy/Projects/incident-intelligence/notebooks/04_model_explainability.ipynb)

The CLI is the source of truth for the current end-to-end workflow.
