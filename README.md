# Incident Intelligence

Incident Intelligence is a machine learning pipeline for **incident root cause classification** using system telemetry signals such as CPU usage, memory growth, request rate, and latency.

The project demonstrates a complete ML workflow:

- Synthetic incident data generation
- Exploratory data analysis
- Baseline model training
- Model evaluation
- Model explainability

The goal is to automatically classify the **underlying cause of production incidents** from operational metrics.

---

## ML Pipeline

![Pipeline](docs/images/pipeline_diagram.png)

The repository implements a full ML lifecycle:

1. Synthetic dataset generation
2. Dataset splitting (train / validation / evaluation)
3. Model training with hyperparameter tuning
4. Model evaluation
5. Model explainability

---

## Dataset

The dataset contains simulated telemetry signals from a distributed service.

Features include:

| Feature | Description |
|------|------|
| avg_cpu_usage | CPU utilization |
| mem_growth | Memory growth rate |
| request_rate | Incoming request rate |
| latency | Request latency |
| dependency_latency | Upstream service latency |
| upstream_error_rate | Dependency error rate |
| error_rate | Application error rate |
| oom_log_count | Out-of-memory events |
| timeout_log_count | Timeout events |

Target variable: `root_cause_label`

Classes:

- `bad_deployment`
- `external_dependency_failure`
- `traffic_spike`
- `memory_leak`
- `cpu_saturation`
- `normal`

---

## Quickstart (recommended: CLI)

### 1) Create and activate a virtual environment (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Install the package in editable mode (required for CLI commands)

```bash
pip install -e .
```

### 4) Run the pipeline via CLI

```bash
incident-generate
incident-train
incident-eval
incident-explain
```

Or run end-to-end:

```bash
incident-pipeline
```

> **Note:** These CLI commands are defined in `pyproject.toml` under `[project.scripts]` and depend on
> modules in `incident_intelligence.cli.*`. If the CLI is not available, use one of the alternatives below.

---

## Alternative 1: Makefile

From the project root:

```bash
make help
make pipeline
```

You can also run steps individually:

```bash
make generate
make train
make evaluate
make explain
```

---

## Alternative 2: Run scripts directly

From the project root:

```bash
python scripts/generate_dataset.py
python scripts/train.py
python scripts/evaluate.py
python scripts/explain.py
```

Or run end-to-end:

```bash
python scripts/run_pipeline.py
```

---

## Project Structure

```text
incident-intelligence/
├── artifacts/                      # Pipeline outputs, logs, and generated assets
├── config/
│   └── class_config.json           # Class/label configuration
├── data/
│   ├── raw/                        # Raw input data
│   ├── processed/                  # Cleaned/transformed data
│   └── incident_root_cause_data.csv
├── models/                         # Saved trained models
├── notebooks/
│   ├── explainability_outputs/     # Explainability plots/tables
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_model_explainability.ipynb
├── scripts/
│   ├── generate_dataset.py         # Build dataset artifacts
│   ├── train.py                    # Train model(s)
│   ├── evaluate.py                 # Evaluate model(s)
│   ├── explain.py                  # Explainability artifacts (e.g., SHAP)
│   └── run_pipeline.py             # End-to-end pipeline runner
├── src/incident_intelligence/
│   ├── api/                        # API-related code
│   ├── cli/                        # CLI entry points (incident-* commands)
│   ├── data/                       # Data processing utilities
│   ├── modeling/                   # Modeling/training utilities
│   ├── __init__.py
│   └── settings.py                 # Central project settings
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Notebooks

- **01_data_generation.ipynb**  
  Creates/validates the working dataset from source inputs.

- **02_eda.ipynb**  
  Performs exploratory data analysis (distributions, missing values, trends, correlations).

- **03_baseline_model.ipynb**  
  Trains baseline models and compares initial performance.

- **04_model_explainability.ipynb**  
  Produces explainability outputs (feature importance and model interpretation artifacts).

Recommended order: **01 → 02 → 03 → 04**

---

## Source Code Organization

Inside `src/incident_intelligence`:

- `cli/`: CLI entry points for the `incident-*` commands (see `pyproject.toml`)
- `data/`: data ingestion, preprocessing, and feature utilities
- `modeling/`: training, inference, and evaluation logic
- `api/`: API-layer code (if serving predictions)
- `settings.py`: shared constants/config loading

---

## Inputs and Outputs

### Inputs

- `data/raw/`
- `data/incident_root_cause_data.csv`
- `config/class_config.json`

### Outputs

- `data/processed/`
- `models/`
- `artifacts/`
- `notebooks/explainability_outputs/`

---

## Notes

- Keep large datasets and model binaries out of git unless required.
- Update `config/class_config.json` and `src/incident_intelligence/settings.py` before running custom experiments.
- Prefer CLI or scripts for reproducibility; use notebooks for analysis and diagnostics.
