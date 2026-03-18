# Incident Intelligence

Incident Intelligence is an end-to-end machine learning pipeline for **incident root cause classification** using operational telemetry signals such as CPU usage, memory growth, request rate, and service latency.

The project demonstrates a full ML workflow including:

- Synthetic incident data generation
- Exploratory data analysis (EDA)
- Model training and evaluation
- Model explainability

The goal is to automatically identify the **underlying cause of production incidents** using system metrics.

---

## Key Features

- Synthetic incident simulation engine for generating labeled telemetry data
- Modular ML pipeline with CLI-based execution
- Multiple classification models with automated comparison
- Automated evaluation outputs including metrics, confusion matrices, comparison plots, and classification reports
- Model explainability using global feature importance and **local incident analysis**
- Fully reproducible dataset generation and training pipeline

---

## Command Line Interface

The project exposes CLI commands for running the pipeline:

| Command                  | Description                      |
| ------------------------ | -------------------------------- |
| `incident-generate`      | Generate synthetic dataset       |
| `incident-train`         | Train candidate models           |
| `incident-evaluate`      | Evaluate trained models          |
| `incident-explain`       | Generate global explainability   |
| `incident-explain-local` | Generate local incident analysis |
| `incident-pipeline`      | Run the full pipeline            |

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [ML Pipeline](#ml-pipeline)
- [Synthetic Dataset](#synthetic-dataset)
- [Modeling Pipeline](#modeling-pipeline)
- [Model Evaluation](#model-evaluation)
- [Model Explainability](#model-explainability)
  - [Local Incident Root Cause Analysis](#local-incident-root-cause-analysis)
- [Quickstart](#quickstart)
- [Running the Pipeline](#running-the-pipeline)
- [Project Structure](#project-structure)
- [Models Evaluated](#models-evaluated)
- [Model Performance](#model-performance)
- [Generated Outputs](#generated-outputs)
- [Feature Dictionary](#feature-dictionary)
- [Notebooks](#notebooks)
- [Reproducibility](#reproducibility)
- [Development Notes](#development-notes)
- [Future Improvements](#future-improvements)

---

## Architecture Overview

The project follows a modular ML pipeline architecture:

- **Config layer** – experiment and dataset configuration ([config/](config/))
- **Core ML logic** – reusable modules ([src/incident_intelligence/](src/incident_intelligence/))
- **CLI entrypoints** – command-line tools for dataset generation, training, evaluation, and explainability ([src/incident_intelligence/cli/](src/incident_intelligence/cli/))
- **Artifacts** – generated models, metrics, plots, reports, and explainability outputs ([artifacts/](artifacts/))

This separation allows the pipeline to be executed via CLI, automated workflows, or orchestration systems.

---

## ML Pipeline

The repository implements a full machine learning lifecycle for incident root cause classification, from synthetic data generation to model training, evaluation, and explainability.

<img src="docs/images/ML-pipeline-RCA.png" width="600" alt="ML Pipeline">

_Figure: High-level machine learning pipeline for incident root cause classification._

The ML pipeline consists of the following stages:

1. Synthetic dataset generation
2. Dataset splitting (train / validation / evaluation)
3. Model training with hyperparameter tuning
4. Model evaluation with saved plots and reports
5. Model explainability

---

## Synthetic Dataset

This project uses a **synthetically generated incident dataset**.

The dataset simulates common production failure scenarios using configurable statistical rules.

<img src="docs/images/synthetic_incident_pipeline.png" width="600" alt="Synthetic Incident Generation Pipeline">

The generator simulates operational telemetry signals such as:

- CPU usage
- memory growth
- request throughput
- service latency
- dependency failures
- error rates

Each synthetic incident is assigned a root cause label and telemetry metrics are generated according to
statistical distributions defined in:

[`generator_spec/class_config.json`](generator_spec/class_config.json).

This file defines simulation rules for each root cause category.

### Synthetic Incident Generation Process

1. Load root-cause configuration rules
2. Sample a root-cause category
3. Generate baseline system metrics
4. Apply class-specific metric perturbations
5. Simulate metric dependencies (CPU, memory, latency)
6. Generate log signals (OOM events, timeout logs)
7. Produce the final dataset and perform stratified splitting

Simulation logic lives in: [`src/incident_intelligence/data/`](src/incident_intelligence/data/)

### Dataset Output

Generated files:

```text
data/raw/incidents_raw.csv
```

The raw dataset in `data/raw/incidents_raw.csv` is split into:

- Training set `data/processed/incidents_root_cause_train.csv`
- Validation set `data/processed/incidents_root_cause_val.csv`
- Evaluation set `data/processed/incidents_root_cause_eval.csv`

Each row represents a synthetic incident snapshot.

> ⚠️ **Important**: All data is synthetic and contains **no production telemetry or PII**

---

## Modeling Pipeline

After dataset generation, the modeling stage trains multiple classification models.

<img src="docs/images/modeling_pipeline.png" width="600" alt="Modeling Pipeline">

The training workflow includes:

1. Feature preprocessing and scaling when required by the model
2. Training multiple classification algorithms
3. Hyperparameter tuning via cross-validation
4. Preliminary validation based model comparison
5. Best model selection

The best performing model is saved as:

```text
artifacts/models/best_model.joblib
```

The trained candidate models are then passed to the evaluation stage for comparison across validation and evaluation datasets.

---

## Model Evaluation

After training candidate models, their performance is evaluated on the validation and evaluation datasets.

<img src="docs/images/model_evaluation_pipeline.png" width="600" alt="Model Evaluation Pipeline">

The evaluation stage measures model performance using several standard classification metrics:

- Accuracy
- Precision (macro)
- Recall (macro)
- F1 Score (macro)
- ROC-AUC (if applicable)

### Evaluation Outputs

The evaluation pipeline now saves both machine-readable metrics and human-readable artifacts automatically.

Generated evaluation artifacts include:

```text
artifacts/
├── metrics/
│   ├── evaluation.json           # detailed evaluation metrics
│   ├── evaluation_summary.csv    # evaluation summary metrics
│   ├── leaderboard_val.csv       # validation leaderboard
│   └── train_val_results.json
├── plots/
│   ├── confusion_matrix_<model>.png    # per-model confusion matrix plots
│   ├── feature_importance_<model>.png. # per-model feature-importance plots when supported
│   └── model_comparison.png            # model comparison chart
└── reports/
    └── <model>_classification_report.md.  # per-model classification reports
```

Example Confusion Matrix:

<img src="docs/images/confusion_matrix.png" width="500" alt="Confusion Matrix">

> ⚠️ **Note**
> Feature-importance plots are only generated for models that expose importances or coefficients. For example, tree-based models and logistic regression are supported, while SVM (RBF) is skipped.

---

## Model Explainability

Explainability artifacts help interpret model predictions and identify important telemetry signals.

Outputs include:

- Global feature importance visualizations
- Feature ranking tables
- Explainability summary reports

Example:

<img src="docs/images/sample_global_importance.png" width="600" alt="Feature Importance Example">

### Local Incident Root Cause Analysis

Local explainability analyzes **individual incident predictions**.

Artificats generated:

- SHAP waterfall plots
- Root cause probability ranking
- Feature contribution breakdown
- HTML incident analysis reports

Example output directory:

```text
artifacts/explain/<model_name>/local/
```

Example files:

```text
row_12_waterfall.png
row_12_report.html
local_explainability_index.html
local_explainability_summary.json
```

Open the following file in a browser:

```text
artifacts/explain/<model_name>/local/local_explainability_index.html
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/incident-intelligence.git
cd incident-intelligence
```

### 2. Create virtual environment (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install project

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Running the Pipeline

### CLI (Recommended)

Run the full pipeline

```bash
incident-pipeline
```

Run individual stages

```bash
incident-generate
incident-train
incident-evaluate
incident-explain
incident-explain-local --model artifacts/models/best_model.joblib
```

These commands correspond to modules under: [src/incident_intelligence/cli/](src/incident_intelligence/cli/)

### Makefile

Run full pipeline

```bash
make pipeline
```

Run individual steps

```bash
make generate
make train
make evaluate
make explain
make explain-local
```

**Note:** The Makefile internally calls the same CLI commands used above.

---

## Project Structure

```text
incident-intelligence/
├── artifacts/                      # Generated models, metrics, plots, reports, explainability assets
├── generator_spec/
│   └── class_config.json           # Synthetic incident simulation rules
├── data/                           # Generated synthetic data files
│   ├── raw/                        # Raw Data generated using simulation rules
│   │   └── incidents_raw.csv
│   └── processed/                  # Train/validation/eval splits of raw data
│       ├── incident_root_cause_train.csv
│       ├── incident_root_cause_val.csv
│       └── incident_root_cause_eval.csv
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_model_explainability.ipynb
├── src/incident_intelligence/
│   ├── cli/                        # CLI commands
│   ├── data/                       # Synthetic data generator
│   ├── modeling/                   # Training, evaluation, explainability
│   ├──  config.py                  # Config loading and CLI/TOML merge logic
│   ├── settings.py                 # Project paths and configuration helpers
│   └── __init__.py
├── pyproject.toml
├── requirements.txt
├── Makefile
└── README.md
```

---

## Models Evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF)

The best-performing model is automatically selected

---

## Model Performance

Example validation leaderboard (results will vary depending on generated data):

| Model               | Accuracy | Precision | Recall   | F1 Score |
| ------------------- | -------- | --------- | -------- | -------- |
| Logistic Regression | 0.82     | 0.81      | 0.80     | 0.80     |
| Random Forest       | 0.88     | 0.87      | 0.86     | 0.86     |
| Gradient Boosting   | **0.90** | **0.89**  | **0.89** | **0.89** |
| SVM (RBF)           | 0.87     | 0.86      | 0.85     | 0.85     |

---

## Generated Outputs

Pipeline outputs are written to:

```text
artifacts/
├── explain/          # Model explainability artifacts
├── metrics/          # Performance metrics and evaluation results
├── models/           # Trained model files
├── plots/            # Saved evaluation plots
└── reports/          # Saved per-model classification reports
```

Artifacts are generated locally and **not stored in version control**.

---

## Feature Dictionary

This section describes all input features and the target variable used for model training.

### Input Features

| Feature               | Type    | Description                                                                                                            |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `avg_cpu_usage`       | float   | Average CPU utilization (%) during the incident observation window. Higher values indicate CPU pressure or saturation. |
| `mem_growth`          | float   | Memory growth rate over the observation period. Elevated values may indicate memory leaks or memory pressure.          |
| `oom_log_count`       | integer | Count of out-of-memory (OOM) events logged during the incident window.                                                 |
| `request_rate`        | float   | Incoming request throughput (requests per unit time) observed during the incident.                                     |
| `error_rate`          | float   | Application/service error rate during the incident window.                                                             |
| `latency`             | float   | End-to-end service latency measured during the incident period.                                                        |
| `upstream_error_rate` | float   | Error rate from upstream/downstream dependencies affecting the service.                                                |
| `dependency_latency`  | float   | Latency contributed by dependent services or external systems.                                                         |
| `timeout_log_count`   | integer | Count of timeout-related log events during the incident window.                                                        |

### Target Variable

| Column             | Type        | Description                                                   |
| ------------------ | ----------- | ------------------------------------------------------------- |
| `root_cause_label` | categorical | Synthetic root-cause classification label (prediction target) |

### Possible Root Cause Labels

Based on the synthetic data generation, the following root cause categories are present:

- `memory_leak` - Memory-related issues causing gradual performance degradation
- `bad_deployment` - Issues introduced by recent code or configuration deployments
- `external_dependency_failure` - Failures caused by external service dependencies
- `cpu_saturation` - CPU resource exhaustion
- `traffic_spike` - Sudden increase in traffic causing system overload
- `normal` - No incident / normal operational state

---

## Notebooks

Exploratory notebooks are included for analysis and experimentation.

Recommended order:

- **[01_data_generation.ipynb](notebooks/01_data_generation.ipynb)**
- **[02_eda.ipynb](notebooks/02_eda.ipynb)**
- **[03_baseline_model.ipynb](notebooks/03_baseline_model.ipynb)**
- **[04_model_explainability.ipynb](notebooks/04_model_explainability.ipynb)**

---

## Reproducibility

The pipeline uses fixed random seeds for reproducibility.

Regenerate dataset:

```bash
incident-generate
```

Run full pipeline:

```bash
incident-pipeline
```

---

## Development Notes

- Prefer CLI commands or Makefile for reproducible workflows
- Use notebooks only for exploration and experimentation.
- Avoid committing generated artifacts to git

---

## Future Improvements

Potential enhancements include:

- Integration with real production telemetry datasets
- Automated hyperparameter tuning (Optuna / Ray Tune)
- Model monitoring and drift detection
- Real-time inference API deployment
- Integration with workflow orchestration tools (Airflow / Prefect)
