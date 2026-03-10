# Incident Intelligence

Incident Intelligence is an end-to-end machine learning pipeline for **incident root cause classification** using operational telemetry signals such as CPU usage, memory growth, request rate, and service latency.

The project demonstrates a full ML workflow including:

- Synthetic incident data generation
- Exploratory data analysis (EDA)
- Model training and evaluation
- Model explainability

The goal is to automatically identify the **underlying cause of production incidents** using system metrics.

---

## Architecture Overview

The project follows a modular ML pipeline architecture:

- **Config layer** – experiment and dataset configuration (`config/`)
- **Core ML logic** – reusable modules (`src/incident_intelligence/`)
- **Pipeline scripts** – CLI entrypoints for dataset generation, training, and evaluation (`scripts/`)
- **Artifacts** – generated models, metrics, and explainability outputs (`artifacts/`)

This separation allows the pipeline to be executed via CLI, automated workflows, or orchestration systems.

---

## ML Pipeline

![Pipeline](docs/images/pipeline_diagram.png)

The repository implements a full ML lifecycle:

1. Synthetic dataset generation
2. Dataset splitting (train / validation / test)
3. Model training with hyperparameter tuning
4. Model evaluation
5. Model explainability

---

## Dataset

This project uses a **synthetically generated** incident dataset for development, testing, and evaluation.

The dataset is generated using a configurable simulation framework that models common production failure patterns such as

- traffic spikes
- memory leaks
- CPU saturation,
- Dependency failures.

Dataset generation is controlled by:

- Simulation logic in `src/incident_intelligence/data/`
- Generation parameters defined in `config/class_config.json`

### Output Dataset

The generated dataset is saved to:

- **File**: `data/raw/incidents_raw.csv`
- **Format**: CSV (UTF-8 encoded)
- **Granularity**: One row per synthetic incident snapshot
- **Privacy**: Contains **no production data or PII**

> ⚠️ **Important**: All data is synthetically generated to simulate realistic incident patterns. This is not real production data.

### Data Split

The output dataset in incidents_raw.csv is split into:

- Training set
- Validation set
- Test set (optional)

Full column descriptions are provided in the [Feature Dictionary](#feature-dictionary) section below.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/incident-intelligence.git
cd incident-intelligence
```

### 2. Create and activate a virtual environment (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Run the pipeline

```bash
incident-pipeline
```

This command will:

- Generate the synthetic dataset
- Train the models
- Evaluate model performance
- Produce explainability artifacts

After completion you should see generated outputs in:

```bash
artifacts/
    models/
    metrics/
    explain/
```

---

## Running the Pipeline

The pipeline can be executed via **CLI commands (recommended)**, a **Makefile**, or **direct scripts**.

### Option 1 - CLI Commands (Recommended)

#### 1. Install the project in editable mode

Run from project root

```bash
pip install -e .
```

#### 2. Run the full pipeline

```bash
incident-pipeline
```

#### 3. Run individual stages

```bash
incident-generate   # dataset only
incident-train      # train only
incident-eval       # evaluate only
incident-explain    # explainability only
```

> Note: If the CLI commands are not available, ensure you ran `pip install -e .` successfully.
> The CLI entry points expect modules under `incident_intelligence.cli.*`. If you prefer not to use the CLI, use one of the alternatives below.

### Option 2 - Makefile

#### 1. Run the full pipeline

```bash
make pipeline  # run full pipeline
```

#### 2. Run individual steps

```bash
make generate
make train
make evaluate
make explain
```

### Option 3 - Run Scripts Directly

#### 1. Run the full pipeline

```bash
python scripts/run_pipeline.py
```

#### 2. Run individual steps

```bash
python scripts/generate_dataset.py
python scripts/train.py
python scripts/evaluate.py
python scripts/explain.py
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
│   ├── data/                       # Data processing utilities
│   ├── modeling/                   # Modeling/training utilities
│   ├── __init__.py
│   └── settings.py                 # Central project settings
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Models Evaluated

The training pipeline evaluates multiple classification algorithms:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (RBF)

Each model is trained using a standardized preprocessing pipeline and evaluated using validation metrics.

The best-performing model is automatically selected and saved as:

`artifacts/models/best_model.joblib`

**Note:** All models are trained using the training dataset and evaluated on a validation set to ensure fair comparison.

---

## Generated Outputs

This section describes all artifacts generated during model training and evaluation. These files are created in the `artifacts/` directory when you run the training pipeline.

### Directory Structure

```
artifacts/
├── explain/          # Model explainability artifacts
├── metrics/          # Performance metrics and evaluation results
└── models/           # Trained model files
```

### Explainability Artifacts (`artifacts/explain/`)

Visual explanations and feature importance analysis for different models.

#### Sample Output

Below is an example of the global feature importance visualization generated for each model:

![Global Feature Importance Example](docs/images/sample_global_importance.png)

_Example: Global feature importance showing the top features ranked by their contribution to model predictions_

The visualizations show:

- **Feature Names**: Input variables from the dataset (e.g., `avg_cpu_usage`, `mem_growth`)
- **Importance Scores**: Quantitative measure of each feature's impact on predictions
- **Ranked Display**: Features ordered from most to least important

#### Generated Files

| Model               | Global Importance Plot                               | CSV Data                                             |
| ------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| Best Model          | `best_model_global_importance.png`                   | `best_model_global_importance.csv`                   |
| Gradient Boosting   | `Gradient_Boosting_pipeline_global_importance.png`   | `Gradient_Boosting_pipeline_global_importance.csv`   |
| Logistic Regression | `Logistic_Regression_pipeline_global_importance.png` | `Logistic_Regression_pipeline_global_importance.csv` |
| Random Forest       | `Random_Forest_pipeline_global_importance.png`       | `Random_Forest_pipeline_global_importance.csv`       |
| SVM (RBF)           | `SVM_(RBF)_pipeline_global_importance.png`           | `SVM_(RBF)_pipeline_global_importance.csv`           |

**Additional Files:**

- `explainability_summary.json` - Summary of explainability metrics across all models

### Model Performance Metrics (`artifacts/metrics/`)

Performance evaluation files generated during training:

| File                     | Description                                    |
| ------------------------ | ---------------------------------------------- |
| `baseline_metrics.json`  | Baseline model performance metrics             |
| `evaluation_summary.csv` | Summary of all model evaluations               |
| `evaluation.json`        | Detailed evaluation metrics in JSON format     |
| `leaderboard_val.csv`    | Validation leaderboard comparing all models    |
| `train_val_results.json` | Training and validation results for all models |

### Trained Models (`artifacts/models/`)

All trained models saved in joblib format for easy deployment:

- `best_model.joblib` - Best performing model from AutoML
- `Gradient_Boosting_pipeline.joblib` - Gradient Boosting classifier pipeline
- `Logistic_Regression_pipeline.joblib` - Logistic Regression classifier pipeline
- `Random_Forest_pipeline.joblib` - Random Forest classifier pipeline
- `SVM_(RBF)_pipeline.joblib` - Support Vector Machine with RBF kernel pipeline

**Note:** Model artifacts are not checked into version control due to file size. Run the training pipeline to generate them locally.

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

### CSV Header Format

```csv
avg_cpu_usage,mem_growth,oom_log_count,request_rate,error_rate,latency,upstream_error_rate,dependency_latency,timeout_log_count,root_cause_label
```

### Sample Data Row

```csv
87.00089104373197,2.504895389677996,4,549.031934807165,2.0242063636527776,441.29667286645815,1.7617296375792422,419.79306285037956,2,memory_leak
```

---

## Notebooks

- **01_data_generation.ipynb** Creates/validates the working dataset from source inputs.

- **02_eda.ipynb** Performs exploratory data analysis (distributions, missing values, trends, correlations).

- **03_baseline_model.ipynb** Trains baseline models and compares initial performance.

- **04_model_explainability.ipynb** Produces explainability outputs (feature importance and model interpretation artifacts).

Recommended order: **01 → 02 → 03 → 04**

---

## Reproducibility

- The pipeline uses fixed random seeds where possible to ensure reproducible results.
- Synthetic dataset generation and model training both use deterministic seeds.
- To regenerate the dataset: incident-generate
- Running the pipeline multiple times with the same configuration should produce similar model results.

---

## Development Notes

- Keep large datasets and model binaries out of git unless required.
- Update `config/class_config.json` and `src/incident_intelligence/settings.py` before running custom experiments.
- Prefer CLI / scripts / Makefile for reproducibility; use notebooks for analysis and diagnostics.

---

## Future Improvements

Potential enhancements include:

- Integration with real production telemetry datasets
- Automated hyperparameter tuning (Optuna / Ray Tune)
- Model monitoring and drift detection
- Real-time inference API deployment
- Integration with workflow orchestration tools (Airflow / Prefect)
