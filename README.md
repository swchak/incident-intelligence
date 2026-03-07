# Incident Intelligence

A comprehensive incident analysis and intelligence system that processes incident data, trains predictive models, and generates reports.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Notebook Guide](#notebook-guide)
- [Source Code Organization](#source-code-organization)
- [Getting Started](#getting-started)
- [Running Scripts and Pipelines](#running-scripts-and-pipelines)
- [Requirements](#requirements)

## Folder Structure

```
incident-intelligence/
├── artifacts/              # Generated model artifacts and outputs
├── config/                 # Configuration files
│   └── class_config.json   # Class configuration settings
├── data/                   # Data directory
│   ├── processed/          # Cleaned and processed datasets
│   ├── raw/                # Raw input data
│   └── incident_root_cause_data.csv  # Main incident dataset
├── models/                 # Trained model files
├── notebooks/              # Jupyter notebooks for analysis and development
│   ├── explainability_outputs/  # Generated explainability visualizations
│   ├── 01_data_generation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_model_explainability.ipynb
├── reports/                # Generated reports and visualizations
├── scripts/                # Executable Python scripts
│   ├── __pycache__/        # Python cache directory
│   ├── evaluate.py         # Model evaluation script
│   ├── generate_dataset.py # Dataset generation script
│   ├── Makefile            # Build and task automation
│   ├── run_pipeline.py     # Main pipeline orchestration
│   └── train.py            # Model training script
├── src/                    # Source code modules
│   └── incident_intelligence/
│       ├── api/            # API endpoints and handlers
│       ├── data/           # Data loading and processing utilities
│       ├── modeling/       # Model architecture and training logic
│       ├── __init__.py     # Package initialization
│       ├── settings.py     # Application settings and configuration
│       └── venv/           # Virtual environment (if present)
├── .gitignore              # Git ignore rules
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # This file
└── requirements.txt        # Python package dependencies
```

## Notebook Guide

All analysis and exploratory notebooks are located in the `notebooks/` directory:

| Notebook                        | Purpose                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------- |
| `01_data_generation.ipynb`      | Generate and prepare the incident dataset from raw sources                             |
| `02_eda.ipynb`                  | Exploratory Data Analysis - understand data distributions, patterns, and relationships |
| `03_baseline_model.ipynb`       | Build and evaluate baseline machine learning models                                    |
| `04_model_explainability.ipynb` | Analyze model interpretability and feature importance using SHAP/LIME                  |

**Subdirectories:**

- `explainability_outputs/` - Generated explainability visualizations and reports

**To view notebooks:**

```bash
jupyter notebook notebooks/
```

**Recommended workflow:**

1. Start with `01_data_generation.ipynb` to prepare data
2. Run `02_eda.ipynb` to explore data characteristics
3. Execute `03_baseline_model.ipynb` to train initial models
4. Review `04_model_explainability.ipynb` to understand model decisions

## Source Code Organization

### `src/incident_intelligence/`

The main source code is organized into the following modules:

- **`api/`** - REST API endpoints and request handlers
- **`data/`** - Data loading, processing, and feature engineering functions
- **`modeling/`** - Machine learning models, training logic, and evaluation utilities
- **`settings.py`** - Configuration management and environment variables
- **`__init__.py`** - Package initialization

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. Navigate to the project directory

```bash
cd /Users/swethachakravarthy/Projects/incident-intelligence
```

2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running Scripts and Pipelines

### 1. Generate Dataset

Prepare and generate the dataset:

```bash
python scripts/generate_dataset.py
```

### 2. Train Models

Train the machine learning models:

```bash
python scripts/train.py
```

### 3. Run Full Pipeline

Execute the complete pipeline (data processing, training, evaluation):

```bash
python scripts/run_pipeline.py
```

### 4. Evaluate Models

Evaluate trained models on test data:

```bash
python scripts/evaluate.py
```

### Using Makefile

If a Makefile exists, you can use:

```bash
make help           # View available commands
make train          # Run training
make evaluate       # Run evaluation
```

## Configuration

Configuration is managed through:

- **`config/class_config.json`** - Class and model configuration
- **`src/incident_intelligence/settings.py`** - Application settings

Update these files before running pipelines to customize behavior.

## Output Locations

- **Models**: `models/`
- **Processed Data**: `data/processed/`
- **Artifacts**: `artifacts/`
- **Explainability Outputs**: `notebooks/explainability_outputs/`

## Dependencies

All required packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

_Last updated: March 6, 2026_
