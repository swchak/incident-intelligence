# Incident Intelligence

An intelligent incident analysis system that automatically analyzes metrics and logs to suggest the most likely root causes to engineers when incidents occur.

## Overview

Incident Intelligence accelerates incident resolution by leveraging data analysis and machine learning to identify potential root causes. When an incident is detected, the system analyzes relevant metrics and logs to provide engineers with actionable insights and prioritized root cause suggestions.


## Features

- **Automatic Incident Analysis**: Analyzes metrics and logs in real-time when incidents are detected
- **Root Cause Suggestions**: Provides prioritized list of likely root causes based on data patterns
- **Fast Resolution**: Helps engineers quickly identify the source of problems
- **Data-Driven Insights**: Uses comprehensive metrics and log analysis for accurate suggestions

## Getting Started

### Prerequisites

- Python 3.8+
- [Additional dependencies to be added]

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/incident-intelligence.git
cd incident-intelligence

# Install dependencies
pip install -r requirements.txt
```

### Usage

[Usage instructions to be added]

## Data Generation

Since real datasets are not available for this project, we generate simulated data using the `generate.py` script. This allows us to create realistic incident scenarios and test the analysis engine.

To generate simulated data:

```bash
source venv/bin/activate
python data/generate.py
```

The script will create synthetic metrics and logs that simulate real incident patterns for testing and development. The generated data is saved to `incident_root_cause_data.csv`, which contains incident scenarios with associated metrics and root causes for training and evaluation.

The incident data is synthetically generated based on common production failure patterns with injected noise and partial overlap to reflect real-world ambiguity and imperfect labeling. This approach ensures the model learns to handle the complexity and uncertainty inherent in actual incident diagnosis scenarios.

### Data Validation

The `validation.py` script validates and analyzes the generated dataset to ensure data quality and integrity. It performs statistical analysis on the incident data by computing mean values of key metrics (CPU usage, error rate, request rate) grouped by root cause labels. This helps verify that the simulated data exhibits realistic patterns and correlations between metrics and their associated root causes.

To run validation:

```bash
python data/validation.py
```

## Contributing

[Contributing guidelines to be added]

## License

[License information to be added]