# SIH PS 26027 — AI-Powered Automatic Block Planning

Maximise asset availability for train operations on Indian Railways using ML-predicted maintenance priorities and constraint-based optimal scheduling.

> **Note**: All datasets are **synthetic**, generated for the Smart India Hackathon demonstration. They do not represent actual Indian Railways infrastructure or operations.

## Architecture

```
Processed Data
     ↓
Feature Engineering
     ↓
XGBoost (Priority Prediction)
     ↓
Maintenance Priority Score
     ↓
OR-Tools CP-SAT (Optimal Scheduling)
     ↓
Optimal Maintenance Block Schedule
     ↓
FastAPI Backend
     ↓
Next.js Dashboard
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements-ml.txt
```

### Run the ML Pipeline

```bash
# Train XGBoost model
python -m src.ml.train

# Generate batch predictions
python -m src.ml.batch_predict
```

### Run the Optimizer

```bash
# Default settings (60s time limit)
python -m src.optimizer.main

# Custom settings
python -m src.optimizer.main --time-limit 120 --workers 8 --seed 42
```

### Run Tests

```bash
pytest tests/optimizer -v
```

## Output Files

### ML Outputs
| File | Description |
|------|-------------|
| `data/processed/maintenance_predictions.csv` | XGBoost priority predictions |
| `artifacts/models/maintenance_priority_xgb.json` | Trained model |

### Optimizer Outputs
| File | Description |
|------|-------------|
| `artifacts/optimization/optimal_schedule.csv` | Task-level schedule with block assignments |
| `artifacts/optimization/maintenance_blocks.csv` | Consolidated maintenance blocks |
| `artifacts/optimization/unscheduled_tasks.csv` | Tasks not scheduled, with reasons |
| `artifacts/optimization/optimization_summary.json` | Solver statistics and KPIs |
| `artifacts/optimization/solver_log.txt` | Human-readable solver log |
| `artifacts/optimization/validation_report.json` | Independent validation results |

## Project Structure

```
sih_99/
├── config/
│   ├── data_config.yaml
│   └── ml_config.yaml
├── data/
│   ├── raw/                          # Raw synthetic datasets
│   ├── processed/                    # Processed datasets
│   └── DATA_DICTIONARY.md
├── src/
│   ├── data/                         # Data generation & feature engineering
│   │   ├── feature_engineering.py
│   │   ├── generate_data.py
│   │   ├── preprocess.py
│   │   └── validate_data.py
│   ├── ml/                           # XGBoost ML pipeline
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── batch_predict.py
│   │   ├── evaluate.py
│   │   └── preprocessing.py
│   └── optimizer/                    # OR-Tools CP-SAT optimizer
│       ├── config.py                 # Centralized configuration
│       ├── data_loader.py            # CSV loading & validation
│       ├── preprocessing.py          # Time conversion & scaling
│       ├── model.py                  # CP-SAT model builder
│       ├── constraints.py            # All constraint families
│       ├── objective.py              # Composite objective function
│       ├── solver.py                 # Solver wrapper
│       ├── schedule.py               # Schedule & block extraction
│       ├── validate.py               # Independent validator
│       └── main.py                   # CLI entry point
├── tests/
│   └── optimizer/                    # Optimizer test suite
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   └── optimization/                # Optimizer outputs
├── docs/
│   └── optimizer.md                  # Technical documentation
└── requirements-ml.txt
```

## Documentation

- [Optimizer Technical Documentation](docs/optimizer.md)
- [Data Dictionary](data/DATA_DICTIONARY.md)

## Technology Stack

- **ML**: XGBoost, scikit-learn, pandas, numpy
- **Optimization**: Google OR-Tools CP-SAT
- **Testing**: pytest
- **Backend**: FastAPI (planned)
- **Frontend**: Next.js (planned)
- **Database**: PostgreSQL (planned)
