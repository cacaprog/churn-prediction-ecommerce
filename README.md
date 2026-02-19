# Olist Seller Churn Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-purple.svg)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

End-to-end machine learning pipeline to predict seller churn for the Olist e-commerce marketplace — enabling proactive retention strategies and measurable revenue protection.

---

## 🎯 Business Problem

Olist's marketplace depends on an active seller base. High churn erodes GMV and increases acquisition costs. This project builds a **dual-stage predictive system** that flags at-risk sellers before they churn, giving account managers a prioritised intervention list with estimated revenue impact.

### Key Results (842 sellers, Jun 2017 – Aug 2018)

| Metric | Value |
|--------|-------|
| Overall Churn Rate | **85.0%** |
| Never-Activated Sellers | **515 (61.2%)** — onboarding failure |
| Dormant Sellers | **201 (23.9%)** — retention failure |
| Active Sellers | **140 (16.6%)** |
| Revenue at Risk | **R$ 272,607** |
| High / Critical Risk Sellers | **669 (79.5%)** |
| Pre-Activation Model AUC | **0.975** (Logistic Regression) |
| Retention Model AUC | **0.706** (Gradient Boosting) |
| Active Sellers Targeted for Intervention | **33** |

---

## 🏗️ Architecture

```
Raw CSVs
   │
   ▼
DataLoader ──► DataPreprocessor ──► ChurnAnalyzer (labels + cohorts)
                                          │
                                          ▼
                                  FeatureEngineer
                                   /            \
                        Pre-Activation         Retention
                          Features              Features
                              │                    │
                              ▼                    ▼
                          ChurnModeler ──────► ChurnModeler
                       (LogReg / RF / GBM)  (LogReg / RF / GBM)
                              │                    │
                              └────────┬───────────┘
                                       ▼
                                 ModelEvaluator
                              (ROC · PR · CM · FI)
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                   Risk Scoring              InsightsReporter
                (overall_churn_risk)    (churn_insights_report.md)
                          │
                          ▼
               InterventionPrioritizer
              (intervention_priority_list.csv)
                          │
                          ▼
               📊 Interactive Dashboard
              (dashboard/index.html)
```

---

## 🚀 Quick Start

### Option A — Local (uv)

```bash
# 1. Clone and enter the project
git clone <repo>
cd olist-ecommerce

# 2. Copy and configure environment variables
cp .env.example .env   # set DATA_PATH to your Olist CSV folder

# 3. Install dependencies (uv required)
make install

# 4. Run the full pipeline
make run-pipeline

# 5. Generate the interactive dashboard
make dashboard         # → opens dashboard/index.html

# 6. View generated reports
ls outputs/            # seller_master, risk_scores, segments, cohorts
ls outputs/figures/    # ROC, PR, confusion matrix, feature importance
```

### Option B — Docker

```bash
# 1. Clone and configure env (same as above)
git clone <repo>
cd olist-ecommerce
cp .env.example .env

# 2. Build the image (one-time, ~2 min)
make docker-build

# 3. Run the full training pipeline
make docker-pipeline

# 4. Launch the MLflow UI → http://localhost:5000
make docker-mlflow
```

> Raw CSVs are read from `./data/raw/` on your host via a bind mount — no copying into the image required.

---

## 🐳 Docker

The project ships a **multi-stage `Dockerfile`** and a **`docker-compose.yml`** that orchestrates three independent services.

### How it works

```
┌─────────────────────────────────────────────────────────┐
│  builder stage (python:3.10-slim + build-essential)      │
│   └─ uv sync → resolves & installs deps into .venv       │
└───────────────────────────┬─────────────────────────────┘
                            │  COPY .venv only
┌───────────────────────────▼─────────────────────────────┐
│  runtime stage (python:3.10-slim, no compiler tools)     │
│   ├─ runs as non-root user (appuser)                     │
│   ├─ bind mount: ./data  → /app/data  (host CSVs)        │
│   └─ named volumes: outputs · models · mlruns · logs     │
└─────────────────────────────────────────────────────────┘
```

### Services

| Service | Profile | Description |
|---------|---------|-------------|
| `pipeline` | `pipeline` | Runs the full training pipeline |
| `inference` | `inference` | Scores sellers with saved models |
| `mlflow` | `mlflow` | Experiment tracking UI on port 5000 |

Services use **Compose profiles** — nothing starts by default. Activate the one you need.

### Commands

```bash
make docker-build      # Build the image (needed once, or after dep changes)
make docker-pipeline   # Train — reads ./data/raw, writes to volumes
make docker-inference  # Score — uses saved models from the models volume
make docker-mlflow     # Start MLflow UI → http://localhost:5000
make docker-down       # Stop & remove containers
make docker-clean      # ⚠ Remove containers, volumes AND the image
```

### When to rebuild

You only need `make docker-build` after:
- Changing `Dockerfile`
- Updating `pyproject.toml` or `uv.lock` (dependency changes)
- Modifying files in `src/`, `scripts/`, or `config/`

Changes to `docker-compose.yml` **never** require a rebuild.

---

## 📊 Interactive Dashboard

After running the pipeline, generate a self-contained stakeholder dashboard:

```bash
make dashboard
# → dashboard/index.html
```

The dashboard is a **single HTML file** (no server required) with five tabs:

| Tab | Contents |
|-----|----------|
| 🏠 **Overview** | 6 KPI cards · key insight pills · status & risk donuts |
| 📅 **Cohort Analysis** | Monthly churn vs activation trend · GMV by cohort |
| 🗂️ **Segmentation** | Churn by business segment · lead type · behaviour profile · state |
| 🤖 **Model Performance** | Metric bars per model · cross-model comparison chart · pipeline architecture |
| 🎯 **Interventions** | Searchable & filterable priority table for 30 high-risk sellers |

Open `dashboard/index.html` in any browser — no dependencies, no server.

---

## 📁 Project Structure

```
olist-ecommerce/
├── 📁 config/
│   └── settings.py              # Centralised config via pydantic-settings (.env)
├── 📁 src/                      # Core library — importable modules
│   ├── pipeline.py              # End-to-end orchestrator + all pipeline classes
│   ├── features.py              # Feature engineering (pre-activation + retention)
│   ├── models.py                # Model training (LogReg, RF, GBM comparison)
│   ├── evaluation.py            # Evaluation + chart generation → outputs/figures/
│   ├── reports.py               # Stakeholder Markdown report generation
│   └── validation/
│       └── schemas.py           # Pydantic data validation schemas
├── 📁 scripts/
│   ├── run_pipeline.py          # Entry point: runs src/pipeline.main()
│   └── generate_dashboard.py    # Reads outputs/ CSVs → dashboard/index.html
├── 📁 dashboard/                # Dashboard source (index.html is gitignored)
│   └── template.html            # HTML/JS/CSS template (Chart.js, dark theme)
├── 📁 notebooks/
│   └── poc_churn_analysis.ipynb # Exploratory analysis
├── 📁 tests/
│   └── test_pipeline.py         # Unit tests
├── 📁 docs/                     # Project documentation
│   ├── dataset.md               # Dataset description
│   ├── glossary.md              # Domain glossary
│   ├── project.md               # Architecture & design notes
│   └── poc.md                   # POC findings
├── 📁 outputs/                  # ← Generated on pipeline run (gitignored)
│   ├── figures/                 # Charts: ROC, PR, confusion matrix, feature importance
│   ├── seller_master.csv        # Full seller dataset with churn labels
│   ├── seller_risk_scores.csv   # Per-seller risk scores
│   ├── cohort_analysis.csv      # Monthly cohort stats
│   ├── segment_analysis_*.csv   # Churn by segment, lead type, state, profile
│   ├── intervention_priority_list.csv
│   └── analysis_summary.txt
├── 📁 models/                   # ← Generated on pipeline run (gitignored)
│   ├── pre_activation_model.joblib
│   └── retention_model.joblib
├── 📁 data/                     # (gitignored) — place Olist CSVs here
│   └── raw/
├── .env.example                 # Required environment variables template
├── Dockerfile                   # Multi-stage image (builder → runtime)
├── docker-compose.yml           # pipeline · inference · mlflow services
├── .dockerignore                # Keeps build context lean
├── Makefile                     # Developer shortcuts (local + Docker)
└── requirements.txt             # Pinned dependencies
```

---

## 🤖 Model Details

### Stage 1 — Pre-Activation Model
Predicts whether a newly-onboarded seller will **never make a sale** (61% of the dataset).

| Model | AUC-ROC | Accuracy | F1 |
|-------|---------|----------|----|
| **Logistic Regression** ✅ | **0.975** | 0.930 | 0.944 |
| Random Forest | 0.962 | 0.937 | 0.947 |
| Gradient Boosting | 0.953 | 0.918 | 0.933 |

### Stage 2 — Retention Model
Predicts whether an **activated seller** will go dormant (60+ days without an order).

| Model | AUC-ROC | Accuracy | F1 |
|-------|---------|----------|----|
| Logistic Regression | 0.647 | 0.597 | 0.638 |
| Random Forest | 0.673 | 0.597 | 0.638 |
| **Gradient Boosting** ✅ | **0.706** | 0.645 | 0.694 |

> Charts for each model (ROC curve, Precision-Recall, Confusion Matrix, Feature Importance) are saved to `reports/figures/` on every run.

---

## 🛠️ Developer Commands

### Local

```bash
make install           # Install dependencies via uv
make run-pipeline      # Run the full end-to-end pipeline
make run-inference     # Score sellers using saved models
make dashboard         # Generate dashboard/index.html from latest outputs
make test              # Run unit tests
make test-coverage     # Run tests with HTML coverage report
make lint              # flake8 + black + isort + bandit checks
make format            # Auto-format with black + isort
make ci-check          # Full local CI simulation (lint → typecheck → tests)
make pre-commit-run    # Run all pre-commit hooks over the codebase
make mlflow-ui         # Open MLflow UI at http://localhost:5000
make clean             # Remove __pycache__, .pytest_cache, htmlcov
make clean-outputs     # Remove generated CSVs, reports, and figures
make setup-dirs        # Create required directories from scratch
```

### Docker

```bash
make docker-build      # Build the Docker image
make docker-pipeline   # Run training pipeline in a container
make docker-inference  # Run inference in a container
make docker-mlflow     # Start MLflow UI at http://localhost:5000
make docker-down       # Stop & remove containers
make docker-clean      # ⚠ Remove containers, volumes AND the image
```

---

## ⚙️ Configuration

All settings are managed through `config/settings.py` and read from `.env`.
No hardcoded paths anywhere in the codebase.

```bash
# .env
DATA_PATH=./data/raw          # Olist raw CSV files
OUTPUT_PATH=./outputs         # Generated CSVs and text files
MODELS_PATH=./models          # Saved .joblib models
FIGURES_PATH=./outputs/figures # Charts and plots
```

See `.env.example` for the full list of configurable values.

---

## 🧪 Testing

```bash
make test             # Run all unit tests
make test-coverage    # Run with coverage (open htmlcov/index.html)
```

---

## 📧 Contact

Cairo Cananea
- Blog: [cairocananea.com.br](https://cairocananea.com.br)
- Linkedin: [Cairo Cananea](https://www.linkedin.com/in/cairocananea/)
- Github: [Cairo Cananea](https://github.com/cacaprog)
