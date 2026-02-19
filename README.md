# Olist Seller Churn Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning pipeline to predict seller churn for Olist e-commerce marketplace, enabling proactive retention strategies and revenue protection.

## 🎯 Business Impact

- **Problem**: High seller churn rate costing significant GMV loss
- **Solution**: Dual-stage ML model (Pre-activation + Retention) with robust AUC performance
- **Outcome**: Early identification of at-risk sellers enables targeted interventions

## 🏗️ Architecture

```
[Raw Data] → [Validation] → [Feature Engineering] → [Dual Models] → [Risk Scoring] → [Intervention List]
                ↓                    ↓                      ↓              ↓                ↓
           Schema Checks       Feature Store          MLflow Registry   Thresholds     Business Report
```

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repo>
cd olist-churn-prediction
cp .env.example .env  # Edit with your paths

# 2. Install dependencies
make install

# 3. Run full pipeline
make run-pipeline

# 4. View outputs
open reports/churn_insights_report.md
```

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Overall Churn Rate | ~35% |
| Pre-activation Model AUC | ~0.87 |
| Retention Model AUC | ~0.83 |
| Revenue at Risk (High/Critical) | Significant |

## 📁 Project Structure

```
olist-churn-prediction/
├── 📁 data/                    # (gitignored)
│   └── raw/
├── 📁 config/
│   └── settings.py            # Configuration management
├── 📁 src/
│   ├── __init__.py
│   ├── pipeline.py            # Main pipeline logic
│   ├── features.py            # Feature engineering
│   ├── models.py              # Model training
│   ├── evaluation.py          # Metrics & validation
│   └── validation/
│       └── schemas.py         # Data validation schemas
├── 📁 notebooks/              # Jupyter notebooks for exploration
├── 📁 tests/                  # Unit tests
│   └── test_pipeline.py
├── 📁 reports/                # Generated outputs
├── 📄 .env.example            # Template for env vars
├── 📄 .gitignore              # Critical: don't commit data/models
├── 📄 README.md               # This file
├── 📄 requirements.txt        # Pinned dependencies
└── 📄 Makefile                # Common commands
```

## 🧪 Testing

```bash
make test              # Run unit tests
make test-coverage     # Run tests with coverage
make lint              # Code quality checks
```

## 📝 Citation

If using this code, please cite:
```
Cananea, C. (2024). Olist Seller Churn Prediction. 
https://github.com/cairocananea/olist-churn
```

## 📧 Contact

Cairo Cananea - cairocananea.com.br
