## 🎯 **MUST-HAVES (Professional Minimum Viable Product)**

### **Tier 1: Critical (Do This Week)**

#### **1. Configuration Management** ⭐ HIGHEST PRIORITY
**Why**: Hardcoded paths = broken pipeline on any other machine
**Time**: 2 hours

```python
# config/settings.py
from pydantic import BaseSettings, Field
from pathlib import Path
import os

class Settings(BaseSettings):
    """Centralized configuration - no more hardcoded paths."""
    DATA_PATH: Path = Field(..., env="DATA_PATH")
    OUTPUT_PATH: Path = Field(..., env="OUTPUT_PATH")
    MODELS_PATH: Path = Field(default=Path("./models"))
    
    # Business rules as config
    NEVER_ACTIVATED_DAYS: int = 90
    DORMANT_DAYS: int = 60
    RANDOM_STATE: int = 42
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# .env file (gitignored)
DATA_PATH=/home/cairo/data/raw
OUTPUT_PATH=/home/cairo/output
```

**Impact**: Pipeline runs anywhere, configurable without code changes

---

#### **2. Data Validation at Entry Points** ⭐ HIGHEST PRIORITY
**Why**: Silent failures in data = wrong predictions = bad business decisions
**Time**: 3 hours

```python
# src/validation/schemas.py
import pandera as pa
from pandera import Column, Check

seller_master_schema = pa.DataFrameSchema({
    "seller_id": Column(str, unique=True, nullable=False),
    "won_date": Column(pa.DateTime, nullable=False),
    "total_orders": Column(int, Check.greater_than_or_equal_to(0), nullable=True),
    "total_gmv": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    "churned": Column(int, Check.isin([0, 1]), nullable=True),
}, strict=True)

# In your pipeline
@pa.check_output(seller_master_schema)
def build_seller_master(self) -> pd.DataFrame:
    # existing code
```

**Impact**: Fail fast with clear errors, prevent garbage-in-garbage-out

---

#### **3. Proper Directory Structure** ⭐ HIGH PRIORITY
**Why**: Navigation chaos signals amateur work
**Time**: 1 hour

```
olist-churn-prediction/
├── 📁 data/                    # (gitignored)
│   ├── raw/
│   ├── processed/
│   └── outputs/
├── 📁 src/
│   ├── __init__.py
│   ├── config.py              # Settings management
│   ├── pipeline.py            # Your main logic (refactored)
│   ├── features.py            # Feature engineering
│   ├── models.py              # Model training
│   └── evaluation.py          # Metrics & validation
├── 📁 notebooks/
│   └── 01_exploratory_analysis.ipynb
├── 📁 tests/
│   └── test_pipeline.py       # At least one test file
├── 📁 reports/                # Generated outputs go here
├── 📄 .env.example            # Template for env vars
├── 📄 .gitignore              # Critical: don't commit data/models
├── 📄 README.md               # Professional documentation
├── 📄 requirements.txt        # Pinned dependencies
└── 📄 Makefile                # Common commands
```

---

#### **4. README.md with Professional Standards** ⭐ HIGH PRIORITY
**Why**: First impression for hiring managers/stakeholders
**Time**: 2 hours

```markdown
# Olist Seller Churn Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning pipeline to predict seller churn for Olist e-commerce marketplace, 
enabling proactive retention strategies and revenue protection.

## 🎯 Business Impact

- **Problem**: 35% seller churn rate costing R$ X million in annual GMV
- **Solution**: Dual-stage ML model (Pre-activation + Retention) with 0.85 AUC
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
| Overall Churn Rate | 35.2% |
| Pre-activation Model AUC | 0.87 |
| Retention Model AUC | 0.83 |
| Revenue at Risk (High/Critical) | R$ 2.4M |

## 📁 Project Structure

... (directory tree)

## 🧪 Testing

```bash
make test              # Run unit tests
make test-integration  # Run integration tests
make lint             # Code quality checks
```

## 📝 Citation

If using this code, please cite:
```
Cananea, C. (2024). Olist Seller Churn Prediction. 
https://github.com/cairocananea/olist-churn
```

## 📧 Contact

Cairo Cananea - cairocananea.com.br
```

---

### **Tier 2: Important (Do This Month)**

#### **5. Requirements.txt with Pinned Versions**
**Why**: "Works on my machine" is not professional
**Time**: 30 minutes

```txt
# requirements.txt (generated via pip freeze)
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
pandera==0.17.2
pydantic==2.0.3
python-dotenv==1.0.0
joblib==1.3.1
matplotlib==3.7.2
seaborn==0.12.2
pytest==7.4.0
```

---

#### **6. Basic Testing (Minimum Viable)**
**Why**: Shows you understand software engineering practices
**Time**: 3 hours

```python
# tests/test_pipeline.py
import pytest
import pandas as pd
from src.pipeline import ChurnAnalyzer
from src.config import Settings

class TestChurnDefinitions:
    """Test business logic correctness."""
    
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'seller_id': ['A', 'B', 'C', 'D'],
            'total_orders': [0, 5, 10, 3],
            'days_to_first_sale': [100, 5, None, 45],
            'days_since_last_sale': [None, 90, 30, 70],
            'total_gmv': [0, 1000, 5000, 800]
        })
    
    def test_never_activated_logic(self, sample_data):
        """Sellers with no orders or >90 days to first sale = churned."""
        analyzer = ChurnAnalyzer(sample_data, Settings())
        result = analyzer.define_churn_labels(sample_data)
        
        assert result.loc[result['seller_id'] == 'A', 'never_activated'].iloc[0] == 1
        assert result.loc[result['seller_id'] == 'B', 'never_activated'].iloc[0] == 0
    
    def test_dormant_logic(self, sample_data):
        """Active sellers with >60 days since last sale = dormant."""
        analyzer = ChurnAnalyzer(sample_data, Settings())
        result = analyzer.define_churn_labels(sample_data)
        
        # Seller B: has orders, last sale 90 days ago = dormant
        assert result.loc[result['seller_id'] == 'B', 'dormant'].iloc[0] == 1
        # Seller C: last sale 30 days ago = active
        assert result.loc[result['seller_id'] == 'C', 'dormant'].iloc[0] == 0

class TestDataValidation:
    """Test data quality checks."""
    
    def test_negative_gmv_raises_error(self):
        """Negative GMV should fail validation."""
        bad_data = pd.DataFrame({
            'seller_id': ['A'],
            'total_gmv': [-100]  # Invalid
        })
        
        with pytest.raises(Exception):
            validate_seller_data(bad_data)
```

---

#### **7. Makefile for Common Operations**
**Why**: Shows operational thinking, easy for others to use
**Time**: 1 hour

```makefile
.PHONY: install test run-pipeline clean lint

install:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ --cov=src --cov-report=html

run-pipeline:
	python -m src.pipeline

run-validation:
	python -m src.validation.validate_data

generate-reports:
	python -m src.reports.generate_insights

lint:
	flake8 src/ tests/
	black --check src/ tests/

format:
	black src/ tests/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/

setup-dirs:
	mkdir -p data/raw data/processed data/outputs models reports
```

---

#### **8. .gitignore (Critical)**
**Why**: Committing data/models is a security/cost mistake
**Time**: 15 minutes

```gitignore
# Data
data/
*.csv
*.parquet
*.xlsx

# Models
models/
*.joblib
*.pkl
*.h5

# Outputs
reports/figures/
reports/*.html
reports/*.pdf

# Environment
.env
.venv/
venv/
__pycache__/
*.pyc

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints
```

---

### **Tier 3: Nice to Have (Do If Time Permits)**

| Priority | Item | Time | Impact |
|----------|------|------|--------|
| 9 | MLflow for model tracking | 4h | Shows MLOps awareness |
| 10 | Docker containerization | 4h | True reproducibility |
| 11 | Pre-commit hooks (black, flake8) | 1h | Code quality automation |
| 12 | GitHub Actions CI | 2h | Automated testing |
