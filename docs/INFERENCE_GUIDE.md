# Running Models on New Data — Inference Guide

> **Audience:** Data scientists, ML engineers, or anyone who needs to generate churn risk scores for a fresh batch of Olist sellers without re-training the models.

---

## Table of Contents

1. [Overview](#1-overview)
2. [What "New Data" Means Here](#2-what-new-data-means-here)
3. [Prerequisites](#3-prerequisites)
4. [Anatomy of the Trained Models](#4-anatomy-of-the-trained-models)
5. [Step-by-Step: Preparing the New Dataset](#5-step-by-step-preparing-the-new-dataset)
6. [Step-by-Step: Running Inference](#6-step-by-step-running-inference)
7. [Complete Inference Script](#7-complete-inference-script)
8. [Loading Models from MLflow](#8-loading-models-from-mlflow)
9. [Understanding the Output](#9-understanding-the-output)
10. [Common Pitfalls & Troubleshooting](#10-common-pitfalls--troubleshooting)
11. [When to Re-Train](#11-when-to-re-train)

---

## 1. Overview

The pipeline trains **two complementary models** that work together to assign a churn risk score to every seller:

| Model | File | Predicts | Applied to |
|---|---|---|---|
| **Pre-Activation Model** | `models/pre_activation_model.joblib` | `never_activated` (1 = never made a sale) | All sellers |
| **Retention Model** | `models/retention_model.joblib` | `dormant` (1 = went silent after selling) | Activated sellers only |

The **overall churn risk** (`0.0 → 1.0`) combines both scores:
- For sellers who never activated → `overall_churn_risk = never_activated_risk`
- For activated sellers → `overall_churn_risk = max(never_activated_risk, dormancy_risk)`

---

## 2. What "New Data" Means Here

"New data" = a fresh export of the same Olist-style raw CSVs for a **new time window** (e.g., the next quarter). The 10 source files must be present:

```
data/raw/
├── olist_marketing_qualified_leads_dataset.csv
├── olist_closed_deals_dataset.csv
├── olist_sellers_dataset.csv
├── olist_orders_dataset.csv
├── olist_order_items_dataset.csv
├── olist_products_dataset.csv
├── olist_customers_dataset.csv
├── olist_order_reviews_dataset.csv
├── olist_order_payments_dataset.csv
└── product_category_name_translation.csv
```

> **Important:** If you only have a subset of new sellers (e.g., sellers who joined this month), you can pass just those rows through the inference script — but the feature engineering logic must still run against the same format.

---

## 3. Prerequisites

### Environment
```bash
# All dependencies are pinned in pyproject.toml / uv.lock
make install          # equivalent to: uv sync
```

### Trained models must exist
```bash
ls models/
# pre_activation_model.joblib
# retention_model.joblib
```

If the `models/` directory is empty, run the full training pipeline first:
```bash
make run-pipeline
```

### Environment variables (optional overrides)
```bash
# .env file — copy from .env.example
DATA_PATH=./data/raw
MODELS_PATH=./models
OUTPUT_PATH=./outputs
MLFLOW_TRACKING_URI=mlruns
```

---

## 4. Anatomy of the Trained Models

### Feature sets

**Pre-Activation Model features** (9 features, all available before the seller's first sale):

| Feature | Type | Source |
|---|---|---|
| `lead_behaviour_profile_encoded` | int | `olist_marketing_qualified_leads_dataset.csv` |
| `business_segment_encoded` | int | `olist_closed_deals_dataset.csv` |
| `lead_type_encoded` | int | `olist_closed_deals_dataset.csv` |
| `business_type_encoded` | int | `olist_closed_deals_dataset.csv` |
| `origin_encoded` | int | `olist_marketing_qualified_leads_dataset.csv` |
| `seller_state_encoded` | int | `olist_sellers_dataset.csv` |
| `declared_monthly_revenue` | float | `olist_closed_deals_dataset.csv` |
| `declared_product_catalog_size` | float | `olist_closed_deals_dataset.csv` |
| `sales_cycle_days` | int | `won_date - first_contact_date` |

**Retention Model features** (11 features, require at least one completed sale):

| Feature | Type | Source |
|---|---|---|
| `lead_behaviour_profile_encoded` | int | MQL dataset |
| `business_segment_encoded` | int | Deals dataset |
| `total_orders` | int | Aggregated from order_items |
| `total_gmv` | float | Aggregated from order_items |
| `unique_customers` | int | Aggregated from orders |
| `days_active` | int | `last_sale_date - first_sale_date` |
| `days_to_first_sale` | int | `first_sale_date - won_date` |
| `avg_order_value` | float | `total_gmv / total_orders` |
| `gmv_per_order` | float | Engineered |
| `customers_per_order` | float | Engineered |
| `activation_speed_score` | int | Engineered (1/2/3) |

> **Critical:** The `LabelEncoder` in `FeatureEngineer` is fit **at runtime** on the data you pass in. It is **not** serialized alongside the model. This means category encoding is always relative to the values present in the new dataset. See [Pitfall #1](#pitfall-1-label-encoding-is-not-saved) for implications.

---

## 5. Step-by-Step: Preparing the New Dataset

The raw CSVs go through a fixed transformation pipeline. You must replicate these exact steps before calling `model.predict_proba()`.

### Step 1 — Load raw CSVs

```python
import pandas as pd

data = {
    "mqls":        pd.read_csv("data/raw/olist_marketing_qualified_leads_dataset.csv"),
    "deals":       pd.read_csv("data/raw/olist_closed_deals_dataset.csv"),
    "sellers":     pd.read_csv("data/raw/olist_sellers_dataset.csv"),
    "orders":      pd.read_csv("data/raw/olist_orders_dataset.csv"),
    "order_items": pd.read_csv("data/raw/olist_order_items_dataset.csv"),
}
```

### Step 2 — Parse dates

```python
data["deals"]["won_date"] = pd.to_datetime(data["deals"]["won_date"])
data["orders"]["order_purchase_timestamp"] = pd.to_datetime(data["orders"]["order_purchase_timestamp"])
data["mqls"]["first_contact_date"] = pd.to_datetime(data["mqls"]["first_contact_date"])
```

### Step 3 — Build the seller master table

```python
# Join closed deals with MQL funnel data
master = data["deals"].merge(
    data["mqls"][["mql_id", "first_contact_date", "landing_page_id", "origin"]],
    on="mql_id",
    how="left",
)
master["sales_cycle_days"] = (master["won_date"] - master["first_contact_date"]).dt.days

master = master.merge(
    data["sellers"][["seller_id", "seller_city", "seller_state"]],
    on="seller_id",
    how="left",
)
```

### Step 4 — Calculate activity metrics

```python
import numpy as np

# Aggregate orders per seller
completed_statuses = ["delivered", "shipped", "approved", "invoiced"]
seller_orders = data["order_items"].merge(
    data["orders"][["order_id", "order_purchase_timestamp", "order_status", "customer_id"]],
    on="order_id",
    how="inner",
)
seller_orders = seller_orders[seller_orders["order_status"].isin(completed_statuses)]

activity = (
    seller_orders.groupby("seller_id")
    .agg(
        order_id=("order_id", "nunique"),
        customer_id=("customer_id", "nunique"),
        price_sum=("price", "sum"),
        price_mean=("price", "mean"),
        price_std=("price", "std"),
        freight_value=("freight_value", "sum"),
        first_sale_date=("order_purchase_timestamp", "min"),
        last_sale_date=("order_purchase_timestamp", "max"),
        total_items_sold=("order_purchase_timestamp", "count"),
    )
    .reset_index()
    .rename(columns={
        "order_id": "total_orders",
        "customer_id": "unique_customers",
        "price_sum": "total_gmv",
        "price_mean": "avg_order_value",
        "price_std": "std_order_value",
        "freight_value": "total_freight",
    })
)

# Time-based metrics
dataset_end_date = data["orders"]["order_purchase_timestamp"].max()
activity["days_active"] = (activity["last_sale_date"] - activity["first_sale_date"]).dt.days + 1
activity["days_since_last_sale"] = (dataset_end_date - activity["last_sale_date"]).dt.days
activity["avg_daily_orders"] = activity["total_orders"] / activity["days_active"].clip(lower=1)

master = master.merge(activity, on="seller_id", how="left")

# Fill non-active sellers
for col in ["total_orders", "total_gmv", "unique_customers", "total_items_sold", "total_freight"]:
    master[col] = master[col].fillna(0)

master["days_to_first_sale"] = (master["first_sale_date"] - master["won_date"]).dt.days
```

### Step 5 — Handle missing values

```python
categorical_cols = [
    "lead_behaviour_profile", "business_segment", "lead_type",
    "business_type", "origin", "seller_state",
]
for col in categorical_cols:
    if col in master.columns:
        master[col] = master[col].fillna("Unknown")

for col in ["declared_monthly_revenue", "declared_product_catalog_size"]:
    if col in master.columns:
        master[col] = master[col].fillna(0)
```

### Step 6 — Apply churn labels (needed for the retention filter)

```python
NEVER_ACTIVATED_DAYS = 90
DORMANT_DAYS = 60

master["never_activated"] = (
    (master["total_orders"] == 0) | (master["days_to_first_sale"] > NEVER_ACTIVATED_DAYS)
).astype(int)

master["dormant"] = (
    (master["total_orders"] > 0)
    & (master["never_activated"] == 0)
    & (master["days_since_last_sale"] > DORMANT_DAYS)
).astype(int)

master["active"] = (
    (master["total_orders"] > 0)
    & (master["days_since_last_sale"] <= DORMANT_DAYS)
).astype(int)
```

---

## 6. Step-by-Step: Running Inference

### Load the trained models

```python
import joblib

pre_model = joblib.load("models/pre_activation_model.joblib")
ret_model = joblib.load("models/retention_model.joblib")
```

### Build feature matrices using `FeatureEngineer`

```python
from config.settings import get_settings
from src.features import FeatureEngineer

settings = get_settings()
engineer = FeatureEngineer(settings)

# Pre-activation features — all sellers
X_pre = engineer.create_pre_activation_features(master.copy())

# Retention features — activated sellers only
activated = master[master["never_activated"] == 0].copy()
X_ret = engineer.create_retention_features(activated).replace([np.inf, -np.inf], 0)
```

### Generate probability scores

```python
# Pre-activation risk for everyone
master["never_activated_risk"] = pre_model.predict_proba(X_pre)[:, 1]

# Dormancy risk for activated sellers only
master.loc[activated.index, "dormancy_risk"] = ret_model.predict_proba(X_ret)[:, 1]

# Combined overall risk
master["overall_churn_risk"] = master.apply(
    lambda row: (
        row["never_activated_risk"]
        if pd.isna(row.get("dormancy_risk"))
        else max(row["never_activated_risk"], row.get("dormancy_risk", 0))
    ),
    axis=1,
)

# Risk category buckets
master["risk_category"] = pd.cut(
    master["overall_churn_risk"],
    bins=[0, 0.3, 0.6, 0.8, 1.0],
    labels=["Low", "Medium", "High", "Critical"],
)
```

### Export results

```python
master.to_csv("outputs/new_seller_risk_scores.csv", index=False)
print(master["risk_category"].value_counts())
```

---

## 7. Complete Inference Script

Save this as `scripts/run_inference.py` and run it with `make run-inference` (see Makefile addition below).

```python
#!/usr/bin/env python3
"""
Inference script: score new sellers with trained churn models.

Usage:
    PYTHONPATH=. uv run python scripts/run_inference.py
    PYTHONPATH=. uv run python scripts/run_inference.py --data-path /path/to/new/data
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.features import FeatureEngineer
from src.pipeline import (
    ChurnAnalyzer,
    DataLoader,
    DataPreprocessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run churn inference on new seller data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override the DATA_PATH setting (directory with raw CSVs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/new_seller_risk_scores.csv"),
        help="Where to write the scored CSV (default: outputs/new_seller_risk_scores.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if args.data_path:
        # Override data path without touching the settings object globally
        settings.DATA_PATH = args.data_path

    # ------------------------------------------------------------------
    # 1. Validate models exist
    # ------------------------------------------------------------------
    pre_model_path = settings.MODELS_PATH / "pre_activation_model.joblib"
    ret_model_path = settings.MODELS_PATH / "retention_model.joblib"

    for path in (pre_model_path, ret_model_path):
        if not path.exists():
            logger.error(f"Model not found: {path}. Run 'make run-pipeline' first.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load & preprocess raw data (same steps as training)
    # ------------------------------------------------------------------
    loader = DataLoader(settings)
    loader.load_all()
    loader.parse_dates()
    master = loader.build_seller_master_table()

    preprocessor = DataPreprocessor(settings)
    master = preprocessor.calculate_seller_activity_metrics(master, loader.data)
    master = preprocessor.handle_missing_values(master)

    if master.empty:
        logger.error("No data found. Check DATA_PATH.")
        sys.exit(1)

    logger.info(f"Loaded {len(master):,} sellers for scoring.")

    # ------------------------------------------------------------------
    # 3. Apply churn labels (needed to route sellers to the right model)
    # ------------------------------------------------------------------
    analyzer = ChurnAnalyzer(settings)
    master = analyzer.define_churn_labels(master)

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    engineer = FeatureEngineer(settings)
    X_pre = engineer.create_pre_activation_features(master.copy())

    activated = master[master["never_activated"] == 0].copy()
    X_ret = engineer.create_retention_features(activated).replace([np.inf, -np.inf], 0)

    # ------------------------------------------------------------------
    # 5. Load models & score
    # ------------------------------------------------------------------
    pre_model = joblib.load(pre_model_path)
    ret_model = joblib.load(ret_model_path)

    logger.info("Scoring sellers with pre-activation model...")
    master["never_activated_risk"] = pre_model.predict_proba(X_pre)[:, 1]

    if not activated.empty:
        logger.info("Scoring activated sellers with retention model...")
        master.loc[activated.index, "dormancy_risk"] = ret_model.predict_proba(X_ret)[:, 1]

    # ------------------------------------------------------------------
    # 6. Combine into overall risk
    # ------------------------------------------------------------------
    master["overall_churn_risk"] = master.apply(
        lambda row: (
            row["never_activated_risk"]
            if pd.isna(row.get("dormancy_risk"))
            else max(row["never_activated_risk"], row.get("dormancy_risk", 0))
        ),
        axis=1,
    )
    master["risk_category"] = pd.cut(
        master["overall_churn_risk"],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    )

    # ------------------------------------------------------------------
    # 7. Export
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols_to_export = [
        "seller_id", "seller_city", "seller_state",
        "business_segment", "lead_behaviour_profile",
        "total_orders", "total_gmv", "days_since_last_sale",
        "never_activated", "dormant", "active",
        "never_activated_risk", "dormancy_risk",
        "overall_churn_risk", "risk_category",
    ]
    available_cols = [c for c in cols_to_export if c in master.columns]
    master[available_cols].to_csv(args.output, index=False)

    logger.info(f"\n✅  Scores saved → {args.output}")
    logger.info("\nRisk distribution:")
    for cat, count in master["risk_category"].value_counts().items():
        pct = count / len(master) * 100
        logger.info(f"  {cat:10s}: {count:>5,} sellers ({pct:.1f}%)")


if __name__ == "__main__":
    main()
```

### Add the Makefile target

Add this to your `Makefile`:

```makefile
# Score new sellers with the trained models (no re-training)
run-inference:
	$(UV_RUN) python scripts/run_inference.py

# Score from a custom data directory
run-inference-custom:
	$(UV_RUN) python scripts/run_inference.py --data-path $(DATA_PATH) --output $(OUTPUT)
```

Then run:
```bash
make run-inference

# With custom paths:
DATA_PATH=/path/to/new/data OUTPUT=outputs/q1_2025_scores.csv make run-inference-custom
```

---

## 8. Loading Models from MLflow

If you prefer to load the registered model version from the MLflow Model Registry instead of from disk:

```python
import mlflow.sklearn

# Load the latest production model by name
pre_model = mlflow.sklearn.load_model("models:/olist-pre-activation-churn/latest")
ret_model = mlflow.sklearn.load_model("models:/olist-retention-churn/latest")
```

Or a specific version:
```python
pre_model = mlflow.sklearn.load_model("models:/olist-pre-activation-churn/1")
```

**Start the MLflow UI to browse registered models:**
```bash
make mlflow-ui
# Then open: http://localhost:5000
```

> The model names (`olist-pre-activation-churn` and `olist-retention-churn`) are registered in `src/models.py` via `mlflow.sklearn.log_model(..., registered_model_name=...)`.

---

## 9. Understanding the Output

### Output columns

| Column | Range | Meaning |
|---|---|---|
| `never_activated_risk` | 0.0 – 1.0 | Probability of never making a first sale |
| `dormancy_risk` | 0.0 – 1.0 | Probability of going dormant (only for activated sellers) |
| `overall_churn_risk` | 0.0 – 1.0 | Combined worst-case churn risk |
| `risk_category` | Low/Medium/High/Critical | Bucket derived from `overall_churn_risk` |

### Risk category thresholds

| Category | Risk Score Range | Recommended Action |
|---|---|---|
| **Low** | 0.00 – 0.30 | Monitor; no immediate action needed |
| **Medium** | 0.30 – 0.60 | Proactive outreach cadence |
| **High** | 0.60 – 0.80 | Assign account manager |
| **Critical** | 0.80 – 1.00 | Immediate intervention (calls, offers) |

---

## 10. Common Pitfalls & Troubleshooting

### Pitfall #1: Label encoding is not saved

The `FeatureEngineer` uses `sklearn.LabelEncoder` which is **refit every time** `create_pre_activation_features()` is called. This means:

- If a new dataset has a category value that didn't exist during training (e.g., a new `seller_state`), it will get a different integer code.
- **For most tree-based models (Random Forest, Gradient Boosting)**, this is usually fine because they learn thresholds, not exact values.
- **If you are worried** about consistency, save the `LabelEncoder` objects after training:

```python
# Save encoders after first training run
import joblib
joblib.dump(engineer.label_encoders, "models/label_encoders.joblib")

# Load and use during inference
le_dict = joblib.load("models/label_encoders.joblib")
for col, le in le_dict.items():
    master[f"{col}_encoded"] = le.transform(master[col].astype(str))
```

> ⚠️ Using saved encoders will raise `ValueError` if new categories appear. Wrap it in a try/except and fall back to `le.classes_[-1]` (Unknown) for unseen values.

### Pitfall #2: Missing columns in the new data

If the new data is missing a column (e.g., no `declared_monthly_revenue`), `FeatureEngineer` gracefully skips it but the model will receive **fewer features** than it was trained on — causing a `ValueError`.

**Fix:** Ensure all required CSVs contain the expected columns. Use the validation module:
```bash
make run-validation
```

### Pitfall #3: All sellers have 0 orders

If your new data window is very recent, all sellers might have `total_orders == 0`, making everyone `never_activated`. This is expected behavior for a very new cohort. The pre-activation model will still score them correctly. The retention model simply won't fire (no activated sellers to score).

### Pitfall #4: `NaN` in `days_since_last_sale`

This happens when a seller has no orders (so `last_sale_date` is NaT). The pipeline handles this by filling with 0 for `total_orders` and leaving time metrics as NaN — which only affects the retention model (which only runs on sellers who **have** made a sale).

### Pitfall #5: Wrong PYTHONPATH

All `import` statements in the project use `from src.xxx import ...` with the project root as the Python root. Always run with:
```bash
PYTHONPATH=. uv run python scripts/run_inference.py
# or use:
make run-inference
```

---

## 11. When to Re-Train

You should re-train the models (via `make run-pipeline`) when:

| Signal | Action |
|---|---|
| **Model AUC-ROC drops > 5% on new data** | Re-train immediately |
| **New business rule changes** (e.g., changing `DORMANT_DAYS` from 60 to 45) | Re-train — the target label definition has changed |
| **Significant data drift** (new seller segments, market changes) | Re-train with fresh data |
| **> 3 months have passed** since the last training run | Consider re-training as a preventive measure |
| **New Olist features are available** | Add to `FeatureEngineer` then re-train |

To check for model drift, compare the risk score distribution of the new scored batch against the distribution from the original training run:

```python
import matplotlib.pyplot as plt

# Load old scores from training
old = pd.read_csv("outputs/seller_risk_scores.csv")["overall_churn_risk"]

# New scored data
new = master["overall_churn_risk"]

plt.hist(old, bins=30, alpha=0.5, label="Training data")
plt.hist(new, bins=30, alpha=0.5, label="New data")
plt.legend()
plt.title("Risk Score Distribution Comparison")
plt.savefig("outputs/drift_check.png")
```

A large visual shift between the two distributions is a strong signal that re-training is warranted.

---

*Generated for the Olist Seller Churn Prediction project — Cairo Cananea*
