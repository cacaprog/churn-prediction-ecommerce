#!/usr/bin/env python3
"""
Inference script: score new sellers with the trained churn prediction models.

This script loads the two saved models (pre-activation + retention) and runs
them on a fresh set of raw Olist CSVs, producing an overall_churn_risk score
and risk_category for every seller — WITHOUT re-training.

Usage
-----
    # Default: uses paths from config/settings.py / .env
    PYTHONPATH=. uv run python scripts/run_inference.py

    # Override raw-data directory and output file:
    PYTHONPATH=. uv run python scripts/run_inference.py \\
        --data-path /path/to/new/data \\
        --output outputs/q2_2025_scores.csv

Makefile shortcuts
------------------
    make run-inference
    DATA_PATH=/new/data OUTPUT=outputs/q2.csv make run-inference-custom

See docs/INFERENCE_GUIDE.md for a full walkthrough.
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when called directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import get_settings  # noqa: E402
from src.features import FeatureEngineer  # noqa: E402
from src.pipeline import ChurnAnalyzer, DataLoader, DataPreprocessor  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score new Olist sellers with the trained churn models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=(
            "Directory containing the 10 raw Olist CSVs. "
            "Defaults to DATA_PATH from config/settings.py / .env."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/new_seller_risk_scores.csv"),
        help="Where to write the scored CSV (default: outputs/new_seller_risk_scores.csv).",
    )
    parser.add_argument(
        "--models-path",
        type=Path,
        default=None,
        help="Override MODELS_PATH from settings.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    settings = get_settings()

    # Allow CLI overrides of path settings
    if args.data_path:
        settings.DATA_PATH = args.data_path
    if args.models_path:
        settings.MODELS_PATH = args.models_path

    logger.info("=" * 70)
    logger.info("OLIST CHURN INFERENCE — Scoring new seller data")
    logger.info("=" * 70)
    logger.info(f"  Data path   : {settings.DATA_PATH}")
    logger.info(f"  Models path : {settings.MODELS_PATH}")
    logger.info(f"  Output      : {args.output}")

    # ------------------------------------------------------------------
    # 1. Validate that trained models exist on disk
    # ------------------------------------------------------------------
    pre_model_path = settings.MODELS_PATH / "pre_activation_model.joblib"
    ret_model_path = settings.MODELS_PATH / "retention_model.joblib"

    missing = [p for p in (pre_model_path, ret_model_path) if not p.exists()]
    if missing:
        for p in missing:
            logger.error(f"Model not found: {p}")
        logger.error("Run 'make run-pipeline' to train the models first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Data loading & preprocessing (identical to training pipeline)
    # ------------------------------------------------------------------
    logger.info("\n[Step 1/5] Loading raw data...")
    loader = DataLoader(settings)
    loader.load_all()
    loader.parse_dates()
    master = loader.build_seller_master_table()

    preprocessor = DataPreprocessor(settings)
    master = preprocessor.calculate_seller_activity_metrics(master, loader.data)
    master = preprocessor.handle_missing_values(master)

    if master.empty:
        logger.error("No sellers found in the data. Check your DATA_PATH.")
        sys.exit(1)

    logger.info(f"  → {len(master):,} sellers loaded.")

    # ------------------------------------------------------------------
    # 3. Apply churn labels (routes sellers to the right model)
    # ------------------------------------------------------------------
    logger.info("\n[Step 2/5] Applying churn labels...")
    analyzer = ChurnAnalyzer(settings)
    master = analyzer.define_churn_labels(master)

    n_activated = (master["never_activated"] == 0).sum()
    logger.info(f"  → Never activated : {master['never_activated'].sum():,}")
    logger.info(f"  → Activated        : {n_activated:,}")

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    logger.info("\n[Step 3/5] Engineering features...")
    engineer = FeatureEngineer(settings)

    X_pre = engineer.create_pre_activation_features(master.copy())
    logger.info(f"  Pre-activation feature matrix: {X_pre.shape}")

    activated = master[master["never_activated"] == 0].copy()
    if not activated.empty:
        X_ret = engineer.create_retention_features(activated).replace(
            [np.inf, -np.inf], 0
        )
        logger.info(f"  Retention feature matrix     : {X_ret.shape}")
    else:
        X_ret = pd.DataFrame()
        logger.warning(
            "  No activated sellers found — retention model will be skipped."
        )

    # ------------------------------------------------------------------
    # 5. Load models & generate probability scores
    # ------------------------------------------------------------------
    logger.info("\n[Step 4/5] Generating risk scores...")
    pre_model = joblib.load(pre_model_path)
    ret_model = joblib.load(ret_model_path)

    # Pre-activation risk for ALL sellers
    master["never_activated_risk"] = pre_model.predict_proba(X_pre)[:, 1]

    # Dormancy risk for activated sellers only
    master["dormancy_risk"] = np.nan
    if not X_ret.empty:
        master.loc[activated.index, "dormancy_risk"] = ret_model.predict_proba(X_ret)[
            :, 1
        ]

    # Combined overall risk = worst of both scores
    master["overall_churn_risk"] = master.apply(
        lambda row: (
            row["never_activated_risk"]
            if pd.isna(row.get("dormancy_risk"))
            else max(row["never_activated_risk"], row.get("dormancy_risk", 0.0))
        ),
        axis=1,
    )

    # Risk category buckets (matches thresholds in pipeline.py)
    master["risk_category"] = pd.cut(
        master["overall_churn_risk"],
        bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    )

    # ------------------------------------------------------------------
    # 6. Export scored data
    # ------------------------------------------------------------------
    logger.info("\n[Step 5/5] Exporting results...")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    preferred_cols = [
        "seller_id",
        "seller_city",
        "seller_state",
        "business_segment",
        "lead_behaviour_profile",
        "total_orders",
        "total_gmv",
        "unique_customers",
        "days_since_last_sale",
        "days_to_first_sale",
        "never_activated",
        "dormant",
        "active",
        "never_activated_risk",
        "dormancy_risk",
        "overall_churn_risk",
        "risk_category",
    ]
    export_cols = [c for c in preferred_cols if c in master.columns]
    master[export_cols].to_csv(args.output, index=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Output file  : {args.output}")
    logger.info(f"  Total sellers: {len(master):,}")
    logger.info("\n  Risk distribution:")
    for category, count in master["risk_category"].value_counts().items():
        pct = count / len(master) * 100
        bar = "█" * int(pct / 2)
        logger.info(f"    {str(category):10s} {count:>5,} ({pct:4.1f}%)  {bar}")

    high_critical = master["risk_category"].isin(["High", "Critical"]).sum()
    logger.info(
        f"\n  ⚠️  {high_critical:,} sellers need attention (High + Critical risk)"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
