#!/usr/bin/env python3
"""
Olist Seller Churn Analysis - Production Pipeline
====================================================

A comprehensive churn analysis pipeline for Olist e-commerce marketplace.

Sections:
    1. Data Loading & Preprocessing
    2. Exploratory Data Analysis (EDA)
    3. Feature Engineering
    4. Model Training & Evaluation
    5. Risk Scoring & Predictions
    6. Results Export

Author: Cairo Cananea
Website: cairocananea.com.br
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

import joblib

from config.settings import Settings, get_settings
from src.validation.schemas import validate_seller_data, validate_risk_scores

warnings.filterwarnings("ignore")


# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_logging() -> logging.Logger:
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                f'churn_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# =============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# =============================================================================


class DataLoader:
    """Handles loading and initial cleaning of all datasets."""

    DATASETS = {
        "mqls": "olist_marketing_qualified_leads_dataset.csv",
        "deals": "olist_closed_deals_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "translation": "product_category_name_translation.csv",
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.data: Dict[str, pd.DataFrame] = {}

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets from the raw data folder."""
        logger.info("=" * 60)
        logger.info("SECTION 1: DATA LOADING & PREPROCESSING")
        logger.info("=" * 60)

        for key, filename in self.DATASETS.items():
            filepath = os.path.join(self.settings.DATA_PATH, filename)
            try:
                self.data[key] = pd.read_csv(filepath)
                logger.info(
                    f"Loaded {key}: {self.data[key].shape[0]:,} rows, {self.data[key].shape[1]} columns"
                )
            except FileNotFoundError:
                logger.error(f"File not found: {filepath}")
                raise

        logger.info(f"Successfully loaded {len(self.data)} datasets")
        return self.data

    def parse_dates(self) -> None:
        """Convert date columns to datetime objects."""
        date_columns = {
            "deals": ["won_date"],
            "orders": ["order_purchase_timestamp", "order_delivered_customer_date"],
            "mqls": ["first_contact_date"],
        }

        for dataset, columns in date_columns.items():
            for col in columns:
                if col in self.data[dataset].columns:
                    self.data[dataset][col] = pd.to_datetime(self.data[dataset][col])

        logger.info("Date parsing complete")

    def build_seller_master_table(self) -> pd.DataFrame:
        """Build the master seller dataset by joining all relevant tables."""
        logger.info("\nBuilding Seller Master Dataset...")

        # Start with closed deals
        master = self.data["deals"].copy()

        # Merge MQL data (funnel information)
        master = master.merge(
            self.data["mqls"][
                ["mql_id", "first_contact_date", "landing_page_id", "origin"]
            ],
            on="mql_id",
            how="left",
        )

        # Calculate sales cycle duration
        master["sales_cycle_days"] = (
            master["won_date"] - master["first_contact_date"]
        ).dt.days

        # Merge seller location data
        master = master.merge(
            self.data["sellers"][["seller_id", "seller_city", "seller_state"]],
            on="seller_id",
            how="left",
        )

        logger.info(f"Master table created: {master.shape[0]:,} sellers")
        return master


class DataPreprocessor:
    """Handles data cleaning, missing value treatment, and feature preparation."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        logger.info("\n--- Handling Missing Values ---")

        # Categorical columns: fill with 'Unknown'
        categorical_cols = [
            "lead_behaviour_profile",
            "business_segment",
            "lead_type",
            "business_type",
            "origin",
            "seller_state",
        ]
        for col in categorical_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna("Unknown")
                    logger.info(
                        f"  {col}: filled {missing_count} missing with 'Unknown'"
                    )

        # Numeric columns: fill with 0 or median
        numeric_cols = ["declared_monthly_revenue", "declared_product_catalog_size"]
        for col in numeric_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna(0)
                    logger.info(f"  {col}: filled {missing_count} missing with 0")

        return df

    def calculate_seller_activity_metrics(
        self, df: pd.DataFrame, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate comprehensive seller activity metrics."""
        logger.info("\n--- Calculating Seller Activity Metrics ---")

        # Merge orders with items
        seller_orders = data["order_items"].merge(
            data["orders"][
                ["order_id", "order_purchase_timestamp", "order_status", "customer_id"]
            ],
            on="order_id",
            how="inner",
        )

        # Filter completed orders only
        completed_statuses = ["delivered", "shipped", "approved", "invoiced"]
        seller_orders = seller_orders[
            seller_orders["order_status"].isin(completed_statuses)
        ]

        # Aggregate seller activity
        activity = (
            seller_orders.groupby("seller_id")
            .agg(
                {
                    "order_id": "nunique",
                    "customer_id": "nunique",
                    "price": ["sum", "mean", "std"],
                    "freight_value": "sum",
                    "order_purchase_timestamp": ["min", "max", "count"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        activity.columns = [
            "seller_id",
            "total_orders",
            "unique_customers",
            "total_gmv",
            "avg_order_value",
            "std_order_value",
            "total_freight",
            "first_sale_date",
            "last_sale_date",
            "total_items_sold",
        ]

        # Calculate time-based metrics
        dataset_end_date = data["orders"]["order_purchase_timestamp"].max()
        activity["days_active"] = (
            activity["last_sale_date"] - activity["first_sale_date"]
        ).dt.days + 1
        activity["days_since_last_sale"] = (
            dataset_end_date - activity["last_sale_date"]
        ).dt.days
        activity["avg_daily_orders"] = activity["total_orders"] / activity[
            "days_active"
        ].clip(lower=1)

        # Merge with master
        df = df.merge(activity, on="seller_id", how="left")

        # Fill NaN for non-active sellers
        numeric_fill = [
            "total_orders",
            "total_gmv",
            "unique_customers",
            "total_items_sold",
            "total_freight",
        ]
        for col in numeric_fill:
            df[col] = df[col].fillna(0)

        # Calculate time to first sale
        df["days_to_first_sale"] = (df["first_sale_date"] - df["won_date"]).dt.days

        logger.info(
            f"Activity metrics calculated for {df[df['total_orders'] > 0].shape[0]:,} active sellers"
        )
        return df


# =============================================================================
# SECTION 2: DATA ANALYSIS
# =============================================================================


class ChurnAnalyzer:
    """Performs comprehensive churn analysis and segmentation."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def define_churn_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply churn definitions to the dataset."""
        logger.info("\n" + "=" * 60)
        logger.info("SECTION 2: DATA ANALYSIS")
        logger.info("=" * 60)
        logger.info("\n--- Defining Churn Labels ---")

        # 1. Never-activated: No orders or first sale > 90 days after won_date
        df["never_activated"] = (
            (df["total_orders"] == 0)
            | (df["days_to_first_sale"] > self.settings.NEVER_ACTIVATED_DAYS)
        ).astype(int)

        # 2. Dormant: Had orders but no activity in last 60 days
        df["dormant"] = (
            (df["total_orders"] > 0)
            & (df["never_activated"] == 0)
            & (df["days_since_last_sale"] > self.settings.DORMANT_DAYS)
        ).astype(int)

        # 3. Active: Recent activity within 60 days
        df["active"] = (
            (df["total_orders"] > 0)
            & (df["days_since_last_sale"] <= self.settings.DORMANT_DAYS)
        ).astype(int)

        # Overall churn flag
        df["churned"] = ((df["never_activated"] == 1) | (df["dormant"] == 1)).astype(
            int
        )

        # Log distribution
        total = len(df)
        logger.info(f"  Total Sellers: {total:,}")
        logger.info(
            f"  Never Activated: {df['never_activated'].sum():,} ({df['never_activated'].mean()*100:.1f}%)"
        )
        logger.info(
            f"  Dormant: {df['dormant'].sum():,} ({df['dormant'].mean()*100:.1f}%)"
        )
        logger.info(
            f"  Active: {df['active'].sum():,} ({df['active'].mean()*100:.1f}%)"
        )
        logger.info(f"  Overall Churn Rate: {df['churned'].mean()*100:.1f}%")

        return df

    def cohort_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate cohort-based churn analysis."""
        logger.info("\n--- Cohort Analysis ---")

        df["cohort_month"] = df["won_date"].dt.to_period("M")

        cohort_summary = (
            df.groupby("cohort_month")
            .agg(
                {
                    "seller_id": "count",
                    "never_activated": "sum",
                    "dormant": "sum",
                    "active": "sum",
                    "churned": "sum",
                    "total_gmv": "sum",
                }
            )
            .reset_index()
        )

        cohort_summary.rename(columns={"seller_id": "total_sellers"}, inplace=True)
        cohort_summary["churn_rate"] = (
            cohort_summary["churned"] / cohort_summary["total_sellers"] * 100
        ).round(1)
        cohort_summary["activation_rate"] = (
            (cohort_summary["total_sellers"] - cohort_summary["never_activated"])
            / cohort_summary["total_sellers"]
            * 100
        ).round(1)

        logger.info(f"Cohorts analyzed: {len(cohort_summary)} months")
        logger.info(
            f"Average churn rate by cohort: {cohort_summary['churn_rate'].mean():.1f}%"
        )

        return cohort_summary

    def segment_analysis(
        self, df: pd.DataFrame, segment_col: str
    ) -> Optional[pd.DataFrame]:
        """Analyze churn rates by segment."""
        if segment_col not in df.columns:
            return None

        # Clean segment column
        clean_col = f"{segment_col}_clean"
        df[clean_col] = df[segment_col].fillna("Unknown")

        analysis = (
            df.groupby(clean_col)
            .agg(
                {
                    "seller_id": "count",
                    "churned": "sum",
                    "never_activated": "sum",
                    "active": "sum",
                    "total_gmv": ["mean", "sum"],
                }
            )
            .reset_index()
        )

        # Flatten columns
        analysis.columns = [
            "Segment",
            "Total_Sellers",
            "Churned",
            "Never_Activated",
            "Active",
            "Avg_GMV",
            "Total_GMV",
        ]

        analysis["Churn_Rate_%"] = (
            analysis["Churned"] / analysis["Total_Sellers"] * 100
        ).round(1)
        analysis["Market_Share_%"] = (
            analysis["Total_Sellers"] / analysis["Total_Sellers"].sum() * 100
        ).round(1)

        return analysis.sort_values("Churn_Rate_%", ascending=False)


# =============================================================================
# SECTION 5: RISK SCORING & PREDICTIONS
# =============================================================================


class RiskScorer:
    """Generates churn risk scores for all sellers."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def calculate_risk_scores(
        self, data: pd.DataFrame, pre_model, pre_features, ret_model, ret_features
    ) -> pd.DataFrame:
        """Generate comprehensive risk scores."""
        logger.info("\n" + "=" * 60)
        logger.info("SECTION 5: RISK SCORING & PREDICTIONS")
        logger.info("=" * 60)
        logger.info("\n--- Generating Risk Scores ---")

        df = data.copy()

        # Pre-activation risk (for all sellers)
        X_pre = df[pre_features]
        df["never_activated_risk"] = pre_model.predict_proba(X_pre)[:, 1]

        # Retention risk (only for activated sellers)
        activated_mask = df["never_activated"] == 0
        X_ret = df.loc[activated_mask, ret_features].replace([np.inf, -np.inf], 0)
        df.loc[activated_mask, "dormancy_risk"] = ret_model.predict_proba(X_ret)[:, 1]

        # Combined overall risk
        df["overall_churn_risk"] = df.apply(
            lambda row: row["never_activated_risk"]
            if pd.isna(row["dormancy_risk"])
            else max(row["never_activated_risk"], row["dormancy_risk"]),
            axis=1,
        )

        # Risk categories
        df["risk_category"] = pd.cut(
            df["overall_churn_risk"],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["Low", "Medium", "High", "Critical"],
        )

        # Log risk distribution
        logger.info("\nRisk Score Distribution:")
        risk_dist = df["risk_category"].value_counts()
        for category, count in risk_dist.items():
            logger.info(f"  {category}: {count:,} sellers ({count/len(df)*100:.1f}%)")

        return df


# =============================================================================
# SECTION 6: VISUALIZATION & EXPORT
# =============================================================================


class ChurnVisualizer:
    """Creates visualizations for analysis results."""

    def __init__(self, settings: Settings):
        self.settings = settings
        sns.set_style("darkgrid")
        plt.rcParams["figure.figsize"] = settings.FIG_SIZE

    def save_plot(self, filename: str) -> None:
        """Save plot to output directory."""
        path = os.path.join(self.settings.OUTPUT_PATH, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {filename}")

    def plot_churn_distribution(self, df: pd.DataFrame) -> None:
        """Plot seller status distribution."""
        fig, ax = plt.subplots(figsize=(8, 8))

        status_counts = {
            "Never\nActivated": df["never_activated"].sum(),
            "Dormant": df["dormant"].sum(),
            "Active": df["active"].sum(),
        }

        colors = ["#FF6B6B", "#FDCB6E", "#6C5CE7"]
        wedges, texts, autotexts = ax.pie(
            status_counts.values(),
            labels=status_counts.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            explode=(0.02, 0.02, 0.02),
        )

        ax.set_title(
            "Seller Status Distribution", fontsize=14, fontweight="bold", pad=20
        )
        self.save_plot("01_seller_status_distribution.png")

    def plot_cohort_analysis(self, cohort_df: pd.DataFrame) -> None:
        """Plot cohort churn rates over time."""
        plt.figure(figsize=(12, 6))

        # Filter to complete cohorts
        cohort_plot = cohort_df[cohort_df["cohort_month"].astype(str) < "2018-09"]

        plt.plot(
            cohort_plot["cohort_month"].astype(str),
            cohort_plot["churn_rate"],
            marker="o",
            linewidth=2,
            color="#E74C3C",
            markersize=8,
        )

        plt.fill_between(
            range(len(cohort_plot)),
            cohort_plot["churn_rate"],
            alpha=0.3,
            color="#E74C3C",
        )

        plt.title("Churn Rate by Onboarding Cohort", fontsize=14, fontweight="bold")
        plt.xlabel("Cohort Month", fontsize=12)
        plt.ylabel("Churn Rate (%)", fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        self.save_plot("02_cohort_churn_rate.png")

    def plot_feature_importance(
        self, importance_df: pd.DataFrame, title: str, filename: str
    ) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(10, 6))

        top_features = importance_df.head(10).sort_values("Importance")

        bars = plt.barh(
            top_features["Feature"], top_features["Importance"], color="#4834D4"
        )

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Importance", fontsize=12)
        plt.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        self.save_plot(filename)

    def plot_risk_distribution(self, df: pd.DataFrame) -> None:
        """Plot overall risk score distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(
            df["overall_churn_risk"],
            bins=30,
            color="#6C5CE7",
            edgecolor="white",
            alpha=0.7,
        )
        ax1.axvline(
            df["overall_churn_risk"].mean(),
            color="#E74C3C",
            linestyle="--",
            label=f'Mean: {df["overall_churn_risk"].mean():.2f}',
        )
        ax1.set_title("Risk Score Distribution", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Churn Risk Score")
        ax1.set_ylabel("Number of Sellers")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Risk categories pie chart
        risk_counts = df["risk_category"].value_counts()
        colors = {
            "Low": "#00B894",
            "Medium": "#FDCB6E",
            "High": "#E17055",
            "Critical": "#D63031",
        }

        ax2.pie(
            risk_counts.values,
            labels=risk_counts.index,
            autopct="%1.1f%%",
            colors=[colors.get(x, "#6C5CE7") for x in risk_counts.index],
            startangle=90,
        )
        ax2.set_title("Risk Categories", fontsize=12, fontweight="bold")

        plt.tight_layout()
        self.save_plot("03_risk_distribution.png")


class ResultsExporter:
    """Exports analysis results and datasets."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def export_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Export DataFrame to CSV."""
        filepath = os.path.join(self.settings.OUTPUT_PATH, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"  Exported: {filename} ({len(df):,} rows)")

    def save_model(self, model, model_name: str) -> None:
        """Save trained model to disk."""
        filepath = os.path.join(self.settings.MODELS_PATH, f"{model_name}.joblib")
        joblib.dump(model, filepath)
        logger.info(f"  Saved model: {model_name}.joblib")

    def generate_summary_report(self, df: pd.DataFrame, cohort_df: pd.DataFrame) -> str:
        """Generate text summary of analysis."""
        report = f"""
================================================================================
OLIST SELLER CHURN ANALYSIS - SUMMARY REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
Total Sellers Analyzed: {len(df):,}
Overall Churn Rate: {df['churned'].mean()*100:.1f}%
Revenue at Risk: R$ {df[df['churned']==1]['total_gmv'].sum():,.2f}

CHURN BREAKDOWN
---------------
Never Activated: {df['never_activated'].sum():,} ({df['never_activated'].mean()*100:.1f}%)
Dormant: {df['dormant'].sum():,} ({df['dormant'].mean()*100:.1f}%)
Active: {df['active'].sum():,} ({df['active'].mean()*100:.1f}%)

RISK ASSESSMENT
---------------
High/Critical Risk Sellers: {(df['risk_category'].isin(['High', 'Critical'])).sum():,}
Percentage at High Risk: {(df['risk_category'].isin(['High', 'Critical'])).mean()*100:.1f}%

COHORT ANALYSIS
---------------
Cohorts Analyzed: {len(cohort_df)}
Average Cohort Churn Rate: {cohort_df['churn_rate'].mean():.1f}%
Best Performing Cohort: {cohort_df.loc[cohort_df['churn_rate'].idxmin(), 'cohort_month']}

TOP SEGMENTS BY CHURN RATE
--------------------------
See exported CSV files for detailed segment analysis.

================================================================================
"""
        # Save report
        report_path = os.path.join(self.settings.OUTPUT_PATH, "analysis_summary.txt")
        with open(report_path, "w") as f:
            f.write(report)

        return report


# =============================================================================
# SECTION 7: INTERVENTION PRIORITIES
# =============================================================================


class InterventionPrioritizer:
    """Generates prioritized intervention lists with business impact estimates."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_intervention_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create prioritized list of sellers for intervention."""
        logger.info("\n--- Generating Intervention Priorities ---")

        # Filter to high and critical risk sellers who are still active
        intervention_mask = (df["risk_category"].isin(["High", "Critical"])) & (
            df["active"] == 1
        )
        intervention_df = df[intervention_mask].copy()

        if len(intervention_df) == 0:
            logger.info("No active high-risk sellers found for intervention")
            return pd.DataFrame()

        # Calculate priority score based on risk and value
        intervention_df["gmv_percentile"] = intervention_df["total_gmv"].rank(pct=True)
        intervention_df["priority_score"] = (
            intervention_df["overall_churn_risk"] * 0.6
            + intervention_df["gmv_percentile"] * 0.4
        )

        # Add intervention urgency
        intervention_df["days_until_dormant"] = (
            self.settings.DORMANT_DAYS - intervention_df["days_since_last_sale"]
        )
        intervention_df["urgency"] = intervention_df["days_until_dormant"].apply(
            lambda x: "Immediate"
            if x <= 7
            else "High"
            if x <= 14
            else "Medium"
            if x <= 30
            else "Low"
        )

        # Sort by priority
        intervention_df = intervention_df.sort_values("priority_score", ascending=False)

        # Select relevant columns for export
        columns = [
            "seller_id",
            "seller_city",
            "seller_state",
            "business_segment",
            "lead_behaviour_profile",
            "total_orders",
            "total_gmv",
            "unique_customers",
            "days_since_last_sale",
            "overall_churn_risk",
            "risk_category",
            "priority_score",
            "urgency",
            "declared_monthly_revenue",
        ]

        available_cols = [col for col in columns if col in intervention_df.columns]
        return intervention_df[available_cols]

    def calculate_business_gains(
        self, intervention_df: pd.DataFrame, scored_data: pd.DataFrame
    ) -> Dict:
        """Calculate potential business gains from interventions."""

        if len(intervention_df) == 0:
            return {
                "total_at_risk": 0,
                "gmv_at_risk": 0,
                "intervention_success_rate": 0.25,
                "projected_gmv_saved": 0,
                "projected_sellers_saved": 0,
            }

        # Assumptions for calculations
        INTERVENTION_SUCCESS_RATE = 0.25  # 25% of intervened sellers can be saved
        AVG_MONTHLY_GMV_MULTIPLIER = 3  # Project 3 months of retained GMV

        total_gmv_at_risk = intervention_df["total_gmv"].sum()
        total_sellers_at_risk = len(intervention_df)

        projected_gmv_saved = (
            total_gmv_at_risk * INTERVENTION_SUCCESS_RATE * AVG_MONTHLY_GMV_MULTIPLIER
        )
        projected_sellers_saved = int(total_sellers_at_risk * INTERVENTION_SUCCESS_RATE)

        return {
            "total_at_risk": total_sellers_at_risk,
            "gmv_at_risk": total_gmv_at_risk,
            "intervention_success_rate": INTERVENTION_SUCCESS_RATE,
            "projected_gmv_saved": projected_gmv_saved,
            "projected_sellers_saved": projected_sellers_saved,
            "avg_seller_gmv": intervention_df["total_gmv"].mean(),
        }

    def generate_intervention_report(
        self, intervention_df: pd.DataFrame, gains: Dict
    ) -> str:
        """Generate intervention priorities text report."""

        report = f"""================================================================================
INTERVENTION PRIORITIES - SELLER RETENTION ACTION PLAN
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
Active Sellers at High Risk: {gains['total_at_risk']:,}
Total GMV at Risk: R$ {gains['gmv_at_risk']:,.2f}
Average GMV per at-risk seller: R$ {gains['avg_seller_gmv']:,.2f}

BUSINESS IMPACT PROJECTION
--------------------------
Assumptions:
- Intervention Success Rate: {gains['intervention_success_rate']*100:.0f}%
- Projected Retention Period: 3 months of GMV

Projected Outcomes with Intervention:
- Sellers Saved: {gains['projected_sellers_saved']:,}
- GMV Protected: R$ {gains['projected_gmv_saved']:,.2f}
- ROI Potential: High (intervention cost << GMV at risk)

PRIORITY TIERS
--------------
"""
        if len(intervention_df) > 0:
            # Immediate urgency
            immediate = intervention_df[intervention_df["urgency"] == "Immediate"]
            if len(immediate) > 0:
                report += f"\n🔴 IMMEDIATE ACTION (≤7 days to dormancy): {len(immediate)} sellers"
                report += f"\n   GMV at Risk: R$ {immediate['total_gmv'].sum():,.2f}"
                report += f"\n   Top Priority Sellers:\n"
                for idx, row in immediate.head(5).iterrows():
                    report += f"   - {row['seller_id'][:8]}... | {row['seller_state']} | GMV: R$ {row['total_gmv']:,.0f} | Risk: {row['overall_churn_risk']:.1%}\n"

            # High urgency
            high = intervention_df[intervention_df["urgency"] == "High"]
            if len(high) > 0:
                report += f"\n🟠 HIGH PRIORITY (8-14 days): {len(high)} sellers"
                report += f"\n   GMV at Risk: R$ {high['total_gmv'].sum():,.2f}\n"

            # Medium urgency
            medium = intervention_df[intervention_df["urgency"] == "Medium"]
            if len(medium) > 0:
                report += f"\n🟡 MEDIUM PRIORITY (15-30 days): {len(medium)} sellers"
                report += f"\n   GMV at Risk: R$ {medium['total_gmv'].sum():,.2f}\n"

        report += f"""

RECOMMENDED INTERVENTION ACTIONS
--------------------------------
1. IMMEDIATE (Next 48 hours):
   - Personal call from Account Manager to top 10 priority sellers
   - Offer: Dedicated support + platform fee waiver for 1 month
   - Goal: Understand pain points, provide immediate assistance

2. SHORT-TERM (Next 2 weeks):
   - Deploy automated email sequence to all High urgency sellers
   - Include: Success stories, feature tutorials, support resources
   - Offer: Group webinar with top-performing sellers

3. ONGOING (Monthly):
   - Review intervention effectiveness (track saved sellers)
   - Refine targeting based on response rates
   - Scale successful intervention patterns

================================================================================
END OF REPORT
================================================================================
"""
        return report

    def export_intervention_data(
        self, intervention_df: pd.DataFrame, report: str
    ) -> None:
        """Export intervention data and report to files."""

        # Export CSV with full intervention list
        if len(intervention_df) > 0:
            csv_path = os.path.join(
                self.settings.OUTPUT_PATH, "intervention_priority_list.csv"
            )
            intervention_df.to_csv(csv_path, index=False)
            logger.info(
                f"  Exported: intervention_priority_list.csv ({len(intervention_df):,} sellers)"
            )

        # Export text report
        report_path = os.path.join(
            self.settings.OUTPUT_PATH, "intervention_priorities.txt"
        )
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"  Exported: intervention_priorities.txt")


def main():
    """
    Full end-to-end orchestrator for the Olist Seller Churn Analysis Pipeline.

    Execution order:
        1.  Load & preprocess raw data
        2.  Build seller master table with activity metrics
        3.  Apply churn labels (never_activated / dormant / active)
        4.  Cohort & segment analyses
        5.  Feature engineering (pre-activation + retention)
        6.  Train Pre-Activation Model  → best of LogReg, RF, GBM
        7.  Train Retention Model       → best of LogReg, RF, GBM
        8.  Evaluate both models        → charts saved to reports/figures/
        9.  Score all sellers           → overall_churn_risk + risk_category
        10. Build intervention priority list
        11. Export CSVs                 → output/
        12. Generate reports/churn_insights_report.md
        13. Generate reports/model_evaluation.md
    """
    import os
    import joblib
    import numpy as np
    from sklearn.model_selection import train_test_split

    from src.features import FeatureEngineer
    from src.models import ChurnModeler
    from src.evaluation import ModelEvaluator
    from src.reports import InsightsReporter

    settings = get_settings()

    # ------------------------------------------------------------------
    # Logging — write to logs/, not project root
    # ------------------------------------------------------------------
    logs_dir = settings.OUTPUT_PATH.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        logs_dir / f'churn_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file)),
        ],
    )
    _logger = logging.getLogger(__name__)
    _logger.info("=" * 80)
    _logger.info("OLIST SELLER CHURN ANALYSIS PIPELINE")
    _logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 1. Data loading & preprocessing
    # ------------------------------------------------------------------
    loader = DataLoader(settings)
    loader.load_all()
    loader.parse_dates()
    master = loader.build_seller_master_table()

    preprocessor = DataPreprocessor(settings)
    master = preprocessor.calculate_seller_activity_metrics(master, loader.data)
    master = preprocessor.handle_missing_values(master)

    if len(master) == 0:
        _logger.error("No data loaded. Check your DATA_PATH setting.")
        return

    _logger.info(f"\nProcessing {len(master):,} sellers...")

    # ------------------------------------------------------------------
    # 2. Churn labelling
    # ------------------------------------------------------------------
    analyzer = ChurnAnalyzer(settings)
    master = analyzer.define_churn_labels(master)

    # ------------------------------------------------------------------
    # 3. Cohort & segment analyses
    # ------------------------------------------------------------------
    cohort = analyzer.cohort_analysis(master)
    segment_analyses: Dict[str, Optional[pd.DataFrame]] = {
        "business_segment": analyzer.segment_analysis(master, "business_segment"),
        "lead_behaviour_profile": analyzer.segment_analysis(
            master, "lead_behaviour_profile"
        ),
        "lead_type": analyzer.segment_analysis(master, "lead_type"),
        "seller_state": analyzer.segment_analysis(master, "seller_state"),
    }

    model_metrics: Dict[str, Dict] = {}
    pre_model = ret_model = None

    if len(master) > 10:
        engineer = FeatureEngineer(settings)
        evaluator = ModelEvaluator(settings)

        # ------------------------------------------------------------------
        # 4. Pre-Activation Model
        # ------------------------------------------------------------------
        full_pre_feats = engineer.create_pre_activation_features(master.copy())
        y_pre = master["never_activated"]

        if len(full_pre_feats) > 10 and y_pre.nunique() > 1:
            X_pre_train, X_pre_test, y_pre_train, y_pre_test = train_test_split(
                full_pre_feats,
                y_pre,
                test_size=settings.TEST_SIZE,
                random_state=settings.RANDOM_STATE,
                stratify=y_pre,
            )
            modeler_pre = ChurnModeler(settings)
            pre_model, _ = modeler_pre.train_pre_activation_model(
                X_pre_train, y_pre_train
            )

            if pre_model is not None:
                joblib.dump(
                    pre_model,
                    os.path.join(
                        str(settings.MODELS_PATH), "pre_activation_model.joblib"
                    ),
                )
                fi_pre = (
                    next(
                        (
                            v
                            for k, v in modeler_pre.feature_importance.items()
                            if "GradientBoosting" in k or "RandomForest" in k
                        ),
                        next(iter(modeler_pre.feature_importance.values()), None),
                    )
                    if modeler_pre.feature_importance
                    else None
                )
                model_metrics["pre_activation"] = evaluator.evaluate_model(
                    pre_model,
                    X_pre_test,
                    y_pre_test,
                    model_name="Pre-Activation Model",
                    feature_importance=fi_pre,
                )

        # ------------------------------------------------------------------
        # 5. Retention Model  (activated sellers only)
        # ------------------------------------------------------------------
        activated = master[master["never_activated"] == 0].copy()
        if len(activated) > 10:
            full_ret_feats = engineer.create_retention_features(activated)
            y_ret = activated["dormant"].loc[full_ret_feats.index]

            if y_ret.nunique() > 1:
                X_ret_train, X_ret_test, y_ret_train, y_ret_test = train_test_split(
                    full_ret_feats,
                    y_ret,
                    test_size=settings.TEST_SIZE,
                    random_state=settings.RANDOM_STATE,
                    stratify=y_ret,
                )
                modeler_ret = ChurnModeler(settings)
                ret_model, _ = modeler_ret.train_retention_model(
                    X_ret_train, y_ret_train
                )

                if ret_model is not None:
                    joblib.dump(
                        ret_model,
                        os.path.join(
                            str(settings.MODELS_PATH), "retention_model.joblib"
                        ),
                    )
                    fi_ret = (
                        next(
                            (
                                v
                                for k, v in modeler_ret.feature_importance.items()
                                if "GradientBoosting" in k or "RandomForest" in k
                            ),
                            next(iter(modeler_ret.feature_importance.values()), None),
                        )
                        if modeler_ret.feature_importance
                        else None
                    )
                    model_metrics["retention"] = evaluator.evaluate_model(
                        ret_model,
                        X_ret_test,
                        y_ret_test,
                        model_name="Retention Model",
                        feature_importance=fi_ret,
                    )

        # Save technical evaluation report
        evaluator.save_report("model_evaluation.md")

        # ------------------------------------------------------------------
        # 6. Risk scoring — full dataset
        # ------------------------------------------------------------------
        if pre_model is not None:
            _logger.info("\n--- Scoring all sellers ---")
            master_pre_feats = engineer.create_pre_activation_features(master.copy())
            master["never_activated_risk"] = pre_model.predict_proba(master_pre_feats)[
                :, 1
            ]

            if ret_model is not None:
                activated_idx = master[master["never_activated"] == 0].index
                master_ret_feats = engineer.create_retention_features(
                    master.loc[activated_idx].copy()
                ).replace([np.inf, -np.inf], 0)
                master.loc[activated_idx, "dormancy_risk"] = ret_model.predict_proba(
                    master_ret_feats
                )[:, 1]

            master["overall_churn_risk"] = master.apply(
                lambda r: r["never_activated_risk"]
                if pd.isna(r.get("dormancy_risk"))
                else max(r["never_activated_risk"], r.get("dormancy_risk", 0)),
                axis=1,
            )
            master["risk_category"] = pd.cut(
                master["overall_churn_risk"],
                bins=[0, 0.3, 0.6, 0.8, 1.0],
                labels=["Low", "Medium", "High", "Critical"],
            )

    # ------------------------------------------------------------------
    # 7. Export CSVs & legacy text outputs
    # ------------------------------------------------------------------
    exporter = ResultsExporter(settings)
    exporter.export_csv(master, "seller_master.csv")
    exporter.export_csv(cohort, "cohort_analysis.csv")
    if "overall_churn_risk" in master.columns:
        exporter.export_csv(master, "seller_risk_scores.csv")
    for seg_col, seg_df in segment_analyses.items():
        if seg_df is not None:
            exporter.export_csv(seg_df, f"segment_analysis_{seg_col}.csv")
    exporter.generate_summary_report(master, cohort)

    if "overall_churn_risk" in master.columns:
        prioritizer = InterventionPrioritizer(settings)
        intervention_df = prioritizer.generate_intervention_list(master)
        gains = prioritizer.calculate_business_gains(intervention_df, master)
        report_text = prioritizer.generate_intervention_report(intervention_df, gains)
        prioritizer.export_intervention_data(intervention_df, report_text)

    # ------------------------------------------------------------------
    # 8. EDA visualisations
    # ------------------------------------------------------------------
    visualizer = ChurnVisualizer(settings)
    visualizer.plot_churn_distribution(master)
    visualizer.plot_cohort_analysis(cohort)
    if "overall_churn_risk" in master.columns:
        visualizer.plot_risk_distribution(master)

    # ------------------------------------------------------------------
    # 9. Stakeholder report → reports/churn_insights_report.md
    # ------------------------------------------------------------------
    reporter = InsightsReporter(settings)
    reporter.build(
        seller_master=master,
        risk_scores=master,
        cohort=cohort,
        segment_analyses=segment_analyses,
        model_metrics=model_metrics,
    )
    reporter.save("churn_insights_report.md")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    _logger.info("\n" + "=" * 80)
    _logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    _logger.info(f"  Outputs  → {settings.OUTPUT_PATH}/")
    _logger.info(f"    ├── *.csv / *.txt          (data & summaries)")
    _logger.info(f"    ├── figures/               (ROC, PR, confusion matrix, feature importance)")
    _logger.info(f"    ├── churn_insights_report.md")
    _logger.info(f"    └── model_evaluation.md")
    _logger.info(f"  Models   → {settings.MODELS_PATH}/")
    _logger.info("=" * 80)


if __name__ == "__main__":
    main()
