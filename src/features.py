"""
Feature Engineering Module
==========================

Creates features for predictive modeling including:
- Pre-activation features (available before first sale)
- Retention features (for activated sellers)
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config.settings import Settings

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for predictive modeling."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def create_pre_activation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for pre-activation churn model.
        These features are available before the seller makes their first sale.
        """
        logger.info("\n--- Creating Pre-Activation Features ---")

        data = df.copy()

        # Categorical features (available before first sale)
        categorical_cols = [
            "lead_behaviour_profile",
            "business_segment",
            "lead_type",
            "business_type",
            "origin",
            "seller_state",
        ]

        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[f"{col}_encoded"] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le

        # Numeric features
        numeric_features = [
            "declared_monthly_revenue",
            "declared_product_catalog_size",
            "sales_cycle_days",
        ]

        for col in numeric_features:
            if col in data.columns:
                data[col] = data[col].fillna(0)

        logger.info(
            f"Created {len(categorical_cols)} encoded + {len(numeric_features)} numeric features"
        )

        # Return only feature columns
        feature_cols = self.get_pre_activation_feature_cols()
        available_cols = [col for col in feature_cols if col in data.columns]
        return data[available_cols]

    def create_retention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for retention model (activated sellers only).
        These features capture seller behavior and engagement patterns.
        """
        logger.info("\n--- Creating Retention Features ---")

        data = df.copy()

        # Activity-based features
        data["gmv_per_order"] = data["total_gmv"] / data["total_orders"].clip(lower=1)
        data["customers_per_order"] = data["unique_customers"] / data[
            "total_orders"
        ].clip(lower=1)
        data["freight_ratio"] = data["total_freight"] / data["total_gmv"].clip(lower=1)

        # Time-based features
        data["activation_speed_score"] = np.where(
            data["days_to_first_sale"] <= 7,
            3,
            np.where(data["days_to_first_sale"] <= 30, 2, 1),
        )

        # Engagement features
        data["is_high_value"] = (
            data["total_gmv"] > data["total_gmv"].quantile(0.75)
        ).astype(int)
        data["is_frequent_seller"] = (
            data["total_orders"] > data["total_orders"].quantile(0.75)
        ).astype(int)

        # Fill infinite values
        data = data.replace([np.inf, -np.inf], 0)

        logger.info("Created 6 additional retention features")

        # Return only feature columns
        feature_cols = self.get_retention_feature_cols()
        available_cols = [col for col in feature_cols if col in data.columns]
        return data[available_cols]

    def get_pre_activation_feature_cols(self) -> List[str]:
        """Return list of pre-activation feature column names."""
        return [
            "lead_behaviour_profile_encoded",
            "business_segment_encoded",
            "lead_type_encoded",
            "business_type_encoded",
            "origin_encoded",
            "seller_state_encoded",
            "declared_monthly_revenue",
            "declared_product_catalog_size",
            "sales_cycle_days",
        ]

    def get_retention_feature_cols(self) -> List[str]:
        """Return list of retention feature column names."""
        return [
            "lead_behaviour_profile_encoded",
            "business_segment_encoded",
            "total_orders",
            "total_gmv",
            "unique_customers",
            "days_active",
            "days_to_first_sale",
            "avg_order_value",
            "gmv_per_order",
            "customers_per_order",
            "activation_speed_score",
        ]
