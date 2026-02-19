"""
Tests for Churn Analysis Pipeline
=================================

Basic tests to validate business logic correctness and data quality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config.settings import Settings, get_settings
from src.pipeline import ChurnAnalyzer, DataLoader, DataPreprocessor


class TestChurnDefinitions:
    """Test business logic correctness."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing churn definitions."""
        return pd.DataFrame(
            {
                "seller_id": ["A", "B", "C", "D", "E"],
                "total_orders": [0, 5, 10, 3, 0],
                "days_to_first_sale": [100, 5, None, 45, 80],
                "days_since_last_sale": [None, 90, 30, 70, None],
                "total_gmv": [0, 1000, 5000, 800, 0],
                "won_date": pd.to_datetime(["2017-01-01"] * 5),
            }
        )

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            DATA_PATH="./data/raw", OUTPUT_PATH="./output", MODELS_PATH="./models"
        )

    def test_never_activated_logic(self, sample_data, settings):
        """Sellers with no orders or >90 days to first sale = churned."""
        analyzer = ChurnAnalyzer(settings)
        result = analyzer.define_churn_labels(sample_data)

        # Seller A: no orders = never_activated
        assert result.loc[result["seller_id"] == "A", "never_activated"].iloc[0] == 1
        # Seller B: has orders, activated quickly = not never_activated
        assert result.loc[result["seller_id"] == "B", "never_activated"].iloc[0] == 0
        # Seller D: 45 days to first sale = not never_activated (under 90)
        assert result.loc[result["seller_id"] == "D", "never_activated"].iloc[0] == 0

    def test_dormant_logic(self, sample_data, settings):
        """Active sellers with >60 days since last sale = dormant."""
        analyzer = ChurnAnalyzer(settings)
        result = analyzer.define_churn_labels(sample_data)

        # Seller B: has orders, last sale 90 days ago = dormant
        assert result.loc[result["seller_id"] == "B", "dormant"].iloc[0] == 1
        # Seller C: last sale 30 days ago = active
        assert result.loc[result["seller_id"] == "C", "dormant"].iloc[0] == 0

    def test_churned_flag(self, sample_data, settings):
        """Churned flag should be set if never_activated or dormant."""
        analyzer = ChurnAnalyzer(settings)
        result = analyzer.define_churn_labels(sample_data)

        # Seller A: never_activated should be churned
        assert result.loc[result["seller_id"] == "A", "churned"].iloc[0] == 1
        # Seller B: dormant should be churned
        assert result.loc[result["seller_id"] == "B", "churned"].iloc[0] == 1
        # Seller C: active should not be churned
        assert result.loc[result["seller_id"] == "C", "churned"].iloc[0] == 0


class TestDataValidation:
    """Test data quality checks."""

    def test_settings_creation(self):
        """Test that settings can be created."""
        settings = get_settings()
        assert settings.NEVER_ACTIVATED_DAYS == 90
        assert settings.DORMANT_DAYS == 60
        assert settings.RANDOM_STATE == 42


class TestFeatureEngineering:
    """Test feature engineering logic."""

    @pytest.fixture
    def sample_features_data(self):
        """Create sample data for feature testing."""
        return pd.DataFrame(
            {
                "seller_id": ["A", "B", "C"],
                "total_orders": [10, 5, 0],
                "total_gmv": [1000, 500, 0],
                "unique_customers": [8, 3, 0],
                "total_freight": [50, 25, 0],
                "days_to_first_sale": [7, 30, None],
                "lead_behaviour_profile": ["Cat", "Wolf", "Unknown"],
                "business_segment": ["Retail", "Wholesale", "Unknown"],
            }
        )

    def test_gmv_per_order_calculation(self, sample_features_data):
        """Test GMV per order feature calculation."""
        from src.features import FeatureEngineer

        settings = Settings()
        engineer = FeatureEngineer(settings)

        result = engineer.create_retention_features(sample_features_data)

        # Check first row (Seller A): 1000 / 10 = 100
        assert result["gmv_per_order"].iloc[0] == 100.0
        # Check second row (Seller B): 500 / 5 = 100
        assert result["gmv_per_order"].iloc[1] == 100.0


class TestConfiguration:
    """Test configuration management."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()

        assert settings.NEVER_ACTIVATED_DAYS == 90
        assert settings.DORMANT_DAYS == 60
        assert settings.RANDOM_STATE == 42
        assert settings.TEST_SIZE == 0.25
        assert settings.CV_FOLDS == 5

    def test_settings_with_custom_values(self):
        """Test that settings can accept custom values."""
        settings = Settings(NEVER_ACTIVATED_DAYS=120, DORMANT_DAYS=90, RANDOM_STATE=123)

        assert settings.NEVER_ACTIVATED_DAYS == 120
        assert settings.DORMANT_DAYS == 90
        assert settings.RANDOM_STATE == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
