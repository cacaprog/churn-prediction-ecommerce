"""
Data validation schemas using Pandera.
Fail fast with clear errors - prevent garbage-in-garbage-out.
"""

from typing import Dict

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Schema for the master seller dataset after processing
seller_master_schema = DataFrameSchema(
    {
        "seller_id": Column(str, unique=True, nullable=False),
        "won_date": Column(pa.DateTime, nullable=False),
        "total_orders": Column(int, Check.greater_than_or_equal_to(0), nullable=True),
        "total_gmv": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
        "never_activated": Column(int, Check.isin([0, 1]), nullable=True),
        "dormant": Column(int, Check.isin([0, 1]), nullable=True),
        "active": Column(int, Check.isin([0, 1]), nullable=True),
        "churned": Column(int, Check.isin([0, 1]), nullable=True),
    },
    strict=False,
)


# Schema for raw MQL dataset
mql_schema = DataFrameSchema(
    {
        "mql_id": Column(str, unique=True, nullable=False),
        "first_contact_date": Column(pa.DateTime, nullable=True),
        "landing_page_id": Column(str, nullable=True),
        "origin": Column(str, nullable=True),
    },
    strict=False,
)


# Schema for raw deals dataset
deals_schema = DataFrameSchema(
    {
        "mql_id": Column(str, nullable=False),
        "seller_id": Column(str, unique=True, nullable=False),
        "won_date": Column(pa.DateTime, nullable=False),
    },
    strict=False,
)


# Schema for risk-scored output
risk_scores_schema = DataFrameSchema(
    {
        "seller_id": Column(str, unique=True, nullable=False),
        "never_activated_risk": Column(
            float,
            [Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(1)],
            nullable=True,
        ),
        "dormancy_risk": Column(
            float,
            [Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(1)],
            nullable=True,
        ),
        "overall_churn_risk": Column(
            float,
            [Check.greater_than_or_equal_to(0), Check.less_than_or_equal_to(1)],
            nullable=False,
        ),
        "risk_category": Column(
            str, Check.isin(["Low", "Medium", "High", "Critical"]), nullable=True
        ),
    },
    strict=False,
)


def validate_seller_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate seller master data against schema.

    Args:
        df: DataFrame to validate

    Returns:
        Validated DataFrame

    Raises:
        SchemaError: If validation fails
    """
    return seller_master_schema.validate(df)


def validate_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate risk score output against schema.

    Args:
        df: DataFrame to validate

    Returns:
        Validated DataFrame
    """
    return risk_scores_schema.validate(df)


def get_dataset_schema(dataset_name: str) -> DataFrameSchema:
    """
    Get the appropriate schema for a dataset.

    Args:
        dataset_name: Name of the dataset ('mqls', 'deals', etc.)

    Returns:
        DataFrameSchema for the dataset
    """
    schemas: Dict[str, DataFrameSchema] = {
        "mqls": mql_schema,
        "deals": deals_schema,
    }
    return schemas.get(dataset_name)
