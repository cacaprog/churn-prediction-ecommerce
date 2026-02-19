"""
Configuration management using Pydantic Settings.
Centralizes all configuration - no more hardcoded paths.
"""

from pathlib import Path
from typing import List, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the Olist churn prediction pipeline."""

    # Paths
    DATA_PATH: Path = Field(default=Path("./data/raw"))
    OUTPUT_PATH: Path = Field(default=Path("./outputs"))
    MODELS_PATH: Path = Field(default=Path("./models"))
    REPORTS_PATH: Path = Field(
        default=Path("./outputs")
    )  # markdown reports alongside CSVs
    FIGURES_PATH: Path = Field(default=Path("./outputs/figures"))  # all charts
    PROCESSED_PATH: Path = Field(default=Path("./data/processed"))

    # Business rules as config
    NEVER_ACTIVATED_DAYS: int = 90
    DORMANT_DAYS: int = 60

    # Model Settings
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.25
    CV_FOLDS: int = 5

    # MLflow
    MLFLOW_TRACKING_URI: str = "mlruns"  # local folder; set to remote URL in .env
    MLFLOW_EXPERIMENT_NAME: str = "olist-churn-prediction"

    # Visual Settings
    FIG_SIZE: Tuple[int, int] = (12, 6)
    COLOR_PALETTE: List[str] = Field(
        default_factory=lambda: ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore"
    )


def get_settings() -> Settings:
    """Get settings instance, creating directories if needed."""
    settings = Settings()

    # Ensure directories exist
    for path in [
        settings.OUTPUT_PATH,
        settings.MODELS_PATH,
        settings.FIGURES_PATH,
        settings.PROCESSED_PATH,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return settings
