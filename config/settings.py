"""
Configuration management using Pydantic Settings.
Centralizes all configuration - no more hardcoded paths.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path
from typing import List, Tuple


class Settings(BaseSettings):
    """Centralized configuration for the Olist churn prediction pipeline."""
    
    # Paths
    DATA_PATH: Path = Field(default=Path("./data/raw"))
    OUTPUT_PATH: Path = Field(default=Path("./output"))
    MODELS_PATH: Path = Field(default=Path("./models"))
    REPORTS_PATH: Path = Field(default=Path("./reports"))
    IMAGES_PATH: Path = Field(default=Path("./output/images"))
    PROCESSED_PATH: Path = Field(default=Path("./data/processed"))
    
    # Business rules as config
    NEVER_ACTIVATED_DAYS: int = 90
    DORMANT_DAYS: int = 60
    
    # Model Settings
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.25
    CV_FOLDS: int = 5
    
    # Visual Settings
    FIG_SIZE: Tuple[int, int] = (12, 6)
    COLOR_PALETTE: List[str] = Field(default_factory=lambda: [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'
    ])
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


def get_settings() -> Settings:
    """Get settings instance, creating directories if needed."""
    settings = Settings()
    
    # Ensure directories exist
    for path in [settings.OUTPUT_PATH, settings.MODELS_PATH, settings.REPORTS_PATH,
                 settings.IMAGES_PATH, settings.PROCESSED_PATH]:
        path.mkdir(parents=True, exist_ok=True)
    
    return settings
