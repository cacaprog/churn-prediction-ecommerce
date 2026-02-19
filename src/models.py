"""
Model Training Module
====================

Handles training and evaluation of churn prediction models:
- Pre-activation model (predicts never-activated sellers)
- Retention model (predicts dormant churn for activated sellers)
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from config.settings import Settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates and compares multiple models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.results: Dict[str, Dict] = {}

    def evaluate_model(
        self,
        name: str,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_proba: np.ndarray,
    ) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        y_pred = model.predict(X_test)

        metrics = {
            "model_name": name,
            "auc_roc": roc_auc_score(y_test, y_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        return metrics

    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform cross-validation."""
        cv = StratifiedKFold(
            n_splits=self.settings.CV_FOLDS,
            shuffle=True,
            random_state=self.settings.RANDOM_STATE,
        )

        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        return {"cv_auc_mean": auc_scores.mean(), "cv_auc_std": auc_scores.std()}

    def select_best_model(self) -> Tuple[str, Dict]:
        """Select best model based on AUC-ROC."""
        best_name = max(self.results, key=lambda x: self.results[x]["auc_roc"])
        return best_name, self.results[best_name]


class ChurnModeler:
    """Handles training and evaluation of churn prediction models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: Dict[str, Any] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.evaluator = ModelEvaluator(settings)

    def train_pre_activation_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Any, List[str]]:
        """
        Train model to predict never-activated sellers.

        Args:
            X: Feature matrix
            y: Target vector (never_activated)

        Returns:
            Tuple of (best_model, feature_columns)
        """
        logger.info("\n" + "=" * 60)
        logger.info("SECTION 4: MODEL TRAINING & EVALUATION")
        logger.info("=" * 60)
        logger.info("\n--- Training Pre-Activation Model ---")

        feature_cols = list(X.columns)

        # Check if we have both classes
        if y.nunique() < 2:
            logger.warning(
                f"Only one class present in target: {y.unique()}. Skipping model training."
            )
            return None, feature_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.settings.TEST_SIZE,
            random_state=self.settings.RANDOM_STATE,
            stratify=y,
        )

        # Train multiple models
        models_to_train = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.settings.RANDOM_STATE,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=self.settings.RANDOM_STATE,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=self.settings.RANDOM_STATE
            ),
        }

        logger.info("\nModel Comparison (Pre-Activation):")
        logger.info("-" * 70)
        logger.info(f"{'Model':<20} {'AUC-ROC':>10} {'Accuracy':>10} {'F1-Score':>10}")
        logger.info("-" * 70)

        for name, model in models_to_train.items():
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = self.evaluator.evaluate_model(
                name, model, X_test, y_test, y_proba
            )
            self.evaluator.results[name] = metrics

            logger.info(
                f"{name:<20} {metrics['auc_roc']:>10.3f} "
                f"{metrics['accuracy']:>10.3f} {metrics['f1_score']:>10.3f}"
            )

            # Store feature importance for tree-based models
            if hasattr(model, "feature_importances_"):
                self.feature_importance[name] = pd.DataFrame(
                    {"Feature": feature_cols, "Importance": model.feature_importances_}
                ).sort_values("Importance", ascending=False)

        # Select best model
        best_name, best_metrics = self.evaluator.select_best_model()
        best_model = models_to_train[best_name]

        logger.info("-" * 70)
        logger.info(f"Best Model: {best_name} (AUC-ROC: {best_metrics['auc_roc']:.3f})")

        self.models["pre_activation"] = best_model

        return best_model, feature_cols

    def train_retention_model(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Any, List[str]]:
        """
        Train model to predict dormant churn for activated sellers.

        Args:
            X: Feature matrix (for activated sellers only)
            y: Target vector (dormant)

        Returns:
            Tuple of (best_model, feature_columns)
        """
        logger.info("\n--- Training Retention Model ---")

        feature_cols = list(X.columns)

        # Check if we have both classes
        if y.nunique() < 2:
            logger.warning(
                f"Only one class present in target: {y.unique()}. Skipping model training."
            )
            return None, feature_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.settings.TEST_SIZE,
            random_state=self.settings.RANDOM_STATE,
            stratify=y,
        )

        # Train multiple models
        models_to_train = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.settings.RANDOM_STATE,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=self.settings.RANDOM_STATE,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=self.settings.RANDOM_STATE
            ),
        }

        self.evaluator.results = {}  # Reset for retention models

        logger.info("\nModel Comparison (Retention):")
        logger.info("-" * 70)
        logger.info(f"{'Model':<20} {'AUC-ROC':>10} {'Accuracy':>10} {'F1-Score':>10}")
        logger.info("-" * 70)

        for name, model in models_to_train.items():
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = self.evaluator.evaluate_model(
                name, model, X_test, y_test, y_proba
            )
            self.evaluator.results[name] = metrics

            logger.info(
                f"{name:<20} {metrics['auc_roc']:>10.3f} "
                f"{metrics['accuracy']:>10.3f} {metrics['f1_score']:>10.3f}"
            )

            if hasattr(model, "feature_importances_"):
                self.feature_importance[f"{name}_retention"] = pd.DataFrame(
                    {"Feature": feature_cols, "Importance": model.feature_importances_}
                ).sort_values("Importance", ascending=False)

        best_name, best_metrics = self.evaluator.select_best_model()
        best_model = models_to_train[best_name]

        logger.info("-" * 70)
        logger.info(f"Best Model: {best_name} (AUC-ROC: {best_metrics['auc_roc']:.3f})")

        self.models["retention"] = best_model

        return best_model, feature_cols
