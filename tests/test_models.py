"""
Tests for Model Training & Evaluation Logic
============================================

Tests the pure-logic parts of ModelEvaluator and ChurnModeler without
requiring real training data or saved model files.
"""

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from config.settings import Settings
from src.models import ChurnModeler, ModelEvaluator


@pytest.fixture
def settings():
    return Settings(
        DATA_PATH="./data/raw", OUTPUT_PATH="./output", MODELS_PATH="./models"
    )


@pytest.fixture
def evaluator(settings):
    return ModelEvaluator(settings)


# ── ModelEvaluator ────────────────────────────────────────────────────────────


class TestModelEvaluator:
    """Tests for the ModelEvaluator helper class."""

    def test_evaluate_model_returns_required_keys(self, settings):
        """evaluate_model() must return all expected metric keys."""
        evaluator = ModelEvaluator(settings)

        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5], "f2": [5, 4, 3, 2, 1]})
        y = pd.Series([0, 1, 0, 1, 0])

        model = DummyClassifier(strategy="most_frequent")
        model.fit(X, y)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = evaluator.evaluate_model("DummyModel", model, X, y, y_proba)

        expected_keys = {
            "model_name",
            "auc_roc",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_evaluate_model_name_preserved(self, settings):
        """model_name must be stored exactly as passed."""
        evaluator = ModelEvaluator(settings)

        X = pd.DataFrame({"f": [0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1])
        model = DummyClassifier(strategy="stratified", random_state=42)
        model.fit(X, y)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = evaluator.evaluate_model("MyModel", model, X, y, y_proba)
        assert metrics["model_name"] == "MyModel"

    def test_evaluate_model_metric_range(self, settings):
        """All scalar metrics must be in [0, 1]."""
        evaluator = ModelEvaluator(settings)

        X = pd.DataFrame({"f": [0, 1, 0, 1, 0, 1]})
        y = pd.Series([0, 1, 0, 1, 0, 1])
        model = DummyClassifier(strategy="stratified", random_state=0)
        model.fit(X, y)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = evaluator.evaluate_model("Test", model, X, y, y_proba)

        for key in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of [0,1]: {metrics[key]}"

    def test_select_best_model_picks_highest_auc(self, settings):
        """select_best_model() should return the entry with the highest auc_roc."""
        evaluator = ModelEvaluator(settings)
        evaluator.results = {
            "ModelA": {"auc_roc": 0.65, "accuracy": 0.7},
            "ModelB": {"auc_roc": 0.82, "accuracy": 0.75},
            "ModelC": {"auc_roc": 0.74, "accuracy": 0.72},
        }

        best_name, best_metrics = evaluator.select_best_model()

        assert best_name == "ModelB"
        assert best_metrics["auc_roc"] == 0.82

    def test_select_best_model_single_entry(self, settings):
        """select_best_model() works when only one model is registered."""
        evaluator = ModelEvaluator(settings)
        evaluator.results = {"OnlyModel": {"auc_roc": 0.55}}

        best_name, _ = evaluator.select_best_model()
        assert best_name == "OnlyModel"


# ── ChurnModeler single-class guard ──────────────────────────────────────────


class TestChurnModelerGuards:
    """
    ChurnModeler must handle degenerate datasets (only one class in target)
    gracefully, returning (None, feature_cols) instead of raising.
    """

    def test_pre_activation_single_class_returns_none(self, settings):
        """Pre-activation model skips training when target has only one class."""
        modeler = ChurnModeler(settings)

        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y = pd.Series([0, 0, 0])  # all same class

        model, feature_cols = modeler.train_pre_activation_model(X, y)

        assert model is None
        assert feature_cols == ["f1", "f2"]

    def test_retention_single_class_returns_none(self, settings):
        """Retention model skips training when target has only one class."""
        modeler = ChurnModeler(settings)

        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y = pd.Series([1, 1, 1])  # all same class

        model, feature_cols = modeler.train_retention_model(X, y)

        assert model is None
        assert feature_cols == ["f1", "f2"]

    def test_pre_activation_returns_feature_cols(self, settings):
        """Feature names must be returned even when training is skipped."""
        modeler = ChurnModeler(settings)

        X = pd.DataFrame({"gmv": [100, 200], "orders": [5, 10]})
        y = pd.Series([0, 0])

        _, cols = modeler.train_pre_activation_model(X, y)
        assert set(cols) == {"gmv", "orders"}
