"""
Model Evaluation Module
=======================

Provides comprehensive model evaluation capabilities:
- ROC curves and confusion matrices (saved to reports/figures/)
- Precision-Recall curves
- Performance metrics calculation
- Markdown report generation (saved to reports/)

All outputs are written through Settings — no hardcoded paths.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from config.settings import Settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates trained churn models and generates stakeholder-ready reports.

    Saves:
    - PNG charts → reports/figures/
    - model_evaluation.md → reports/
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.report_content: List[str] = []
        self._figures_path = settings.FIGURES_PATH
        self._figures_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _add_section(self, title: str, content: str) -> None:
        """Append a ## section to the report buffer."""
        self.report_content.append(f"## {title}\n\n{content}\n")

    def _add_image(self, title: str, filename: str) -> None:
        """Append an image reference (relative path) to the report buffer."""
        # Relative path so the markdown renders from the reports/ root
        self.report_content.append(f"### {title}\n\n![{title}](figures/{filename})\n")

    def _save_figure(self, filename: str) -> None:
        """Save the current matplotlib figure to reports/figures/."""
        path = self._figures_path / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved figure: outputs/figures/{filename}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        feature_importance: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Evaluate a trained model: compute metrics, generate plots, buffer report.

        Args:
            model: A fitted sklearn estimator
            X_test: Test feature matrix
            y_test: True labels
            model_name: Human-readable model name (used in filenames + report)
            feature_importance: Optional DataFrame with Feature / Importance columns

        Returns:
            Dict of scalar evaluation metrics
        """
        logger.info(f"Evaluating {model_name} ...")

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Scalar metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        class_report = classification_report(y_test, y_pred, zero_division=0)

        # Report section: numeric metrics
        metrics_summary = (
            f"- **Accuracy:** {accuracy:.3f}\n"
            f"- **Precision:** {precision:.3f}\n"
            f"- **Recall:** {recall:.3f}\n"
            f"- **F1 Score:** {f1:.3f}\n"
            f"- **ROC AUC:** {roc_auc:.3f}\n\n"
            "```\n" + class_report + "\n```"
        )
        self._add_section(f"{model_name} — Performance Metrics", metrics_summary)

        # Plots
        slug = model_name.replace(" ", "_").lower()
        self._plot_roc_curve(fpr, tpr, roc_auc, model_name, f"roc_curve_{slug}.png")
        self._plot_precision_recall(y_test, y_prob, model_name, f"pr_curve_{slug}.png")
        self._plot_confusion_matrix(
            y_test, y_pred, model_name, f"confusion_matrix_{slug}.png"
        )
        if feature_importance is not None and len(feature_importance) > 0:
            self._plot_feature_importance(
                feature_importance, model_name, f"feature_importance_{slug}.png"
            )

        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
        }

    # -------------------------------------------------------------------------
    # Plot methods
    # -------------------------------------------------------------------------

    def _plot_roc_curve(
        self, fpr, tpr, roc_auc: float, model_name: str, filename: str
    ) -> None:
        """ROC curve with AUC annotation."""
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#6C5CE7", lw=2.5, label=f"AUC = {roc_auc:.3f}")
        ax.plot(
            [0, 1],
            [0, 1],
            color="#b2bec3",
            lw=1.5,
            linestyle="--",
            label="Random classifier",
        )
        ax.fill_between(fpr, tpr, alpha=0.08, color="#6C5CE7")
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure(filename)
        self._add_image(f"ROC Curve: {model_name}", filename)

    def _plot_precision_recall(
        self, y_test: pd.Series, y_prob: np.ndarray, model_name: str, filename: str
    ) -> None:
        """Precision-Recall curve."""
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(rec, prec)
        baseline = y_test.mean()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(rec, prec, color="#00B894", lw=2.5, label=f"AP = {pr_auc:.3f}")
        ax.axhline(
            baseline,
            color="#b2bec3",
            lw=1.5,
            linestyle="--",
            label=f"Baseline ({baseline:.2f})",
        )
        ax.fill_between(rec, prec, alpha=0.08, color="#00B894")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(
            f"Precision-Recall Curve — {model_name}", fontsize=14, fontweight="bold"
        )
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure(filename)
        self._add_image(f"Precision-Recall Curve: {model_name}", filename)

    def _plot_confusion_matrix(
        self, y_test: pd.Series, y_pred: np.ndarray, model_name: str, filename: str
    ) -> None:
        """Annotated confusion matrix heatmap."""
        cm = confusion_matrix(y_test, y_pred)
        labels = ["Not Churned", "Churned"]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"size": 14},
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        self._save_figure(filename)
        self._add_image(f"Confusion Matrix: {model_name}", filename)

    def _plot_feature_importance(
        self, importance_df: pd.DataFrame, model_name: str, filename: str
    ) -> None:
        """Horizontal bar chart of top-10 feature importances."""
        top = importance_df.head(10).sort_values("Importance")

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.barh(
            top["Feature"], top["Importance"], color="#4834D4", edgecolor="white"
        )
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )
        ax.set_title(
            f"Feature Importance — {model_name}", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel("Importance", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        self._save_figure(filename)
        self._add_image(f"Feature Importance: {model_name}", filename)

    # -------------------------------------------------------------------------
    # Report persistence
    # -------------------------------------------------------------------------

    def save_report(self, filename: str = "model_evaluation.md") -> str:
        """
        Write the accumulated report buffer to reports/<filename>.

        Returns:
            Absolute path to the saved file.
        """
        header = (
            "# Churn Prediction Models — Technical Evaluation Report\n\n"
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
            "---\n\n"
        )
        full_content = header + "\n".join(self.report_content)

        path = self.settings.REPORTS_PATH / filename
        path.write_text(full_content, encoding="utf-8")
        logger.info(f"Evaluation report saved: outputs/{filename}")
        return str(path)
