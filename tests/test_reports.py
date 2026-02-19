"""
Tests for Reports & Insights Modules
======================================

Tests the pure-logic helpers of InsightsReporter (reports.py) and
the report-buffer helpers of ModelEvaluator (evaluation.py) without
requiring file I/O or trained models.
"""

from pathlib import Path

import pandas as pd
import pytest

from config.settings import Settings

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_settings(tmp_path):
    """Settings pointing to a real temp directory so mkdir() doesn't fail."""
    return Settings(
        DATA_PATH=str(tmp_path / "data/raw"),
        OUTPUT_PATH=str(tmp_path / "outputs"),
        MODELS_PATH=str(tmp_path / "models"),
        REPORTS_PATH=str(tmp_path / "outputs"),
        FIGURES_PATH=str(tmp_path / "outputs/figures"),
        PROCESSED_PATH=str(tmp_path / "data/processed"),
    )


@pytest.fixture
def master_df():
    """Minimal seller master DataFrame with required churn columns."""
    return pd.DataFrame(
        {
            "seller_id": ["A", "B", "C", "D"],
            "churned": [1, 1, 0, 1],
            "never_activated": [1, 0, 0, 0],
            "dormant": [0, 1, 0, 1],
            "active": [0, 0, 1, 0],
            "total_gmv": [500.0, 1500.0, 3000.0, 800.0],
        }
    )


@pytest.fixture
def risk_df():
    """Minimal risk-scores DataFrame."""
    return pd.DataFrame(
        {
            "seller_id": ["A", "B", "C", "D"],
            "overall_churn_risk": [0.9, 0.7, 0.2, 0.85],
            "risk_category": ["Critical", "High", "Low", "High"],
            "active": [0, 0, 1, 0],
            "total_gmv": [500.0, 1500.0, 3000.0, 800.0],
        }
    )


@pytest.fixture
def cohort_df():
    """Minimal cohort DataFrame."""
    return pd.DataFrame(
        {
            "cohort_month": ["2017-06", "2017-07", "2017-08"],
            "total_sellers": [50, 60, 55],
            "never_activated": [20, 25, 22],
            "dormant": [10, 12, 11],
            "active": [20, 23, 22],
            "churn_rate": [60.0, 61.7, 60.0],
            "activation_rate": [40.0, 38.3, 40.0],
        }
    )


# ── InsightsReporter ──────────────────────────────────────────────────────────


class TestInsightsReporter:
    """Tests for the InsightsReporter class in reports.py."""

    def test_build_populates_sections(
        self, tmp_settings, master_df, risk_df, cohort_df
    ):
        """build() should populate at least the core report sections."""
        from src.reports import InsightsReporter

        reporter = InsightsReporter(tmp_settings)
        reporter.build(
            seller_master=master_df,
            risk_scores=risk_df,
            cohort=cohort_df,
        )

        assert len(reporter._sections) > 0

    def test_build_returns_self_for_chaining(
        self, tmp_settings, master_df, risk_df, cohort_df
    ):
        """build() must return self to enable fluent chaining."""
        from src.reports import InsightsReporter

        reporter = InsightsReporter(tmp_settings)
        result = reporter.build(
            seller_master=master_df,
            risk_scores=risk_df,
            cohort=cohort_df,
        )
        assert result is reporter

    def test_save_writes_file(self, tmp_settings, master_df, risk_df, cohort_df):
        """save() writes a .md file with the report content."""
        from src.reports import InsightsReporter

        reporter = InsightsReporter(tmp_settings)
        reporter.build(
            seller_master=master_df,
            risk_scores=risk_df,
            cohort=cohort_df,
        )
        path_str = reporter.save("test_report.md")
        path = Path(path_str)

        assert path.exists()
        content = path.read_text()
        assert "Olist Seller Churn" in content

    def test_add_table_produces_markdown(self, tmp_settings):
        """_add_table() should produce a string with Markdown table separators."""
        from src.reports import InsightsReporter

        reporter = InsightsReporter(tmp_settings)
        df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [0.91, 0.78]})
        result = reporter._add_table(df)

        assert "|" in result
        assert "---" in result
        assert "Alice" in result

    def test_add_table_respects_max_rows(self, tmp_settings):
        """_add_table() should cap output at max_rows."""
        from src.reports import InsightsReporter

        reporter = InsightsReporter(tmp_settings)
        df = pd.DataFrame({"x": range(20)})
        result = reporter._add_table(df, max_rows=5)

        # Header + separator + 5 data rows = 7 lines
        lines = [ln for ln in result.split("\n") if ln.strip()]
        assert len(lines) == 7

    def test_h2_appends_section(self, tmp_settings):
        """_h2() must append a section starting with '## '."""
        from src.reports import InsightsReporter

        reporter = InsightsReporter(tmp_settings)
        reporter._h2("Test Section", "Some body text.")

        assert len(reporter._sections) == 1
        assert reporter._sections[0].startswith("## Test Section")

    def test_build_resets_sections_on_second_call(
        self, tmp_settings, master_df, risk_df, cohort_df
    ):
        """Calling build() twice should not accumulate duplicate sections."""
        from src.reports import InsightsReporter

        reporter = InsightsReporter(tmp_settings)
        reporter.build(seller_master=master_df, risk_scores=risk_df, cohort=cohort_df)
        count_first = len(reporter._sections)

        reporter.build(seller_master=master_df, risk_scores=risk_df, cohort=cohort_df)
        count_second = len(reporter._sections)

        assert count_first == count_second


# ── ModelEvaluator (evaluation.py) report-buffer helpers ─────────────────────


class TestModelEvaluatorReportBuffer:
    """Tests for the report-buffer helpers in src/evaluation.py ModelEvaluator."""

    def test_add_section_appends_to_buffer(self, tmp_settings):
        """_add_section() must append a '## '-prefixed entry."""
        from src.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(tmp_settings)
        evaluator._add_section("My Section", "Detail text here.")

        assert len(evaluator.report_content) == 1
        assert evaluator.report_content[0].startswith("## My Section")
        assert "Detail text here." in evaluator.report_content[0]

    def test_add_image_appends_markdown_image(self, tmp_settings):
        """_add_image() must produce a valid Markdown image reference."""
        from src.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(tmp_settings)
        evaluator._add_image("ROC Curve", "roc_curve_lr.png")

        assert len(evaluator.report_content) == 1
        assert "![ROC Curve]" in evaluator.report_content[0]
        assert "roc_curve_lr.png" in evaluator.report_content[0]

    def test_save_report_creates_file(self, tmp_settings):
        """save_report() must write the buffered content to disk."""
        from src.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(tmp_settings)
        evaluator._add_section("Section 1", "Content A.")
        evaluator._add_section("Section 2", "Content B.")

        path_str = evaluator.save_report("test_eval.md")
        path = Path(path_str)

        assert path.exists()
        text = path.read_text()
        assert "Churn Prediction Models" in text
        assert "Content A." in text
        assert "Content B." in text

    def test_save_report_includes_header_timestamp(self, tmp_settings):
        """The saved report must include the generated timestamp header."""
        from src.evaluation import ModelEvaluator

        evaluator = ModelEvaluator(tmp_settings)
        path_str = evaluator.save_report("ts_test.md")
        content = Path(path_str).read_text()

        assert "_Generated:" in content
