"""
Reports Module
==============

Generates stakeholder-ready Markdown reports from in-memory analysis results.
Replaces the old scripts/generate_churn_insights.py approach of re-reading CSVs.

Outputs:
  - reports/churn_insights_report.md  (executive insights)
  - reports/churn_recommendations.md (filled recommendations from template)

All paths resolved through Settings — no hardcoded strings.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)


class InsightsReporter:
    """
    Generates a stakeholder-ready insights report from pipeline outputs.

    Usage:
        reporter = InsightsReporter(settings)
        reporter.build(
            seller_master=master_df,
            risk_scores=scored_df,
            cohort=cohort_df,
            segment_analyses={"business_segment": seg_df, ...},
            model_metrics={"pre_activation": {...}, "retention": {...}},
        )
        reporter.save()
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._sections: List[str] = []

    # -------------------------------------------------------------------------
    # Section builders
    # -------------------------------------------------------------------------

    def _h2(self, title: str, body: str) -> None:
        self._sections.append(f"## {title}\n\n{body}\n")

    def _h3(self, title: str, body: str) -> None:
        self._sections.append(f"### {title}\n\n{body}\n")

    def _add_table(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """Convert a DataFrame to a Markdown table string (no tabulate needed)."""
        subset = df.head(max_rows)
        cols = list(subset.columns)
        # Header row
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = []
        for _, row in subset.iterrows():
            cells = []
            for v in row:
                if isinstance(v, float):
                    cells.append(f"{v:.2f}")
                else:
                    cells.append(str(v))
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, sep] + rows)

    # -------------------------------------------------------------------------
    # Individual sections
    # -------------------------------------------------------------------------

    def _section_executive_summary(
        self,
        master: pd.DataFrame,
        risk_scores: pd.DataFrame,
        model_metrics: Dict[str, Dict],
    ) -> None:
        total = len(master)
        churned = int(master["churned"].sum())
        churn_rate = master["churned"].mean() * 100
        churned_gmv = master.loc[master["churned"] == 1, "total_gmv"].sum()
        total_gmv = master["total_gmv"].sum()
        gmv_share = (churned_gmv / total_gmv * 100) if total_gmv > 0 else 0

        high_critical = risk_scores["risk_category"].isin(["High", "Critical"]).sum()
        revenue_at_risk = (
            risk_scores.loc[
                risk_scores["risk_category"].isin(["High", "Critical"])
                & (
                    risk_scores.get("churned", risk_scores["overall_churn_risk"]) == 0
                    if "churned" in risk_scores.columns
                    else risk_scores["overall_churn_risk"] > 0
                ),
                "total_gmv",
            ].sum()
            if "total_gmv" in risk_scores.columns
            else 0.0
        )

        pre_auc = model_metrics.get("pre_activation", {}).get("roc_auc", "N/A")
        ret_auc = model_metrics.get("retention", {}).get("roc_auc", "N/A")

        pre_auc_str = f"{pre_auc:.3f}" if isinstance(pre_auc, float) else pre_auc
        ret_auc_str = f"{ret_auc:.3f}" if isinstance(ret_auc, float) else ret_auc

        body = (
            f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}  \n"
            f"**Dataset:** {total:,} closed deals (Olist marketplace, Jun 2017 – Aug 2018)\n\n"
            "### Key Findings at a Glance\n\n"
            "| Metric | Value | Impact |\n"
            "|--------|-------|--------|\n"
            f"| Total Sellers Analyzed | {total:,} | Baseline population |\n"
            f"| Overall Churn Rate | {churn_rate:.1f}% | {churned:,} sellers lost |\n"
            f"| GMV from Churned Sellers | {gmv_share:.1f}% of total GMV | Revenue exposure |\n"
            f"| High / Critical Risk Sellers | {high_critical:,} | Priority intervention targets |\n"
            f"| Active Revenue at Risk | R$ {revenue_at_risk:,.2f} | Protectable with intervention |\n"
            f"| Pre-Activation Model AUC | {pre_auc_str} | Predictive performance |\n"
            f"| Retention Model AUC | {ret_auc_str} | Predictive performance |\n"
        )
        self._h2("Executive Summary", body)

    def _section_churn_breakdown(self, master: pd.DataFrame) -> None:
        total = len(master)
        never = int(master["never_activated"].sum())
        dormant = int(master["dormant"].sum())
        active = int(master["active"].sum())

        body = (
            "Three mutually exclusive seller states were defined:\n\n"
            "| Status | Count | Share | Definition |\n"
            "|--------|-------|-------|------------|\n"
            f"| 🔴 Never Activated | {never:,} | {never/total*100:.1f}% | 0 orders within 90 days of onboarding |\n"
            f"| 🟠 Dormant | {dormant:,} | {dormant/total*100:.1f}% | Had orders but inactive for 60+ days |\n"
            f"| 🟢 Active | {active:,} | {active/total*100:.1f}% | Order within last 60 days |\n\n"
            "> **Key insight:** The Never-Activated segment dominates churn. "
            "This is an onboarding failure, not a retention failure — requiring "
            "fundamentally different interventions than dormancy recovery.\n"
        )
        self._h2("Churn Breakdown", body)

    def _section_cohort_analysis(self, cohort: pd.DataFrame) -> None:
        avg_rate = cohort["churn_rate"].mean()
        best_cohort = cohort.loc[cohort["churn_rate"].idxmin()]
        worst_cohort = cohort.loc[cohort["churn_rate"].idxmax()]

        body = (
            f"- **Cohorts Analyzed:** {len(cohort)} months (Jun 2017 – Aug 2018)\n"
            f"- **Average Cohort Churn Rate:** {avg_rate:.1f}%\n"
            f"- **Best Cohort:** `{best_cohort['cohort_month']}` — {best_cohort['churn_rate']:.1f}% churn\n"
            f"- **Worst Cohort:** `{worst_cohort['cohort_month']}` — {worst_cohort['churn_rate']:.1f}% churn\n\n"
            "**Cohort Summary Table:**\n\n"
        )
        display_cols = [
            c
            for c in [
                "cohort_month",
                "total_sellers",
                "never_activated",
                "dormant",
                "active",
                "churn_rate",
                "activation_rate",
            ]
            if c in cohort.columns
        ]
        body += self._add_table(cohort[display_cols])
        self._h2("Cohort Analysis", body)

    def _section_segment_analysis(
        self, segment_analyses: Dict[str, Optional[pd.DataFrame]]
    ) -> None:
        body = ""
        for seg_name, df in segment_analyses.items():
            if df is None or len(df) == 0:
                continue
            readable = seg_name.replace("_", " ").title()
            body += f"### By {readable}\n\n"
            body += self._add_table(df) + "\n\n"
            worst = df.iloc[0]
            best = df.iloc[-1]
            body += (
                f"> **Highest churn:** `{worst['Segment']}` at {worst['Churn_Rate_%']:.1f}%  \n"
                f"> **Lowest churn:** `{best['Segment']}` at {best['Churn_Rate_%']:.1f}%\n\n"
            )
        if body:
            self._h2("Segment Analysis", body)

    def _section_risk_distribution(self, risk_scores: pd.DataFrame) -> None:
        risk_counts = risk_scores["risk_category"].value_counts()
        total = len(risk_scores)

        rows = ""
        for cat in ["Critical", "High", "Medium", "Low"]:
            count = risk_counts.get(cat, 0)
            pct = count / total * 100
            emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(
                cat, ""
            )
            rows += f"| {emoji} {cat} | {count:,} | {pct:.1f}% |\n"

        body = (
            "| Risk Category | Sellers | Share |\n"
            "|---------------|---------|-------|\n" + rows
        )
        self._h2("Risk Score Distribution", body)

    def _section_intervention_priorities(self, risk_scores: pd.DataFrame) -> None:
        if (
            "active" not in risk_scores.columns
            or "overall_churn_risk" not in risk_scores.columns
        ):
            return

        at_risk_active = risk_scores[
            risk_scores["risk_category"].isin(["High", "Critical"])
            & (risk_scores["active"] == 1)
        ].copy()

        if len(at_risk_active) == 0:
            self._h2(
                "Intervention Priorities", "_No active high-risk sellers identified._\n"
            )
            return

        # Sort by risk × GMV
        if "total_gmv" in at_risk_active.columns:
            at_risk_active["priority_score"] = (
                at_risk_active["overall_churn_risk"] * 0.6
                + at_risk_active["total_gmv"].rank(pct=True) * 0.4
            )
            at_risk_active = at_risk_active.sort_values(
                "priority_score", ascending=False
            )

        gmv_at_risk = (
            at_risk_active["total_gmv"].sum()
            if "total_gmv" in at_risk_active.columns
            else 0
        )
        projected_saved = gmv_at_risk * 0.25 * 3  # 25% save rate × 3 months

        display = [
            c
            for c in [
                "seller_id",
                "seller_state",
                "business_segment",
                "total_gmv",
                "overall_churn_risk",
                "risk_category",
                "urgency",
            ]
            if c in at_risk_active.columns
        ]

        body = (
            f"- **Active sellers at High/Critical risk:** {len(at_risk_active):,}\n"
            f"- **Total GMV at risk:** R$ {gmv_at_risk:,.2f}\n"
            f"- **Projected GMV saved** (25% save rate, 3-month horizon): R$ {projected_saved:,.2f}\n\n"
            "**Top 15 Priority Sellers:**\n\n"
        )
        body += self._add_table(at_risk_active[display], max_rows=15)
        self._h2("Intervention Priorities", body)

    def _section_recommended_actions(self) -> None:
        body = (
            "### Immediate (Next 7 Days)\n\n"
            "1. **Critical-risk active sellers** — Personal outreach from Account Manager "
            "within 48 hours. Offer: 1-month platform fee waiver + dedicated support.\n"
            "2. **High-risk active sellers** — Trigger automated email nurture sequence "
            "(5 touchpoints over 2 weeks).\n\n"
            "### Short-Term (30 Days)\n\n"
            "3. **Never-Activated Recovery** — Launch reactivation campaign targeting "
            "sellers with 0 orders but registered < 90 days. Include onboarding webinar.\n"
            "4. **Segment-Specific Programs** — Build tailored playbooks for highest-churn "
            "business segments and DISC profiles identified in the analysis.\n\n"
            "### Medium-Term (90 Days)\n\n"
            "5. **Deploy Automated Scoring** — Schedule monthly re-scoring of all sellers "
            "using the trained models. Automate alerts for Critical-risk transitions.\n"
            "6. **Onboarding Redesign** — Reform the first-30-days seller journey: "
            "automated check-ins, catalog setup assistance, first-sale milestone rewards.\n\n"
            "### Success KPIs\n\n"
            "| KPI | Target |\n"
            "|-----|--------|\n"
            "| Intervention response rate | > 40% |\n"
            "| Seller save rate (3 months post-intervention) | > 25% |\n"
            "| Never-Activated rate reduction (next cohort) | – 10 pp |\n"
            "| Active seller ratio improvement | + 5 pp |\n"
        )
        self._h2("Recommended Actions", body)

    def _section_model_appendix(self, model_metrics: Dict[str, Dict]) -> None:
        rows = ""
        for model_name, m in model_metrics.items():
            readable = model_name.replace("_", " ").title()
            rows += (
                (
                    f"| {readable} "
                    f"| {m.get('roc_auc', 'N/A'):.3f} "
                    f"| {m.get('precision', 'N/A'):.3f} "
                    f"| {m.get('recall', 'N/A'):.3f} "
                    f"| {m.get('f1_score', 'N/A'):.3f} |\n"
                )
                if all(
                    isinstance(m.get(k), float)
                    for k in ["roc_auc", "precision", "recall", "f1_score"]
                )
                else (f"| {readable} | N/A | N/A | N/A | N/A |\n")
            )

        body = (
            "| Model | AUC-ROC | Precision | Recall | F1-Score |\n"
            "|-------|---------|-----------|--------|----------|\n"
            + rows
            + "\n> Full ROC curves, Precision-Recall curves, and Confusion Matrices "
            "are in `reports/figures/`. See `model_evaluation.md` for the technical report.\n"
        )
        self._h2("Appendix: Model Performance", body)

    # -------------------------------------------------------------------------
    # Main build + save
    # -------------------------------------------------------------------------

    def build(
        self,
        seller_master: pd.DataFrame,
        risk_scores: pd.DataFrame,
        cohort: pd.DataFrame,
        segment_analyses: Optional[Dict[str, Optional[pd.DataFrame]]] = None,
        model_metrics: Optional[Dict[str, Dict]] = None,
    ) -> "InsightsReporter":
        """
        Assemble all report sections.

        Args:
            seller_master:     Full seller dataset with churn labels.
            risk_scores:       Dataset with overall_churn_risk + risk_category columns.
            cohort:            Cohort analysis DataFrame.
            segment_analyses:  Dict of {segment_col: DataFrame} from ChurnAnalyzer.segment_analysis.
            model_metrics:     Dict of {model_key: metrics_dict} from ModelEvaluator.evaluate_model.

        Returns:
            self (fluent API)
        """
        model_metrics = model_metrics or {}
        segment_analyses = segment_analyses or {}

        self._sections = []  # reset
        self._section_executive_summary(seller_master, risk_scores, model_metrics)
        self._section_churn_breakdown(seller_master)
        self._section_cohort_analysis(cohort)
        if segment_analyses:
            self._section_segment_analysis(segment_analyses)
        self._section_risk_distribution(risk_scores)
        self._section_intervention_priorities(risk_scores)
        self._section_recommended_actions()
        if model_metrics:
            self._section_model_appendix(model_metrics)

        return self

    def save(self, filename: str = "churn_insights_report.md") -> str:
        """
        Write the report to reports/<filename>.

        Returns:
            Absolute path to the saved file.
        """
        header = (
            "# Olist Seller Churn Analysis — Insights Report\n\n"
            f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            "Confidential — Internal Use Only_\n\n"
            "---\n\n"
        )
        full_content = header + "\n".join(self._sections)

        path = self.settings.REPORTS_PATH / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(full_content, encoding="utf-8")
        logger.info(f"Insights report saved: reports/{filename}")
        return str(path)
