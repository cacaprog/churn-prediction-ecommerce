# Olist Seller Churn Analysis — Insights Report

_Generated: 2026-02-19 14:58:53 | Confidential — Internal Use Only_

---

## Executive Summary

**Analysis Date:** 2026-02-19  
**Dataset:** 842 closed deals (Olist marketplace, Jun 2017 – Aug 2018)

### Key Findings at a Glance

| Metric | Value | Impact |
|--------|-------|--------|
| Total Sellers Analyzed | 842 | Baseline population |
| Overall Churn Rate | 85.0% | 716 sellers lost |
| GMV from Churned Sellers | 40.4% of total GMV | Revenue exposure |
| High / Critical Risk Sellers | 669 | Priority intervention targets |
| Active Revenue at Risk | R$ 33,097.00 | Protectable with intervention |
| Pre-Activation Model AUC | 0.966 | Predictive performance |
| Retention Model AUC | 0.577 | Predictive performance |


## Churn Breakdown

Three mutually exclusive seller states were defined:

| Status | Count | Share | Definition |
|--------|-------|-------|------------|
| 🔴 Never Activated | 515 | 61.2% | 0 orders within 90 days of onboarding |
| 🟠 Dormant | 201 | 23.9% | Had orders but inactive for 60+ days |
| 🟢 Active | 140 | 16.6% | Order within last 60 days |

> **Key insight:** The Never-Activated segment dominates churn. This is an onboarding failure, not a retention failure — requiring fundamentally different interventions than dormancy recovery.


## Cohort Analysis

- **Cohorts Analyzed:** 12 months (Jun 2017 – Aug 2018)
- **Average Cohort Churn Rate:** 89.6%
- **Best Cohort:** `2018-05` — 73.0% churn
- **Worst Cohort:** `2017-12` — 100.0% churn

**Cohort Summary Table:**

| cohort_month | total_sellers | never_activated | dormant | active | churn_rate | activation_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2017-12 | 3 | 2 | 1 | 0 | 100.00 | 33.30 |
| 2018-01 | 73 | 41 | 25 | 11 | 90.40 | 43.80 |
| 2018-02 | 113 | 67 | 29 | 18 | 85.00 | 40.70 |
| 2018-03 | 147 | 85 | 42 | 22 | 86.40 | 42.20 |
| 2018-04 | 207 | 119 | 60 | 34 | 86.50 | 42.50 |
| 2018-05 | 122 | 59 | 30 | 34 | 73.00 | 51.60 |
| 2018-06 | 57 | 34 | 11 | 12 | 78.90 | 40.40 |
| 2018-07 | 37 | 27 | 3 | 7 | 81.10 | 27.00 |
| 2018-08 | 33 | 31 | 0 | 2 | 93.90 | 6.10 |
| 2018-09 | 23 | 23 | 0 | 0 | 100.00 | 0.00 |

## Segment Analysis

### By Business Segment

| Segment | Total_Sellers | Churned | Never_Activated | Active | Avg_GMV | Total_GMV | Churn_Rate_% | Market_Share_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Unknown | 1 | 1 | 1 | 0 | 0.00 | 0.00 | 100.00 | 0.10 |
| religious | 1 | 1 | 1 | 0 | 0.00 | 0.00 | 100.00 | 0.10 |
| perfume | 2 | 2 | 2 | 0 | 0.00 | 0.00 | 100.00 | 0.20 |
| other | 3 | 3 | 3 | 0 | 0.00 | 0.00 | 100.00 | 0.40 |
| jewerly | 8 | 8 | 8 | 0 | 0.00 | 0.00 | 100.00 | 1.00 |
| air_conditioning | 3 | 3 | 2 | 0 | 883.33 | 2650.00 | 100.00 | 0.40 |
| handcrafted | 12 | 12 | 11 | 2 | 72.99 | 875.90 | 100.00 | 1.40 |
| gifts | 5 | 5 | 3 | 0 | 209.15 | 1045.76 | 100.00 | 0.60 |
| fashion_accessories | 19 | 18 | 15 | 2 | 412.16 | 7831.08 | 94.70 | 2.30 |
| computers | 34 | 32 | 22 | 2 | 346.24 | 11772.29 | 94.10 | 4.00 |

> **Highest churn:** `Unknown` at 100.0%  
> **Lowest churn:** `games_consoles` at 50.0%

### By Lead Behaviour Profile

| Segment | Total_Sellers | Churned | Never_Activated | Active | Avg_GMV | Total_GMV | Churn_Rate_% | Market_Share_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cat, wolf | 8 | 8 | 7 | 0 | 7.38 | 59.00 | 100.00 | 1.00 |
| eagle, wolf | 3 | 3 | 3 | 0 | 0.00 | 0.00 | 100.00 | 0.40 |
| shark, cat | 1 | 1 | 1 | 0 | 0.00 | 0.00 | 100.00 | 0.10 |
| shark, wolf | 1 | 1 | 1 | 0 | 0.00 | 0.00 | 100.00 | 0.10 |
| Unknown | 177 | 155 | 106 | 28 | 1309.16 | 231720.67 | 87.60 | 21.00 |
| shark | 24 | 21 | 17 | 3 | 1965.10 | 47162.42 | 87.50 | 2.90 |
| eagle | 123 | 106 | 77 | 18 | 737.60 | 90724.83 | 86.20 | 14.60 |
| wolf | 95 | 80 | 58 | 15 | 225.03 | 21377.89 | 84.20 | 11.30 |
| cat | 407 | 339 | 243 | 75 | 693.62 | 282303.58 | 83.30 | 48.30 |
| eagle, cat | 3 | 2 | 2 | 1 | 219.00 | 657.00 | 66.70 | 0.40 |

> **Highest churn:** `cat, wolf` at 100.0%  
> **Lowest churn:** `eagle, cat` at 66.7%

### By Lead Type

| Segment | Total_Sellers | Churned | Never_Activated | Active | Avg_GMV | Total_GMV | Churn_Rate_% | Market_Share_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| other | 3 | 3 | 3 | 0 | 0.00 | 0.00 | 100.00 | 0.40 |
| offline | 104 | 98 | 80 | 9 | 186.57 | 19403.54 | 94.20 | 12.40 |
| online_beginner | 57 | 53 | 38 | 4 | 391.27 | 22302.52 | 93.00 | 6.80 |
| online_small | 77 | 69 | 51 | 8 | 681.50 | 52475.28 | 89.60 | 9.10 |
| industry | 123 | 109 | 85 | 16 | 237.22 | 29178.39 | 88.60 | 14.60 |
| Unknown | 6 | 5 | 3 | 1 | 7513.15 | 45078.90 | 83.30 | 0.70 |
| online_medium | 332 | 276 | 192 | 63 | 621.99 | 206501.77 | 83.10 | 39.40 |
| online_big | 126 | 93 | 55 | 35 | 2308.50 | 290870.95 | 73.80 | 15.00 |
| online_top | 14 | 10 | 8 | 4 | 585.29 | 8194.04 | 71.40 | 1.70 |

> **Highest churn:** `other` at 100.0%  
> **Lowest churn:** `online_top` at 71.4%

### By Seller State

| Segment | Total_Sellers | Churned | Never_Activated | Active | Avg_GMV | Total_GMV | Churn_Rate_% | Market_Share_% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BA | 3 | 3 | 1 | 0 | 1031.60 | 3094.80 | 100.00 | 0.40 |
| DF | 4 | 4 | 0 | 0 | 612.57 | 2450.29 | 100.00 | 0.50 |
| PE | 1 | 1 | 0 | 0 | 319.90 | 319.90 | 100.00 | 0.10 |
| Unknown | 462 | 462 | 462 | 0 | 0.00 | 0.00 | 100.00 | 54.90 |
| GO | 7 | 6 | 0 | 1 | 1123.20 | 7862.38 | 85.70 | 0.80 |
| PR | 32 | 26 | 2 | 8 | 927.78 | 29688.91 | 81.20 | 3.80 |
| RS | 19 | 15 | 5 | 6 | 2280.39 | 43327.45 | 78.90 | 2.30 |
| ES | 4 | 3 | 1 | 1 | 2132.07 | 8528.29 | 75.00 | 0.50 |
| MG | 26 | 18 | 3 | 8 | 1148.33 | 29856.52 | 69.20 | 3.10 |
| RJ | 26 | 17 | 3 | 10 | 2172.80 | 56492.84 | 65.40 | 3.10 |

> **Highest churn:** `BA` at 100.0%  
> **Lowest churn:** `PB` at 0.0%



## Risk Score Distribution

| Risk Category | Sellers | Share |
|---------------|---------|-------|
| 🔴 Critical | 378 | 44.9% |
| 🟠 High | 291 | 34.6% |
| 🟡 Medium | 114 | 13.5% |
| 🟢 Low | 59 | 7.0% |


## Intervention Priorities

- **Active sellers at High/Critical risk:** 33
- **Total GMV at risk:** R$ 33,332.40
- **Projected GMV saved** (25% save rate, 3-month horizon): R$ 24,999.30

**Top 15 Priority Sellers:**

| seller_id | seller_state | business_segment | total_gmv | overall_churn_risk | risk_category |
| --- | --- | --- | --- | --- | --- |
| a63bfbaa882c8f4542891b4e2246cc7f | RJ | health_beauty | 6772.00 | 0.99 | Critical |
| 9b1585752613ec342d03bbab9997ec48 | RJ | car_accessories | 4248.98 | 0.92 | Critical |
| 77a515caa36327151d1cc6c32a9f00e1 | SP | stationery | 2217.39 | 0.93 | Critical |
| 56e361f411e38dcef17cdc2a3d99628b | SP | audio_video_electronics | 1298.00 | 0.96 | Critical |
| 11fb6f6d341adbe19e81733701704635 | SP | home_decor | 1197.00 | 0.97 | Critical |
| b586cd24c010a13916af621b0325fbba | SC | home_decor | 1108.94 | 0.96 | Critical |
| 723cd880edaacdb998898b67c8f9da30 | SP | bed_bath_table | 2136.53 | 0.84 | Critical |
| ffc470761de7d0232558ba5e786e57b7 | SP | construction_tools_house_garden | 1649.01 | 0.81 | Critical |
| e5cbe890e679490127e9a390b46bbd20 | SP | food_supplement | 908.99 | 0.90 | Critical |
| 64c9a1db4e73e19aaafd3286dc448c96 | PR | household_utilities | 439.00 | 0.99 | Critical |
| 5bc24d989e71e93c33e50a7782431b0e | MG | car_accessories | 3148.00 | 0.65 | High |
| 88ef59b51bdaa941d10a853429f2b6ce | PR | audio_video_electronics | 252.90 | 0.99 | Critical |
| 1961c3e1272bfeceb05d0b78b5bbfdaf | SP | games_consoles | 657.00 | 0.83 | Critical |
| 751e274377499a8503fd6243ad9c56f6 | SP | audio_video_electronics | 218.00 | 0.97 | Critical |
| cb6c9f5888a7a090c75beaf615925792 | ES | construction_tools_house_garden | 779.89 | 0.72 | High |

## Recommended Actions

### Immediate (Next 7 Days)

1. **Critical-risk active sellers** — Personal outreach from Account Manager within 48 hours. Offer: 1-month platform fee waiver + dedicated support.
2. **High-risk active sellers** — Trigger automated email nurture sequence (5 touchpoints over 2 weeks).

### Short-Term (30 Days)

3. **Never-Activated Recovery** — Launch reactivation campaign targeting sellers with 0 orders but registered < 90 days. Include onboarding webinar.
4. **Segment-Specific Programs** — Build tailored playbooks for highest-churn business segments and DISC profiles identified in the analysis.

### Medium-Term (90 Days)

5. **Deploy Automated Scoring** — Schedule monthly re-scoring of all sellers using the trained models. Automate alerts for Critical-risk transitions.
6. **Onboarding Redesign** — Reform the first-30-days seller journey: automated check-ins, catalog setup assistance, first-sale milestone rewards.

### Success KPIs

| KPI | Target |
|-----|--------|
| Intervention response rate | > 40% |
| Seller save rate (3 months post-intervention) | > 25% |
| Never-Activated rate reduction (next cohort) | – 10 pp |
| Active seller ratio improvement | + 5 pp |


## Appendix: Model Performance

| Model | AUC-ROC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Pre Activation | 0.966 | 0.924 | 0.946 | 0.935 |
| Retention | 0.577 | 0.633 | 0.620 | 0.626 |

> Full ROC curves, Precision-Recall curves, and Confusion Matrices are in `reports/figures/`. See `model_evaluation.md` for the technical report.

