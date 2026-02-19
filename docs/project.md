# **FINAL EXECUTION PLAN: OLIST MARKETPLACE CHURN ANALYSIS**
## Data-Driven Strategy for Dual-Side Retention & Growth

---

## **🎯 REFINED UNDERSTANDING**

### **DISC Behavior Profiles Decoded:**
- **Eagle (Influence):** Social, persuasive, enthusiastic - likely responsive to relationship-building
- **Cat (Steadiness):** Patient, reliable, team-oriented - likely needs support and reassurance
- **Wolf (Conscientiousness):** Analytical, detail-oriented, quality-focused - likely needs data and processes
- **Shark (Dominance):** Results-driven, competitive, decisive - likely wants quick wins

**Strategic Hypothesis:** Different profiles need different onboarding/retention approaches. A "Cat" seller may churn from lack of hand-holding, while a "Shark" churns from slow results.

### **Business Context Clarified:**
- **Time Window:** Jun 2017 - Jun 2018 (12-month snapshot)
- **Funnel Reality:** 8,000 MQLs → 842 deals = 10.5% conversion (industry benchmark for B2B SaaS is 10-15%, so you're competitive)
- **Deal Close:** Seller paid subscription, officially joined platform
- **Missing Data:** Those 0.0 values in `declared_monthly_revenue` and `catalog_size` suggest optional fields or data entry gaps

### **Dual Objective:**
**Maximize both seller AND customer retention** → Compound effect on marketplace GMV and sustainability

---

## **📊 ANALYSIS ROADMAP**

I'll structure this as a **4-week sprint** with weekly deliverables. Here's what happens each week:

---

## **WEEK 1: FOUNDATION & DISCOVERY**

### **Day 1-2: Data Integration & Quality Assessment**

**Tasks:**
1. **Build master dataset** linking all tables:
   - MQLs → Closed Deals → Sellers → Orders → Customers → Reviews → Products

2. **Data quality audit:**
   - Missing value analysis (especially those 0.0 fields)
   - Date range validation
   - Duplicate detection
   - Outlier identification

3. **Establish ground truth:**
   - How many sellers from the 842 deals actually went live (had ≥1 order)?
   - What's the average time from `won_date` to first sale?
   - What % of 842 sellers are still active at dataset end date?

**Expected Findings:**
- Likely discover that significant % of sellers never activate (common in marketplaces)
- Identify data quirks that need handling

---

### **Day 3-4: Cohort Analysis & Churn Definition**

**For Sellers:**

**Churn Definitions:**
1. **Never-activated churn:** Seller closed deal but 0 orders in first 90 days
2. **Dormant churn:** Seller had ≥1 order but 0 orders in last 60 days
3. **Declining churn:** Seller active but GMV down >50% from peak month

**Cohort Segmentation:**
- Monthly onboarding cohorts (Jul 2017 → Jun 2018)
- By business segment (15+ categories visible)
- By lead type (online_big, online_medium, online_small, offline, industry)
- By DISC profile (Cat, Eagle, Wolf, Shark)
- By business type (reseller vs. manufacturer)

**For Customers:**

**Churn Definitions:**
1. **One-and-done:** First purchase, no repeat in 60/90/120 days
2. **Lapsed repeat:** Had 2+ purchases but none in 90 days
3. **Declining engagement:** 30%+ increase in inter-purchase interval

**Cohort Segmentation:**
- Monthly first-purchase cohorts
- By state/region
- By product category preference
- By seller type purchased from

**Deliverable:**
- **Cohort Dashboard** showing retention curves, churn rates by segment
- **Benchmark Report** with industry standards (I'll use web search for Brazilian e-commerce benchmarks)

---

### **Day 5: Stakeholder Checkpoint #1**

**Presentation (60 min):**
- Data landscape overview
- Preliminary churn rates (high-level numbers)
- Cohort retention curves
- Initial hypotheses for Week 2 deep-dive
- Q&A and priority alignment

**Decision Point:** Confirm which segments are highest priority for deeper analysis

---

## **WEEK 2: DEEP DIVE & DRIVER ANALYSIS**

### **Day 6-8: Feature Engineering Intensive**

I'll create **150+ features** across these categories:

**Seller Features:**

*Activity Metrics:*
- Total orders, GMV, AOV (average order value)
- Orders/month trajectory (accelerating, stable, declining)
- Product diversity (categories sold, unique products)
- Geographic reach (states served)
- Inventory velocity (order frequency)

*Quality Metrics:*
- Avg review score, % 5-star, % 1-2 star
- Delivery performance (% on-time, avg delay)
- Cancellation rate
- Response time (if available in future data)

*Economic Metrics:*
- GMV vs. declared revenue (expectation gap)
- Revenue per product (catalog efficiency)
- Month-over-month growth rate
- Days to first sale (activation speed)
- Customer concentration (% of GMV from top 3 customers = risk)

*Funnel Features:*
- Lead source (organic vs paid)
- Time from MQL to close
- SDR/SR who managed (proxy for sales quality)
- Landing page (proxy for expectations set)

*Behavioral:*
- DISC profile
- Lead type
- Business segment
- Reseller vs manufacturer

**Customer Features:**

*RFM (Recency, Frequency, Monetary):*
- Days since last purchase
- Total orders
- Total spend, AOV
- Purchase acceleration (time between orders shortening/lengthening)

*Experience Metrics:*
- Avg review score given
- % of orders with issues (late delivery, low rating)
- Avg delivery time experienced
- Product category diversity (browsing breadth)

*Seller Relationship:*
- Number of unique sellers purchased from
- % of spend with top seller (loyalty concentration)
- Experienced seller churn (did their seller disappear?)
- Avg seller quality (based on seller's overall rating)

*Geographic/Demographic Proxies:*
- State (São Paulo vs others - economic proxy)
- Urban vs. rural (based on city)
- Local vs. distant purchases (shipping distance)

**Interaction Features:**
- Customer tenure × review score
- Seller DISC profile × customer satisfaction
- Product category × delivery performance

---

### **Day 9-10: Exploratory Data Analysis (EDA)**

**Comparative Analysis:**

For **every feature**, I'll compare:
- Active vs. churned (both seller and customer)
- High performers vs. low performers
- Retained vs. at-risk

**Statistical Tests:**
- T-tests for continuous variables
- Chi-square for categorical variables
- Correlation matrices
- Feature importance via random forest

**Visualizations:**
- Distribution plots (churned vs. active)
- Box plots (feature by segment)
- Heatmaps (correlation matrices)
- Survival curves (time-to-churn by segment)
- Sankey diagrams (funnel flow)

**Key Questions:**
- Which DISC profile has highest/lowest churn?
- Do resellers vs. manufacturers behave differently?
- Does lead source predict seller longevity?
- Do customers buying from high-review sellers churn less?
- Does seller churn cause customer churn?

**Deliverable:**
- **EDA Report** (30+ pages) with statistical findings
- **Insight Summary** (5-page executive brief)

---

### **Day 10: Stakeholder Checkpoint #2**

**Presentation (90 min):**
- Top 10 churn drivers for sellers
- Top 10 churn drivers for customers
- Surprising findings ("Cat" sellers churn 2x more than expected)
- Segment deep-dives
- Preview of Week 3 modeling approach

---

## **WEEK 3: PREDICTIVE MODELING & OPTIMIZATION**

### **Day 11-13: Model Development**

**Seller Churn Models:**

**Model 1: Pre-Activation Risk (0-90 days post-close)**
- **Features:** Pre-sale data only (DISC, segment, lead source, declared revenue, SDR/SR)
- **Target:** Did seller achieve first sale within 90 days?
- **Use Case:** Prioritize onboarding support

**Model 2: Active Seller Churn Risk (ongoing)**
- **Features:** Full activity data (orders, reviews, GMV trends)
- **Target:** Will seller be dormant in next 60 days?
- **Use Case:** Proactive retention interventions

**Customer Churn Models:**

**Model 3: First-Purchase Conversion (post-first-order)**
- **Features:** First order experience (delivery, review, category, seller quality)
- **Target:** Will customer make 2nd purchase within 90 days?
- **Use Case:** Post-purchase engagement campaigns

**Model 4: Repeat Customer Retention**
- **Features:** Full RFM + experience data
- **Target:** Will customer churn in next 90 days?
- **Use Case:** Win-back programs, VIP retention

**Modeling Approach:**

For each model, I'll train:
1. **Logistic Regression** (baseline, interpretable)
2. **Random Forest** (captures non-linear relationships)
3. **XGBoost** (usually best performance)
4. **LightGBM** (faster, comparable to XGBoost)

**Validation:**
- Time-based split (train on first 9 months, test on last 3)
- 5-fold cross-validation within training set
- Evaluation metrics:
  - **AUC-ROC** (overall discrimination)
  - **Precision@20%** (top quintile accuracy - most actionable)
  - **Recall@20%** (how many churners caught in top 20%)
  - **Calibration** (are probabilities accurate?)

**Feature Importance:**
- SHAP values (explains individual predictions)
- Permutation importance
- Partial dependence plots

---

### **Day 14-15: Funnel Optimization Analysis**

**Lead Source ROI:**
```
For each source (organic, paid, social, email, referral):
- MQL → Deal conversion rate
- Avg time to close
- Avg seller GMV (first 6 months)
- Seller retention rate (6-month)
- **Cost per acquisition** (if you have ad spend data) → ROI
```

**Landing Page Performance:**
- Conversion rates by page
- Seller quality by page (proxy: avg GMV, retention)
- Recommendations for page optimization

**SDR/SR Analysis:**
- Conversion rates by rep
- Quality of sellers closed (GMV, retention)
- Time-to-close efficiency
- **Best practices:** What do top performers do differently?

**DISC Profile Strategy:**
- Which profiles convert best?
- Which profiles have best retention?
- Should sales approach differ by profile?
  - **Eagle:** Emphasize community, networking, support
  - **Cat:** Emphasize reliability, step-by-step guidance, low risk
  - **Wolf:** Emphasize data, analytics, optimization tools
  - **Shark:** Emphasize revenue potential, competitive advantage, speed

**Segment Opportunity Analysis:**
- Which business segments are under-served? (high customer demand, few sellers)
- Which are over-saturated? (many sellers, low GMV per seller)
- Where should sales focus acquisition?

**Deliverable:**
- **Funnel Optimization Report** with channel reallocation recommendations
- **Sales Playbook** tailored by DISC profile

---

## **WEEK 4: INTEGRATION & ACTIONABLE STRATEGY**

### **Day 16-17: Interconnected Analysis**

**Marketplace Dynamics:**

**Seller → Customer Impact:**
- When a high-volume seller churns, how many customers affected?
- Do customers who buy from churned sellers also churn?
- Identify "critical sellers" whose loss would be catastrophic

**Customer → Seller Impact:**
- Do sellers in categories with high customer churn also churn?
- Does poor seller performance (reviews) predict customer churn?

**Category Health:**
- Which categories are thriving (seller growth + customer retention)?
- Which are dying (seller churn + customer churn)?
- Portfolio optimization: where to invest marketing vs. sunset?

**Geographic Patterns:**
- Are there regions with systemic issues (delivery, seller density)?
- Opportunity zones (high demand, low supply)?

---

### **Day 18: Benchmark Research**

I'll use web search to find:
- Brazilian e-commerce churn benchmarks
- Marketplace retention standards (Mercado Livre, Amazon Brasil comparables)
- SaaS B2B churn rates (for seller side)
- D2C retention benchmarks (for customer side)

This contextualizes your performance: "Your 35% seller churn is above the 25% industry average - opportunity to improve."

---

### **Day 19-20: Strategic Recommendations & Tools**

**Deliverable 1: Action Playbook (3 separate documents)**

**A. Seller Success Playbook:**

*Pre-Activation (Days 0-30):*
- **Cat sellers:** Weekly check-ins, catalog-building workshops, hand-holding
- **Eagle sellers:** Community intro, seller networking events, peer mentorship
- **Wolf sellers:** Analytics training, data dashboards, optimization guides
- **Shark sellers:** Revenue targets, competitive benchmarks, fast-track support

*At-Risk Intervention (Model score >70%):*
- Proactive outreach within 48 hours
- Root cause diagnosis (survey)
- Personalized support (category expert, logistics help, marketing co-op)

*Win-Back (Churned <90 days):*
- Reactivation offers
- "What went wrong" interview
- Platform improvements communication

**B. Customer Retention Playbook:**

*Segment 1: First-Time Buyers (0-30 days post-purchase):*
- Post-purchase survey (delivery experience, product quality)
- Next-purchase incentive (10% off, free shipping)
- Category recommendations (browsing expansion)

*Segment 2: At-Risk Repeat Customers (Model score >60%):*
- Personalized email: "We miss you! Here's what's new"
- VIP offers for high-value customers
- Seller recommendations (introduce to new sellers)

*Segment 3: Seller-Churn Affected:*
- "Your favorite seller is no longer available, try these alternatives"
- Category expansion suggestions
- Apology credit if seller churned mid-order

**C. Sales Optimization Playbook:**

*Lead Routing:*
- Route by DISC profile to specialized SDRs
- Prioritize high-intent leads (organic search) for top SRs

*Sales Process:*
- Expectation-setting: Align `declared_revenue` with realistic ramp
- Category counseling: Steer away from oversaturated segments
- Onboarding preview: Show them what first 90 days looks like

*Performance Management:*
- SDR/SR scorecards: Conversion + 6-month seller retention
- Best practice sharing (top performer playbook)

---

**Deliverable 2: Operational Dashboards (3 tools)**

I'll build these as interactive visualizations:

**Dashboard 1: Seller Health Monitor**
- Real-time churn risk scores for all active sellers
- Segment filters (DISC, category, cohort)
- Intervention triggers (auto-flag when risk >70%)
- Performance trends (GMV, orders, reviews)

**Dashboard 2: Customer Retention Tracker**
- Cohort retention curves (live updating)
- At-risk customer segments
- Win-back campaign effectiveness
- LTV trends by segment

**Dashboard 3: Funnel Analytics**
- MQL → Deal conversion by source/page/SDR
- Time-to-close trends
- Seller quality by acquisition channel
- ROI calculator (input: ad spend, output: projected GMV)

---

**Deliverable 3: Executive Presentation (25 slides)**

**Structure:**
1. **State of the Marketplace** (Slides 1-5)
   - Current churn rates (seller + customer)
   - Benchmark comparison
   - Economic impact quantification

2. **Root Cause Analysis** (Slides 6-12)
   - Top 5 seller churn drivers
   - Top 5 customer churn drivers
   - DISC profile insights
   - Segment deep-dives

3. **Predictive Insights** (Slides 13-17)
   - Model performance summary
   - Risk score distribution (how many at-risk right now?)
   - Feature importance (what matters most?)
   - Early warning indicators

4. **Strategic Recommendations** (Slides 18-23)
   - Prioritized action plan (quick wins vs. long-term)
   - Resource allocation (where to invest)
   - Expected impact (churn reduction %, GMV lift)
   - Implementation roadmap (6-month plan)

5. **Next Steps** (Slides 24-25)
   - Immediate actions (Week 1)
   - Dashboard deployment
   - Ongoing monitoring plan

---

### **Day 20: Final Delivery**

**Package Contents:**

📁 **1. Reports Folder**
- Executive Summary (10 pages)
- Seller Churn Analysis (40 pages)
- Customer Churn Analysis (40 pages)
- Funnel Optimization Study (25 pages)
- Model Documentation (20 pages)

📁 **2. Playbooks Folder**
- Seller Success Playbook (PDF)
- Customer Retention Playbook (PDF)
- Sales Optimization Playbook (PDF)

📁 **3. Data Products Folder**
- `seller_risk_scores.csv` (all sellers with churn probability)
- `customer_risk_scores.csv` (all customers with churn probability)
- `segment_profiles.csv` (characteristics of each segment)
- `intervention_priority_list.csv` (top 100 sellers + customers to contact)

📁 **4. Code & Models Folder**
- Python notebooks (fully commented, reproducible)
- Trained models (pickled for deployment)
- SQL queries (for database integration)
- Dashboard templates (Power BI / Tableau)

📁 **5. Presentation Folder**
- Executive deck (PPT)
- Technical appendix (detailed methodology)

---

## **🎯 EXPECTED IMPACT (Quantified)**

Let me project realistic outcomes based on industry benchmarks:

### **Baseline Assumptions:**
- 842 sellers, assume 50% churn annually = 421 churned sellers
- ~96,000 customers, assume 60% churn (one-time buyers) = 57,600 churned
- Avg seller GMV: $5,000/month (conservative)
- Avg customer LTV: $200 (2-3 orders)

### **Projected Improvements:**

**Seller Retention:**
- **Reduce pre-activation churn by 40%** (better onboarding)
  - Save ~85 sellers from never activating
  - Additional GMV: 85 × $5K/mo × 12 mo = **$5.1M annually**

- **Reduce active seller churn by 15%** (early intervention)
  - Save ~50 active sellers
  - Additional GMV: 50 × $5K/mo × 12 mo = **$3M annually**

**Customer Retention:**
- **Convert 10% more one-time buyers to repeat**
  - 5,760 customers × $200 LTV = **$1.15M**

- **Extend repeat customer lifetime by 20%**
  - 38,400 repeat customers × $200 LTV × 0.2 = **$1.54M**

**Funnel Optimization:**
- **Improve MQL → Deal conversion by 3 percentage points** (10.5% → 13.5%)
  - On 8,000 MQLs = 240 additional sellers/year
  - 240 × $5K/mo × 12 × 50% retention = **$7.2M GMV**

### **Total Projected Annual Impact:**
**$17.99M in additional GMV** from retention improvements alone.

If Olist takes 10-15% commission, that's **$1.8M - $2.7M in additional revenue**.

**ROI:** If this 4-week project costs $50K-$100K, the **ROI is 18-54x in Year 1**.

---
