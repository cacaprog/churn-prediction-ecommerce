## **1-WEEK PROOF OF CONCEPT: SELLER CHURN QUICK-WIN**


### **🎯 SPRINT OBJECTIVE**
Prove analytical value by identifying the top 3 seller churn drivers and building a working prediction model that flags at-risk sellers for immediate intervention.


## **📅 DAY-BY-DAY EXECUTION**

### **DAY 1: Data Foundation (Monday)**

**Morning (4 hours):**
- Build core dataset linking: MQLs → Closed Deals → Sellers → Order Items → Orders
- Establish seller activity timeline (won_date → first_sale_date → last_sale_date)
- Define churn metrics:
  - **Never-activated:** 0 orders within 90 days of won_date
  - **Dormant:** Had orders but 0 in last 60 days of dataset
  - **Active:** Had order in last 60 days

**Afternoon (4 hours):**
- Calculate baseline metrics:
  - How many of 842 deals actually went live?
  - Average time-to-first-sale
  - Overall churn rate by definition
  - Monthly cohort retention curves (Jun 2017 → Jun 2018)

**End-of-Day Output:**
- One-page metrics summary
- Cohort retention visualization
- Data quality notes (missing values, date issues)


### **DAY 2: Segment Deep-Dive (Tuesday)**

**Morning (4 hours):**
Analyze churn rates across key dimensions:
1. **DISC Profile** (Cat, Eagle, Wolf, Shark)
2. **Business Segment** (15 categories)
3. **Lead Type** (online_big, online_medium, online_small, offline, industry)
4. **Business Type** (reseller vs manufacturer vs not declared)
5. **Lead Source** (via `origin` field from MQL data)

**Afternoon (4 hours):**
- Statistical significance testing (Chi-square for categorical comparisons)
- Identify highest/lowest churn segments
- Calculate segment sizes (avoid focusing on tiny segments)
- Early hypotheses generation

**End-of-Day Output:**
- Segment comparison table with churn rates
- Top 3 hypothesis candidates (e.g., "Cat sellers churn 2x more than Sharks")


### **DAY 3: Feature Engineering (Wednesday)**

**Morning (4 hours):**
Create 30-40 features focusing on **pre-activation predictors** (data available at deal close):

*Seller Profile Features:*
- DISC profile (one-hot encoded)
- Business segment, lead type, business type
- Has company (yes/no)
- Has GTIN (product barcodes - quality signal)
- Declared catalog size (0 vs >0, actual value)
- Declared monthly revenue (0 vs >0, actual value)
- Average stock level

*Funnel Features:*
- Lead source/origin
- Landing page ID
- Time from first_contact_date to won_date (sales cycle length)
- SDR and SR IDs (as categorical - some reps may close better sellers)

*Expectation Gap:*
- Ratio of declared_revenue to business segment median
- Catalog size vs segment median

**Afternoon (4 hours):**
Add **post-activation features** (for sellers who went live):
- Days to first sale
- First month orders count
- First month GMV
- First month average order value
- Number of unique customers (first 30 days)
- Product diversity (unique products sold, first 30 days)

**End-of-Day Output:**
- Feature dataset ready for modeling
- Feature description document


### **DAY 4: Model Building (Thursday)**

**Morning (3 hours):**
Build **Model 1: Pre-Activation Churn Risk**
- Target: Never activated (0 orders in 90 days post-close)
- Features: Only pre-sale data (profile, segment, funnel)
- Train/test split: First 8 months for training, last 4 for testing
- Algorithms:
  1. Logistic Regression (baseline, interpretable)
  2. Random Forest (best performance likely)

**Afternoon (3 hours):**
Build **Model 2: Active Seller Retention**
- Target: Dormant in next 60 days (for sellers who activated)
- Features: Pre-sale + activity data (GMV trend, orders, first-month performance)
- Same train/test methodology

**Evening (2 hours):**
- Model evaluation (AUC, precision@top-20%, recall)
- Feature importance extraction
- Generate risk scores for all 842 sellers

**End-of-Day Output:**
- Two trained models
- Performance summary
- `seller_risk_scores.csv` with churn probabilities


### **DAY 5: Insights & Recommendations (Friday)**

**Morning (3 hours):**
Synthesize findings:
1. **Top 3 Churn Drivers** (from model + segment analysis)
   - Example: "Cat sellers with <10 declared catalog size have 68% never-activate rate vs. 22% baseline"
2. **Segment Opportunities**
   - Which profiles/segments to prioritize in sales?
   - Which need intervention redesign?
3. **Risk Cohort Identification**
   - How many sellers currently at high risk (>70% probability)?
   - Which are highest value (based on declared revenue)?

**Afternoon (3 hours):**
Build **Intervention Priority List:**
- Top 50 at-risk sellers ranked by:
  - Churn probability (model score)
  - Potential value (declared revenue)
  - Days since last order (urgency)
- Suggested action per seller based on profile:
  - Cat: "Schedule 1:1 support call, offer catalog workshop"
  - Shark: "Share competitor benchmark, propose growth tactics"

**Evening (2 hours):**
Create **Executive Summary Presentation (10 slides):**
1. Title: "Seller Churn POC - Week 1 Findings"
2. Churn landscape (rates, cohorts)
3. Top 3 drivers (with visuals)
4. DISC profile insights
5. Model performance summary
6. Risk distribution (how many at-risk now?)
7. Intervention priority list (sample)
8. Quick-win recommendations
9. ROI projection (churn reduction → GMV saved)
10. Next steps (if green-lit for full 4-week)

**End-of-Day Output:**
- Executive presentation (PPT)
- Intervention list (CSV)
- Code repository (Python notebooks)


### **📊 DELIVERABLES PACKAGE (Friday EOD)**

**1. Executive Presentation (10 slides)**
- Suitable for 30-minute stakeholder review
- Clear visuals, minimal text
- Actionable insights highlighted

**2. Priority Intervention List**
- `top_50_at_risk_sellers.csv`
- Columns: seller_id, churn_probability, profile, suggested_action, contact_urgency

**3. Data Assets**
- `seller_risk_scores.csv` (all 842 sellers scored)
- `segment_churn_analysis.csv` (churn rates by dimension)

**4. Code Repository**
- Jupyter notebooks (commented, reproducible)
- Can be rerun monthly as new data arrives

**5. One-Page Summary**
- For quick executive circulation
- Key numbers, top insight, next step recommendation


### **🎯 EXPECTED OUTCOMES**

**Immediate Value:**
1. **Identify 50-100 sellers for urgent intervention** this week
   - Assume 20% save rate from outreach = 10-20 sellers retained
   - At $5K/month GMV each = $50K-$100K monthly GMV saved
   - **Annualized: $600K-$1.2M**

2. **Validate top churn hypothesis**
   - Example: "Sellers declaring $0 revenue churn at 3x rate"
   - Immediate sales process fix: require revenue estimate or counsel expectations

3. **Prove model works**
   - If Model 1 achieves >70% AUC, demonstrates predictive value
   - Sets foundation for ongoing scoring system

**Strategic Insight:**
- Clear answer: "Should we invest in full 4-week analysis?"
- Quantified ROI projection based on Week 1 findings
- Specific roadmap for Weeks 2-4 if approved


### **CRITICAL SUCCESS FACTORS**

**To Make This Work:**

1. **Data Access:** I need all CSV files loaded and queryable by Monday AM

2. **Stakeholder Availability:**
   - 30-min kickoff Monday (confirm churn definitions, priorities)
   - 30-min Friday presentation (review findings, decide on next phase)

3. **Decision Criteria:**
   - What churn reduction % would justify full project?
   - What model performance (AUC) proves value?
   - How many at-risk sellers flagged = "worth it"?

4. **Quick Wins Execution:**
   - If we identify 50 at-risk sellers Friday, can someone action the outreach?
   - Having immediate impact validates the work
