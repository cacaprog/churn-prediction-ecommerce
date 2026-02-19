# Glossary of Terms: E-commerce Churn Analysis

This glossary provides definitions for key terms used in the analysis of churn, specifically tailored for the Olist Ecommerce seller churn project. It includes general data science and business metrics as well as project-specific definitions.

## 1. Core Churn Definitions (Project Specific)

These definitions are specific to the "Seller Churn Quick-Win" Proof of Concept.

*   **Seller Churn**: In this B2B2C context, refers to sellers leaving the platform or ceasing to generate revenue, rather than end-consumers stopping purchases.
*   **Never-activated**: A seller who has closed a deal but has recorded **0 orders within 90 days** of the `won_date`.
*   **Dormant**: A seller who has historically made sales but has had **0 orders in the last 60 days** of the dataset's observation period.
*   **Active Seller**: A seller who has recorded at least one order in the **last 60 days**.
*   **Time-to-First-Sale**: The time elapsed (in days) between the date the deal was won (`won_date`) and the date of the seller's first completed sale.

## 2. General Business & E-commerce Metrics

*   **GMV (Gross Merchandise Value)**: The total value of merchandise sold over a given period through the site. It is a measure of the growth of the business/platform.
*   **CAC (Customer Acquisition Cost)**: The total cost of acquiring a new customer (or seller), including marketing and sales expenses.
*   **CLV (Customer Lifetime Value)**: The total predicted revenue or profit a business can expect from a single customer account throughout the business relationship.
*   **ARPU (Average Revenue Per User)**: The total revenue divided by the number of active users (or sellers) in a specific period.
*   **AOV (Average Order Value)**: The average dollar amount spent each time a customer places an order. Calculated as Total Revenue / Number of Orders.
*   **Cohort Analysis**: A behavioral analytics feature that groups users/sellers who share a common characteristic (usually `start_date` or `won_date`) to analyze their behavior over time.
    *   *Example*: Analyzing the retention rate of all sellers who joined in June 2017 vs. June 2018.
*   **MQL (Marketing Qualified Lead)**: A lead who has indicated interest in what the brand has to offer based on marketing efforts and is more likely to become a customer than other leads.
*   **Sales Cycle Length**: The amount of time it takes to close a deal, measured from the first contact (`first_contact_date`) to the deal closed date (`won_date`).

## 3. Data Science & Modeling Terms

*   **Binary Classification**: A type of machine learning task where the goal is to categorize data into one of two distinct groups (e.g., `Churn` vs. `No Churn`).
*   **Logistic Regression**: A statistical model used to predict the probability of a binary outcome (1/0, Yes/No, Churn/Active). It provides interpretable coefficients indicating how each feature affects the probability.
*   **Random Forest**: An ensemble learning method that operates by constructing a multitude of decision trees at training time. It is often more accurate than simple regression but less interpretable.
*   **Feature Importance**: A score assigned to features in a machine learning model that indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model.
*   **AUC (Area Under the ROC Curve)**: A performance measurement for classification problems at various threshold settings. AUC tells how much the model is capable of distinguishing between classes. Higher is better (1.0 is perfect, 0.5 is random).
*   **Precision**: The ratio of correctly predicted positive observations to the total predicted positive observations.
    *   *In Churn Context*: Of all the sellers we PREDICTED would churn, how many ALREADY churned?
*   **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to the all observations in actual class.
    *   *In Churn Context*: Of all the sellers who ACTUALLY churned, how many did we catch?
*   **Precision@Top-K% (e.g., Top-20%)**: A metric that evaluates the precision of the model only for the top K% of most confident predictions. This is critical for resource-constrained interventions (e.g., "We can only call the top 50 riskiest sellers").
*   **F1-Score**: The weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account.
*   **Survival Analysis**: A set of statistical approaches used to analyze the expected duration of time until one or more events happen, such as seller churn.
*   **Kaplan-Meier Estimator**: A non-parametric statistic used to estimate the survival function from lifetime data. It helps visualize the probability of a seller "surviving" (staying active) past a certain time point.

## 4. Olist/Project Specific Features

*   **DISC Profile**: A behavioral self-assessment tool used to categorize communication styles of sellers.
    *   *Categories*: Cat, Eagle, Wolf, Shark.
    *   *Relevance*: Different personality types may require different retention strategies.
*   **SDR (Sales Development Rep)**: The representative responsible for outbound prospecting and qualifying leads.
*   **SR (Sales Representative)**: The representative responsible for closing the deal with the seller.
*   **Lead Source/Origin**: The channel through which the lead was acquired (e.g., social, organic_search, paid_search).
*   **Business Segment**: The market category the seller operates in (e.g., health_beauty, audio_video_electronics).
*   **Catalog Size**: The number of unique products a seller declares or actually lists on the platform.
