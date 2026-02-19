# Churn Prediction Models — Technical Evaluation Report

_Generated: 2026-02-19 17:59:31_

---

## Pre-Activation Model — Performance Metrics

- **Accuracy:** 0.919
- **Precision:** 0.924
- **Recall:** 0.946
- **F1 Score:** 0.935
- **ROC AUC:** 0.966

```
              precision    recall  f1-score   support

           0       0.91      0.88      0.89        82
           1       0.92      0.95      0.93       129

    accuracy                           0.92       211
   macro avg       0.92      0.91      0.91       211
weighted avg       0.92      0.92      0.92       211

```

### ROC Curve: Pre-Activation Model

![ROC Curve: Pre-Activation Model](figures/roc_curve_pre-activation_model.png)

### Precision-Recall Curve: Pre-Activation Model

![Precision-Recall Curve: Pre-Activation Model](figures/pr_curve_pre-activation_model.png)

### Confusion Matrix: Pre-Activation Model

![Confusion Matrix: Pre-Activation Model](figures/confusion_matrix_pre-activation_model.png)

### Feature Importance: Pre-Activation Model

![Feature Importance: Pre-Activation Model](figures/feature_importance_pre-activation_model.png)

## Retention Model — Performance Metrics

- **Accuracy:** 0.549
- **Precision:** 0.627
- **Recall:** 0.640
- **F1 Score:** 0.634
- **ROC AUC:** 0.579

```
              precision    recall  f1-score   support

           0       0.42      0.41      0.41        32
           1       0.63      0.64      0.63        50

    accuracy                           0.55        82
   macro avg       0.52      0.52      0.52        82
weighted avg       0.55      0.55      0.55        82

```

### ROC Curve: Retention Model

![ROC Curve: Retention Model](figures/roc_curve_retention_model.png)

### Precision-Recall Curve: Retention Model

![Precision-Recall Curve: Retention Model](figures/pr_curve_retention_model.png)

### Confusion Matrix: Retention Model

![Confusion Matrix: Retention Model](figures/confusion_matrix_retention_model.png)

### Feature Importance: Retention Model

![Feature Importance: Retention Model](figures/feature_importance_retention_model.png)
