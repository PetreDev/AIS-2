# Web Application Vulnerability Detection – Report Template

## 1. Introduction

- Problem statement and motivation.
- Dataset description (CSIC 2010).
- Brief overview of the selected machine learning approach (Logistic Regression).

## 2. Data Preparation

- Loading procedure and sampling strategy (20,000 rows; 80/20 split).
- Data cleaning steps and handling of missing values.
- Overview of the sample dataset exported for submission.

## 3. Exploratory Analysis & Feature Engineering

- Basic statistics (attack ratio, request length distribution, method distribution).
- Description of engineered features:
  - Numeric signals (lengths, special character ratios/uniqueness, digit & uppercase density, suspicious keyword counts, entropy).
  - TF-IDF character n-grams.
  - Method encoding.
- Rationale for each feature group.

## 4. Modelling Strategy

- Logistic Regression configuration (class weighting, hyperparameter grid, solver).
- Cross-validation setup and scoring metrics.
- Summary of selected hyperparameters (`best_model_params.json`).

## 5. Evaluation

- Present key metrics from `evaluation_metrics.json`.
- Confusion matrix, ROC curve, probability histogram.
- Discussion of false positives/negatives (insights from CSV exports).

## 6. Interpretation & Insights

- Highlight significant behaviours or patterns revealed by the model.
- Feature importance discussion leveraging `top_logistic_coefficients.csv` and qualitative analysis of prominent n-grams.
- Limitations of the approach.

## 7. Recommendations & Future Work

- Suggestions for improving detection performance.
- Potential enhancements (additional features, alternative algorithms, model stacking).
- Ideas for deployment or integration into security monitoring workflows.

## 8. References

- Cite dataset sources, libraries, and any external material.
