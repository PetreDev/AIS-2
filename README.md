# Web Application Vulnerability Detection

This project implements an end-to-end machine learning pipeline that detects malicious HTTP requests using the CSIC 2010 dataset. The workflow covers data preparation, feature engineering, model training with hyperparameter tuning, evaluation, and artefact generation (figures, metrics, sampled data).

## Project Structure

- `src/` – Python package containing reusable modules for data loading, feature extraction, modelling, evaluation, and reporting.
- `csic_database.csv/` – Original CSIC 2010 dataset (not included in submissions).
- `outputs/` – Generated artefacts such as metrics, plots, and the trained model.
- `data_samples/` – Sampled subsets required for submission.
- `reports/` – Slot for the written report and supporting material.
- `requirements.txt` – Python dependencies.

## Environment Setup

1. Activate the provided virtual environment or create a new one.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Pipeline

Execute the complete workflow (data loading → feature engineering → training → evaluation):

```
python -m src.run_pipeline
```

Key configuration values (e.g., train/test split, TF-IDF settings, hyperparameter grid) are defined in `src/config.py`. Adjust them as needed before running the pipeline.

## Feature Engineering

- **Numeric features** – request/URL/body lengths, special character statistics (counts, ratios, uniqueness), parameter density, digit/uppercase ratios, suspicious keyword frequency, entropy.
- **Textual features** – character-level TF-IDF n-grams computed over the concatenated URL and body.
- **Categorical features** – request method one-hot encoded.

All transformations are composed via `ColumnTransformer` to guarantee consistent preprocessing during training and inference.

## Modelling and Evaluation

- **Algorithm** – Logistic Regression with class weighting (`balanced`) and a tuned regularization strength (`C`).
- **Hyperparameter search** – Grid search over multiple `C` values with 5-fold cross-validation, optimising ROC-AUC while tracking Accuracy and F1.
- **Metrics** – Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curve, and probability histograms.
- **Error analysis** – False positives/negatives exported for manual inspection.

## Generated Artefacts

Running the pipeline populates the `outputs/` directory with:

- `evaluation_metrics.json` – Metrics and classification report.
- `basic_statistics.json` – Dataset-level exploratory statistics.
- `dataset_split_summary.json` – Train/test sizes and class balance.
- `grid_search_results.csv` – Cross-validation scores.
- `best_model_params.json` – Selected hyperparameters.
- `false_positives.csv`, `false_negatives.csv` – Misclassified samples.
- `figures/` – Confusion matrices, ROC curve, and probability histogram.
- `logistic_regression_model.joblib` – Serialized pipeline (preprocessing + classifier).
- `top_logistic_coefficients.csv` – Most influential positive and negative indicators learned by the model.

A 10,000-row sample of the dataset is exported to `data_samples/csic_sample_10k.csv` for submission requirements.

## Next Steps

- Convert the contents of `outputs/` into visual material for the written report.
- Extend feature engineering with additional statistical or context-aware indicators if higher accuracy is required.
- Compare against alternative models (e.g., Random Forest, Gradient Boosting) to validate performance.
- Incorporate insights from `top_logistic_coefficients.csv` and misclassification exports into the written report.
