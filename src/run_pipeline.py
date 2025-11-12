from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from .analytics import compute_basic_statistics
from .config import ExperimentConfig, ensure_directories
from .data_utils import load_raw_dataset, prepare_feature_frame, train_test_split_dataset
from .evaluation import (
    generate_evaluation_results,
    plot_confusion_matrix,
    plot_roc_curve_figure,
    plot_score_histogram,
    save_metrics_to_json,
)
from .features import build_preprocessing_pipeline
from .interpretability import export_top_coefficients
from .modeling import build_model_pipeline
from .reporting import export_false_predictions, export_sample_dataset, save_basic_statistics


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = ExperimentConfig.from_root(project_root)

    ensure_directories(
        [
            config.paths.outputs,
            config.paths.figures,
            config.paths.data_samples,
        ]
    )

    LOGGER.info("Loading dataset from %s", config.paths.dataset)
    raw_dataframe = load_raw_dataset(config.paths)
    features, labels = prepare_feature_frame(raw_dataframe, config.data)
    LOGGER.info("Loaded %d samples (attacks: %d)", len(labels), labels.sum())

    LOGGER.info("Computing basic statistics")
    basic_stats = compute_basic_statistics(features, labels)
    save_basic_statistics(basic_stats, config.paths.outputs / "basic_statistics.json")

    LOGGER.info("Exporting sample dataset for submission")
    export_sample_dataset(
        raw_dataframe,
        sample_size=10_000,
        random_state=config.data.random_state,
        output_path=config.paths.data_samples / "csic_sample_10k.csv",
    )

    LOGGER.info("Splitting dataset into train/test")
    X_train, X_test, y_train, y_test = train_test_split_dataset(features, labels, config.data)
    with (config.paths.outputs / "dataset_split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "positive_ratio_train": float(y_train.mean()),
                "positive_ratio_test": float(y_test.mean()),
                "random_state": config.data.random_state,
            },
            file,
            indent=2,
        )

    LOGGER.info("Building preprocessing pipeline")
    preprocessing = build_preprocessing_pipeline(config.features)

    LOGGER.info("Setting up model and grid search")
    model_search = build_model_pipeline(preprocessing, config.model)

    LOGGER.info("Training model with hyperparameter optimisation")
    model_search.fit(X_train, y_train)

    LOGGER.info("Best hyperparameters: %s", model_search.best_params_)
    with (config.paths.outputs / "best_model_params.json").open("w", encoding="utf-8") as file:
        json.dump(model_search.best_params_, file, indent=2)

    cv_results = pd.DataFrame(model_search.cv_results_)
    cv_results.to_csv(config.paths.outputs / "grid_search_results.csv", index=False)

    LOGGER.info("Evaluating on test set")
    y_pred = model_search.predict(X_test)
    y_scores = model_search.predict_proba(X_test)[:, 1]

    evaluation = generate_evaluation_results(y_test.to_numpy(), y_pred, y_scores)
    save_metrics_to_json(evaluation, config.paths.outputs / "evaluation_metrics.json")

    LOGGER.info("Saving evaluation plots")
    plot_confusion_matrix(
        evaluation.confusion_matrix,
        labels=["Benign", "Attack"],
        output_path=config.paths.figures / "confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        evaluation.confusion_matrix,
        labels=["Benign", "Attack"],
        output_path=config.paths.figures / "confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_roc_curve_figure(y_test.to_numpy(), y_scores, config.paths.figures / "roc_curve.png")
    plot_score_histogram(y_test.to_numpy(), y_scores, config.paths.figures / "probability_histogram.png")

    LOGGER.info("Exporting misclassified samples for manual review")
    if evaluation.false_positive_indices:
        export_false_predictions(
            X_test,
            y_test,
            y_pred,
            y_scores,
            evaluation.false_positive_indices,
            config.paths.outputs / "false_positives.csv",
        )
    if evaluation.false_negative_indices:
        export_false_predictions(
            X_test,
            y_test,
            y_pred,
            y_scores,
            evaluation.false_negative_indices,
            config.paths.outputs / "false_negatives.csv",
        )

    LOGGER.info("Persisting trained model")
    best_model = model_search.best_estimator_
    joblib.dump(best_model, config.paths.outputs / "logistic_regression_model.joblib")

    LOGGER.info("Exporting most influential features")
    export_top_coefficients(
        best_model,
        top_k=25,
        output_path=config.paths.outputs / "top_logistic_coefficients.csv",
    )

    LOGGER.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()

