from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResults:
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Dict[str, float]]
    false_positive_indices: List[int]
    false_negative_indices: List[int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": self.metrics,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
            "false_positive_indices": self.false_positive_indices,
            "false_negative_indices": self.false_negative_indices,
        }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_scores),
    }
    return metrics


def generate_evaluation_results(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> EvaluationResults:
    """Assemble detailed evaluation artefacts."""

    metrics = compute_metrics(y_true, y_pred, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    false_positive_indices = np.where((y_pred == 1) & (y_true == 0))[0].tolist()
    false_negative_indices = np.where((y_pred == 0) & (y_true == 1))[0].tolist()

    return EvaluationResults(
        metrics=metrics,
        confusion_matrix=cm,
        classification_report=report,
        false_positive_indices=false_positive_indices,
        false_negative_indices=false_negative_indices,
    )


def save_metrics_to_json(results: EvaluationResults, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results.to_dict(), file, indent=2)


def plot_confusion_matrix(
    cm: np.ndarray, labels: List[str], output_path: Path, normalize: bool = False
) -> None:
    """Plot and save the confusion matrix."""

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm if not normalize else cm / cm.sum(axis=1, keepdims=True),
        display_labels=labels,
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_curve_figure(
    y_true: np.ndarray, y_scores: np.ndarray, output_path: Path
) -> None:
    """Plot ROC curve."""

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_score_histogram(
    y_true: np.ndarray, y_scores: np.ndarray, output_path: Path
) -> None:
    """Plot histogram of predicted probabilities for positive class."""

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        x=y_scores,
        hue=y_true,
        multiple="stack",
        bins=30,
        palette="Set1",
        ax=ax,
        legend=True,
    )
    ax.set_xlabel("Predicted probability of attack")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Predicted Probabilities")
    ax.legend(title="True label", labels=["Benign", "Attack"])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_class_distribution(labels: pd.Series, output_path: Path) -> None:
    """Visualise the class distribution (attack vs. benign)."""

    label_series = pd.Series(labels).astype(int)
    counts = (
        label_series.map({0: "Benign", 1: "Attack"})
        .value_counts()
        .sort_index()
    )
    colors = ["#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(
        counts.index,
        counts.values,
        color=colors[: len(counts)],
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_method_distribution(features: pd.DataFrame, output_path: Path) -> None:
    """Plot the distribution of HTTP methods."""

    method_counts = (
        features["method"].fillna("UNKNOWN").astype(str).str.upper().value_counts()
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(
        method_counts.index,
        method_counts.values,
        color="#1f77b4",
    )
    ax.set_xlabel("Method")
    ax.set_ylabel("Count")
    ax.set_title("HTTP Method Distribution")
    for index, value in enumerate(method_counts.values):
        ax.text(index, value, f"{int(value)}", ha="center", va="bottom")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_feature_distribution(
    numeric_features: pd.DataFrame,
    labels: pd.Series,
    feature_name: str,
    output_path: Path,
    *,
    bins: int = 50,
) -> None:
    """Plot the distribution of a numeric feature split by class."""

    if feature_name not in numeric_features.columns:
        raise KeyError(f"Feature '{feature_name}' not found in numeric features.")

    label_array = pd.Series(labels).astype(int).to_numpy()
    feature_array = numeric_features[feature_name].to_numpy()

    benign_mask = label_array == 0
    attack_mask = label_array == 1

    benign_data = feature_array[benign_mask]
    attack_data = feature_array[attack_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.histplot(
        benign_data,
        bins=bins,
        color="#1f77b4",
        stat="count",
        alpha=0.8,
        ax=axes[0],
    )
    axes[0].set_title("Benign")
    axes[0].set_xlabel(feature_name.replace("_", " ").title())
    axes[0].set_ylabel("Density")

    sns.histplot(
        attack_data,
        bins=bins,
        color="#d62728",
        stat="count",
        alpha=0.8,
        ax=axes[1],
    )
    axes[1].set_title("Attack")
    axes[1].set_xlabel(feature_name.replace("_", " ").title())

    for axis, data, color in zip(
        axes,
        (benign_data, attack_data),
        ["#1f77b4", "#d62728"],
    ):
        if len(data) > 0:
            mean_value = data.mean()
            axis.axvline(mean_value, color=color, linestyle="--", linewidth=1.5)
            ymax = axis.get_ylim()[1]
            text_y = ymax * 0.9 if ymax > 0 else 0.1
            axis.text(
                mean_value,
                text_y,
                f"Mean = {mean_value:.2f}",
                color=color,
                rotation=90,
                va="top",
                ha="right",
                backgroundcolor="white",
            )

    fig.suptitle(
        f"{feature_name.replace('_', ' ').title()} Distribution by Class",
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(metrics: Dict[str, float], output_path: Path) -> None:
    """Plot a bar chart summarising evaluation metrics."""

    metric_series = pd.Series(metrics)
    metric_series = metric_series.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = metric_series.index.str.upper()
    ax.barh(
        categories,
        metric_series.values,
        color="#1f77b4",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_ylabel("Metric")
    ax.set_title("Evaluation Metric Summary")
    for bar in ax.patches:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(width + 0.01, y, f"{width:.3f}", va="center", ha="left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_top_coefficients(
    coefficients: pd.DataFrame,
    output_path: Path,
    *,
    top_k: int = 15,
) -> None:
    """Plot the most influential features learned by the linear model."""

    if {"feature", "coefficient", "direction"} - set(coefficients.columns):
        raise KeyError(
            "Coefficients DataFrame must contain 'feature', 'coefficient', and 'direction' columns."
        )

    attack_top = (
        coefficients[coefficients["direction"] == "attack"]
        .nlargest(top_k, "coefficient")
    )
    benign_top = (
        coefficients[coefficients["direction"] == "benign"]
        .nsmallest(top_k, "coefficient")
    )
    top_features = pd.concat([benign_top, attack_top], ignore_index=True)
    top_features = top_features.sort_values("coefficient")
    top_features["direction_label"] = top_features["direction"].str.capitalize()

    palette = {"Attack": "#d62728", "Benign": "#1f77b4"}

    fig_height = 0.4 * len(top_features) + 2
    fig, ax = plt.subplots(figsize=(10, max(fig_height, 6)))
    sns.barplot(
        data=top_features,
        x="coefficient",
        y="feature",
        hue="direction_label",
        palette=palette,
        orient="h",
        ax=ax,
    )
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_k} Positive and Negative Coefficients")
    ax.axvline(0, color="black", linewidth=1)
    ax.legend(title="Direction", loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)