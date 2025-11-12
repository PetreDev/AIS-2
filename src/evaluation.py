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

