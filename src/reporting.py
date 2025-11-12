from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .analytics import BasicStatistics


def save_basic_statistics(statistics: BasicStatistics, output_path: Path) -> None:
    """Persist basic statistics to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(statistics.to_dict(), file, indent=2)


def export_false_predictions(
    features: pd.DataFrame,
    true_labels: pd.Series,
    predicted_labels: Iterable[int],
    predicted_scores: Iterable[float],
    indices: Iterable[int],
    output_path: Path,
) -> None:
    """Save misclassified samples for manual review."""

    index_list = list(indices)
    subset = features.iloc[index_list].copy()
    subset["true_label"] = true_labels.iloc[index_list].to_numpy()
    subset["predicted_label"] = pd.Series(predicted_labels).iloc[index_list].to_numpy()
    subset["predicted_score"] = pd.Series(predicted_scores).iloc[index_list].to_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=True)


def export_sample_dataset(
    raw_dataframe: pd.DataFrame, sample_size: int, random_state: int, output_path: Path
) -> None:
    """Export a representative sample of the dataset for submission."""

    sample = (
        raw_dataframe.sample(n=sample_size, random_state=random_state, replace=False)
        if len(raw_dataframe) > sample_size
        else raw_dataframe.copy()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)

