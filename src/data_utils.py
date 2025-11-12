from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DataConfig, PathsConfig


def load_raw_dataset(paths: PathsConfig) -> pd.DataFrame:
    """Load the CSIC dataset into a pandas DataFrame with cleaned column names."""

    dataframe = pd.read_csv(paths.dataset)

    # Normalise leading/trailing whitespace in column names.
    dataframe.columns = [col.strip() if isinstance(col, str) else col for col in dataframe.columns]

    # Ensure the first column (labels: Normal/Anomalous) has a meaningful name.
    first_column = dataframe.columns[0]
    if not first_column or first_column.startswith("Unnamed"):
        dataframe = dataframe.rename(columns={first_column: "label"})
    else:
        dataframe = dataframe.rename(columns={first_column: "label"})

    return dataframe


def prepare_feature_frame(raw_df: pd.DataFrame, data_cfg: DataConfig) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the feature DataFrame used by the modelling pipeline.

    Returns:
        features: DataFrame containing columns required for the pipeline.
        labels: Series of binary labels (1 = attack, 0 = benign).
    """

    dataframe = raw_df.copy()

    if data_cfg.drop_duplicates and data_cfg.duplicate_subset:
        dataframe = dataframe.drop_duplicates(subset=list(data_cfg.duplicate_subset))

    label_series = dataframe[data_cfg.label_column]
    numeric_labels = pd.to_numeric(label_series, errors="coerce")

    if numeric_labels.isna().any():
        mapped_labels = (
            label_series.astype(str)
            .str.strip()
            .str.lower()
            .map({"normal": 0, "anomalous": 1, "attack": 1, "benign": 0})
        )
        numeric_labels = numeric_labels.fillna(mapped_labels)

    if numeric_labels.isna().any():
        raise ValueError("Unable to parse classification labels into numeric values.")

    dataframe["classification"] = numeric_labels.astype(int)

    if data_cfg.sample_size and len(dataframe) > data_cfg.sample_size:
        stratify_labels = dataframe["classification"] if data_cfg.stratify else None
        dataframe, _ = train_test_split(
            dataframe,
            train_size=data_cfg.sample_size,
            random_state=data_cfg.random_state,
            stratify=stratify_labels,
        )
        dataframe = dataframe.sort_index()

    labels = dataframe["classification"]

    # Basic text columns for downstream feature engineering.
    features = pd.DataFrame(
        {
            "method": dataframe["Method"].fillna("UNKNOWN").str.upper().str.strip(),
            "url": dataframe["URL"].fillna("").astype(str),
            "body": dataframe.get(
                "content", pd.Series("", index=dataframe.index, dtype=str)
            ).fillna("").astype(str),
        }
    )
    features["raw_request"] = (features["url"] + " " + features["body"]).str.strip()

    return features, labels


def train_test_split_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    data_cfg: DataConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and test subsets."""

    stratify_labels = labels if data_cfg.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=data_cfg.test_size,
        random_state=data_cfg.random_state,
        stratify=stratify_labels,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )

