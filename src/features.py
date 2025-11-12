from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import FeatureConfig


def shannon_entropy(text: str) -> float:
    """Compute the Shannon entropy of a text string."""

    if not text:
        return 0.0
    counts = np.array(list(Counter(text).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


class HTTPRequestFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract numeric features from HTTP request fields."""

    def __init__(
        self,
        special_characters: Sequence[str],
        suspicious_keywords: Sequence[str],
        max_entropy_length: int = 2048,
    ) -> None:
        self.special_characters = tuple(special_characters)
        self.suspicious_keywords = tuple(suspicious_keywords)
        self._keywords_lower = tuple(keyword.lower() for keyword in self.suspicious_keywords)
        self.max_entropy_length = max_entropy_length
        self._feature_names: List[str] = [
            "url_length",
            "body_length",
            "request_length",
            "special_char_count",
            "special_char_ratio",
            "parameter_count",
            "unique_special_char_count",
            "digit_count",
            "digit_ratio",
            "uppercase_ratio",
            "suspicious_keyword_count",
            "entropy",
        ]

    def fit(self, X: pd.DataFrame, y: Iterable | None = None):  # type: ignore[override]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # type: ignore[override]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=["url", "body"])

        if X.empty:
            # Return empty array with correct number of features
            return np.empty((0, len(self._feature_names)), dtype=float)

        if "url" not in X.columns or "body" not in X.columns:
            raise ValueError("Input DataFrame must contain 'url' and 'body' columns")

        url = X["url"].astype(str)
        body = X["body"].astype(str)
        combined = (url + " " + body).str.strip()
        combined_lower = combined.str.lower()

        url_length = url.str.len().to_numpy(dtype=float)
        body_length = body.str.len().to_numpy(dtype=float)
        request_length = combined.str.len().to_numpy(dtype=float)

        # Vectorized special character counting
        special_char_count = np.zeros(len(combined), dtype=float)
        for char in self.special_characters:
            escaped_char = re.escape(char)  # Escape regex special characters
            special_char_count += combined.str.count(escaped_char).to_numpy(dtype=float)
        special_char_ratio = np.where(request_length == 0, 0.0, special_char_count / request_length)

        # Vectorized parameter counting
        parameter_count = (
            combined.str.count(re.escape("=")).add(combined.str.count(re.escape("&"))).to_numpy(dtype=float)
        )

        # Vectorized unique special character counting
        unique_special_char_count = np.zeros(len(combined), dtype=float)
        for char in self.special_characters:
            unique_special_char_count += combined.str.contains(char, regex=False).astype(float).to_numpy()
        unique_special_char_count = unique_special_char_count.astype(float)

        # Vectorized digit counting
        digit_count = combined.apply(lambda x: sum(c.isdigit() for c in x)).to_numpy(dtype=float)
        digit_ratio = np.where(request_length == 0, 0.0, digit_count / request_length)

        # Vectorized uppercase counting
        uppercase_count = combined.apply(lambda x: sum(c.isupper() for c in x)).to_numpy(dtype=float)
        uppercase_ratio = np.where(request_length == 0, 0.0, uppercase_count / request_length)

        # Vectorized suspicious keyword counting
        suspicious_keyword_count = np.zeros(len(combined_lower), dtype=float)
        for keyword in self._keywords_lower:
            suspicious_keyword_count += combined_lower.str.count(keyword).to_numpy(dtype=float)

        truncated = combined.str.slice(0, self.max_entropy_length)
        entropy_values = truncated.apply(shannon_entropy).to_numpy(dtype=float)

        return np.vstack(
            (
                url_length,
                body_length,
                request_length,
                special_char_count,
                special_char_ratio,
                parameter_count,
                unique_special_char_count,
                digit_count,
                digit_ratio,
                uppercase_ratio,
                suspicious_keyword_count,
                entropy_values,
            )
        ).T

    def get_feature_names_out(self, input_features=None) -> np.ndarray:  # type: ignore[override]
        return np.asarray(self._feature_names, dtype=object)


def build_preprocessing_pipeline(feature_cfg: FeatureConfig) -> ColumnTransformer:
    """Create the full preprocessing pipeline combining numeric and text features."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "custom_features",
                HTTPRequestFeatureExtractor(
                    special_characters=feature_cfg.special_characters,
                    suspicious_keywords=feature_cfg.suspicious_keywords,
                    max_entropy_length=feature_cfg.max_entropy_string_length,
                ),
            ),
            ("scaler", MaxAbsScaler()),
        ]
    )

    text_vectoriser = TfidfVectorizer(
        analyzer=feature_cfg.tfidf_analyzer,
        ngram_range=feature_cfg.tfidf_ngram_range,
        min_df=feature_cfg.tfidf_min_df,
        max_features=feature_cfg.tfidf_max_features,
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore"),
            )
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            # ("numeric", numeric_pipeline, ["url", "body"]),  # Disabled
            ("tfidf", text_vectoriser, "raw_request"),
            ("categorical", categorical_pipeline, ["method"]),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocessing

