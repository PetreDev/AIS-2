from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Tuple


@dataclass(frozen=True)
class PathsConfig:
    """Collection of project paths used across the pipeline."""

    project_root: Path
    dataset: Path
    outputs: Path
    figures: Path
    data_samples: Path

    @staticmethod
    def from_root(root: Path) -> "PathsConfig":
        return PathsConfig(
            project_root=root,
            dataset=root / "csic_database.csv" / "csic_database.csv",
            outputs=root / "outputs",
            figures=root / "outputs" / "figures",
            data_samples=root / "data_samples",
        )


@dataclass(frozen=True)
class DataConfig:
    """Configuration related to data ingestion and splitting."""

    label_column: str = "classification"
    sample_size: int | None = 20000
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    drop_duplicates: bool = False
    duplicate_subset: Tuple[str, ...] | None = ("URL", "content", "classification")


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature extraction and vectorisation."""

    special_characters: Tuple[str, ...] = ("'", '"', "<", ">", ";", "(", ")", "=", "%")
    suspicious_keywords: Tuple[str, ...] = (
        "select",
        "union",
        "insert",
        "update",
        "delete",
        "drop",
        "sleep",
        "benchmark",
        "script",
        "iframe",
        "<img",
        "onerror",
        "onload",
    )
    max_entropy_string_length: int = 2048
    tfidf_ngram_range: Tuple[int, int] = (3, 5)
    tfidf_min_df: int = 5
    tfidf_max_features: int | None = 50000
    tfidf_analyzer: str = "char_wb"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the classification model."""

    c_values: Tuple[float, ...] = (0.1, 1.0, 10.0)
    penalty: str = "l2"
    solver: str = "liblinear"
    max_iter: int = 200
    class_weight: str | None = "balanced"
    scoring: Tuple[str, ...] = ("accuracy", "f1", "roc_auc")
    refit: str = "roc_auc"
    cv_folds: int = 5
    random_state: int | None = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration object passed through the pipeline."""

    paths: PathsConfig
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @staticmethod
    def from_root(project_root: Path) -> "ExperimentConfig":
        return ExperimentConfig(paths=PathsConfig.from_root(project_root))


def ensure_directories(paths: Iterable[Path]) -> None:
    """Ensure that a set of directories exists on disk."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

