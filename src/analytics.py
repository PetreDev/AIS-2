from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BasicStatistics:
    total_requests: int
    attack_requests: int
    benign_requests: int
    attack_ratio: float
    average_request_length: float
    min_request_length: int
    max_request_length: int
    method_distribution: Dict[str, float]

    def to_dict(self) -> Dict[str, float | int | Dict[str, float]]:
        return {
            "total_requests": self.total_requests,
            "attack_requests": self.attack_requests,
            "benign_requests": self.benign_requests,
            "attack_ratio": self.attack_ratio,
            "average_request_length": self.average_request_length,
            "min_request_length": self.min_request_length,
            "max_request_length": self.max_request_length,
            "method_distribution": self.method_distribution,
        }


def compute_basic_statistics(features: pd.DataFrame, labels: pd.Series) -> BasicStatistics:
    """Calculate dataset-level statistics useful for exploratory analysis."""

    total_requests = int(len(labels))
    attack_requests = int(labels.sum())
    benign_requests = total_requests - attack_requests
    attack_ratio = attack_requests / total_requests if total_requests else 0.0

    request_lengths = features["url"].str.len() + features["body"].str.len()
    average_length = float(request_lengths.mean())
    min_length = int(request_lengths.min()) if total_requests else 0
    max_length = int(request_lengths.max()) if total_requests else 0

    method_counts = features["method"].value_counts(normalize=True).to_dict()
    method_distribution = {method: float(freq) for method, freq in method_counts.items()}

    return BasicStatistics(
        total_requests=total_requests,
        attack_requests=attack_requests,
        benign_requests=benign_requests,
        attack_ratio=attack_ratio,
        average_request_length=average_length,
        min_request_length=min_length,
        max_request_length=max_length,
        method_distribution=method_distribution,
    )

