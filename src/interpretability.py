from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _get_feature_names(preprocessing: Pipeline) -> np.ndarray:
    """Retrieve feature names from a fitted preprocessing pipeline."""

    try:
        feature_names = preprocessing.get_feature_names_out()
    except AttributeError as exc:  # pragma: no cover - defensive fallback
        raise AttributeError(
            "Unable to extract feature names from preprocessing pipeline. "
            "Ensure scikit-learn >= 1.3 is installed."
        ) from exc
    return np.asarray(feature_names, dtype=object)


def export_top_coefficients(
    model_pipeline: Pipeline,
    top_k: int,
    output_path: Path,
    attack_label: int = 1,
) -> None:
    """
    Persist the top-k positive and negative coefficients from a linear model.

    Args:
        model_pipeline: Fitted pipeline with `preprocessing` and linear `classifier`.
        top_k: Number of coefficients to include for each direction.
        output_path: Target CSV path.
        attack_label: Class index treated as the positive (attack) class.
    """

    preprocessing = model_pipeline.named_steps["preprocessing"]
    classifier = model_pipeline.named_steps["classifier"]

    feature_names = _get_feature_names(preprocessing)

    classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise AttributeError("Classifier must expose `classes_` after fitting.")

    coef_matrix = classifier.coef_

    if coef_matrix.shape[0] == len(classes):
        try:
            class_index = int(np.where(classes == attack_label)[0][0])
        except IndexError as exc:
            raise ValueError(
                f"Attack label {attack_label} not found among classifier classes: {classes}"
            ) from exc
        coefficients = coef_matrix[class_index]
    elif coef_matrix.shape[0] == len(classes) - 1:
        positive_class = classes[-1]
        if attack_label != positive_class:
            raise ValueError(
                "Binary classifier stores only positive-class coefficients; "
                f"expected attack_label={positive_class}, received {attack_label}."
            )
        coefficients = coef_matrix[0]
    else:  # pragma: no cover - defensive branch
        raise ValueError(
            f"Unexpected coefficient matrix shape {coef_matrix.shape} for classes {classes}."
        )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    )

    top_positive = feature_importance.nlargest(top_k, "coefficient").assign(direction="attack")
    top_negative = feature_importance.nsmallest(top_k, "coefficient").assign(direction="benign")

    combined = pd.concat([top_positive, top_negative], axis=0, ignore_index=True)
    combined = combined[["feature", "coefficient", "abs_coefficient", "direction"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

