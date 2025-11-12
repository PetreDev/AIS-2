from __future__ import annotations

from typing import Dict, Iterable

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .config import ModelConfig


def build_model_pipeline(preprocessing: Pipeline, model_cfg: ModelConfig) -> GridSearchCV:
    """Create the full modelling pipeline coupled with hyperparameter optimisation."""

    classifier = LogisticRegression(
        penalty=model_cfg.penalty,
        C=1.0,
        solver=model_cfg.solver,
        class_weight=model_cfg.class_weight,
        max_iter=model_cfg.max_iter,
        random_state=model_cfg.random_state,
        n_jobs=None,  # liblinear does not support parallel fitting through n_jobs
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )

    param_grid: Dict[str, Iterable[float]] = {"classifier__C": model_cfg.c_values}

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=list(model_cfg.scoring),
        refit=model_cfg.refit,
        cv=model_cfg.cv_folds,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    return grid_search

