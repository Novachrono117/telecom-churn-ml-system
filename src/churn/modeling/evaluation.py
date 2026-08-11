"""Paired cross-validation of the baselines on the training pool.

Two properties make this module the leakage-critical piece of Phase 5.

**Preprocessing is refitted inside every fold.** The estimators handed in are
complete pipelines — cleaner, ``ColumnTransformer``, classifier — and each fold
starts from a fresh :func:`sklearn.base.clone`. The scaler's mean, the encoder's
categories and the classifier's coefficients are therefore estimated on the
training fold alone; the validation fold is only ever transformed and predicted.
Calling ``preprocessor.fit_transform`` on the whole pool before splitting would
leak every validation fold into every other fold's preprocessing.

**Every model sees exactly the same folds.** ``StratifiedKFold`` with a fixed
``random_state`` yields an identical partition on each call, so the comparison
is paired: a difference between two models cannot be an artefact of one having
drawn an easier split.

Per-fold metrics and out-of-fold predictions come from a **single** pass. Using
``cross_validate`` plus ``cross_val_predict`` would fit each model twice and
leave open the possibility of the two passes disagreeing.

This module never loads data. It receives ``X`` and ``y`` already prepared, so
it has no way to reach the holdout even by accident.
"""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from churn.config import get_config
from churn.modeling.metrics import (
    DEFAULT_THRESHOLD,
    METRIC_NAMES,
    ClassificationMetrics,
    compute_metrics,
)

logger = logging.getLogger(__name__)

#: Cross-validation policy for Phase 5. Stratified because the target is
#: imbalanced and an unstratified fold could shift the positive rate enough to
#: make the fold-to-fold spread a sampling artefact.
N_SPLITS = 5
SHUFFLE = True
CV_STRATEGY = "StratifiedKFold"


@dataclass(frozen=True)
class FoldResult:
    """Metrics of one validation fold."""

    fold: int
    n_train: int
    n_validation: int
    metrics: ClassificationMetrics


@dataclass(frozen=True)
class ModelEvaluation:
    """Everything Phase 5 measures about one baseline."""

    name: str
    folds: tuple[FoldResult, ...]
    oof_probability: np.ndarray
    oof_metrics: ClassificationMetrics
    converged: bool
    max_iterations_used: int | None

    def metric_values(self, metric: str) -> np.ndarray:
        """Return the per-fold values of one metric."""
        return np.array([getattr(fold.metrics, metric) for fold in self.folds], dtype=float)

    def mean(self, metric: str) -> float:
        """Mean of a metric across folds."""
        return float(self.metric_values(metric).mean())

    def std(self, metric: str) -> float:
        """Sample standard deviation (ddof=1) of a metric across folds."""
        return float(self.metric_values(metric).std(ddof=1))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return ``{metric: {"mean": ..., "std": ...}}`` for every metric."""
        return {name: {"mean": self.mean(name), "std": self.std(name)} for name in METRIC_NAMES}


def build_splitter(seed: int | None = None) -> StratifiedKFold:
    """Return the Phase 5 cross-validation splitter.

    The same object — or any object built by this function — produces the same
    partition every time it is asked, which is what makes the comparison paired.

    Args:
        seed: Random state. Defaults to the configured project seed.

    Returns:
        The configured :class:`~sklearn.model_selection.StratifiedKFold`.
    """
    random_state = get_config().seed if seed is None else seed
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=random_state)


def _final_estimator(estimator: BaseEstimator) -> BaseEstimator:
    steps = getattr(estimator, "steps", None)
    return steps[-1][1] if steps else estimator


def evaluate_model(
    name: str,
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
    threshold: float = DEFAULT_THRESHOLD,
) -> ModelEvaluation:
    """Cross-validate one estimator, collecting fold metrics and OOF predictions.

    Args:
        name: Model name, for logging and reporting.
        estimator: Unfitted estimator; cloned before every fold.
        features: Canonical feature matrix of the training pool.
        target: Encoded target aligned with ``features``.
        splitter: Cross-validation splitter, shared across models.
        threshold: Decision rule for the threshold-dependent metrics.

    Returns:
        The :class:`ModelEvaluation`.

    Raises:
        RuntimeError: If any row fails to receive an out-of-fold prediction.
    """
    y = np.asarray(target)
    oof = np.full(len(y), np.nan, dtype=float)
    folds: list[FoldResult] = []
    converged = True
    iterations: list[int] = []

    for number, (train_index, validation_index) in enumerate(splitter.split(features, y), start=1):
        model = clone(estimator)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(features.iloc[train_index], y[train_index])
        if any(issubclass(entry.category, ConvergenceWarning) for entry in caught):
            converged = False
            logger.warning("%s: fold %d did not converge", name, number)

        probability = model.predict_proba(features.iloc[validation_index])[:, 1]
        oof[validation_index] = probability

        n_iter = getattr(_final_estimator(model), "n_iter_", None)
        if n_iter is not None:
            iterations.append(int(np.max(n_iter)))

        folds.append(
            FoldResult(
                fold=number,
                n_train=int(len(train_index)),
                n_validation=int(len(validation_index)),
                metrics=compute_metrics(y[validation_index], probability, threshold),
            )
        )

    if np.isnan(oof).any():
        raise RuntimeError(
            f"{int(np.isnan(oof).sum())} row(s) received no out-of-fold prediction for {name!r}. "
            "Every row must be in exactly one validation fold."
        )

    logger.info(
        "%s: %d folds, mean ROC-AUC %.4f",
        name,
        len(folds),
        float(np.mean([fold.metrics.roc_auc for fold in folds])),
    )
    return ModelEvaluation(
        name=name,
        folds=tuple(folds),
        oof_probability=oof,
        oof_metrics=compute_metrics(y, oof, threshold),
        converged=converged,
        max_iterations_used=max(iterations) if iterations else None,
    )


def evaluate_models(
    models: Mapping[str, BaseEstimator],
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> OrderedDict[str, ModelEvaluation]:
    """Cross-validate several estimators against the same folds.

    Args:
        models: Mapping of name to unfitted estimator.
        features: Canonical feature matrix of the training pool.
        target: Encoded target aligned with ``features``.
        splitter: Shared splitter. Built from the configured seed when omitted.
        threshold: Decision rule for the threshold-dependent metrics.

    Returns:
        An ordered mapping of name to :class:`ModelEvaluation`, in input order.
    """
    shared = splitter or build_splitter()
    results: OrderedDict[str, ModelEvaluation] = OrderedDict()
    for name, estimator in models.items():
        results[name] = evaluate_model(name, estimator, features, target, shared, threshold)
    return results
