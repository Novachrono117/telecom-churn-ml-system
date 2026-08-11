"""Classification metrics for the baseline comparison.

Two families, kept apart on purpose because they answer different questions.

**Threshold-independent** — how well the model *ranks* customers by risk:

* ``roc_auc`` — ``sklearn.metrics.roc_auc_score``.
* ``average_precision`` — ``sklearn.metrics.average_precision_score``. This is
  reported as **Average Precision**, never as "PR-AUC", because the two are not
  the same computation: AP is a weighted sum of precisions at each threshold,
  whereas a PR-AUC obtained by trapezoidal integration interpolates between
  operating points and is optimistically biased. Calling it AP removes the
  ambiguity about which one produced the number.

**Threshold-dependent** — what happens when the ranking is turned into a
decision at the 0.5 default rule. Nothing here is optimised: 0.5 is an
operational reference for Phase 5, not a business decision.

``accuracy`` is computed but is auxiliary. At a 26.5% positive rate a model that
never predicts churn already reaches ~73%, so accuracy cannot rank these models
and is never used to promote one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

#: Operational reference rule for Phase 5. Not optimised, not a business choice.
DEFAULT_THRESHOLD = 0.5

#: Metric names in reporting order. ``accuracy`` stays last: auxiliary only.
METRIC_NAMES: tuple[str, ...] = (
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
    "accuracy",
)

#: Metrics that do not depend on the decision threshold.
THRESHOLD_INDEPENDENT: tuple[str, ...] = ("roc_auc", "average_precision")


@dataclass(frozen=True)
class ClassificationMetrics:
    """One evaluation of predicted probabilities against known labels."""

    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics as a plain mapping, in :data:`METRIC_NAMES` order."""
        values = asdict(self)
        return {name: values[name] for name in METRIC_NAMES}


def compute_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> ClassificationMetrics:
    """Evaluate predicted positive-class probabilities.

    ``zero_division=0`` is passed explicitly because the majority baseline
    predicts no positives at all: precision is then undefined (0/0) and is
    reported as ``0.0``. That is the honest reading — a model that never fires
    has no precision to speak of — and it keeps the run free of warnings that
    would otherwise hide a real one.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Predicted probability of the positive class.
        threshold: Decision rule applied to obtain hard labels.

    Returns:
        The six metrics.
    """
    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability, dtype=float)
    y_predicted = (y_probability >= threshold).astype(int)

    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_probability)),
        average_precision=float(average_precision_score(y_true, y_probability)),
        precision=float(precision_score(y_true, y_predicted, zero_division=0)),
        recall=float(recall_score(y_true, y_predicted, zero_division=0)),
        f1=float(f1_score(y_true, y_predicted, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_predicted)),
    )


def confusion_at_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Return the 2x2 confusion matrix ``[[TN, FP], [FN, TP]]`` at ``threshold``."""
    y_predicted = (np.asarray(y_probability, dtype=float) >= threshold).astype(int)
    return confusion_matrix(np.asarray(y_true), y_predicted, labels=[0, 1])


def positive_prevalence(y_true: np.ndarray) -> float:
    """Return the share of positives.

    Used as the interpretation reference for Average Precision: a ranking with
    no signal converges to an AP equal to the positive prevalence, so an AP is
    only meaningful relative to that number.
    """
    values = np.asarray(y_true)
    return float(values.mean()) if values.size else 0.0
