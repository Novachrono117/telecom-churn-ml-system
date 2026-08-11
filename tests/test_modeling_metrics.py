"""Metric computation, checked against hand-computable cases.

Synthetic vectors with known answers, not the real dataset: a metric test that
depends on the data cannot tell a metric bug from a data change.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from churn.modeling.metrics import (
    DEFAULT_THRESHOLD,
    METRIC_NAMES,
    THRESHOLD_INDEPENDENT,
    compute_metrics,
    confusion_at_threshold,
    positive_prevalence,
)


def test_perfect_ranking_and_perfect_decision() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_probability = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_probability)

    assert metrics.roc_auc == 1.0
    assert metrics.average_precision == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.accuracy == 1.0


def test_inverted_ranking_scores_zero_auc() -> None:
    metrics = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1]))

    assert metrics.roc_auc == 0.0


def test_constant_probability_is_a_coin_flip_for_roc_auc() -> None:
    """A model that gives everyone the same score cannot rank anything."""
    metrics = compute_metrics(np.array([0, 0, 0, 1]), np.zeros(4))

    assert metrics.roc_auc == 0.5


def test_constant_probability_gives_average_precision_equal_to_prevalence() -> None:
    y_true = np.array([0, 0, 0, 1, 0, 0, 1, 0])

    metrics = compute_metrics(y_true, np.full(8, 0.3))

    assert metrics.average_precision == pytest.approx(positive_prevalence(y_true))


def test_precision_is_zero_not_nan_when_nothing_is_predicted_positive() -> None:
    """The majority baseline predicts no positives; 0/0 must not become NaN."""
    metrics = compute_metrics(np.array([0, 1, 0, 1]), np.zeros(4))

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_hand_computed_precision_recall_f1() -> None:
    # predicted positive: rows 0, 1, 2 -> TP=2 (rows 0,1), FP=1 (row 2)
    # actual positive:    rows 0, 1, 3 -> FN=1 (row 3)
    y_true = np.array([1, 1, 0, 1, 0])
    y_probability = np.array([0.9, 0.7, 0.6, 0.2, 0.1])

    metrics = compute_metrics(y_true, y_probability)

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(3 / 5)


def test_threshold_is_inclusive_at_the_boundary() -> None:
    """A probability exactly at 0.5 counts as a positive prediction."""
    metrics = compute_metrics(np.array([0, 1]), np.array([0.49, DEFAULT_THRESHOLD]))

    assert metrics.recall == 1.0
    assert metrics.precision == 1.0


def test_average_precision_is_sklearn_average_precision_score() -> None:
    """Named AP, and computed as AP — not a trapezoidal PR-AUC."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_probability = np.array([0.1, 0.9, 0.35, 0.8, 0.6, 0.2, 0.4, 0.7])

    metrics = compute_metrics(y_true, y_probability)

    assert metrics.average_precision == pytest.approx(
        average_precision_score(y_true, y_probability)
    )


def test_threshold_independent_metrics_ignore_the_threshold() -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_probability = np.array([0.1, 0.9, 0.35, 0.8, 0.6])

    at_half = compute_metrics(y_true, y_probability, threshold=0.5)
    at_tenth = compute_metrics(y_true, y_probability, threshold=0.1)

    for metric in THRESHOLD_INDEPENDENT:
        assert getattr(at_half, metric) == getattr(at_tenth, metric)
    assert at_half.recall != at_tenth.recall


def test_confusion_matrix_layout_is_tn_fp_fn_tp() -> None:
    y_true = np.array([0, 0, 1, 1, 1])
    y_probability = np.array([0.1, 0.9, 0.8, 0.7, 0.2])

    matrix = confusion_at_threshold(y_true, y_probability)

    assert matrix.tolist() == [[1, 1], [1, 2]]


def test_confusion_matrix_keeps_both_classes_when_one_is_never_predicted() -> None:
    matrix = confusion_at_threshold(np.array([0, 1]), np.zeros(2))

    assert matrix.shape == (2, 2)


def test_as_dict_returns_every_metric_in_order() -> None:
    metrics = compute_metrics(np.array([0, 1]), np.array([0.2, 0.8]))

    assert list(metrics.as_dict()) == list(METRIC_NAMES)


def test_prevalence_of_an_empty_vector_is_zero() -> None:
    assert positive_prevalence(np.array([])) == 0.0
