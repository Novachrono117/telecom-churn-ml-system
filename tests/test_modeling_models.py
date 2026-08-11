"""The three baselines must be exactly what the protocol declares.

These are contract tests on configuration, not on behaviour: a silent change
from ``class_weight=None`` to ``"balanced"``, or a stray ``C``, would turn the
reference point into a different experiment while every downstream number kept
claiming to be a baseline.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import get_config
from churn.modeling.models import (
    CLASSIFIER_STEP,
    LOGISTIC_MAX_ITER,
    LOGISTIC_REGRESSION,
    LOGISTIC_SOLVER,
    MAJORITY,
    PREPROCESSOR_STEP,
    STRATIFIED_RANDOM,
    build_baselines,
)


def test_exactly_three_baselines_in_reporting_order() -> None:
    models = build_baselines()

    assert list(models) == [MAJORITY, STRATIFIED_RANDOM, LOGISTIC_REGRESSION]


def test_every_baseline_is_a_full_pipeline() -> None:
    for name, model in build_baselines().items():
        assert isinstance(model, Pipeline), name
        assert list(model.named_steps) == [PREPROCESSOR_STEP, CLASSIFIER_STEP], name


def test_majority_uses_most_frequent() -> None:
    classifier = build_baselines()[MAJORITY].named_steps[CLASSIFIER_STEP]

    assert isinstance(classifier, DummyClassifier)
    assert classifier.strategy == "most_frequent"


def test_stratified_uses_the_configured_seed() -> None:
    classifier = build_baselines()[STRATIFIED_RANDOM].named_steps[CLASSIFIER_STEP]

    assert isinstance(classifier, DummyClassifier)
    assert classifier.strategy == "stratified"
    assert classifier.random_state == get_config().seed


def test_logistic_regression_is_untuned() -> None:
    classifier = build_baselines()[LOGISTIC_REGRESSION].named_steps[CLASSIFIER_STEP]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.C == 1.0
    assert classifier.penalty == "l2"
    assert classifier.solver == LOGISTIC_SOLVER
    assert classifier.max_iter == LOGISTIC_MAX_ITER


def test_logistic_regression_does_not_reweight_classes() -> None:
    """`class_weight="balanced"` would make this a different experiment."""
    classifier = build_baselines()[LOGISTIC_REGRESSION].named_steps[CLASSIFIER_STEP]

    assert classifier.class_weight is None


def test_logistic_regression_carries_no_unused_random_state() -> None:
    """lbfgs is deterministic; a random_state would imply randomness that is absent."""
    classifier = build_baselines()[LOGISTIC_REGRESSION].named_steps[CLASSIFIER_STEP]

    assert classifier.random_state is None


def test_baselines_are_returned_unfitted() -> None:
    for name, model in build_baselines().items():
        assert not hasattr(model.named_steps[CLASSIFIER_STEP], "classes_"), name


def test_every_baseline_shares_the_same_preprocessing_shape() -> None:
    """The comparison is between models, not between pipeline shapes."""
    shapes = {
        name: [step for step, _ in model.named_steps[PREPROCESSOR_STEP].steps]
        for name, model in build_baselines().items()
    }

    assert len(set(map(tuple, shapes.values()))) == 1
