"""The predictive primitives, exercised without a server.

The decision rule is tested here directly rather than through a client. Rule 30
of the boundary specification asks for the ``>=`` behaviour at ``probability ==
threshold``; constructing a customer whose score lands exactly on a 16-digit
threshold would be an exercise in luck, and a test that depends on luck proves
nothing about the comparison. The primitive is called with the threshold itself.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from churn.modeling.freeze import apply_decision_rule
from churn.preprocessing.contracts import FEATURE_COLUMNS
from churn.preprocessing.exceptions import DataQualityError, FeatureContractError
from churn.serving.inference import (
    CHURN,
    RETAINED,
    build_feature_frame,
    canonical_probability,
    churn_probabilities,
    decide,
    decision_label,
    prepared_features,
)
from churn.serving.service import ChurnInferenceService

THRESHOLD = 0.3272694566222328


# --------------------------------------------------------------------------- #
# The decision rule: `>=`, not `>`.
# --------------------------------------------------------------------------- #


def test_a_probability_exactly_at_the_threshold_is_positive() -> None:
    """The boundary case the whole rule turns on."""
    assert decide(THRESHOLD, THRESHOLD) == 1


def test_the_comparison_is_not_strictly_greater() -> None:
    """If `>` had been implemented, the threshold itself would score 0."""
    assert decide(THRESHOLD, THRESHOLD) != 0
    assert decide(np.nextafter(THRESHOLD, 0.0), THRESHOLD) == 0
    assert decide(np.nextafter(THRESHOLD, 1.0), THRESHOLD) == 1


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, 0), (0.32, 0), (0.3272694566222327, 0), (0.33, 1), (1.0, 1)],
)
def test_the_rule_across_the_range(probability: float, expected: int) -> None:
    assert decide(probability, THRESHOLD) == expected


def test_the_serving_helper_is_the_frozen_comparison() -> None:
    """`decide` delegates; it does not restate `>=`."""
    probabilities = np.array([0.0, 0.2, THRESHOLD, 0.9, 1.0])

    scalar = [decide(float(value), THRESHOLD) for value in probabilities]

    assert scalar == list(apply_decision_rule(probabilities, THRESHOLD))


def test_the_serving_rule_is_the_frozen_rule_not_a_copy_of_it() -> None:
    """`decide` delegates. A second implementation of `>=` could drift from the freeze."""
    import churn.serving.inference as inference_module

    assert inference_module.apply_decision_rule is apply_decision_rule


@pytest.mark.parametrize(("label", "word"), [(0, RETAINED), (1, CHURN)])
def test_decision_labels(label: int, word: str) -> None:
    assert decision_label(label) == word


def test_a_non_binary_label_has_no_word() -> None:
    with pytest.raises(ValueError):
        decision_label(2)


# --------------------------------------------------------------------------- #
# prepare_features is the only door into the pipeline.
# --------------------------------------------------------------------------- #


def test_prepared_features_returns_the_canonical_column_order(record: dict[str, Any]) -> None:
    shuffled = dict(reversed(list(record.items())))

    features = prepared_features([shuffled])

    assert tuple(features.columns) == FEATURE_COLUMNS
    assert len(features) == 1


def test_the_input_order_of_a_payload_is_irrelevant(record: dict[str, Any]) -> None:
    """JSON has no order semantics, so the contract normalises rather than trusting."""
    forward = prepared_features([record])
    backward = prepared_features([dict(reversed(list(record.items())))])

    pd.testing.assert_frame_equal(forward, backward)


def test_build_feature_frame_preserves_record_order(record: dict[str, Any]) -> None:
    first = {**record, "tenure": 1}
    second = {**record, "tenure": 2}

    frame = build_feature_frame([first, second])

    assert list(frame["tenure"]) == [1, 2]
    assert list(frame.index) == [0, 1]


def test_an_empty_batch_has_no_feature_frame() -> None:
    with pytest.raises(ValueError):
        build_feature_frame([])


def test_the_identifier_never_reaches_the_matrix(record: dict[str, Any]) -> None:
    with pytest.raises(FeatureContractError):
        prepared_features([{**record, "customerID": "0000-ABCDE"}])


def test_the_target_never_reaches_the_matrix(record: dict[str, Any]) -> None:
    with pytest.raises(FeatureContractError):
        prepared_features([{**record, "Churn": "Yes"}])


def test_a_missing_feature_is_refused(record: dict[str, Any]) -> None:
    incomplete = {key: value for key, value in record.items() if key != "Contract"}

    with pytest.raises((FeatureContractError, DataQualityError)):
        prepared_features([incomplete])


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_category_is_refused_by_the_feature_contract(
    record: dict[str, Any], blank: object
) -> None:
    """The authority on "blank" is the frozen preprocessing, not the HTTP schema."""
    with pytest.raises(DataQualityError):
        prepared_features([{**record, "Contract": blank}])


def test_a_structural_zero_is_preserved(record: dict[str, Any]) -> None:
    """Blank TotalCharges at tenure == 0 is a customer who has not been billed yet."""
    features = prepared_features([{**record, "tenure": 0, "TotalCharges": ""}])

    assert features.loc[0, "TotalCharges"] == ""  # untouched here; the cleaner resolves it


def test_a_blank_total_charges_at_positive_tenure_is_refused(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """The rule lives in the pipeline's cleaner, so it fires when the model is applied."""
    with pytest.raises(DataQualityError):
        service.predict_one({**record, "tenure": 7, "TotalCharges": "   "})


def test_an_unreadable_total_charges_is_refused(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    with pytest.raises(DataQualityError):
        service.predict_one({**record, "TotalCharges": "not a number"})


# --------------------------------------------------------------------------- #
# Probabilities.
# --------------------------------------------------------------------------- #


def test_probabilities_are_read_from_the_column_classes_names(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    pipeline = service.artifacts.pipeline
    features = prepared_features([record])

    served = churn_probabilities(pipeline, features)
    raw = pipeline.predict_proba(features)
    column = int(np.flatnonzero(np.asarray(pipeline.classes_) == 1)[0])

    assert served[0] == raw[0, column]
    assert column == service.artifacts.positive_class_column


def test_probabilities_stay_inside_the_unit_interval(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    variants = [
        {**record, "tenure": tenure, "Contract": contract}
        for tenure in (0, 1, 24, 72)
        for contract in ("Month-to-month", "One year", "Two year")
    ]

    probabilities = churn_probabilities(service.artifacts.pipeline, prepared_features(variants))

    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


# --------------------------------------------------------------------------- #
# The canonical scoring primitive.
# --------------------------------------------------------------------------- #


def test_the_canonical_primitive_scores_exactly_one_row(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """Always a 1-row frame, whatever the caller is doing.

    This is what removes GEMV/GEMM as an observable difference: the classifier only
    ever sees one matrix shape, so there is only one summation order to round.
    """
    shapes: list[tuple[int, int]] = []
    pipeline = service.artifacts.pipeline
    original = type(pipeline).predict_proba

    def capture(self: object, X: Any) -> Any:  # noqa: ANN401, N803
        shapes.append(X.shape)
        return original(self, X)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(type(pipeline), "predict_proba", capture)
        service.predict_batch([record, {**record, "tenure": 50}, {**record, "tenure": 9}])

    assert shapes == [(1, 19), (1, 19), (1, 19)]


def test_the_canonical_primitive_returns_a_plain_float(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    probability = canonical_probability(service.artifacts.pipeline, record)

    assert type(probability) is float
    assert 0.0 <= probability <= 1.0


def test_the_canonical_primitive_is_what_the_service_calls(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """No second scoring path: the service's answer IS the primitive's output."""
    assert service.predict_one(record).churn_probability == canonical_probability(
        service.artifacts.pipeline, record
    )
