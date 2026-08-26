"""``ChurnInferenceService``: the primitive, called without a server."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from churn.serving.artifacts import FREEZE_COMMIT, FREEZE_COMMIT_SHORT, SERVING_VERSION
from churn.serving.errors import BatchTooLargeError
from churn.serving.inference import Prediction
from churn.serving.service import ChurnInferenceService
from conftest import PREDICTION_FIELDS, assert_predictions_agree, ulp_distance

THRESHOLD = 0.3272694566222328


def test_the_service_is_callable_without_http(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    prediction = service.predict_one(record)

    assert isinstance(prediction, Prediction)
    assert 0.0 <= prediction.churn_probability <= 1.0
    assert prediction.prediction in {0, 1}
    assert prediction.decision in {"retained", "churn"}
    assert prediction.threshold == THRESHOLD
    assert prediction.comparison == ">="
    assert prediction.calibration_policy == "NONE"


def test_the_decision_follows_the_frozen_rule_and_nothing_else(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    variants = [
        {**record, "tenure": tenure, "Contract": contract, "TotalCharges": str(tenure * 70.0)}
        for tenure in (0, 3, 24, 60)
        for contract in ("Month-to-month", "Two year")
    ]

    for prediction in service.predict_batch(variants):
        assert prediction.prediction == int(prediction.churn_probability >= THRESHOLD)
        assert prediction.decision == ("churn" if prediction.prediction == 1 else "retained")


def test_the_prediction_carries_no_clock_and_no_identifier(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """Determinism is a property of the fields, so the fields are enumerated."""
    fields = set(vars(service.predict_one(record)))

    assert fields == {
        "churn_probability",
        "prediction",
        "decision",
        "threshold",
        "comparison",
        "calibration_policy",
        "model_fingerprint",
    }


def test_the_same_record_always_gets_the_same_answer(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    assert service.predict_one(record) == service.predict_one(dict(record))


# --------------------------------------------------------------------------- #
# Batch.
# --------------------------------------------------------------------------- #


def test_a_batch_preserves_input_order(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    records = [
        {**record, "tenure": 1, "Contract": "Month-to-month"},
        {**record, "tenure": 70, "Contract": "Two year"},
        {**record, "tenure": 5, "Contract": "One year"},
    ]

    batched = service.predict_batch(records)
    one_by_one = [service.predict_one(item) for item in records]

    for batched_item, single in zip(batched, one_by_one, strict=True):
        assert_predictions_agree(batched_item, single)


def test_batch_and_single_are_exactly_equal_in_every_field(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """The endpoint-invariance contract, asserted with `==` and no tolerance.

    Seven synthetic records, each scored twice: once alone, once inside the batch.
    Both paths go through the same one-row primitive, so the floats are the same
    float — not close, the same.
    """
    records = [
        {
            **record,
            "tenure": tenure,
            "MonthlyCharges": 20.0 + tenure,
            "TotalCharges": str(tenure * 55.5),
        }
        for tenure in range(1, 8)
    ]

    batched = service.predict_batch(records)

    assert len(batched) == len(records)
    for index, item in enumerate(records):
        single = service.predict_one(item)
        assert single.churn_probability == batched[index].churn_probability
        assert single.prediction == batched[index].prediction
        assert single.decision == batched[index].decision
        assert single.threshold == batched[index].threshold
        assert single.comparison == batched[index].comparison
        assert_predictions_agree(single, batched[index])


def test_the_probability_is_bit_identical_at_every_batch_size(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """Batch composition is not an input to the model, so it must not move a float.

    This is the test that would have caught the original GEMV/GEMM divergence: the
    same record scored alone, in a pair, and in a crowd of a hundred. Before the
    service was made canonical, the batch-of-one and the batch-of-a-hundred
    disagreed in the last bit.
    """
    alone = service.predict_one(record)

    for size in (1, 2, 21, 100, 500):
        filler = [{**record, "tenure": 40} for _ in range(size - 1)]
        crowded = service.predict_batch([record, *filler])[0]

        assert crowded.churn_probability == alone.churn_probability, (
            f"batch of {size} moved the probability by "
            f"{ulp_distance(alone.churn_probability, crowded.churn_probability)} ULP"
        )
        assert crowded.prediction == alone.prediction


def test_position_inside_the_batch_does_not_move_a_probability(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """Not just the size of the batch — where the record sits in it."""
    other = {**record, "tenure": 68, "Contract": "Two year"}
    alone = service.predict_one(record)

    first = service.predict_batch([record, other, other])[0]
    middle = service.predict_batch([other, record, other])[1]
    last = service.predict_batch([other, other, record])[2]

    assert first.churn_probability == alone.churn_probability
    assert middle.churn_probability == alone.churn_probability
    assert last.churn_probability == alone.churn_probability


def test_a_repeated_record_is_scored_identically_at_both_positions(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """The A, B, A, C case: order preserved, and both A's identical."""
    a = {**record, "tenure": 3, "Contract": "Month-to-month"}
    b = {**record, "tenure": 66, "Contract": "Two year"}
    c = {**record, "tenure": 24, "Contract": "One year"}

    scored = service.predict_batch([a, b, a, c])

    assert len(scored) == 4
    assert_predictions_agree(scored[0], scored[2])
    assert_predictions_agree(scored[0], service.predict_one(a))
    assert_predictions_agree(scored[1], service.predict_one(b))
    assert_predictions_agree(scored[3], service.predict_one(c))
    # The four are not accidentally all the same record.
    assert len({item.churn_probability for item in scored}) == 3


def test_the_prediction_response_fields_are_exactly_the_declared_ones(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """The equivalence contract covers every field, so the field set is pinned."""
    assert set(vars(service.predict_one(record))) == set(PREDICTION_FIELDS)


def test_the_operational_limit_refuses_rather_than_truncates(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    bounded = ChurnInferenceService(artifacts=service.artifacts, max_batch_size=3)

    assert len(bounded.predict_batch([record] * 3)) == 3
    with pytest.raises(BatchTooLargeError) as error:
        bounded.predict_batch([record] * 4)

    assert "does not change any prediction" in str(error.value)


def test_the_limit_does_not_change_what_a_record_scores(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    bounded = ChurnInferenceService(artifacts=service.artifacts, max_batch_size=1)

    assert bounded.predict_one(record) == service.predict_one(record)


def test_an_empty_batch_is_refused(service: ChurnInferenceService) -> None:
    with pytest.raises(ValueError):
        service.predict_batch([])


# --------------------------------------------------------------------------- #
# Identity.
# --------------------------------------------------------------------------- #


def test_the_identity_names_the_frozen_model_not_this_release(
    service: ChurnInferenceService,
) -> None:
    identity = service.identity()

    assert identity.freeze_commit == FREEZE_COMMIT
    assert identity.freeze_commit_short == FREEZE_COMMIT_SHORT
    assert identity.serving_version == SERVING_VERSION
    assert identity.freeze_commit != identity.serving_version


def test_the_identity_describes_the_frozen_policy(service: ChurnInferenceService) -> None:
    identity = service.identity()

    assert identity.estimator == "LogisticRegression"
    assert identity.threshold == THRESHOLD
    assert identity.comparison == ">="
    assert identity.calibration_policy == "NONE"
    assert identity.threshold_policy == "F1_MAXIMIZATION"
    assert identity.positive_class_label == 1
    assert identity.positive_class_column == 1
    assert identity.positive_class_meaning == "Churn = Yes"
    assert identity.n_features == 19
    assert identity.n_transformed_features == 46
    assert len(identity.feature_names) == 19


def test_the_identity_exposes_nothing_sensitive(service: ChurnInferenceService) -> None:
    """No filesystem path, no secret, no dataset statistic."""
    rendered = repr(service.identity())

    assert "artifacts/model" not in rendered
    assert "C:\\" not in rendered and "/home/" not in rendered
    assert "customerID" not in rendered


def test_the_service_holds_no_per_request_state(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    before = service.predict_one(record)
    service.predict_batch([{**record, "tenure": t} for t in range(1, 30)])
    after = service.predict_one(record)

    assert before == after


def test_the_service_is_immutable(service: ChurnInferenceService) -> None:
    """One shared instance per process is only safe if nothing can mutate it."""
    with pytest.raises(FrozenInstanceError):
        service.max_batch_size = 1

    with pytest.raises(FrozenInstanceError):
        service.artifacts.threshold = 0.5
