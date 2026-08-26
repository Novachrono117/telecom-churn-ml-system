"""The HTTP boundary, driven end to end against the real frozen model."""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from churn.serving.api import create_app, get_service
from churn.serving.artifacts import FREEZE_COMMIT, FREEZE_COMMIT_SHORT, SERVING_VERSION
from churn.serving.errors import (
    CODE_BATCH_TOO_LARGE,
    CODE_INTERNAL_ERROR,
    CODE_INVALID_FEATURE_VALUE,
    CODE_INVALID_REQUEST_SCHEMA,
    CODE_SERVICE_NOT_READY,
    ERROR_CODES,
)
from churn.serving.service import ChurnInferenceService
from conftest import assert_predictions_agree

PREDICT = "/api/v1/predict"
BATCH = "/api/v1/predict/batch"
MODEL = "/api/v1/model"
THRESHOLD = 0.3272694566222328


# --------------------------------------------------------------------------- #
# Health.
# --------------------------------------------------------------------------- #


def test_liveness_answers_without_touching_the_model(client: TestClient) -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert response.json()["model_fingerprint"] is None


def test_readiness_reports_the_verified_model(
    client: TestClient, service: ChurnInferenceService
) -> None:
    body = client.get("/health/ready").json()

    assert body["status"] == "ready"
    assert body["model_fingerprint"] == service.model_fingerprint
    assert body["startup_gates_passed"] == len(service.artifacts.checks)
    assert body["startup_gates_passed"] > 0


def test_readiness_is_not_ready_when_the_lifespan_never_ran(
    service: ChurnInferenceService,
) -> None:
    """Constructed but never started: no service, and the endpoint says so."""
    unstarted = TestClient(create_app(service=service))

    response = unstarted.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == CODE_SERVICE_NOT_READY


def test_prediction_is_refused_when_the_service_is_not_ready(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    unstarted = TestClient(create_app(service=service))

    response = unstarted.post(PREDICT, json=record)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == CODE_SERVICE_NOT_READY


# --------------------------------------------------------------------------- #
# Model metadata.
# --------------------------------------------------------------------------- #


def test_model_metadata_identifies_the_frozen_model(
    client: TestClient, service: ChurnInferenceService
) -> None:
    body = client.get(MODEL).json()

    assert body["model_fingerprint"] == service.model_fingerprint
    assert body["freeze_commit"] == FREEZE_COMMIT
    assert body["freeze_commit_short"] == FREEZE_COMMIT_SHORT
    assert body["serving_version"] == SERVING_VERSION
    assert body["threshold"] == THRESHOLD
    assert body["comparison"] == ">="
    assert body["calibration_policy"] == "NONE"
    assert body["threshold_policy"] == "F1_MAXIMIZATION"
    assert body["positive_class_label"] == 1
    assert body["n_features"] == 19
    assert body["n_transformed_features"] == 46


def test_model_metadata_distinguishes_model_version_from_serving_version(
    client: TestClient,
) -> None:
    body = client.get(MODEL).json()

    assert body["freeze_commit"] != body["serving_version"]


def test_model_metadata_leaks_no_path_and_no_secret(client: TestClient) -> None:
    rendered = client.get(MODEL).text

    assert "artifacts/model" not in rendered
    assert "C:\\" not in rendered
    assert "Traceback" not in rendered


# --------------------------------------------------------------------------- #
# Single prediction.
# --------------------------------------------------------------------------- #


def test_a_valid_record_is_scored(client: TestClient, record: dict[str, Any]) -> None:
    response = client.post(PREDICT, json=record)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["decision"] in ("retained", "churn")
    assert body["threshold"] == THRESHOLD
    assert body["comparison"] == ">="
    assert body["calibration_policy"] == "NONE"


def test_the_decision_is_the_threshold_rule(client: TestClient, record: dict[str, Any]) -> None:
    body = client.post(PREDICT, json=record).json()

    assert body["prediction"] == int(body["churn_probability"] >= body["threshold"])


def test_the_response_is_deterministic(client: TestClient, record: dict[str, Any]) -> None:
    """Same payload, same artefacts, byte-identical answer."""
    first = client.post(PREDICT, json=record)
    second = client.post(PREDICT, json=record)

    assert first.json() == second.json()
    assert first.content == second.content


def test_key_order_in_the_payload_is_irrelevant(client: TestClient, record: dict[str, Any]) -> None:
    """JSON has no order semantics; prepare_features normalises to FEATURE_COLUMNS."""
    reversed_record = dict(reversed(list(record.items())))

    assert client.post(PREDICT, json=record).json() == (
        client.post(PREDICT, json=reversed_record).json()
    )


def test_a_category_never_seen_in_training_still_gets_a_prediction(
    client: TestClient, record: dict[str, Any]
) -> None:
    """The encoder was frozen with handle_unknown='ignore' so this can happen.

    What this proves is that the boundary does not *block* a valid new category:
    the schema accepts it, the feature contract accepts it, and the frozen encoder
    absorbs it. It proves nothing about whether the resulting score is
    trustworthy — an unseen category is encoded as an all-zero block, which is a
    drift question and belongs to Phase 12, not here.
    """
    response = client.post(PREDICT, json={**record, "Contract": "Three year prepaid"})

    assert response.status_code == 200
    assert 0.0 <= response.json()["churn_probability"] <= 1.0


def test_an_unseen_payment_method_is_also_accepted(
    client: TestClient, record: dict[str, Any]
) -> None:
    response = client.post(PREDICT, json={**record, "PaymentMethod": "Instant transfer (PIX)"})

    assert response.status_code == 200


# --------------------------------------------------------------------------- #
# Rejected payloads.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("field", ["customerID", "Churn"])
def test_the_identifier_and_the_target_are_rejected_not_ignored(
    client: TestClient, record: dict[str, Any], field: str
) -> None:
    response = client.post(PREDICT, json={**record, field: "whatever"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == CODE_INVALID_REQUEST_SCHEMA
    assert any(field in detail for detail in response.json()["error"]["details"])


def test_an_engineered_feature_is_rejected(client: TestClient, record: dict[str, Any]) -> None:
    response = client.post(PREDICT, json={**record, "tenure_bucket": "0-12"})

    assert response.status_code == 422


def test_a_missing_feature_is_rejected(client: TestClient, record: dict[str, Any]) -> None:
    del record["PaymentMethod"]

    response = client.post(PREDICT, json=record)

    assert response.status_code == 422


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_category_is_rejected(
    client: TestClient, record: dict[str, Any], blank: str
) -> None:
    response = client.post(PREDICT, json={**record, "InternetService": blank})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == CODE_INVALID_FEATURE_VALUE


def test_a_null_category_is_rejected(client: TestClient, record: dict[str, Any]) -> None:
    response = client.post(PREDICT, json={**record, "InternetService": None})

    assert response.status_code == 422


def test_a_structural_zero_total_charges_is_accepted(
    client: TestClient, record: dict[str, Any]
) -> None:
    """Blank TotalCharges at tenure == 0: a customer who has not been billed yet."""
    response = client.post(PREDICT, json={**record, "tenure": 0, "TotalCharges": ""})

    assert response.status_code == 200


@pytest.mark.parametrize("value", ["", "   ", None])
def test_a_blank_total_charges_at_positive_tenure_is_rejected(
    client: TestClient, record: dict[str, Any], value: object
) -> None:
    response = client.post(PREDICT, json={**record, "tenure": 9, "TotalCharges": value})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == CODE_INVALID_FEATURE_VALUE


def test_an_unreadable_total_charges_is_rejected(
    client: TestClient, record: dict[str, Any]
) -> None:
    response = client.post(PREDICT, json={**record, "TotalCharges": "n/a"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == CODE_INVALID_FEATURE_VALUE


def test_a_senior_citizen_string_is_rejected_rather_than_silently_ignored(
    client: TestClient, record: dict[str, Any]
) -> None:
    """The encoder learned integers; "0" would be an unknown category, scored as absent."""
    response = client.post(PREDICT, json={**record, "SeniorCitizen": "zero"})

    assert response.status_code == 422


def test_no_range_constraint_was_invented_for_the_numeric_features(
    client: TestClient, record: dict[str, Any]
) -> None:
    """A value beyond the training range is out of distribution, not invalid input."""
    response = client.post(
        PREDICT, json={**record, "MonthlyCharges": 999.0, "tenure": 400, "TotalCharges": 399600.0}
    )

    assert response.status_code == 200


def test_an_error_never_carries_a_traceback_or_a_path(
    client: TestClient, record: dict[str, Any]
) -> None:
    text = client.post(PREDICT, json={**record, "TotalCharges": "n/a"}).text

    assert "Traceback" not in text
    assert "site-packages" not in text
    assert "churn_pipeline.joblib" not in text


def test_every_error_code_is_from_the_declared_set(
    client: TestClient, record: dict[str, Any]
) -> None:
    responses = [
        client.post(PREDICT, json={**record, "customerID": "x"}),
        client.post(PREDICT, json={**record, "Contract": ""}),
        client.post(BATCH, json={"records": []}),
    ]

    for response in responses:
        assert response.json()["error"]["code"] in ERROR_CODES


def test_an_unexpected_failure_answers_500_without_internals(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    class Exploding(ChurnInferenceService):
        def _score(self, record: Any) -> Any:  # noqa: ANN401
            raise RuntimeError("secret detail /etc/passwd")

    app = create_app(service=service)
    app.dependency_overrides[get_service] = lambda: Exploding(artifacts=service.artifacts)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(PREDICT, json=record)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == CODE_INTERNAL_ERROR
    assert "secret detail" not in response.text
    assert "Traceback" not in response.text


# --------------------------------------------------------------------------- #
# Batch.
# --------------------------------------------------------------------------- #


def test_a_batch_is_scored_in_input_order(client: TestClient, record: dict[str, Any]) -> None:
    records = [
        {**record, "tenure": 1, "Contract": "Month-to-month"},
        {**record, "tenure": 71, "Contract": "Two year"},
        {**record, "tenure": 12, "Contract": "One year"},
    ]

    body = client.post(BATCH, json={"records": records}).json()

    assert body["count"] == 3
    singles = [client.post(PREDICT, json=item).json() for item in records]
    assert body["predictions"] == singles


def test_batch_and_single_return_identical_json(client: TestClient, record: dict[str, Any]) -> None:
    """Endpoint invariance over HTTP, asserted on the decoded JSON with `==`.

    The envelopes differ — batch wraps its results in `{count, predictions}` — so
    the comparison is between the corresponding PredictionResponse objects, which
    must be equal in every field, probability included.
    """
    records = [{**record, "tenure": tenure} for tenure in (0, 2, 30, 65)]
    records[0]["TotalCharges"] = ""

    batched = client.post(BATCH, json={"records": records}).json()["predictions"]

    for index, item in enumerate(records):
        single = client.post(PREDICT, json=item).json()
        assert batched[index] == single
        assert_predictions_agree(batched[index], single)


def test_the_same_record_through_both_endpoints_is_numerically_identical(
    client: TestClient, record: dict[str, Any]
) -> None:
    """One record, two routes, one float."""
    single = client.post(PREDICT, json=record).json()
    inside_batch = client.post(BATCH, json={"records": [record]}).json()["predictions"][0]
    inside_crowd = client.post(BATCH, json={"records": [record] * 50}).json()["predictions"][0]

    assert single["churn_probability"] == inside_batch["churn_probability"]
    assert single["churn_probability"] == inside_crowd["churn_probability"]
    assert single == inside_batch == inside_crowd


def test_a_batch_with_a_repeated_record_preserves_order_and_repeats_the_answer(
    client: TestClient, record: dict[str, Any]
) -> None:
    """A, B, A, C over HTTP: positions preserved, both A's byte-identical."""
    a = {**record, "tenure": 3, "Contract": "Month-to-month"}
    b = {**record, "tenure": 66, "Contract": "Two year"}
    c = {**record, "tenure": 24, "Contract": "One year"}

    body = client.post(BATCH, json={"records": [a, b, a, c]}).json()
    predictions = body["predictions"]

    assert body["count"] == 4
    assert predictions[0] == predictions[2]
    assert predictions[0] == client.post(PREDICT, json=a).json()
    assert predictions[1] == client.post(PREDICT, json=b).json()
    assert predictions[3] == client.post(PREDICT, json=c).json()
    assert len({item["churn_probability"] for item in predictions}) == 3


def test_reordering_the_batch_reorders_the_answers(
    client: TestClient, record: dict[str, Any]
) -> None:
    first = {**record, "tenure": 1, "Contract": "Month-to-month"}
    second = {**record, "tenure": 70, "Contract": "Two year"}

    forward = client.post(BATCH, json={"records": [first, second]}).json()["predictions"]
    backward = client.post(BATCH, json={"records": [second, first]}).json()["predictions"]

    assert forward == list(reversed(backward))


def test_an_empty_batch_is_rejected(client: TestClient) -> None:
    response = client.post(BATCH, json={"records": []})

    assert response.status_code == 422


def test_a_batch_beyond_the_limit_is_refused(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    bounded = ChurnInferenceService(artifacts=service.artifacts, max_batch_size=2)

    with TestClient(create_app(service=bounded)) as client:
        assert client.post(BATCH, json={"records": [record] * 2}).status_code == 200
        response = client.post(BATCH, json={"records": [record] * 3})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == CODE_BATCH_TOO_LARGE


def test_one_invalid_record_fails_the_whole_batch(
    client: TestClient, record: dict[str, Any]
) -> None:
    """No partial answer: a caller must not have to guess which rows were scored."""
    response = client.post(BATCH, json={"records": [record, {**record, "Contract": ""}]})

    assert response.status_code == 422


def test_the_batch_response_carries_no_clock(client: TestClient, record: dict[str, Any]) -> None:
    body = client.post(BATCH, json={"records": [record]}).json()

    assert set(body) == {"count", "predictions"}
    assert set(body["predictions"][0]) == {
        "churn_probability",
        "prediction",
        "decision",
        "threshold",
        "comparison",
        "calibration_policy",
        "model_fingerprint",
    }


# --------------------------------------------------------------------------- #
# Dependency injection.
# --------------------------------------------------------------------------- #


def test_the_http_layer_accepts_an_injected_service(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    """Unit-testable transport: the layer does not build its own model."""
    app = create_app(service=service)
    app.dependency_overrides[get_service] = lambda: ChurnInferenceService(
        artifacts=service.artifacts, max_batch_size=1
    )

    with TestClient(app) as client:
        assert client.post(PREDICT, json=record).status_code == 200
        assert client.post(BATCH, json={"records": [record] * 2}).status_code == 413
