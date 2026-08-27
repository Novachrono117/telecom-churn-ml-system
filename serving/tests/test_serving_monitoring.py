"""Monitoring inside the serving process: it observes, and it changes nothing.

The claim this file exists to establish is narrow and absolute: **turning monitoring
on does not alter a single predictive field.** Everything else here — the endpoint,
the counters, the health reporting — is secondary to that, because a monitoring
layer that could move a probability would be a defect in the prediction boundary
rather than a feature of the observability one.

The Phase 11 guarantees are re-run with monitoring enabled, not assumed to survive:
exact single/batch equivalence, batch order, the 34 startup gates, and fail-closed
artefact verification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from churn.monitoring.settings import (
    EXPECTED_REFERENCE_PROFILE_SHA256,
    MONITORING_ENDPOINT,
    REFERENCE_TRUST_ANCHOR,
    STATUS_INSUFFICIENT_DATA,
    default_reference_profile_path,
    get_monitoring_policy,
)
from churn.serving.api import create_app
from churn.serving.errors import ArtifactIntegrityError
from churn.serving.monitoring import (
    HEALTH_DEGRADED,
    HEALTH_DISABLED,
    HEALTH_HEALTHY,
    OBSERVER_FAILURE_EVENT,
    MonitoringStartupError,
    ServingMonitor,
)
from churn.serving.settings import ServingSettings
from conftest import assert_predictions_agree

PREDICT = "/api/v1/predict"
BATCH = "/api/v1/predict/batch"
MINIMUM = get_monitoring_policy().minimum_window_size

#: A value no reference population could contain, used to prove that an exception
#: message derived from a request never reaches a log.
SECRET_CUSTOMER_VALUE = "SECRET_CUSTOMER_VALUE_ABC123"


def _walk(node: Any, path: str = "") -> list[tuple[str, Any]]:
    """Return every ``(path, leaf)`` in a nested structure, keys included as leaves."""
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append((f"{path}.{key}", key))
            found.extend(_walk(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk(value, f"{path}[{index}]"))
    else:
        found.append((path, node))
    return found


@pytest.fixture(scope="session")
def monitoring_settings(policy_path: Path) -> ServingSettings:
    return ServingSettings(policy_path=policy_path, monitoring_enabled=True)


@pytest.fixture
def monitored(monitoring_settings: ServingSettings) -> Iterator[TestClient]:
    """A client with monitoring on, over the real frozen model and real reference."""
    with TestClient(create_app(settings=monitoring_settings)) as client:
        yield client


# --------------------------------------------------------------------------- #
# The central claim.
# --------------------------------------------------------------------------- #


def test_the_prediction_is_identical_with_monitoring_on_and_off(
    settings: ServingSettings, monitoring_settings: ServingSettings, record: dict[str, Any]
) -> None:
    """Same payload, same artefacts, two processes. Byte-identical answers."""
    with TestClient(create_app(settings=settings)) as off:
        without = off.post(PREDICT, json=record)
    with TestClient(create_app(settings=monitoring_settings)) as on:
        With = on.post(PREDICT, json=record)

    assert without.json() == With.json()
    assert without.content == With.content
    assert_predictions_agree(without.json(), With.json())


def test_a_batch_is_identical_with_monitoring_on_and_off(
    settings: ServingSettings, monitoring_settings: ServingSettings, record: dict[str, Any]
) -> None:
    records = [{**record, "tenure": tenure} for tenure in (0, 5, 40, 70)]
    records[0]["TotalCharges"] = ""
    payload = {"records": records}

    with TestClient(create_app(settings=settings)) as off:
        without = off.post(BATCH, json=payload).json()
    with TestClient(create_app(settings=monitoring_settings)) as on:
        With = on.post(BATCH, json=payload).json()

    assert without == With


def test_single_and_batch_stay_exactly_equal_with_monitoring_on(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """The Phase 11 contract, re-established under the new configuration."""
    records = [{**record, "tenure": tenure} for tenure in (1, 12, 36, 60)]

    batched = monitored.post(BATCH, json={"records": records}).json()["predictions"]

    for index, item in enumerate(records):
        single = monitored.post(PREDICT, json=item).json()
        assert batched[index] == single
        assert_predictions_agree(batched[index], single)


def test_batch_order_is_preserved_with_monitoring_on(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    a = {**record, "tenure": 3, "Contract": "Month-to-month"}
    b = {**record, "tenure": 66, "Contract": "Two year"}
    c = {**record, "tenure": 24, "Contract": "One year"}

    predictions = monitored.post(BATCH, json={"records": [a, b, a, c]}).json()["predictions"]

    assert predictions[0] == predictions[2]
    assert predictions[0] == monitored.post(PREDICT, json=a).json()
    assert predictions[1] == monitored.post(PREDICT, json=b).json()
    assert len({item["churn_probability"] for item in predictions}) == 3


def test_the_thirty_four_startup_gates_still_pass(monitored: TestClient) -> None:
    """Monitoring must not weaken the frozen model's verification."""
    body = monitored.get("/health/ready").json()

    assert body["status"] == "ready"
    assert body["startup_gates_passed"] == 34


# --------------------------------------------------------------------------- #
# The endpoint.
# --------------------------------------------------------------------------- #


def test_the_endpoint_is_absent_when_monitoring_is_off(client: TestClient) -> None:
    """Opt-in: deploying Phase 12 code does not alter the published Phase 11 surface."""
    assert client.get(MONITORING_ENDPOINT).status_code == 404
    assert MONITORING_ENDPOINT not in client.get("/openapi.json").json()["paths"]


def test_the_phase_11_route_list_is_unchanged_by_default(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])

    assert paths == {
        "/health/live",
        "/health/ready",
        "/api/v1/model",
        "/api/v1/predict",
        "/api/v1/predict/batch",
    }


def test_the_endpoint_reports_aggregates_when_monitoring_is_on(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    monitored.post(PREDICT, json=record)
    monitored.post(BATCH, json={"records": [record] * 4})

    body = monitored.get(MONITORING_ENDPOINT).json()

    assert body["monitoring_enabled"] is True
    assert body["n_records"] == 5
    assert body["data_quality"]["requests_total"] == 2
    assert body["data_quality"]["records_total"] == 5
    assert body["data_quality"]["successful_records"] == 5
    assert body["reference_population"] == "training_pool"
    assert body["reference_n"] == 5634
    assert len(body["reference_profile_sha256"]) == 64


def test_a_small_window_reports_counts_without_a_verdict(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    monitored.post(PREDICT, json=record)

    body = monitored.get(MONITORING_ENDPOINT).json()

    assert body["status"] == STATUS_INSUFFICIENT_DATA
    assert body["window_has_verdict"] is False
    assert body["details_suppressed"] is True
    assert body["n_records"] == 1
    assert body["minimum_window_size"] == 100
    # The counters stay: a suppressed window must not look like a dead process.
    assert body["data_quality"]["requests_total"] == 1
    assert body["collector_health"] == HEALTH_HEALTHY


def test_the_endpoint_never_claims_a_performance_verdict(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    monitored.post(PREDICT, json=record)

    body = monitored.get(MONITORING_ENDPOINT).json()

    assert body["performance_degradation"]["evaluated"] is False
    assert "labels" in body["performance_degradation"]["reason"]
    assert "NOT evidence that the model degraded" in body["interpretation"]
    assert "calibration_policy is NONE" in body["calibration"]


def test_there_is_no_http_reset_endpoint(monitored: TestClient) -> None:
    """An unauthenticated endpoint that erases drift evidence is a bad trade."""
    paths = monitored.get("/openapi.json").json()["paths"]

    assert MONITORING_ENDPOINT in paths
    assert set(paths[MONITORING_ENDPOINT]) == {"get"}
    assert f"{MONITORING_ENDPOINT}/reset" not in paths
    assert monitored.post(f"{MONITORING_ENDPOINT}/reset").status_code in (404, 405)


def test_rejected_records_are_counted_by_class(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    monitored.post(PREDICT, json={**record, "customerID": "x"})  # schema
    monitored.post(PREDICT, json={**record, "Contract": ""})  # feature contract

    quality = monitored.get(MONITORING_ENDPOINT).json()["data_quality"]

    assert quality["invalid_schema_records"] == 1
    assert quality["invalid_feature_records"] == 1
    assert quality["successful_records"] == 0


def test_an_unseen_category_is_observable_through_the_endpoint(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """The API accepts it, the encoder ignores it, monitoring makes it visible.

    Over a window large enough to publish distributions. Below the minimum the
    per-feature entry does not exist at all, and the global
    ``unseen_category_records`` counter is what an operator has — see the privacy
    tests, where that is the point rather than a limitation.
    """
    unseen = {**record, "PaymentMethod": "Instant transfer (PIX)"}
    response = monitored.post(BATCH, json={"records": [unseen] * MINIMUM})

    body = monitored.get(MONITORING_ENDPOINT).json()
    entry = body["feature_drift"]["categorical"]["PaymentMethod"]

    assert response.status_code == 200
    assert entry["unseen_count"] == MINIMUM
    assert entry["n_distinct_unseen_observed"] == 1
    assert body["data_quality"]["unseen_category_records"] == MINIMUM


def test_a_structural_violation_is_observable_and_still_scored(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """Watched, not enforced: Phase 12 does not retroactively reject Phase 11 input."""
    broken = {**record, "InternetService": "No", "OnlineSecurity": "No"}

    response = monitored.post(BATCH, json={"records": [broken] * MINIMUM})
    body = monitored.get(MONITORING_ENDPOINT).json()

    assert response.status_code == 200
    assert body["structural_consistency"]["records_with_any_violation"] == MINIMUM
    assert body["data_quality"]["structural_violation_records"] == MINIMUM


# --------------------------------------------------------------------------- #
# Privacy.
# --------------------------------------------------------------------------- #


def test_the_endpoint_leaks_no_payload(monitored: TestClient, record: dict[str, Any]) -> None:
    """A one-record window must not hand back that record's feature values.

    Its mean, min and max would each BE the value — and so would a histogram bin
    holding a single count. Below the operational minimum the distributions are gone
    entirely, so a low-traffic deployment cannot be polled into revealing individual
    customers one request at a time.
    """
    monitored.post(PREDICT, json={**record, "MonthlyCharges": 123.45})

    body = monitored.get(MONITORING_ENDPOINT).json()

    assert "customerID" not in monitored.get(MONITORING_ENDPOINT).text
    assert "123.45" not in monitored.get(MONITORING_ENDPOINT).text
    assert body["feature_drift"] is None
    assert body["details_suppressed"] is True
    assert body["small_window_note"] is not None


def test_a_sufficient_window_does_report_its_statistics(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """The contrast: an average over a hundred records is an aggregate."""
    monitored.post(BATCH, json={"records": [record] * 120})

    body = monitored.get(MONITORING_ENDPOINT).json()
    entry = body["feature_drift"]["numeric"]["MonthlyCharges"]

    assert body["window_has_verdict"] is True
    assert body["small_window_note"] is None
    assert entry["mean"] is not None
    assert entry["mean_shift_in_reference_sd"] is not None


def test_the_endpoint_leaks_no_unseen_category_text(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """Neither the text nor its digest, at any window size."""
    secret = "SuperSecretPlan"
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    monitored.post(BATCH, json={"records": [{**record, "Contract": secret}] * MINIMUM})

    rendered = monitored.get(MONITORING_ENDPOINT).text

    assert secret not in rendered
    assert digest not in rendered
    entry = monitored.get(MONITORING_ENDPOINT).json()["feature_drift"]["categorical"]["Contract"]
    assert entry["n_distinct_unseen_observed"] == 1


def test_no_row_level_probability_is_reported(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """Not in a large window, and not even as a one-record summary in a small one."""
    single = monitored.post(PREDICT, json=record).json()
    small = monitored.get(MONITORING_ENDPOINT).json()

    # With one record the mean WOULD BE that score and the positive rate WOULD BE
    # that decision, so the whole section is withheld.
    assert small["prediction_drift"] is None
    assert str(single["churn_probability"]) not in monitored.get(MONITORING_ENDPOINT).text

    monitored.post(BATCH, json={"records": [record] * MINIMUM})
    prediction_drift = monitored.get(MONITORING_ENDPOINT).json()["prediction_drift"]

    assert "probabilities" not in prediction_drift
    assert "scores" not in prediction_drift
    assert set(prediction_drift) >= {"mean", "std", "bin_counts", "predicted_positive_rate"}


# --------------------------------------------------------------------------- #
# Health, and the readiness boundary.
# --------------------------------------------------------------------------- #


def test_readiness_reports_collector_health(monitored: TestClient, client: TestClient) -> None:
    assert monitored.get("/health/ready").json()["monitoring"] == HEALTH_HEALTHY
    assert client.get("/health/ready").json()["monitoring"] == HEALTH_DISABLED


def test_drift_never_makes_the_service_not_ready(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """A shifted population is not a corrupt artefact.

    A hundred and fifty records far from the reference: the monitoring status goes
    CRITICAL, and readiness stays 200. Conflating the two would pull a perfectly
    functioning instance out of rotation because customers changed.
    """
    drifted = {
        **record,
        "tenure": 71,
        "Contract": "Two year",
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "MonthlyCharges": 20.0,
        "TotalCharges": "1420.00",
        "PaymentMethod": "Bank transfer (automatic)",
        "PaperlessBilling": "No",
    }
    monitored.post(BATCH, json={"records": [drifted] * 150})

    monitoring = monitored.get(MONITORING_ENDPOINT).json()
    readiness = monitored.get("/health/ready")

    assert monitoring["status"] == "CRITICAL"
    assert monitoring["prediction_drift"]["psi"]["status"] == "CRITICAL"
    assert readiness.status_code == 200
    assert readiness.json()["status"] == "ready"
    assert readiness.json()["monitoring"] == HEALTH_HEALTHY


def test_an_artifact_failure_still_makes_the_service_not_ready(
    service: Any, record: dict[str, Any]
) -> None:
    """The other half: readiness is about the artefact, and still is."""
    unstarted = TestClient(create_app(service=service))

    assert unstarted.get("/health/ready").status_code == 503
    assert unstarted.post(PREDICT, json=record).status_code == 503


# --------------------------------------------------------------------------- #
# Failure isolation.
# --------------------------------------------------------------------------- #


def test_a_failing_collector_does_not_change_a_prediction(
    settings: ServingSettings,
    monitoring_settings: ServingSettings,
    record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prediction path is authoritative; a broken histogram costs an observation."""
    from churn.monitoring.collector import MonitoringCollector

    with TestClient(create_app(settings=settings)) as off:
        expected = off.post(PREDICT, json=record).json()

    def explode(*_: object, **__: object) -> None:
        raise RuntimeError("collector is broken")

    app = create_app(settings=monitoring_settings)
    with TestClient(app) as client:
        monkeypatch.setattr(MonitoringCollector, "observe", explode)
        response = client.post(PREDICT, json=record)
        health = client.get("/health/ready").json()
        monitoring = client.get(MONITORING_ENDPOINT).json()

    assert response.status_code == 200
    assert response.json() == expected
    assert health["status"] == "ready"
    assert health["monitoring"] == HEALTH_DEGRADED
    assert monitoring["collector_health"] == HEALTH_DEGRADED
    assert monitoring["collector_failures"] >= 1


def test_a_collector_failure_is_never_silent(
    monitoring_settings: ServingSettings,
    record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from churn.monitoring.collector import MonitoringCollector

    def explode(*_: object, **__: object) -> None:
        raise RuntimeError("collector is broken")

    app = create_app(settings=monitoring_settings)
    with caplog.at_level("ERROR", logger="churn"), TestClient(app) as client:
        monkeypatch.setattr(MonitoringCollector, "observe", explode)
        client.post(PREDICT, json=record)

    logged = "\n".join(entry.getMessage() for entry in caplog.records)
    assert OBSERVER_FAILURE_EVENT in logged
    assert "exception_type=RuntimeError" in logged


# --------------------------------------------------------------------------- #
# Fail-closed startup.
# --------------------------------------------------------------------------- #


def test_an_absent_reference_stops_monitoring_startup(policy_path: Path, tmp_path: Path) -> None:
    """A drift number against a baseline that does not exist looks like a signal."""
    settings = ServingSettings(
        policy_path=policy_path,
        monitoring_enabled=True,
        reference_profile_path=tmp_path / "missing.json",
    )

    with pytest.raises(MonitoringStartupError) as error:
        ServingMonitor.from_settings(settings)

    assert "does not start" in str(error.value)
    assert not (tmp_path / "missing.json").exists()


def test_a_tampered_reference_stops_monitoring_startup(policy_path: Path, tmp_path: Path) -> None:
    """A reference that claims the holdout, or the target, is refused."""
    payload = json.loads(default_reference_profile_path().read_text(encoding="utf-8"))
    payload["provenance"]["holdout_used"] = True
    tampered = tmp_path / "reference_profile.json"
    tampered.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    settings = ServingSettings(
        policy_path=policy_path,
        monitoring_enabled=True,
        reference_profile_path=tampered,
    )

    with pytest.raises(MonitoringStartupError) as error:
        ServingMonitor.from_settings(settings)

    assert "holdout_not_used" in str(error.value)


def test_a_file_that_is_not_a_reference_is_refused(policy_path: Path, tmp_path: Path) -> None:
    impostor = tmp_path / "reference_profile.json"
    impostor.write_text('{"schema_version": 1}', encoding="utf-8")

    settings = ServingSettings(
        policy_path=policy_path,
        monitoring_enabled=True,
        reference_profile_path=impostor,
    )

    with pytest.raises(MonitoringStartupError) as error:
        ServingMonitor.from_settings(settings)

    assert "does not parse" in str(error.value)


def test_a_monitoring_startup_failure_is_a_serving_startup_failure() -> None:
    """It shares the fail-closed hierarchy, so nothing treats it as recoverable."""
    from churn.serving.errors import ServingStartupError

    assert issubclass(MonitoringStartupError, ServingStartupError)
    assert not issubclass(MonitoringStartupError, ArtifactIntegrityError)


# --------------------------------------------------------------------------- #
# Small-window privacy, over the real HTTP boundary.
# --------------------------------------------------------------------------- #


def test_a_one_record_window_leaks_nothing_through_the_endpoint(monitored: TestClient) -> None:
    """One synthetic record with unmistakable values, then a sweep of the response.

    The assertion is not "field X is null". It is that no value the record carried
    appears anywhere in the JSON — as a value, as a key, or as a count sitting beside
    a level name — because ``{"Two year": 1}`` identifies the customer exactly as
    well as a field called ``contract`` would.
    """
    marked = {
        "tenure": 3,
        "MonthlyCharges": 97.31,
        "TotalCharges": "291.93",
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
    }
    prediction = monitored.post(PREDICT, json=marked)
    body = monitored.get(MONITORING_ENDPOINT).json()
    rendered = monitored.get(MONITORING_ENDPOINT).text

    assert prediction.status_code == 200
    assert body["n_records"] == 1
    assert body["status"] == STATUS_INSUFFICIENT_DATA
    assert body["details_suppressed"] is True

    # No categorical value of the record, at any depth, as a key or as a value.
    categorical = {"Two year", "Mailed check", "Fiber optic"}
    offenders = [
        (path, leaf)
        for path, leaf in _walk(body, "response")
        if isinstance(leaf, str) and leaf in categorical
    ]
    assert not offenders, offenders

    # No numeric value of the record either, in any rendering.
    for value in ("97.31", "291.93", str(prediction.json()["churn_probability"])):
        assert value not in rendered

    # And no structure of the {"level": 1} or [0, 0, 1, 0] shape to infer it from.
    assert body["feature_drift"] is None
    assert body["prediction_drift"] is None
    assert body["structural_consistency"] is None


def test_a_window_of_ninety_nine_is_suppressed_exactly_like_a_window_of_one(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """The minimum is a threshold, not a gradient: 99 gets nothing 1 does not."""
    monitored.post(BATCH, json={"records": [record] * (MINIMUM - 1)})

    body = monitored.get(MONITORING_ENDPOINT).json()

    assert body["n_records"] == MINIMUM - 1
    assert body["status"] == STATUS_INSUFFICIENT_DATA
    assert body["details_suppressed"] is True
    assert body["feature_drift"] is None
    assert body["prediction_drift"] is None
    assert body["structural_consistency"] is None
    assert body["data_quality"]["successful_records"] == MINIMUM - 1


def test_at_exactly_the_minimum_the_distributions_appear(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """And the record that crosses the boundary is the one that unlocks them."""
    monitored.post(BATCH, json={"records": [record] * (MINIMUM - 1)})
    assert monitored.get(MONITORING_ENDPOINT).json()["details_suppressed"] is True

    monitored.post(PREDICT, json=record)
    body = monitored.get(MONITORING_ENDPOINT).json()

    assert body["n_records"] == MINIMUM
    assert body["details_suppressed"] is False
    assert body["window_has_verdict"] is True
    assert body["small_window_note"] is None
    assert body["feature_drift"]["numeric"]["tenure"]["mean"] is not None
    assert sum(body["feature_drift"]["numeric"]["tenure"]["bin_counts"]) == MINIMUM
    assert body["prediction_drift"]["predicted_positive_rate"] is not None
    assert body["structural_consistency"]["violations_by_rule"] is not None


def test_the_counters_below_the_minimum_are_exactly_the_operational_ones(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """What a suppressed window still tells an operator, enumerated."""
    monitored.post(PREDICT, json=record)

    body = monitored.get(MONITORING_ENDPOINT).json()
    present = {key for key, value in body.items() if value is not None}

    assert present == {
        "monitoring_enabled",
        "status",
        "collector_health",
        "collector_failures",
        "n_records",
        "minimum_window_size",
        "window_has_verdict",
        "details_suppressed",
        "reference_profile_sha256",
        "expected_reference_profile_sha256",
        "reference_profile_sha256_source",
        "reference_population",
        "reference_n",
        "section_status",
        "data_quality",
        "performance_degradation",
        "interpretation",
        "calibration",
        "rejection_counting_note",
        "small_window_note",
    }


# --------------------------------------------------------------------------- #
# Sanitised failure logging.
# --------------------------------------------------------------------------- #


def test_a_collector_failure_never_logs_anything_derived_from_the_payload(
    settings: ServingSettings,
    monitoring_settings: ServingSettings,
    record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The observer raises with a customer value in its message. Nothing logs it.

    An exception raised inside the collector was raised while holding feature values,
    so ``str(error)`` is text the request controls. A traceback renders that message,
    which is why neither it nor ``logger.exception`` may be used here: the log is the
    one surface nobody audits for payload.
    """
    from churn.monitoring.collector import MonitoringCollector

    with TestClient(create_app(settings=settings)) as off:
        expected = off.post(PREDICT, json=record).json()

    def explode(*_: object, **__: object) -> None:
        raise RuntimeError(SECRET_CUSTOMER_VALUE)

    app = create_app(settings=monitoring_settings)
    with caplog.at_level("DEBUG"), TestClient(app) as client:
        monkeypatch.setattr(MonitoringCollector, "observe", explode)
        response = client.post(PREDICT, json=record)
        readiness = client.get("/health/ready").json()
        monitoring = client.get(MONITORING_ENDPOINT).json()

    logged = "\n".join(f"{entry.getMessage()}\n{entry.exc_text or ''}" for entry in caplog.records)

    # The value never appears — not in a message, not in a traceback.
    assert SECRET_CUSTOMER_VALUE not in logged
    assert not any(entry.exc_info for entry in caplog.records)
    # What IS logged is an event, a type and a count.
    assert OBSERVER_FAILURE_EVENT in logged
    assert "exception_type=RuntimeError" in logged
    # And the prediction is untouched, byte for byte, still 200.
    assert response.status_code == 200
    assert response.json() == expected
    assert readiness["status"] == "ready"
    assert readiness["monitoring"] == HEALTH_DEGRADED
    assert monitoring["collector_health"] == HEALTH_DEGRADED
    assert monitoring["collector_failures"] >= 1


def test_a_failure_while_counting_a_rejection_is_sanitised_too(
    monitoring_settings: ServingSettings,
    record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The rejection path holds a payload the schema just refused. Same rule."""
    from churn.monitoring.collector import MonitoringCollector

    def explode(*_: object, **__: object) -> None:
        raise ValueError(SECRET_CUSTOMER_VALUE)

    app = create_app(settings=monitoring_settings)
    with caplog.at_level("DEBUG"), TestClient(app) as client:
        monkeypatch.setattr(MonitoringCollector, "record_request", explode)
        response = client.post(PREDICT, json={**record, "customerID": "abc"})

    logged = "\n".join(f"{entry.getMessage()}\n{entry.exc_text or ''}" for entry in caplog.records)

    assert response.status_code == 422
    assert SECRET_CUSTOMER_VALUE not in logged
    assert "exception_type=ValueError" in logged


def test_the_failure_tally_keeps_a_type_and_never_a_message(
    monitoring_settings: ServingSettings,
    record: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In memory as well as in the log: the monitor holds a class name."""
    from churn.monitoring.collector import MonitoringCollector

    def explode(*_: object, **__: object) -> None:
        raise RuntimeError(SECRET_CUSTOMER_VALUE)

    app = create_app(settings=monitoring_settings)
    with TestClient(app) as client:
        monkeypatch.setattr(MonitoringCollector, "observe", explode)
        client.post(PREDICT, json=record)
        monitor = client.app.state.monitor

    assert monitor.failures.last_error == "RuntimeError"
    assert SECRET_CUSTOMER_VALUE not in monitor.failures.last_error


# --------------------------------------------------------------------------- #
# The reference trust anchor.
# --------------------------------------------------------------------------- #


def test_the_pinned_digest_is_the_committed_reference_profile() -> None:
    """The anchor and the artefact move together, or startup would fail on main."""
    payload = default_reference_profile_path().read_bytes()

    assert hashlib.sha256(payload).hexdigest() == EXPECTED_REFERENCE_PROFILE_SHA256


def test_the_endpoint_names_what_the_digest_was_checked_against(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """Reporting a digest without naming its expectation proves nothing."""
    monitored.post(PREDICT, json=record)

    body = monitored.get(MONITORING_ENDPOINT).json()

    assert body["reference_profile_sha256"] == EXPECTED_REFERENCE_PROFILE_SHA256
    assert body["expected_reference_profile_sha256"] == EXPECTED_REFERENCE_PROFILE_SHA256
    assert body["reference_profile_sha256_source"] == REFERENCE_TRUST_ANCHOR


def test_a_reference_whose_digest_is_not_the_pinned_one_stops_startup(
    policy_path: Path, tmp_path: Path
) -> None:
    """Tampering that breaks no invariant is still refused, because of the pin.

    A note is edited: every structural check still passes, the population is still
    the training pool, the histograms still total. Only the digest moves. Without the
    pin this profile would be accepted and monitoring would report drift against a
    baseline nobody approved.
    """
    payload = json.loads(default_reference_profile_path().read_text(encoding="utf-8"))
    payload["notes"] = [*payload["notes"], "an edit that breaks no invariant"]
    tampered = tmp_path / "reference_profile.json"
    tampered.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    settings = ServingSettings(
        policy_path=policy_path,
        monitoring_enabled=True,
        reference_profile_path=tampered,
    )

    with pytest.raises(MonitoringStartupError) as error:
        ServingMonitor.from_settings(settings)

    assert "profile_digest_matches" in str(error.value)


def test_a_tampered_reference_never_lets_monitoring_reach_healthy(
    policy_path: Path, tmp_path: Path
) -> None:
    """Fail-closed at the process level, not merely at the constructor."""
    payload = json.loads(default_reference_profile_path().read_text(encoding="utf-8"))
    payload["notes"] = ["replaced"]
    tampered = tmp_path / "reference_profile.json"
    tampered.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    settings = ServingSettings(
        policy_path=policy_path,
        monitoring_enabled=True,
        reference_profile_path=tampered,
    )

    with pytest.raises(MonitoringStartupError), TestClient(create_app(settings=settings)):
        pass  # pragma: no cover - the context manager never opens


# --------------------------------------------------------------------------- #
# Windows, through the serving process.
# --------------------------------------------------------------------------- #


def test_closing_a_window_in_the_process_starts_an_empty_one(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """Window A of 100 reports a verdict; window B of 25 is suppressed again."""
    monitored.post(BATCH, json={"records": [record] * MINIMUM})
    window_a = monitored.get(MONITORING_ENDPOINT).json()

    closed = monitored.app.state.monitor.service.close_window()
    empty = monitored.get(MONITORING_ENDPOINT).json()

    monitored.post(BATCH, json={"records": [record] * 25})
    window_b = monitored.get(MONITORING_ENDPOINT).json()

    assert window_a["n_records"] == MINIMUM
    assert window_a["details_suppressed"] is False
    assert closed.n_records == MINIMUM
    assert empty["n_records"] == 0
    assert window_b["n_records"] == 25
    assert window_b["status"] == STATUS_INSUFFICIENT_DATA
    assert window_b["details_suppressed"] is True


def test_closing_a_window_changes_no_prediction(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """A counter operation cannot move a score. Checked, not assumed."""
    before = monitored.post(PREDICT, json=record).json()

    monitored.app.state.monitor.service.close_window()
    after = monitored.post(PREDICT, json=record).json()

    assert after == before
    assert monitored.get("/api/v1/model").json() == monitored.get("/api/v1/model").json()


# --------------------------------------------------------------------------- #
# Bounded unseen-cardinality tracking, over the serving boundary.
# --------------------------------------------------------------------------- #


def test_the_endpoint_qualifies_a_saturated_distinct_count(
    monitoring_settings: ServingSettings, record: dict[str, Any]
) -> None:
    """A number that has become a lower bound must not be served as a cardinality.

    The app is built with a deliberately tiny cap so saturation is reachable in a
    test; production reads the configured 1024. What is being checked is the
    reporting contract, which does not depend on the value.
    """
    from churn.monitoring.collector import MonitoringCollector

    app = create_app(settings=monitoring_settings)
    with TestClient(app) as client:
        monitor = client.app.state.monitor
        # Rebuild the collector with a small cap, then re-run the window through the
        # real HTTP path so nothing is fabricated in the assertions below.
        object.__setattr__(
            monitor.service,
            "collector",
            MonitoringCollector(monitor.service.reference, 8),
        )
        client.post(
            BATCH,
            json={
                "records": [
                    {**record, "PaymentMethod": f"scheme_{index:04d}"} for index in range(MINIMUM)
                ]
            },
        )
        body = client.get(MONITORING_ENDPOINT).json()

    entry = body["feature_drift"]["categorical"]["PaymentMethod"]

    assert entry["n_distinct_unseen_observed"] == 8
    assert entry["distinct_unseen_tracking_cap"] == 8
    assert entry["distinct_unseen_tracking_saturated"] is True
    assert entry["distinct_unseen_is_lower_bound"] is True
    # The occurrence side stays exact, and it is what carries the status.
    assert entry["unseen_count"] == MINIMUM
    assert entry["unseen_rate"]["value"] == 1.0
    assert entry["unseen_rate"]["status"] == "CRITICAL"
    assert body["data_quality"]["unseen_category_records"] == MINIMUM
    assert "lower bound" in body["unseen_cardinality"]


def test_an_unsaturated_window_reports_an_exact_distinct_count(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """The contrast: under the configured cap the number means what it says."""
    schemes = ["PIX", "Crypto", "Voucher"]
    monitored.post(
        BATCH,
        json={
            "records": [
                {**record, "PaymentMethod": schemes[index % len(schemes)]}
                for index in range(MINIMUM)
            ]
        },
    )

    entry = monitored.get(MONITORING_ENDPOINT).json()["feature_drift"]["categorical"][
        "PaymentMethod"
    ]

    assert entry["n_distinct_unseen_observed"] == len(schemes)
    assert entry["distinct_unseen_tracking_saturated"] is False
    assert entry["distinct_unseen_is_lower_bound"] is False
    assert entry["distinct_unseen_tracking_cap"] == 1024


def test_a_cardinality_attack_leaks_no_value_and_no_digest(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """Bounding the set changes nothing about what may leave it."""
    values = [f"SECRET_SCHEME_{index:04d}" for index in range(MINIMUM)]
    monitored.post(
        BATCH,
        json={"records": [{**record, "PaymentMethod": value} for value in values]},
    )

    rendered = monitored.get(MONITORING_ENDPOINT).text

    for value in values[:20]:
        assert value not in rendered
        assert hashlib.sha256(value.encode("utf-8")).hexdigest() not in rendered


def test_a_small_window_reveals_nothing_about_unseen_cardinality(
    monitored: TestClient, record: dict[str, Any]
) -> None:
    """The suppression policy is not reopened by the new diagnostic fields."""
    monitored.post(PREDICT, json={**record, "PaymentMethod": "Instant transfer (PIX)"})

    body = monitored.get(MONITORING_ENDPOINT).json()
    keys = {leaf for path, leaf in _walk(body, "response") if path.endswith(f".{leaf}")}

    assert body["details_suppressed"] is True
    assert body["feature_drift"] is None
    for field in (
        "n_distinct_unseen_observed",
        "distinct_unseen_tracking_saturated",
        "distinct_unseen_is_lower_bound",
        "distinct_unseen_tracking_cap",
        "unseen_rate",
        "unseen_count",
    ):
        assert field not in keys, field
    # The note explaining the diagnostic is absent too: a suppressed window says
    # nothing about unseen cardinality, not even what the number would have meant.
    assert body.get("unseen_cardinality") is None
    # The global counter is what a small window gives an operator, and it stays.
    assert body["data_quality"]["unseen_category_records"] == 1
