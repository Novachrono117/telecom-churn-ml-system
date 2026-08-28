"""The explanation endpoint, over the real boundary.

The claim this file exists to establish is narrow and absolute: **``/api/v1/explain``
returns the same prediction ``/api/v1/predict`` returns.** Not a prediction that
usually matches, and not one that matches to six decimals — the same float, because
both come out of the same canonical scoring call.

Everything else here is secondary to that, because an explanation of a number the API
never emitted would be a defect in the prediction boundary dressed up as a feature of
the observability one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from churn.serving.api import create_app
from churn.serving.settings import ServingSettings

PREDICT = "/api/v1/predict"
BATCH = "/api/v1/predict/batch"
EXPLAIN = "/api/v1/explain"
PORTFOLIO = "/api/v1/portfolio"
MONITORING = "/api/v1/monitoring"


@pytest.fixture(scope="session")
def portfolio_settings(policy_path: Path) -> ServingSettings:
    return ServingSettings(policy_path=policy_path, portfolio_ui_enabled=True)


@pytest.fixture
def demo(portfolio_settings: ServingSettings) -> Iterator[TestClient]:
    """A client with the demo on, over the real frozen model and real metadata."""
    with TestClient(create_app(settings=portfolio_settings)) as client:
        yield client


def _variants(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "baseline": record,
        "no_internet": {
            **record,
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "Contract": "Two year",
        },
        "no_phone": {**record, "PhoneService": "No", "MultipleLines": "No phone service"},
        "structural_zero": {**record, "tenure": 0, "TotalCharges": ""},
        "unseen_category": {**record, "PaymentMethod": "Instant transfer (PIX)"},
        "senior": {**record, "SeniorCitizen": 1, "tenure": 68, "Contract": "One year"},
    }


# --------------------------------------------------------------------------- #
# The central claim.
# --------------------------------------------------------------------------- #


def test_predict_and_explain_agree_exactly_on_every_shared_field(
    demo: TestClient, record: dict[str, Any]
) -> None:
    """Same payload, same process, identical predictive fields — for every variant."""
    shared = (
        "churn_probability",
        "prediction",
        "decision",
        "threshold",
        "comparison",
        "calibration_policy",
    )

    for name, payload in _variants(record).items():
        predicted = demo.post(PREDICT, json=payload)
        explained = demo.post(EXPLAIN, json=payload)

        assert predicted.status_code == 200, name
        assert explained.status_code == 200, name
        for field in shared:
            assert predicted.json()[field] == explained.json()[field], f"{name}.{field}"
        # Bit-for-bit on the probability, not merely close.
        assert repr(predicted.json()["churn_probability"]) == repr(
            explained.json()["churn_probability"]
        ), name


def test_the_explanation_goes_through_the_canonical_scoring_path(
    demo: TestClient, record: dict[str, Any]
) -> None:
    """One scoring implementation in the process, so the two cannot diverge.

    Breaking the canonical primitive must break both endpoints. If explain survived
    it, explain would be scoring somewhere else.
    """
    from churn.serving import service as service_module

    baseline = demo.post(EXPLAIN, json=record).json()["churn_probability"]
    calls: list[str] = []
    original = service_module.canonical_features_and_probability

    def counted(*args: Any, **kwargs: Any):
        calls.append("scored")
        return original(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service_module, "canonical_features_and_probability", counted)
        predicted = demo.post(PREDICT, json=record).json()
        assert len(calls) == 1
        explained = demo.post(EXPLAIN, json=record).json()
        assert len(calls) == 2

    # Both endpoints reached the same primitive, exactly once each, and returned the
    # same float. There is no second scoring implementation for them to diverge on.
    assert predicted["churn_probability"] == explained["churn_probability"] == baseline


def test_the_batch_endpoint_still_matches_explain(demo: TestClient, record: dict[str, Any]) -> None:
    """Single, batch and explain are three views of one number."""
    records = [{**record, "tenure": tenure} for tenure in (1, 12, 36, 60)]

    batched = demo.post(BATCH, json={"records": records}).json()["predictions"]

    for index, payload in enumerate(records):
        explained = demo.post(EXPLAIN, json=payload).json()
        assert batched[index]["churn_probability"] == explained["churn_probability"]
        assert batched[index]["prediction"] == explained["prediction"]


def test_the_prediction_is_unchanged_by_the_demo_being_enabled(
    settings: ServingSettings, portfolio_settings: ServingSettings, record: dict[str, Any]
) -> None:
    """Turning the product on must not move a predictive field."""
    with TestClient(create_app(settings=settings)) as off:
        without = off.post(PREDICT, json=record)
    with TestClient(create_app(settings=portfolio_settings)) as on:
        with_ui = on.post(PREDICT, json=record)

    assert without.json() == with_ui.json()
    assert without.content == with_ui.content


# --------------------------------------------------------------------------- #
# The decomposition, over HTTP.
# --------------------------------------------------------------------------- #


def test_the_response_reconstructs_the_logit(demo: TestClient, record: dict[str, Any]) -> None:
    body = demo.post(EXPLAIN, json=record).json()

    total = sum(item["contribution_log_odds"] for item in body["contributions"])

    assert body["intercept"] + total == pytest.approx(body["model_logit"], abs=1e-12)
    assert body["reconstruction"]["within_tolerance"] is True
    assert body["explanation_method"] == "EXACT_LOGISTIC_DECOMPOSITION"
    assert len(body["contributions"]) == 19


def test_the_response_carries_all_nineteen_features_and_no_more(
    demo: TestClient, record: dict[str, Any]
) -> None:
    body = demo.post(EXPLAIN, json=record).json()

    features = {item["feature"] for item in body["contributions"]}

    assert len(features) == 19
    assert "customerID" not in features
    assert "Churn" not in features


def test_a_coupled_block_is_returned_as_one_row_summing_exactly(
    demo: TestClient, record: dict[str, Any]
) -> None:
    payload = _variants(record)["no_internet"]

    body = demo.post(EXPLAIN, json=payload).json()
    blocks = body["grouped_contributions"]
    by_feature = {item["feature"]: item for item in body["contributions"]}

    assert [block["name"] for block in blocks] == ["no_internet_service"]
    block = blocks[0]
    assert block["n_members"] == 7
    assert block["contribution_log_odds"] == sum(
        by_feature[name]["contribution_log_odds"] for name in block["members"]
    )
    # Grouped plus ungrouped still reconstructs the logit.
    total = block["contribution_log_odds"] + sum(
        by_feature[name]["contribution_log_odds"] for name in body["ungrouped_features"]
    )
    assert body["intercept"] + total == pytest.approx(body["model_logit"], abs=1e-12)


def test_an_inconsistent_record_is_explained_ungrouped(
    demo: TestClient, record: dict[str, Any]
) -> None:
    """Accepted, scored, counted by monitoring — and not presented as one block."""
    broken = {**record, "InternetService": "No", "OnlineSecurity": "No"}

    response = demo.post(EXPLAIN, json=broken)

    assert response.status_code == 200
    assert response.json()["grouped_contributions"] == []
    assert len(response.json()["ungrouped_features"]) == 19


def test_an_unseen_category_is_scored_and_flagged(demo: TestClient, record: dict[str, Any]) -> None:
    """The API contract is unchanged: a new category is still accepted and scored."""
    response = demo.post(EXPLAIN, json=_variants(record)["unseen_category"])
    body = response.json()

    entry = next(item for item in body["contributions"] if item["feature"] == "PaymentMethod")

    assert response.status_code == 200
    assert entry["is_unseen_level"] is True
    assert entry["contribution_log_odds"] == 0.0
    assert entry["active_level"] is None


def test_the_response_carries_its_caveats(demo: TestClient, record: dict[str, Any]) -> None:
    body = demo.post(EXPLAIN, json=record).json()

    assert "not causal effects" in body["causal_note"]
    assert "no post-hoc calibration" in body["calibration_note"]
    assert body["calibration_policy"] == "NONE"


# --------------------------------------------------------------------------- #
# The contract at the edge.
# --------------------------------------------------------------------------- #


def test_a_structural_zero_is_accepted(demo: TestClient, record: dict[str, Any]) -> None:
    response = demo.post(EXPLAIN, json={**record, "tenure": 0, "TotalCharges": ""})

    assert response.status_code == 200
    entry = next(
        item for item in response.json()["contributions"] if item["feature"] == "TotalCharges"
    )
    assert entry["value_display"] == "(blank)"


def test_a_blank_total_at_positive_tenure_is_rejected(
    demo: TestClient, record: dict[str, Any]
) -> None:
    """The frozen rule, unchanged and not reimplemented anywhere else."""
    response = demo.post(EXPLAIN, json={**record, "tenure": 5, "TotalCharges": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_FEATURE_VALUE"


@pytest.mark.parametrize("extra", ["customerID", "Churn", "tenure_bucket"])
def test_extra_fields_are_rejected(demo: TestClient, record: dict[str, Any], extra: str) -> None:
    response = demo.post(EXPLAIN, json={**record, extra: "x"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST_SCHEMA"


def test_a_missing_feature_is_rejected(demo: TestClient, record: dict[str, Any]) -> None:
    payload = dict(record)
    payload.pop("Contract")

    response = demo.post(EXPLAIN, json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST_SCHEMA"


def test_a_blank_category_is_rejected(demo: TestClient, record: dict[str, Any]) -> None:
    response = demo.post(EXPLAIN, json={**record, "Contract": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] in {
        "INVALID_FEATURE_VALUE",
        "INVALID_FEATURE_PAYLOAD",
    }


def test_no_error_response_carries_a_traceback_or_a_path(
    demo: TestClient, record: dict[str, Any]
) -> None:
    response = demo.post(EXPLAIN, json={**record, "tenure": 5, "TotalCharges": ""})

    rendered = response.text

    assert "Traceback" not in rendered
    assert "site-packages" not in rendered
    assert "\\\\" not in rendered
    assert "src/churn" not in rendered


# --------------------------------------------------------------------------- #
# Metadata.
# --------------------------------------------------------------------------- #


def test_the_metadata_endpoint_serves_versioned_numbers(demo: TestClient) -> None:
    body = demo.get(PORTFOLIO).json()

    assert body["provenance"]["holdout_reopened"] is False
    assert body["provenance"]["metrics_recomputed"] is False
    assert [entry["key"] for entry in body["evaluation"]["headline"]][:2] == [
        "average_precision",
        "roc_auc",
    ]
    assert "accuracy" not in {entry["key"] for entry in body["evaluation"]["headline"]}
    assert "optimistic bias" in body["evaluation"]["caveat"]
    assert body["policy"]["calibration_policy"] == "NONE"


def test_the_metadata_offers_the_training_contract_levels(demo: TestClient) -> None:
    """A UI convenience. The API still accepts anything non-blank — proved above."""
    levels = demo.get(PORTFOLIO).json()["known_levels"]

    assert len(levels) == 16
    assert levels["Contract"] == ["Month-to-month", "One year", "Two year"]
    assert "No internet service" in levels["OnlineSecurity"]
    assert "customerID" not in levels


def test_the_metadata_exposes_no_path_and_no_row(demo: TestClient) -> None:
    rendered = demo.get(PORTFOLIO).text

    assert "customerID" not in rendered
    for needle in ("data/raw", ".csv", "site-packages", "C:\\\\"):
        assert needle not in rendered


def test_the_metadata_endpoint_is_read_only(demo: TestClient) -> None:
    paths = demo.get("/openapi.json").json()["paths"]

    assert set(paths[PORTFOLIO]) == {"get"}
    assert set(paths[EXPLAIN]) == {"post"}


# --------------------------------------------------------------------------- #
# Fail-closed startup.
# --------------------------------------------------------------------------- #


def test_absent_metadata_stops_the_demo(policy_path: Path, tmp_path: Path) -> None:
    """Metrics invented at serving time would not be the frozen evaluation."""
    from churn.serving.portfolio import PortfolioStartupError

    settings = ServingSettings(
        policy_path=policy_path,
        portfolio_ui_enabled=True,
        portfolio_metadata_path=tmp_path / "missing.json",
    )

    with pytest.raises(PortfolioStartupError) as error, TestClient(create_app(settings=settings)):
        pass  # pragma: no cover - the context manager never opens

    assert "does not start" in str(error.value)


def test_absent_assets_stop_the_demo(policy_path: Path, tmp_path: Path) -> None:
    """A page that renders half of itself is worse than one that refuses."""
    from churn.serving.portfolio import PortfolioStartupError

    settings = ServingSettings(
        policy_path=policy_path,
        portfolio_ui_enabled=True,
        portfolio_static_path=tmp_path,
    )

    with pytest.raises(PortfolioStartupError) as error, TestClient(create_app(settings=settings)):
        pass  # pragma: no cover

    assert "incomplete" in str(error.value)


def test_a_portfolio_startup_failure_is_a_serving_startup_failure() -> None:
    from churn.serving.errors import ServingStartupError
    from churn.serving.portfolio import PortfolioStartupError

    assert issubclass(PortfolioStartupError, ServingStartupError)


def test_the_new_error_code_is_outside_the_phase_11_set() -> None:
    """Extending the recorded set would change a published contract.

    ``EXPLANATION_UNAVAILABLE`` belongs to a route that exists only when the demo is
    enabled, so on a Phase 11 deployment it cannot be emitted and the recorded set is
    still exactly true. That is why it lives in its own set rather than in the frozen
    one whose contents ``serving_results.json`` verifies byte for byte.
    """
    from churn.serving.errors import (
        ALL_ERROR_CODES,
        CODE_EXPLANATION_UNAVAILABLE,
        ERROR_CODES,
        PORTFOLIO_ERROR_CODES,
    )

    assert CODE_EXPLANATION_UNAVAILABLE not in ERROR_CODES
    assert CODE_EXPLANATION_UNAVAILABLE in PORTFOLIO_ERROR_CODES
    assert ALL_ERROR_CODES == ERROR_CODES | PORTFOLIO_ERROR_CODES
    assert len(ERROR_CODES) == 6


def test_every_error_the_demo_can_emit_is_a_declared_code(
    demo: TestClient, record: dict[str, Any]
) -> None:
    from churn.serving.errors import ALL_ERROR_CODES

    failures = (
        {**record, "customerID": "x"},
        {**record, "Contract": ""},
        {**record, "tenure": 5, "TotalCharges": ""},
    )
    for payload in failures:
        response = demo.post(EXPLAIN, json=payload)
        assert response.json()["error"]["code"] in ALL_ERROR_CODES


# --------------------------------------------------------------------------- #
# Monitoring, alongside.
# --------------------------------------------------------------------------- #


def test_an_explained_record_is_observed_like_a_predicted_one(
    policy_path: Path, record: dict[str, Any]
) -> None:
    """The explain endpoint scores a real record, so monitoring must see it.

    Otherwise the operational counters would disagree with each other: a request
    counted, a record never observed.
    """
    settings = ServingSettings(
        policy_path=policy_path, portfolio_ui_enabled=True, monitoring_enabled=True
    )
    with TestClient(create_app(settings=settings)) as client:
        client.post(EXPLAIN, json=record)
        body = client.get("/api/v1/monitoring").json()

    assert body["n_records"] == 1
    assert body["data_quality"]["requests_total"] == 1
    assert body["data_quality"]["records_total"] == 1
    assert body["data_quality"]["successful_records"] == 1


# --------------------------------------------------------------------------- #
# Metadata integrity: the demo serves the pinned summary or it does not start.
#
# Every number the page presents as measured comes from one file, and a setting names
# which file. Parsing it and validating its shape answers "is this shaped like the
# summary"; it cannot answer "is this the summary". A file with an edited average
# precision satisfies every structural invariant. So the digest is compared against a
# constant in source, and a mismatch is a startup failure — not a fallback to empty
# metadata, and not the numbers shown unverified.
# --------------------------------------------------------------------------- #


def _write_metadata(destination: Path, mutate=None) -> Path:
    """Copy the committed summary to ``destination``, optionally changing one field.

    The real artefact is never opened for writing: it is read, decoded, mutated in
    memory and written somewhere pytest owns.
    """
    from churn.portfolio.metadata import default_metadata_path

    payload = json.loads(default_metadata_path().read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(payload)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def _inflate_average_precision(payload: dict[str, Any]) -> None:
    for entry in payload["evaluation"]["headline"]:
        if entry["key"] == "average_precision":
            entry["value"] = 0.97
            entry["ci_lower"] = 0.96
            entry["ci_upper"] = 0.98


def _inflate_roc_auc(payload: dict[str, Any]) -> None:
    for entry in payload["evaluation"]["headline"]:
        if entry["key"] == "roc_auc":
            entry["value"] = 0.99


def _soften_the_caveat(payload: dict[str, Any]) -> None:
    """Keep the phrase the structural invariant looks for; drop what it admits."""
    payload["evaluation"]["caveat"] = "The estimate may carry optimistic bias."


TAMPERINGS = {
    "average_precision": _inflate_average_precision,
    "roc_auc": _inflate_roc_auc,
    "analyst_caveat": _soften_the_caveat,
}


def test_intact_metadata_starts_the_demo(policy_path: Path, tmp_path: Path) -> None:
    """A byte-identical copy at another path is still the pinned summary."""
    settings = ServingSettings(
        policy_path=policy_path,
        portfolio_ui_enabled=True,
        portfolio_metadata_path=_write_metadata(tmp_path / "portfolio_metadata.json"),
    )
    with TestClient(create_app(settings=settings)) as client:
        response = client.get(PORTFOLIO)

    assert response.status_code == 200
    integrity = response.json()["integrity"]
    assert integrity["verified"] is True
    assert integrity["actual_sha256"] == integrity["expected_sha256"]


def test_the_metadata_response_names_where_its_expectation_comes_from(
    demo: TestClient,
) -> None:
    """An operator reading the response should not have to guess what was compared."""
    from churn.portfolio.metadata import (
        EXPECTED_PORTFOLIO_METADATA_SHA256,
        PORTFOLIO_METADATA_TRUST_ANCHOR,
    )

    integrity = demo.get(PORTFOLIO).json()["integrity"]

    assert integrity["expected_sha256"] == EXPECTED_PORTFOLIO_METADATA_SHA256
    assert integrity["source_of_expected_sha256"] == PORTFOLIO_METADATA_TRUST_ANCHOR
    assert len(integrity["actual_sha256"]) == 64


@pytest.mark.parametrize("name", sorted(TAMPERINGS))
def test_tampered_metadata_stops_the_demo(name: str, policy_path: Path, tmp_path: Path) -> None:
    """Structurally valid, semantically intact, presentationally false — refused."""
    from churn.portfolio.metadata import load_portfolio_metadata, verify_portfolio_metadata
    from churn.serving.portfolio import PortfolioStartupError

    path = _write_metadata(tmp_path / "portfolio_metadata.json", TAMPERINGS[name])

    # The premise: nothing but the digest objects to this file.
    assert all(holds for _, holds, _ in verify_portfolio_metadata(load_portfolio_metadata(path)))

    settings = ServingSettings(
        policy_path=policy_path,
        portfolio_ui_enabled=True,
        portfolio_metadata_path=path,
    )
    with pytest.raises(PortfolioStartupError) as error, TestClient(create_app(settings=settings)):
        pass  # pragma: no cover - the context manager never opens

    message = str(error.value)
    assert "metadata_digest_matches" in message
    assert "expected_sha256" in message
    assert "actual_sha256" in message
    assert "source_of_expected_sha256" in message


def test_a_tampered_summary_never_reaches_a_response(policy_path: Path, tmp_path: Path) -> None:
    """No silent degradation: the routes do not exist, so nothing unverified is shown."""
    from churn.serving.portfolio import PortfolioStartupError

    settings = ServingSettings(
        policy_path=policy_path,
        portfolio_ui_enabled=True,
        portfolio_metadata_path=_write_metadata(
            tmp_path / "portfolio_metadata.json", _inflate_average_precision
        ),
    )
    with pytest.raises(PortfolioStartupError):
        with TestClient(create_app(settings=settings)) as client:
            client.get(PORTFOLIO)  # pragma: no cover - startup never completes


def test_the_real_artefact_was_not_touched_by_the_tampering_tests() -> None:
    """These tests build files in tmp_path; the committed summary is read-only here."""
    from churn.portfolio.metadata import (
        EXPECTED_PORTFOLIO_METADATA_SHA256,
        default_metadata_path,
        load_portfolio_metadata,
        metadata_digest,
    )

    committed = load_portfolio_metadata(default_metadata_path())

    assert metadata_digest(committed) == EXPECTED_PORTFOLIO_METADATA_SHA256


def test_metadata_is_not_read_when_the_demo_is_off(policy_path: Path, tmp_path: Path) -> None:
    """With the UI off there is nothing to display, so there is nothing to verify."""
    settings = ServingSettings(
        policy_path=policy_path,
        portfolio_metadata_path=tmp_path / "does-not-exist.json",
    )
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/health/ready").status_code == 200
        assert client.get(PORTFOLIO).status_code == 404


# --------------------------------------------------------------------------- #
# Untrusted input: a category is a string, and a string is data.
#
# The page offers a "custom / unseen value" field, so a categorical can carry
# arbitrary text. It comes back in value_display and active_level. The backend's job
# is to treat it as an ordinary category and return it unchanged — not to strip it,
# not to escape it, and certainly not to interpret it.
# --------------------------------------------------------------------------- #

MARKUP_VALUES = (
    '<img src=x onerror="alert(1)">',
    "<script>alert(1)</script>",
    "</script><svg onload=alert(1)>",
    '"><svg/onload=confirm(document.domain)>',
    "javascript:alert(1)",
)


@pytest.mark.parametrize("value", MARKUP_VALUES)
def test_a_markup_shaped_unseen_value_is_returned_as_data(
    demo: TestClient, record: dict[str, Any], value: str
) -> None:
    """It scores like any unseen category, and comes back as the string it is."""
    response = demo.post(EXPLAIN, json={**record, "PaymentMethod": value})

    assert response.status_code == 200
    body = response.json()
    payment = next(item for item in body["contributions"] if item["feature"] == "PaymentMethod")

    assert payment["is_unseen_level"] is True
    # value_display carries the string the visitor typed, verbatim: not stripped, not
    # escaped, not rewritten. Escaping belongs to the medium that renders it, and the
    # medium here is JSON followed by textContent.
    assert payment["value_display"] == value
    assert payment["active_level"] is None
    # An unseen level contributes nothing: the frozen encoder ignores it.
    assert payment["contribution_log_odds"] == 0.0


@pytest.mark.parametrize("value", MARKUP_VALUES)
def test_a_markup_shaped_value_is_json_encoded_never_interpolated(
    demo: TestClient, record: dict[str, Any], value: str
) -> None:
    """The response is JSON, and the raw bytes carry no unescaped tag.

    The API answers ``application/json``; a browser does not parse that as markup, and
    the page inserts it with textContent. This asserts the first half — the response
    is data — while the frontend audit asserts the second.
    """
    response = demo.post(EXPLAIN, json={**record, "Contract": value})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert json.loads(response.text)["contributions"] is not None
    # Round-trips exactly: nothing was stripped, nothing was rewritten.
    contract = next(
        item for item in response.json()["contributions"] if item["feature"] == "Contract"
    )
    assert contract["value_display"] == value


@pytest.mark.parametrize("value", MARKUP_VALUES)
def test_a_markup_shaped_value_predicts_the_same_as_it_explains(
    demo: TestClient, record: dict[str, Any], value: str
) -> None:
    payload = {**record, "PaymentMethod": value}
    predicted = demo.post(PREDICT, json=payload).json()
    explained = demo.post(EXPLAIN, json=payload).json()

    assert predicted["churn_probability"] == explained["churn_probability"]
    assert predicted["prediction"] == explained["prediction"]


def test_no_error_message_echoes_untrusted_input_as_markup(
    demo: TestClient, record: dict[str, Any]
) -> None:
    """A rejected value may be quoted back; it is quoted as text, in a JSON string."""
    response = demo.post(EXPLAIN, json={**record, "Contract": ""})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert "<" not in response.json()["error"]["message"]


# --------------------------------------------------------------------------- #
# One analysis is one observation.
#
# The explanation response already carries every predictive field the page renders, so
# the page posts to /api/v1/explain and nowhere else. Two calls for one action would
# score the same customer twice and count one analysis as two records in the window.
#
# /api/v1/predict is unchanged and still observes exactly once when a client calls it:
# these are two independent operations, and a client that deliberately calls both gets
# two observations because that is two requests.
# --------------------------------------------------------------------------- #


def _monitored(policy_path: Path) -> TestClient:
    return TestClient(
        create_app(
            settings=ServingSettings(
                policy_path=policy_path, portfolio_ui_enabled=True, monitoring_enabled=True
            )
        )
    )


def test_one_analysis_is_observed_exactly_once(policy_path: Path, record: dict[str, Any]) -> None:
    """N before, N + 1 after. Never N + 2."""
    with _monitored(policy_path) as client:
        before = client.get(MONITORING).json()
        client.post(EXPLAIN, json=record)
        after = client.get(MONITORING).json()

    assert before["n_records"] == 0
    assert after["n_records"] == before["n_records"] + 1
    assert after["data_quality"]["successful_records"] == 1
    assert after["data_quality"]["records_total"] == 1
    assert after["data_quality"]["requests_total"] == 1


def test_a_direct_prediction_call_is_observed_exactly_once(
    policy_path: Path, record: dict[str, Any]
) -> None:
    """The Phase 11 endpoint keeps its Phase 12 semantics, demo or no demo."""
    with _monitored(policy_path) as client:
        before = client.get(MONITORING).json()
        client.post(PREDICT, json=record)
        after = client.get(MONITORING).json()

    assert after["n_records"] == before["n_records"] + 1
    assert after["data_quality"]["successful_records"] == 1


def test_calling_both_endpoints_is_two_observations_because_it_is_two_requests(
    policy_path: Path, record: dict[str, Any]
) -> None:
    """Not a defect: the page never does this, but an API client is entitled to."""
    with _monitored(policy_path) as client:
        client.post(PREDICT, json=record)
        client.post(EXPLAIN, json=record)
        body = client.get(MONITORING).json()

    assert body["n_records"] == 2
    assert body["data_quality"]["successful_records"] == 2


def test_reading_the_monitoring_endpoint_observes_nothing(
    policy_path: Path, record: dict[str, Any]
) -> None:
    """The page refreshes it after every analysis; that refresh must not count."""
    with _monitored(policy_path) as client:
        client.post(EXPLAIN, json=record)
        for _ in range(5):
            client.get(MONITORING)
        body = client.get(MONITORING).json()

    assert body["n_records"] == 1
    assert body["data_quality"]["records_total"] == 1


def test_the_analysis_probability_is_unaffected_by_monitoring(
    demo: TestClient, policy_path: Path, record: dict[str, Any]
) -> None:
    """Observation is observation: the same payload gets the same float either way."""
    without = demo.post(EXPLAIN, json=record).json()
    with _monitored(policy_path) as client:
        with_monitoring = client.post(EXPLAIN, json=record).json()

    assert without["churn_probability"] == with_monitoring["churn_probability"]
    assert without["model_logit"] == with_monitoring["model_logit"]
