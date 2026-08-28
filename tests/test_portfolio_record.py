"""The Phase 13 record must be reproducible, and every claim in it must name a test.

A record is only worth reading if it cannot drift from the system it describes. This
file checks both halves: the file on disk equals a fresh build, and nothing it asserts
is an unverified promise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from churn.portfolio.metadata import load_portfolio_metadata
from churn.portfolio.results import (
    COMPONENT,
    PHASE,
    RECORD_PATH,
    SERVING_VERIFIED_BY,
    VERIFIED_BY,
    build_record,
    canonical_json,
    read_record,
    record_invariants,
)

ASSETS = ("index.html", "css/app.css", "js/app.js")


@pytest.fixture(scope="module")
def record():
    return build_record(load_portfolio_metadata(), ASSETS)


def test_the_record_is_deterministic() -> None:
    first = canonical_json(build_record(load_portfolio_metadata(), ASSETS))
    second = canonical_json(build_record(load_portfolio_metadata(), ASSETS))

    assert first == second
    for stamp in ("timestamp", "generated_at", "hostname", "run_id", "duration"):
        assert stamp not in first


def test_the_committed_record_matches_a_fresh_build() -> None:
    assert read_record() == build_record(load_portfolio_metadata(), ASSETS)
    assert canonical_json(read_record()) == RECORD_PATH.read_text(encoding="utf-8")


def test_every_invariant_holds(record) -> None:
    failed = [name for name, holds, _ in record_invariants(record) if not holds]

    assert not failed, failed
    assert len(record_invariants(record)) >= 20


def test_the_record_identifies_the_phase(record) -> None:
    assert record["phase"] == PHASE
    assert record["component"] == COMPONENT
    assert record["model_changed"] is False
    assert record["fit_calls"] == 0
    assert record["holdout_loaded"] is False
    assert record["new_experiment_performed"] is False


def test_the_record_pins_the_frozen_model(record) -> None:
    frozen = record["frozen_model"]

    assert (
        frozen["model_fingerprint_sha256"]
        == "a57568e18ec0333e1c85b590f98e75da93ebbfafff063860ec5daf4e460d939f"
    )
    assert (
        frozen["pipeline_sha256"]
        == "574fde36c6e2e991de7c3dfdab21981dc41eae2810d003ecbfb179dd504dc3d8"
    )
    assert frozen["threshold"] == 0.3272694566222328
    assert frozen["comparison"] == ">="
    assert frozen["calibration_policy"] == "NONE"


def test_the_record_declares_the_method_and_denies_the_alternatives(record) -> None:
    explanation = record["explanation"]

    assert explanation["local_explanation_method"] == "EXACT_LOGISTIC_DECOMPOSITION"
    assert explanation["shap_used"] is False
    assert explanation["lime_used"] is False
    assert explanation["permutation_importance_used"] is False
    assert explanation["sampling_used"] is False
    assert explanation["random_seed_used"] is False
    assert explanation["prediction_explanation_probability_match"] is True
    assert explanation["explanation_scores_independently"] is False


def test_the_record_says_the_ui_is_opt_in(record) -> None:
    product = record["product"]

    assert product["ui_enabled_by_default"] is False
    assert product["environment_flag"] == "CHURN_SERVING_PORTFOLIO_UI"
    assert product["phase11_route_list_unchanged_by_default"] is True
    assert product["phase11_serving_record_rewritten"] is False
    assert product["phase12_monitoring_record_rewritten"] is False


def test_the_record_declares_no_new_dependency(record) -> None:
    product = record["product"]

    assert product["new_runtime_dependencies"] == []
    assert product["frontend_framework"] is None
    assert product["frontend_build_step"] is False
    assert product["external_runtime_assets"] is False
    assert product["cdn_used"] is False
    assert product["runs_offline"] is True


def test_the_record_forbids_an_invented_decision_layer(record) -> None:
    decision = record["decision_presentation"]

    assert decision["risk_bands_invented"] is False
    assert decision["retention_recommendations_generated"] is False
    assert decision["decision_computed_in_frontend"] is False
    assert decision["threshold_hardcoded_in_frontend"] is False
    assert decision["calibration_claimed"] is False


def test_the_record_carries_the_coupling_blocks(record) -> None:
    coupling = record["structural_coupling"]

    assert coupling["n_blocks"] == 2
    assert coupling["derived_from"] == "churn.monitoring.structural.STRUCTURAL_RULES"
    assert coupling["group_value_is_exact_sum"] is True
    assert coupling["grouping_changes_arithmetic"] is False
    assert coupling["prepare_features_changed"] is False
    assert sum(block["n_members"] for block in coupling["blocks"]) == 9


def test_every_claim_names_a_test_that_exists(record) -> None:
    """A record that cites a test nobody wrote is a record that asserts nothing."""
    from churn.config import PROJECT_ROOT

    evidence = {**record["verified_by"], **record["verified_by_serving_suite"]}
    for claim, reference in evidence.items():
        path, test = reference.split("::")
        source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
        assert f"def {test}(" in source, f"{claim} cites {reference}, which does not exist"


def test_the_evidence_map_is_complete(record) -> None:
    assert set(VERIFIED_BY) <= set(record["verified_by"])
    assert set(SERVING_VERIFIED_BY) <= set(record["verified_by_serving_suite"])
    assert len(record["notes"]) >= 15


def test_the_record_names_no_dataset_and_no_customer(record) -> None:
    rendered = json.dumps(record)

    assert "customerID" not in rendered
    for needle in ("data/raw", "data/interim", ".csv"):
        assert needle not in rendered


def test_the_record_lives_where_the_other_experiment_records_do() -> None:
    assert RECORD_PATH.parent == Path(RECORD_PATH).parent
    assert RECORD_PATH.name == "portfolio_results.json"
    assert RECORD_PATH.parent.name == "experiments"


# --------------------------------------------------------------------------- #
# Claims added at the Phase 13 close: pinned metadata, DOM safety, and a flow
# described in words that are literally true.
# --------------------------------------------------------------------------- #


def test_the_record_pins_the_metadata_it_describes(record) -> None:
    """The recorded digest is the summary's own, and it is the pinned one."""
    from churn.portfolio.metadata import (
        EXPECTED_PORTFOLIO_METADATA_SHA256,
        metadata_digest,
    )

    integrity = record["metadata_integrity"]

    assert integrity["portfolio_metadata_sha256"] == metadata_digest(load_portfolio_metadata())
    assert integrity["portfolio_metadata_expected_sha256"] == EXPECTED_PORTFOLIO_METADATA_SHA256
    assert integrity["portfolio_metadata_integrity_validated"] is True
    assert integrity["portfolio_metadata_digest_pinned"] is True
    assert integrity["portfolio_metadata_digest_mismatches"] == 0


def test_the_record_says_where_the_expectation_lives(record) -> None:
    """A pin inside the file it checks would be an echo, not an expectation."""
    integrity = record["metadata_integrity"]

    assert (
        integrity["source_of_expected_sha256"]
        == "churn.portfolio.metadata.EXPECTED_PORTFOLIO_METADATA_SHA256"
    )
    assert integrity["expected_digest_stored_in_the_checked_file"] is False
    assert integrity["expected_digest_stored_in_decision_policy"] is False
    assert "does not start" in integrity["behaviour_on_mismatch"]
    assert integrity["loaded_when_ui_disabled"] is False


def test_the_record_declares_the_dom_safety_strategy(record) -> None:
    security = record["security"]

    assert security["user_controlled_values_rendered_as_text"] is True
    assert security["dangerous_dom_sinks_used"] is False
    assert security["dom_xss_guarded"] is True
    assert security["html_string_constructed_from_data"] is False
    assert security["sanitizer_library_added"] is False
    assert "innerHTML" in security["dom_sinks_banned"]
    assert "textContent" in security["dom_write_primitives_used"]


def test_the_record_describes_the_request_flow_literally(record) -> None:
    """The wording must be true of the JavaScript, not of an earlier intention."""
    flow = record["ui_request_flow"]

    assert flow["ui_analysis_endpoint"] == "/api/v1/explain"
    assert flow["ui_scoring_calls_per_submission"] == 1
    assert flow["ui_monitoring_observations_per_submission"] == 1
    assert flow["ui_calls_predict_and_explain_for_one_action"] is False
    assert flow["predict_endpoint_called_by_the_page"] is False
    assert flow["network_calls_per_analysis"] == [
        "POST /api/v1/explain",
        "GET /api/v1/monitoring",
    ]


def test_the_record_does_not_claim_the_page_calls_the_prediction_endpoint(record) -> None:
    """The retired wording said the frontend reused /predict. It never did."""
    explanation = record["explanation"]

    assert "prediction_endpoint_reused" not in explanation
    assert explanation["canonical_prediction_service_reused"] is True
    assert explanation["prediction_endpoint_unchanged"] is True
    assert (
        explanation["canonical_scoring_entry_point"]
        == "churn.serving.service.ChurnInferenceService.score_with_features"
    )


def test_the_record_keeps_predict_as_an_independent_operation(record) -> None:
    """Two deliberate API calls are two observations; one UI action is one."""
    flow = record["ui_request_flow"]

    assert flow["predict_observations_when_called_directly"] == 1
    assert flow["explain_observations_when_called_directly"] == 1
    assert flow["monitoring_refresh_is_read_only"] is True
    assert flow["monitoring_refresh_scores_anything"] is False


def test_the_new_claims_hold_as_invariants(record) -> None:
    names = {name for name, _, _ in record_invariants(record)}

    assert {
        "one_submission_is_one_observation",
        "the_monitoring_refresh_scores_nothing",
        "portfolio_metadata_is_pinned",
        "the_expectation_is_independent_of_the_checked_file",
        "untrusted_values_are_rendered_as_text",
    } <= names
    assert all(holds for _, holds, _ in record_invariants(record))
