"""The Phase 13 machine-readable record: what the product is, as data.

Built from the running system — the metadata that exists, the coupling blocks the
equivalences imply, the assets on disk — rather than transcribed by hand, and
deterministic for the same reason every other record here is: no timestamp, no
hostname, no run id, so ``--verify`` can compare bytes.

Every prohibition it asserts names the test that establishes it. The ones worth
reading first are the two that separate this phase from a demo that would have been
easier to build: ``local_explanation_method = EXACT_LOGISTIC_DECOMPOSITION`` (not an
approximation), and ``prediction_explanation_probability_match = true`` (not two
endpoints that usually agree).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from churn.config import PROJECT_ROOT
from churn.portfolio.coupling import COUPLING_BLOCKS, block_records
from churn.portfolio.explanation import EXPLANATION_METHOD
from churn.portfolio.metadata import (
    EXPECTED_PORTFOLIO_METADATA_SHA256,
    PORTFOLIO_METADATA_RELATIVE_PATH,
    PORTFOLIO_METADATA_TRUST_ANCHOR,
    PortfolioMetadata,
    metadata_digest,
)

logger = logging.getLogger(__name__)

RECORD_PATH = PROJECT_ROOT / "reports" / "experiments" / "portfolio_results.json"

SCHEMA_VERSION = 1
EXPERIMENT = "phase13-portfolio-product"
COMPONENT = "PORTFOLIO_PRODUCT"
PHASE = 13

_TESTS = "tests"

#: ``(claim, module, test)`` — where each prohibition is actually established.
_EVIDENCE: tuple[tuple[str, str, str], ...] = (
    ("fit_calls", "test_portfolio_isolation", "test_the_explanation_path_calls_no_fit"),
    ("holdout_loaded", "test_portfolio_isolation", "test_the_holdout_is_never_reachable"),
    ("shap_used", "test_portfolio_isolation", "test_no_approximate_attribution_library_is_used"),
    ("lime_used", "test_portfolio_isolation", "test_no_approximate_attribution_library_is_used"),
    (
        "local_explanation_method",
        "test_portfolio_explanation",
        "test_the_contributions_reconstruct_the_logit_exactly",
    ),
    (
        "probability_reconstruction",
        "test_portfolio_explanation",
        "test_the_sigmoid_of_the_logit_reconstructs_the_probability",
    ),
    (
        "explanation_never_scores",
        "test_portfolio_explanation",
        "test_the_explanation_core_never_calls_predict_proba",
    ),
    (
        "all_raw_features_covered",
        "test_portfolio_explanation",
        "test_every_raw_feature_appears_exactly_once",
    ),
    (
        "reconstruction_gate",
        "test_portfolio_explanation",
        "test_a_decomposition_that_does_not_close_is_refused",
    ),
    (
        "coupled_groups_sum_exactly",
        "test_portfolio_coupling",
        "test_a_grouped_block_equals_the_sum_of_its_members",
    ),
    (
        "coupling_follows_phase10",
        "test_portfolio_coupling",
        "test_the_blocks_are_derived_from_the_frozen_equivalences",
    ),
    (
        "inconsistent_records_are_not_grouped",
        "test_portfolio_coupling",
        "test_a_broken_equivalence_is_not_presented_as_one_block",
    ),
    (
        "metadata_holdout_not_reopened",
        "test_portfolio_metadata",
        "test_the_metadata_reopens_no_holdout",
    ),
    (
        "metadata_deterministic",
        "test_portfolio_metadata",
        "test_the_metadata_is_deterministic",
    ),
    (
        "accuracy_not_headline",
        "test_portfolio_metadata",
        "test_accuracy_is_never_a_headline_metric",
    ),
    (
        "analyst_exposure_caveat",
        "test_portfolio_metadata",
        "test_the_analyst_exposure_caveat_travels_with_the_metrics",
    ),
)

VERIFIED_BY: dict[str, str] = {
    claim: f"{_TESTS}/{module}.py::{test}" for claim, module, test in _EVIDENCE
}

#: Claims established in the serving suite, because they are about the boundary.
SERVING_VERIFIED_BY: dict[str, str] = {
    "prediction_explanation_probability_match": (
        "serving/tests/test_serving_portfolio.py::"
        "test_predict_and_explain_agree_exactly_on_every_shared_field"
    ),
    "canonical_prediction_service_reused": (
        "serving/tests/test_serving_portfolio.py::"
        "test_the_explanation_goes_through_the_canonical_scoring_path"
    ),
    "prediction_endpoint_unchanged": (
        "serving/tests/test_portfolio_ui.py::"
        "test_the_phase_11_endpoints_answer_identically_with_the_demo_on"
    ),
    "ui_scoring_calls_per_submission": (
        "serving/tests/test_portfolio_ui.py::test_one_submission_makes_exactly_one_scoring_call"
    ),
    "ui_monitoring_observations_per_submission": (
        "serving/tests/test_serving_portfolio.py::test_one_analysis_is_observed_exactly_once"
    ),
    "portfolio_metadata_integrity_validated": (
        "serving/tests/test_serving_portfolio.py::test_tampered_metadata_stops_the_demo"
    ),
    "dangerous_dom_sinks_used": (
        "serving/tests/test_portfolio_ui.py::test_the_script_uses_no_dangerous_dom_sink"
    ),
    "dom_xss_guarded": (
        "serving/tests/test_serving_portfolio.py::"
        "test_a_markup_shaped_unseen_value_is_returned_as_data"
    ),
    "ui_enabled_by_default": (
        "serving/tests/test_portfolio_ui.py::test_the_demo_is_absent_when_the_ui_is_off"
    ),
    "phase11_route_list_unchanged_by_default": (
        "serving/tests/test_portfolio_ui.py::test_the_phase_11_route_list_is_unchanged_by_default"
    ),
    "external_runtime_assets": (
        "serving/tests/test_portfolio_ui.py::test_the_page_requests_nothing_from_the_internet"
    ),
    "analytics_used": (
        "serving/tests/test_portfolio_ui.py::test_the_page_carries_no_analytics_and_no_storage"
    ),
    "no_frontend_source_of_truth": (
        "serving/tests/test_portfolio_ui.py::test_the_script_hardcodes_no_model_constant"
    ),
    "monitoring_small_window_respected": (
        "serving/tests/test_portfolio_ui.py::test_the_page_cannot_bypass_the_small_window_policy"
    ),
}

NOTES: tuple[str, ...] = (
    "Phase 13 makes an already-validated system demonstrable. It adds NO modelling: no fit, no "
    "tuning, no feature engineering, no recalibration, no threshold selection and no new "
    "experiment. Every number the product shows comes from a frozen artefact or from the "
    "request the visitor just submitted.",
    "The local explanation is an EXACT ALGEBRAIC DECOMPOSITION, not an attribution estimate. A "
    "logistic regression's response surface is known in closed form, so SHAP, LIME and "
    "permutation importance would add a dependency, a random seed and an approximation error in "
    "exchange for a worse answer to a question already answered exactly. The primitives are "
    "reused from Phase 10 rather than reimplemented: two implementations of one identity are two "
    "chances to drift.",
    "The explanation endpoint NEVER SCORES. It calls the same canonical single-record path the "
    "prediction endpoints call and decomposes the row that path returned, so /api/v1/predict and "
    "/api/v1/explain cannot disagree about a probability — the value is copied, not recomputed.",
    "An explanation is published only if its arithmetic closes. Every explained request "
    "re-derives intercept + sum(contributions), pushes it through the logistic function and "
    "compares it with the probability that was served, at the Phase 10 tolerance. A mismatch "
    "returns EXPLANATION_UNAVAILABLE; the prediction endpoints are unaffected and remain "
    "authoritative.",
    "Structurally coupled features are presented as ONE row whose value is the exact sum of its "
    "members. A customer without internet produces seven correct terms that are one fact about "
    "the product's encoding, and listing them separately would read as seven independent "
    "reasons. Grouping changes the layout, never the arithmetic: the grand total still "
    "reconstructs the logit. A record that BREAKS the equivalence is deliberately not grouped.",
    "The metadata the product presents as measured is PINNED. Its digest is compared at "
    "startup against churn.portfolio.metadata.EXPECTED_PORTFOLIO_METADATA_SHA256, a constant in "
    "source rather than a field of the file being checked, exactly as Phase 12 pins its "
    "reference profile. Structural validation could not close this on its own: a file with an "
    "edited average precision satisfies every invariant while showing a number nobody measured. "
    "On a mismatch the process does not start — it does not fall back to empty metadata and it "
    "does not display the metrics unverified.",
    "The frontend inserts every dynamic value as TEXT. A categorical may carry an arbitrary "
    "'custom / unseen value' that returns in value_display, active_level and error messages, so "
    "the script constructs no HTML string anywhere: createElement, textContent, value, "
    "replaceChildren and appendChild only. innerHTML, outerHTML, insertAdjacentHTML, "
    "document.write, eval and new Function do not appear in app.js, and a test enforces the "
    "absence. No sanitiser was added, because a value that is never parsed as HTML has nothing "
    "to sanitise.",
    "ONE SUBMISSION IS ONE OBSERVATION. The main action posts to /api/v1/explain and nowhere "
    "else: that response already carries churn_probability, prediction, decision, threshold, "
    "comparison and calibration_policy, so calling /api/v1/predict as well would score the same "
    "customer twice, double the latency, and count one analysis as two records in the monitoring "
    "window. /api/v1/predict is untouched and remains the endpoint for ordinary API consumers; a "
    "client that deliberately calls both gets two observations, which is correct, because that "
    "is two requests.",
    "The UI is OPT-IN and off by default, for the same reason monitoring is: Phase 11 published "
    "a route list and recorded it, so a demo that added routes by default would silently change "
    "a published contract. reports/experiments/serving_results.json is unchanged and still "
    "verifies byte for byte.",
    "The frontend is a PRESENTATION LAYER and holds no source of truth. The threshold, the "
    "comparison, the calibration policy, the fingerprint and every metric arrive from the API; "
    "the only thing the script computes is the pixel position of a marker on an axis.",
    "Categorical dropdowns are a UI CONVENIENCE, not a contract change. The API still accepts "
    "any non-blank category, because the frozen encoder was fitted with handle_unknown='ignore' "
    "so a new contract term or payment method can still be scored. The page offers a "
    "'custom / unseen value' input precisely so that behaviour stays demonstrable.",
    "Structural prefilling in the form is UI ASSISTANCE. Setting InternetService = No fills the "
    "six add-ons for convenience; the user can override any of them, the API accepts the result, "
    "and Phase 12 monitoring counts it as a structural inconsistency exactly as before. "
    "prepare_features is untouched.",
    "TotalCharges keeps its frozen rule. The page explains that a blank is admissible only at "
    "tenure 0 and sends the field as typed; the structural-zero decision belongs to the frozen "
    "preprocessing and is not reimplemented in JavaScript.",
    "The held-out metrics are COPIED from Phase 9D's committed artefact by an offline build "
    "step. The serving process reads a small versioned summary instead of the evaluation record, "
    "so nothing in the runtime is in a position to re-evaluate. evaluations_performed stays 1.",
    "Accuracy is reported as AUXILIARY, never as the headline. On a population where 73.5 % of "
    "customers do not churn, a model that predicts nobody churns scores close to it while "
    "finding no one. Average precision and ROC-AUC lead, and they are named as ranking quality "
    "rather than as accuracy or certainty.",
    "The analyst-exposure limitation carried since Phase 3 is a FIELD of the metadata artefact, "
    "not a line of HTML, so the metrics cannot be displayed without it. A portfolio page is "
    "exactly where an inconvenient caveat goes missing.",
    "There is no risk banding. A single threshold was justified in Phase 9B, so the product "
    "reports which side of it a customer falls on and invents no low/medium/high cuts.",
    "There are no retention recommendations. The dataset carries no costs, no capacity and no "
    "treatment effects, so a discount or a call-list would be an invention. The product predicts "
    "and explains; acting on that needs business inputs it does not have.",
    "Nothing is persisted. No payload, no identifier, no probability and no explanation is "
    "written anywhere, and the page uses no localStorage, sessionStorage, IndexedDB or cookie. "
    "No analytics, no telemetry and no third-party request of any kind.",
    "Every runtime asset is local. The page loads one stylesheet and one script from this "
    "service and requests nothing else, so the demo runs fully offline.",
)


def build_record(
    metadata: PortfolioMetadata,
    static_assets: tuple[str, ...],
) -> dict[str, Any]:
    """Return the Phase 13 record as a plain, JSON-encodable mapping.

    The recorded metadata digest is computed from the summary passed in, not copied
    from a constant, so a record built against a summary that drifted from the pin
    would disagree with ``expected_sha256`` instead of quietly restating it.
    """
    provenance = metadata.provenance
    evaluation = metadata.evaluation
    metadata_sha256 = metadata_digest(metadata)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "component": COMPONENT,
        "model_changed": False,
        "fit_calls": 0,
        "holdout_loaded": False,
        "threshold_changed": False,
        "calibration_changed": False,
        "new_experiment_performed": False,
        "frozen_model": {
            "freeze_commit": provenance.freeze_commit,
            "model_fingerprint_sha256": provenance.model_fingerprint_sha256,
            "pipeline_sha256": provenance.pipeline_sha256,
            "decision_policy_sha256": provenance.decision_policy_sha256,
            "estimator": metadata.policy.estimator,
            "calibration_policy": metadata.policy.calibration_policy,
            "threshold_policy": metadata.policy.threshold_policy,
            "threshold": metadata.policy.threshold,
            "comparison": metadata.policy.comparison,
            "n_features": metadata.policy.n_features,
            "n_transformed_features": metadata.policy.n_transformed_features,
        },
        "product": {
            "ui_enabled_by_default": False,
            "environment_flag": "CHURN_SERVING_PORTFOLIO_UI",
            "demo_route": "/demo",
            "static_mount": "/demo/static",
            "static_assets": list(static_assets),
            "stack": "HTML5, modern CSS, vanilla ES6 JavaScript, inline SVG",
            "frontend_framework": None,
            "frontend_build_step": False,
            "new_runtime_dependencies": [],
            "external_runtime_assets": False,
            "cdn_used": False,
            "web_fonts_loaded": False,
            "runs_offline": True,
            "phase11_route_list_unchanged_by_default": True,
            "phase11_serving_record_rewritten": False,
            "phase12_monitoring_record_rewritten": False,
        },
        "explanation": {
            "endpoint": "/api/v1/explain",
            "canonical_prediction_service_reused": True,
            "prediction_endpoint_unchanged": True,
            "canonical_scoring_path": (
                "churn.serving.inference.canonical_features_and_probability"
            ),
            "canonical_scoring_entry_point": (
                "churn.serving.service.ChurnInferenceService.score_with_features"
            ),
            "explanation_scores_independently": False,
            "local_explanation_method": EXPLANATION_METHOD,
            "identity": "logit(P) = intercept + sum_f contribution_f ; P = sigmoid(logit)",
            "shap_used": False,
            "lime_used": False,
            "permutation_importance_used": False,
            "sampling_used": False,
            "random_seed_used": False,
            "prediction_explanation_probability_match": True,
            "reconstruction_gate": True,
            "reconstruction_failure_behaviour": (
                "the explanation endpoint returns EXPLANATION_UNAVAILABLE and no "
                "explanation is shown; the prediction endpoints are unaffected"
            ),
            "reused_from_phase10": [
                "churn.modeling.interpretation.extract_terms",
                "churn.modeling.interpretation.feature_groups",
                "churn.modeling.interpretation.transform_features",
                "churn.modeling.interpretation.group_contributions",
                "churn.modeling.interpretation.sigmoid",
                "churn.modeling.interpretation.logit_of",
            ],
            "phase10_source_modified": False,
            "phase10_artefacts_modified": False,
            "contribution_fields": [
                "feature",
                "kind",
                "value_display",
                "active_level",
                "is_unseen_level",
                "contribution_log_odds",
                "direction",
            ],
            "ordering": "local to the request, by absolute contribution",
            "ordering_is_global_feature_importance": False,
            "causal_interpretation_claimed": False,
        },
        "ui_request_flow": {
            "ui_analysis_endpoint": "/api/v1/explain",
            "network_calls_per_analysis": [
                "POST /api/v1/explain",
                "GET /api/v1/monitoring",
            ],
            "ui_scoring_calls_per_submission": 1,
            "ui_monitoring_observations_per_submission": 1,
            "ui_calls_predict_and_explain_for_one_action": False,
            "monitoring_refresh_is_read_only": True,
            "monitoring_refresh_scores_anything": False,
            "predict_endpoint_called_by_the_page": False,
            "predict_observations_when_called_directly": 1,
            "explain_observations_when_called_directly": 1,
            "rationale": (
                "the explanation response already carries every predictive field the "
                "page renders, so a second call would score the same customer twice "
                "and count one analysis as two monitoring records"
            ),
        },
        "metadata_integrity": {
            "portfolio_metadata_integrity_validated": True,
            "portfolio_metadata_digest_pinned": True,
            "portfolio_metadata_sha256": metadata_sha256,
            "portfolio_metadata_expected_sha256": EXPECTED_PORTFOLIO_METADATA_SHA256,
            "portfolio_metadata_digest_mismatches": 0,
            "source_of_expected_sha256": PORTFOLIO_METADATA_TRUST_ANCHOR,
            "expected_digest_stored_in_the_checked_file": False,
            "expected_digest_stored_in_decision_policy": False,
            "digest_scope": "canonical serialisation of the summary",
            "enforced_at": "portfolio startup, when the UI is enabled",
            "loaded_when_ui_disabled": False,
            "behaviour_on_mismatch": (
                "PortfolioStartupError; the process does not start, does not fall "
                "back to empty metadata and does not display unverified metrics"
            ),
        },
        "structural_coupling": {
            "blocks": block_records(),
            "n_blocks": len(COUPLING_BLOCKS),
            "derived_from": "churn.monitoring.structural.STRUCTURAL_RULES",
            "group_value_is_exact_sum": True,
            "grouping_changes_arithmetic": False,
            "inconsistent_records_grouped": False,
            "api_contract_changed": False,
            "prepare_features_changed": False,
        },
        "evaluation_shown": {
            "source": PORTFOLIO_METADATA_RELATIVE_PATH,
            "copied_from": provenance.holdout_results_path,
            "holdout_results_sha256": provenance.holdout_results_sha256,
            "holdout_reopened": provenance.holdout_reopened,
            "metrics_recomputed": provenance.metrics_recomputed,
            "evaluations_performed": evaluation.evaluations_performed,
            "n_samples": evaluation.n_samples,
            "headline_metrics": [entry.key for entry in evaluation.headline],
            "auxiliary_metrics": [entry.key for entry in evaluation.auxiliary],
            "accuracy_is_headline": False,
            "analyst_exposure_caveat_shown": True,
            "runtime_reads_holdout_results": False,
        },
        "decision_presentation": {
            "threshold_source": "GET /api/v1/model",
            "threshold_hardcoded_in_frontend": False,
            "decision_computed_in_frontend": False,
            "score_rounded_before_decision": False,
            "risk_bands_invented": False,
            "sides": ["below decision threshold", "at/above decision threshold"],
            "retention_recommendations_generated": False,
            "calibration_claimed": False,
        },
        "monitoring_integration": {
            "monitoring_endpoint_consumed": True,
            "small_window_policy_respected": True,
            "suppressed_distributions_reconstructed": False,
            "disabled_state_reported_honestly": True,
        },
        "privacy": {
            "payloads_persisted": False,
            "identifiers_persisted": False,
            "row_level_predictions_persisted": False,
            "explanations_persisted": False,
            "customer_id_accepted": False,
            "analytics_used": False,
            "third_party_requests": False,
            "browser_storage_used": False,
            "real_customer_data_embedded": False,
            "synthetic_examples_labelled_as_such": True,
        },
        "security": {
            "static_root_from_operator_configuration": True,
            "user_supplied_paths_accepted": False,
            "upload_endpoints": False,
            "write_endpoints": False,
            "error_responses_carry_tracebacks": False,
            "error_responses_carry_filesystem_paths": False,
            "user_controlled_values_rendered_as_text": True,
            "dangerous_dom_sinks_used": False,
            "dom_xss_guarded": True,
            "dom_sinks_banned": [
                "innerHTML",
                "outerHTML",
                "insertAdjacentHTML",
                "document.write",
                "eval",
                "new Function",
            ],
            "dom_write_primitives_used": [
                "createElement",
                "textContent",
                "value",
                "setAttribute",
                "replaceChildren",
                "appendChild",
            ],
            "html_string_constructed_from_data": False,
            "sanitizer_library_added": False,
            "inline_event_handlers_in_html": False,
            "inline_script_in_html": False,
        },
        "accessibility": {
            "labels_associated_with_inputs": True,
            "semantic_headings": True,
            "native_buttons": True,
            "focus_states": True,
            "aria_live_regions": ["result", "errors", "status"],
            "skip_link": True,
            "reduced_motion_respected": True,
            "formal_wcag_audit": False,
        },
        "responsiveness": {
            "breakpoints": [1080, 720],
            "primary_target": "desktop",
            "pixel_perfect_visual_tests": False,
        },
        "deployment": {
            "cloud_deployment_performed": False,
            "domain_registered": False,
            "runs_locally": True,
            "single_command": (
                "CHURN_SERVING_PORTFOLIO_UI=1 uv run --project serving python -m churn.serving"
            ),
        },
        "verified_by": dict(VERIFIED_BY),
        "verified_by_serving_suite": dict(SERVING_VERIFIED_BY),
        "notes": list(NOTES),
    }


def canonical_json(record: dict[str, Any]) -> str:
    """Return the one serialisation this project writes and hashes."""
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def write_record(record: dict[str, Any], path: Path | None = None) -> Path:
    """Write the record with LF endings and a trailing newline."""
    destination = path or RECORD_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(record), encoding="utf-8", newline="\n")
    logger.info("Wrote portfolio record: %s", destination)
    return destination


def read_record(path: Path | None = None) -> dict[str, Any]:
    """Read the committed record.

    Raises:
        FileNotFoundError: If it has not been produced yet.
    """
    source = path or RECORD_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Portfolio record not found at {source}. Produce it with "
            "`uv run python scripts/build_portfolio_record.py`."
        )
    return json.loads(source.read_text(encoding="utf-8"))


def record_invariants(record: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Return ``(name, holds, detail)`` for every claim the record must not break."""
    product = record["product"]
    explanation = record["explanation"]
    coupling = record["structural_coupling"]
    shown = record["evaluation_shown"]
    decision = record["decision_presentation"]
    privacy = record["privacy"]
    security = record["security"]
    flow = record["ui_request_flow"]
    integrity = record["metadata_integrity"]

    return [
        ("phase_is_thirteen", record["phase"] == PHASE, str(record["phase"])),
        ("component_is_the_product", record["component"] == COMPONENT, record["component"]),
        ("model_unchanged", record["model_changed"] is False, "model_changed=false"),
        ("no_fit", record["fit_calls"] == 0, str(record["fit_calls"])),
        ("no_holdout", record["holdout_loaded"] is False, "holdout_loaded=false"),
        (
            "threshold_and_calibration_unchanged",
            record["threshold_changed"] is False and record["calibration_changed"] is False,
            "both false",
        ),
        (
            "ui_is_opt_in",
            product["ui_enabled_by_default"] is False
            and product["phase11_route_list_unchanged_by_default"] is True,
            product["environment_flag"],
        ),
        (
            "phase11_and_phase12_records_untouched",
            product["phase11_serving_record_rewritten"] is False
            and product["phase12_monitoring_record_rewritten"] is False,
            "serving_results.json and monitoring_results.json unchanged",
        ),
        (
            "no_new_dependency_and_no_build_step",
            product["new_runtime_dependencies"] == []
            and product["frontend_build_step"] is False
            and product["frontend_framework"] is None,
            product["stack"],
        ),
        (
            "no_external_runtime_asset",
            product["external_runtime_assets"] is False
            and product["cdn_used"] is False
            and product["web_fonts_loaded"] is False
            and product["runs_offline"] is True,
            "everything local",
        ),
        (
            "explanation_is_exact",
            explanation["local_explanation_method"] == EXPLANATION_METHOD
            and explanation["shap_used"] is False
            and explanation["lime_used"] is False
            and explanation["permutation_importance_used"] is False
            and explanation["sampling_used"] is False,
            EXPLANATION_METHOD,
        ),
        (
            "explanation_reuses_the_prediction_path",
            explanation["canonical_prediction_service_reused"] is True
            and explanation["prediction_endpoint_unchanged"] is True
            and explanation["explanation_scores_independently"] is False
            and explanation["prediction_explanation_probability_match"] is True,
            explanation["canonical_scoring_path"],
        ),
        (
            "one_submission_is_one_observation",
            flow["ui_analysis_endpoint"] == "/api/v1/explain"
            and flow["ui_scoring_calls_per_submission"] == 1
            and flow["ui_monitoring_observations_per_submission"] == 1
            and flow["ui_calls_predict_and_explain_for_one_action"] is False
            and flow["predict_endpoint_called_by_the_page"] is False,
            " + ".join(flow["network_calls_per_analysis"]),
        ),
        (
            "the_monitoring_refresh_scores_nothing",
            flow["monitoring_refresh_is_read_only"] is True
            and flow["monitoring_refresh_scores_anything"] is False,
            "GET only",
        ),
        (
            "portfolio_metadata_is_pinned",
            integrity["portfolio_metadata_integrity_validated"] is True
            and integrity["portfolio_metadata_digest_pinned"] is True
            and integrity["portfolio_metadata_digest_mismatches"] == 0
            and integrity["portfolio_metadata_sha256"]
            == integrity["portfolio_metadata_expected_sha256"],
            integrity["source_of_expected_sha256"],
        ),
        (
            "the_expectation_is_independent_of_the_checked_file",
            integrity["expected_digest_stored_in_the_checked_file"] is False
            and integrity["expected_digest_stored_in_decision_policy"] is False,
            integrity["enforced_at"],
        ),
        (
            "untrusted_values_are_rendered_as_text",
            security["user_controlled_values_rendered_as_text"] is True
            and security["dangerous_dom_sinks_used"] is False
            and security["dom_xss_guarded"] is True
            and security["html_string_constructed_from_data"] is False
            and security["sanitizer_library_added"] is False,
            f"{len(security['dom_sinks_banned'])} sink(s) banned",
        ),
        (
            "explanation_is_gated",
            explanation["reconstruction_gate"] is True,
            "reconstruction checked per request",
        ),
        (
            "phase10_untouched",
            explanation["phase10_source_modified"] is False
            and explanation["phase10_artefacts_modified"] is False,
            "primitives reused, nothing modified",
        ),
        (
            "ordering_is_not_called_importance",
            explanation["ordering_is_global_feature_importance"] is False
            and explanation["causal_interpretation_claimed"] is False,
            explanation["ordering"],
        ),
        (
            "coupling_is_presentation_only",
            coupling["group_value_is_exact_sum"] is True
            and coupling["grouping_changes_arithmetic"] is False
            and coupling["api_contract_changed"] is False
            and coupling["prepare_features_changed"] is False,
            f"{coupling['n_blocks']} block(s)",
        ),
        (
            "inconsistent_records_are_not_grouped",
            coupling["inconsistent_records_grouped"] is False,
            "a broken equivalence is shown ungrouped",
        ),
        (
            "holdout_not_reopened",
            shown["holdout_reopened"] is False
            and shown["metrics_recomputed"] is False
            and shown["runtime_reads_holdout_results"] is False
            and shown["evaluations_performed"] == 1,
            f"{shown['evaluations_performed']} evaluation(s)",
        ),
        (
            "accuracy_is_not_headline",
            shown["accuracy_is_headline"] is False
            and "accuracy" not in shown["headline_metrics"]
            and {"average_precision", "roc_auc"} <= set(shown["headline_metrics"]),
            f"headline={shown['headline_metrics']}",
        ),
        (
            "analyst_exposure_caveat_shown",
            shown["analyst_exposure_caveat_shown"] is True,
            "carried since Phase 3",
        ),
        (
            "no_invented_decision_layer",
            decision["threshold_hardcoded_in_frontend"] is False
            and decision["decision_computed_in_frontend"] is False
            and decision["risk_bands_invented"] is False
            and decision["retention_recommendations_generated"] is False
            and decision["calibration_claimed"] is False,
            " / ".join(decision["sides"]),
        ),
        (
            "monitoring_privacy_respected",
            record["monitoring_integration"]["small_window_policy_respected"] is True
            and record["monitoring_integration"]["suppressed_distributions_reconstructed"] is False,
            "the frontend cannot bypass suppression",
        ),
        (
            "nothing_persisted",
            privacy["payloads_persisted"] is False
            and privacy["identifiers_persisted"] is False
            and privacy["row_level_predictions_persisted"] is False
            and privacy["explanations_persisted"] is False
            and privacy["browser_storage_used"] is False,
            "none",
        ),
        (
            "no_analytics_and_no_third_party",
            privacy["analytics_used"] is False and privacy["third_party_requests"] is False,
            "none",
        ),
        (
            "no_real_customer_data_embedded",
            privacy["real_customer_data_embedded"] is False
            and privacy["synthetic_examples_labelled_as_such"] is True,
            "synthetic fixtures only",
        ),
        (
            "static_serving_is_safe",
            security["user_supplied_paths_accepted"] is False
            and security["upload_endpoints"] is False
            and security["write_endpoints"] is False
            and security["error_responses_carry_tracebacks"] is False,
            "fixed root, no user paths",
        ),
        (
            "no_cloud_deployment",
            record["deployment"]["cloud_deployment_performed"] is False,
            "runs locally",
        ),
        (
            "every_claim_has_evidence",
            set(VERIFIED_BY) <= set(record["verified_by"])
            and set(SERVING_VERIFIED_BY) <= set(record["verified_by_serving_suite"]),
            "verified_by",
        ),
    ]


__all__ = [
    "COMPONENT",
    "EXPERIMENT",
    "PHASE",
    "RECORD_PATH",
    "SCHEMA_VERSION",
    "SERVING_VERIFIED_BY",
    "VERIFIED_BY",
    "build_record",
    "canonical_json",
    "read_record",
    "record_invariants",
    "write_record",
]
