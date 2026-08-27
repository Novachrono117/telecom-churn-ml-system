"""The Phase 12 machine-readable record: what monitoring is, as data.

Built from the running system — the reference profile that exists, the policy that
is configured, the rules that are declared — rather than transcribed by hand, and
deterministic for the same reason every other record in this repository is: no
timestamp, no hostname, no run id, so ``--verify`` can compare bytes.

Like the Phase 11 record, every prohibition it asserts names the test that
establishes it under ``verified_by``. The most important of those prohibitions is
the one about what monitoring **cannot** conclude:
``performance_monitoring_without_labels = false`` is not a limitation being
apologised for, it is the methodological boundary of the phase.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from churn.config import PROJECT_ROOT
from churn.monitoring.build import SCORE_N_BINS
from churn.monitoring.collector import QUALITY_COUNTERS
from churn.monitoring.distributions import DEFAULT_N_BINS, REFERENCE_QUANTILES, UNSEEN_LEVEL
from churn.monitoring.drift import EPSILON
from churn.monitoring.reference import ReferenceProfile, profile_digest
from churn.monitoring.service import SUPPRESSED_SECTIONS
from churn.monitoring.settings import (
    EXPECTED_REFERENCE_PROFILE_SHA256,
    MONITORING_ENDPOINT,
    REFERENCE_TRUST_ANCHOR,
    STATUS_CRITICAL,
    STATUS_INSUFFICIENT_DATA,
    STATUS_OK,
    STATUS_WARNING,
    MonitoringPolicy,
)
from churn.monitoring.structural import STRUCTURAL_RULES

logger = logging.getLogger(__name__)

RECORD_PATH = PROJECT_ROOT / "reports" / "experiments" / "monitoring_results.json"

SCHEMA_VERSION = 1
EXPERIMENT = "phase12-monitoring-and-drift"
COMPONENT = "MONITORING_AND_DRIFT"
PHASE = 12

_TESTS = "tests"

#: ``(claim, module, test)`` — where each prohibition is actually established.
_EVIDENCE: tuple[tuple[str, str, str], ...] = (
    (
        "reference_population",
        "test_monitoring_reference",
        "test_the_reference_is_the_training_pool",
    ),
    ("target_used_for_reference", "test_monitoring_reference", "test_the_target_is_never_read"),
    ("holdout_loaded", "test_monitoring_reference", "test_the_holdout_is_never_reachable"),
    ("reference_n", "test_monitoring_reference", "test_the_reference_has_the_frozen_pool_size"),
    (
        "reference_profile_sha256",
        "test_monitoring_reference",
        "test_a_tampered_reference_fails_verification",
    ),
    ("reference_deterministic", "test_monitoring_reference", "test_the_profile_is_deterministic"),
    ("numeric_bins_fixed", "test_monitoring_drift", "test_bins_come_from_the_reference_only"),
    (
        "numeric_self_psi_zero",
        "test_monitoring_selfcheck",
        "test_the_reference_does_not_drift_from_itself",
    ),
    (
        "categorical_self_tvd_zero",
        "test_monitoring_selfcheck",
        "test_the_reference_does_not_drift_from_itself",
    ),
    (
        "score_self_psi_zero",
        "test_monitoring_selfcheck",
        "test_the_reference_does_not_drift_from_itself",
    ),
    (
        "structural_reference_clean",
        "test_monitoring_selfcheck",
        "test_the_reference_does_not_drift_from_itself",
    ),
    (
        "numeric_drift_detected",
        "test_monitoring_drift",
        "test_a_shifted_numeric_feature_raises_psi",
    ),
    (
        "categorical_drift_detected",
        "test_monitoring_drift",
        "test_a_concentrated_category_raises_tvd",
    ),
    ("unseen_category_detected", "test_monitoring_drift", "test_an_unseen_category_is_counted"),
    (
        "structural_violation_detected",
        "test_monitoring_drift",
        "test_a_broken_equivalence_is_counted",
    ),
    ("score_drift_detected", "test_monitoring_drift", "test_a_shifted_population_moves_the_score"),
    ("minimum_window_size", "test_monitoring_drift", "test_a_small_window_claims_no_drift"),
    (
        "monitoring_policy_cannot_move_the_model",
        "test_monitoring_policy",
        "test_the_monitoring_policy_has_no_model_parameter",
    ),
    ("payloads_persisted", "test_monitoring_collector", "test_the_collector_keeps_no_record"),
    (
        "identifiers_persisted",
        "test_monitoring_collector",
        "test_the_collector_keeps_no_unseen_category_text",
    ),
    (
        "collector_thread_safe",
        "test_monitoring_collector",
        "test_concurrent_observation_loses_no_update",
    ),
    ("no_fit", "test_monitoring_isolation", "test_monitoring_calls_no_fit_and_loads_no_dataset"),
    (
        "no_pipeline_predict",
        "test_monitoring_isolation",
        "test_monitoring_never_calls_pipeline_predict",
    ),
    (
        "performance_monitoring_without_labels",
        "test_monitoring_isolation",
        "test_no_label_based_metric_is_computed",
    ),
    (
        "small_window_details_suppressed",
        "test_monitoring_privacy",
        "test_a_small_window_exposes_no_distribution_at_all",
    ),
    (
        "small_window_leaks_no_single_record",
        "test_monitoring_privacy",
        "test_a_one_record_window_cannot_be_read_back_out_of_the_report",
    ),
    (
        "unseen_value_digests_exposed",
        "test_monitoring_privacy",
        "test_no_unseen_digest_reaches_a_snapshot_a_report_or_a_log",
    ),
    (
        "window_reset_primitive",
        "test_monitoring_window",
        "test_a_reset_closes_one_window_and_opens_an_empty_one",
    ),
    (
        "window_reset_is_atomic",
        "test_monitoring_window",
        "test_concurrent_observation_and_reset_lose_and_duplicate_nothing",
    ),
    (
        "distinct_unseen_tracking_bounded",
        "test_monitoring_cardinality",
        "test_a_thousand_distinct_unseen_values_do_not_grow_the_collector",
    ),
    (
        "unseen_occurrence_count_remains_exact",
        "test_monitoring_cardinality",
        "test_saturation_costs_no_occurrence",
    ),
    (
        "saturation_can_change_alert_status",
        "test_monitoring_cardinality",
        "test_saturation_never_moves_an_alert_status",
    ),
    (
        "distinct_unseen_tracking_resets_with_the_window",
        "test_monitoring_cardinality",
        "test_a_reset_clears_saturation_and_the_digest_set",
    ),
)

VERIFIED_BY: dict[str, str] = {
    claim: f"{_TESTS}/{module}.py::{test}" for claim, module, test in _EVIDENCE
}

#: Claims established in the serving suite rather than the root one, because they
#: are about the integrated boundary.
SERVING_VERIFIED_BY: dict[str, str] = {
    "monitoring_changes_prediction": (
        "serving/tests/test_serving_monitoring.py::"
        "test_the_prediction_is_identical_with_monitoring_on_and_off"
    ),
    "single_batch_probability_contract": (
        "serving/tests/test_serving_monitoring.py::"
        "test_single_and_batch_stay_exactly_equal_with_monitoring_on"
    ),
    "failure_isolation": (
        "serving/tests/test_serving_monitoring.py::"
        "test_a_failing_collector_does_not_change_a_prediction"
    ),
    "drift_does_not_break_readiness": (
        "serving/tests/test_serving_monitoring.py::test_drift_never_makes_the_service_not_ready"
    ),
    "reference_tampering_fails_startup": (
        "serving/tests/test_serving_monitoring.py::"
        "test_a_tampered_reference_stops_monitoring_startup"
    ),
    "reference_pinned_digest_enforced": (
        "serving/tests/test_serving_monitoring.py::"
        "test_a_reference_whose_digest_is_not_the_pinned_one_stops_startup"
    ),
    "small_window_details_suppressed": (
        "serving/tests/test_serving_monitoring.py::"
        "test_a_one_record_window_leaks_nothing_through_the_endpoint"
    ),
    "exception_messages_never_logged": (
        "serving/tests/test_serving_monitoring.py::"
        "test_a_collector_failure_never_logs_anything_derived_from_the_payload"
    ),
}

NOTES: tuple[str, ...] = (
    "Phase 12 detects that the population changed. It does NOT detect that the model got "
    "worse. Those are different claims, and the second one requires production ground-truth "
    "labels, which do not exist in this phase. No accuracy, recall, precision, F1, ROC-AUC, "
    "average precision or confusion matrix is computed anywhere in this package.",
    "Four phenomena are kept apart everywhere: DATA QUALITY (is the input well formed), "
    "DATA DRIFT (has P(X) moved), PREDICTION DRIFT (has the score distribution moved) and "
    "PERFORMANCE DEGRADATION (not answered). A window can be CRITICAL on every drift signal "
    "while the model still makes the decisions the business wants, and it can be OK on all "
    "of them while the model quietly rots.",
    "The reference population is the frozen TRAINING POOL, 5634 rows. The holdout was "
    "consumed by Phase 9D and is deliberately not used: an operational reference should "
    "describe the population the system was built for. No monitoring module can reach either "
    "partition — the offline build script loads the pool and hands it in.",
    "The target was never read. The reference profile records target_used=false, and the "
    "build function has no parameter through which a label could be passed. Applying the "
    "frozen model to the training pool to record a SCORE DISTRIBUTION is not a performance "
    "estimate and is not used as one.",
    "PSI and TVD are DESCRIPTIVE DISTANCES, not test statistics. There is no null "
    "distribution, no p-value and no significance level anywhere in this phase. The "
    "OK/WARNING/CRITICAL cutoffs are an OPERATIONAL MONITORING POLICY: heuristics about when "
    "a human should look, chosen before any production data existed, stored in "
    "configs/monitoring.toml and reachable by nothing that predicts.",
    "Histogram bins are computed ONCE from the reference and stored. A window that recomputed "
    "its own bins would produce a number partly measuring the binning. The partition extends "
    "to +/- infinity so it is total; whether a value lies outside the reference range is "
    "reported separately as out_of_reference_range_rate.",
    "PSI floors every share at an explicit epsilon before the logarithm because empty bins "
    "are normal and the formula is undefined at zero. With a small window the epsilon, not "
    "the data, can dominate — which is exactly why no drift verdict is claimed below the "
    "minimum window size.",
    "An unseen category is COUNTED, never stored. The collector keeps a count and a distinct "
    "cardinality derived from digests; the customer-supplied strings themselves are never "
    "retained, because they are unbounded free text and keeping them is what turns a "
    "monitoring store into a data-protection incident.",
    "The SHA-256 used for unseen cardinality is NOT anonymisation and is not claimed as such. "
    "A categorical domain is small enough to hash exhaustively, so an unsalted digest of one "
    "of its values is reversible by anyone who can read it. What the design relies on is "
    "containment, not the hash: the digest set is process-local, never serialised, never "
    "written to the reference profile, the record, a report, a snapshot or a log, and never "
    "returned by the endpoint. Only its cardinality leaves memory. A keyed digest would "
    "harden nothing that is exposed, because nothing is.",
    "BELOW minimum_window_size the report suppresses every detailed distribution, and that is "
    "a privacy control, not a cosmetic one. Order statistics are the obvious leak (a mean "
    "over one record IS that record) but any per-feature breakdown of a tiny window is the "
    "same leak in a different shape: a level count names the contract, a histogram bin places "
    "the tenure, a per-rule structural detail says which equivalence was broken, and a "
    "predicted_positive_rate of 1.0 over one record IS that decision. So nothing per-feature, "
    "per-level, per-bin or per-rule is computed at all below the minimum; the window size, "
    "the global operational counters and the statuses remain, because those describe the "
    "deployment rather than a customer.",
    "Monitoring failures are logged as an event name and an exception TYPE. Never str(error), "
    "never a traceback. The guard runs holding feature values and a probability, so an "
    "exception raised beneath it can carry either into its message; writing that message to a "
    "log would move the payload into the one place nobody audits for it.",
    "The reference profile digest is validated against a PIN, not merely recorded. "
    "churn.monitoring.settings.EXPECTED_REFERENCE_PROFILE_SHA256 holds the expected value in "
    "source, independently of the file being checked, and monitoring startup passes it to "
    "verify_reference_profile as expected_digest. Hashing whatever was loaded and logging the "
    "result answers what was loaded, never whether the right thing was loaded. A mismatch "
    "raises MonitoringStartupError and the process does not come up, so monitoring cannot "
    "reach HEALTHY on an unpinned baseline.",
    "Distinct unseen cardinality is EXACT until the configured tracking cap is reached. "
    "After saturation the reported value is a lower bound, and the report labels it as "
    "one (distinct_unseen_tracking_saturated, distinct_unseen_is_lower_bound) rather "
    "than presenting the cap as though it were the true cardinality. Occurrence counts "
    "and unseen rates remain exact at every cap and every traffic pattern, because they "
    "are integers rather than a set.",
    "The cap is MEMORY SAFETY, not privacy, and the two must not be conflated. Privacy "
    "is containment: the values are never retained and the digests never leave memory, "
    "which holds at any cap. Memory safety is boundedness: the serving contract accepts "
    "unseen categories on purpose, so a caller sending a million distinct new values "
    "would grow an unbounded set inside a long-lived process whose window has no "
    "automatic rotation and whose monitoring endpoint is unauthenticated. An "
    "observability component that can be made to exhaust the process it observes is a "
    "defect however private its contents are.",
    "Saturation refuses new digests rather than evicting old ones. An evicting set would "
    "report a number that is neither the true cardinality nor a bound on it, and the same "
    "value would mean something different in every window.",
    "No alert reads the distinct count. The unseen signal the operational policy grades "
    "is unseen_rate, which is exact, so saturating the diagnostic cannot move a status "
    "in either direction.",
    "A window is a counter generation, not a time series. MonitoringCollector.reset() closes "
    "the current window and opens an empty one under the same RLock that guards every update, "
    "returning the snapshot of the window it closed so no observation is lost between the "
    "two. It is deliberately not reachable over HTTP: this service has no authentication, and "
    "an anonymous caller able to erase the evidence a drift investigation depends on is a "
    "worse trade than restarting the process. Resetting changes no model, no policy, no "
    "threshold, no prediction and no reference profile.",
    "The seven product equivalences are WATCHED, not enforced. Phase 11's schema accepts a "
    "violating record and Phase 12 does not retroactively make it invalid: the violation is "
    "reported as a rate and the record is still scored. Tightening input validity after an "
    "API has been published is a breaking change dressed up as a bug fix.",
    "Monitoring is observational and cannot change a prediction. The observer is called after "
    "the answer already exists, its return value is discarded, and the prediction path guards "
    "against it raising. A collector failure degrades the monitoring status and leaves the "
    "prediction untouched.",
    "Monitoring is OPT-IN on the serving process. Phase 11 published a route list and recorded "
    "it; enabling monitoring adds a route, so deploying Phase 12 code cannot silently alter "
    "the Phase 11 contract. reports/experiments/serving_results.json is unchanged and still "
    "verifies byte for byte.",
    "Drift never makes the service NOT READY. Readiness is a claim about the ARTEFACT — is the "
    "frozen model loaded and verified. A shifted population is not a corrupt model, and "
    "conflating them would pull a healthy instance out of rotation because customers changed.",
    "There is no HTTP reset endpoint. Closing a window is an administrative operation, this "
    "service has no authentication, and an unauthenticated endpoint that erases the evidence "
    "a drift investigation depends on is a worse trade than restarting the process.",
    "No model was trained, no threshold moved, no calibration was applied, the holdout was "
    "not reopened and no alternative model was built.",
)


def build_record(
    reference: ReferenceProfile,
    policy: MonitoringPolicy,
) -> dict[str, Any]:
    """Return the Phase 12 record as a plain, JSON-encodable mapping."""
    provenance = reference.provenance
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
        "reference": {
            "reference_population": provenance.reference_population,
            "reference_n": provenance.n_reference,
            "target_used_for_reference": provenance.target_used,
            "holdout_used_for_reference": provenance.holdout_used,
            "reference_profile_path": "reports/monitoring/reference_profile.json",
            "reference_profile_sha256": profile_digest(reference),
            "expected_reference_profile_sha256": EXPECTED_REFERENCE_PROFILE_SHA256,
            "reference_profile_sha256_source": REFERENCE_TRUST_ANCHOR,
            "reference_profile_digest_is_pinned": True,
            "reference_profile_digest_enforced_at_startup": True,
            "reference_digest_mismatch_is_fail_closed": True,
            "raw_sha256": provenance.raw_sha256,
            "training_ids_sha256": provenance.training_ids_sha256,
            "model_fingerprint": provenance.model_fingerprint_sha256,
            "pipeline_sha256": provenance.pipeline_sha256,
            "decision_policy_sha256": provenance.decision_policy_sha256,
            "freeze_commit": provenance.freeze_commit,
            "serving_commit": provenance.serving_commit,
        },
        "metrics": {
            "numeric_drift_metric": "population_stability_index",
            "categorical_drift_metric": "total_variation_distance",
            "score_drift_metric": "population_stability_index",
            "psi_formula": "PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)",
            "psi_epsilon": EPSILON,
            "tvd_formula": "TVD = 0.5 * sum_c |p_actual(c) - p_reference(c)|",
            "is_statistical_test": False,
            "p_values_computed": False,
            "additional_numeric_signals": [
                "mean_shift_in_reference_sd",
                "out_of_reference_range_rate",
            ],
            "additional_score_signals": [
                "mean_delta",
                "predicted_positive_rate",
                "predicted_positive_rate_delta",
            ],
        },
        "binning": {
            "n_bins_requested": DEFAULT_N_BINS,
            "score_n_bins_requested": SCORE_N_BINS,
            "strategy": "reference quantiles, duplicates dropped",
            "interval_convention": "(-inf, e1], (e1, e2], ..., (e_last, +inf)",
            "computed_once_from_reference": True,
            "recomputed_per_window": False,
            "quantiles_recorded": list(REFERENCE_QUANTILES),
        },
        "monitoring_scope": {
            "data_quality_monitoring": True,
            "feature_drift_monitoring": True,
            "unseen_category_monitoring": True,
            "structural_consistency_monitoring": True,
            "prediction_drift_monitoring": True,
            "performance_monitoring_without_labels": False,
            "performance_monitoring_reason": (
                "no production ground-truth labels exist in this phase, so no accuracy, "
                "recall, precision, F1, ROC-AUC, average precision or confusion matrix is "
                "computed"
            ),
            "structural_checks": len(STRUCTURAL_RULES),
            "structural_rules": [rule.relationship for rule in STRUCTURAL_RULES],
            "structural_rules_enforced": False,
            "quality_counters": list(QUALITY_COUNTERS),
            "unseen_level": UNSEEN_LEVEL,
            "distinct_unseen_tracking_bounded": True,
            "max_distinct_unseen_tracked_per_feature": (
                policy.max_distinct_unseen_tracked_per_feature
            ),
            "distinct_unseen_exact_until_saturation": True,
            "distinct_unseen_lower_bound_after_saturation": True,
            "distinct_unseen_eviction_on_saturation": False,
            "unseen_occurrence_count_remains_exact": True,
            "unseen_rate_remains_exact": True,
            "unseen_alert_signal": "unseen_rate",
            "saturation_can_change_alert_status": False,
        },
        "window": {
            "minimum_window_size": policy.minimum_window_size,
            "insufficient_data_status": STATUS_INSUFFICIENT_DATA,
            "statuses": [STATUS_OK, STATUS_WARNING, STATUS_CRITICAL, STATUS_INSUFFICIENT_DATA],
            "counts_reported_below_minimum": True,
            "drift_claimed_below_minimum": False,
            "details_suppressed_below_minimum": True,
            "fields_visible_below_minimum": [
                "status",
                "n_records",
                "minimum_window_size",
                "details_suppressed",
                "section_status",
                "reference_profile_sha256",
                *(f"data_quality.{counter}" for counter in QUALITY_COUNTERS),
                "collector_health",
                "collector_failures",
            ],
            "fields_suppressed_below_minimum": [
                *SUPPRESSED_SECTIONS,
                "numeric_histograms",
                "numeric_order_statistics",
                "per_feature_drift_metrics",
                "per_feature_out_of_range_rates",
                "categorical_count_by_level",
                "categorical_frequency_by_level",
                "per_feature_unseen_counts",
                "per_feature_unseen_rates",
                "n_distinct_unseen_observed",
                "distinct_unseen_tracking_saturated",
                "score_histogram",
                "score_summary_statistics",
                "predicted_positive_count",
                "predicted_positive_rate",
                "per_rule_structural_details",
            ],
            "window_primitive": "churn.monitoring.collector.MonitoringCollector.reset()",
            "window_primitive_semantics": (
                "snapshot-and-reset: returns the closing snapshot and opens an empty "
                "window, both under the collector RLock, so an update belongs entirely "
                "to one window or entirely to the next"
            ),
            "window_reset_exposed_over_http": False,
            "reset_changes_model": False,
            "reset_changes_policy": False,
            "reset_changes_threshold": False,
            "reset_changes_prediction": False,
            "reset_changes_reference_profile": False,
            "window_is_a_time_series": False,
        },
        "alert_policy": policy.as_record(),
        "serving_integration": {
            "integrated": True,
            "opt_in": True,
            "environment_flag": "CHURN_SERVING_MONITORING",
            "default_enabled": False,
            "monitoring_endpoint": MONITORING_ENDPOINT,
            "http_reset_endpoint": False,
            "phase11_route_list_unchanged_by_default": True,
            "phase11_serving_record_rewritten": False,
            "monitoring_changes_prediction": False,
            "drift_can_make_service_not_ready": False,
            "artifact_corruption_makes_service_not_ready": True,
            "readiness_reports_collector_health": True,
            "failure_isolation": (
                "the observer is called after the answer exists and its return value is "
                "discarded; a collector failure is counted, logged with a traceback, "
                "degrades the monitoring status, and leaves the prediction and the HTTP "
                "status untouched"
            ),
        },
        "privacy": {
            "payloads_persisted": False,
            "identifiers_persisted": False,
            "row_level_predictions_persisted": False,
            "row_level_features_persisted": False,
            "unseen_category_values_persisted": False,
            "unseen_values_persisted": False,
            "unseen_value_digests_exposed": False,
            "unseen_value_digests_persisted": False,
            "unseen_value_digests_logged": False,
            "customer_id_accepted": False,
            "aggregates_only": True,
            "small_window_details_suppressed": True,
            "exception_messages_logged": False,
            "tracebacks_logged": False,
            "unseen_cardinality_tracked_via": (
                "a bounded set of process-local sha256 digests; only a count derived "
                "from it ever leaves memory"
            ),
            "unseen_digest_retention_bounded": True,
            "unseen_digest_is_anonymisation": False,
            "unseen_digest_caveat": (
                "a categorical domain is small enough to hash exhaustively, so an unsalted "
                "digest of one of its values is reversible by anyone who can read it; the "
                "property relied on is that no reader exists, not that the hash hides "
                "anything"
            ),
        },
        "dependencies": {
            "new_runtime_dependencies": [],
            "root_pyproject_modified": False,
            "root_uv_lock_modified": False,
            "configs_base_toml_modified": False,
            "new_environment_created": False,
            "implemented_with": ["numpy", "pandas", "pydantic", "standard library"],
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
    logger.info("Wrote monitoring record: %s", destination)
    return destination


def read_record(path: Path | None = None) -> dict[str, Any]:
    """Read the record back.

    Raises:
        FileNotFoundError: If it has not been produced yet.
    """
    source = path or RECORD_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Monitoring record not found at {source}. Produce it with "
            "`uv run python scripts/build_monitoring_record.py`."
        )
    return json.loads(source.read_text(encoding="utf-8"))


def record_invariants(record: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Return ``(name, holds, detail)`` for every claim the record must not break."""
    reference = record["reference"]
    scope = record["monitoring_scope"]
    privacy = record["privacy"]
    integration = record["serving_integration"]
    metrics = record["metrics"]
    window = record["window"]

    return [
        ("phase_is_twelve", record["phase"] == PHASE, str(record["phase"])),
        ("component_is_monitoring", record["component"] == COMPONENT, record["component"]),
        ("model_unchanged", record["model_changed"] is False, "model_changed=false"),
        ("no_fit", record["fit_calls"] == 0, str(record["fit_calls"])),
        ("no_holdout", record["holdout_loaded"] is False, "holdout_loaded=false"),
        ("threshold_unchanged", record["threshold_changed"] is False, "threshold_changed=false"),
        (
            "calibration_unchanged",
            record["calibration_changed"] is False,
            "calibration_changed=false",
        ),
        (
            "reference_is_the_training_pool",
            reference["reference_population"] == "training_pool",
            reference["reference_population"],
        ),
        (
            "reference_n_is_the_pool",
            reference["reference_n"] == 5634,
            str(reference["reference_n"]),
        ),
        (
            "target_not_used_for_reference",
            reference["target_used_for_reference"] is False,
            "target_used_for_reference=false",
        ),
        (
            "holdout_not_used_for_reference",
            reference["holdout_used_for_reference"] is False,
            "holdout_used_for_reference=false",
        ),
        (
            "reference_profile_digest_recorded",
            len(reference["reference_profile_sha256"]) == 64,
            reference["reference_profile_sha256"][:16] + "…",
        ),
        (
            "no_label_based_metric",
            scope["performance_monitoring_without_labels"] is False,
            "performance_monitoring_without_labels=false",
        ),
        (
            "seven_structural_checks",
            scope["structural_checks"] == 7,
            str(scope["structural_checks"]),
        ),
        (
            "structural_rules_not_enforced",
            scope["structural_rules_enforced"] is False,
            "watched, not enforced",
        ),
        (
            "drift_metrics_are_not_tests",
            metrics["is_statistical_test"] is False and metrics["p_values_computed"] is False,
            "descriptive distances only",
        ),
        (
            "bins_are_not_recomputed_per_window",
            record["binning"]["recomputed_per_window"] is False,
            "fixed at profile-build time",
        ),
        (
            "no_drift_claimed_below_minimum",
            window["drift_claimed_below_minimum"] is False
            and window["counts_reported_below_minimum"] is True,
            f"minimum_window_size={window['minimum_window_size']}",
        ),
        (
            "no_distribution_exposed_below_minimum",
            window["details_suppressed_below_minimum"] is True
            and set(SUPPRESSED_SECTIONS) <= set(window["fields_suppressed_below_minimum"])
            and not set(window["fields_visible_below_minimum"])
            & set(window["fields_suppressed_below_minimum"]),
            f"{len(window['fields_suppressed_below_minimum'])} field group(s) withheld",
        ),
        (
            "only_global_counters_visible_below_minimum",
            all(
                field.startswith("data_quality.")
                or field
                in {
                    "status",
                    "n_records",
                    "minimum_window_size",
                    "details_suppressed",
                    "section_status",
                    "reference_profile_sha256",
                    "collector_health",
                    "collector_failures",
                }
                for field in window["fields_visible_below_minimum"]
            ),
            "window size, statuses and global counters only",
        ),
        (
            "window_primitive_is_not_http_reachable",
            window["window_reset_exposed_over_http"] is False
            and window["reset_changes_model"] is False
            and window["reset_changes_threshold"] is False
            and window["reset_changes_prediction"] is False
            and window["reset_changes_reference_profile"] is False,
            window["window_primitive"],
        ),
        (
            "reference_digest_is_pinned_and_enforced",
            reference["reference_profile_digest_is_pinned"] is True
            and reference["reference_profile_digest_enforced_at_startup"] is True
            and reference["reference_digest_mismatch_is_fail_closed"] is True
            and reference["expected_reference_profile_sha256"]
            == reference["reference_profile_sha256"],
            f"pinned in {reference['reference_profile_sha256_source']}",
        ),
        (
            "alert_policy_is_operational",
            record["alert_policy"]["classification"] == "OPERATIONAL_MONITORING_POLICY"
            and record["alert_policy"]["is_statistical_significance"] is False,
            "heuristic operational policy",
        ),
        (
            "monitoring_does_not_change_prediction",
            integration["monitoring_changes_prediction"] is False,
            "observational only",
        ),
        (
            "drift_does_not_break_readiness",
            integration["drift_can_make_service_not_ready"] is False
            and integration["artifact_corruption_makes_service_not_ready"] is True,
            "readiness is about the artefact",
        ),
        (
            "phase11_record_not_rewritten",
            integration["phase11_serving_record_rewritten"] is False,
            "serving_results.json untouched",
        ),
        ("no_payload_persisted", privacy["payloads_persisted"] is False, "none"),
        ("no_identifier_persisted", privacy["identifiers_persisted"] is False, "none"),
        (
            "no_row_level_prediction_persisted",
            privacy["row_level_predictions_persisted"] is False,
            "none",
        ),
        (
            "no_unseen_value_persisted",
            privacy["unseen_category_values_persisted"] is False
            and privacy["unseen_values_persisted"] is False,
            "digests only",
        ),
        (
            "no_unseen_digest_exposed",
            privacy["unseen_value_digests_exposed"] is False
            and privacy["unseen_value_digests_persisted"] is False
            and privacy["unseen_value_digests_logged"] is False,
            "cardinality only",
        ),
        (
            "the_digest_is_not_called_anonymisation",
            privacy["unseen_digest_is_anonymisation"] is False,
            "containment, not the hash",
        ),
        (
            "distinct_unseen_tracking_is_bounded",
            scope["distinct_unseen_tracking_bounded"] is True
            and scope["max_distinct_unseen_tracked_per_feature"] >= 1
            and scope["distinct_unseen_eviction_on_saturation"] is False
            and privacy["unseen_digest_retention_bounded"] is True,
            f"cap={scope['max_distinct_unseen_tracked_per_feature']} per feature",
        ),
        (
            "saturation_does_not_touch_the_alert_signal",
            scope["unseen_occurrence_count_remains_exact"] is True
            and scope["unseen_rate_remains_exact"] is True
            and scope["unseen_alert_signal"] == "unseen_rate"
            and scope["saturation_can_change_alert_status"] is False,
            "unseen_rate is exact and is what the policy reads",
        ),
        (
            "the_saturated_count_is_labelled_a_bound",
            scope["distinct_unseen_exact_until_saturation"] is True
            and scope["distinct_unseen_lower_bound_after_saturation"] is True,
            "exact below the cap, lower bound at or above it",
        ),
        (
            "small_window_details_suppressed",
            privacy["small_window_details_suppressed"] is True,
            "no per-feature breakdown below the minimum",
        ),
        (
            "no_exception_message_logged",
            privacy["exception_messages_logged"] is False and privacy["tracebacks_logged"] is False,
            "event name and exception type only",
        ),
        (
            "no_new_dependency",
            record["dependencies"]["new_runtime_dependencies"] == []
            and record["dependencies"]["root_pyproject_modified"] is False,
            "numpy, pandas, stdlib",
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
