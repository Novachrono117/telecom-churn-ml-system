"""The Phase 11 machine-readable record: what the serving boundary is, as data.

Built from the *running* system rather than transcribed by hand. Every fingerprint,
threshold, gate name and endpoint path in the record is read from the artefacts the
service actually loaded and from the application it actually builds, so the file
cannot describe a boundary that does not exist.

**Deterministic by construction.** No timestamp, no duration, no hostname, no run
identifier. Regenerating it on an unchanged repository reproduces the file byte for
byte, which is what makes a diff of it mean something. That convention is the
repository's, and it is why ``--verify`` can compare the file with a fresh build.

**Claims are not self-certified.** Every prohibition the record asserts —
``fit_calls = 0``, ``holdout_loaded = false``, ``pipeline_predict_used = false`` —
names the test that establishes it under ``verified_by``. A record that merely
asserted them would be a statement of intent; this one is an index into evidence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from churn.config import PROJECT_ROOT
from churn.serving.api import API_PREFIX, create_app
from churn.serving.artifacts import (
    DECISION_POLICY_SHA256,
    FREEZE_COMMIT,
    FREEZE_COMMIT_SHORT,
    FREEZE_PHASE,
    SERVING_VERSION,
)
from churn.serving.errors import ERROR_CODES
from churn.serving.provenance import (
    SOURCE_ROLES,
    runtime_effective_roles,
    training_only_roles,
)
from churn.serving.runtime import PINNED_LIBRARY_KEYS, PYTHON_KEY
from churn.serving.service import ChurnInferenceService
from churn.serving.settings import (
    DEFAULT_MAX_BATCH_SIZE,
    ENV_PREFIX,
    FORBIDDEN_ENV_VARS,
    MAX_ALLOWED_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

RECORD_PATH = PROJECT_ROOT / "reports" / "experiments" / "serving_results.json"

SCHEMA_VERSION = 1
EXPERIMENT = "phase11-production-inference-api"
COMPONENT = "PRODUCTION_INFERENCE_API"
PHASE = 11

#: Where each prohibition is actually established, as ``(claim, module, test)``.
#: The record indexes evidence; it does not stand in for it.
_EVIDENCE: tuple[tuple[str, str, str], ...] = (
    (
        "model_changed",
        "test_artifacts",
        "test_startup_loads_the_frozen_model_and_every_gate_passes",
    ),
    ("fit_calls", "test_isolation", "test_serving_a_request_calls_no_fit_and_loads_no_dataset"),
    ("holdout_loaded", "test_isolation", "test_the_holdout_is_never_loaded"),
    (
        "training_data_loaded",
        "test_isolation",
        "test_serving_a_request_calls_no_fit_and_loads_no_dataset",
    ),
    (
        "prepare_features_required",
        "test_isolation",
        "test_the_pipeline_only_ever_receives_prepared_features",
    ),
    ("predict_proba_used", "test_isolation", "test_predict_proba_is_what_produces_the_answer"),
    (
        "pipeline_predict_used",
        "test_static_audit",
        "test_no_forbidden_call_is_written_in_the_serving_source",
    ),
    (
        "artifact_validation_on_startup",
        "test_artifacts",
        "test_the_gate_names_cover_every_documented_failure_mode",
    ),
    ("fail_closed", "test_artifacts", "test_a_tampered_pipeline_file_is_refused"),
    ("payloads_persisted", "test_isolation", "test_serving_writes_nothing_to_the_repository"),
    (
        "identifiers_persisted",
        "test_api",
        "test_the_identifier_and_the_target_are_rejected_not_ignored",
    ),
    (
        "threshold_source",
        "test_settings",
        "test_a_frozen_decision_cannot_be_overridden_from_the_environment",
    ),
    (
        "single_batch_equivalence",
        "test_service",
        "test_batch_and_single_are_exactly_equal_in_every_field",
    ),
    (
        "no_upstream_artefact_written",
        "test_isolation",
        "test_serving_creates_no_file_anywhere_in_reports_or_artifacts",
    ),
    ("runtime_compatibility", "test_runtime", "test_any_divergent_component_fails_closed"),
    (
        "runtime_source_provenance_validated",
        "test_provenance",
        "test_a_divergent_runtime_source_fails_the_startup",
    ),
    (
        "single_batch_probability_contract",
        "test_service",
        "test_the_probability_is_bit_identical_at_every_batch_size",
    ),
    (
        "canonical_scoring_strategy",
        "test_service",
        "test_position_inside_the_batch_does_not_move_a_probability",
    ),
    (
        "openapi_contract",
        "test_openapi",
        "test_the_request_declares_exactly_the_nineteen_contracted_features",
    ),
)

#: ``{claim: "serving/tests/<module>.py::<test>"}``.
VERIFIED_BY: dict[str, str] = {
    claim: f"serving/tests/{module}.py::{test}" for claim, module, test in _EVIDENCE
}

NOTES: tuple[str, ...] = (
    "Phase 11 versions the SERVING APPLICATION. It does not version the model. The model is "
    f"the object frozen in Phase 9C at commit {FREEZE_COMMIT_SHORT}, and nothing in this phase "
    "refits, reselects, recalibrates or rethresholds it. model_changed is false and is checked, "
    "not assumed.",
    "FastAPI and Uvicorn are NOT declared in the root project, and were not added to it. "
    "pyproject.toml and uv.lock are part of the Phase 9C provenance: their digests are recorded "
    "in the decision policy and `scripts/freeze_model.py --verify` fails if either moves. The "
    "serving dependencies live in an isolated uv project under serving/, which depends on the "
    "root project as an editable path dependency. Both root manifests are byte-identical to "
    "commit 47d6dd3.",
    "The startup gate pins the SHA-256 of reports/decision_policy.json itself. Without that pin "
    "the policy would be its own only authority, and a hand-edited threshold would pass every "
    "gate that asks whether the policy agrees with itself. The pinned digest is not invented "
    "here: Phases 9D, 9E and 10 each independently recorded the digest of the policy they read, "
    "and all three agree.",
    "The runtime versions are checked at startup and a mismatch FAILS the startup rather than "
    "warning. A joblib artefact pickles its classes by reference, so a different scikit-learn, "
    "numpy or pandas can change what the loaded object computes while every stored fingerprint "
    "stays identical.",
    "prepare_features runs BEFORE the persisted pipeline and is therefore outside everything the "
    "model's own fingerprints can see. Its source digest is checked at startup, and a test "
    "asserts the pipeline is never handed anything that did not pass through it.",
    "The decision comes exclusively from probability >= threshold. pipeline.predict is never "
    "called: it would apply scikit-learn's own 0.5 cut, which is not the frozen policy. The "
    "positive-class column is resolved from classes_, never assumed to be column 1.",
    "The batch limit is an operational bound on the REQUEST and changes no prediction. It is "
    "configured in the serving layer only; configs/base.toml was not touched, because it belongs "
    "to the frozen provenance.",
    "There is no environment variable for the threshold, the calibration policy or the positive "
    "class, and setting one is REFUSED at startup rather than ignored. Ignoring it would leave an "
    "operator believing a decision boundary had moved when it had not.",
    "Single and batch scoring share one code path: predict_one is predict_batch of length one. "
    "The decision for a record is therefore identical across the two endpoints. The PROBABILITY "
    "can differ in the final bit: the classifier's decision function is a matrix product, and "
    "BLAS dispatches a 1x46 product to a different kernel than an Nx46 one, summing the same 46 "
    "terms in a different order. Measured at most 1 ULP (1.11e-16 absolute); everything before "
    "the classifier is bit-identical. The probability is deliberately NOT rounded to hide this: "
    "Phase 9C stored the threshold unrounded because a value rounded in the sixth decimal can "
    "move customers across the boundary, and rounding the score would be that mistake mirrored.",
    "The calibration policy is NONE, so the score is the model's raw output and is not a "
    "calibrated probability. The OpenAPI description says so explicitly rather than letting a "
    "reader assume otherwise.",
    "No monitoring, drift metric, baseline distribution or alert rule is implemented here. That "
    "is Phase 12. The unseen-category test in this phase proves only that a new category is not "
    "BLOCKED; it says nothing about whether the resulting score should be trusted.",
    "No Dockerfile and no frontend were produced. Containerisation and any visual interface "
    "belong to later phases; the local run command is documented instead.",
)

_UNSEEN_CATEGORY_NOTE = "OneHotEncoder(handle_unknown='ignore'), unchanged from Phase 4"
_BLANK_POLICY_NOTE = (
    "blank, empty and whitespace-only values are rejected by the frozen feature contract"
)
_TOTAL_CHARGES_NOTE = (
    "blank with tenure == 0 is a structural zero and is accepted; blank with tenure > 0 and any "
    "unreadable value raise DataQualityError. The rule is the frozen preprocessing's, not "
    "re-implemented at the HTTP layer"
)
_ISOLATION_REASON = (
    "FastAPI and Uvicorn are absent from the root project, and the root pyproject.toml and "
    "uv.lock are part of the Phase 9C provenance: editing either would fail "
    "scripts/freeze_model.py --verify"
)
_POLICY_ARTEFACT = "reports/decision_policy.json"


def _endpoints() -> dict[str, Any]:
    """Read the published routes from the application, not from a list kept by hand."""
    app = create_app()
    return {
        "api_prefix": API_PREFIX,
        "single_endpoint": f"{API_PREFIX}/predict",
        "batch_endpoint": f"{API_PREFIX}/predict/batch",
        "metadata_endpoint": f"{API_PREFIX}/model",
        "health_endpoints": ["/health/live", "/health/ready"],
        "openapi_endpoint": "/openapi.json",
        "published_paths": sorted(app.openapi()["paths"]),
    }


def build_record(service: ChurnInferenceService) -> dict[str, Any]:
    """Return the Phase 11 record as a plain, JSON-encodable mapping."""
    artifacts = service.artifacts
    identity = service.identity()
    policy = artifacts.policy

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "component": COMPONENT,
        "serving_version": SERVING_VERSION,
        "model_changed": False,
        "fit_calls": 0,
        "holdout_loaded": False,
        "training_data_loaded": False,
        "raw_dataset_loaded": False,
        "frozen_model": {
            "freeze_phase": FREEZE_PHASE,
            "freeze_commit": FREEZE_COMMIT,
            "freeze_commit_short": FREEZE_COMMIT_SHORT,
            "estimator": identity.estimator,
            "model_fingerprint": artifacts.model_fingerprint,
            "pipeline_sha256": artifacts.pipeline_sha256,
            "decision_policy_sha256": artifacts.policy_sha256,
            "decision_policy_sha256_pinned_in_serving": DECISION_POLICY_SHA256,
            "policy_digest_pinned": True,
            "n_features": artifacts.n_features,
            "n_transformed_features": artifacts.n_transformed_features,
        },
        "decision_policy": {
            "threshold": artifacts.threshold,
            "comparison": artifacts.comparison,
            "threshold_policy": artifacts.threshold_policy,
            "threshold_source": _POLICY_ARTEFACT,
            "calibration_policy": artifacts.calibration_policy,
            "calibration_source": _POLICY_ARTEFACT,
            "calibrated": policy.calibration.calibrated,
            "positive_class": artifacts.positive_class_label,
            "positive_class_column": artifacts.positive_class_column,
            "positive_class_resolved_from": "pipeline.classes_",
            "positive_class_meaning": identity.positive_class_meaning,
            "boundary_note": policy.decision_rule.boundary_note,
        },
        "inference_chain": {
            "prepare_features_required": True,
            "prepare_features_entry_point": policy.inference_contract.entry_point,
            "predict_proba_used": True,
            "pipeline_predict_used": False,
            "chain": list(policy.inference_contract.chain),
        },
        "feature_contract": {
            "n_features": artifacts.n_features,
            "feature_names": list(artifacts.feature_columns),
            "identifier_accepted": False,
            "target_accepted": False,
            "engineered_features_accepted": False,
            "extra_fields": "forbid",
            "categorical_enums_closed": False,
            "numeric_range_constraints_declared": False,
            "unseen_category_handling": _UNSEEN_CATEGORY_NOTE,
            "blank_policy": _BLANK_POLICY_NOTE,
            "total_charges_rule": _TOTAL_CHARGES_NOTE,
        },
        "api": _endpoints(),
        "operational_limits": {
            "default_max_batch_size": DEFAULT_MAX_BATCH_SIZE,
            "max_allowed_batch_size": MAX_ALLOWED_BATCH_SIZE,
            "configured_max_batch_size": service.max_batch_size,
            "limit_changes_predictions": False,
            "env_prefix": ENV_PREFIX,
            "forbidden_env_vars": list(FORBIDDEN_ENV_VARS),
        },
        "startup": {
            "artifact_validation_on_startup": True,
            "fail_closed": True,
            "rebuilds_artefacts_on_failure": False,
            "model_loaded_once_per_process": True,
            "runtime_compatibility_checked": True,
            "n_gates": len(artifacts.checks),
            "gates": [check.name for check in artifacts.checks],
        },
        "runtime": {
            "recorded_by_freeze": dict(policy.environment),
            "installed": dict(artifacts.runtime),
            "python_compared_at": "major.minor",
            "libraries_compared_exactly": list(PINNED_LIBRARY_KEYS),
            "python_key": PYTHON_KEY,
        },
        "environment_isolation": {
            "isolated_serving_environment_required": True,
            "reason": _ISOLATION_REASON,
            "serving_project": "serving/pyproject.toml",
            "serving_lock": "serving/uv.lock",
            "root_project_dependency": "editable path dependency on the repository root",
            "root_manifests_modified": False,
            "configs_base_toml_modified": False,
            "ml_libraries_pinned_exactly": list(PINNED_LIBRARY_KEYS),
        },
        "privacy": {
            "payloads_persisted": False,
            "identifiers_persisted": False,
            "predictions_persisted": False,
            "payload_logged": False,
            "probability_logged": False,
            "customer_id_accepted": False,
            "logged_fields": ["method", "path", "status_code", "latency_ms", "batch_size"],
        },
        "security": {
            "extra_fields_forbidden": True,
            "batch_bounded": True,
            "model_path_from_request": False,
            "pickle_path_from_request": False,
            "traceback_exposed": False,
            "eval_or_exec_used": False,
            "dynamic_import_from_request": False,
            "error_codes": sorted(ERROR_CODES),
        },
        "determinism": {
            "response_fields_are_pure_functions_of_input_and_artefacts": True,
            "timestamp_in_response": False,
            "request_id_in_response": False,
            "single_batch_decision_contract": "exact",
            "single_batch_probability_contract": "exact",
            "canonical_scoring_strategy": "row-wise",
            "batch_shape_affects_probability": False,
            "probability_rounded": False,
            "superseded_implementation": {
                "strategy": "vectorised N-row matrix",
                "observed_max_ulps": 1,
                "observed_max_absolute": 1.1102230246251565e-16,
                "cause": "BLAS dispatches a 1x46 product (GEMV) to a different kernel "
                "than an Nx46 one (GEMM); the same 46 terms accumulate in a different "
                "order and floating-point addition is not associative",
                "preprocessing_was_bit_identical": True,
                "why_replaced": "a caller must not need to know how a request was "
                "batched in order to reproduce a probability",
            },
        },
        "runtime_source_provenance": {
            "runtime_source_provenance_validated": True,
            "enforced_at": "startup",
            "fail_closed": True,
            "runtime_source_digest_mismatches": 0,
            "digest_definition": "SHA-256 of the module source, newlines normalised to "
            "\n — the same function the freeze recorded them with",
            "source_resolution": "sys.modules[<module>].__file__; no absolute repository "
            "path is assumed, stored or compared",
            "recorded_source_files": sorted(role.path for role in SOURCE_ROLES),
            "runtime_effective_source_files": [role.path for role in runtime_effective_roles()],
            "training_only_source_files": [role.path for role in training_only_roles()],
            "classification": [
                {
                    "path": role.path,
                    "module": role.module,
                    "runtime_effective": role.runtime_effective,
                    "evidence": role.evidence,
                    "reason": role.reason,
                }
                for role in SOURCE_ROLES
            ],
            "why_not_gate_everything": "three recorded files are never called and have "
            "none of their values read by this process; gating them would refuse "
            "startup over an edit that cannot reach a prediction",
            "note": "pickle integrity protects serialized state; source provenance "
            "additionally protects the behavior of custom Python code resolved at runtime",
        },
        "out_of_scope": {
            "monitoring_implemented": False,
            "drift_metrics_implemented": False,
            "frontend_implemented": False,
            "containerisation_implemented": False,
            "deferred_to": "Phase 12 for monitoring; later phases for deployment and any UI",
        },
        "verified_by": dict(VERIFIED_BY),
        "notes": list(NOTES),
    }


def write_record(record: dict[str, Any], path: Path | None = None) -> Path:
    """Write the record as indented JSON with a trailing newline and LF endings."""
    destination = path or RECORD_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote serving record: %s", destination)
    return destination


def read_record(path: Path | None = None) -> dict[str, Any]:
    """Read the record back.

    Raises:
        FileNotFoundError: If it has not been produced yet.
    """
    source = path or RECORD_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Serving record not found at {source}. Produce it with "
            "`uv run --project serving python scripts/build_serving_record.py`."
        )
    return json.loads(source.read_text(encoding="utf-8"))


def record_invariants(record: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """Return ``(name, holds, detail)`` for every claim the record must not break."""
    chain = record["inference_chain"]
    privacy = record["privacy"]
    startup = record["startup"]
    rule = record["decision_policy"]

    return [
        ("phase_is_eleven", record["phase"] == PHASE, str(record["phase"])),
        ("component_is_the_boundary", record["component"] == COMPONENT, record["component"]),
        ("model_unchanged", record["model_changed"] is False, "model_changed=false"),
        ("no_fit", record["fit_calls"] == 0, str(record["fit_calls"])),
        ("no_holdout", record["holdout_loaded"] is False, "holdout_loaded=false"),
        ("no_training_data", record["training_data_loaded"] is False, "training_data_loaded=false"),
        ("prepare_features_required", chain["prepare_features_required"] is True, "required"),
        ("predict_proba_used", chain["predict_proba_used"] is True, "used"),
        ("pipeline_predict_unused", chain["pipeline_predict_used"] is False, "unused"),
        (
            "threshold_from_policy",
            rule["threshold_source"] == _POLICY_ARTEFACT,
            _POLICY_ARTEFACT,
        ),
        (
            "calibration_from_policy",
            rule["calibration_source"] == _POLICY_ARTEFACT,
            _POLICY_ARTEFACT,
        ),
        ("comparison_is_greater_or_equal", rule["comparison"] == ">=", rule["comparison"]),
        ("positive_class_is_one", rule["positive_class"] == 1, str(rule["positive_class"])),
        (
            "positive_class_resolved",
            rule["positive_class_resolved_from"] == "pipeline.classes_",
            "classes_",
        ),
        ("calibration_is_none", rule["calibration_policy"] == "NONE", rule["calibration_policy"]),
        ("startup_validates", startup["artifact_validation_on_startup"] is True, "validated"),
        ("fail_closed", startup["fail_closed"] is True, "fail_closed"),
        ("never_rebuilds", startup["rebuilds_artefacts_on_failure"] is False, "never"),
        ("loaded_once_per_process", startup["model_loaded_once_per_process"] is True, "once"),
        ("no_payload_persisted", privacy["payloads_persisted"] is False, "none"),
        ("no_identifier_persisted", privacy["identifiers_persisted"] is False, "none"),
        ("identifier_rejected", privacy["customer_id_accepted"] is False, "rejected"),
        (
            "root_manifests_untouched",
            record["environment_isolation"]["root_manifests_modified"] is False,
            "untouched",
        ),
        (
            "monitoring_deferred",
            record["out_of_scope"]["monitoring_implemented"] is False,
            "Phase 12",
        ),
        ("nineteen_features", record["feature_contract"]["n_features"] == 19, "19"),
        ("every_claim_has_evidence", set(VERIFIED_BY) <= set(record["verified_by"]), "verified_by"),
    ]


__all__ = [
    "COMPONENT",
    "EXPERIMENT",
    "PHASE",
    "RECORD_PATH",
    "SCHEMA_VERSION",
    "VERIFIED_BY",
    "build_record",
    "read_record",
    "record_invariants",
    "write_record",
]
