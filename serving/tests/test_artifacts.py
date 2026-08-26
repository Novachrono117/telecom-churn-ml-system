"""The startup gates, and the fail-closed behaviour they exist to produce.

Each test corrupts exactly one thing and asserts which gate noticed. A gate that
cannot be provoked is a gate nobody has evidence for.

Nothing in this module rebuilds an artefact, and every test asserts that the
service did not either: a serving process that can regenerate its own model can
also regenerate a different one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from churn.modeling.freeze import file_digest, load_pipeline, model_fingerprint
from churn.modeling.freeze_results import DecisionPolicy, load_decision_policy
from churn.preprocessing.contracts import FEATURE_COLUMNS
from churn.serving.artifacts import (
    DECISION_POLICY_SHA256,
    FREEZE_COMMIT,
    SERVING_VERSION,
    load_frozen_artifacts,
    pipeline_checks,
    policy_checks,
    resolve_pipeline_path,
)
from churn.serving.errors import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    RuntimeCompatibilityError,
    ServingStartupError,
)
from churn.serving.service import ChurnInferenceService
from churn.serving.settings import ServingSettings
from conftest import dump_pipeline_copy, read_policy_json, write_policy_json

FROZEN_THRESHOLD = 0.3272694566222328
FROZEN_FINGERPRINT = "a57568e18ec0333e1c85b590f98e75da93ebbfafff063860ec5daf4e460d939f"
FROZEN_PIPELINE_SHA = "574fde36c6e2e991de7c3dfdab21981dc41eae2810d003ecbfb179dd504dc3d8"


def failed_gates(error: BaseException) -> str:
    return str(error)


# --------------------------------------------------------------------------- #
# The happy path: what a verified startup actually established.
# --------------------------------------------------------------------------- #


def test_startup_loads_the_frozen_model_and_every_gate_passes(
    service: ChurnInferenceService,
) -> None:
    artifacts = service.artifacts

    assert all(check.passed for check in artifacts.checks)
    assert artifacts.model_fingerprint == FROZEN_FINGERPRINT
    assert artifacts.pipeline_sha256 == FROZEN_PIPELINE_SHA
    assert artifacts.policy_sha256 == DECISION_POLICY_SHA256
    assert artifacts.threshold == FROZEN_THRESHOLD
    assert artifacts.comparison == ">="
    assert artifacts.calibration_policy == "NONE"
    assert artifacts.threshold_policy == "F1_MAXIMIZATION"
    assert artifacts.positive_class_label == 1
    assert artifacts.n_features == 19
    assert artifacts.feature_columns == FEATURE_COLUMNS


def test_the_gate_names_cover_every_documented_failure_mode(
    service: ChurnInferenceService,
) -> None:
    names = {check.name for check in service.artifacts.checks}

    assert {
        "decision_policy_digest_matches",
        "threshold_is_finite_and_a_probability",
        "decision_comparison_is_greater_or_equal",
        "calibration_policy_is_the_frozen_one",
        "policy_positive_class_is_unambiguous",
        "feature_contract_matches_the_installed_contract",
        "prepare_features_source_unchanged",
        "pipeline_file_digest_matches",
        "model_fingerprint_matches",
        "positive_class_column_resolves_from_classes_",
        "transformed_feature_names_match",
        "loaded_classifier_has_no_calibration_wrapper",
    } <= names


def test_the_model_version_is_not_the_serving_version(service: ChurnInferenceService) -> None:
    """Phase 11 versions the application. The model is still the one frozen in 9C."""
    artifacts = service.artifacts

    assert artifacts.freeze_commit == FREEZE_COMMIT
    assert len(FREEZE_COMMIT) == 40
    assert artifacts.serving_version == SERVING_VERSION
    assert artifacts.freeze_commit != artifacts.serving_version


def test_the_pipeline_path_comes_from_trusted_configuration(
    settings: ServingSettings, service: ChurnInferenceService, tmp_path: Path
) -> None:
    """Never from a request. There is no code path from a payload to a filename."""
    policy = service.artifacts.policy

    assert resolve_pipeline_path(policy, settings).is_file()
    override = ServingSettings(policy_path=settings.policy_path, pipeline_path=tmp_path / "x")
    assert resolve_pipeline_path(policy, override) == tmp_path / "x"


# --------------------------------------------------------------------------- #
# Fail-closed: absent artefacts.
# --------------------------------------------------------------------------- #


def test_an_absent_policy_stops_the_startup_and_builds_nothing(tmp_path: Path) -> None:
    missing = tmp_path / "decision_policy.json"

    with pytest.raises(ArtifactNotFoundError):
        load_frozen_artifacts(ServingSettings(policy_path=missing))

    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_an_absent_pipeline_stops_the_startup_and_builds_nothing(
    frozen_copy: tuple[Path, Path], tmp_path: Path
) -> None:
    policy_copy, pipeline_copy = frozen_copy
    pipeline_copy.unlink()
    settings = ServingSettings(policy_path=policy_copy, pipeline_path=pipeline_copy)

    with pytest.raises(ArtifactNotFoundError) as error:
        load_frozen_artifacts(settings)

    assert "does not rebuild it" in str(error.value)
    assert not pipeline_copy.exists()


def test_a_startup_failure_is_never_a_repair_trigger(frozen_copy: tuple[Path, Path]) -> None:
    """Every startup error is a ServingStartupError. None of them has a repair path."""
    policy_copy, pipeline_copy = frozen_copy
    pipeline_copy.unlink()

    with pytest.raises(ServingStartupError):
        load_frozen_artifacts(ServingSettings(policy_path=policy_copy, pipeline_path=pipeline_copy))


# --------------------------------------------------------------------------- #
# Fail-closed: tampered artefacts.
# --------------------------------------------------------------------------- #


def test_a_tampered_pipeline_file_is_refused(frozen_copy: tuple[Path, Path]) -> None:
    policy_copy, pipeline_copy = frozen_copy
    pipeline_copy.write_bytes(pipeline_copy.read_bytes() + b"\x00")

    with pytest.raises(ArtifactIntegrityError) as error:
        load_frozen_artifacts(ServingSettings(policy_path=policy_copy, pipeline_path=pipeline_copy))

    assert "pipeline_file_digest_matches" in failed_gates(error.value)


def test_a_tampered_policy_threshold_is_refused(frozen_copy: tuple[Path, Path]) -> None:
    """The policy is not its own only authority; its bytes are pinned too."""
    policy_copy, pipeline_copy = frozen_copy
    payload = read_policy_json(policy_copy)
    payload["threshold"]["final_threshold"] = 0.5
    write_policy_json(policy_copy, payload)

    with pytest.raises(ArtifactIntegrityError) as error:
        load_frozen_artifacts(ServingSettings(policy_path=policy_copy, pipeline_path=pipeline_copy))

    assert "decision_policy_digest_matches" in failed_gates(error.value)


def test_a_tampered_policy_calibration_is_refused(frozen_copy: tuple[Path, Path]) -> None:
    policy_copy, pipeline_copy = frozen_copy
    payload = read_policy_json(policy_copy)
    payload["calibration"]["calibration_policy"] = "C1"
    write_policy_json(policy_copy, payload)

    with pytest.raises(ArtifactIntegrityError):
        load_frozen_artifacts(ServingSettings(policy_path=policy_copy, pipeline_path=pipeline_copy))


def test_a_file_that_is_not_a_decision_policy_is_refused(tmp_path: Path) -> None:
    impostor = tmp_path / "decision_policy.json"
    impostor.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError) as error:
        load_frozen_artifacts(ServingSettings(policy_path=impostor))

    assert "does not parse" in str(error.value)


def test_an_incompatible_runtime_stops_the_startup(
    settings: ServingSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from churn.serving import runtime as runtime_module

    monkeypatch.setattr(
        runtime_module,
        "runtime_versions",
        lambda: {
            "python": "3.9",
            "scikit_learn": "1.0.0",
            "numpy": "1.0.0",
            "pandas": "1.0.0",
            "joblib": "1.0.0",
        },
    )

    with pytest.raises(RuntimeCompatibilityError) as error:
        load_frozen_artifacts(settings)

    assert "scikit_learn" in str(error.value)


# --------------------------------------------------------------------------- #
# The gates in isolation.
# --------------------------------------------------------------------------- #


@pytest.fixture
def policy(policy_path: Path) -> DecisionPolicy:
    return load_decision_policy(policy_path)


def test_every_policy_gate_passes_on_the_real_freeze(policy: DecisionPolicy) -> None:
    assert all(check.passed for check in policy_checks(policy, DECISION_POLICY_SHA256))


def test_a_threshold_outside_zero_one_is_refused(policy: DecisionPolicy) -> None:
    broken = policy.model_copy(
        update={"threshold": policy.threshold.model_copy(update={"final_threshold": 1.5})}
    )

    checks = {check.name: check.passed for check in policy_checks(broken, DECISION_POLICY_SHA256)}

    assert checks["threshold_is_finite_and_a_probability"] is False


def test_a_greater_than_comparison_is_refused(policy: DecisionPolicy) -> None:
    """`>` and `>=` differ on exactly the customers sitting at the threshold."""
    broken = policy.model_copy(
        update={"decision_rule": policy.decision_rule.model_copy(update={"comparison": ">"})}
    )

    checks = {check.name: check.passed for check in policy_checks(broken, DECISION_POLICY_SHA256)}

    assert checks["decision_comparison_is_greater_or_equal"] is False


def test_a_positive_class_that_does_not_match_its_column_is_refused(
    policy: DecisionPolicy,
) -> None:
    broken = policy.model_copy(
        update={
            "decision_rule": policy.decision_rule.model_copy(update={"positive_class_column": 0})
        }
    )

    checks = {check.name: check.passed for check in policy_checks(broken, DECISION_POLICY_SHA256)}

    assert checks["policy_positive_class_is_unambiguous"] is False


def test_a_feature_contract_that_is_not_the_installed_one_is_refused(
    policy: DecisionPolicy,
) -> None:
    contract = policy.feature_contract
    broken = policy.model_copy(
        update={
            "feature_contract": contract.model_copy(
                update={"categorical": contract.categorical[:-1], "n_features": 18}
            )
        }
    )

    checks = {check.name: check.passed for check in policy_checks(broken, DECISION_POLICY_SHA256)}

    assert checks["feature_contract_matches_the_installed_contract"] is False


def test_an_engineered_feature_in_the_contract_is_refused(policy: DecisionPolicy) -> None:
    contract = policy.feature_contract
    broken = policy.model_copy(
        update={"feature_contract": contract.model_copy(update={"engineered": ["tenure_bucket"]})}
    )

    checks = {check.name: check.passed for check in policy_checks(broken, DECISION_POLICY_SHA256)}

    assert checks["no_engineered_feature_in_the_contract"] is False


def test_a_moved_prepare_features_is_refused(policy: DecisionPolicy) -> None:
    """The gate that covers the part of inference the model fingerprints cannot see."""
    contract = policy.inference_contract
    broken = policy.model_copy(
        update={
            "inference_contract": contract.model_copy(
                update={"prepare_features_source_sha256": "0" * 64}
            )
        }
    )

    checks = {check.name: check.passed for check in policy_checks(broken, DECISION_POLICY_SHA256)}

    assert checks["prepare_features_source_unchanged"] is False


def test_a_policy_read_from_other_bytes_is_refused(policy: DecisionPolicy) -> None:
    checks = {check.name: check.passed for check in policy_checks(policy, "0" * 64)}

    assert checks["decision_policy_digest_matches"] is False


def test_every_pipeline_gate_passes_on_the_real_artefact(
    pipeline_path: Path, policy: DecisionPolicy
) -> None:
    checks, pipeline, file_sha, fingerprint = pipeline_checks(pipeline_path, policy)

    assert all(check.passed for check in checks)
    assert pipeline is not None
    assert file_sha == FROZEN_PIPELINE_SHA
    assert fingerprint == FROZEN_FINGERPRINT


def test_a_swapped_model_is_caught_by_the_fingerprint_even_when_the_bytes_agree(
    tmp_path: Path, pipeline_path: Path, policy: DecisionPolicy
) -> None:
    """The strong version of the integrity story.

    A byte digest only proves the file is the file somebody recorded. Here an
    attacker-equivalent scenario is built: the coefficients are altered, the file
    is re-serialised, and the *policy is updated to accept the new bytes*. The
    file-digest gate passes — and the model fingerprint, which describes what the
    model does rather than how it was stored, still refuses it.
    """
    swapped = load_pipeline(pipeline_path)
    classifier = swapped.named_steps["classifier"]
    classifier.coef_ = np.asarray(classifier.coef_, dtype=float) + 0.1

    swapped_path = dump_pipeline_copy(swapped, tmp_path / "swapped.joblib")
    complicit = policy.model_copy(
        update={
            "artifacts": policy.artifacts.model_copy(
                update={"pipeline_sha256": file_digest(swapped_path)}
            )
        }
    )

    checks = {check.name: check.passed for check in pipeline_checks(swapped_path, complicit)[0]}

    assert checks["pipeline_file_digest_matches"] is True
    assert checks["model_fingerprint_matches"] is False
    assert model_fingerprint(swapped) != FROZEN_FINGERPRINT


def test_a_divergent_recorded_fingerprint_is_refused(
    pipeline_path: Path, policy: DecisionPolicy
) -> None:
    broken = policy.model_copy(
        update={
            "artifacts": policy.artifacts.model_copy(update={"model_fingerprint_sha256": "0" * 64})
        }
    )

    checks = {check.name: check.passed for check in pipeline_checks(pipeline_path, broken)[0]}

    assert checks["model_fingerprint_matches"] is False


def test_the_loaded_classifier_is_the_estimator_the_policy_names(
    pipeline_path: Path, policy: DecisionPolicy
) -> None:
    """A calibration wrapper would show up here: the frozen policy is NONE."""
    broken = policy.model_copy(
        update={"model": policy.model.model_copy(update={"estimator": "CalibratedClassifierCV"})}
    )

    checks = {check.name: check.passed for check in pipeline_checks(pipeline_path, broken)[0]}

    assert checks["loaded_classifier_has_no_calibration_wrapper"] is False


def test_all_failures_are_reported_at_once(frozen_copy: tuple[Path, Path]) -> None:
    """Three broken things should be one error, not three restarts."""
    policy_copy, pipeline_copy = frozen_copy
    payload = read_policy_json(policy_copy)
    payload["threshold"]["final_threshold"] = 1.5
    payload["decision_rule"]["comparison"] = ">"
    write_policy_json(policy_copy, payload)

    with pytest.raises(ArtifactIntegrityError) as error:
        load_frozen_artifacts(ServingSettings(policy_path=policy_copy, pipeline_path=pipeline_copy))

    message = failed_gates(error.value)
    assert "decision_policy_digest_matches" in message
    assert "threshold_is_finite_and_a_probability" in message
    assert "decision_comparison_is_greater_or_equal" in message
