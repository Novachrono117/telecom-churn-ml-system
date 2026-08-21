"""Phase 9C: the freeze, its fingerprints, and the integrity primitives.

The property this phase depends on is not "a file was written" — it is that the
object loaded later is provably the object frozen earlier, and that the rule
applied to it is unambiguous. Both are asserted here on real behaviour: a
round-trip through disk, a fingerprint that moves when a coefficient moves, and a
decision rule tested at the boundary where ``>`` and ``>=`` disagree.

The round-trip tests fit on **synthetic fixtures**, so they need no data file and
can never reach a partition they should not. The policy tests read the real
artefacts and skip when the freeze has not been produced yet. Nothing here reads
a holdout row.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import io
import json
import pathlib
import re

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import PROJECT_ROOT
from churn.modeling.calibration_results import load_results as load_calibration_results
from churn.modeling.freeze import (
    CONFIGURATION_FILES,
    DECISION_COMPARISON,
    INFERENCE_SOURCE_FILES,
    KNOWN_CALIBRATION_POLICIES,
    KNOWN_THRESHOLD_POLICIES,
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
    SERIALIZATION_COMPRESS,
    SERIALIZATION_PROTOCOL,
    FrozenModelError,
    IntegrityError,
    apply_decision_rule,
    check_decision_rule,
    check_pipeline_artefact,
    check_policy_vocabulary,
    check_source_provenance,
    check_threshold,
    dump_pipeline,
    file_digest,
    fit_final_pipeline,
    function_source_digest,
    load_pipeline,
    model_fingerprint,
    model_state,
    positive_class_column,
    positive_probability,
    raise_on_failure,
    source_digest,
    text_digest,
    transformed_feature_names,
)
from churn.modeling.freeze_results import (
    POLICY_NAME,
    SCHEMA_VERSION,
    UPSTREAM_ARTEFACTS,
    DecisionPolicy,
    load_decision_policy,
    verify_decision_policy,
)
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.modeling.threshold_results import load_results as load_threshold_results
from churn.modeling.tuning import build_modern_logistic
from churn.preprocessing.contracts import FEATURE_COLUMNS, build_feature_matrix, prepare_features
from churn.preprocessing.target import encode_target


def _load_script(name: str):
    """Import a file in ``scripts/`` as a module. It is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: The orchestration script, imported so its ``verify`` entry point can be
#: exercised directly instead of only through a subprocess.
freeze_model = _load_script("freeze_model")


@pytest.fixture(scope="module")
def synthetic_xy(comparison_frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """A synthetic feature matrix and target. No data file is read."""
    return (
        build_feature_matrix(comparison_frame),
        np.asarray(encode_target(comparison_frame["Churn"])),
    )


@pytest.fixture(scope="module")
def fitted(synthetic_xy: tuple[pd.DataFrame, np.ndarray]):
    """The frozen pipeline fitted on the synthetic fixture."""
    features, target = synthetic_xy
    return fit_final_pipeline(features, target)


# --- the final fit ------------------------------------------------------------


def test_the_final_fit_uses_only_the_data_it_is_handed(
    synthetic_xy: tuple[pd.DataFrame, np.ndarray],
) -> None:
    """``fit_final_pipeline`` loads nothing, so it cannot reach another partition."""
    features, target = synthetic_xy

    pipeline = fit_final_pipeline(features, target)

    assert pipeline.named_steps[CLASSIFIER_STEP].n_features_in_ == len(
        transformed_feature_names(pipeline)
    )
    # The scaler saw exactly the rows it was handed, and no others: a fit that
    # had reached another partition would have seen more.
    assert model_state(pipeline)["scaler"]["n_samples_seen"] == len(features)


def test_the_fitted_pipeline_is_the_frozen_candidate(fitted) -> None:
    classifier = fitted.named_steps[CLASSIFIER_STEP]

    assert list(fitted.named_steps) == [PREPROCESSOR_STEP, CLASSIFIER_STEP]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.get_params()["C"] == build_modern_logistic().get_params()["C"]
    assert classifier.class_weight is None
    assert not hasattr(classifier, "calibrated_classifiers_")


def test_the_fit_is_deterministic(synthetic_xy: tuple[pd.DataFrame, np.ndarray]) -> None:
    """Two fits on the same data produce bit-identical learned parameters."""
    features, target = synthetic_xy

    first = model_fingerprint(fit_final_pipeline(features, target))
    second = model_fingerprint(fit_final_pipeline(features, target))

    assert first == second


# --- classes and the positive-class column ------------------------------------


def test_the_estimator_classes_are_the_encoded_binary_target(fitted) -> None:
    classes = fitted.named_steps[CLASSIFIER_STEP].classes_.tolist()

    assert classes == [NEGATIVE_CLASS_LABEL, POSITIVE_CLASS_LABEL]


def test_the_positive_class_column_is_resolved_from_classes(fitted) -> None:
    column = positive_class_column(fitted)
    classes = fitted.named_steps[CLASSIFIER_STEP].classes_

    assert classes[column] == POSITIVE_CLASS_LABEL


def test_the_positive_probability_is_the_column_classes_points_at(
    fitted, synthetic_xy: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, _ = synthetic_xy
    column = positive_class_column(fitted)

    probability = positive_probability(fitted, features)

    assert np.array_equal(probability, fitted.predict_proba(features)[:, column])
    assert probability.min() >= 0.0 and probability.max() <= 1.0


def test_an_unidentifiable_positive_class_is_refused(fitted) -> None:
    """A model whose classes do not contain the positive label cannot decide."""
    import copy

    broken = copy.deepcopy(fitted)
    broken.named_steps[CLASSIFIER_STEP].classes_ = np.array([7, 9])

    with pytest.raises(FrozenModelError, match="not identifiable"):
        positive_class_column(broken)


# --- the decision rule --------------------------------------------------------


def test_the_rule_is_greater_or_equal_at_the_boundary() -> None:
    """A probability exactly at the threshold is POSITIVE. ``>`` would differ here."""
    probability = np.array([0.3272694566222327, 0.3272694566222328, 0.3272694566222329])

    decisions = apply_decision_rule(probability, 0.3272694566222328)

    assert decisions.tolist() == [0, 1, 1]
    assert DECISION_COMPARISON == ">="


def test_the_rule_agrees_with_the_plain_comparison(
    fitted, synthetic_xy: tuple[pd.DataFrame, np.ndarray]
) -> None:
    features, _ = synthetic_xy
    probability = positive_probability(fitted, features)

    for threshold in (0.1, 0.3272694566222328, 0.5, 0.9):
        assert np.array_equal(
            apply_decision_rule(probability, threshold), (probability >= threshold).astype(int)
        )


def test_the_decision_rule_check_rejects_a_strict_comparison() -> None:
    checks = {check.name: check for check in check_decision_rule(">", 1, [0, 1])}

    assert checks["decision_comparison_is_greater_or_equal"].passed is False


def test_the_decision_rule_check_rejects_the_wrong_positive_column() -> None:
    checks = {check.name: check for check in check_decision_rule(">=", 0, [0, 1])}

    assert checks["positive_class_column_is_unambiguous"].passed is False


# --- serialisation round trip -------------------------------------------------


def test_a_dumped_pipeline_can_be_read_back(fitted, tmp_path) -> None:
    path = dump_pipeline(fitted, tmp_path / "pipeline.joblib")

    reloaded = load_pipeline(path)

    assert path.is_file()
    assert list(reloaded.named_steps) == [PREPROCESSOR_STEP, CLASSIFIER_STEP]


def test_probabilities_survive_serialisation_exactly(
    fitted, synthetic_xy: tuple[pd.DataFrame, np.ndarray], tmp_path
) -> None:
    features, _ = synthetic_xy
    before = positive_probability(fitted, features)

    reloaded = load_pipeline(dump_pipeline(fitted, tmp_path / "pipeline.joblib"))
    after = positive_probability(reloaded, features)

    assert np.array_equal(before, after)


def test_the_reloaded_pipeline_decides_identically(
    fitted, synthetic_xy: tuple[pd.DataFrame, np.ndarray], tmp_path
) -> None:
    features, _ = synthetic_xy
    threshold = 0.3272694566222328
    reloaded = load_pipeline(dump_pipeline(fitted, tmp_path / "pipeline.joblib"))

    before = apply_decision_rule(positive_probability(fitted, features), threshold)
    after = apply_decision_rule(positive_probability(reloaded, features), threshold)

    assert np.array_equal(before, after)
    assert model_fingerprint(fitted) == model_fingerprint(reloaded)


def test_serialisation_settings_are_pinned(fitted, tmp_path) -> None:
    """Repeated dumps of the same object are byte-identical under the pinned settings."""
    first = dump_pipeline(fitted, tmp_path / "a.joblib")
    second = dump_pipeline(fitted, tmp_path / "b.joblib")

    assert file_digest(first) == file_digest(second)
    assert SERIALIZATION_PROTOCOL == 5
    assert SERIALIZATION_COMPRESS == 0


def test_loading_an_absent_pipeline_names_the_command_that_rebuilds_it(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="freeze_model.py"):
        load_pipeline(tmp_path / "absent.joblib")


# --- fingerprints -------------------------------------------------------------


def test_the_model_fingerprint_moves_when_a_coefficient_moves(fitted) -> None:
    import copy

    before = model_fingerprint(fitted)
    tampered = copy.deepcopy(fitted)
    tampered.named_steps[CLASSIFIER_STEP].coef_[0, 0] += 1e-12

    assert model_fingerprint(tampered) != before


def test_the_model_fingerprint_moves_when_the_scaler_moves(fitted) -> None:
    import copy

    before = model_fingerprint(fitted)
    tampered = copy.deepcopy(fitted)
    tampered.named_steps[PREPROCESSOR_STEP].named_steps["encode"].named_transformers_[
        "numeric"
    ].mean_[0] += 1e-9

    assert model_fingerprint(tampered) != before


def test_the_file_digest_detects_a_corrupted_artefact(fitted, tmp_path) -> None:
    """One flipped byte, and the file check fails while the policy is unchanged."""
    path = dump_pipeline(fitted, tmp_path / "pipeline.joblib")
    expected_file = file_digest(path)
    expected_model = model_fingerprint(fitted)

    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(bytes(payload))

    checks, _ = check_pipeline_artefact(path, expected_file, expected_model)
    named = {check.name: check for check in checks}

    assert named["pipeline_file_digest_matches"].passed is False


def test_a_swapped_model_fails_the_fingerprint_check(
    fitted, synthetic_xy: tuple[pd.DataFrame, np.ndarray], tmp_path
) -> None:
    """A different fitted model, correctly serialised, is still not the frozen one."""
    features, target = synthetic_xy
    expected_model = model_fingerprint(fitted)

    other = fit_final_pipeline(features.iloc[: len(features) // 2], target[: len(target) // 2])
    path = dump_pipeline(other, tmp_path / "other.joblib")

    checks, _ = check_pipeline_artefact(path, file_digest(path), expected_model)
    named = {check.name: check for check in checks}

    assert named["pipeline_file_digest_matches"].passed is True
    assert named["model_fingerprint_matches"].passed is False


def test_a_missing_artefact_is_a_failed_check_not_a_crash(tmp_path) -> None:
    checks, pipeline = check_pipeline_artefact(tmp_path / "absent.joblib", "a" * 64, "b" * 64)

    assert pipeline is None
    assert checks[0].name == "pipeline_file_present"
    assert checks[0].passed is False


def test_an_unfitted_or_foreign_object_cannot_be_fingerprinted(tmp_path) -> None:
    path = tmp_path / "foreign.joblib"
    joblib.dump({"not": "a pipeline"}, path)

    checks, _ = check_pipeline_artefact(path, file_digest(path), "b" * 64)
    named = {check.name: check for check in checks}

    assert named["pipeline_has_the_frozen_structure"].passed is False


def test_the_canonical_encoding_does_not_sort_away_order() -> None:
    """Reordering the features must change the fingerprint, not be normalised away."""
    assert text_digest(["a", "b"]) != text_digest(["b", "a"])


# --- integrity primitives -----------------------------------------------------


def test_threshold_checks_accept_the_frozen_value() -> None:
    checks = {c.name: c for c in check_threshold(0.3272694566222328, 0.3272694566222328)}

    assert all(check.passed for check in checks.values())


@pytest.mark.parametrize(
    ("value", "failing"),
    [
        (None, "threshold_present"),
        ("0.32", "threshold_is_numeric"),
        (float("nan"), "threshold_is_finite_and_a_probability"),
        (1.5, "threshold_is_finite_and_a_probability"),
        (0.5, "threshold_matches_phase9b_exactly"),
    ],
)
def test_threshold_checks_reject_a_broken_value(value: object, failing: str) -> None:
    checks = {c.name: c for c in check_threshold(value, 0.3272694566222328)}

    assert checks[failing].passed is False


def test_a_boolean_is_not_accepted_as_a_threshold() -> None:
    """``True`` is an ``int`` in Python; a policy carrying it is corrupt, not valid."""
    checks = {c.name: c for c in check_threshold(True, 0.3272694566222328)}

    assert checks["threshold_is_numeric"].passed is False


def test_policy_vocabulary_rejects_free_text() -> None:
    checks = {c.name: c for c in check_policy_vocabulary("sigmoid-ish", "whatever")}

    assert checks["calibration_policy_recognised"].passed is False
    assert checks["threshold_policy_recognised"].passed is False


def test_raise_on_failure_lists_every_failure() -> None:
    checks = check_threshold(0.5, 0.3272694566222328) + check_policy_vocabulary("X", "Y")

    with pytest.raises(IntegrityError) as error:
        raise_on_failure(checks)

    message = str(error.value)
    assert "threshold_matches_phase9b_exactly" in message
    assert "calibration_policy_recognised" in message
    assert "threshold_policy_recognised" in message


def test_raise_on_failure_is_transparent_when_everything_passes() -> None:
    checks = check_threshold(0.3272694566222328, 0.3272694566222328)

    assert raise_on_failure(checks) == checks


# --- the generated policy -----------------------------------------------------


def _policy_path():
    return PROJECT_ROOT / "reports" / "decision_policy.json"


def _policy() -> DecisionPolicy:
    if not _policy_path().is_file():
        pytest.skip("the freeze has not been produced yet")
    return load_decision_policy()


def test_the_policy_has_a_valid_schema() -> None:
    policy = _policy()

    assert policy.schema_version == SCHEMA_VERSION
    assert policy.policy == POLICY_NAME
    assert policy.frozen_at_phase == "9C"
    # Re-validating the raw bytes proves the file itself parses, not only an
    # object that happened to be constructed in memory.
    assert DecisionPolicy.model_validate_json(_policy_path().read_text(encoding="utf-8"))


def test_the_policy_preserves_the_threshold_without_rounding() -> None:
    policy = _policy()
    raw = _policy_path().read_text(encoding="utf-8")

    assert policy.threshold.final_threshold == 0.3272694566222328
    assert policy.threshold.rounded is False
    assert '"final_threshold": 0.3272694566222328' in raw


def test_the_policy_freezes_the_phase9a_and_phase9b_decisions() -> None:
    policy = _policy()

    assert policy.calibration.calibration_policy == "NONE"
    assert policy.calibration.calibrated is False
    assert policy.threshold.threshold_policy == "F1_MAXIMIZATION"
    assert policy.calibration.calibration_policy in KNOWN_CALIBRATION_POLICIES
    assert policy.threshold.threshold_policy in KNOWN_THRESHOLD_POLICIES


def test_the_policy_states_the_rule_unambiguously() -> None:
    policy = _policy()
    rule = policy.decision_rule

    assert rule.comparison == ">="
    assert rule.classes == [0, 1]
    assert rule.positive_class_label == 1
    assert rule.positive_class_column == 1
    assert rule.classes[rule.positive_class_column] == rule.positive_class_label
    assert "Churn" in rule.positive_class_meaning


def test_the_policy_freezes_the_model_and_its_representation() -> None:
    policy = _policy()

    assert policy.model.estimator == "LogisticRegression"
    assert policy.model.reopened_here is False
    assert policy.model.hyperparameters == {
        "C": 1.0,
        "class_weight": None,
        "l1_ratio": 0.0,
        "max_iter": 100,
        "solver": "lbfgs",
    }
    assert policy.feature_contract.n_features == len(FEATURE_COLUMNS) == 19
    assert len(policy.feature_contract.numeric) == 3
    assert len(policy.feature_contract.categorical) == 16
    assert policy.feature_contract.engineered == []
    assert policy.feature_contract.feature_names_sha256 == text_digest(list(FEATURE_COLUMNS))


def test_the_policy_records_the_prepare_features_boundary() -> None:
    policy = _policy()

    assert policy.feature_contract.entry_point.endswith("prepare_features")
    assert "prepare_features" in policy.decision_rule.probability_expression


def test_the_policy_was_fitted_on_the_whole_training_pool() -> None:
    policy = _policy()

    assert policy.model.fitted_on["partition"] == "training pool"
    assert policy.model.fitted_on["fraction_of_pool_used"] == 1.0
    assert policy.model.fitted_on["n_rows"] == 5634
    assert policy.model.convergence["converged"] is True


def test_the_policy_carries_no_timestamp_and_no_holdout_quantity() -> None:
    raw = _policy_path().read_text(encoding="utf-8") if _policy_path().is_file() else None
    if raw is None:
        pytest.skip("the freeze has not been produced yet")
    payload = json.loads(raw)

    assert not [key for key in payload if re.search("time|stamp|generated", key, re.I)]
    assert not re.search(r"\d{4}-\d{2}-\d{2}", raw)
    assert payload["holdout_touched"] is False
    assert payload["reopened_decisions"] == []


def test_the_full_verification_passes_against_the_real_artefacts() -> None:
    """The freeze on disk passes every check the --verify command runs."""
    policy = _policy()

    checks, pipeline = verify_decision_policy(
        policy, load_threshold_results(), load_calibration_results()
    )
    failed = [check.name for check in checks if not check.passed]

    assert not failed, f"failed checks: {failed}"
    assert pipeline is not None
    assert len(checks) >= 20


def test_the_frozen_pipeline_on_disk_reproduces_its_fingerprints() -> None:
    policy = _policy()
    path = PROJECT_ROOT / policy.artifacts.pipeline_path
    if not path.is_file():
        pytest.skip("the pipeline artefact has not been produced yet")

    pipeline = load_pipeline(path)

    assert file_digest(path) == policy.artifacts.pipeline_sha256
    assert model_fingerprint(pipeline) == policy.artifacts.model_fingerprint_sha256
    assert policy.artifacts.authoritative_fingerprint == "model_fingerprint_sha256"
    assert policy.artifacts.versioned_in_git is False


def test_earlier_phase_artefacts_are_unchanged() -> None:
    """Every upstream digest recorded in the freeze still matches the file."""
    policy = _policy()
    recorded = policy.provenance["upstream_artefact_digests"]

    assert set(recorded) == set(UPSTREAM_ARTEFACTS)
    for artefact, digest in recorded.items():
        assert file_digest(PROJECT_ROOT / artefact) == digest, (
            f"{artefact} changed since the freeze"
        )


# --- the inference boundary: source provenance --------------------------------


@pytest.fixture
def sandbox(tmp_path):
    """A throw-away copy of everything the freeze fingerprints, plus the freeze.

    Every provenance test mutates **this** tree, never the repository. The real
    files are read once, copied, and then left alone.
    """
    policy = _policy()
    pipeline_source = PROJECT_ROOT / policy.artifacts.pipeline_path
    if not pipeline_source.is_file():
        pytest.skip("the pipeline artefact has not been produced yet")

    for relative in (
        *policy.code_provenance.source_digests,
        *policy.code_provenance.configuration_digests,
        policy.artifacts.pipeline_path,
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())

    policy_path = tmp_path / "reports" / "decision_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_bytes((PROJECT_ROOT / "reports" / "decision_policy.json").read_bytes())
    return tmp_path


def _provenance_checks(policy: DecisionPolicy, root, prepare_digest: str | None = None):
    return {
        check.name: check
        for check in check_source_provenance(
            policy.code_provenance.source_digests,
            policy.code_provenance.configuration_digests,
            prepare_digest or policy.inference_contract.prepare_features_source_sha256,
            root,
        )
    }


def test_the_policy_freezes_the_inference_boundary() -> None:
    policy = _policy()
    contract = policy.inference_contract

    assert contract.entry_point == "churn.preprocessing.contracts.prepare_features"
    assert contract.prepare_features_source_sha256 == function_source_digest(prepare_features)
    assert len(contract.prepare_features_source_sha256) == 64
    assert "prepare_features" in contract.chain[1]
    assert str(policy.threshold.final_threshold) in contract.chain[-1]


def test_every_file_that_can_change_a_prediction_is_fingerprinted() -> None:
    policy = _policy()
    frozen = set(policy.code_provenance.source_digests)

    assert frozen == set(INFERENCE_SOURCE_FILES)
    assert set(policy.code_provenance.configuration_digests) == set(CONFIGURATION_FILES)
    # The class the pickle binds by reference, and the code outside the pipeline.
    assert "src/churn/preprocessing/transformers.py" in frozen
    assert "src/churn/preprocessing/contracts.py" in frozen
    assert "src/churn/modeling/tuning.py" in frozen
    assert "src/churn/modeling/freeze.py" in frozen
    assert "uv.lock" in policy.code_provenance.configuration_digests
    # Every frozen path carries a stated reason.
    for path in frozen | set(policy.code_provenance.configuration_digests):
        assert policy.code_provenance.why_each_file[path]


def test_the_recorded_source_digests_match_the_repository() -> None:
    policy = _policy()

    for path, digest in policy.code_provenance.source_digests.items():
        assert source_digest(PROJECT_ROOT / path) == digest, f"{path} moved since the freeze"
    for path, digest in policy.code_provenance.configuration_digests.items():
        assert source_digest(PROJECT_ROOT / path) == digest, f"{path} moved since the freeze"


def test_provenance_passes_on_an_unmodified_copy(sandbox) -> None:
    checks = _provenance_checks(_policy(), sandbox)

    assert all(check.passed for check in checks.values()), {
        name: check.detail for name, check in checks.items() if not check.passed
    }


def test_a_changed_prepare_features_fingerprint_fails_verification(sandbox) -> None:
    """The recorded digest no longer matches the function that would run."""
    checks = _provenance_checks(_policy(), sandbox, prepare_digest="0" * 64)

    assert checks["prepare_features_source_unchanged"].passed is False


@pytest.mark.parametrize(
    "target",
    [
        "src/churn/preprocessing/contracts.py",
        "src/churn/preprocessing/transformers.py",
        "src/churn/modeling/tuning.py",
        "src/churn/modeling/freeze.py",
    ],
)
def test_a_changed_inference_source_file_fails_verification(sandbox, target: str) -> None:
    """One edited character in any of them, and the freeze stops verifying."""
    edited = sandbox / target
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")

    checks = _provenance_checks(_policy(), sandbox)

    assert checks["inference_source_modules_unchanged"].passed is False
    assert target in checks["inference_source_modules_unchanged"].detail


def test_a_deleted_inference_source_file_fails_verification(sandbox) -> None:
    (sandbox / "src/churn/preprocessing/transformers.py").unlink()

    checks = _provenance_checks(_policy(), sandbox)

    assert checks["inference_source_modules_unchanged"].passed is False
    assert "missing" in checks["inference_source_modules_unchanged"].detail


@pytest.mark.parametrize("target", ["configs/base.toml", "pyproject.toml", "uv.lock"])
def test_a_changed_configuration_or_lock_fails_verification(sandbox, target: str) -> None:
    edited = sandbox / target
    edited.write_text(edited.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")

    checks = _provenance_checks(_policy(), sandbox)

    assert checks["configuration_and_dependency_pins_unchanged"].passed is False
    assert target in checks["configuration_and_dependency_pins_unchanged"].detail


def test_the_source_digest_ignores_line_endings_only() -> None:
    """Newlines are normalised; nothing else is."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        base = pathlib.Path(directory)
        lf, crlf, changed = base / "a.py", base / "b.py", base / "c.py"
        lf.write_bytes(b"x = 1\ny = 2\n")
        crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
        changed.write_bytes(b"x = 1\ny = 3\n")

        assert source_digest(lf) == source_digest(crlf)
        assert source_digest(lf) != source_digest(changed)
        # A byte hash would have failed the first assertion.
        assert file_digest(lf) != file_digest(crlf)


def test_a_docstring_edit_also_invalidates_the_freeze(sandbox) -> None:
    """The documented cost of whole-file hashing, asserted rather than assumed."""
    target = sandbox / "src/churn/preprocessing/contracts.py"
    target.write_text('"""Edited docstring."""\n' + target.read_text(encoding="utf-8"), "utf-8")

    checks = _provenance_checks(_policy(), sandbox)

    assert checks["inference_source_modules_unchanged"].passed is False


# --- verify is fail-closed ----------------------------------------------------


def test_verify_succeeds_on_an_intact_sandbox(sandbox) -> None:
    assert freeze_model.verify(sandbox / "reports" / "decision_policy.json", sandbox) == 0


def test_verify_fails_when_the_pipeline_is_absent(sandbox) -> None:
    """Fail-closed: a missing model is an error, never a trigger to rebuild it."""
    pipeline = sandbox / _policy().artifacts.pipeline_path
    pipeline.unlink()

    exit_code = freeze_model.verify(sandbox / "reports" / "decision_policy.json", sandbox)

    assert exit_code == 1
    assert not pipeline.exists(), "--verify recreated the pipeline it was meant to check"


def test_verify_fails_when_the_pipeline_is_tampered_with(sandbox) -> None:
    pipeline = sandbox / _policy().artifacts.pipeline_path
    payload = bytearray(pipeline.read_bytes())
    payload[-1] ^= 0x01
    pipeline.write_bytes(bytes(payload))
    before = file_digest(pipeline)

    exit_code = freeze_model.verify(sandbox / "reports" / "decision_policy.json", sandbox)

    assert exit_code == 1
    assert file_digest(pipeline) == before, "--verify repaired the artefact instead of failing"


def test_verify_fails_when_a_source_file_moved(sandbox) -> None:
    target = sandbox / "src/churn/preprocessing/transformers.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    policy_path = sandbox / "reports" / "decision_policy.json"
    before = file_digest(policy_path)

    exit_code = freeze_model.verify(policy_path, sandbox)

    assert exit_code == 1
    assert file_digest(policy_path) == before, "--verify refreshed the policy instead of failing"


def test_verify_fails_when_the_policy_is_absent(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="freeze_model.py"):
        freeze_model.verify(tmp_path / "absent.json", tmp_path)


def test_the_command_reports_a_missing_policy_as_a_nonzero_exit(monkeypatch, tmp_path) -> None:
    """`main` turns the fail-closed error into an exit code, not a rebuild."""
    monkeypatch.setattr(freeze_model, "verify", lambda: (_ for _ in ()).throw(FileNotFoundError()))

    assert freeze_model.main(["--verify"]) == 1


# --- verify is read-only ------------------------------------------------------


#: Everything `--verify` must leave untouched: the freeze itself plus every
#: upstream artefact it is anchored to.
def _freeze_artefacts() -> list[pathlib.Path]:
    policy = _policy()
    return [
        PROJECT_ROOT / "reports" / "decision_policy.json",
        PROJECT_ROOT / "reports" / "model_freeze_report.md",
        PROJECT_ROOT / policy.artifacts.pipeline_path,
        *(PROJECT_ROOT / path for path in UPSTREAM_ARTEFACTS),
    ]


def _fingerprints() -> dict[str, tuple[str, int, int]]:
    return {
        str(path): (file_digest(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in _freeze_artefacts()
        if path.is_file()
    }


def test_verify_changes_no_byte_and_no_mtime_of_any_artefact() -> None:
    """SHA-256, size and mtime of every artefact, before and after a real run."""
    if not (PROJECT_ROOT / "reports" / "decision_policy.json").is_file():
        pytest.skip("the freeze has not been produced yet")

    before = _fingerprints()
    assert len(before) >= 10

    exit_code = freeze_model.main(["--verify"])

    after = _fingerprints()
    assert exit_code == 0
    assert set(before) == set(after), "--verify added or removed a file"
    for path, values in before.items():
        assert after[path] == values, f"--verify modified {path}"


def test_verify_never_fits_dumps_or_writes(monkeypatch) -> None:
    """Writes and fits raise; `--verify` still runs to completion.

    The strongest of the three kinds of evidence: it does not observe that
    nothing changed, it makes changing anything impossible and shows the command
    never tried.
    """
    if not (PROJECT_ROOT / "reports" / "decision_policy.json").is_file():
        pytest.skip("the freeze has not been produced yet")

    attempted: list[str] = []

    def forbid(name: str):
        def guard(*args, **kwargs):
            attempted.append(name)
            raise AssertionError(f"--verify called {name}")

        return guard

    monkeypatch.setattr(Pipeline, "fit", forbid("Pipeline.fit"))
    monkeypatch.setattr(LogisticRegression, "fit", forbid("LogisticRegression.fit"))
    monkeypatch.setattr(joblib, "dump", forbid("joblib.dump"))
    monkeypatch.setattr(pathlib.Path, "write_text", forbid("Path.write_text"))
    monkeypatch.setattr(pathlib.Path, "write_bytes", forbid("Path.write_bytes"))
    monkeypatch.setattr(freeze_model, "dump_pipeline", forbid("dump_pipeline"))
    monkeypatch.setattr(freeze_model, "write_decision_policy", forbid("write_decision_policy"))
    monkeypatch.setattr(freeze_model, "fit_final_pipeline", forbid("fit_final_pipeline"))

    # Opening anything under the repository for writing is recorded rather than
    # raised: pytest's own machinery may legitimately open files mid-test, and a
    # blanket raise would prove nothing about this command.
    opened_for_writing: list[str] = []
    real_open = builtins.open
    real_io_open = io.open

    def watch(original):
        def guarded(file, mode="r", *args, **kwargs):
            if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
                try:
                    resolved = pathlib.Path(file).resolve()
                except (TypeError, ValueError, OSError):
                    resolved = None
                if resolved is not None and resolved.is_relative_to(PROJECT_ROOT):
                    opened_for_writing.append(str(resolved))
            return original(file, mode, *args, **kwargs)

        return guarded

    monkeypatch.setattr(builtins, "open", watch(real_open))
    monkeypatch.setattr(io, "open", watch(real_io_open))

    exit_code = freeze_model.main(["--verify"])

    assert exit_code == 0
    assert attempted == [], f"--verify attempted: {attempted}"
    assert opened_for_writing == [], f"--verify opened for writing: {opened_for_writing}"


def test_the_verify_function_references_no_write_or_fit_operation() -> None:
    """A narrow AST audit of one function, as the third kind of evidence."""
    tree = ast.parse((PROJECT_ROOT / "scripts" / "freeze_model.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "verify"
    )

    used: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)

    forbidden = {
        "fit",
        "fit_final_pipeline",
        "dump",
        "dump_pipeline",
        "joblib",
        "write_text",
        "write_bytes",
        "write_decision_policy",
        "build_decision_policy",
        "build_report",
        "freeze",
        "mkdir",
        "unlink",
        "open",
    }
    found = sorted(used & forbidden)
    assert not found, f"verify() references {found}"


def test_the_freeze_and_verify_paths_are_separate_functions() -> None:
    """`--verify` must not be able to fall through into the build path."""
    tree = ast.parse((PROJECT_ROOT / "scripts" / "freeze_model.py").read_text(encoding="utf-8"))
    main_function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    # The verify branch returns; it never continues into the writing code below.
    verify_calls = [
        node
        for node in ast.walk(main_function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "verify"
    ]
    assert len(verify_calls) == 1, "the --verify branch must return immediately"


# --- audits -------------------------------------------------------------------

_PHASE9C_SOURCES = [
    PROJECT_ROOT / "src" / "churn" / "modeling" / "freeze.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "freeze_results.py",
    PROJECT_ROOT / "scripts" / "freeze_model.py",
]

FORBIDDEN_IDENTIFIERS = (
    "TunedThresholdClassifierCV",
    "CalibratedClassifierCV",
    "GridSearchCV",
    "RandomizedSearchCV",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "build_hist_gradient_boosting",
    "build_native_pipeline",
    "build_calibrated_estimator",
    "select_f1_threshold",
    "run_threshold_nested_cv",
    "CONTRACT_TENURE",
    "XGBClassifier",
    "LGBMClassifier",
    "optuna",
)


def _identifiers(source) -> set[str]:
    used: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.alias):
            used.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                used.add(node.asname)
    return used


@pytest.mark.parametrize("source", _PHASE9C_SOURCES, ids=lambda path: path.name)
def test_phase9c_code_never_reaches_the_holdout(source) -> None:
    used = _identifiers(source)

    assert "load_holdout" not in used, f"{source.name} imports or calls the holdout loader"
    assert "holdout" not in used, f"{source.name} reads a holdout partition"


@pytest.mark.parametrize("source", _PHASE9C_SOURCES, ids=lambda path: path.name)
def test_phase9c_code_stays_inside_its_scope(source) -> None:
    used = _identifiers(source)

    found = sorted(name for name in FORBIDDEN_IDENTIFIERS if name in used)
    assert not found, (
        f"{source.name} references {found}: this phase decides nothing — no model comparison, no "
        "tuning, no calibration, no threshold search, no engineered feature"
    )


@pytest.mark.parametrize("source", _PHASE9C_SOURCES, ids=lambda path: path.name)
def test_no_phase9c_call_sets_class_weight(source) -> None:
    tree = ast.parse(source.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "class_weight":
                continue
            assert isinstance(keyword.value, ast.Constant) and keyword.value.value is None, (
                f"{source.name} passes a non-None class_weight; it is frozen at None"
            )


@pytest.mark.parametrize("source", _PHASE9C_SOURCES, ids=lambda path: path.name)
def test_only_the_script_loads_data_and_only_the_training_pool(source) -> None:
    used = _identifiers(source)

    if source.name == "freeze_model.py":
        assert "load_training_pool" in used
        assert "load_split" not in used
        return
    assert "load_training_pool" not in used
    assert "load_raw_typed" not in used


@pytest.mark.parametrize("source", _PHASE9C_SOURCES, ids=lambda path: path.name)
def test_no_metric_is_computed_anywhere_in_this_phase(source) -> None:
    """A freeze that scored something could let a number leak into a decision."""
    used = _identifiers(source)

    scoring = sorted(
        name
        for name in (
            "accuracy_score",
            "f1_score",
            "precision_score",
            "recall_score",
            "roc_auc_score",
            "average_precision_score",
            "brier_score_loss",
            "log_loss",
            "confusion_matrix",
            "compute_metrics",
            "cross_val_score",
            "cross_validate",
        )
        if name in used
    )
    assert not scoring, f"{source.name} computes {scoring}: this phase evaluates nothing"


def test_the_report_generator_reads_no_clock() -> None:
    used = _identifiers(PROJECT_ROOT / "scripts" / "freeze_model.py")

    found = sorted(
        name
        for name in ("datetime", "date", "today", "now", "utcnow", "monotonic", "perf_counter")
        if name in used
    )
    assert not found, f"freeze_model.py reads wall-clock state via {found}"


def test_the_report_states_the_holdout_is_untouched() -> None:
    report = PROJECT_ROOT / "reports" / "model_freeze_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")

    text = report.read_text(encoding="utf-8")

    assert "holdout_touched: false" in text
    assert "No holdout/test performance has been measured at this stage." in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
