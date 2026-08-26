"""Startup gate: locate, verify and load the frozen model — or refuse to serve.

This module is the only place in the serving layer that touches the filesystem,
and it runs **once per process**. Everything downstream receives an already
verified :class:`FrozenArtifacts` and never re-reads anything.

What it proves before the service is allowed to exist
-----------------------------------------------------
Every check below answers a way the process could otherwise serve predictions
from something that is not the frozen model:

* the policy parses into the Phase 9C schema, is the Phase 9C freeze, and has the
  bytes it was frozen with — without that last one the file would be its own only
  authority, and a hand-edited threshold would pass every gate that asks whether
  the policy agrees with itself;
* the runtime is the environment the model was produced in;
* the threshold, the comparison, the classes and the positive column are the
  frozen ones and are internally consistent;
* the pipeline file has the recorded bytes;
* the loaded object has the recorded **model fingerprint** — the authoritative
  identity, invariant to serialisation;
* the classifier carries no calibration wrapper, because the frozen calibration
  policy is ``NONE``;
* the installed feature contract is the frozen one, including the source of
  ``prepare_features`` itself;
* every source file the policy recorded that is actually **executed or read** by
  this process still has the digest the freeze recorded.

That last check deserves its own sentence. ``prepare_features`` runs *before* the
persisted pipeline, so it is outside everything the model's own fingerprints can
see: editing it would change the values handed to the model while the pipeline
digest, the model fingerprint, the feature names and the threshold all stayed
identical. Phase 9C recorded its source digest precisely so a later process could
detect that, and this is the later process.

What it deliberately does **not** do
------------------------------------
It never rebuilds anything. A missing pipeline is an error, not a trigger to run
``scripts/freeze_model.py``: the serving boundary does not train, and a service
that can regenerate its own model can also regenerate a *different* one.

It does not gate the *whole* of ``code_provenance``. Three of the eight recorded
files are never executed by this process, and refusing to start over an edit to a
training builder that cannot reach a prediction would cost availability and protect
nothing. Which files are gated, and on what traced evidence, is
:mod:`churn.serving.provenance`; the whole-repository check stays with
``scripts/freeze_model.py --verify``, which is a repository gate and runs where the
repository is.

The runtime-effective files ARE gated here, and that closes the gap the model's own
fingerprints cannot: the pipeline pickles its classes by reference, so an edit to
``TotalChargesCleaner.transform`` changes what the loaded object computes while
``pipeline_sha256`` and ``model_fingerprint_sha256`` both stay identical. Pickle
integrity protects serialised state; source provenance additionally protects the
behaviour of the custom Python code resolved at runtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from churn.config import PROJECT_ROOT
from churn.modeling.freeze import (
    DECISION_COMPARISON,
    KNOWN_CALIBRATION_POLICIES,
    KNOWN_THRESHOLD_POLICIES,
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
    PREPARE_FEATURES,
    PREPARE_FEATURES_QUALNAME,
    FrozenModelError,
    file_digest,
    function_source_digest,
    load_pipeline,
    model_fingerprint,
    positive_class_column,
    text_digest,
    transformed_feature_names,
)
from churn.modeling.freeze_results import POLICY_NAME, DecisionPolicy, load_decision_policy
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.preprocessing.contracts import FEATURE_COLUMNS
from churn.serving.errors import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ServingStartupError,
)
from churn.serving.provenance import runtime_source_checks
from churn.serving.runtime import VersionCheck, runtime_versions, verify_runtime_compatibility
from churn.serving.settings import ServingSettings, load_settings

logger = logging.getLogger(__name__)

#: The commit that froze the model. It is the model's version, and it is **not**
#: the version of this serving code: Phase 11 ships an application around an
#: object frozen in Phase 9C and changes nothing about it. A later serving
#: release keeps this value; only a new freeze may move it.
#:
#: It is a constant here rather than a policy field because the policy is written
#: *by* the commit that freezes it and therefore cannot contain its own hash.
FREEZE_COMMIT = "9d1db4962769093a617f1db2db66354536acffb3"
FREEZE_COMMIT_SHORT = FREEZE_COMMIT[:7]

#: Version of the serving application itself. Independent of the model.
SERVING_VERSION = "1.0.0"

#: Phase that produced the frozen artefacts this service loads.
FREEZE_PHASE = "9C"

#: SHA-256 of ``reports/decision_policy.json`` as it was frozen.
#:
#: Without this, the policy would be its own only authority: a hand-edited
#: threshold, calibration policy or positive class would load cleanly, pass every
#: internal consistency gate, and be served — because each of those gates asks
#: whether the policy agrees with itself. This pin is what makes the file's
#: *content* checkable at all, exactly as the policy pins the pipeline's bytes.
#:
#: The value is not invented here. Phases 9D, 9E and 10 each recorded the digest
#: of the policy they read, independently, and all three agree:
#: ``holdout_results.json``, ``error_analysis_results.json`` and
#: ``model_interpretation_results.json``. Only a deliberate re-freeze may move it.
#:
#: It is a byte digest, and the repository pins its checkout to LF in
#: ``.gitattributes``, which is what keeps it stable across platforms.
DECISION_POLICY_SHA256 = "bd7aa800ae36ac04ecf951f7377ce71d16d084faeee771b01eb6c9655a4bc714"

#: Calibration policy the freeze settled on. Recorded here so the startup gate
#: can state what it expects rather than accepting whatever it finds.
EXPECTED_CALIBRATION_POLICY = "NONE"


@dataclass(frozen=True)
class StartupCheck:
    """One named startup gate and what it observed."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class FrozenArtifacts:
    """The verified frozen model, and the facts a request needs from it.

    Immutable and shared by every request in the process. Nothing here is
    recomputed per call and nothing is mutated after startup: the pipeline is
    read-only during serving, which is what makes one shared instance safe.
    """

    policy: DecisionPolicy
    pipeline: Pipeline
    policy_sha256: str
    pipeline_sha256: str
    model_fingerprint: str
    positive_class_column: int
    positive_class_label: int
    threshold: float
    comparison: str
    calibration_policy: str
    threshold_policy: str
    estimator: str
    feature_columns: tuple[str, ...]
    n_transformed_features: int
    runtime: dict[str, str]
    checks: tuple[StartupCheck, ...]

    @property
    def n_features(self) -> int:
        """Number of raw features the boundary accepts."""
        return len(self.feature_columns)

    @property
    def freeze_commit(self) -> str:
        """Commit that froze the model. The model's version."""
        return FREEZE_COMMIT

    @property
    def serving_version(self) -> str:
        """Version of the serving application. Not the model's version."""
        return SERVING_VERSION


def _check(name: str, passed: bool, detail: str) -> StartupCheck:
    return StartupCheck(name=name, passed=bool(passed), detail=detail)


def _raise_on_failure(checks: list[StartupCheck]) -> None:
    """Raise listing every failed gate, not only the first."""
    failed = [check for check in checks if not check.passed]
    if failed:
        raise ArtifactIntegrityError(
            "The frozen model did not pass the serving startup gates:\n  "
            + "\n  ".join(f"{check.name}: {check.detail}" for check in failed)
            + "\nThe service does not start. No artefact is rebuilt: this boundary "
            "serves a frozen model, it does not produce one."
        )


def resolve_pipeline_path(policy: DecisionPolicy, settings: ServingSettings) -> Path:
    """Return where the pipeline is expected, from trusted process configuration.

    The path comes from the settings or from the policy — never from a request.
    An HTTP client cannot name a file this process will unpickle.
    """
    if settings.pipeline_path is not None:
        return settings.pipeline_path
    return PROJECT_ROOT / policy.artifacts.pipeline_path


def _read_policy(path: Path) -> DecisionPolicy:
    if not path.is_file():
        raise ArtifactNotFoundError(
            "The frozen decision policy is absent, so there is no threshold, no "
            "positive class and no model identity to serve. The service does not "
            "start and does not produce one."
        )
    try:
        return load_decision_policy(path)
    except ValidationError as error:
        raise ArtifactIntegrityError(
            f"The decision policy does not parse into the Phase {FREEZE_PHASE} schema "
            f"({error.error_count()} violation(s)). A file that is not a decision policy "
            "cannot be served from."
        ) from error


def policy_checks(policy: DecisionPolicy, policy_sha256: str) -> list[StartupCheck]:
    """Gates that need only the policy and its digest.

    Public so that each gate can be exercised on its own, rather than only
    through a startup that stops at the first family of failures.
    """
    contract = policy.feature_contract
    declared = tuple(contract.numeric) + tuple(contract.categorical)
    threshold = policy.threshold.final_threshold
    rule = policy.decision_rule

    return [
        _check(
            "decision_policy_digest_matches",
            policy_sha256 == DECISION_POLICY_SHA256,
            f"actual={policy_sha256[:16]}… frozen={DECISION_POLICY_SHA256[:16]}…",
        ),
        _check(
            "policy_is_the_phase9c_freeze",
            policy.policy == POLICY_NAME and policy.frozen_at_phase == FREEZE_PHASE,
            f"policy={policy.policy!r} phase={policy.frozen_at_phase!r}",
        ),
        _check(
            "threshold_is_finite_and_a_probability",
            isinstance(threshold, float) and 0.0 < threshold < 1.0,
            f"threshold={threshold!r}",
        ),
        _check(
            "decision_comparison_is_greater_or_equal",
            rule.comparison == DECISION_COMPARISON,
            f"comparison={rule.comparison!r}",
        ),
        _check(
            "threshold_policy_recognised",
            policy.threshold.threshold_policy in KNOWN_THRESHOLD_POLICIES,
            f"{policy.threshold.threshold_policy!r} in {sorted(KNOWN_THRESHOLD_POLICIES)}",
        ),
        _check(
            "calibration_policy_recognised",
            policy.calibration.calibration_policy in KNOWN_CALIBRATION_POLICIES,
            f"{policy.calibration.calibration_policy!r} in {sorted(KNOWN_CALIBRATION_POLICIES)}",
        ),
        _check(
            "calibration_policy_is_the_frozen_one",
            policy.calibration.calibration_policy == EXPECTED_CALIBRATION_POLICY
            and not policy.calibration.calibrated,
            f"policy={policy.calibration.calibration_policy!r} "
            f"calibrated={policy.calibration.calibrated}",
        ),
        _check(
            "policy_classes_are_the_encoded_binary_target",
            list(rule.classes) == [NEGATIVE_CLASS_LABEL, POSITIVE_CLASS_LABEL],
            f"classes={list(rule.classes)}",
        ),
        _check(
            "policy_positive_class_is_unambiguous",
            rule.positive_class_label == POSITIVE_CLASS_LABEL
            and 0 <= rule.positive_class_column < len(rule.classes)
            and rule.classes[rule.positive_class_column] == POSITIVE_CLASS_LABEL,
            f"label={rule.positive_class_label} column={rule.positive_class_column} "
            f"classes={list(rule.classes)}",
        ),
        _check(
            "inference_entry_point_is_prepare_features",
            contract.entry_point == PREPARE_FEATURES_QUALNAME
            and policy.inference_contract.entry_point == PREPARE_FEATURES_QUALNAME,
            f"entry_point={contract.entry_point!r}",
        ),
        _check(
            "feature_contract_matches_the_installed_contract",
            declared == FEATURE_COLUMNS and contract.n_features == len(FEATURE_COLUMNS),
            f"policy={len(declared)} features, installed={len(FEATURE_COLUMNS)}"
            + ("" if declared == FEATURE_COLUMNS else f"; policy order={list(declared)}"),
        ),
        _check(
            "no_engineered_feature_in_the_contract",
            not contract.engineered,
            f"engineered={list(contract.engineered)}",
        ),
        _check(
            "feature_names_digest_matches",
            text_digest(list(FEATURE_COLUMNS))
            == contract.feature_names_sha256
            == policy.inference_contract.feature_names_sha256,
            f"actual={text_digest(list(FEATURE_COLUMNS))[:16]}… "
            f"recorded={contract.feature_names_sha256[:16]}…",
        ),
        _check(
            "prepare_features_source_unchanged",
            function_source_digest(PREPARE_FEATURES)
            == policy.inference_contract.prepare_features_source_sha256,
            f"actual={function_source_digest(PREPARE_FEATURES)[:16]}… "
            f"recorded={policy.inference_contract.prepare_features_source_sha256[:16]}…",
        ),
        *(
            _check(name, passed, detail)
            for name, passed, detail in runtime_source_checks(policy.code_provenance.source_digests)
        ),
    ]


def pipeline_checks(
    path: Path,
    policy: DecisionPolicy,
) -> tuple[list[StartupCheck], Pipeline | None, str, str]:
    """Gates that need the serialised pipeline. Returns ``(checks, pipeline, file, model)``.

    Public for the same reason as :func:`policy_checks`: a gate that cannot be
    provoked in isolation cannot be shown to work.
    """
    if not path.is_file():
        raise ArtifactNotFoundError(
            "The frozen pipeline artefact is absent. The service does not start and "
            "does not rebuild it: regenerating a model inside a serving process would "
            "make the object served unprovable."
        )

    actual_file = file_digest(path)
    checks = [
        _check(
            "pipeline_file_digest_matches",
            actual_file == policy.artifacts.pipeline_sha256,
            f"actual={actual_file[:16]}… recorded={policy.artifacts.pipeline_sha256[:16]}…",
        )
    ]

    try:
        pipeline = load_pipeline(path)
    except Exception as error:  # noqa: BLE001 - any failure here is a failed gate
        # Startup diagnostics are for the operator's console, never for a client:
        # this branch cannot be reached through an HTTP request.
        checks.append(_check("pipeline_loads", False, f"{type(error).__name__}: {error}"))
        return checks, None, actual_file, ""
    checks.append(_check("pipeline_loads", True, type(pipeline).__name__))

    try:
        actual_model = model_fingerprint(pipeline)
    except FrozenModelError as error:
        checks.append(_check("pipeline_has_the_frozen_structure", False, str(error)))
        return checks, None, actual_file, ""

    checks.append(_check("pipeline_has_the_frozen_structure", True, "preprocessor -> classifier"))
    checks.append(
        _check(
            "model_fingerprint_matches",
            actual_model == policy.artifacts.model_fingerprint_sha256,
            f"actual={actual_model[:16]}… "
            f"recorded={policy.artifacts.model_fingerprint_sha256[:16]}…",
        )
    )

    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    classes = [int(label) for label in classifier.classes_]
    checks.append(
        _check(
            "loaded_classes_match_the_policy",
            classes == list(policy.decision_rule.classes),
            f"loaded={classes} policy={list(policy.decision_rule.classes)}",
        )
    )
    checks.append(
        _check(
            "loaded_classifier_has_no_calibration_wrapper",
            type(classifier).__name__ == policy.model.estimator,
            f"classifier={type(classifier).__name__} policy={policy.model.estimator!r}",
        )
    )

    try:
        column = positive_class_column(pipeline)
    except FrozenModelError as error:
        checks.append(_check("positive_class_column_resolves_from_classes_", False, str(error)))
        return checks, None, actual_file, actual_model

    checks.append(
        _check(
            "positive_class_column_resolves_from_classes_",
            column == policy.decision_rule.positive_class_column,
            f"resolved={column} policy={policy.decision_rule.positive_class_column} "
            f"classes={classes}",
        )
    )

    transformed = transformed_feature_names(pipeline)
    checks.append(
        _check(
            "transformed_feature_names_match",
            text_digest(transformed) == policy.feature_contract.transformed_feature_names_sha256
            and len(transformed) == policy.feature_contract.n_transformed_features,
            f"{len(transformed)} columns, digest={text_digest(transformed)[:16]}…",
        )
    )
    checks.append(
        _check(
            "preprocessor_step_present",
            PREPROCESSOR_STEP in pipeline.named_steps,
            f"steps={list(pipeline.named_steps)}",
        )
    )
    return checks, pipeline, actual_file, actual_model


def load_frozen_artifacts(settings: ServingSettings | None = None) -> FrozenArtifacts:
    """Run every startup gate and return the verified frozen model.

    Called once per process. The returned object is immutable and is what every
    request reads; nothing here runs again per request.

    Args:
        settings: Trusted process configuration. Defaults to :func:`load_settings`.

    Returns:
        The verified :class:`FrozenArtifacts`.

    Raises:
        ArtifactNotFoundError: If the policy or the pipeline is absent.
        ArtifactIntegrityError: If any gate fails.
        RuntimeCompatibilityError: If the runtime is not the frozen environment.
    """
    resolved = settings or load_settings()

    policy = _read_policy(resolved.policy_path)
    policy_sha256 = file_digest(resolved.policy_path)
    runtime_checks: list[VersionCheck] = verify_runtime_compatibility(policy.environment)

    checks = policy_checks(policy, policy_sha256)
    pipeline_path = resolve_pipeline_path(policy, resolved)
    pipeline_gates, pipeline, pipeline_sha, fingerprint = pipeline_checks(pipeline_path, policy)
    checks.extend(pipeline_gates)
    _raise_on_failure(checks)

    if pipeline is None:  # pragma: no cover - unreachable: a None pipeline failed a gate
        raise ArtifactIntegrityError("The frozen pipeline could not be loaded.")

    checks.extend(
        _check(f"runtime_{check.component}_matches", check.compatible, check.actual)
        for check in runtime_checks
    )

    artifacts = FrozenArtifacts(
        policy=policy,
        pipeline=pipeline,
        policy_sha256=policy_sha256,
        pipeline_sha256=pipeline_sha,
        model_fingerprint=fingerprint,
        positive_class_column=policy.decision_rule.positive_class_column,
        positive_class_label=policy.decision_rule.positive_class_label,
        threshold=policy.threshold.final_threshold,
        comparison=policy.decision_rule.comparison,
        calibration_policy=policy.calibration.calibration_policy,
        threshold_policy=policy.threshold.threshold_policy,
        estimator=policy.model.estimator,
        feature_columns=FEATURE_COLUMNS,
        n_transformed_features=policy.feature_contract.n_transformed_features,
        runtime=runtime_versions(),
        checks=tuple(checks),
    )
    logger.info(
        "Frozen model loaded: fingerprint=%s freeze_commit=%s threshold=%r calibration=%s "
        "(%d startup gates passed)",
        artifacts.model_fingerprint[:16],
        FREEZE_COMMIT_SHORT,
        artifacts.threshold,
        artifacts.calibration_policy,
        len(artifacts.checks),
    )
    return artifacts


__all__ = [
    "DECISION_POLICY_SHA256",
    "EXPECTED_CALIBRATION_POLICY",
    "FREEZE_COMMIT",
    "FREEZE_COMMIT_SHORT",
    "FREEZE_PHASE",
    "SERVING_VERSION",
    "FrozenArtifacts",
    "ServingStartupError",
    "StartupCheck",
    "load_frozen_artifacts",
    "pipeline_checks",
    "policy_checks",
    "resolve_pipeline_path",
]
