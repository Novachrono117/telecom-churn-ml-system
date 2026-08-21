"""The frozen decision policy: the contract Phase 9D and Phase 11 read.

This artefact is the whole point of Phase 9C. It states, in one versioned file,
what the model is, what it expects, how its probability becomes a decision, and
which bytes produced it — so that after the holdout is opened it is still
provable that the object evaluated was the object frozen beforehand.

**It lives in ``reports/``, not in ``artifacts/``.** ``artifacts/`` is
git-ignored by an existing repository convention for regenerable binaries, and a
freeze record that is not versioned cannot prove it predates anything. The
precedent followed here is ``reports/split_manifest.json``: the same kind of
object — a frozen contract carrying fingerprints — kept in the same place, for
the same reason. The fitted pipeline itself stays a regenerable artefact; the
policy carries its digests, so a rebuilt file can be checked rather than trusted.

Two deliberate absences, as in every earlier phase: **no timestamp**, so a rerun
on an unchanged repository reproduces the file byte for byte, and **no holdout
quantity of any kind**. The only holdout-shaped field is ``holdout_touched``,
which is ``false``.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

from churn.config import PROJECT_ROOT, get_config
from churn.modeling.freeze import (
    CONFIGURATION_FILES,
    DECISION_COMPARISON,
    INFERENCE_SOURCE_FILES,
    NEGATIVE_CLASS_LABEL,
    PIPELINE_RELATIVE_PATH,
    POSITIVE_CLASS_LABEL,
    POSITIVE_CLASS_MEANING,
    PREPARE_FEATURES,
    PREPARE_FEATURES_QUALNAME,
    SERIALIZATION_COMPRESS,
    SERIALIZATION_PROTOCOL,
    IntegrityCheck,
    check_decision_rule,
    check_pipeline_artefact,
    check_policy_vocabulary,
    check_source_provenance,
    check_threshold,
    file_digest,
    function_source_digest,
    model_fingerprint,
    positive_class_column,
    raise_on_failure,
    source_digests,
    text_digest,
    transformed_feature_names,
)
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

POLICY_PATH = PROJECT_ROOT / "reports" / "decision_policy.json"

SCHEMA_VERSION = 1
POLICY_NAME = "phase9c-decision-policy-freeze"

#: Upstream artefacts this freeze is anchored to. Their digests go into the
#: record, so a reader learns which bytes were read, not merely which names.
UPSTREAM_ARTEFACTS: tuple[str, ...] = (
    "reports/split_manifest.json",
    "reports/experiments/baseline_results.json",
    "reports/experiments/feature_engineering_results.json",
    "reports/experiments/model_comparison_results.json",
    "reports/experiments/hgb_representation_results.json",
    "reports/experiments/tuning_results.json",
    "reports/experiments/calibration_results.json",
    "reports/experiments/threshold_results.json",
)

_PRECISION = 6


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


class ModelRecord(BaseModel):
    """What the estimator is, and what it was fitted on."""

    model_config = ConfigDict(frozen=True)

    builder: str
    pipeline_builder: str
    estimator: str
    selected_in: str
    hyperparameters: dict[str, object]
    preprocessing: dict[str, object]
    fitted_on: dict[str, object]
    convergence: dict[str, object]
    reopened_here: bool


class FeatureContractRecord(BaseModel):
    """What the persisted pipeline expects to be handed."""

    model_config = ConfigDict(frozen=True)

    entry_point: str
    boundary: str
    n_features: int = Field(ge=1)
    numeric: list[str]
    categorical: list[str]
    engineered: list[str]
    feature_names_sha256: str = Field(min_length=64, max_length=64)
    n_transformed_features: int = Field(ge=1)
    transformed_feature_names_sha256: str = Field(min_length=64, max_length=64)


class CalibrationRecord(BaseModel):
    """The frozen calibration policy, and where it was decided."""

    model_config = ConfigDict(frozen=True)

    calibration_policy: str
    calibrated: bool
    decided_in: str
    probability_expression: str
    note: str


class ThresholdRecord(BaseModel):
    """The frozen threshold, and where it was decided."""

    model_config = ConfigDict(frozen=True)

    threshold_policy: str
    final_threshold: float
    decided_in: str
    rounded: bool
    note: str


class DecisionRuleRecord(BaseModel):
    """How a probability becomes a decision, with nothing left implicit."""

    model_config = ConfigDict(frozen=True)

    expression: str
    comparison: str
    probability_expression: str
    classes: list[int]
    positive_class_label: int
    positive_class_column: int = Field(ge=0)
    positive_class_meaning: str
    negative_class_label: int
    boundary_note: str


class InferenceContractRecord(BaseModel):
    """The part of the predictive function that lives outside the pipeline.

    ``prepare_features`` runs before the persisted object is reached, so a change
    to it alters what the model is handed without moving a single one of the
    model's own fingerprints. Freezing its source closes that gap.
    """

    model_config = ConfigDict(frozen=True)

    chain: list[str]
    entry_point: str
    prepare_features_module: str
    prepare_features_source_sha256: str = Field(min_length=64, max_length=64)
    feature_names_sha256: str = Field(min_length=64, max_length=64)
    note: str


class CodeProvenanceRecord(BaseModel):
    """Digests of everything whose edit could change a prediction."""

    model_config = ConfigDict(frozen=True)

    strategy: str
    digest_definition: str
    caveat: str
    source_digests: dict[str, str]
    configuration_digests: dict[str, str]
    why_each_file: dict[str, str]
    not_covered: list[str]


class ArtefactRecord(BaseModel):
    """The serialised pipeline: where it is, and which bytes it should be."""

    model_config = ConfigDict(frozen=True)

    pipeline_path: str
    pipeline_sha256: str = Field(min_length=64, max_length=64)
    pipeline_bytes: int = Field(ge=1)
    model_fingerprint_sha256: str = Field(min_length=64, max_length=64)
    serialization: dict[str, object]
    versioned_in_git: bool
    authoritative_fingerprint: str
    note: str


class DecisionPolicy(BaseModel):
    """The complete Phase 9C freeze."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    policy: str
    frozen_at_phase: str
    raw_sha256: str = Field(min_length=64, max_length=64)
    training_ids_sha256: str = Field(min_length=64, max_length=64)
    random_seed: int = Field(ge=0)
    model: ModelRecord
    feature_contract: FeatureContractRecord
    inference_contract: InferenceContractRecord
    code_provenance: CodeProvenanceRecord
    calibration: CalibrationRecord
    threshold: ThresholdRecord
    decision_rule: DecisionRuleRecord
    artifacts: ArtefactRecord
    provenance: dict[str, object]
    environment: dict[str, str]
    holdout_touched: bool
    holdout_status: str
    evaluation_status: str
    reopened_decisions: list[str]
    notes: list[str]


NOTES: tuple[str, ...] = (
    "Phase 9C decides nothing. It materialises decisions taken in Phases 5-8B (estimator and "
    "hyperparameters), 9A (calibration policy) and 9B (threshold policy) as one persisted object "
    "plus this contract.",
    "The final fit uses the WHOLE frozen training pool. Earlier phases fitted on four fifths at a "
    "time because they needed a validation fold; that need is over once the configuration is "
    "frozen. No metric was computed from this fit, and none could change anything if it were.",
    "No holdout row was loaded, transformed, predicted or counted at any point in this phase. The "
    "modules that do the work never load data at all: the training pool is handed to them.",
    "NO HOLDOUT OR TEST PERFORMANCE HAS BEEN MEASURED AT THIS STAGE. There is no performance "
    "figure anywhere in this repository yet.",
    "The persisted pipeline starts AFTER prepare_features(). It expects a validated, canonically "
    "ordered feature matrix; producing one is the caller's responsibility. That boundary was "
    "fixed in Phase 4 and is not moved here.",
    "Two fingerprints are recorded and they answer different questions. "
    "model_fingerprint_sha256 hashes the learned parameters under a canonical JSON encoding that "
    "round-trips every double exactly: it is the IDENTITY of the model and is invariant to how "
    "the object was serialised. pipeline_sha256 hashes the file bytes: it catches a corrupted or "
    "swapped file but is a property of the file, not of the model, and is not guaranteed to "
    "survive a change of environment. The first is authoritative.",
    "The serialised pipeline is a regenerable artefact and is NOT committed: artifacts/ is "
    "git-ignored by an existing repository convention. Rebuilding it from the frozen split, the "
    "frozen builder and the frozen training pool must reproduce both fingerprints, which is a "
    "stronger guarantee than storing the binary would be.",
    "joblib protocol and compression are pinned rather than left at their defaults. A future "
    "interpreter raising pickle.DEFAULT_PROTOCOL would otherwise change every byte of the file, "
    "and its digest, without anything about the model having changed.",
    "The decision comparison is >=, written down explicitly. > and >= differ on exactly the "
    "customers whose probability equals the threshold, and a rule that leaves the boundary "
    "implicit is not a frozen rule.",
    "The positive-class probability column is resolved from classes_, not assumed to be column 1. "
    "The assumption happens to hold here and is still the wrong thing to record: reading the "
    "wrong column would invert every decision without raising anything.",
    "The threshold is stored unrounded. A threshold rounded in the sixth decimal can move "
    "customers across the decision boundary.",
    "class_weight stays None and stays closed. It changes the objective the model is fitted on, "
    "so reopening it would reopen model selection; the threshold only changes a post-fit "
    "decision rule, which is why it could be chosen in Phase 9B.",
    "This phase builds no serving layer. What exists here are persistence and integrity "
    "primitives; the public prediction interface is Phase 11's subject.",
)


def build_decision_policy(
    pipeline: Pipeline,
    pipeline_path: Path,
    threshold_results: object,
    calibration_results: object,
    tuning_results: object,
    n_training_rows: int,
    training_prevalence: float,
    digests: Mapping[str, str],
) -> DecisionPolicy:
    """Assemble the freeze record from the fitted pipeline and the frozen inputs.

    Every policy value is **read from the upstream artefacts**, never restated:
    the threshold comes from the Phase 9B record, the calibration policy from the
    Phase 9A record, the hyperparameters from the fitted object itself. A freeze
    that retyped its inputs could disagree with them.
    """
    config = get_config()
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    transformed = transformed_feature_names(pipeline)
    classes = [int(value) for value in classifier.classes_]
    column = positive_class_column(pipeline)
    threshold = float(threshold_results.selection.final_threshold)

    return DecisionPolicy(
        schema_version=SCHEMA_VERSION,
        policy=POLICY_NAME,
        frozen_at_phase="9C",
        raw_sha256=threshold_results.raw_sha256,
        training_ids_sha256=threshold_results.training_ids_sha256,
        random_seed=config.seed,
        model=ModelRecord(
            builder="churn.modeling.tuning.build_modern_logistic",
            pipeline_builder="churn.modeling.tuning.build_modern_logistic_pipeline",
            estimator=type(classifier).__name__,
            selected_in=tuning_results.experiment,
            hyperparameters={
                key: value
                for key, value in sorted(classifier.get_params().items())
                if key in {"C", "l1_ratio", "solver", "class_weight", "max_iter"}
            },
            preprocessing={
                "builder": "churn.features.pipeline.build_feature_preprocessor",
                "steps": [
                    "TotalChargesCleaner",
                    "StandardScaler on the 3 numeric features",
                    "OneHotEncoder(handle_unknown='ignore') on the 16 categorical features",
                ],
                "n_transformed_features": len(transformed),
                "fitted_on_the_training_pool_only": True,
            },
            fitted_on={
                "partition": "training pool",
                "loader": "churn.preprocessing.splitting.load_training_pool",
                "n_rows": int(n_training_rows),
                "positive_prevalence": _round(training_prevalence),
                "fraction_of_pool_used": 1.0,
                "note": (
                    "the whole pool, not a fold: this fit materialises a frozen configuration "
                    "and produces no estimate, so no validation split is withheld"
                ),
            },
            convergence={
                "converged": bool(int(classifier.n_iter_[0]) < int(classifier.max_iter)),
                "n_iter": int(classifier.n_iter_[0]),
                "max_iter": int(classifier.max_iter),
            },
            reopened_here=False,
        ),
        feature_contract=FeatureContractRecord(
            entry_point="churn.preprocessing.contracts.prepare_features",
            boundary=(
                "the persisted pipeline is applied to the OUTPUT of prepare_features(). It "
                "expects a validated feature matrix in canonical column order and does not "
                "validate one itself"
            ),
            n_features=len(FEATURE_COLUMNS),
            numeric=list(NUMERIC_FEATURES),
            categorical=list(CATEGORICAL_FEATURES),
            engineered=[],
            feature_names_sha256=text_digest(list(FEATURE_COLUMNS)),
            n_transformed_features=len(transformed),
            transformed_feature_names_sha256=text_digest(transformed),
        ),
        inference_contract=InferenceContractRecord(
            chain=[
                "raw input",
                f"{PREPARE_FEATURES_QUALNAME}(...)",
                "frozen sklearn pipeline (preprocessor -> classifier)",
                "predict_proba(...)",
                "positive-class probability (column resolved from classes_)",
                f"probability >= {threshold}",
            ],
            entry_point=PREPARE_FEATURES_QUALNAME,
            prepare_features_module="churn.preprocessing.contracts",
            prepare_features_source_sha256=function_source_digest(PREPARE_FEATURES),
            feature_names_sha256=text_digest(list(FEATURE_COLUMNS)),
            note=(
                "prepare_features runs BEFORE the persisted pipeline and is therefore outside "
                "everything the model's own fingerprints can see. A change to it would alter the "
                "values handed to the model while pipeline_sha256, model_fingerprint_sha256, the "
                "feature names, the threshold and the hyperparameters all stayed identical. Its "
                "source is fingerprinted here so that cannot happen silently"
            ),
        ),
        code_provenance=CodeProvenanceRecord(
            strategy=(
                "whole-file SHA-256 of every source file that materially determines the frozen "
                "inference function, plus the configuration and the dependency pins. No source "
                "parsing, no AST normalisation, no comment stripping"
            ),
            digest_definition=(
                "SHA-256 of the file's UTF-8 text with newlines normalised to \\n. Not a raw byte "
                "hash: a byte hash changes when the checkout's line endings change, which would "
                "fail the freeze for a reason unrelated to the code. .gitattributes already pins "
                "this repository to LF, so the two agree today; normalising keeps the digest valid "
                "in a checkout that ignores it, such as a zip download or a Colab upload"
            ),
            caveat=(
                "WHOLE-FILE hashing means an UNRELATED edit in one of these files — a docstring, a "
                "new helper, a reordered import — also invalidates the freeze. That is deliberate "
                "and fail-closed: the alternative, hashing only the functions believed to matter, "
                "would silently miss a changed constant, a changed helper or a changed default. A "
                "freeze that survives an edit it should have caught is worse than one that fails "
                "on an edit it need not have caught"
            ),
            source_digests=source_digests(INFERENCE_SOURCE_FILES),
            configuration_digests=source_digests(CONFIGURATION_FILES),
            why_each_file={
                "src/churn/preprocessing/contracts.py": (
                    "defines prepare_features, the feature contract and the canonical column "
                    "order — the entire input side of the inference function"
                ),
                "src/churn/preprocessing/transformers.py": (
                    "defines TotalChargesCleaner, which is pickled BY REFERENCE inside the "
                    "persisted pipeline: the stream carries the module path and unpickling "
                    "imports whatever the module currently defines, so editing its transform "
                    "changes what the loaded pipeline computes while both model fingerprints "
                    "stay identical"
                ),
                "src/churn/preprocessing/pipeline.py": (
                    "configures the ColumnTransformer — which columns are scaled, which are "
                    "one-hot encoded, handle_unknown, the step names other code reads it by"
                ),
                "src/churn/features/pipeline.py": (
                    "assembles the preprocessing pipeline and fixes its step order"
                ),
                "src/churn/modeling/tuning.py": (
                    "defines build_modern_logistic and build_modern_logistic_pipeline, the "
                    "builders that produce the frozen estimator"
                ),
                "src/churn/modeling/models.py": (
                    "holds the solver, max_iter and step-name constants those builders read"
                ),
                "src/churn/preprocessing/target.py": (
                    "defines encode_target, which fixes which label is the positive class and "
                    "therefore what classes_ and the probability column mean"
                ),
                "src/churn/modeling/freeze.py": (
                    "implements the decision rule itself: the >= comparison, the resolution of "
                    "the positive-class column and the serialisation settings. Changing >= to > "
                    "here would change every decision at the boundary"
                ),
                "configs/base.toml": "fixes the seed, the target column and the positive label",
                "pyproject.toml": "declares the dependency constraints",
                "uv.lock": (
                    "pins the exact resolved versions whose behaviour the fitted object and the "
                    "transformations depend on"
                ),
            },
            not_covered=[
                "The train/holdout partition is not covered here: it is gated separately and more "
                "strongly by reports/split_manifest.json, whose digest is in "
                "provenance.upstream_artefact_digests and whose reproducibility is proven by "
                "`scripts/build_split.py --verify`.",
                "churn/modeling/freeze_results.py is deliberately excluded. It records the freeze; "
                "it does not participate in producing a prediction, so an edit to it cannot change "
                "one.",
                "Third-party library source is not hashed file by file. uv.lock pins the exact "
                "resolved versions, which is the auditable equivalent.",
                "The Python interpreter build is recorded in `environment` but not fingerprinted.",
            ],
        ),
        calibration=CalibrationRecord(
            calibration_policy=calibration_results.selection.selected_policy,
            calibrated=False,
            decided_in=calibration_results.experiment,
            probability_expression="pipeline.predict_proba(X)[:, positive_class_column]",
            note=(
                "NONE was the outcome of the Phase 9A gate, not an omission: neither sigmoid nor "
                "isotonic calibration cleared the pre-registered eligibility rule, so no "
                "calibration layer was adopted and the frozen logistic regression keeps emitting "
                "its own probabilities. See reports/calibration_report.md; the result is not "
                "reinterpreted here"
            ),
        ),
        threshold=ThresholdRecord(
            threshold_policy=threshold_results.selection.selected_threshold_policy,
            final_threshold=threshold,
            decided_in=threshold_results.experiment,
            rounded=False,
            note=(
                "selected in Phase 9B by applying the pre-registered F1-maximisation policy to "
                "out-of-fold probabilities of the training pool, after a nested evaluation showed "
                "the procedure beating the 0.5 default in 5/5 outer folds. See "
                "reports/threshold_report.md; the result is not reinterpreted here"
            ),
        ),
        decision_rule=DecisionRuleRecord(
            expression="positive = probability >= final_threshold",
            comparison=DECISION_COMPARISON,
            probability_expression=(
                "pipeline.predict_proba(prepare_features(X))[:, positive_class_column]"
            ),
            classes=classes,
            positive_class_label=POSITIVE_CLASS_LABEL,
            positive_class_column=column,
            positive_class_meaning=POSITIVE_CLASS_MEANING,
            negative_class_label=NEGATIVE_CLASS_LABEL,
            boundary_note=(
                "a customer whose probability equals the threshold exactly is predicted POSITIVE. "
                "The comparison is >=, not >"
            ),
        ),
        artifacts=ArtefactRecord(
            pipeline_path=PIPELINE_RELATIVE_PATH,
            pipeline_sha256=file_digest(pipeline_path),
            pipeline_bytes=pipeline_path.stat().st_size,
            model_fingerprint_sha256=model_fingerprint(pipeline),
            serialization={
                "library": "joblib",
                "joblib_version": joblib.__version__,
                "protocol": SERIALIZATION_PROTOCOL,
                "compress": SERIALIZATION_COMPRESS,
                "pinned": True,
            },
            versioned_in_git=False,
            authoritative_fingerprint="model_fingerprint_sha256",
            note=(
                "the file is regenerable by `uv run python scripts/freeze_model.py` from the "
                "frozen split, the frozen builder and the frozen training pool; both fingerprints "
                "must come out unchanged"
            ),
        ),
        provenance={
            "upstream_artefact_digests": dict(digests),
            "references": {
                "tuning": {
                    "artefact": "reports/experiments/tuning_results.json",
                    "experiment": tuning_results.experiment,
                    "schema_version": tuning_results.schema_version,
                },
                "calibration": {
                    "artefact": "reports/experiments/calibration_results.json",
                    "experiment": calibration_results.experiment,
                    "schema_version": calibration_results.schema_version,
                },
                "threshold": {
                    "artefact": "reports/experiments/threshold_results.json",
                    "experiment": threshold_results.experiment,
                    "schema_version": threshold_results.schema_version,
                },
            },
            "configuration": "configs/base.toml",
            "split": {
                "manifest": "reports/split_manifest.json",
                "note": (
                    "the manifest digest above covers the whole frozen partition. No holdout "
                    "identifier, row or statistic is copied into this policy"
                ),
            },
        },
        environment={
            "python": ".".join(str(part) for part in sys.version_info[:2]),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        holdout_touched=False,
        holdout_status=(
            "untouched. No holdout row was loaded, transformed, predicted, thresholded or counted "
            "in this phase, and no holdout statistic exists anywhere in this repository"
        ),
        evaluation_status=(
            "no holdout/test performance has been measured at this stage. The first and only "
            "holdout evaluation is Phase 9D"
        ),
        reopened_decisions=[],
        notes=list(NOTES),
    )


def write_decision_policy(policy: DecisionPolicy, path: Path | None = None) -> Path:
    """Write the freeze record as indented JSON."""
    destination = path or POLICY_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(policy.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote decision policy: %s", destination)
    return destination


def load_decision_policy(path: Path | None = None) -> DecisionPolicy:
    """Read and validate the freeze record.

    Validation is the schema check the integrity report reports on: a file that
    does not parse into this model is not a decision policy, and saying so here
    is better than discovering it field by field later.

    Raises:
        FileNotFoundError: If the freeze has not been produced yet.
    """
    source = path or POLICY_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Decision policy not found at {source}. Produce it with "
            "`uv run python scripts/freeze_model.py`."
        )
    return DecisionPolicy.model_validate_json(source.read_text(encoding="utf-8"))


def verify_decision_policy(
    policy: DecisionPolicy,
    threshold_results: object,
    calibration_results: object,
    root: Path | None = None,
) -> tuple[list[IntegrityCheck], Pipeline | None]:
    """Run every integrity check on a loaded policy and its pipeline.

    Returns the full list rather than raising, so a caller can print all of them;
    :func:`churn.modeling.freeze.raise_on_failure` turns them into an error.

    Args:
        policy: The loaded freeze record.
        threshold_results: The Phase 9B record, the source of truth for the
            threshold and its policy.
        calibration_results: The Phase 9A record, the source of truth for the
            calibration policy.
        root: Repository root, for resolving the pipeline path. Defaults to the
            real one.

    Returns:
        ``(checks, pipeline)``.
    """
    base = root or PROJECT_ROOT
    checks: list[IntegrityCheck] = [
        IntegrityCheck(
            name="policy_schema_valid",
            passed=True,
            detail=f"{POLICY_NAME} v{policy.schema_version} parsed and validated",
        )
    ]

    checks.extend(
        check_threshold(
            policy.threshold.final_threshold,
            threshold_results.selection.final_threshold,
        )
    )
    checks.extend(
        check_policy_vocabulary(
            policy.calibration.calibration_policy, policy.threshold.threshold_policy
        )
    )
    checks.append(
        IntegrityCheck(
            name="calibration_policy_matches_phase9a",
            passed=policy.calibration.calibration_policy
            == calibration_results.selection.selected_policy,
            detail=(
                f"policy={policy.calibration.calibration_policy!r} "
                f"phase9a={calibration_results.selection.selected_policy!r}"
            ),
        )
    )
    checks.append(
        IntegrityCheck(
            name="threshold_policy_matches_phase9b",
            passed=policy.threshold.threshold_policy
            == threshold_results.selection.selected_threshold_policy,
            detail=(
                f"policy={policy.threshold.threshold_policy!r} "
                f"phase9b={threshold_results.selection.selected_threshold_policy!r}"
            ),
        )
    )
    checks.extend(
        check_decision_rule(
            policy.decision_rule.comparison,
            policy.decision_rule.positive_class_column,
            policy.decision_rule.classes,
        )
    )
    checks.append(
        IntegrityCheck(
            name="no_engineered_feature_in_the_contract",
            passed=policy.feature_contract.engineered == [],
            detail=f"engineered={policy.feature_contract.engineered}",
        )
    )
    checks.append(
        IntegrityCheck(
            name="class_weight_still_none",
            passed=policy.model.hyperparameters.get("class_weight") is None,
            detail=f"class_weight={policy.model.hyperparameters.get('class_weight')!r}",
        )
    )
    checks.append(
        IntegrityCheck(
            name="holdout_untouched",
            passed=policy.holdout_touched is False,
            detail=f"holdout_touched={policy.holdout_touched}",
        )
    )
    # The part of the predictive function that lives outside the .joblib. Without
    # these three, a change to prepare_features or to TotalChargesCleaner would
    # pass every other check in this list.
    checks.extend(
        check_source_provenance(
            policy.code_provenance.source_digests,
            policy.code_provenance.configuration_digests,
            policy.inference_contract.prepare_features_source_sha256,
            base,
        )
    )
    checks.append(
        IntegrityCheck(
            name="inference_entry_point_unchanged",
            passed=policy.inference_contract.entry_point == PREPARE_FEATURES_QUALNAME,
            detail=f"entry_point={policy.inference_contract.entry_point!r}",
        )
    )

    artefact_checks, pipeline = check_pipeline_artefact(
        base / policy.artifacts.pipeline_path,
        policy.artifacts.pipeline_sha256,
        policy.artifacts.model_fingerprint_sha256,
    )
    checks.extend(artefact_checks)

    if pipeline is not None:
        transformed = transformed_feature_names(pipeline)
        checks.append(
            IntegrityCheck(
                name="transformed_feature_names_match",
                passed=text_digest(transformed)
                == policy.feature_contract.transformed_feature_names_sha256,
                detail=f"{len(transformed)} columns",
            )
        )
        checks.append(
            IntegrityCheck(
                name="loaded_pipeline_agrees_on_the_positive_column",
                passed=positive_class_column(pipeline)
                == policy.decision_rule.positive_class_column,
                detail=f"column={positive_class_column(pipeline)}",
            )
        )
        classifier = pipeline.named_steps[CLASSIFIER_STEP]
        checks.append(
            IntegrityCheck(
                name="loaded_pipeline_hyperparameters_match",
                passed=all(
                    classifier.get_params()[key] == value
                    for key, value in policy.model.hyperparameters.items()
                ),
                detail=", ".join(f"{k}={v}" for k, v in policy.model.hyperparameters.items()),
            )
        )
        checks.append(
            IntegrityCheck(
                name="loaded_pipeline_has_no_calibration_wrapper",
                passed=type(classifier).__name__ == policy.model.estimator
                and not hasattr(classifier, "calibrated_classifiers_"),
                detail=f"classifier={type(classifier).__name__}",
            )
        )
        checks.append(
            IntegrityCheck(
                name="loaded_pipeline_has_the_two_frozen_steps",
                passed=list(pipeline.named_steps) == [PREPROCESSOR_STEP, CLASSIFIER_STEP],
                detail=f"steps={list(pipeline.named_steps)}",
            )
        )

    return checks, pipeline


def verify_or_raise(
    policy: DecisionPolicy,
    threshold_results: object,
    calibration_results: object,
    root: Path | None = None,
) -> tuple[list[IntegrityCheck], Pipeline]:
    """Verify a policy and raise :class:`IntegrityError` if anything failed."""
    checks, pipeline = verify_decision_policy(policy, threshold_results, calibration_results, root)
    raise_on_failure(checks)
    assert pipeline is not None  # raise_on_failure would have fired otherwise
    return checks, pipeline
