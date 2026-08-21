"""Machine-readable record of the Phase 9B decision-threshold policy.

Carries four things a later reader cannot reconstruct from the numbers: the
**exact selection rule** including its tie-breaking, the separation between the
one metric that decided and the tables that only describe, the pre-registered
eligibility rule with the branch it took, and the frozen threshold itself at full
double precision.

Thresholds are written **unrounded** while metrics are rounded to six decimals.
That asymmetry is deliberate: a metric rounded in the seventh decimal is still
the same measurement, but a threshold rounded in the sixth can move customers
across the decision boundary, and the number recorded here has to be the number
that will be applied to the holdout.

It also records the SHA-256 of every upstream artefact this phase was anchored
to. A reference by name says which file was meant; a digest says which bytes were
actually read.

Same two deliberate absences as every earlier phase: **no timestamp**, so the
determinism check is possible, and **no holdout quantity of any kind**.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT, get_config
from churn.modeling.comparison_results import (
    ArtefactReference,
    ReproductionCheck,
    ReproductionError,
    WarningRecord,
)
from churn.modeling.evaluation import CV_STRATEGY, N_SPLITS, SHUFFLE
from churn.modeling.metrics import DEFAULT_THRESHOLD
from churn.modeling.threshold import (
    CONFUSION_NAMES,
    CURVE_GRID_POINTS,
    DECISION_METRIC_NAMES,
    F1_POLICY,
    GUARDRAIL_METRIC_NAMES,
    MIN_IMPROVED_FOLDS,
    RECALL_TARGETS,
    SELECTION_METRIC,
    THRESHOLD_INNER_N_SPLITS,
    THRESHOLD_INNER_SHUFFLE,
    TIE_TOLERANCE,
    TOP_K_FRACTIONS,
    InnerThresholdSelection,
    OperatingPoint,
    RecallScenario,
    ThresholdRun,
    ThresholdStability,
    ThresholdSweep,
    TopKRecord,
)
from churn.modeling.tuning import build_modern_logistic
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "threshold_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase9b-decision-threshold-policy"

_PRECISION = 6

#: The calibration policy Phase 9A froze. Anything else and the threshold chosen
#: here would belong to a probability that no longer exists.
EXPECTED_CALIBRATION_POLICY = "NONE"

#: The Phase 9A experiment whose probabilities this phase cuts.
REFERENCE_EXPERIMENT = "C0"
REPRODUCTION_TOLERANCE = 1e-9

#: Tolerance for the claim that the nested outer probabilities and the
#: independently recomputed training-pool probabilities are the same vector.
IDENTITY_TOLERANCE = 1e-12


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return str(value)


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in binary so line endings cannot alter it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CalibrationPolicyMismatchError(RuntimeError):
    """This phase would cut a probability other than the one Phase 9A froze."""


def verify_calibration_policy(calibration_results: object) -> str:
    """Check that Phase 9A left the probability uncalibrated.

    A threshold is a cut on a score. If a calibrator had been adopted, the score
    would have been rewritten and a threshold chosen on the raw probability would
    select different customers than the same number on the calibrated one. This
    phase is written for ``NONE``; if the Phase 9A record ever says otherwise,
    the code must change rather than silently cut the wrong quantity.

    Args:
        calibration_results: The loaded Phase 9A record.

    Returns:
        The verified calibration policy.

    Raises:
        CalibrationPolicyMismatchError: If the policy is not ``NONE``.
    """
    policy = calibration_results.selection.selected_policy
    if policy != EXPECTED_CALIBRATION_POLICY:
        raise CalibrationPolicyMismatchError(
            f"Phase 9A selected calibration policy {policy!r}, but this phase is written for "
            f"{EXPECTED_CALIBRATION_POLICY!r} and cuts predict_proba directly. A threshold chosen "
            "on the raw score is not the same operating point on a calibrated one."
        )
    logger.info("Calibration policy verified against the Phase 9A record: %s", policy)
    return policy


def verify_probability_source(
    default_run: ThresholdRun,
    calibration_results: object,
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> ReproductionCheck:
    """Check the outer probabilities against the frozen Phase 9A C0 numbers.

    The outer loop of this phase fits the frozen pipeline on each outer training
    fold and predicts the outer validation fold — which is exactly what Phase 9A's
    C0 did, on the same partition and the same data. The two threshold-independent
    metrics recorded by both phases must therefore agree to the recorded
    precision. A disagreement means the estimator, the preprocessing or the
    partition moved, and the threshold selected here would belong to a probability
    this repository has never evaluated.

    Args:
        default_run: The D0 run, whose folds carry the outer probabilities.
        calibration_results: The loaded Phase 9A record.
        tolerance: Maximum tolerated absolute difference.

    Returns:
        The :class:`ReproductionCheck`.

    Raises:
        ReproductionError: If any fold differs by more than ``tolerance``.
    """
    reference = next(
        record
        for record in calibration_results.policies
        if record.experiment == REFERENCE_EXPERIMENT
    )

    largest = 0.0
    offenders: list[str] = []
    for fold, stored in zip(default_run.folds, reference.folds, strict=True):
        for metric in GUARDRAIL_METRIC_NAMES:
            difference = abs(_round(fold.metrics.value(metric)) - stored.metrics[metric])
            largest = max(largest, difference)
            if difference > tolerance:
                offenders.append(f"fold {fold.fold} {metric}: {difference:.3e}")

    if offenders:
        raise ReproductionError(
            "The outer probabilities do not reproduce the frozen Phase 9A C0 numbers. The "
            "probability source has drifted, so a threshold selected on it would not be a "
            "threshold of the frozen model.\n  " + "\n  ".join(offenders)
        )

    logger.info("Outer probabilities reproduce Phase 9A C0 within %.3e.", largest)
    return ReproductionCheck(
        experiment=default_run.experiment,
        reference_artefact="reports/experiments/calibration_results.json",
        reference_schema_version=calibration_results.schema_version,
        reference_key=f"policies[{REFERENCE_EXPERIMENT}]",
        tolerance=tolerance,
        max_absolute_difference=largest,
        reproduced=True,
    )


def verify_training_pool_probabilities(
    nested: np.ndarray,
    recomputed: np.ndarray,
    tolerance: float = IDENTITY_TOLERANCE,
) -> ReproductionCheck:
    """Check that the training-pool OOF vector is the nested outer vector.

    The final threshold is fixed by running the same procedure once on the whole
    training pool, with the same 5-fold splitter and the same frozen estimator.
    That is, by construction, the identical computation the outer loop already
    performed — so the two vectors must coincide. Recomputing it independently and
    comparing turns "same splitter, same model" from a sentence in a report into a
    property this run demonstrated.

    Args:
        nested: Outer out-of-fold probabilities from the nested evaluation.
        recomputed: Probabilities from an independent pass over the same folds.
        tolerance: Maximum tolerated absolute difference.

    Returns:
        The :class:`ReproductionCheck`.

    Raises:
        ReproductionError: If the two vectors differ by more than ``tolerance``.
    """
    largest = float(np.max(np.abs(np.asarray(nested) - np.asarray(recomputed))))
    if largest > tolerance:
        raise ReproductionError(
            "The independently recomputed training-pool out-of-fold probabilities differ from the "
            f"nested outer probabilities by {largest:.3e}. They are meant to be the same "
            "computation; a difference means the two passes did not share a splitter, an "
            "estimator or the data."
        )
    logger.info("Training-pool OOF reproduces the nested outer probabilities within %.3e.", largest)
    return ReproductionCheck(
        experiment="full_training_out_of_fold",
        reference_artefact="in-run: nested outer out-of-fold probabilities",
        reference_schema_version=SCHEMA_VERSION,
        reference_key="policies[D0].outer_out_of_fold_probability",
        tolerance=tolerance,
        max_absolute_difference=largest,
        reproduced=True,
    )


class OperatingPointRecord(BaseModel):
    """One threshold applied to one probability vector."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    predicted_positive_rate: float
    true_negatives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    true_positives: int = Field(ge=0)


class ThresholdDeltaRecord(BaseModel):
    """Per-fold difference of one metric between two threshold policies.

    Counts are named by direction, not by merit: a higher predicted positive rate
    is neither an improvement nor a regression, it is a larger contact list.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    reference: str
    per_fold: list[float]
    mean: float
    std: float
    folds_higher: int = Field(ge=0)
    folds_tied: int = Field(ge=0)
    folds_lower: int = Field(ge=0)


class ThresholdFoldRecord(BaseModel):
    """One outer fold of one threshold policy."""

    model_config = ConfigDict(frozen=True)

    fold: int = Field(ge=1)
    n_train: int = Field(ge=1)
    n_validation: int = Field(ge=1)
    threshold: float
    metrics: dict[str, float]
    confusion: dict[str, int]


class ThresholdPolicyRecord(BaseModel):
    """One threshold policy evaluated on the outer folds."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    label: str
    model: str
    policy: str
    threshold_source: str
    folds: list[ThresholdFoldRecord]
    mean: dict[str, float]
    std: dict[str, float]
    paired_deltas: dict[str, dict[str, ThresholdDeltaRecord]]


class OuterThresholdRecord(BaseModel):
    """The threshold one outer fold selected, and what it was selected from.

    The metrics here are **inner out-of-fold selection metrics**: the value the
    search maximised on rows inside the outer training fold. They are not
    performance, and the outer-fold metrics are reported separately.
    """

    model_config = ConfigDict(frozen=True)

    fold: int = Field(ge=1)
    n_outer_train: int = Field(ge=1)
    n_inner_splits: int = Field(ge=2)
    n_candidates: int = Field(ge=1)
    n_tied_within_tolerance: int = Field(ge=1)
    default_in_candidates: bool
    threshold: float
    inner_out_of_fold_selection_metrics: OperatingPointRecord


class EligibilityRecord(BaseModel):
    """Whether the selection procedure earned the right to replace 0.5."""

    model_config = ConfigDict(frozen=True)

    rule: list[str]
    rule_fixed_before_results: bool
    min_improved_folds: int = Field(ge=1)
    minimum_gain_cutoff: float | None
    decision_metric: str
    status: str
    rationale: str
    mean_delta_f1: float
    folds_f1_improved: int = Field(ge=0)
    folds_f1_tied: int = Field(ge=0)
    folds_f1_worsened: int = Field(ge=0)


class SelectionRecord(BaseModel):
    """The frozen decision policy carried into the final holdout evaluation."""

    model_config = ConfigDict(frozen=True)

    selected_threshold_policy: str
    final_threshold: float
    rationale: str
    decision_rule: str
    optimised: bool
    full_training_out_of_fold_operating_point: OperatingPointRecord
    operating_point_label: str
    f1_max_on_full_training_out_of_fold: OperatingPointRecord
    f1_max_adopted: bool
    phase9c_input: str


class ThresholdStabilityRecord(BaseModel):
    """Fold-to-fold dispersion of the selected thresholds."""

    model_config = ConfigDict(frozen=True)

    per_fold: list[float]
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    spread: float
    n_distinct: int = Field(ge=1)
    role: str


class RecallScenarioRecord(BaseModel):
    """A hypothetical operating point reaching a given recall. Descriptive only."""

    model_config = ConfigDict(frozen=True)

    target_recall: float
    attainable: bool
    threshold: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    predicted_positive_rate: float | None


class TopKRecordModel(BaseModel):
    """One hypothetical contact-capacity level. Descriptive only."""

    model_config = ConfigDict(frozen=True)

    fraction: float
    n_contacted: int = Field(ge=0)
    churners_captured: int = Field(ge=0)
    recall: float
    precision: float
    lift: float
    probability_at_cut: float | None


class CurvePoint(BaseModel):
    """One point of the reporting grid of the operating curve."""

    model_config = ConfigDict(frozen=True)

    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positive_rate: float


class CostFramework(BaseModel):
    """The conceptual cost model, with no numbers in it.

    The formula is recorded because it is what makes the limitation precise: the
    cost-optimal threshold is a function of two quantities this dataset does not
    contain, so no cost-optimal threshold can be computed here. Both cost fields
    are ``null`` and must stay that way until real figures exist.
    """

    model_config = ConfigDict(frozen=True)

    formula: str
    cost_false_negative: float | None
    cost_false_positive: float | None
    optimal_threshold_computed: bool
    note: str


class ThresholdResults(BaseModel):
    """The complete Phase 9B record."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    experiment: str
    raw_sha256: str = Field(min_length=64, max_length=64)
    training_ids_sha256: str = Field(min_length=64, max_length=64)
    n_training_rows: int = Field(ge=1)
    training_positive_prevalence: float = Field(gt=0.0, lt=1.0)
    random_seed: int = Field(ge=0)
    scikit_learn_version: str
    references: dict[str, ArtefactReference]
    upstream_artefact_digests: dict[str, str]
    reproduction: list[ReproductionCheck]
    frozen_estimator: dict[str, object]
    frozen_preprocessing: dict[str, object]
    feature_set: dict[str, object]
    calibration_policy: str
    probability_source: dict[str, object]
    outer_cross_validation: dict[str, object]
    inner_threshold_cross_validation: dict[str, object]
    threshold_policy: dict[str, object]
    tie_breaking_rule: list[str]
    default_threshold: float
    numeric_precision: dict[str, object]
    outer_thresholds: list[OuterThresholdRecord]
    policies: list[ThresholdPolicyRecord]
    ranking_invariance: dict[str, object]
    eligibility: EligibilityRecord
    selection: SelectionRecord
    threshold_stability: ThresholdStabilityRecord
    hypothetical_recall_scenarios: dict[str, object]
    top_k_capacity_analysis: dict[str, object]
    operating_curve: dict[str, object]
    cost_framework: CostFramework
    class_weight: str | None
    engineered_features_used: list[str]
    holdout_touched: bool
    warnings: list[WarningRecord]
    methodological_notes: list[str]


THRESHOLD_POLICY_RULE: tuple[str, ...] = (
    "Candidates are every distinct predicted probability in the vector being searched, plus the "
    "default 0.5, which is added unconditionally so the procedure can return 'the default was "
    "already best'.",
    "The decision rule is: predict the positive class when probability >= threshold.",
    "Each candidate is scored by the F1 score of the positive class, and by nothing else.",
    "The candidate with the maximum F1 is selected.",
    "Ties are resolved by the tie-breaking rule, not by argmax over an array order.",
    "Probabilities are never rounded before the search: rounding would merge cuts that separate "
    "different customers.",
)

TIE_BREAKING_RULE: tuple[str, ...] = (
    f"Two candidates count as tied when their F1 values are within {TIE_TOLERANCE:.0e} of the "
    "maximum. 'Equal' is not a well-defined operation on floats, so the tolerance is explicit.",
    "Among the tied candidates, the threshold closest to 0.5 is selected. This is a PARSIMONY "
    "rule: when the evidence does not distinguish two operating points, the one that moves less "
    "from the default is taken. It is not a claim that 0.5 is correct and it is not a business "
    "conclusion.",
    "If two tied candidates are exactly equidistant from 0.5, the larger threshold is selected. "
    "This second tie-break exists only to make the answer unique and deterministic.",
)

ELIGIBILITY_RULE: tuple[str, ...] = (
    "The F1-maximisation procedure (D1) is ELIGIBLE_THRESHOLD_POLICY to replace the default "
    "threshold when: (1) its mean paired delta F1 against D0 across the outer folds is > 0; and "
    f"(2) F1 improves STRICTLY in at least {MIN_IMPROVED_FOLDS} of the 5 outer folds.",
    "A fold in which both arms scored identically is a TIE, never an improvement. A tie is "
    "exactly what happens when the search returns the default, and counting it as a win would "
    "let a procedure that changed nothing look like one that helped.",
    "There is no minimum-gain cutoff. Magnitude is reported separately and weighed by a reader.",
    "This is an engineering heuristic about direction and consistency, not a significance test. "
    "Five folds cannot support one and none was run.",
    "If D1 is not eligible, the selected policy is DEFAULT_0_5 and the final threshold is 0.5.",
    "If D1 is eligible, the selected policy is F1_MAXIMIZATION and the final threshold comes from "
    "applying the same POLICY_F1 once to the out-of-fold probabilities of the whole training "
    "pool. That application is a FREEZE, not a new performance estimate.",
)

METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "Every number in this artefact is cross-validated on the frozen Phase 4 training pool. No "
    "metric, threshold, curve, scenario or count involves the holdout, which stays untouched.",
    "F1-max is a RESEARCH OPERATING POINT, not a business-optimal threshold. This dataset carries "
    "no cost of losing a customer and no cost of a retention contact, so no economically optimal "
    "threshold can be computed and none is claimed.",
    "The selection metric was pre-registered as F1 of the positive class before any Phase 9B "
    "number was read, and was not changed after the results were seen.",
    "The threshold-selection PROCEDURE is what the nested design evaluates, not one threshold. "
    "Inside each outer training fold an inner 4-fold produces an out-of-sample probability for "
    "every outer-train row, POLICY_F1 reads only those probabilities, and the resulting threshold "
    "is applied to the outer validation fold without any further adjustment. No outer-validation "
    "row takes part in a fit, in the probabilities the search reads, or in the F1 it maximises.",
    "A per-fold threshold that moves across folds is not a defect of the estimate. It is part of "
    "what the nested design measures, and its dispersion is reported as such.",
    "D0 and D1 read the SAME outer probability vector and differ only by where it is cut. Average "
    "Precision and ROC-AUC are invariant to the threshold, so they must be identical between the "
    "arms; the run gates on that identity rather than asserting it.",
    "The final threshold is fixed by applying POLICY_F1 to the out-of-fold probabilities of the "
    "whole training pool. Its metrics are OPERATING-POINT SELECTION METRICS, computed on the same "
    "rows the threshold was chosen from, and are optimistically biased by construction. They are "
    "not performance and must never be relabelled as such.",
    "The recall scenarios are hypothetical and descriptive. No stakeholder in this project has "
    "stated a recall target, and no scenario can select the frozen threshold.",
    "The top-k table is a hypothetical capacity analysis. It uses only the descending order of "
    "the probabilities, no evidence exists that a real retention team has any of these "
    "capacities, and it does not select the frozen threshold.",
    "Lift is precision divided by the training-pool prevalence, the only base rate available "
    "here. No external or historical base rate was used.",
    "A threshold is applied AFTER the fit. Moving it changes decisions, not coefficients, which "
    "is why it can be chosen at this stage without reopening model selection. class_weight would "
    "not be: it changes the objective the model is fitted on, so it stays None and stays closed.",
    "The calibration policy remains NONE. The threshold acts directly on predict_proba(X)[:, 1] "
    "of the frozen, uncalibrated logistic regression.",
    "The estimator, its hyperparameters, the 19 original features and the preprocessing are "
    "frozen inputs. Nothing here reopens model selection, tuning or calibration.",
    "Threshold selection is an explicit, auditable implementation. TunedThresholdClassifierCV was "
    "deliberately not used: the candidate set, the decision rule, the objective and the "
    "tie-breaking all had to be visible and testable rather than delegated to a default.",
    "Thresholds are recorded unrounded; metrics are rounded to six decimals. A metric rounded in "
    "the seventh decimal is the same measurement, but a threshold rounded in the sixth can move "
    "customers across the boundary.",
    "The operating curve in this artefact is tabulated on a uniform reporting grid. The search "
    "itself never used that grid: it read every distinct probability.",
    "No fitted estimator was persisted. This phase persists the protocol, the metrics and the "
    "decision.",
)


def _operating_point(point: OperatingPoint) -> OperatingPointRecord:
    return OperatingPointRecord(
        threshold=float(point.threshold),
        precision=_round(point.precision),
        recall=_round(point.recall),
        f1=_round(point.f1),
        accuracy=_round(point.accuracy),
        predicted_positive_rate=_round(point.predicted_positive_rate),
        true_negatives=point.true_negatives,
        false_positives=point.false_positives,
        false_negatives=point.false_negatives,
        true_positives=point.true_positives,
    )


def _delta(delta: object) -> ThresholdDeltaRecord:
    return ThresholdDeltaRecord(
        metric=delta.metric,
        reference=delta.reference,
        per_fold=[_round(value) for value in delta.deltas],
        mean=_round(delta.mean),
        std=_round(delta.std),
        folds_higher=delta.n_higher,
        folds_tied=delta.n_tied,
        folds_lower=delta.n_lower,
    )


#: Metrics summarised per policy. The confusion cells are recorded per fold but
#: not averaged: a mean true-positive count across folds describes the fold sizes
#: as much as the decision rule.
SUMMARY_METRICS: tuple[str, ...] = DECISION_METRIC_NAMES + GUARDRAIL_METRIC_NAMES


def _policy(run: ThresholdRun, threshold_source: str) -> ThresholdPolicyRecord:
    return ThresholdPolicyRecord(
        experiment=run.experiment,
        label=run.label,
        model=run.model,
        policy=run.policy,
        threshold_source=threshold_source,
        folds=[
            ThresholdFoldRecord(
                fold=fold.fold,
                n_train=fold.n_train,
                n_validation=fold.n_validation,
                threshold=float(fold.threshold),
                metrics={metric: _round(fold.metrics.value(metric)) for metric in SUMMARY_METRICS},
                confusion={metric: int(fold.metrics.value(metric)) for metric in CONFUSION_NAMES},
            )
            for fold in run.folds
        ],
        mean={metric: _round(run.mean(metric)) for metric in SUMMARY_METRICS},
        std={metric: _round(run.std(metric)) for metric in SUMMARY_METRICS},
        paired_deltas={
            reference: {metric: _delta(delta) for metric, delta in deltas.items()}
            for reference, deltas in run.deltas.items()
        },
    )


def build_eligibility_record(status: str, rationale: str, delta: object) -> EligibilityRecord:
    """Wrap the eligibility decision, with the quantities the rule read."""
    return EligibilityRecord(
        rule=list(ELIGIBILITY_RULE),
        rule_fixed_before_results=True,
        min_improved_folds=MIN_IMPROVED_FOLDS,
        minimum_gain_cutoff=None,
        decision_metric=SELECTION_METRIC,
        status=status,
        rationale=rationale,
        mean_delta_f1=_round(delta.mean),
        folds_f1_improved=delta.n_higher,
        folds_f1_tied=delta.n_tied,
        folds_f1_worsened=delta.n_lower,
    )


def build_selection_record(
    policy: str,
    final_threshold: float,
    rationale: str,
    operating_point: OperatingPoint,
    f1_max_point: OperatingPoint,
) -> SelectionRecord:
    """Wrap the frozen decision policy carried into Phase 9C."""
    adopted = policy == F1_POLICY
    return SelectionRecord(
        selected_threshold_policy=policy,
        final_threshold=float(final_threshold),
        rationale=rationale,
        decision_rule="predict churn when predict_proba(X)[:, 1] >= final_threshold",
        optimised=adopted,
        full_training_out_of_fold_operating_point=_operating_point(operating_point),
        operating_point_label=(
            "full-training out-of-fold operating-point selection metrics. Computed on the "
            "training pool, on the same rows any threshold search read, and therefore "
            "optimistically biased. NOT final performance: no performance figure exists in this "
            "repository until the holdout is evaluated"
        ),
        f1_max_on_full_training_out_of_fold=_operating_point(f1_max_point),
        f1_max_adopted=adopted,
        phase9c_input=(
            "Phase 9C freezes the estimator, the calibration policy NONE and this threshold into "
            "one decision policy; Phase 9D then evaluates the untouched holdout exactly once"
        ),
    )


def build_stability_record(stability: ThresholdStability) -> ThresholdStabilityRecord:
    """Wrap the threshold dispersion, with its role stated."""
    return ThresholdStabilityRecord(
        per_fold=[float(value) for value in stability.thresholds],
        mean=float(stability.mean),
        median=float(stability.median),
        std=float(stability.std),
        minimum=float(stability.minimum),
        maximum=float(stability.maximum),
        spread=float(stability.spread),
        n_distinct=stability.n_distinct,
        role=(
            "descriptive. Dispersion is reported and, where wide, documented as a limitation. It "
            "is NOT a post-hoc criterion: the eligibility rule was fixed before these thresholds "
            "existed and is not modified by them"
        ),
    )


def _recall_scenario(scenario: RecallScenario) -> RecallScenarioRecord:
    point = scenario.operating_point
    return RecallScenarioRecord(
        target_recall=float(scenario.target_recall),
        attainable=scenario.attainable,
        threshold=None if point is None else float(point.threshold),
        precision=None if point is None else _round(point.precision),
        recall=None if point is None else _round(point.recall),
        f1=None if point is None else _round(point.f1),
        predicted_positive_rate=None if point is None else _round(point.predicted_positive_rate),
    )


def _top_k(record: TopKRecord) -> TopKRecordModel:
    return TopKRecordModel(
        fraction=float(record.fraction),
        n_contacted=record.n_contacted,
        churners_captured=record.churners_captured,
        recall=_round(record.recall),
        precision=_round(record.precision),
        lift=_round(record.lift),
        probability_at_cut=(
            None if record.probability_at_cut is None else float(record.probability_at_cut)
        ),
    )


def _curve(sweep: ThresholdSweep) -> list[CurvePoint]:
    return [
        CurvePoint(
            threshold=_round(float(sweep.thresholds[index])),
            precision=_round(float(sweep.precision[index])),
            recall=_round(float(sweep.recall[index])),
            f1=_round(float(sweep.f1[index])),
            predicted_positive_rate=_round(float(sweep.predicted_positive_rate[index])),
        )
        for index in range(len(sweep))
    ]


def build_results(
    evaluation: object,
    selections: Sequence[InnerThresholdSelection],
    reproduction: Sequence[ReproductionCheck],
    ranking_invariance_difference: float,
    eligibility_record: EligibilityRecord,
    selection: SelectionRecord,
    stability: ThresholdStabilityRecord,
    scenarios: Sequence[RecallScenario],
    top_k: Sequence[TopKRecord],
    curve: ThresholdSweep,
    references: Mapping[str, ArtefactReference],
    digests: Mapping[str, str],
    raw_sha256: str,
    training_ids_sha256: str,
    target: np.ndarray,
    captured: Sequence[object],
) -> ThresholdResults:
    """Assemble the Phase 9B record."""
    config = get_config()
    labels = np.asarray(target)
    estimator = build_modern_logistic()

    return ThresholdResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        raw_sha256=raw_sha256,
        training_ids_sha256=training_ids_sha256,
        n_training_rows=int(labels.size),
        training_positive_prevalence=_round(float(labels.mean())),
        random_seed=config.seed,
        scikit_learn_version=sklearn.__version__,
        references=dict(references),
        upstream_artefact_digests=dict(digests),
        reproduction=list(reproduction),
        frozen_estimator={
            "builder": "churn.modeling.tuning.build_modern_logistic",
            "estimator": "LogisticRegression",
            "selected_in": "phase8b-nested-hyperparameter-tuning",
            "hyperparameters": {
                key: _jsonable(value)
                for key, value in sorted(estimator.get_params().items())
                if key in {"C", "l1_ratio", "solver", "class_weight", "max_iter"}
            },
            "reopened_here": False,
        },
        frozen_preprocessing={
            "builder": "churn.features.pipeline.build_feature_preprocessor",
            "steps": [
                "prepare_features (contract validation and canonical column order)",
                "TotalChargesCleaner",
                "StandardScaler on the 3 numeric features",
                "OneHotEncoder(handle_unknown='ignore') on the 16 categorical features",
            ],
            "n_transformed_features": 46,
            "refitted_inside_every_fold": True,
        },
        feature_set={
            "n_features": len(FEATURE_COLUMNS),
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
            "engineered": [],
        },
        calibration_policy=EXPECTED_CALIBRATION_POLICY,
        probability_source={
            "expression": "build_modern_logistic_pipeline().predict_proba(X)[:, 1]",
            "calibrated": False,
            "decided_in": "phase9a-calibration-gate",
            "reference_experiment": REFERENCE_EXPERIMENT,
            "note": (
                "The threshold cuts the raw probability of the frozen logistic regression. A "
                "calibrator would have rewritten that score, and a threshold chosen on one is not "
                "the same operating point on the other, which is why Phase 9A ran first"
            ),
        },
        outer_cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": N_SPLITS,
            "shuffle": SHUFFLE,
            "random_state": config.seed,
            "shared_across_policies": True,
            "identical_to_previous_phases": True,
            "role": (
                "evaluates the threshold-selection procedure; never takes part in choosing a "
                "threshold"
            ),
        },
        inner_threshold_cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": THRESHOLD_INNER_N_SPLITS,
            "shuffle": THRESHOLD_INNER_SHUFFLE,
            "random_state": config.seed,
            "scope": "inside each outer training fold only",
            "role": (
                "produces out-of-sample probabilities for every outer-train row; POLICY_F1 reads "
                "those and nothing else"
            ),
        },
        threshold_policy={
            "name": "POLICY_F1",
            "objective": SELECTION_METRIC,
            "objective_class": "positive class (Churn = Yes)",
            "pre_registered": True,
            "rule": list(THRESHOLD_POLICY_RULE),
            "rationale": (
                "Precision and recall are both relevant and F1 weighs them symmetrically, which "
                "is a defensible default when no cost ratio is available. It requires no invented "
                "business number"
            ),
            "scope_statement": (
                "F1-max is a research operating point, not a business-optimal threshold"
            ),
            "changed_after_seeing_results": False,
        },
        tie_breaking_rule=list(TIE_BREAKING_RULE),
        default_threshold=float(DEFAULT_THRESHOLD),
        numeric_precision={
            "metrics_rounded_to": _PRECISION,
            "thresholds_rounded": False,
            "probabilities_rounded_before_search": False,
            "note": (
                "A metric rounded in the seventh decimal is the same measurement; a threshold "
                "rounded in the sixth can move customers across the decision boundary"
            ),
        },
        outer_thresholds=[
            OuterThresholdRecord(
                fold=item.fold,
                n_outer_train=item.n_outer_train,
                n_inner_splits=item.n_inner_splits,
                n_candidates=item.selection.n_candidates,
                n_tied_within_tolerance=item.selection.n_tied_within_tolerance,
                default_in_candidates=item.selection.default_in_candidates,
                threshold=float(item.threshold),
                inner_out_of_fold_selection_metrics=_operating_point(
                    item.selection.operating_point
                ),
            )
            for item in selections
        ],
        policies=[
            _policy(evaluation.default_run, "fixed at 0.5; not optimised"),
            _policy(
                evaluation.selected_run,
                "POLICY_F1 on the inner out-of-fold probabilities of the outer training fold",
            ),
        ],
        ranking_invariance={
            "metrics": list(GUARDRAIL_METRIC_NAMES),
            "expectation": (
                "identical between D0 and D1: both arms read the same outer probability vector "
                "and neither metric depends on a threshold"
            ),
            "max_absolute_difference": ranking_invariance_difference,
            "holds": ranking_invariance_difference == 0.0,
            "gate": "the run aborts if any outer fold disagrees",
        },
        eligibility=eligibility_record,
        selection=selection,
        threshold_stability=stability,
        hypothetical_recall_scenarios={
            "status": "hypothetical operating scenarios",
            "decisional": False,
            "targets": list(RECALL_TARGETS),
            "rule": (
                "recall is non-increasing in the threshold, so the LARGEST threshold reaching the "
                "target is also the most precise one that does; that is the threshold reported"
            ),
            "source": "out-of-fold probabilities of the whole training pool",
            "note": (
                "No stakeholder in this project has stated a recall target. These rows describe "
                "what the trade-off looks like elsewhere on the curve and cannot select the "
                "frozen threshold"
            ),
            "scenarios": [_recall_scenario(scenario).model_dump() for scenario in scenarios],
        },
        top_k_capacity_analysis={
            "status": "hypothetical capacity analysis",
            "decisional": False,
            "fractions": list(TOP_K_FRACTIONS),
            "ordering": "descending probability; ties broken by ascending row position",
            "n_contacted_rule": "floor(fraction * n_training_rows), so a budget is never overspent",
            "lift_baseline": "training-pool positive prevalence",
            "source": "out-of-fold probabilities of the whole training pool",
            "note": (
                "No evidence exists that a real retention team has any of these capacities. The "
                "probability at the cut is reported so the ranking view and the threshold view "
                "can be compared, never so one can be substituted for the other"
            ),
            "levels": [_top_k(record).model_dump() for record in top_k],
        },
        operating_curve={
            "source": "out-of-fold probabilities of the whole training pool",
            "grid": "uniform, reporting only",
            "n_points": CURVE_GRID_POINTS,
            "note": (
                "The search read every distinct probability, not this grid. The grid exists so the "
                "curve can be tabulated at a fixed size"
            ),
            "points": [point.model_dump() for point in _curve(curve)],
        },
        cost_framework=CostFramework(
            formula="ExpectedCost(t) = cost_FN * FN(t) + cost_FP * FP(t)",
            cost_false_negative=None,
            cost_false_positive=None,
            optimal_threshold_computed=False,
            note=(
                "Recorded to make the limitation precise, not to be evaluated. Choosing values "
                "for the two costs would produce a number that looks like an economic optimum "
                "and whose entire content is the assumption. No cost-optimal threshold is "
                "computed or claimed anywhere in this phase"
            ),
        ),
        class_weight=None,
        engineered_features_used=[],
        holdout_touched=False,
        warnings=[
            WarningRecord(
                category=item.category,
                message=item.message,
                source=item.source,
                count=item.count,
            )
            for item in captured
        ],
        methodological_notes=list(METHODOLOGICAL_NOTES),
    )


def write_results(results: ThresholdResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote threshold results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> ThresholdResults:
    """Read and validate a Phase 9B record.

    Raises:
        FileNotFoundError: If it has not been generated yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Threshold results not found at {source}. Generate them with "
            "`uv run python scripts/run_threshold.py`."
        )
    return ThresholdResults.model_validate_json(source.read_text(encoding="utf-8"))
