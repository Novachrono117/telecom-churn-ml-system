"""Machine-readable record of the Phase 9A calibration gate.

Carries three things a later reader cannot reconstruct from the numbers: the
**exact ECE definition** used (there is no canonical one), the separation
between the metrics that decided and the metrics that only watched, and the
pre-registered rules with the branch they took.

It also records the SHA-256 of every upstream artefact this phase was anchored
to. A reference by name says which file was meant; a digest says which bytes
were actually read.

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
from churn.modeling.calibration import (
    AVERAGE_PRECISION,
    BRIER,
    CALIBRATION_ENSEMBLE,
    CALIBRATION_INNER_N_SPLITS,
    CALIBRATION_INNER_SHUFFLE,
    CALIBRATION_METRIC_NAMES,
    DECISION_METRICS,
    ECE_N_BINS,
    ECE_STRATEGY,
    LOG_LOSS,
    LOWER_IS_BETTER,
    MIN_IMPROVED_FOLDS,
    PROBABILITY_METRICS,
    RANKING_METRICS,
    RELIABILITY_N_BINS,
    RELIABILITY_STRATEGY,
    ROC_AUC,
    CalibrationRun,
)
from churn.modeling.comparison_results import (
    ArtefactReference,
    ReproductionCheck,
    ReproductionError,
    WarningRecord,
)
from churn.modeling.evaluation import CV_STRATEGY, N_SPLITS, SHUFFLE
from churn.modeling.tuning import build_modern_logistic
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "calibration_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase9a-calibration-gate"

_PRECISION = 6


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


class FrozenCandidateMismatchError(RuntimeError):
    """This phase would calibrate an estimator other than the frozen candidate."""


#: The Phase 8B decision this phase is only valid downstream of.
EXPECTED_SELECTED_EXPERIMENT = "T0"


def verify_frozen_candidate(tuning_results: object) -> dict[str, object]:
    """Check that the estimator built here is the one Phase 8B froze.

    The Phase 8B record states which candidate won and with which
    hyperparameters. This phase builds its estimator from a factory. If the two
    ever disagree — because the factory changed, or because a later tuning run
    selected something else — then the calibration policy produced here would
    belong to no model in this repository, and it would say nothing about the one
    that goes to the holdout.

    Args:
        tuning_results: The loaded Phase 8B record.

    Returns:
        The verified hyperparameters.

    Raises:
        FrozenCandidateMismatchError: On any disagreement.
    """
    selection = tuning_results.selection
    built = {
        key: value
        for key, value in build_modern_logistic().get_params().items()
        if key in {"C", "l1_ratio", "solver", "class_weight", "max_iter"}
    }
    recorded = dict(selection.frozen_hyperparameters)

    problems: list[str] = []
    if selection.selected_experiment != EXPECTED_SELECTED_EXPERIMENT:
        problems.append(
            f"Phase 8B selected {selection.selected_experiment!r}, but this phase is written for "
            f"{EXPECTED_SELECTED_EXPERIMENT!r}"
        )
    if built != recorded:
        problems.append(f"builder produces {built}, the Phase 8B record froze {recorded}")
    if built.get("class_weight") is not None:
        problems.append(f"class_weight is {built.get('class_weight')!r}; it must stay None")

    if problems:
        raise FrozenCandidateMismatchError(
            "The estimator this phase would calibrate is not the frozen Phase 8B candidate:\n  "
            + "\n  ".join(problems)
        )

    logger.info("Frozen candidate verified against the Phase 8B record.")
    return built


#: The Phase 8B experiment C0 must reproduce. Same estimator, same folds, same
#: data: the ranking metrics have to come out identical, and if they do not,
#: something changed that this phase is not allowed to change.
REFERENCE_EXPERIMENT = "T0"
REPRODUCTION_TOLERANCE = 1e-9


def verify_uncalibrated_reference(
    run: CalibrationRun,
    tuning_results: object,
    tolerance: float = REPRODUCTION_TOLERANCE,
) -> ReproductionCheck:
    """Check C0 against the frozen Phase 8B T0 numbers.

    C0 is the frozen pipeline on the frozen outer folds, which is exactly what
    T0 was. The two ranking metrics recorded by both phases must therefore agree
    to the recorded precision. A disagreement means the estimator, the
    preprocessing or the partition moved, and no calibration delta measured
    against a drifted reference would mean anything.

    Args:
        run: The C0 run.
        tuning_results: The loaded Phase 8B record.
        tolerance: Maximum tolerated absolute difference.

    Returns:
        The :class:`ReproductionCheck`.

    Raises:
        ReproductionError: If any fold differs by more than ``tolerance``.
    """
    reference = next(
        record for record in tuning_results.procedures if record.experiment == REFERENCE_EXPERIMENT
    )

    largest = 0.0
    offenders: list[str] = []
    for fold, stored in zip(run.folds, reference.folds, strict=True):
        for metric in (AVERAGE_PRECISION, ROC_AUC):
            difference = abs(_round(getattr(fold.metrics, metric)) - stored.metrics[metric])
            largest = max(largest, difference)
            if difference > tolerance:
                offenders.append(f"fold {fold.fold} {metric}: {difference:.3e}")

    if offenders:
        raise ReproductionError(
            "C0 does not reproduce the frozen Phase 8B T0 numbers. The uncalibrated reference "
            "has drifted, so no calibration delta measured against it is interpretable.\n  "
            + "\n  ".join(offenders)
        )

    logger.info("C0 reproduces Phase 8B T0 within %.3e.", largest)
    return ReproductionCheck(
        experiment=run.experiment,
        reference_artefact="reports/experiments/tuning_results.json",
        reference_schema_version=tuning_results.schema_version,
        reference_key=f"procedures[{REFERENCE_EXPERIMENT}]",
        tolerance=tolerance,
        max_absolute_difference=largest,
        reproduced=True,
    )


class CalibrationDelta(BaseModel):
    """Per-fold difference of one metric, with its direction stated."""

    model_config = ConfigDict(frozen=True)

    metric: str
    reference: str
    lower_is_better: bool
    per_fold: list[float]
    mean: float
    std: float
    folds_improved: int = Field(ge=0)
    folds_tied: int = Field(ge=0)
    folds_worsened: int = Field(ge=0)


class CalibrationFoldRecord(BaseModel):
    """One outer fold of one policy."""

    model_config = ConfigDict(frozen=True)

    fold: int = Field(ge=1)
    n_train: int = Field(ge=1)
    n_validation: int = Field(ge=1)
    metrics: dict[str, float]


class PolicyRecord(BaseModel):
    """One calibration policy evaluated on the outer folds."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    label: str
    model: str
    method: str | None
    calibrated: bool
    folds: list[CalibrationFoldRecord]
    mean: dict[str, float]
    std: dict[str, float]
    outer_out_of_fold: dict[str, float]
    n_distinct_oof_probabilities: int = Field(ge=1)
    paired_deltas: dict[str, dict[str, CalibrationDelta]]
    converged: bool


class EligibilityRecord(BaseModel):
    """Whether one calibration method earned adoption."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    method: str
    status: str
    rationale: str
    mean_delta_brier: float
    folds_brier_improved: int = Field(ge=0)
    mean_delta_log_loss: float
    folds_log_loss_improved: int = Field(ge=0)
    mean_delta_average_precision: float
    folds_average_precision_worsened: int = Field(ge=0)


class SelectionRecord(BaseModel):
    """The pre-registered rule and the branch it took."""

    model_config = ConfigDict(frozen=True)

    rule: list[str]
    rule_fixed_before_results: bool
    min_improved_folds: int = Field(ge=1)
    minimum_gain_cutoff: float | None
    decision_metrics: list[str]
    guardrail_metrics: list[str]
    selected_policy: str
    selected_label: str
    rationale: str
    brier_stability: dict[str, float]
    threshold_status: str
    phase9b_input: str


class EceDefinition(BaseModel):
    """The exact ECE this phase computed. There is no canonical one."""

    model_config = ConfigDict(frozen=True)

    formula: str
    n_bins: int = Field(ge=1)
    strategy: str
    empty_bin_behaviour: str
    duplicate_edge_behaviour: str
    caveat: str


class CalibrationResults(BaseModel):
    """The complete Phase 9A record."""

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
    ranking_expectations: dict[str, str]
    frozen_estimator: dict[str, object]
    frozen_preprocessing: dict[str, object]
    feature_set: dict[str, object]
    outer_cross_validation: dict[str, object]
    calibration_inner_cross_validation: dict[str, object]
    calibration_methods: list[dict[str, object]]
    metric_protocol: dict[str, object]
    ece_definition: EceDefinition
    reliability_diagram: dict[str, object]
    threshold_selected: bool
    class_weight: str | None
    engineered_features_used: list[str]
    policies: list[PolicyRecord]
    eligibility: list[EligibilityRecord]
    selection: SelectionRecord
    warnings: list[WarningRecord]
    methodological_notes: list[str]


METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "Every metric is cross-validated on the frozen Phase 4 training pool. No metric, "
    "distribution, plot or probability involves the holdout, which stays untouched.",
    "The outer folds are the identical StratifiedKFold partition used by Phases 5 to 8B, so "
    "every paired delta is a per-fold difference on the same partition.",
    "Calibration is CROSS-FITTED. Inside each outer training fold the inner 4-fold produces an "
    "out-of-sample score for every outer-train row, the calibrator is fitted on those scores, "
    "and the base pipeline is then refitted on the whole outer training fold. No row is ever "
    "used to fit the estimator that scores it and the calibrator that transforms that score. A "
    "calibrator fitted on in-sample scores would learn to correct a distortion that does not "
    "exist out of sample.",
    "The whole pipeline sits inside the calibrator, not just the classifier, so the cleaner, the "
    "scaler and the encoder are refitted inside every inner calibration fold.",
    "ensemble=False is pinned rather than left at the 'auto' default. This is what isolates the "
    "question: the base estimator is refitted on the entire outer training fold, exactly as C0 "
    "is, so the arms differ by the calibration layer alone. ensemble=True would instead average "
    "four estimators each fitted on three quarters of the fold, adding a variance-reduction "
    "effect on top of calibration and leaving the comparison unable to attribute the result.",
    "Brier score and log loss are PROPER SCORING RULES of probabilistic quality. They are not "
    "measures of calibration alone: each decomposes into a calibration term and a "
    "refinement/discrimination term, so a movement can come from sharper probabilities rather "
    "than better-aligned ones. The decision reads them together with the reliability diagram, "
    "the ranking guardrails and fold-to-fold consistency.",
    "Average Precision and ROC-AUC are GUARDRAILS only. No calibrator is adopted because a "
    "ranking metric improved.",
    "No threshold-dependent metric was computed at all. Precision, recall, F1 and accuracy need "
    "a decision rule, this phase does not choose one, and a metric that is never computed cannot "
    "accidentally influence the decision.",
    "No threshold was selected. Phase 9B selects it, on the training pool, acting on whatever "
    "probability this phase froze.",
    "The estimator, its hyperparameters, class_weight=None, the 19 original features and the "
    "preprocessing are all frozen inputs. Nothing here reopens model selection.",
    "Expected Calibration Error is reported under an explicitly stated definition because there "
    "is no canonical one: implementations differ in binning, weighting and empty-bin handling. "
    "It is a supporting diagnostic and decided nothing on its own.",
    "Isotonic regression is more flexible than a sigmoid fit and was not assumed to be better. "
    "It is monotone but not strictly increasing, so it can map distinct scores onto one plateau "
    "and introduce tied probabilities, reducing ranking RESOLUTION. Where a lower Average "
    "Precision or ROC-AUC accompanies that reduction, it is reported as a trade-off of the "
    "probabilistic representation and must not be read as a calibration error. The count of "
    "distinct out-of-fold probabilities records how far the collapse went.",
    "Whether a logistic regression needs calibration is an EMPIRICAL question here, not a "
    "property assumed from the family. Its probabilistic structure and log-loss objective often "
    "yield adequately calibrated probabilities, but regularisation, model misspecification and "
    "the characteristics of a dataset can still produce miscalibration. The conclusion recorded "
    "in this artefact is scoped to this dataset and this protocol: additional calibration was, "
    "or was not, supported by the measurements.",
    "The eligibility and selection rules were fixed before any Phase 9A number was read, carry "
    "no minimum-gain cutoff, and every branch is exercised by tests on synthetic inputs.",
    "Outer out-of-fold probabilities cover every training row exactly once, always produced by a "
    "policy whose entire fitting and calibration ran without that row. They are cross-validated "
    "training-pool results, never test performance.",
    "No fitted estimator and no calibrator was persisted. This phase persists the protocol, the "
    "metrics and the decision.",
)


def _delta(delta: object) -> CalibrationDelta:
    return CalibrationDelta(
        metric=delta.metric,
        reference=delta.reference,
        lower_is_better=delta.lower_is_better,
        per_fold=[_round(value) for value in delta.deltas],
        mean=_round(delta.mean),
        std=_round(delta.std),
        folds_improved=delta.n_improved,
        folds_tied=delta.n_tied,
        folds_worsened=delta.n_worsened,
    )


def _policy(run: CalibrationRun) -> PolicyRecord:
    return PolicyRecord(
        experiment=run.experiment,
        label=run.label,
        model=run.model,
        method=run.method,
        calibrated=run.calibrated,
        folds=[
            CalibrationFoldRecord(
                fold=fold.fold,
                n_train=fold.n_train,
                n_validation=fold.n_validation,
                metrics={
                    metric: _round(getattr(fold.metrics, metric))
                    for metric in CALIBRATION_METRIC_NAMES
                },
            )
            for fold in run.folds
        ],
        mean={metric: _round(run.mean(metric)) for metric in CALIBRATION_METRIC_NAMES},
        std={metric: _round(run.std(metric)) for metric in CALIBRATION_METRIC_NAMES},
        outer_out_of_fold={
            metric: _round(getattr(run.oof_metrics, metric)) for metric in CALIBRATION_METRIC_NAMES
        },
        n_distinct_oof_probabilities=run.n_distinct_oof_probabilities,
        paired_deltas={
            reference: {metric: _delta(delta) for metric, delta in deltas.items()}
            for reference, deltas in run.deltas.items()
        },
        converged=run.converged,
    )


def build_eligibility_record(
    experiment: str,
    method: str,
    status: str,
    rationale: str,
    deltas: Mapping[str, object],
) -> EligibilityRecord:
    """Wrap one eligibility decision, with the three quantities the rule read."""
    return EligibilityRecord(
        experiment=experiment,
        method=method,
        status=status,
        rationale=rationale,
        mean_delta_brier=_round(deltas[BRIER].mean),
        folds_brier_improved=deltas[BRIER].n_improved,
        mean_delta_log_loss=_round(deltas[LOG_LOSS].mean),
        folds_log_loss_improved=deltas[LOG_LOSS].n_improved,
        mean_delta_average_precision=_round(deltas[AVERAGE_PRECISION].mean),
        folds_average_precision_worsened=deltas[AVERAGE_PRECISION].n_worsened,
    )


def build_selection_record(
    policy: str,
    label: str,
    rationale: str,
    stability: Mapping[str, float],
) -> SelectionRecord:
    """Wrap the selection decision, with the rule it was taken under."""
    return SelectionRecord(
        rule=[
            "A calibration method is ELIGIBLE_CALIBRATION when: (1) its mean paired delta Brier "
            f"is < 0; (2) Brier improves in at least {MIN_IMPROVED_FOLDS} of the 5 outer folds; "
            "(3) its mean paired delta log loss is <= 0; and (4) Average Precision does not "
            "deteriorate relevantly and consistently.",
            "Condition 4 uses the project's standard consistency test and no magnitude cutoff: "
            "AP counts as deteriorated when its mean delta is negative AND it worsens in at "
            f"least {MIN_IMPROVED_FOLDS} of the 5 outer folds.",
            "If conditions 1 and 2 hold but 3 fails, the two probability losses disagree "
            "materially and the method is INCONCLUSIVE_CALIBRATION, not adopted.",
            "If no method is eligible, the calibration policy is NONE and the uncalibrated "
            "logistic regression is carried forward.",
            "If exactly one method is eligible, it is selected.",
            "If both are eligible, compare them directly on the same folds: first Brier, then "
            f"log loss; a method with a mean delta in its favour winning at least "
            f"{MIN_IMPROVED_FOLDS} of the 5 outer folds is selected.",
            "Otherwise the pre-registered default applies: sigmoid, for lower flexibility and "
            "lower overfitting risk. Stability and complexity are reported but are not turned "
            "into a numeric cutoff.",
        ],
        rule_fixed_before_results=True,
        min_improved_folds=MIN_IMPROVED_FOLDS,
        minimum_gain_cutoff=None,
        decision_metrics=list(DECISION_METRICS),
        guardrail_metrics=list(RANKING_METRICS),
        selected_policy=policy,
        selected_label=label,
        rationale=rationale,
        brier_stability={key: _round(value) for key, value in dict(stability).items()},
        threshold_status=(
            "not selected here. Phase 9B selects the threshold on the training pool, acting on "
            "the probability this phase froze"
        ),
        phase9b_input=(
            "Phase 9B receives the frozen estimator plus this calibration policy, and selects a "
            "threshold on exactly the probability the policy produces"
        ),
    )


RANKING_EXPECTATIONS: dict[str, str] = {
    "C1_sigmoid": (
        "Sigmoid calibration applied to the score of a single estimator is a strictly monotone "
        "transformation, so barring numerical or pathological effects it should not materially "
        "change the ranking. Average Precision and ROC-AUC for C1 against C0 are therefore "
        "checked as an implementation guardrail: a material change would indicate something "
        "other than calibration happened and would have to be investigated before the run is "
        "accepted. Small numerical differences are legitimate and are reported rather than "
        "asserted away."
    ),
    "C2_isotonic": (
        "Isotonic calibration is monotone but NOT strictly increasing: it can map several "
        "distinct scores onto one plateau, introducing tied probabilities and reducing ranking "
        "resolution. Average Precision and ROC-AUC may therefore differ from the uncalibrated "
        "reference. Any such difference is reported as a trade-off of the probabilistic "
        "representation, not as a calibration error, and the count of distinct out-of-fold "
        "probabilities is recorded so the magnitude of the collapse is visible."
    ),
}


def build_results(
    runs: Mapping[str, CalibrationRun],
    reproduction: Sequence[ReproductionCheck],
    eligibility_records: Sequence[EligibilityRecord],
    selection: SelectionRecord,
    references: Mapping[str, ArtefactReference],
    digests: Mapping[str, str],
    raw_sha256: str,
    training_ids_sha256: str,
    target: np.ndarray,
    captured: Sequence[object],
) -> CalibrationResults:
    """Assemble the Phase 9A record."""
    config = get_config()
    labels = np.asarray(target)
    estimator = build_modern_logistic()

    return CalibrationResults(
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
        ranking_expectations=dict(RANKING_EXPECTATIONS),
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
            "refitted_inside_every_inner_calibration_fold": True,
        },
        feature_set={
            "n_features": len(FEATURE_COLUMNS),
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
            "engineered": [],
        },
        outer_cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": N_SPLITS,
            "shuffle": SHUFFLE,
            "random_state": config.seed,
            "shared_across_policies": True,
            "identical_to_previous_phases": True,
            "role": "evaluation of the calibration policy; never used to fit a calibrator",
        },
        calibration_inner_cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": CALIBRATION_INNER_N_SPLITS,
            "shuffle": CALIBRATION_INNER_SHUFFLE,
            "random_state": config.seed,
            "ensemble": CALIBRATION_ENSEMBLE,
            "scope": "inside each outer training fold only",
            "role": (
                "produce out-of-sample scores for the calibrator; the base estimator is then "
                "refitted on the whole outer training fold"
            ),
            "ensemble_rationale": (
                "False isolates the calibration layer: the estimator that predicts is fitted on "
                "the same rows C0 is fitted on. True would average four estimators fitted on "
                "three quarters each, mixing variance reduction into the measured effect"
            ),
        },
        calibration_methods=[
            {
                "experiment": "C0",
                "method": "none",
                "ensemble": None,
                "description": "the frozen pipeline, its own probabilities, no transformation",
            },
            {
                "experiment": "C1",
                "method": "sigmoid",
                "ensemble": CALIBRATION_ENSEMBLE,
                "description": (
                    "Platt scaling: a two-parameter logistic fit mapping score to probability. "
                    "Strictly monotone, and rigid by construction — a limitation and a protection"
                ),
            },
            {
                "experiment": "C2",
                "method": "isotonic",
                "ensemble": CALIBRATION_ENSEMBLE,
                "description": (
                    "a free monotone step function. More flexible, and therefore more able to "
                    "track the particular calibration split it was fitted on. Monotone but not "
                    "strictly increasing, so it can create ties"
                ),
            },
        ],
        metric_protocol={
            "probability_metrics": list(PROBABILITY_METRICS),
            "ranking_guardrail_metrics": list(RANKING_METRICS),
            "decision_metrics": list(DECISION_METRICS),
            "lower_is_better": sorted(LOWER_IS_BETTER),
            "threshold_dependent_metrics_computed": [],
            "note": (
                "Brier score and log loss decide. Average Precision and ROC-AUC watch. A "
                "calibrator improving a ranking metric is not evidence for adopting it: both "
                "methods here are monotone and cannot reorder customers, so these metrics are "
                "read as a check that the ranking survived, never as a gain"
            ),
        },
        ece_definition=EceDefinition(
            formula=(
                "ECE = sum over non-empty bins b of (n_b / N) * |mean(predicted probability in "
                "b) - mean(observed label in b)|"
            ),
            n_bins=ECE_N_BINS,
            strategy=ECE_STRATEGY,
            empty_bin_behaviour="excluded; an empty bin carries weight 0 and contributes nothing",
            duplicate_edge_behaviour=(
                "duplicate quantile edges are collapsed, so the effective number of bins can be "
                "smaller than n_bins when the score distribution has ties"
            ),
            caveat=(
                "There is no canonical ECE. Implementations differ in binning, weighting and "
                "empty-bin handling, so this number is comparable within this report and must "
                "not be compared to an ECE from elsewhere without checking its definition. It is "
                "a supporting diagnostic and decided nothing on its own"
            ),
        ),
        reliability_diagram={
            "n_bins": RELIABILITY_N_BINS,
            "strategy": RELIABILITY_STRATEGY,
            "shared_bins_across_policies": True,
            "source": "outer out-of-fold probabilities on the training pool",
            "note": (
                "The bin configuration was declared before any diagram was drawn. Bins chosen "
                "after seeing which curve looks best would be a presentation decision"
            ),
        },
        threshold_selected=False,
        class_weight=None,
        engineered_features_used=[],
        policies=[_policy(runs[key]) for key in runs],
        eligibility=list(eligibility_records),
        selection=selection,
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


def write_results(results: CalibrationResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote calibration results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> CalibrationResults:
    """Read and validate a Phase 9A record.

    Raises:
        FileNotFoundError: If it has not been generated yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Calibration results not found at {source}. Generate them with "
            "`uv run python scripts/run_calibration.py`."
        )
    return CalibrationResults.model_validate_json(source.read_text(encoding="utf-8"))
