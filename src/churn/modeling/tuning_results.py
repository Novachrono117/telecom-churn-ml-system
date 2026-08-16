"""Machine-readable record of the Phase 8B nested tuning study.

Carries three things a later reader cannot reconstruct from the numbers alone:
the evidence that the logistic **compatibility migration** changed nothing, the
separation between the nested estimate and the final full-training search, and
the pre-registered selection rule with the branch it actually took.

Same two deliberate absences as every earlier phase: **no timestamp**, so the
determinism check is possible, and **no holdout quantity of any kind**.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT, get_config
from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC
from churn.modeling.comparison_results import (
    ArtefactReference,
    FoldMetrics,
    PairedDelta,
    ReproductionCheck,
    ReproductionError,
    WarningRecord,
    verify_reproduction,
)
from churn.modeling.evaluation import CV_STRATEGY, N_SPLITS, SHUFFLE, ModelEvaluation
from churn.modeling.families import HIST_GRADIENT_BOOSTING_PARAMS
from churn.modeling.metrics import DEFAULT_THRESHOLD, METRIC_NAMES
from churn.modeling.results import BaselineResults
from churn.modeling.tuning import (
    INNER_N_SPLITS,
    INNER_SHUFFLE,
    SEARCH_SCORING,
    T1_SEARCH_SPACE,
    T2_FIXED_PARAMS,
    T2_SEARCH_SPACE,
    TUNED_HGB,
    TUNED_LOGISTIC,
    NestedRun,
    build_modern_logistic,
    frozen_hgb_configuration,
    n_candidates,
    selection_frequency,
)
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "tuning_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase8b-nested-hyperparameter-tuning"

#: Model of `baseline_results.json` that T0 must reproduce.
BASELINE_MODEL = "logistic_regression"

_PRECISION = 6
_TOLERANCE = 1e-9


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return str(value)


class MigrationCheck(BaseModel):
    """Evidence that the modern logistic spelling changed nothing."""

    model_config = ConfigDict(frozen=True)

    legacy_configuration: dict[str, object]
    modern_configuration: dict[str, object]
    tolerance: float
    max_metric_difference: float
    max_oof_probability_difference: float
    metrics_equivalent: bool
    probabilities_equivalent: bool
    legacy_emits_deprecation_warning: bool
    modern_emits_deprecation_warning: bool
    classification: str
    note: str


class SearchSpace(BaseModel):
    """One declared grid."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    estimator: str
    parameters: dict[str, list[object]]
    n_candidates: int = Field(ge=1)
    fixed_parameters: dict[str, object]
    scoring: str
    contains_frozen_configuration: bool


class OuterFoldRecord(FoldMetrics):
    """An outer fold, plus what the inner search chose inside it."""

    best_params: dict[str, object] | None = None
    best_inner_score: float | None = None


class ProcedureRecord(BaseModel):
    """One procedure evaluated on the outer folds."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    label: str
    model: str
    tuned: bool
    folds: list[OuterFoldRecord]
    mean: dict[str, float]
    std: dict[str, float]
    outer_out_of_fold: dict[str, float]
    paired_deltas: dict[str, dict[str, PairedDelta]]
    hyperparameter_selection_frequency: dict[str, dict[str, int]]
    converged: bool


class EligibilityRecord(BaseModel):
    """Whether one procedure may replace the frozen baseline."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    status: str
    rationale: str
    mean_paired_delta_ap: float
    outer_folds_improved: int = Field(ge=0)
    outer_folds_unchanged: int = Field(ge=0)
    outer_folds_worsened: int = Field(ge=0)


class FinalSearchRecord(BaseModel):
    """A search over the whole training pool. Not a generalisation estimate."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    best_params: dict[str, object]
    best_cv_score: float
    scoring: str
    is_generalisation_estimate: bool
    note: str


class SelectionRecord(BaseModel):
    """The pre-registered rule and the branch it took."""

    model_config = ConfigDict(frozen=True)

    rule: list[str]
    rule_fixed_before_results: bool
    eligibility_min_positive_folds: int = Field(ge=1)
    minimum_gain_cutoff: float | None
    selected_experiment: str
    selected_label: str
    rationale: str
    frozen_hyperparameters: dict[str, object]
    estimator_implementation: str
    legacy_builder_status: str
    threshold_status: str
    calibration_status: str
    class_weight_status: str
    phase9_sequence: list[str]


class TuningResults(BaseModel):
    """The complete Phase 8B record."""

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
    feature_set: dict[str, object]
    outer_cross_validation: dict[str, object]
    inner_cross_validation: dict[str, object]
    metric_protocol: dict[str, object]
    decision_threshold: float
    threshold_optimised: bool
    class_weight: str | None
    calibration_performed: bool
    engineered_features_used: list[str]
    compatibility_migration: MigrationCheck
    reproduction: list[ReproductionCheck]
    search_spaces: list[SearchSpace]
    procedures: list[ProcedureRecord]
    eligibility: list[EligibilityRecord]
    selection: SelectionRecord
    final_full_training_searches: list[FinalSearchRecord]
    excluded_from_scope: list[dict[str, str]]
    warnings: list[WarningRecord]
    methodological_notes: list[str]


METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "Every metric is cross-validated on the frozen Phase 4 training pool. No metric was "
    "computed on the holdout, which stays untouched until Phase 9.",
    "The outer folds are the identical StratifiedKFold partition used by Phases 5 to 8A, so "
    "every paired delta is a per-fold difference on the same partition.",
    "Nested cross-validation estimates the TUNING PROCEDURE, not a single configuration. The "
    "search runs entirely inside each outer training fold; the outer validation fold is "
    "predicted once, by the estimator the search had already committed to.",
    "The inner cross-validation never sees an outer validation row. Preprocessing is refitted "
    "inside every inner fold, because the estimator handed to GridSearchCV is the complete "
    "pipeline and nothing is pre-fitted before the search.",
    "GridSearchCV selects on average_precision alone. No multi-metric objective was built: "
    "ROC-AUC and the 0.5 diagnostics accompany the evaluation and select nothing.",
    "The logistic migration from penalty='l2' to l1_ratio=0.0 is a COMPATIBILITY MIGRATION, "
    "not a model change. It is gated on reproducing the legacy per-fold metrics and the "
    "legacy out-of-fold probabilities within tolerance before any tuning runs.",
    "The legacy builder is preserved and still reproduces the historical artefacts. The "
    "modern builder is used only for the new Phase 8B experiments.",
    "class_weight is None for every candidate and stays frozen there. Cost-sensitive learning "
    "through class weights is DEFERRED and out of scope for the main line of this project: it "
    "changes the fit, so reopening it would reopen model selection rather than extend it. The "
    "threshold changes only the post-training decision rule and is therefore separable. Recall "
    "was not compensated through class weights.",
    "Paired fold counts are reported as improved / tied / worsened. A tie is not an improvement "
    "and never counts towards the eligibility rule, which requires strictly positive deltas; "
    "folds_improved and folds_worsened do not have to sum to the number of folds.",
    "No wall-clock metadata is written into this artefact or into the Phase 8B report. Both are "
    "a pure function of their inputs, so re-running the generator on an unchanged repository "
    "reproduces them byte for byte and any diff is a real change. Git already records when "
    "something was produced.",
    "The decision threshold was not optimised. 0.5 is a diagnostic reference only, and no "
    "threshold search, F1-max, Youden rule or TunedThresholdClassifierCV was used.",
    "No probability was calibrated and no calibration metric ranked anything.",
    "No engineered feature was used, and the categorical representation is the common one-hot "
    "block. The native categorical representation was closed in Phase 8A and is not reopened.",
    "Random forest is out of scope for this phase by decision, recorded in excluded_from_scope. "
    "That is a scoping decision, not a claim that the family is universally inferior.",
    "The eligibility rule and the selection rule were fixed before any Phase 8B result was "
    "read, carry no minimum-gain cutoff, and every branch of the selection rule is exercised "
    "by tests on synthetic inputs.",
    "The final full-training search is NOT a generalisation estimate. Its cross-validated "
    "score is computed on the same rows the selection used and is optimistically biased by "
    "construction; the honest estimate of the procedure is the nested outer result.",
    "Outer out-of-fold predictions cover every training row exactly once, always produced by a "
    "procedure whose entire selection ran without that row. They are nested cross-validated "
    "outer-OOF results on the training pool, never test performance.",
    "No fitted estimator was persisted. This phase persists the protocol, the metrics, the "
    "selected hyperparameters and the decision.",
)


def verify_migration(
    legacy: ModelEvaluation,
    modern: ModelEvaluation,
    legacy_warned: bool,
    modern_warned: bool,
    tolerance: float = _TOLERANCE,
) -> MigrationCheck:
    """Check that the modern logistic spelling reproduces the legacy one.

    Compares **every** per-fold metric and, additionally, the out-of-fold
    probability vectors. Metrics can agree while probabilities differ in a way
    that would surface later, so the stronger check is the one that decides.

    Raises:
        ReproductionError: If either comparison exceeds ``tolerance``. Nothing
            downstream may run in that case: the migration would then be a model
            change wearing the label of a compatibility fix.
    """
    if len(legacy.folds) != len(modern.folds):
        raise ReproductionError(
            f"The migration gate compared {len(modern.folds)} modern folds against "
            f"{len(legacy.folds)} legacy folds."
        )

    max_metric = 0.0
    offenders: list[str] = []
    for old, new in zip(legacy.folds, modern.folds, strict=True):
        for metric in METRIC_NAMES:
            difference = abs(getattr(old.metrics, metric) - getattr(new.metrics, metric))
            max_metric = max(max_metric, difference)
            if difference > tolerance:
                offenders.append(f"fold {old.fold} {metric}: {difference:.3e}")

    max_probability = float(
        np.max(np.abs(np.asarray(legacy.oof_probability) - np.asarray(modern.oof_probability)))
    )
    if max_probability > tolerance:
        offenders.append(f"out-of-fold probabilities: {max_probability:.3e}")

    if offenders:
        raise ReproductionError(
            "The modern logistic regression does not reproduce the frozen legacy one. This is "
            "a model change, not a compatibility migration, and no tuning may be interpreted "
            "until the cause is found.\n  " + "\n  ".join(offenders)
        )

    logger.info(
        "Compatibility migration verified: metrics within %.3e, probabilities within %.3e.",
        max_metric,
        max_probability,
    )
    return MigrationCheck(
        legacy_configuration={
            "C": 1.0,
            "penalty": "l2",
            "class_weight": None,
            "solver": "lbfgs",
            "max_iter": 100,
        },
        modern_configuration={
            key: _jsonable(value)
            for key, value in sorted(build_modern_logistic().get_params().items())
            if key in {"C", "l1_ratio", "class_weight", "solver", "max_iter"}
        },
        tolerance=tolerance,
        max_metric_difference=max_metric,
        max_oof_probability_difference=max_probability,
        metrics_equivalent=True,
        probabilities_equivalent=True,
        legacy_emits_deprecation_warning=legacy_warned,
        modern_emits_deprecation_warning=modern_warned,
        classification="COMPATIBILITY_MIGRATION",
        note=(
            "penalty='l2' and l1_ratio=0.0 describe the same L2 penalty. The legacy builder is "
            "preserved so the historical artefacts keep reproducing; the modern builder is used "
            "for the new experiments and no longer emits the removal warning."
        ),
    )


def verify_baseline_reproduction(run: NestedRun, reference: BaselineResults) -> ReproductionCheck:
    """Check T0 against the frozen Phase 5 logistic regression."""
    model = next(item for item in reference.models if item.name == BASELINE_MODEL)
    return verify_reproduction(
        experiment=run.experiment,
        evaluation=run.as_evaluation(),
        reference_folds=[fold.metrics for fold in model.folds],
        reference_artefact="reports/experiments/baseline_results.json",
        reference_schema_version=reference.schema_version,
        reference_key=f"models[{model.name}]",
    )


def _procedure(run: NestedRun) -> ProcedureRecord:
    evaluation = run.as_evaluation()
    summary = evaluation.summary()
    return ProcedureRecord(
        experiment=run.experiment,
        label=run.label,
        model=run.model,
        tuned=run.tuned,
        folds=[
            OuterFoldRecord(
                fold=fold.fold,
                n_train=fold.n_train,
                n_validation=fold.n_validation,
                metrics={metric: _round(getattr(fold.metrics, metric)) for metric in METRIC_NAMES},
                best_params=(
                    {
                        key.split("__", 1)[-1]: _jsonable(value)
                        for key, value in fold.best_params.items()
                    }
                    if fold.best_params
                    else None
                ),
                best_inner_score=(
                    _round(fold.best_inner_score) if fold.best_inner_score is not None else None
                ),
            )
            for fold in run.folds
        ],
        mean={metric: _round(summary[metric]["mean"]) for metric in METRIC_NAMES},
        std={metric: _round(summary[metric]["std"]) for metric in METRIC_NAMES},
        outer_out_of_fold={
            metric: _round(getattr(run.oof_metrics, metric)) for metric in METRIC_NAMES
        },
        paired_deltas={
            reference: {
                metric: PairedDelta(
                    metric=metric,
                    reference=reference,
                    per_fold=[_round(delta) for delta in comparison.deltas],
                    mean=_round(comparison.mean),
                    std=_round(comparison.std),
                    folds_improved=comparison.n_positive,
                    folds_worsened=comparison.n_negative,
                    folds_unchanged=comparison.n_zero,
                )
                for metric, comparison in comparisons.items()
            }
            for reference, comparisons in run.paired_deltas.items()
        },
        hyperparameter_selection_frequency={
            name: dict(values) for name, values in selection_frequency(run).items()
        },
        converged=run.converged,
    )


def _search_spaces() -> list[SearchSpace]:
    logistic_grid = {key: list(values) for key, values in T1_SEARCH_SPACE.items()}
    hgb_grid = {key: list(values) for key, values in T2_SEARCH_SPACE.items()}
    frozen = frozen_hgb_configuration()
    return [
        SearchSpace(
            experiment=TUNED_LOGISTIC,
            estimator="LogisticRegression",
            parameters=logistic_grid,
            n_candidates=n_candidates(T1_SEARCH_SPACE),
            fixed_parameters={
                "l1_ratio": 0.0,
                "solver": "lbfgs",
                "class_weight": None,
                "max_iter": 100,
            },
            scoring=SEARCH_SCORING,
            contains_frozen_configuration=1.0 in logistic_grid[next(iter(logistic_grid))],
        ),
        SearchSpace(
            experiment=TUNED_HGB,
            estimator="HistGradientBoostingClassifier",
            parameters=hgb_grid,
            n_candidates=n_candidates(T2_SEARCH_SPACE),
            fixed_parameters={
                name: _jsonable(HIST_GRADIENT_BOOSTING_PARAMS[name])
                for name in T2_FIXED_PARAMS
                if name != "random_state"
            }
            | {"random_state": get_config().seed},
            scoring=SEARCH_SCORING,
            contains_frozen_configuration=all(
                frozen[parameter] in values for parameter, values in hgb_grid.items()
            ),
        ),
    ]


def build_results(
    runs: Mapping[str, NestedRun],
    migration: MigrationCheck,
    reproduction: Sequence[ReproductionCheck],
    eligibility_records: Sequence[EligibilityRecord],
    selection: SelectionRecord,
    final_searches: Sequence[FinalSearchRecord],
    baseline: BaselineResults,
    comparison_metric_protocol: Mapping[str, object],
    references: Mapping[str, ArtefactReference],
    target: np.ndarray,
    captured: Sequence[object],
) -> TuningResults:
    """Assemble the Phase 8B record."""
    config = get_config()
    labels = np.asarray(target)

    return TuningResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        raw_sha256=baseline.raw_sha256,
        training_ids_sha256=baseline.training_ids_sha256,
        n_training_rows=int(labels.size),
        training_positive_prevalence=_round(float(labels.mean())),
        random_seed=config.seed,
        scikit_learn_version=sklearn.__version__,
        references=dict(references),
        feature_set={
            "n_features": len(FEATURE_COLUMNS),
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
            "engineered": [],
            "representation": "common_ohe",
            "n_transformed_features": 46,
        },
        outer_cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": N_SPLITS,
            "shuffle": SHUFFLE,
            "random_state": config.seed,
            "shared_across_procedures": True,
            "identical_to_previous_phases": True,
            "role": "outer estimate of the tuning procedure; never used for selection",
        },
        inner_cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": INNER_N_SPLITS,
            "shuffle": INNER_SHUFFLE,
            "random_state": config.seed,
            "scope": "inside each outer training fold only",
            "role": "hyperparameter selection",
        },
        metric_protocol={
            **comparison_metric_protocol,
            "primary_metric_used_here": PRIMARY_METRIC,
            "secondary_metric_used_here": SECONDARY_METRIC,
            "search_selection_metric": SEARCH_SCORING,
        },
        decision_threshold=DEFAULT_THRESHOLD,
        threshold_optimised=False,
        class_weight=None,
        calibration_performed=False,
        engineered_features_used=[],
        compatibility_migration=migration,
        reproduction=list(reproduction),
        search_spaces=_search_spaces(),
        procedures=[_procedure(runs[key]) for key in runs],
        eligibility=list(eligibility_records),
        selection=selection,
        final_full_training_searches=list(final_searches),
        excluded_from_scope=[
            {
                "family": "random_forest",
                "reason": (
                    "Phase 7 measured a mean paired delta of about -0.0553 AP against the "
                    "logistic regression with 0/5 folds positive, and no later experiment "
                    "produced evidence to reopen it. A scoping decision, not a claim that the "
                    "family is universally inferior."
                ),
            },
            {
                "family": "hist_gradient_boosting_native_categorical",
                "reason": (
                    "Phase 8A classified the native categorical representation "
                    "REPRESENTATION_NOT_SUPPORTED (mean delta about -0.00022 AP, 2/5 folds) and "
                    "it stayed below the logistic regression in 5/5 folds. Including it would "
                    "multiply experimental factors inside a tuning study."
                ),
            },
            {
                "family": "contract_tenure",
                "reason": (
                    "Retained candidate from Phase 6, deliberately unused here so that a "
                    "feature effect is not mixed into a hyperparameter effect."
                ),
            },
        ],
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


def build_eligibility_record(
    experiment: str,
    status: str,
    rationale: str,
    comparison: object,
) -> EligibilityRecord:
    """Wrap one eligibility decision for the artefact."""
    return EligibilityRecord(
        experiment=experiment,
        status=status,
        rationale=rationale,
        mean_paired_delta_ap=_round(comparison.mean),
        outer_folds_improved=comparison.n_positive,
        outer_folds_unchanged=comparison.n_zero,
        outer_folds_worsened=comparison.n_negative,
    )


def build_selection_record(
    selected: str,
    rationale: str,
    frozen_hyperparameters: Mapping[str, object],
    label: str,
) -> SelectionRecord:
    """Wrap the selection decision, with the rule it was taken under."""
    return SelectionRecord(
        rule=[
            "A tuned procedure is ELIGIBLE_TO_REPLACE_BASELINE when its mean paired delta AP "
            "against T0 is > 0 and AP improves in at least 4 of the 5 outer folds, where "
            "'improves' means a strictly positive delta and a tied fold counts as neither.",
            "If neither is eligible, the frozen baseline T0 is kept.",
            "If exactly one is eligible, it is selected.",
            "If both are eligible, compare T2 against T1 directly: a procedure with mean delta "
            "AP > 0 winning at least 4 of 5 outer folds is selected.",
            "Otherwise prefer the logistic regression, for lower complexity and higher "
            "interpretability.",
        ],
        rule_fixed_before_results=True,
        eligibility_min_positive_folds=4,
        minimum_gain_cutoff=None,
        selected_experiment=selected,
        selected_label=label,
        rationale=rationale,
        frozen_hyperparameters={
            key: _jsonable(value) for key, value in dict(frozen_hyperparameters).items()
        },
        estimator_implementation=(
            "churn.modeling.tuning.build_modern_logistic — LogisticRegression(C=1.0, "
            "l1_ratio=0.0, solver='lbfgs', class_weight=None, max_iter=100). This is the "
            "implementation Phase 9 must instantiate. The compatibility gate proved it "
            "reproduces the legacy penalty='l2' configuration exactly: maximum per-fold metric "
            "difference 0.0 and maximum out-of-fold probability difference 0.0."
        ),
        legacy_builder_status=(
            "PRESERVED FOR HISTORICAL REPRODUCTION ONLY. churn.modeling.models."
            "build_logistic_baseline still passes penalty='l2' so the Phase 5 to 8A artefacts "
            "keep reproducing byte for byte. It must not be used for any new experiment."
        ),
        threshold_status="not optimised; 0.5 is a diagnostic reference and is decided in Phase 9",
        calibration_status="not performed; calibration is a separate question for a later phase",
        class_weight_status=(
            "FROZEN at None and DEFERRED. class_weight modifies the fit itself, so reopening it "
            "would reopen model selection and tuning rather than extend them. The threshold "
            "modifies only the decision rule applied after training and can therefore be settled "
            "on its own. Cost-sensitive training is explicitly out of scope for the main line of "
            "this project."
        ),
        phase9_sequence=[
            "A. Analyse and select the decision threshold using the training pool only, through "
            "cross-validation and outer out-of-fold predictions. The holdout takes no part in it.",
            "B. Decide whether calibration is needed using the training pool only, under a "
            "leakage-safe protocol.",
            "C. Freeze the estimator, the preprocessing, the calibration if any, and the "
            "threshold.",
            "D. Only then run the first and only final evaluation on the holdout.",
            "E. Error analysis on the holdout may follow the final evaluation as DESCRIPTIVE "
            "analysis only. It must not feed back into the model, the features, the "
            "hyperparameters, the calibration or the threshold.",
        ],
    )


def write_results(results: TuningResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote tuning results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> TuningResults:
    """Read and validate a Phase 8B record.

    Raises:
        FileNotFoundError: If it has not been generated yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Tuning results not found at {source}. Generate them with "
            "`uv run python scripts/run_tuning.py`."
        )
    return TuningResults.model_validate_json(source.read_text(encoding="utf-8"))
