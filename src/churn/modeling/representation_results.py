"""Machine-readable record of the Phase 8A representation gate.

Anchored to `model_comparison_results.json`: R0 and R1 must reproduce M0 and M2
fold by fold, and the provenance of the artefact they are checked against is
carried here so a reader can tell which numbers this phase subtracted from.

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
    ModelComparisonResults,
    PairedDelta,
    ReproductionCheck,
    WarningRecord,
    verify_reproduction,
)
from churn.modeling.evaluation import CV_STRATEGY, N_SPLITS, SHUFFLE
from churn.modeling.families import HIST_GRADIENT_BOOSTING
from churn.modeling.metrics import DEFAULT_THRESHOLD, METRIC_NAMES
from churn.modeling.representation import (
    CATEGORICAL_MASK,
    COMMON_REPRESENTATION,
    MAX_CATEGORICAL_CARDINALITY,
    NATIVE_HGB,
    NATIVE_REPRESENTATION,
    ORDINAL_ENCODER_PARAMS,
    REFERENCE_HGB_COMMON,
    REFERENCE_LOGISTIC,
    RepresentationRun,
    build_hgb_native,
    categorical_indices,
    classify_representation,
    standing_against_logistic,
)
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "hgb_representation_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase8a-hgb-representation-gate"

#: Experiments of `model_comparison_results.json` that R0 and R1 must reproduce.
REFERENCE_EXPERIMENTS: dict[str, str] = {
    REFERENCE_LOGISTIC: "M0",
    REFERENCE_HGB_COMMON: "M2",
}

_PRECISION = 6


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return str(value)


class OrdinalEncoderConfig(BaseModel):
    """How the categorical columns are coded, and what the codes do not mean."""

    model_config = ConfigDict(frozen=True)

    encoder: str
    parameters: dict[str, object]
    fitted_per_fold: bool
    unknown_category_behaviour: str
    ordinal_meaning: str


class RepresentationDefinition(BaseModel):
    """The representation under test, described structurally."""

    model_config = ConfigDict(frozen=True)

    name: str
    raw_features: int = Field(ge=1)
    numeric_features: list[str]
    categorical_features: list[str]
    n_transformed_features: int = Field(ge=1)
    categorical_mask: list[bool]
    categorical_indices: list[int]
    n_categorical_columns: int = Field(ge=0)
    n_numeric_columns: int = Field(ge=0)
    categorical_features_declared_explicitly: bool
    max_categorical_cardinality: int = Field(ge=1)
    observed_training_cardinality: dict[str, int]
    engineered_features: list[str]


class ExperimentRecord(BaseModel):
    """One experiment of the gate, with deltas keyed by reference experiment."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    label: str
    model: str
    representation: str
    n_transformed_features: int = Field(ge=1)
    folds: list[FoldMetrics]
    mean: dict[str, float]
    std: dict[str, float]
    out_of_fold: dict[str, float]
    paired_deltas: dict[str, dict[str, PairedDelta]]
    converged: bool
    max_iterations_used: int | None


class RepresentationVerdict(BaseModel):
    """The two readings this phase produces, and the decision it does not take."""

    model_config = ConfigDict(frozen=True)

    representation_classification: str
    representation_rationale: str
    standing_against_logistic: str
    standing_rationale: str
    tuning_candidate_selected: bool
    decision_note: str


class RepresentationResults(BaseModel):
    """The complete Phase 8A record."""

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
    cross_validation: dict[str, object]
    metric_protocol: dict[str, object]
    decision_threshold: float
    tuning_performed: bool
    calibration_performed: bool
    engineered_features_used: list[str]
    reproduction: list[ReproductionCheck]
    representations: dict[str, RepresentationDefinition]
    ordinal_encoder: OrdinalEncoderConfig
    hgb_effective_hyperparameters: dict[str, dict[str, object]]
    experiments: list[ExperimentRecord]
    verdict: RepresentationVerdict
    warnings: list[WarningRecord]
    methodological_notes: list[str]


METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "Every metric is cross-validated on the frozen Phase 4 training pool. No metric was "
    "computed on the holdout, which stays untouched until Phase 9.",
    "All experiments use the identical StratifiedKFold partition as Phases 5, 6 and 7, so "
    "every delta is a paired per-fold difference.",
    "R0 and R1 are rebuilt through the Phase 7 pipeline builder and checked fold by fold "
    "against model_comparison_results.json before R2 is interpreted.",
    "R2 differs from R1 in exactly one respect: the 16 contracted categorical columns are "
    "ordinal-coded and declared categorical to the estimator instead of one-hot encoded. "
    "Every hyperparameter is the frozen Phase 7 M2 configuration, derived from the same "
    "dictionary rather than restated.",
    "The ordinal codes carry no ordinal meaning. They stay nominal because the same 16 "
    "positions are declared through categorical_features, so the estimator partitions sets "
    "of categories instead of thresholding a number line.",
    "categorical_features is an explicit positional mask. It is never 'from_dtype' and never "
    "inferred from a dtype produced upstream.",
    "The ordinal encoder is fitted inside each training fold. A category seen only in a "
    "validation fold does not enter categories_ and is encoded as NaN, which the estimator's "
    "native missing-value handling routes at split time.",
    "NaN for an unseen category is a serving behaviour for a new but valid category. It is "
    "not blank handling: blanks, None and empty strings are rejected earlier by the Phase 4 "
    "feature contract, and a new valid category must not be confused with invalid input.",
    "Categorical cardinality is checked inside the training fold only, and no column is "
    "grouped, hashed or truncated to fit the estimator's limit.",
    "No hyperparameter was tuned, searched or selected, and no threshold was moved. The 0.5 "
    "rule is a diagnostic reference and selected nothing.",
    "No engineered feature was used. contract_tenure remains a retained candidate from "
    "Phase 6 and is deliberately absent: adding it would mix a representation effect with a "
    "feature effect in one number.",
    "The two comparisons are kept apart. R2 vs R1 measures the representation inside one "
    "family; R2 vs R0 measures where the result stands against the reference model. They "
    "answer different questions and are never summed.",
    "No calibration was performed and no calibration metric was used to rank anything.",
    "Deltas are read on direction and consistency across folds. The between-fold standard "
    "deviation is NOT a bar a gain must clear, and no significance test was run: five folds "
    "cannot support one.",
    "Out-of-fold predictions cover every training row exactly once, always produced by a "
    "model that never saw that row while fitting. They are not test metrics, and no "
    "per-customer prediction is persisted.",
    "This phase selects no tuning candidate. It produces evidence for a human review that "
    "happens before any tuning phase begins.",
)


def verify_against_comparison(
    run: RepresentationRun,
    reference: ModelComparisonResults,
) -> ReproductionCheck:
    """Check R0 or R1 against its Phase 7 experiment, fold by fold."""
    target = REFERENCE_EXPERIMENTS[run.experiment]
    record = next(item for item in reference.main_comparison if item.experiment == target)
    return verify_reproduction(
        experiment=run.experiment,
        evaluation=run.evaluation,
        reference_folds=[fold.metrics for fold in record.folds],
        reference_artefact="reports/experiments/model_comparison_results.json",
        reference_schema_version=reference.schema_version,
        reference_key=f"main_comparison[{record.experiment}]",
    )


def _record(run: RepresentationRun) -> ExperimentRecord:
    summary = run.evaluation.summary()
    return ExperimentRecord(
        experiment=run.experiment,
        label=run.label,
        model=run.model,
        representation=run.representation,
        n_transformed_features=run.n_transformed_features,
        folds=[
            FoldMetrics(
                fold=fold.fold,
                n_train=fold.n_train,
                n_validation=fold.n_validation,
                metrics={metric: _round(getattr(fold.metrics, metric)) for metric in METRIC_NAMES},
            )
            for fold in run.evaluation.folds
        ],
        mean={metric: _round(summary[metric]["mean"]) for metric in METRIC_NAMES},
        std={metric: _round(summary[metric]["std"]) for metric in METRIC_NAMES},
        out_of_fold={
            metric: _round(getattr(run.evaluation.oof_metrics, metric)) for metric in METRIC_NAMES
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
                )
                for metric, comparison in comparisons.items()
            }
            for reference, comparisons in run.paired_deltas.items()
        },
        converged=run.evaluation.converged,
        max_iterations_used=run.evaluation.max_iterations_used,
    )


def _representations(
    runs: Mapping[str, RepresentationRun],
    cardinality: Mapping[str, int],
) -> dict[str, RepresentationDefinition]:
    common = runs[REFERENCE_HGB_COMMON]
    native = runs[NATIVE_HGB]
    return {
        COMMON_REPRESENTATION: RepresentationDefinition(
            name=COMMON_REPRESENTATION,
            raw_features=len(FEATURE_COLUMNS),
            numeric_features=list(NUMERIC_FEATURES),
            categorical_features=list(CATEGORICAL_FEATURES),
            n_transformed_features=common.n_transformed_features,
            categorical_mask=[],
            categorical_indices=[],
            n_categorical_columns=common.n_transformed_features - len(NUMERIC_FEATURES),
            n_numeric_columns=len(NUMERIC_FEATURES),
            categorical_features_declared_explicitly=False,
            max_categorical_cardinality=MAX_CATEGORICAL_CARDINALITY,
            observed_training_cardinality={},
            engineered_features=[],
        ),
        NATIVE_REPRESENTATION: RepresentationDefinition(
            name=NATIVE_REPRESENTATION,
            raw_features=len(FEATURE_COLUMNS),
            numeric_features=list(NUMERIC_FEATURES),
            categorical_features=list(CATEGORICAL_FEATURES),
            n_transformed_features=native.n_transformed_features,
            categorical_mask=list(CATEGORICAL_MASK),
            categorical_indices=list(categorical_indices()),
            n_categorical_columns=sum(CATEGORICAL_MASK),
            n_numeric_columns=len(CATEGORICAL_MASK) - sum(CATEGORICAL_MASK),
            categorical_features_declared_explicitly=True,
            max_categorical_cardinality=MAX_CATEGORICAL_CARDINALITY,
            observed_training_cardinality=dict(cardinality),
            engineered_features=[],
        ),
    }


def build_results(
    runs: Mapping[str, RepresentationRun],
    reproduction: Sequence[ReproductionCheck],
    comparison: ModelComparisonResults,
    cardinality: Mapping[str, int],
    target: np.ndarray,
    captured: Sequence[object],
) -> RepresentationResults:
    """Assemble the Phase 8A record.

    Args:
        runs: Result of :func:`churn.modeling.representation.run_representation_gate`.
        reproduction: The reproduction checks that gated this run.
        comparison: The Phase 7 record, for the inherited protocol and provenance.
        cardinality: Categories observed per categorical column, training pool only.
        target: Encoded training-pool target.
        captured: Warnings raised during the run.

    Returns:
        The validated :class:`RepresentationResults`.
    """
    config = get_config()
    labels = np.asarray(target)
    native = runs[NATIVE_HGB]
    n_folds = len(native.evaluation.folds)

    against_common = native.paired_deltas[REFERENCE_HGB_COMMON][PRIMARY_METRIC]
    against_logistic = native.paired_deltas[REFERENCE_LOGISTIC][PRIMARY_METRIC]
    classification, rationale = classify_representation(against_common, n_folds)
    standing, standing_rationale = standing_against_logistic(against_logistic, n_folds)

    common_hgb = next(item for item in comparison.models if item.family == HIST_GRADIENT_BOOSTING)

    return RepresentationResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        raw_sha256=comparison.raw_sha256,
        training_ids_sha256=comparison.training_ids_sha256,
        n_training_rows=int(labels.size),
        training_positive_prevalence=_round(float(labels.mean())),
        random_seed=config.seed,
        scikit_learn_version=sklearn.__version__,
        references={
            "model_comparison": ArtefactReference(
                artefact="reports/experiments/model_comparison_results.json",
                schema_version=comparison.schema_version,
                experiment=comparison.experiment,
                raw_sha256=comparison.raw_sha256,
                training_ids_sha256=comparison.training_ids_sha256,
            ),
        },
        cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": N_SPLITS,
            "shuffle": SHUFFLE,
            "random_state": config.seed,
            "shared_across_experiments": True,
            "identical_to_model_comparison": True,
        },
        metric_protocol={
            **comparison.metric_protocol,
            "primary_metric_used_here": PRIMARY_METRIC,
            "secondary_metric_used_here": SECONDARY_METRIC,
        },
        decision_threshold=DEFAULT_THRESHOLD,
        tuning_performed=False,
        calibration_performed=False,
        engineered_features_used=[],
        reproduction=list(reproduction),
        representations=_representations(runs, cardinality),
        ordinal_encoder=OrdinalEncoderConfig(
            encoder="OrdinalEncoder",
            parameters={key: _jsonable(value) for key, value in ORDINAL_ENCODER_PARAMS.items()},
            fitted_per_fold=True,
            unknown_category_behaviour=(
                "A category absent from the training fold is encoded as NaN and routed to the "
                "estimator's native missing-value handling for categorical splits. This is "
                "serving behaviour for a new but valid category, not blank handling: blanks are "
                "rejected earlier by the Phase 4 feature contract."
            ),
            ordinal_meaning=(
                "None. The integer codes are an internal representation with no order. They stay "
                "nominal because the same 16 positions are declared through categorical_features, "
                "so the estimator partitions sets of categories rather than thresholding a number "
                "line."
            ),
        ),
        hgb_effective_hyperparameters={
            REFERENCE_HGB_COMMON: dict(common_hgb.effective_hyperparameters),
            NATIVE_HGB: {
                key: _jsonable(value)
                for key, value in sorted(build_hgb_native().get_params().items())
            },
        },
        experiments=[_record(runs[key]) for key in runs],
        verdict=RepresentationVerdict(
            representation_classification=classification,
            representation_rationale=rationale,
            standing_against_logistic=standing,
            standing_rationale=standing_rationale,
            tuning_candidate_selected=False,
            decision_note=(
                "No tuning candidate was selected. This gate measures one factor — the "
                "categorical representation — and the decision about which models, if any, "
                "are worth tuning is deferred to a human review of this evidence."
            ),
        ),
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


def write_results(results: RepresentationResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote representation gate results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> RepresentationResults:
    """Read and validate a Phase 8A record.

    Raises:
        FileNotFoundError: If it has not been generated yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Representation gate results not found at {source}. Generate them with "
            "`uv run python scripts/run_hgb_representation.py`."
        )
    return RepresentationResults.model_validate_json(source.read_text(encoding="utf-8"))
