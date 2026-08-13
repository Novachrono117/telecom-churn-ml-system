"""Machine-readable record of the Phase 7 model-family comparison.

It carries the provenance chain of the two artefacts this phase is anchored to —
the Phase 5 baseline and the Phase 6 ablation — because a Phase 7 delta is only
interpretable next to the numbers it was subtracted from, and because two of
those numbers must be *reproduced*, not merely referenced.

Same two deliberate absences as Phase 5 and Phase 6: **no timestamp**, so the
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
from churn.features.groups import FEATURE_GROUPS
from churn.features.results import AblationResults
from churn.modeling.comparison import CapturedWarning, FamilyRun
from churn.modeling.evaluation import CV_STRATEGY, N_SPLITS, SHUFFLE, ModelEvaluation
from churn.modeling.families import FAMILIES, MAIN_EXPERIMENTS
from churn.modeling.metrics import DEFAULT_THRESHOLD, METRIC_NAMES
from churn.modeling.results import BaselineResults
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "model_comparison_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase7-model-family-comparison"

#: Reference experiment of the Phase 6 record that S0 must reproduce.
E3_EXPERIMENT = "E3"

_PRECISION = 6
_TOLERANCE = 1e-9


class ReproductionError(RuntimeError):
    """An experiment failed to reproduce the frozen numbers it must match."""


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


class ReproductionCheck(BaseModel):
    """Evidence that an experiment matched a frozen artefact fold by fold."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    reference_artefact: str
    reference_schema_version: int
    reference_key: str
    tolerance: float
    max_absolute_difference: float
    reproduced: bool


class ArtefactReference(BaseModel):
    """An artefact this phase reads and is anchored to."""

    model_config = ConfigDict(frozen=True)

    artefact: str
    schema_version: int
    experiment: str
    raw_sha256: str = Field(min_length=64, max_length=64)
    training_ids_sha256: str = Field(min_length=64, max_length=64)


class ModelDefinition(BaseModel):
    """What a family is, and the exact configuration it ran with."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    family: str
    label: str
    estimator: str
    hypothesis: str
    notes: str
    tuned: bool
    effective_hyperparameters: dict[str, object]


class FeatureSet(BaseModel):
    """Which columns entered a set of experiments."""

    model_config = ConfigDict(frozen=True)

    n_features: int = Field(ge=1)
    numeric: list[str]
    categorical: list[str]
    engineered: list[str]


class FoldMetrics(BaseModel):
    """Absolute metrics of one validation fold."""

    model_config = ConfigDict(frozen=True)

    fold: int = Field(ge=1)
    n_train: int = Field(ge=1)
    n_validation: int = Field(ge=1)
    metrics: dict[str, float]


class PairedDelta(BaseModel):
    """Fold-by-fold difference of one metric against a named reference."""

    model_config = ConfigDict(frozen=True)

    metric: str
    reference: str
    per_fold: list[float]
    mean: float
    std: float
    folds_improved: int = Field(ge=0)
    folds_worsened: int = Field(ge=0)


class ExperimentRecord(BaseModel):
    """One experiment: a family on a feature set, with its deltas."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    family: str
    feature_groups: list[str]
    n_input_features: int = Field(ge=1)
    n_transformed_features: int = Field(ge=1)
    folds: list[FoldMetrics]
    mean: dict[str, float]
    std: dict[str, float]
    out_of_fold: dict[str, float]
    paired_deltas: dict[str, PairedDelta]
    converged: bool
    max_iterations_used: int | None


class WarningRecord(BaseModel):
    """A warning the run actually raised."""

    model_config = ConfigDict(frozen=True)

    category: str
    message: str
    source: str
    count: int = Field(ge=1)


class ModelComparisonResults(BaseModel):
    """The complete Phase 7 record."""

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
    main_feature_set: FeatureSet
    sensitivity_feature_set: FeatureSet
    models: list[ModelDefinition]
    reproduction: list[ReproductionCheck]
    main_comparison: list[ExperimentRecord]
    sensitivity_analysis: list[ExperimentRecord]
    warnings: list[WarningRecord]
    methodological_notes: list[str]


METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "Every metric is cross-validated on the frozen Phase 4 training pool. No metric was "
    "computed on the holdout, which stays untouched until Phase 9.",
    "All experiments use the identical StratifiedKFold partition as Phases 5 and 6, so "
    "every delta is a paired per-fold difference.",
    "The representation is held constant across families: the same TotalChargesCleaner, "
    "StandardScaler and OneHotEncoder block, refitted inside every fold. This was chosen "
    "deliberately to isolate the estimator family — varying the encoding together with the "
    "estimator would make a delta unattributable.",
    "A common representation is not necessarily optimal for every family. Standardisation "
    "does not affect tree split points, and HistGradientBoosting can consume raw categories "
    "through categorical_features instead of one-hot columns; neither option was exercised "
    "here. These results therefore estimate the performance of the tested configurations "
    "under a common representation, not the ceiling of each algorithm.",
    "No hyperparameter was tuned, searched or selected. Every configuration is the "
    "declared base configuration of its family, fixed before any result was seen.",
    "HistGradientBoosting runs with early_stopping=False. The default would carve an "
    "internal validation split out of each training fold and stop on it, which is model "
    "selection inside a fold and would make its number incomparable with the others.",
    "No class weighting, resampling or threshold optimisation was applied. The 0.5 rule "
    "is a diagnostic reference, not a decision, and it selected nothing.",
    "No calibration was performed and no calibration metric was used to rank a family. "
    "Calibration is a separate question and belongs to a later phase.",
    "Deltas are read on direction and consistency across folds. The between-fold standard "
    "deviation is NOT a bar a gain must clear: it measures dispersion across partitions, "
    "not the uncertainty of a difference between two models on the same partitions.",
    "No significance test was run. Five folds cannot support one.",
    "The sensitivity analysis pairs each family against ITS OWN main-comparison run, so a "
    "family effect and a feature effect are never summed into one number.",
    "The four Phase 6 candidates that did not pass its heuristic were not reopened here.",
    "Out-of-fold predictions cover every training row exactly once, always produced by a "
    "model that never saw that row while fitting. They are not test metrics.",
    "Associations, not causes: a family ranking churners better says nothing about what "
    "makes a customer churn.",
)


def verify_reproduction(
    experiment: str,
    evaluation: ModelEvaluation,
    reference_folds: Sequence[Mapping[str, float]],
    reference_artefact: str,
    reference_schema_version: int,
    reference_key: str,
    tolerance: float = _TOLERANCE,
) -> ReproductionCheck:
    """Check that an experiment reproduces frozen per-fold metrics.

    The stored values are rounded, so the fresh ones are rounded the same way
    before the comparison and the tolerance then guards the rounding itself.

    Args:
        experiment: Identifier of the experiment being checked, e.g. ``"M0"``.
        evaluation: Its evaluation.
        reference_folds: Per-fold ``{metric: value}`` mappings from the artefact,
            in fold order.
        reference_artefact: Repository-relative path of that artefact.
        reference_schema_version: Its schema version.
        reference_key: What inside it is being reproduced.
        tolerance: Maximum absolute difference tolerated per metric.

    Returns:
        The :class:`ReproductionCheck`.

    Raises:
        ReproductionError: If the fold count differs or any metric differs by
            more than ``tolerance``. Nothing downstream may be interpreted in
            that case: a protocol that no longer reproduces its own references
            cannot attribute a delta to a family or to a feature.
    """
    if len(reference_folds) != len(evaluation.folds):
        raise ReproductionError(
            f"{experiment} produced {len(evaluation.folds)} folds against "
            f"{len(reference_folds)} in {reference_artefact} ({reference_key})."
        )

    largest = 0.0
    offenders: list[str] = []
    for fold, reference in zip(evaluation.folds, reference_folds, strict=True):
        for metric in METRIC_NAMES:
            difference = abs(_round(getattr(fold.metrics, metric)) - reference[metric])
            largest = max(largest, difference)
            if difference > tolerance:
                offenders.append(f"fold {fold.fold} {metric}: {difference:.3e}")

    if offenders:
        raise ReproductionError(
            f"{experiment} does not reproduce {reference_key} in {reference_artefact}. "
            "Do not interpret any model family or sensitivity delta until the cause is "
            "found.\n  " + "\n  ".join(offenders)
        )

    logger.info("%s reproduces %s (max difference %.3e).", experiment, reference_key, largest)
    return ReproductionCheck(
        experiment=experiment,
        reference_artefact=reference_artefact,
        reference_schema_version=reference_schema_version,
        reference_key=reference_key,
        tolerance=tolerance,
        max_absolute_difference=largest,
        reproduced=True,
    )


def verify_baseline(run: FamilyRun, reference: BaselineResults) -> ReproductionCheck:
    """Check M0 against the frozen Phase 5 logistic regression."""
    model = next(item for item in reference.models if item.name == run.family)
    return verify_reproduction(
        experiment=run.experiment,
        evaluation=run.evaluation,
        reference_folds=[fold.metrics for fold in model.folds],
        reference_artefact="reports/experiments/baseline_results.json",
        reference_schema_version=reference.schema_version,
        reference_key=f"models[{model.name}]",
    )


def verify_contract_tenure(run: FamilyRun, reference: AblationResults) -> ReproductionCheck:
    """Check S0 against E3 of the frozen Phase 6 ablation."""
    record = next(item for item in reference.experiments if item.experiment == E3_EXPERIMENT)
    return verify_reproduction(
        experiment=run.experiment,
        evaluation=run.evaluation,
        reference_folds=[fold.metrics for fold in record.folds],
        reference_artefact="reports/experiments/feature_engineering_results.json",
        reference_schema_version=reference.schema_version,
        reference_key=f"experiments[{record.experiment}]",
    )


def _feature_set(feature_groups: Sequence[str]) -> FeatureSet:
    engineered = [column for name in feature_groups for column in FEATURE_GROUPS[name].numeric]
    categorical = [column for name in feature_groups for column in FEATURE_GROUPS[name].categorical]
    return FeatureSet(
        n_features=len(FEATURE_COLUMNS) + len(engineered) + len(categorical),
        numeric=list(NUMERIC_FEATURES),
        categorical=list(CATEGORICAL_FEATURES),
        engineered=engineered + categorical,
    )


def _record(run: FamilyRun) -> ExperimentRecord:
    summary = run.evaluation.summary()
    return ExperimentRecord(
        experiment=run.experiment,
        family=run.family,
        feature_groups=list(run.feature_groups),
        n_input_features=_feature_set(run.feature_groups).n_features,
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
            metric: PairedDelta(
                metric=metric,
                reference=str(run.reference),
                per_fold=[_round(delta) for delta in comparison.deltas],
                mean=_round(comparison.mean),
                std=_round(comparison.std),
                folds_improved=comparison.n_positive,
                folds_worsened=comparison.n_negative,
            )
            for metric, comparison in run.deltas.items()
        },
        converged=run.evaluation.converged,
        max_iterations_used=run.evaluation.max_iterations_used,
    )


def build_results(
    main: Mapping[str, FamilyRun],
    sensitivity: Mapping[str, FamilyRun],
    reproduction: Sequence[ReproductionCheck],
    baseline: BaselineResults,
    ablation: AblationResults,
    target: np.ndarray,
    captured: Sequence[CapturedWarning],
) -> ModelComparisonResults:
    """Assemble the Phase 7 record.

    Args:
        main: Result of :func:`churn.modeling.comparison.run_main_comparison`.
        sensitivity: Result of :func:`churn.modeling.comparison.run_sensitivity`.
        reproduction: The reproduction checks that gated this run.
        baseline: The Phase 5 record, for the inherited protocol and provenance.
        ablation: The Phase 6 record, for provenance.
        target: Encoded training-pool target.
        captured: Warnings raised during the run.

    Returns:
        The validated :class:`ModelComparisonResults`.
    """
    config = get_config()
    labels = np.asarray(target)

    definitions: list[ModelDefinition] = []
    for run in main.values():
        family = FAMILIES[run.family]
        classifier = family.build()
        definitions.append(
            ModelDefinition(
                experiment=MAIN_EXPERIMENTS[run.family],
                family=run.family,
                label=family.label,
                estimator=type(classifier).__name__,
                hypothesis=family.hypothesis,
                notes=family.notes,
                tuned=False,
                effective_hyperparameters={
                    key: _jsonable(value) for key, value in sorted(classifier.get_params().items())
                },
            )
        )

    return ModelComparisonResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        raw_sha256=baseline.raw_sha256,
        training_ids_sha256=baseline.training_ids_sha256,
        n_training_rows=int(labels.size),
        training_positive_prevalence=_round(float(labels.mean())),
        random_seed=config.seed,
        scikit_learn_version=sklearn.__version__,
        references={
            "baseline": ArtefactReference(
                artefact="reports/experiments/baseline_results.json",
                schema_version=baseline.schema_version,
                experiment=baseline.experiment,
                raw_sha256=baseline.raw_sha256,
                training_ids_sha256=baseline.training_ids_sha256,
            ),
            "feature_engineering": ArtefactReference(
                artefact="reports/experiments/feature_engineering_results.json",
                schema_version=ablation.schema_version,
                experiment=ablation.experiment,
                raw_sha256=ablation.raw_sha256,
                training_ids_sha256=ablation.training_ids_sha256,
            ),
        },
        cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": N_SPLITS,
            "shuffle": SHUFFLE,
            "random_state": config.seed,
            "shared_across_models": True,
            "identical_to_baseline": True,
            "identical_to_ablation": True,
        },
        metric_protocol={
            **baseline.metric_protocol.model_dump(),
            "primary_metric_used_here": "average_precision",
            "secondary_metric_used_here": "roc_auc",
        },
        decision_threshold=DEFAULT_THRESHOLD,
        tuning_performed=False,
        calibration_performed=False,
        main_feature_set=_feature_set(()),
        sensitivity_feature_set=_feature_set(
            next(iter(sensitivity.values())).feature_groups if sensitivity else ()
        ),
        models=definitions,
        reproduction=list(reproduction),
        main_comparison=[_record(run) for run in main.values()],
        sensitivity_analysis=[_record(run) for run in sensitivity.values()],
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


def write_results(results: ModelComparisonResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote model comparison results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> ModelComparisonResults:
    """Read and validate a Phase 7 record.

    Raises:
        FileNotFoundError: If it has not been generated yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Model comparison results not found at {source}. Generate them with "
            "`uv run python scripts/run_model_comparison.py`."
        )
    return ModelComparisonResults.model_validate_json(source.read_text(encoding="utf-8"))
