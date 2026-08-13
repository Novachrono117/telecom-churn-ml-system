"""Machine-readable record of the Phase 6 ablation study.

Carries the same provenance chain as the Phase 5 record, plus an explicit
reference to the baseline artefact it is paired against — a delta is only
meaningful next to the numbers it was subtracted from.

No timestamp, for the same reason as in Phase 5: it would change on every run
without adding traceability and would make the determinism check impossible.
No holdout metric of any kind.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT, get_config
from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC, ExperimentResult
from churn.features.groups import FEATURE_GROUPS
from churn.modeling.evaluation import CV_STRATEGY, N_SPLITS, SHUFFLE
from churn.modeling.metrics import DEFAULT_THRESHOLD, METRIC_NAMES
from churn.modeling.results import BaselineResults
from churn.preprocessing.contracts import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "feature_engineering_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase6-feature-engineering-ablation"

_PRECISION = 6


class BaselineReproductionError(RuntimeError):
    """E0 did not reproduce the frozen Phase 5 baseline."""


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


class CandidateDefinition(BaseModel):
    """What a candidate feature is, and what it claims."""

    model_config = ConfigDict(frozen=True)

    group: str
    hypothesis: str
    added_numeric: list[str]
    added_categorical: list[str]
    stateful: bool
    notes: str


class PairedDelta(BaseModel):
    """Fold-by-fold difference of one metric against the baseline."""

    model_config = ConfigDict(frozen=True)

    metric: str
    per_fold: list[float]
    mean: float
    std: float
    folds_improved: int = Field(ge=0)
    folds_worsened: int = Field(ge=0)


class FoldMetrics(BaseModel):
    """Absolute metrics of one validation fold."""

    model_config = ConfigDict(frozen=True)

    fold: int = Field(ge=1)
    metrics: dict[str, float]


class ExperimentRecord(BaseModel):
    """One ablation experiment."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    label: str
    feature_groups: list[str]
    n_input_features: int = Field(ge=1)
    n_transformed_features: int = Field(ge=1)
    folds: list[FoldMetrics]
    mean: dict[str, float]
    std: dict[str, float]
    paired_deltas: dict[str, PairedDelta]
    classification: str | None
    rationale: str
    out_of_fold: dict[str, float] | None


class BaselineReproduction(BaseModel):
    """Evidence that E0 reproduced the frozen Phase 5 numbers."""

    model_config = ConfigDict(frozen=True)

    reference_artefact: str
    reference_schema_version: int
    reference_model: str
    tolerance: float
    max_absolute_difference: float
    reproduced: bool


class AblationResults(BaseModel):
    """The complete Phase 6 record."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    experiment: str
    raw_sha256: str = Field(min_length=64, max_length=64)
    training_ids_sha256: str = Field(min_length=64, max_length=64)
    n_training_rows: int = Field(ge=1)
    training_positive_prevalence: float = Field(gt=0.0, lt=1.0)
    random_seed: int = Field(ge=0)
    scikit_learn_version: str
    cross_validation: dict[str, object]
    metric_protocol: dict[str, object]
    decision_threshold: float
    baseline_reproduction: BaselineReproduction
    candidates: list[CandidateDefinition]
    experiments: list[ExperimentRecord]
    combined_experiment: list[str] | None
    methodological_notes: list[str]


METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "Every metric is cross-validated on the frozen Phase 4 training pool. No metric was "
    "computed on the holdout, which stays untouched until Phase 9.",
    "All experiments use the identical StratifiedKFold partition as the Phase 5 baseline, "
    "so every delta is a paired per-fold difference.",
    "Deltas are compared on direction and consistency across folds. The baseline's "
    "between-fold standard deviation is NOT used as a bar a gain must clear: it measures "
    "dispersion across partitions, not the uncertainty of a difference between two models.",
    "No significance test was run. Five folds cannot support one, and the "
    "PROMISING/INCONCLUSIVE/NOT_SUPPORTED labels are engineering heuristics, not inference.",
    "No minimum-gain cutoff is applied. Magnitude, interpretability and complexity are "
    "reported separately and weighed by a reader.",
    "The classifier is frozen at the Phase 5 configuration for every experiment. No "
    "hyperparameter, threshold or fold assignment was changed to benefit a candidate.",
    "No original feature was removed. This phase tests added representation; feature "
    "selection is a different question and is not mixed in here.",
    "Only charge_intensity is stateful. Its tier medians are learned inside each training "
    "fold; no median from the full-dataset EDA is reused.",
    "No candidate reads the target. Every derived column is a function of the features.",
    "Threshold-dependent metrics at 0.5 are diagnostic and were not used to select any candidate.",
    "Associations, not causes: a feature improving prediction says nothing about whether "
    "it causes churn.",
)


def verify_baseline_reproduction(
    reproduction: ExperimentResult,
    reference: BaselineResults,
    reference_model: str = "logistic_regression",
    tolerance: float = 1e-9,
) -> BaselineReproduction:
    """Check that E0 reproduces the frozen Phase 5 per-fold metrics.

    Args:
        reproduction: The E0 experiment result.
        reference: The Phase 5 record loaded from disk.
        reference_model: Model name inside that record.
        tolerance: Maximum absolute difference tolerated per metric.

    Returns:
        The :class:`BaselineReproduction` evidence.

    Raises:
        BaselineReproductionError: If any per-fold metric differs by more than
            ``tolerance``. The ablation must not be interpreted in that case:
            a protocol that no longer reproduces its own baseline cannot
            attribute a delta to a feature.
    """
    frozen = next(model for model in reference.models if model.name == reference_model)
    if len(frozen.folds) != len(reproduction.evaluation.folds):
        raise BaselineReproductionError(
            f"E0 produced {len(reproduction.evaluation.folds)} folds against "
            f"{len(frozen.folds)} in {reference.experiment}."
        )

    largest = 0.0
    offenders: list[str] = []
    for reference_fold, fold in zip(frozen.folds, reproduction.evaluation.folds, strict=True):
        if reference_fold.fold != fold.fold:
            raise BaselineReproductionError(
                f"Fold identity mismatch: {reference_fold.fold} vs {fold.fold}."
            )
        for metric in METRIC_NAMES:
            # The reference stores rounded values, so E0 is rounded the same way
            # before the comparison; the tolerance then guards the rounding itself.
            difference = abs(_round(getattr(fold.metrics, metric)) - reference_fold.metrics[metric])
            largest = max(largest, difference)
            if difference > tolerance:
                offenders.append(f"fold {fold.fold} {metric}: {difference:.3e}")

    if offenders:
        raise BaselineReproductionError(
            "E0 does not reproduce the frozen Phase 5 baseline. Do not interpret any "
            "candidate feature until the cause is found.\n  " + "\n  ".join(offenders)
        )

    logger.info("E0 reproduces the Phase 5 baseline (max difference %.3e).", largest)
    return BaselineReproduction(
        reference_artefact="reports/experiments/baseline_results.json",
        reference_schema_version=reference.schema_version,
        reference_model=reference_model,
        tolerance=tolerance,
        max_absolute_difference=largest,
        reproduced=True,
    )


def build_results(
    experiments: Mapping[str, ExperimentResult],
    reference: BaselineResults,
    reproduction: BaselineReproduction,
    target: np.ndarray,
    oof_experiments: tuple[str, ...],
) -> AblationResults:
    """Assemble the Phase 6 record.

    Args:
        experiments: Ordered mapping of experiment id to result.
        reference: The Phase 5 record, for the inherited protocol.
        reproduction: Evidence produced by :func:`verify_baseline_reproduction`.
        target: Encoded training-pool target.
        oof_experiments: Experiments whose out-of-fold metrics are recorded.

    Returns:
        The validated :class:`AblationResults`.
    """
    config = get_config()
    labels = np.asarray(target)

    records: list[ExperimentRecord] = []
    for result in experiments.values():
        summary = result.evaluation.summary()
        records.append(
            ExperimentRecord(
                experiment=result.experiment,
                label=result.label,
                feature_groups=list(result.feature_groups),
                n_input_features=len(FEATURE_COLUMNS)
                + sum(
                    len(FEATURE_GROUPS[name].numeric) + len(FEATURE_GROUPS[name].categorical)
                    for name in result.feature_groups
                ),
                n_transformed_features=result.n_transformed_features,
                folds=[
                    FoldMetrics(
                        fold=fold.fold,
                        metrics={
                            metric: _round(getattr(fold.metrics, metric)) for metric in METRIC_NAMES
                        },
                    )
                    for fold in result.evaluation.folds
                ],
                mean={metric: _round(summary[metric]["mean"]) for metric in METRIC_NAMES},
                std={metric: _round(summary[metric]["std"]) for metric in METRIC_NAMES},
                paired_deltas={
                    metric: PairedDelta(
                        metric=metric,
                        per_fold=[_round(delta) for delta in comparison.deltas],
                        mean=_round(comparison.mean),
                        std=_round(comparison.std),
                        folds_improved=comparison.n_positive,
                        folds_worsened=comparison.n_negative,
                    )
                    for metric, comparison in result.comparisons.items()
                },
                classification=result.classification,
                rationale=result.rationale,
                out_of_fold=(
                    {
                        metric: _round(getattr(result.evaluation.oof_metrics, metric))
                        for metric in METRIC_NAMES
                    }
                    if result.experiment in oof_experiments
                    else None
                ),
            )
        )

    combined = experiments.get("E6")
    return AblationResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        raw_sha256=reference.raw_sha256,
        training_ids_sha256=reference.training_ids_sha256,
        n_training_rows=int(labels.size),
        training_positive_prevalence=_round(float(labels.mean())),
        random_seed=config.seed,
        scikit_learn_version=sklearn.__version__,
        cross_validation={
            "strategy": CV_STRATEGY,
            "n_splits": N_SPLITS,
            "shuffle": SHUFFLE,
            "random_state": config.seed,
            "shared_across_experiments": True,
            "identical_to_baseline": True,
        },
        metric_protocol={
            **reference.metric_protocol.model_dump(),
            "primary_metric_used_here": PRIMARY_METRIC,
            "secondary_metric_used_here": SECONDARY_METRIC,
        },
        decision_threshold=DEFAULT_THRESHOLD,
        baseline_reproduction=reproduction,
        candidates=[
            CandidateDefinition(
                group=group.name,
                hypothesis=group.hypothesis,
                added_numeric=list(group.numeric),
                added_categorical=list(group.categorical),
                stateful=group.stateful,
                notes=group.notes,
            )
            for group in FEATURE_GROUPS.values()
        ],
        experiments=records,
        combined_experiment=list(combined.feature_groups) if combined else None,
        methodological_notes=list(METHODOLOGICAL_NOTES),
    )


def write_results(results: AblationResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote ablation results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> AblationResults:
    """Read and validate a Phase 6 record.

    Raises:
        FileNotFoundError: If it has not been generated yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Ablation results not found at {source}. Generate them with "
            "`uv run python scripts/run_feature_engineering.py`."
        )
    return AblationResults.model_validate_json(source.read_text(encoding="utf-8"))
