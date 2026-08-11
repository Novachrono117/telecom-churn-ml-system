"""Machine-readable record of the baseline experiment.

The report in ``reports/baseline_report.md`` is written for a reader; this is
written for a machine and for a future self who wants to know exactly what
produced a number. It carries the provenance chain — raw digest, training
fingerprint, feature set, seeds, library version — alongside every per-fold
value, so no result in the repository is a claim without a trace.

Two deliberate absences:

* **No timestamp.** It would change on every run without adding traceability
  (the git history already dates the artefact) and it would make the
  determinism check impossible.
* **No holdout metric of any kind.** The holdout is untouched until Phase 9.
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
from churn.modeling.evaluation import CV_STRATEGY, N_SPLITS, SHUFFLE, ModelEvaluation
from churn.modeling.metrics import (
    DEFAULT_THRESHOLD,
    METRIC_NAMES,
    confusion_at_threshold,
    positive_prevalence,
)
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

#: Default location of the versioned experiment record.
RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "baseline_results.json"

#: Bumped when the meaning of a field changes.
#: v2 adds ``metric_protocol``: the metric hierarchy and comparison method that
#: later phases must follow, recorded as structured data rather than prose so
#: Phase 6 can read it instead of re-deciding it.
SCHEMA_VERSION = 2

EXPERIMENT_NAME = "phase5-baselines"

#: Decimal places used when serialising metrics. Floating point is already
#: deterministic for identical inputs; rounding only keeps the file readable and
#: free of noise digits that carry no information.
_PRECISION = 6


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


class CrossValidationConfig(BaseModel):
    """How the folds were built."""

    model_config = ConfigDict(frozen=True)

    strategy: str
    n_splits: int = Field(ge=2)
    shuffle: bool
    random_state: int = Field(ge=0)
    shared_across_models: bool


class FeatureSet(BaseModel):
    """Which columns entered the models."""

    model_config = ConfigDict(frozen=True)

    n_features: int = Field(ge=1)
    numeric: list[str]
    categorical: list[str]
    engineered: list[str]


class MetricProtocol(BaseModel):
    """How later phases must read and compare these metrics."""

    model_config = ConfigDict(frozen=True)

    primary_development_metric: str
    secondary_ranking_metric: str
    diagnostic_metrics: list[str]
    auxiliary_metrics: list[str]
    threshold_optimised: bool
    comparison_method: str
    rationale: str


class FoldMetrics(BaseModel):
    """Metrics of a single validation fold."""

    model_config = ConfigDict(frozen=True)

    fold: int = Field(ge=1)
    n_train: int = Field(ge=1)
    n_validation: int = Field(ge=1)
    metrics: dict[str, float]


class ModelResult(BaseModel):
    """Everything measured for one baseline."""

    model_config = ConfigDict(frozen=True)

    name: str
    estimator: str
    hyperparameters: dict[str, object]
    folds: list[FoldMetrics]
    mean: dict[str, float]
    std: dict[str, float]
    out_of_fold: dict[str, float]
    out_of_fold_confusion: dict[str, int]
    converged: bool
    max_iterations_used: int | None


class BaselineResults(BaseModel):
    """The complete Phase 5 experiment record."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    experiment: str
    raw_sha256: str = Field(min_length=64, max_length=64)
    training_ids_sha256: str = Field(min_length=64, max_length=64)
    n_training_rows: int = Field(ge=1)
    training_positive_prevalence: float = Field(gt=0.0, lt=1.0)
    feature_set: FeatureSet
    random_seed: int = Field(ge=0)
    scikit_learn_version: str
    cross_validation: CrossValidationConfig
    decision_threshold: float
    metric_names: list[str]
    metric_protocol: MetricProtocol
    models: list[ModelResult]
    methodological_notes: list[str]


#: The metric hierarchy later phases follow, and how they must compare against
#: these baselines. Recorded as data, not prose, so Phase 6 reads the protocol
#: instead of re-deciding it.
METRIC_PROTOCOL = MetricProtocol(
    primary_development_metric="average_precision",
    secondary_ranking_metric="roc_auc",
    diagnostic_metrics=["precision", "recall", "f1"],
    auxiliary_metrics=["accuracy"],
    threshold_optimised=False,
    comparison_method=(
        "Paired per-fold differences on the identical StratifiedKFold partition: "
        "delta_metric_fold = metric_candidate_fold - metric_baseline_fold. Judge a "
        "candidate on the consistency and magnitude of those deltas. Do not compare a "
        "candidate's gain against a baseline's between-fold standard deviation: that "
        "figure describes how performance varies across partitions, not the "
        "uncertainty of a difference between two models measured on the same partitions."
    ),
    rationale=(
        "Average Precision leads because the positive class is the minority and the "
        "task is to discriminate churners: AP is computed entirely on positive-class "
        "retrieval and degrades visibly when a model finds fewer churners, whereas "
        "ROC-AUC is dominated by the abundant negatives and moves less. ROC-AUC stays "
        "as the secondary ranking view because it is prevalence-independent and "
        "comparable across samples. No single metric decides anything on its own: a "
        "candidate must also be weighed on trade-offs, added complexity and "
        "interpretability, and it need not improve every metric at once."
    ),
)

#: Statements that must travel with the numbers, not only with the prose report.
METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "Every metric here is cross-validated on the frozen Phase 4 training pool. "
    "No metric was computed on the holdout, which stays untouched until Phase 9.",
    "Preprocessing is refitted inside every fold: the scaler's statistics and the "
    "encoder's categories come from the training fold alone.",
    "All models share the same StratifiedKFold partition, so the comparison is paired.",
    "'average_precision' is sklearn.metrics.average_precision_score, not a trapezoidal "
    "PR-AUC. Interpret it against the training positive prevalence recorded here.",
    "The 0.5 decision rule is an operational reference, not a tuned or business-chosen "
    "threshold. Threshold analysis belongs to Phase 9.",
    "Logistic regression is untuned: C, penalty and class_weight are scikit-learn "
    "defaults. No hyperparameter search was run.",
    "No feature engineering, resampling or class weighting was applied. The models see "
    "only the 19 original features.",
    "'accuracy' is auxiliary. It was not used to rank or promote any model.",
    "The per-fold standard deviations describe variability across partitions. They are "
    "NOT a margin of error for the difference between two models and must not be used "
    "as a threshold a future gain has to clear. See metric_protocol.comparison_method.",
    "No paired comparison was performed in this phase; these are reference values only.",
    "The analyst-exposure limitation recorded in the EDA still applies: exploratory "
    "hypotheses were formed with the full dataset in view.",
)


def build_results(
    evaluations: Mapping[str, ModelEvaluation],
    models: Mapping[str, object],
    target: np.ndarray,
    raw_sha256: str,
    training_ids_sha256: str,
) -> BaselineResults:
    """Assemble the experiment record from the evaluations.

    Args:
        evaluations: Result of :func:`churn.modeling.evaluation.evaluate_models`.
        models: The estimators that produced them, for hyperparameter capture.
        target: Encoded training-pool target the evaluations were scored against.
        raw_sha256: Digest of the raw dataset.
        training_ids_sha256: Fingerprint of the training partition.

    Returns:
        The validated :class:`BaselineResults`.
    """
    config = get_config()
    labels = np.asarray(target)
    results: list[ModelResult] = []

    for name, evaluation in evaluations.items():
        classifier = models[name][-1]
        summary = evaluation.summary()
        matrix = confusion_at_threshold(labels, evaluation.oof_probability, DEFAULT_THRESHOLD)
        results.append(
            ModelResult(
                name=name,
                estimator=type(classifier).__name__,
                hyperparameters={
                    key: _jsonable(value) for key, value in sorted(classifier.get_params().items())
                },
                folds=[
                    FoldMetrics(
                        fold=fold.fold,
                        n_train=fold.n_train,
                        n_validation=fold.n_validation,
                        metrics={
                            metric: _round(getattr(fold.metrics, metric)) for metric in METRIC_NAMES
                        },
                    )
                    for fold in evaluation.folds
                ],
                mean={metric: _round(summary[metric]["mean"]) for metric in METRIC_NAMES},
                std={metric: _round(summary[metric]["std"]) for metric in METRIC_NAMES},
                out_of_fold={
                    metric: _round(getattr(evaluation.oof_metrics, metric))
                    for metric in METRIC_NAMES
                },
                out_of_fold_confusion={
                    "true_negative": int(matrix[0, 0]),
                    "false_positive": int(matrix[0, 1]),
                    "false_negative": int(matrix[1, 0]),
                    "true_positive": int(matrix[1, 1]),
                },
                converged=evaluation.converged,
                max_iterations_used=evaluation.max_iterations_used,
            )
        )

    return BaselineResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        raw_sha256=raw_sha256,
        training_ids_sha256=training_ids_sha256,
        n_training_rows=int(labels.size),
        training_positive_prevalence=_round(positive_prevalence(labels)),
        feature_set=FeatureSet(
            n_features=len(FEATURE_COLUMNS),
            numeric=list(NUMERIC_FEATURES),
            categorical=list(CATEGORICAL_FEATURES),
            engineered=[],
        ),
        random_seed=config.seed,
        scikit_learn_version=sklearn.__version__,
        cross_validation=CrossValidationConfig(
            strategy=CV_STRATEGY,
            n_splits=N_SPLITS,
            shuffle=SHUFFLE,
            random_state=config.seed,
            shared_across_models=True,
        ),
        decision_threshold=DEFAULT_THRESHOLD,
        metric_names=list(METRIC_NAMES),
        metric_protocol=METRIC_PROTOCOL,
        models=results,
        methodological_notes=list(METHODOLOGICAL_NOTES),
    )


def write_results(results: BaselineResults, path: Path | None = None) -> Path:
    """Write the experiment record as indented JSON.

    Args:
        results: Record to persist.
        path: Destination. Defaults to :data:`RESULTS_PATH`.

    Returns:
        The path written.
    """
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote baseline results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> BaselineResults:
    """Read and validate an experiment record.

    Args:
        path: Source. Defaults to :data:`RESULTS_PATH`.

    Returns:
        The validated record.

    Raises:
        FileNotFoundError: If the record has not been generated yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Baseline results not found at {source}. Generate them with "
            "`uv run python scripts/run_baselines.py`."
        )
    return BaselineResults.model_validate_json(source.read_text(encoding="utf-8"))
