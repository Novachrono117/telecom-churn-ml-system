"""Phase 9D: the first and only evaluation of the frozen system on the holdout.

This module measures. It cannot train: there is no ``fit`` call in it, no search,
no calibrator and no threshold selection, and it never chooses anything. The
estimator arrives already fitted from disk, the threshold arrives already chosen
from the Phase 9C freeze, and the only thing produced here is a number.

**Why that separation is the whole point.** Every estimate this project has
published so far was cross-validated on the training pool, which means it was
computed on data that had participated in choosing something. The holdout has
never participated in anything: not in a fit, not in a fold, not in a feature
decision, not in the calibration gate, not in the threshold search. That is what
makes the number below an estimate of generalisation rather than a description of
fitting. It is also single-use — once these results exist, no further decision
can be taken from them without destroying the property that made them worth
having.

**The threshold does not move here.** Every metric that depends on a decision is
computed at the single frozen value, including inside every bootstrap
replication. Re-selecting a threshold per replication would measure a procedure,
not this operating point, and this phase is not evaluating a procedure.

**Uncertainty is estimated, not tested.** The bootstrap in
:func:`bootstrap_intervals` resamples the holdout observations and recomputes the
metrics; the probabilities are re-indexed, never recomputed, because no model is
refitted. What comes out is a percentile interval — a statement about how much
this estimate would move under resampling. It is not a hypothesis test, it is not
a significance claim, and it decides nothing.

Metrics are separated into three groups on purpose:

* **discrimination** — threshold-independent, computed on the probabilities;
* **operating point** — what the frozen decision rule actually does;
* **auxiliary** — accuracy, which a constant predictor already scores ~73% on at
  this prevalence, and the two probability losses, reported as diagnostics that
  were never selection criteria.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churn.modeling.freeze import apply_decision_rule, positive_probability

logger = logging.getLogger(__name__)

#: What this phase is. Recorded in the artefact so no later reader has to infer
#: it from the file name.
EVALUATION_TYPE = "FINAL_HOLDOUT"
HOLDOUT_PURPOSE = "FINAL_EVALUATION"

#: Threshold-independent metrics. These are the primary results: they describe
#: how well the model ranks customers by risk, independently of where it cuts.
DISCRIMINATION_METRICS: tuple[str, ...] = ("average_precision", "roc_auc")

#: Metrics of the frozen decision rule. What the system actually does.
OPERATING_POINT_METRICS: tuple[str, ...] = (
    "f1",
    "precision",
    "recall",
    "specificity",
    "balanced_accuracy",
    "predicted_positive_rate",
)

#: Auxiliary metrics. Reported, never decisive. ``accuracy`` is here because at a
#: ~26.5% positive rate a rule that never fires already reaches ~73.5%, so it
#: cannot distinguish a useful model from a silent one.
AUXILIARY_METRICS: tuple[str, ...] = (
    "accuracy",
    "negative_predictive_value",
    "false_positive_rate",
    "false_negative_rate",
)

#: Probabilistic diagnostics. Phase 9A measured these under cross-validation and
#: decided against a calibration layer; they are reported here to close that
#: thread. They were **not** selection criteria and are not used as any here.
PROBABILITY_DIAGNOSTICS: tuple[str, ...] = ("brier_score", "log_loss")

#: Confusion-matrix cells, in the order they are reported.
CONFUSION_NAMES: tuple[str, ...] = (
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
)

#: Bootstrap configuration.
#:
#: 2000 replications is the smallest round number that makes the 2.5% and 97.5%
#: percentiles rest on ~50 replications each rather than a handful, so the
#: interval bounds are not themselves dominated by resampling noise. Going higher
#: would narrow the Monte-Carlo error on the bounds a little further and change
#: no conclusion; going much lower would make the bounds unstable between runs.
N_BOOTSTRAP = 2000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_METHOD = "bootstrap_percentile"

#: Metrics carried through the bootstrap.
BOOTSTRAP_METRICS: tuple[str, ...] = (
    "average_precision",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "specificity",
)


class HoldoutEvaluationError(RuntimeError):
    """The holdout evaluation cannot proceed as specified."""


@dataclass(frozen=True)
class Confusion:
    """The 2x2 confusion matrix, with its orientation stated by field name."""

    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def total(self) -> int:
        """Every row is in exactly one cell, so this is the sample size."""
        return (
            self.true_negatives + self.false_positives + self.false_negatives + self.true_positives
        )

    @property
    def actual_positives(self) -> int:
        """Rows whose true label is churn."""
        return self.true_positives + self.false_negatives

    @property
    def actual_negatives(self) -> int:
        """Rows whose true label is retained."""
        return self.true_negatives + self.false_positives

    @property
    def predicted_positives(self) -> int:
        """Rows the frozen rule flagged."""
        return self.true_positives + self.false_positives

    def as_dict(self) -> dict[str, int]:
        """Return the four cells in :data:`CONFUSION_NAMES` order."""
        return {name: int(getattr(self, name)) for name in CONFUSION_NAMES}


def confusion_of(y_true: np.ndarray, y_predicted: np.ndarray) -> Confusion:
    """Return the confusion matrix with labels pinned to ``[0, 1]``.

    ``labels=[0, 1]`` is passed explicitly. Without it, a vector containing only
    one class would produce a 1x1 matrix and the unpacking below would silently
    assign the wrong cells — the kind of failure that produces a plausible
    number rather than an error.
    """
    matrix = confusion_matrix(np.asarray(y_true), np.asarray(y_predicted), labels=[0, 1])
    (true_negatives, false_positives), (false_negatives, true_positives) = matrix
    return Confusion(
        true_negatives=int(true_negatives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_positives=int(true_positives),
    )


def _ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, or ``0.0`` when it is undefined."""
    return float(numerator / denominator) if denominator else 0.0


@dataclass(frozen=True)
class HoldoutMetrics:
    """Every metric of one evaluation, grouped by what it is allowed to claim."""

    threshold: float
    confusion: Confusion
    average_precision: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    specificity: float
    balanced_accuracy: float
    predicted_positive_rate: float
    accuracy: float
    negative_predictive_value: float
    false_positive_rate: float
    false_negative_rate: float
    brier_score: float
    log_loss: float
    actual_positive_rate: float

    def value(self, metric: str) -> float:
        """Return one metric by name, from any group."""
        if metric in CONFUSION_NAMES:
            return float(getattr(self.confusion, metric))
        return float(getattr(self, metric))

    def group(self, names: Sequence[str]) -> dict[str, float]:
        """Return a named group of metrics, in the given order."""
        return {name: self.value(name) for name in names}


def compute_holdout_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> HoldoutMetrics:
    """Evaluate the frozen decision rule and the probabilities behind it.

    Every threshold-dependent quantity is derived from the same confusion matrix,
    so the reported metrics cannot disagree with the reported cells.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Probability of the positive class.
        threshold: The frozen decision threshold. Never chosen here.

    Returns:
        The :class:`HoldoutMetrics`.

    Raises:
        HoldoutEvaluationError: If the inputs are misaligned or empty.
    """
    labels = np.asarray(y_true).astype(int).ravel()
    probability = np.asarray(y_probability, dtype=float).ravel()
    if labels.size != probability.size:
        raise HoldoutEvaluationError(
            f"{labels.size} labels against {probability.size} probabilities; an evaluation needs "
            "one probability per label."
        )
    if labels.size == 0:
        raise HoldoutEvaluationError("An evaluation needs at least one row.")

    predicted = apply_decision_rule(probability, threshold)
    cells = confusion_of(labels, predicted)

    recall = _ratio(cells.true_positives, cells.actual_positives)
    precision = _ratio(cells.true_positives, cells.predicted_positives)
    specificity = _ratio(cells.true_negatives, cells.actual_negatives)
    f1_denominator = 2.0 * cells.true_positives + cells.false_positives + cells.false_negatives

    return HoldoutMetrics(
        threshold=float(threshold),
        confusion=cells,
        average_precision=float(average_precision_score(labels, probability)),
        roc_auc=float(roc_auc_score(labels, probability)),
        f1=_ratio(2.0 * cells.true_positives, f1_denominator),
        precision=precision,
        recall=recall,
        specificity=specificity,
        balanced_accuracy=(recall + specificity) / 2.0,
        predicted_positive_rate=_ratio(cells.predicted_positives, cells.total),
        accuracy=_ratio(cells.true_positives + cells.true_negatives, cells.total),
        negative_predictive_value=_ratio(
            cells.true_negatives, cells.true_negatives + cells.false_negatives
        ),
        false_positive_rate=_ratio(cells.false_positives, cells.actual_negatives),
        false_negative_rate=_ratio(cells.false_negatives, cells.actual_positives),
        brier_score=float(brier_score_loss(labels, probability)),
        log_loss=float(log_loss(labels, probability, labels=[0, 1])),
        actual_positive_rate=_ratio(cells.actual_positives, cells.total),
    )


def score_holdout(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return P(Churn = Yes) for every holdout row.

    A thin wrapper on purpose: it routes through the Phase 9C primitive that
    resolves the probability column from ``classes_``, so the column read here is
    provably the column the frozen policy names. ``pipeline.predict`` is never
    used — it would apply scikit-learn's internal 0.5, not the frozen threshold.
    """
    probability = positive_probability(pipeline, features)
    logger.info("Scored %d holdout rows.", probability.size)
    return probability


# --- bootstrap ----------------------------------------------------------------

#: A metric that cannot be computed on a resample. ``nan`` rather than a
#: substituted value: a degenerate replication carries no information, and
#: filling it with ``0.0`` would silently drag an interval downwards.
_UNDEFINED = float("nan")


def _needs_positive(labels: np.ndarray) -> bool:
    return bool(labels.sum() > 0)


def _needs_negative(labels: np.ndarray) -> bool:
    return bool((1 - labels).sum() > 0)


def _bootstrap_metric_functions() -> dict[str, Callable[[np.ndarray, np.ndarray, float], float]]:
    """Return one callable per bootstrapped metric, each guarding its own domain.

    A replication is **degenerate** for a metric when that metric is
    mathematically undefined on the resample — not merely inconvenient:

    * Average Precision, ROC-AUC, recall and F1 need at least one actual
      positive; ROC-AUC needs at least one of each class;
    * precision needs at least one predicted positive;
    * specificity needs at least one actual negative.

    Each returns ``nan`` in that case and the count is reported, so an interval
    can never be computed from replications that could not produce a value.
    """

    def average_precision(labels: np.ndarray, probability: np.ndarray, _: float) -> float:
        if not _needs_positive(labels):
            return _UNDEFINED
        return float(average_precision_score(labels, probability))

    def roc_auc(labels: np.ndarray, probability: np.ndarray, _: float) -> float:
        if not (_needs_positive(labels) and _needs_negative(labels)):
            return _UNDEFINED
        return float(roc_auc_score(labels, probability))

    def f1(labels: np.ndarray, probability: np.ndarray, threshold: float) -> float:
        if not _needs_positive(labels):
            return _UNDEFINED
        cells = confusion_of(labels, apply_decision_rule(probability, threshold))
        denominator = 2.0 * cells.true_positives + cells.false_positives + cells.false_negatives
        return _ratio(2.0 * cells.true_positives, denominator)

    def precision(labels: np.ndarray, probability: np.ndarray, threshold: float) -> float:
        cells = confusion_of(labels, apply_decision_rule(probability, threshold))
        if cells.predicted_positives == 0:
            return _UNDEFINED
        return _ratio(cells.true_positives, cells.predicted_positives)

    def recall(labels: np.ndarray, probability: np.ndarray, threshold: float) -> float:
        if not _needs_positive(labels):
            return _UNDEFINED
        cells = confusion_of(labels, apply_decision_rule(probability, threshold))
        return _ratio(cells.true_positives, cells.actual_positives)

    def specificity(labels: np.ndarray, probability: np.ndarray, threshold: float) -> float:
        if not _needs_negative(labels):
            return _UNDEFINED
        cells = confusion_of(labels, apply_decision_rule(probability, threshold))
        return _ratio(cells.true_negatives, cells.actual_negatives)

    return {
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile bootstrap interval for one metric.

    Deliberately **not** called a significance result. It describes how far this
    estimate would move if the holdout had been a different sample of the same
    size from the same population. It tests nothing.
    """

    metric: str
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: str
    n_bootstrap: int
    n_valid: int
    n_degenerate: int

    @property
    def contains_estimate(self) -> bool:
        """Whether the point estimate lies inside its own interval."""
        return self.ci_lower <= self.estimate <= self.ci_upper


def bootstrap_intervals(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    point: HoldoutMetrics,
    metrics: Sequence[str] = BOOTSTRAP_METRICS,
    n_bootstrap: int = N_BOOTSTRAP,
    confidence_level: float = CONFIDENCE_LEVEL,
    seed: int = 42,
) -> dict[str, BootstrapInterval]:
    """Return percentile bootstrap intervals for the requested metrics.

    The procedure, stated so it can be checked rather than trusted:

    1. draw ``n_bootstrap`` resamples of the holdout **observations**, with
       replacement, each the size of the holdout;
    2. re-index the existing probabilities — **no model is refitted**, so this
       measures the uncertainty of the evaluation, not of the training;
    3. recompute each metric on each resample, at the **same frozen threshold**;
    4. take the empirical 2.5th and 97.5th percentiles of the valid values.

    ``seed`` is fixed and recorded, so the interval is reproducible to the last
    digit. The point estimate is the metric on the real holdout, never the mean
    of the replications: the bootstrap describes the spread around the estimate,
    it does not replace it.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Probability of the positive class.
        threshold: The frozen threshold, held constant in every replication.
        point: Metrics of the real holdout, the source of the point estimates.
        metrics: Which metrics to bootstrap.
        n_bootstrap: Number of replications.
        confidence_level: Nominal coverage, e.g. ``0.95``.
        seed: Random seed, recorded in the artefact.

    Returns:
        ``{metric: BootstrapInterval}``, in ``metrics`` order.

    Raises:
        HoldoutEvaluationError: If a metric is unknown, or a replication budget
            leaves no valid value for some metric.
    """
    labels = np.asarray(y_true).astype(int).ravel()
    probability = np.asarray(y_probability, dtype=float).ravel()
    functions = _bootstrap_metric_functions()

    unknown = [name for name in metrics if name not in functions]
    if unknown:
        raise HoldoutEvaluationError(f"No bootstrap implementation for {unknown}.")

    generator = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metrics}

    for _ in range(int(n_bootstrap)):
        index = generator.integers(0, labels.size, size=labels.size)
        resampled_labels = labels[index]
        resampled_probability = probability[index]
        for name in metrics:
            samples[name].append(
                functions[name](resampled_labels, resampled_probability, threshold)
            )

    lower_percentile = 100.0 * (1.0 - confidence_level) / 2.0
    upper_percentile = 100.0 - lower_percentile

    intervals: dict[str, BootstrapInterval] = {}
    for name in metrics:
        values = np.asarray(samples[name], dtype=float)
        valid = values[~np.isnan(values)]
        if valid.size == 0:
            raise HoldoutEvaluationError(
                f"Every bootstrap replication was degenerate for {name!r}; no interval can be "
                "computed from replications that could not produce a value."
            )
        intervals[name] = BootstrapInterval(
            metric=name,
            estimate=point.value(name),
            ci_lower=float(np.percentile(valid, lower_percentile)),
            ci_upper=float(np.percentile(valid, upper_percentile)),
            confidence_level=float(confidence_level),
            method=BOOTSTRAP_METHOD,
            n_bootstrap=int(n_bootstrap),
            n_valid=int(valid.size),
            n_degenerate=int(values.size - valid.size),
        )

    logger.info(
        "Bootstrap done: %d replications, seed %d, %s intervals.",
        n_bootstrap,
        seed,
        BOOTSTRAP_METHOD,
    )
    return intervals


@dataclass(frozen=True)
class GeneralizationCheck:
    """A descriptive holdout-versus-development difference for one metric.

    Not an error, not a gate, and not a test. It is the observed difference
    between a single-sample holdout estimate and a cross-validated one, reported
    so a reader can see it. No cutoff exists anywhere in this project that turns
    it into a verdict, and none is introduced here.
    """

    metric: str
    holdout: float
    development: float
    development_source: str
    difference: float
    comparable: bool
    caveat: str


def generalization_check(
    metric: str,
    holdout_value: float,
    development_value: float,
    development_source: str,
    comparable: bool,
    caveat: str,
) -> GeneralizationCheck:
    """Build one descriptive holdout-versus-development comparison."""
    return GeneralizationCheck(
        metric=metric,
        holdout=float(holdout_value),
        development=float(development_value),
        development_source=development_source,
        difference=float(holdout_value) - float(development_value),
        comparable=bool(comparable),
        caveat=caveat,
    )
