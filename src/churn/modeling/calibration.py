"""Phase 9A: does the frozen logistic regression need a calibration layer?

One question, and deliberately only one. **Are the probabilities this model emits
usable as probabilities?** Not "is it a good ranker" — Phases 5 to 8B already
answered that, and a model can rank perfectly while being systematically
overconfident. Ranking and calibration are different properties of the same
score, and a monotone transformation changes the second without touching the
first.

That asymmetry is why this phase runs before any threshold is chosen. A
calibrator rewrites the probabilities; a threshold is a cut on them. Choosing the
cut first would attach it to a score that stops existing the moment a calibrator
is adopted — 0.42 on raw scores and 0.42 on calibrated scores are different
operating points, with different recall.

**Cross-fitted, or it measures nothing.** A calibrator fitted on the scores its
own base estimator produced in-sample sees an optimistically distorted score
distribution and learns to correct a distortion that does not exist out of
sample. So no row may ever be used to fit the estimator that scores it, to fit
the calibrator that transforms that score, and to evaluate the result::

    for each outer fold (5, the frozen partition):
        inside the outer TRAIN only:
            inner 4-fold -> a cross-validated score for every outer-train row
            fit the calibrator on those out-of-sample scores
            refit the base pipeline on the WHOLE outer train
        predict the outer VALIDATION fold exactly once

**One estimator, one calibrator** — ``ensemble=False``. The alternative,
``ensemble=True``, keeps one (estimator, calibrator) pair per inner fold and
averages their predictions, which would make C1 and C2 differ from C0 by *two*
things at once: the calibration layer, and an average of four models each fitted
on three quarters of the data. Averaging models is a variance-reduction
technique with its own effect on the Brier score, and this phase exists to
isolate the calibration layer. With ``ensemble=False`` the base estimator is
refitted on the entire outer training fold, exactly as C0 is, so the only
difference between the arms is the transformation applied to the score.

**Lower is better here**, unlike every earlier phase. Brier score and log loss
are losses, so an improvement is a *negative* delta. The paired comparison in
this module carries an explicit orientation flag rather than relying on a reader
remembering which way each metric points.

Note what those two losses are and are not. Both are **proper scoring rules**:
they are minimised by the true probabilities, which is why they can be trusted
to rank probability quality. They are **not** measures of calibration alone —
each decomposes into a calibration term and a refinement/discrimination term, so
a movement in either can come from sharper probabilities rather than
better-aligned ones. The decision therefore reads them together with the
reliability diagram, the ranking guardrails and fold-to-fold consistency.

Out of scope by protocol: no threshold is selected, no threshold-dependent metric
is even computed, the estimator and its hyperparameters are frozen,
``class_weight`` stays ``None``, no feature is added, and no model family is
reopened.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import get_config
from churn.modeling.tuning import build_modern_logistic_pipeline

logger = logging.getLogger(__name__)

#: Experiment identifiers.
UNCALIBRATED = "C0"
SIGMOID_CALIBRATION = "C1"
ISOTONIC_CALIBRATION = "C2"

#: Calibration methods. Exactly two, both declared before any result was read.
SIGMOID = "sigmoid"
ISOTONIC = "isotonic"

#: Model keys, for figure colours and labels.
MODEL_KEYS: dict[str, str] = {
    UNCALIBRATED: "logistic_uncalibrated",
    SIGMOID_CALIBRATION: "logistic_sigmoid",
    ISOTONIC_CALIBRATION: "logistic_isotonic",
}

#: Metric names. Split by what they answer, because this phase decides on the
#: first group and only watches the second.
BRIER = "brier_score"
LOG_LOSS = "log_loss"
ECE = "expected_calibration_error"
AVERAGE_PRECISION = "average_precision"
ROC_AUC = "roc_auc"

#: Probability quality — the question of this phase.
PROBABILITY_METRICS: tuple[str, ...] = (BRIER, LOG_LOSS, ECE)

#: Ranking quality — guardrails only. A calibrator is never adopted because one
#: of these went up; they exist to catch a calibrator that damaged the ranking.
RANKING_METRICS: tuple[str, ...] = (AVERAGE_PRECISION, ROC_AUC)

CALIBRATION_METRIC_NAMES: tuple[str, ...] = PROBABILITY_METRICS + RANKING_METRICS

#: Metrics on which a smaller value is better. All three probability losses.
LOWER_IS_BETTER: frozenset[str] = frozenset(PROBABILITY_METRICS)

#: The two metrics the decision rests on, in priority order.
DECISION_METRICS: tuple[str, ...] = (BRIER, LOG_LOSS)

#: Expected Calibration Error configuration. Declared here, before any diagram
#: was drawn, because bin counts chosen after looking at a reliability plot are
#: chosen to make a curve look convincing.
ECE_N_BINS = 10
ECE_STRATEGY = "quantile"

#: Reliability diagram configuration. The same bins for all three experiments —
#: comparing curves binned differently would compare the binnings.
RELIABILITY_N_BINS = 10
RELIABILITY_STRATEGY = "quantile"

#: Inner cross-validation, used only inside an outer training fold, to fit the
#: calibrator on scores from a model that did not see the rows being scored.
CALIBRATION_INNER_N_SPLITS = 4
CALIBRATION_INNER_SHUFFLE = True

#: ``ensemble`` is pinned rather than left at the ``"auto"`` default: the
#: protocol has to state what ran, and a future change of that default would
#: otherwise silently alter it.
#:
#: ``False`` is the methodologically correct value **for this question**. It uses
#: the inner folds only to produce out-of-sample scores for the calibrator, then
#: refits the base estimator on the whole outer training fold — the same data C0
#: is fitted on. ``True`` would instead average four estimators fitted on three
#: quarters each, adding a variance-reduction effect on top of the calibration
#: layer and leaving the comparison unable to say which of the two moved the
#: number.
CALIBRATION_ENSEMBLE = False

#: Folds in which a method must improve for the improvement to count as
#: consistent. The same direction-and-consistency heuristic used since Phase 6 —
#: not a significance test, and carrying no minimum-gain cutoff.
MIN_IMPROVED_FOLDS = 4

ELIGIBLE = "ELIGIBLE_CALIBRATION"
NOT_ELIGIBLE = "NOT_ELIGIBLE_CALIBRATION"
INCONCLUSIVE = "INCONCLUSIVE_CALIBRATION"

#: The policy carried to Phase 9B when no calibrator earns adoption.
NO_CALIBRATION = "NONE"


@dataclass(frozen=True)
class CalibrationMetrics:
    """Probability quality first, ranking guardrails second.

    Deliberately carries **no** precision, recall, F1 or accuracy. Those need a
    threshold, this phase does not choose one, and a metric that cannot be
    computed cannot accidentally decide anything.
    """

    brier_score: float
    log_loss: float
    expected_calibration_error: float
    average_precision: float
    roc_auc: float

    def as_dict(self) -> dict[str, float]:
        """Return the metrics in :data:`CALIBRATION_METRIC_NAMES` order."""
        return {name: float(getattr(self, name)) for name in CALIBRATION_METRIC_NAMES}


def expected_calibration_error(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    n_bins: int = ECE_N_BINS,
    strategy: str = ECE_STRATEGY,
) -> float:
    """Return the Expected Calibration Error under an explicitly stated definition.

    There is no single ECE: implementations differ in binning, in weighting and
    in what they do with empty bins, and two papers reporting "ECE 0.02" may not
    be reporting the same quantity. This one is::

        ECE = sum over non-empty bins b of  (n_b / N) * | mean(p in b) - mean(y in b) |

    that is, the sample-weighted mean absolute gap between predicted confidence
    and observed frequency.

    Binning:

    * ``"quantile"`` places the edges at the empirical quantiles of the
      predicted probabilities, so every bin holds a comparable number of rows.
      Uniform-width bins on a skewed score distribution leave the high-probability
      bins nearly empty, and a gap measured over eleven customers then dominates
      a term it cannot support.
    * ``"uniform"`` places the edges evenly across the observed probability range.

    Empty bins contribute **nothing** — they carry weight ``0``, so they neither
    add to nor dilute the score. Duplicate quantile edges are collapsed, which
    means the *effective* number of bins can be smaller than ``n_bins`` when the
    score distribution has ties; that is reported honestly rather than padded.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Predicted probability of the positive class.
        n_bins: Requested number of bins.
        strategy: ``"quantile"`` or ``"uniform"``.

    Returns:
        The ECE. ``0.0`` for an empty input.

    Raises:
        ValueError: If ``n_bins`` is below 1 or ``strategy`` is unknown.
    """
    predicted, observed, counts = reliability_bins(y_true, y_probability, n_bins, strategy)
    if counts.size == 0:
        return 0.0
    return float(np.sum(counts / counts.sum() * np.abs(predicted - observed)))


def reliability_bins(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    n_bins: int = RELIABILITY_N_BINS,
    strategy: str = RELIABILITY_STRATEGY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mean_predicted, observed_frequency, count)`` per non-empty bin.

    The same binning :func:`expected_calibration_error` uses, exposed so that the
    reliability diagram and the ECE cannot drift apart: a diagram drawn with one
    binning next to a number computed with another describes two things while
    looking like it describes one.

    Each policy is binned by **its own** quantiles. That is what makes the bins
    equally populated, which is the point of quantile binning; reusing one
    policy's edges for another would pile most of the other's mass into a single
    bin and make its curve unreadable. The *strategy* is identical across
    policies, and that is what the comparison rests on.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Predicted probability of the positive class.
        n_bins: Requested number of bins.
        strategy: ``"quantile"`` or ``"uniform"``.

    Returns:
        Three arrays of equal length, one entry per non-empty bin.

    Raises:
        ValueError: If ``n_bins`` is below 1 or ``strategy`` is unknown.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}.")

    labels = np.asarray(y_true, dtype=float)
    probability = np.asarray(y_probability, dtype=float)
    if labels.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, np.array([], dtype=int)

    if strategy == "quantile":
        edges = np.quantile(probability, np.linspace(0.0, 1.0, n_bins + 1))
    elif strategy == "uniform":
        edges = np.linspace(probability.min(), probability.max(), n_bins + 1)
    else:
        raise ValueError(f"Unknown binning strategy {strategy!r}; use 'quantile' or 'uniform'.")

    edges = np.unique(edges)
    if edges.size < 2:
        return (
            np.array([probability.mean()]),
            np.array([labels.mean()]),
            np.array([labels.size], dtype=int),
        )

    assignment = np.clip(np.searchsorted(edges, probability, side="left") - 1, 0, edges.size - 2)

    predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for index in range(edges.size - 1):
        selected = assignment == index
        count = int(selected.sum())
        if count == 0:
            continue
        predicted.append(float(probability[selected].mean()))
        observed.append(float(labels[selected].mean()))
        counts.append(count)

    return np.array(predicted), np.array(observed), np.array(counts, dtype=int)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> CalibrationMetrics:
    """Evaluate predicted probabilities as probabilities, and as a ranking.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Predicted probability of the positive class.

    Returns:
        The five metrics.
    """
    labels = np.asarray(y_true)
    probability = np.asarray(y_probability, dtype=float)

    return CalibrationMetrics(
        brier_score=float(brier_score_loss(labels, probability)),
        log_loss=float(log_loss(labels, probability, labels=[0, 1])),
        expected_calibration_error=expected_calibration_error(labels, probability),
        average_precision=float(average_precision_score(labels, probability)),
        roc_auc=float(roc_auc_score(labels, probability)),
    )


def build_frozen_pipeline() -> Pipeline:
    """Return the Phase 8B candidate, unfitted.

    Built by the Phase 8B factory rather than restated here, so "the frozen
    estimator" is a fact about the code instead of a claim in a docstring. Any
    drift in that factory breaks this phase's tests rather than silently
    changing what was calibrated.
    """
    return build_modern_logistic_pipeline()


def build_calibration_splitter(seed: int | None = None) -> StratifiedKFold:
    """Return the inner splitter used to fit calibrators inside an outer fold."""
    random_state = get_config().seed if seed is None else seed
    return StratifiedKFold(
        n_splits=CALIBRATION_INNER_N_SPLITS,
        shuffle=CALIBRATION_INNER_SHUFFLE,
        random_state=random_state,
    )


def build_calibrated_estimator(method: str, inner: StratifiedKFold) -> CalibratedClassifierCV:
    """Wrap the **complete** frozen pipeline in a cross-fitted calibrator.

    The whole pipeline goes inside the calibrator, not just the classifier. That
    is what makes it leakage-safe: every inner fold refits the cleaner, the
    scaler and the encoder on its own training part, so no row contributes to
    the transformation applied to it, and the calibrator is fitted on scores
    from a model that never saw the rows it is calibrating.

    With :data:`CALIBRATION_ENSEMBLE` at ``False`` the inner folds produce those
    out-of-sample scores and nothing else: the estimator that finally predicts is
    refitted on the whole outer training fold, so the arm differs from the
    uncalibrated reference by the calibration layer alone.

    Args:
        method: ``"sigmoid"`` or ``"isotonic"``.
        inner: Inner splitter. Only ever given an outer training fold.

    Returns:
        The unfitted calibrated estimator.

    Raises:
        ValueError: If ``method`` is not one of the two declared methods.
    """
    if method not in (SIGMOID, ISOTONIC):
        raise ValueError(
            f"Unknown calibration method {method!r}. This phase evaluates exactly "
            f"{SIGMOID!r} and {ISOTONIC!r}; no other method is in scope."
        )
    return CalibratedClassifierCV(
        estimator=build_frozen_pipeline(),
        method=method,
        cv=inner,
        ensemble=CALIBRATION_ENSEMBLE,
        n_jobs=1,
    )


@dataclass(frozen=True)
class PairedMetricDelta:
    """Per-fold difference of one metric, with the direction stated explicitly.

    Every earlier phase compared metrics where higher is better. Here the two
    metrics that decide are losses. Rather than leaving a reader to remember
    which way each one points, the orientation is carried in the data and
    ``n_improved`` is computed from it.
    """

    metric: str
    reference: str
    lower_is_better: bool
    deltas: tuple[float, ...]

    @property
    def mean(self) -> float:
        """Mean of the per-fold deltas."""
        return float(np.mean(self.deltas))

    @property
    def std(self) -> float:
        """Sample standard deviation (ddof=1) of the per-fold deltas."""
        return float(np.std(self.deltas, ddof=1))

    @property
    def n_improved(self) -> int:
        """Folds in which the candidate beat its reference, whichever way that is."""
        if self.lower_is_better:
            return int(sum(delta < 0 for delta in self.deltas))
        return int(sum(delta > 0 for delta in self.deltas))

    @property
    def n_worsened(self) -> int:
        """Folds in which the candidate lost to its reference."""
        if self.lower_is_better:
            return int(sum(delta > 0 for delta in self.deltas))
        return int(sum(delta < 0 for delta in self.deltas))

    @property
    def n_tied(self) -> int:
        """Folds in which the two scored identically."""
        return int(sum(delta == 0 for delta in self.deltas))


@dataclass(frozen=True)
class CalibrationFold:
    """One outer fold of one calibration policy."""

    fold: int
    n_train: int
    n_validation: int
    metrics: CalibrationMetrics


@dataclass(frozen=True)
class CalibrationRun:
    """One calibration policy evaluated on the frozen outer folds."""

    experiment: str
    label: str
    method: str | None
    calibrated: bool
    folds: tuple[CalibrationFold, ...]
    oof_probability: np.ndarray
    oof_metrics: CalibrationMetrics
    converged: bool
    deltas: dict[str, dict[str, PairedMetricDelta]] = field(default_factory=dict)

    @property
    def model(self) -> str:
        """Figure key for this policy."""
        return MODEL_KEYS[self.experiment]

    def values(self, metric: str) -> np.ndarray:
        """Per-fold values of one metric."""
        return np.array([getattr(fold.metrics, metric) for fold in self.folds], dtype=float)

    def mean(self, metric: str) -> float:
        """Mean of a metric across the outer folds."""
        return float(self.values(metric).mean())

    def std(self, metric: str) -> float:
        """Sample standard deviation (ddof=1) of a metric across the outer folds."""
        return float(self.values(metric).std(ddof=1))

    @property
    def n_distinct_oof_probabilities(self) -> int:
        """How many distinct probabilities the policy emits out of fold.

        Informative for isotonic regression, which is a step function: a sharp
        drop here means the calibrator has collapsed a continuous score into a
        handful of plateaus, which is visible in a reliability diagram as flat
        segments and matters for any later threshold that has to land somewhere.
        """
        return int(np.unique(self.oof_probability).size)


def run_calibration_cv(
    experiment: str,
    label: str,
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    outer: StratifiedKFold,
    method: str | None = None,
) -> CalibrationRun:
    """Evaluate one calibration policy on the outer folds.

    The uncalibrated reference goes through this same loop, so "C0, C1 and C2
    share the outer partition" is a structural property of the code rather than
    an assertion in a report.

    Args:
        experiment: Identifier such as ``"C1"``.
        label: Human-readable name.
        estimator: Unfitted estimator; cloned before every outer fold.
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        outer: The frozen outer splitter.
        method: Calibration method, or ``None`` for the uncalibrated reference.

    Returns:
        The :class:`CalibrationRun`.

    Raises:
        RuntimeError: If any row fails to receive an outer out-of-fold probability.
    """
    y = np.asarray(target)
    oof = np.full(len(y), np.nan, dtype=float)
    folds: list[CalibrationFold] = []
    converged = True

    for number, (train_index, validation_index) in enumerate(outer.split(features, y), start=1):
        # The outer validation fold is not referenced anywhere in this block.
        # Everything the calibrator learns, it learns from the outer train.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = clone(estimator).fit(features.iloc[train_index], y[train_index])

        if any(issubclass(entry.category, ConvergenceWarning) for entry in caught):
            converged = False
            logger.warning("%s: outer fold %d did not converge", experiment, number)
        for entry in caught:
            if not issubclass(entry.category, ConvergenceWarning):
                warnings.warn_explicit(entry.message, entry.category, entry.filename, entry.lineno)

        # First and only use of the outer validation fold.
        probability = model.predict_proba(features.iloc[validation_index])[:, 1]
        oof[validation_index] = probability

        folds.append(
            CalibrationFold(
                fold=number,
                n_train=int(len(train_index)),
                n_validation=int(len(validation_index)),
                metrics=compute_calibration_metrics(y[validation_index], probability),
            )
        )
        logger.info("%s: outer fold %d done", experiment, number)

    if np.isnan(oof).any():
        raise RuntimeError(
            f"{int(np.isnan(oof).sum())} row(s) received no outer out-of-fold probability for "
            f"{experiment!r}. Every row must be in exactly one outer validation fold."
        )

    return CalibrationRun(
        experiment=experiment,
        label=label,
        method=method,
        calibrated=method is not None,
        folds=tuple(folds),
        oof_probability=oof,
        oof_metrics=compute_calibration_metrics(y, oof),
        converged=converged,
    )


def compare(
    run: CalibrationRun,
    reference: CalibrationRun,
    metrics: Sequence[str] = CALIBRATION_METRIC_NAMES,
) -> dict[str, PairedMetricDelta]:
    """Return per-outer-fold deltas of ``run`` against ``reference``.

    Raises:
        ValueError: If the two runs were not evaluated on the same folds.
    """
    if [fold.fold for fold in run.folds] != [fold.fold for fold in reference.folds]:
        raise ValueError(
            f"{run.experiment} and {reference.experiment} were not evaluated on the same outer "
            "folds; a paired delta between them would not be paired."
        )
    return {
        metric: PairedMetricDelta(
            metric=metric,
            reference=reference.experiment,
            lower_is_better=metric in LOWER_IS_BETTER,
            deltas=tuple(
                float(getattr(a.metrics, metric) - getattr(b.metrics, metric))
                for a, b in zip(run.folds, reference.folds, strict=True)
            ),
        )
        for metric in metrics
    }


def pair(run: CalibrationRun, reference: CalibrationRun) -> None:
    """Attach paired per-outer-fold deltas of ``run`` against ``reference``."""
    run.deltas[reference.experiment] = compare(run, reference)


def eligibility(
    deltas: dict[str, PairedMetricDelta],
    n_folds: int,
) -> tuple[str, str]:
    """Decide whether a calibration method has earned adoption.

    The rule, fixed before any Phase 9A number existed:

    1. mean paired delta Brier ``< 0``;
    2. Brier improves in at least :data:`MIN_IMPROVED_FOLDS` of the outer folds;
    3. mean paired delta log loss ``<= 0``;
    4. no relevant and consistent deterioration of Average Precision.

    Condition 4 is operationalised the way every consistency claim in this
    project is — direction plus agreement across folds, never a magnitude
    cutoff: AP counts as deteriorated when its mean delta is negative **and** it
    worsens in at least :data:`MIN_IMPROVED_FOLDS` folds. A single bad fold is
    noise; four of five moving the same way is not.

    Conditions 1-2 passing while 3 fails is the case the protocol names
    explicitly: Brier and log loss disagree materially, and the honest answer is
    :data:`INCONCLUSIVE` rather than adopting on the metric that happened to
    agree. The two losses penalise different failures — log loss is unbounded
    and punishes confident mistakes far harder than Brier does — so a
    disagreement usually means the calibrator helped the bulk of the
    distribution and hurt its tail.

    Returns:
        ``(status, rationale)``.
    """
    brier, log_loss_delta, ap = deltas[BRIER], deltas[LOG_LOSS], deltas[AVERAGE_PRECISION]

    summary = (
        f"mean delta Brier {brier.mean:+.5f} improving in {brier.n_improved}/{n_folds} folds; "
        f"mean delta log loss {log_loss_delta.mean:+.5f} improving in "
        f"{log_loss_delta.n_improved}/{n_folds}; mean delta AP {ap.mean:+.5f} worsening in "
        f"{ap.n_worsened}/{n_folds}"
    )

    if not (brier.mean < 0 and brier.n_improved >= MIN_IMPROVED_FOLDS):
        return NOT_ELIGIBLE, (
            f"{summary}. The Brier score does not improve consistently, which is the primary "
            "condition, so calibration is not adopted."
        )
    if log_loss_delta.mean > 0:
        return INCONCLUSIVE, (
            f"{summary}. Brier improves but log loss does not, and the protocol treats a "
            "material disagreement between the two probability losses as inconclusive rather "
            "than adopting on the one that agrees."
        )
    if ap.mean < 0 and ap.n_worsened >= MIN_IMPROVED_FOLDS:
        return NOT_ELIGIBLE, (
            f"{summary}. The ranking guardrail tripped: Average Precision deteriorates "
            "consistently, so the probability gain was bought with ranking quality."
        )
    return ELIGIBLE, (f"{summary}. Both probability losses improve and the ranking guardrail held.")


def select_policy(
    statuses: dict[str, str],
    head_to_head: dict[str, PairedMetricDelta] | None,
    stability: dict[str, float] | None,
    n_folds: int,
) -> tuple[str, str]:
    """Apply the pre-registered selection rule and return the calibration policy.

    Written and tested before the real numbers were read, and every branch is
    exercised on synthetic inputs so the branch the real result takes is not the
    only one anybody checked.

    Priority when both methods are eligible: Brier, then log loss, then
    stability and complexity. The first two can produce a winner on evidence —
    a consistent advantage across folds. The last two cannot be turned into a
    threshold without inventing a cutoff this project does not allow, so they
    are expressed as the **default**: when neither loss produces a consistent
    winner, sigmoid is selected for having one parameter pair against isotonic's
    free monotone step function.

    Args:
        statuses: Eligibility status keyed by experiment, for C1 and C2.
        head_to_head: Deltas of C2 against C1. Required when both are eligible.
        stability: Fold-to-fold standard deviation of the Brier score per
            experiment, reported in the rationale. Never decisive on its own.
        n_folds: Number of outer folds.

    Returns:
        ``(policy, rationale)`` where policy is ``"NONE"``, ``"C1"`` or ``"C2"``.

    Raises:
        ValueError: If both are eligible and the head-to-head is missing.
    """
    sigmoid_ok = statuses.get(SIGMOID_CALIBRATION) == ELIGIBLE
    isotonic_ok = statuses.get(ISOTONIC_CALIBRATION) == ELIGIBLE

    if not sigmoid_ok and not isotonic_ok:
        return NO_CALIBRATION, (
            "No calibration method cleared the eligibility rule, so no calibration layer is "
            "adopted and the frozen logistic regression keeps emitting its own probabilities. "
            "Adding a transformation has to be justified; the default when it is not is to keep "
            "the simpler object."
        )
    if sigmoid_ok and not isotonic_ok:
        return SIGMOID_CALIBRATION, "Only sigmoid calibration cleared the eligibility rule."
    if isotonic_ok and not sigmoid_ok:
        return ISOTONIC_CALIBRATION, "Only isotonic calibration cleared the eligibility rule."

    if head_to_head is None:
        raise ValueError("Both methods are eligible: the C2-vs-C1 comparison is required.")

    spread = ""
    if stability:
        spread = " Fold-to-fold Brier SD: " + ", ".join(
            f"{key} {value:.5f}" for key, value in sorted(stability.items())
        )

    for metric in DECISION_METRICS:
        delta = head_to_head[metric]
        if delta.mean < 0 and delta.n_improved >= MIN_IMPROVED_FOLDS:
            return ISOTONIC_CALIBRATION, (
                f"Both cleared the rule and isotonic won the direct comparison on {metric}: "
                f"mean delta {delta.mean:+.5f} in {delta.n_improved}/{n_folds} outer folds.{spread}"
            )
        if delta.mean > 0 and delta.n_worsened >= MIN_IMPROVED_FOLDS:
            return SIGMOID_CALIBRATION, (
                f"Both cleared the rule and sigmoid won the direct comparison on {metric}: "
                f"mean delta {delta.mean:+.5f} against it in {delta.n_worsened}/{n_folds} outer "
                f"folds.{spread}"
            )

    return SIGMOID_CALIBRATION, (
        "Both cleared the rule and neither probability loss produced a consistent winner, so the "
        "pre-registered default applies: sigmoid, for its lower flexibility. It fits two "
        "parameters, while isotonic fits a free monotone step function and has more room to "
        f"track the calibration split it was given.{spread}"
    )
