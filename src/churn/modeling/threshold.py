"""Phase 9B: which decision threshold gets frozen before the holdout is opened.

One question, and deliberately only one. **Where do we cut the probability the
frozen logistic regression emits, so that a ranking becomes a decision?** Phase
9A settled what that probability *is* — calibration policy ``NONE``, the raw
``predict_proba`` of the frozen candidate — and this phase settles nothing about
the model: a threshold is applied *after* the fit, so moving it cannot reopen
model selection the way ``class_weight`` would.

**The metric was pre-registered.** ``POLICY_F1`` selects the threshold maximising
the F1 score of the positive class, because precision and recall both matter here
and F1 weighs them symmetrically without requiring a cost ratio nobody in this
project is entitled to invent. This dataset carries no monetary cost of a lost
customer and no cost of a retention campaign, so the operating point chosen here
is a **research operating point, not a business-optimal threshold**. That
sentence is not a hedge; it is the honest scope of what F1-max can claim.

**Choosing a threshold on pooled out-of-fold scores and then reporting the F1 it
achieves on those same rows measures nothing.** The threshold was fitted to them.
So the *procedure* is evaluated nested::

    for each outer fold (5, the frozen partition):
        inside the outer TRAIN only:
            inner 4-fold -> an out-of-sample probability for every outer-train row
            POLICY_F1 on those inner-OOF probabilities -> this fold's threshold
            refit the frozen pipeline on the WHOLE outer train
        predict the outer VALIDATION once, cut it at that threshold

No row of an outer validation fold takes part in a fit, in the probabilities the
threshold search reads, or in the F1 the search maximises. What that estimates is
**the threshold-selection procedure**, not one number; a per-fold threshold that
moves is not a defect of the estimate, it is part of what is being measured.

**D0 and D1 differ by the cut and by nothing else.** Both arms read the *same*
outer probability vector, so Average Precision and ROC-AUC — which are invariant
to any threshold — must come out identical. :func:`verify_ranking_invariance`
turns that into a gate rather than a claim.

Out of scope by protocol: the estimator, its hyperparameters, ``class_weight``,
the preprocessing and the 19 original features are frozen inputs; the calibration
policy stays ``NONE``; no cost number is invented; the recall scenarios and the
top-k capacity analysis are descriptive and cannot select anything; and the
holdout is not touched.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import get_config
from churn.modeling.metrics import DEFAULT_THRESHOLD
from churn.modeling.tuning import build_modern_logistic_pipeline

logger = logging.getLogger(__name__)

#: Experiment identifiers.
DEFAULT_EXPERIMENT = "D0"
NESTED_EXPERIMENT = "D1"

#: Model keys, for figure colours and labels.
MODEL_KEYS: dict[str, str] = {
    DEFAULT_EXPERIMENT: "threshold_default",
    NESTED_EXPERIMENT: "threshold_nested_f1",
}

#: Threshold policies. ``DEFAULT_0_5`` is the unoptimised baseline decision rule;
#: ``F1_MAXIMIZATION`` is the pre-registered candidate procedure.
DEFAULT_POLICY = "DEFAULT_0_5"
F1_POLICY = "F1_MAXIMIZATION"

#: The pre-registered selection metric. Fixed before any Phase 9B number existed,
#: and deliberately singular: a second metric would need a weighting nobody chose.
SELECTION_METRIC = "f1"

#: Threshold-dependent metrics, in reporting order. ``accuracy`` is auxiliary —
#: at a 26.5% positive rate a model that never fires already reaches ~73%, so it
#: cannot rank operating points and never decides anything.
PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
ACCURACY = "accuracy"
PREDICTED_POSITIVE_RATE = "predicted_positive_rate"

DECISION_METRIC_NAMES: tuple[str, ...] = (
    PRECISION,
    RECALL,
    F1,
    ACCURACY,
    PREDICTED_POSITIVE_RATE,
)

#: Ranking invariants. A threshold cannot move either of them, which is exactly
#: why they are checked: a difference between the arms would mean the two were
#: not reading the same probabilities.
AVERAGE_PRECISION = "average_precision"
ROC_AUC = "roc_auc"
GUARDRAIL_METRIC_NAMES: tuple[str, ...] = (AVERAGE_PRECISION, ROC_AUC)

#: Confusion-matrix cells recorded per fold, so a reader can recompute every
#: threshold-dependent metric instead of trusting the ones that were reported.
CONFUSION_NAMES: tuple[str, ...] = (
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
)

#: Numerical tolerance for calling two F1 values tied. Explicit, because "equal"
#: is not a well-defined operation on floats: two thresholds that partition the
#: data identically can produce F1 values differing in the last bit.
TIE_TOLERANCE = 1e-12

#: Inner cross-validation, used **only** inside an outer training fold, to give
#: the threshold search probabilities from a model that did not see the rows it
#: is scoring.
THRESHOLD_INNER_N_SPLITS = 4
THRESHOLD_INNER_SHUFFLE = True

#: Outer folds in which D1 must strictly beat D0 for the procedure to be adopted.
#: The same direction-and-consistency heuristic used since Phase 6 — not a
#: significance test, and carrying no minimum-gain cutoff.
MIN_IMPROVED_FOLDS = 4

ELIGIBLE = "ELIGIBLE_THRESHOLD_POLICY"
NOT_ELIGIBLE = "NOT_ELIGIBLE_THRESHOLD_POLICY"

#: Recall levels for the descriptive scenario table. **Not** requirements: no
#: stakeholder in this project has stated a recall target, and none of these
#: rows can select the frozen threshold.
RECALL_TARGETS: tuple[float, ...] = (0.60, 0.70, 0.80)

#: Contact-capacity fractions for the descriptive top-k table. Hypothetical: no
#: evidence exists that any real retention team has this capacity.
TOP_K_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25)

#: Reporting grid for the operating curve written into the artefact. The *search*
#: never uses it — it reads every distinct probability. This grid exists so the
#: curve is readable in a table of fixed size.
CURVE_GRID_POINTS = 101


class RankingInvarianceError(RuntimeError):
    """Two threshold policies disagreed on a threshold-independent metric."""


@dataclass(frozen=True)
class OperatingPoint:
    """One threshold applied to one probability vector.

    Carries the four confusion cells alongside the derived rates, so every metric
    in the report can be recomputed by a reader rather than taken on trust.
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    predicted_positive_rate: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def as_dict(self) -> dict[str, float]:
        """Return the rate metrics in :data:`DECISION_METRIC_NAMES` order."""
        return {name: float(getattr(self, name)) for name in DECISION_METRIC_NAMES}

    def confusion(self) -> dict[str, int]:
        """Return the confusion cells in :data:`CONFUSION_NAMES` order."""
        return {name: int(getattr(self, name)) for name in CONFUSION_NAMES}


@dataclass(frozen=True)
class DecisionMetrics:
    """An operating point plus the two ranking invariants of its probabilities."""

    operating_point: OperatingPoint
    average_precision: float
    roc_auc: float

    def value(self, metric: str) -> float:
        """Return one metric by name, from either group."""
        if metric in DECISION_METRIC_NAMES or metric in CONFUSION_NAMES:
            return float(getattr(self.operating_point, metric))
        return float(getattr(self, metric))

    def as_dict(self) -> dict[str, float]:
        """Return every recorded metric, rates then confusion then invariants."""
        return {
            **self.operating_point.as_dict(),
            **{name: float(value) for name, value in self.operating_point.confusion().items()},
            AVERAGE_PRECISION: float(self.average_precision),
            ROC_AUC: float(self.roc_auc),
        }


@dataclass(frozen=True)
class ThresholdSweep:
    """Every candidate threshold evaluated at once, as parallel arrays."""

    thresholds: np.ndarray
    true_positives: np.ndarray
    false_positives: np.ndarray
    false_negatives: np.ndarray
    true_negatives: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    accuracy: np.ndarray
    predicted_positive_rate: np.ndarray
    n_rows: int
    n_positives: int

    def __len__(self) -> int:
        """Number of candidate thresholds evaluated."""
        return int(self.thresholds.size)

    def at(self, index: int) -> OperatingPoint:
        """Return the operating point at one position of the sweep."""
        return OperatingPoint(
            threshold=float(self.thresholds[index]),
            precision=float(self.precision[index]),
            recall=float(self.recall[index]),
            f1=float(self.f1[index]),
            accuracy=float(self.accuracy[index]),
            predicted_positive_rate=float(self.predicted_positive_rate[index]),
            true_negatives=int(self.true_negatives[index]),
            false_positives=int(self.false_positives[index]),
            false_negatives=int(self.false_negatives[index]),
            true_positives=int(self.true_positives[index]),
        )


def candidate_thresholds(
    probabilities: np.ndarray,
    default: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Return every threshold at which the classification can change, plus ``default``.

    The decision rule is ``probability >= threshold``. Sweeping the threshold
    upwards, the predicted-positive set changes **exactly** when the threshold
    crosses one of the observed probabilities: any value strictly between two
    observed probabilities produces the same partition as the higher of the two.
    The distinct observed probabilities are therefore the complete set of
    materially different cuts, and a denser grid would only add duplicates.

    ``default`` is added unconditionally so that ``0.5`` is always a member of
    the search space — the procedure must be able to return "the default was
    already the best" rather than being structurally unable to say it.

    **Nothing is rounded.** Rounding the probabilities before the search would
    merge cuts that separate different customers, and the threshold that came
    back would not be a threshold of the score the model actually emits.

    Args:
        probabilities: Predicted probabilities of the positive class.
        default: Threshold always included as a candidate.

    Returns:
        The candidates, sorted ascending and deduplicated.
    """
    values = np.asarray(probabilities, dtype=float).ravel()
    return np.unique(np.concatenate([values, np.array([float(default)], dtype=float)]))


def sweep_thresholds(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> ThresholdSweep:
    """Evaluate the decision rule ``probability >= threshold`` at many thresholds.

    Implemented as one sorted pass rather than a loop over thresholds, but the
    quantities are the ordinary ones: with the rows sorted by probability,
    ``searchsorted`` gives how many fall below each threshold, and a cumulative
    sum of the labels gives how many of those are churners.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Predicted probability of the positive class.
        thresholds: Candidates to evaluate. Defaults to
            :func:`candidate_thresholds` of ``y_probability``.

    Returns:
        The :class:`ThresholdSweep`.

    Raises:
        ValueError: If the two inputs have different lengths, or are empty.
    """
    labels = np.asarray(y_true).astype(int).ravel()
    probability = np.asarray(y_probability, dtype=float).ravel()
    if labels.size != probability.size:
        raise ValueError(
            f"y_true has {labels.size} rows and y_probability has {probability.size}; "
            "a threshold sweep needs one probability per label."
        )
    if labels.size == 0:
        raise ValueError("A threshold sweep needs at least one row.")

    if thresholds is None:
        grid = candidate_thresholds(probability)
    else:
        grid = np.asarray(thresholds, dtype=float).ravel()

    n_rows = int(labels.size)
    n_positives = int(labels.sum())

    order = np.argsort(probability, kind="stable")
    sorted_probability = probability[order]
    cumulative_positives = np.concatenate(([0], np.cumsum(labels[order])))

    below = np.searchsorted(sorted_probability, grid, side="left")
    predicted_positive = (n_rows - below).astype(float)
    true_positives = (n_positives - cumulative_positives[below]).astype(float)
    false_positives = predicted_positive - true_positives
    false_negatives = float(n_positives) - true_positives
    true_negatives = n_rows - predicted_positive - false_negatives

    zeros = np.zeros_like(grid, dtype=float)
    precision = np.divide(
        true_positives, predicted_positive, out=zeros.copy(), where=predicted_positive > 0
    )
    recall = true_positives / float(n_positives) if n_positives else zeros.copy()
    f1_denominator = 2.0 * true_positives + false_positives + false_negatives
    f1 = np.divide(2.0 * true_positives, f1_denominator, out=zeros.copy(), where=f1_denominator > 0)

    return ThresholdSweep(
        thresholds=grid,
        true_positives=true_positives.astype(int),
        false_positives=false_positives.astype(int),
        false_negatives=false_negatives.astype(int),
        true_negatives=true_negatives.astype(int),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=(true_positives + true_negatives) / float(n_rows),
        predicted_positive_rate=predicted_positive / float(n_rows),
        n_rows=n_rows,
        n_positives=n_positives,
    )


def evaluate_at_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> OperatingPoint:
    """Return the operating point of one threshold.

    Routed through :func:`sweep_thresholds` on purpose: the numbers a report
    shows for a single threshold and the numbers the search compares come from
    one implementation, so they cannot drift apart.
    """
    return sweep_thresholds(y_true, y_probability, np.array([float(threshold)], dtype=float)).at(0)


@dataclass(frozen=True)
class ThresholdSelection:
    """The threshold ``POLICY_F1`` returned, and what it was chosen from."""

    policy: str
    operating_point: OperatingPoint
    n_candidates: int
    n_tied_within_tolerance: int
    tie_tolerance: float
    default_in_candidates: bool
    best_f1: float

    @property
    def threshold(self) -> float:
        """The selected threshold."""
        return self.operating_point.threshold


def select_f1_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    default: float = DEFAULT_THRESHOLD,
    tie_tolerance: float = TIE_TOLERANCE,
) -> ThresholdSelection:
    """Apply ``POLICY_F1``: the threshold maximising F1 of the positive class.

    The rule, fixed before any Phase 9B number was read:

    1. candidates are every distinct predicted probability, plus ``default``;
    2. positive when ``probability >= threshold``;
    3. score each candidate by F1 of the positive class, and by nothing else;
    4. take the maximum;
    5. break ties deterministically.

    **Ties are real and must be resolved by a stated rule, not by argmax.** Many
    candidates can reach the same maximal F1 — and ``numpy.argmax`` would silently
    return the lowest-indexed one, making the answer an artefact of the sort
    order. Two thresholds count as tied when their F1 values are within
    ``tie_tolerance`` of the maximum. Among the tied ones the threshold **closest
    to 0.5** wins, and if two are exactly equidistant the **larger** one wins.

    Preferring the candidate closest to 0.5 is a **parsimony rule**: when the
    evidence does not distinguish two operating points, the one that moves less
    from the default is chosen. It is not a claim that 0.5 is right, and it is
    not a business conclusion. The second tie-break, preferring the larger
    threshold, exists only to make the answer unique.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Predicted probability of the positive class.
        default: Threshold always present among the candidates.
        tie_tolerance: Absolute F1 difference below which two candidates tie.

    Returns:
        The :class:`ThresholdSelection`.
    """
    grid = candidate_thresholds(y_probability, default)
    sweep = sweep_thresholds(y_true, y_probability, grid)

    best = float(sweep.f1.max())
    tied = np.flatnonzero(sweep.f1 >= best - tie_tolerance)

    distance = np.abs(sweep.thresholds[tied] - float(default))
    # lexsort takes the LAST key as primary: distance ascending first, then the
    # threshold descending, so "closest to the default, larger on a draw".
    chosen = int(tied[np.lexsort((-sweep.thresholds[tied], distance))[0]])

    return ThresholdSelection(
        policy=F1_POLICY,
        operating_point=sweep.at(chosen),
        n_candidates=len(sweep),
        n_tied_within_tolerance=int(tied.size),
        tie_tolerance=float(tie_tolerance),
        default_in_candidates=bool(np.any(sweep.thresholds == float(default))),
        best_f1=best,
    )


def build_frozen_pipeline() -> Pipeline:
    """Return the Phase 8B candidate, unfitted and uncalibrated.

    Built by the Phase 8B factory rather than restated here, so "the frozen
    estimator" is a fact about the code instead of a claim in a docstring. There
    is no calibration wrapper anywhere in this phase: the Phase 9A policy is
    ``NONE``, so the threshold acts directly on ``predict_proba(X)[:, 1]``.
    """
    return build_modern_logistic_pipeline()


def build_threshold_inner_splitter(seed: int | None = None) -> StratifiedKFold:
    """Return the inner splitter used to select a threshold inside an outer fold."""
    random_state = get_config().seed if seed is None else seed
    return StratifiedKFold(
        n_splits=THRESHOLD_INNER_N_SPLITS,
        shuffle=THRESHOLD_INNER_SHUFFLE,
        random_state=random_state,
    )


def _fit(estimator: BaseEstimator, features: pd.DataFrame, target: np.ndarray) -> tuple:
    """Fit a clone, reporting convergence and re-emitting unrelated warnings.

    ``catch_warnings(record=True)`` intercepts *every* warning raised during the
    fit, not only the convergence one being inspected. Anything else is re-emitted
    so a genuine warning is not silently swallowed by a context manager that
    exists for an unrelated purpose.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = clone(estimator).fit(features, target)

    converged = not any(issubclass(entry.category, ConvergenceWarning) for entry in caught)
    for entry in caught:
        if not issubclass(entry.category, ConvergenceWarning):
            warnings.warn_explicit(entry.message, entry.category, entry.filename, entry.lineno)
    return model, converged


def out_of_fold_probabilities(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: np.ndarray,
    splitter: StratifiedKFold,
) -> tuple[np.ndarray, bool]:
    """Cross-validated probability for every row, from a model that never saw it.

    Args:
        estimator: Unfitted complete pipeline; cloned before every fold.
        features: Feature matrix.
        target: Encoded target aligned with ``features``.
        splitter: Cross-validation splitter.

    Returns:
        ``(probabilities, converged)``.

    Raises:
        RuntimeError: If any row fails to receive a probability.
    """
    y = np.asarray(target)
    probabilities = np.full(len(y), np.nan, dtype=float)
    converged = True

    for train_index, validation_index in splitter.split(features, y):
        model, fold_converged = _fit(estimator, features.iloc[train_index], y[train_index])
        converged = converged and fold_converged
        probabilities[validation_index] = model.predict_proba(features.iloc[validation_index])[:, 1]

    if np.isnan(probabilities).any():
        raise RuntimeError(
            f"{int(np.isnan(probabilities).sum())} row(s) received no out-of-fold probability. "
            "Every row must be in exactly one validation fold."
        )
    return probabilities, converged


def decision_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
) -> DecisionMetrics:
    """Return the operating point at ``threshold`` plus the ranking invariants."""
    labels = np.asarray(y_true)
    probability = np.asarray(y_probability, dtype=float)
    return DecisionMetrics(
        operating_point=evaluate_at_threshold(labels, probability, threshold),
        average_precision=float(average_precision_score(labels, probability)),
        roc_auc=float(roc_auc_score(labels, probability)),
    )


@dataclass(frozen=True)
class InnerThresholdSelection:
    """What the threshold search saw, and returned, inside one outer fold."""

    fold: int
    n_outer_train: int
    n_inner_splits: int
    selection: ThresholdSelection

    @property
    def threshold(self) -> float:
        """The threshold this outer fold carries to its validation rows."""
        return self.selection.threshold


@dataclass(frozen=True)
class ThresholdFold:
    """One outer fold of one threshold policy."""

    fold: int
    n_train: int
    n_validation: int
    threshold: float
    metrics: DecisionMetrics


@dataclass(frozen=True)
class ThresholdRun:
    """One threshold policy evaluated on the frozen outer folds."""

    experiment: str
    label: str
    policy: str
    folds: tuple[ThresholdFold, ...]
    deltas: dict[str, dict[str, PairedThresholdDelta]] = field(default_factory=dict)

    @property
    def model(self) -> str:
        """Figure key for this policy."""
        return MODEL_KEYS[self.experiment]

    def values(self, metric: str) -> np.ndarray:
        """Per-fold values of one metric."""
        return np.array([fold.metrics.value(metric) for fold in self.folds], dtype=float)

    def mean(self, metric: str) -> float:
        """Mean of a metric across the outer folds."""
        return float(self.values(metric).mean())

    def std(self, metric: str) -> float:
        """Sample standard deviation (ddof=1) of a metric across the outer folds."""
        return float(self.values(metric).std(ddof=1))


@dataclass(frozen=True)
class NestedThresholdEvaluation:
    """Both arms, the per-fold selections, and the shared outer probabilities."""

    default_run: ThresholdRun
    selected_run: ThresholdRun
    selections: tuple[InnerThresholdSelection, ...]
    oof_probability: np.ndarray
    converged: bool


def run_threshold_nested_cv(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    outer: StratifiedKFold,
    inner: StratifiedKFold,
    default: float = DEFAULT_THRESHOLD,
) -> NestedThresholdEvaluation:
    """Evaluate the threshold-selection procedure against the default rule.

    Both arms come out of this single loop, which is what makes "D0 and D1 share
    the outer partition and the outer probabilities" a structural property of the
    code rather than an assertion in a report.

    The outer validation fold is referenced exactly once per fold, after the
    threshold for that fold has already been fixed.

    Args:
        estimator: Unfitted frozen pipeline; cloned before every fit.
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        outer: The frozen outer splitter.
        inner: Inner splitter, only ever given an outer training fold.
        default: The unoptimised baseline threshold, D0's decision rule.

    Returns:
        The :class:`NestedThresholdEvaluation`.

    Raises:
        RuntimeError: If any row fails to receive an outer out-of-fold probability.
    """
    y = np.asarray(target)
    oof = np.full(len(y), np.nan, dtype=float)
    selections: list[InnerThresholdSelection] = []
    default_folds: list[ThresholdFold] = []
    selected_folds: list[ThresholdFold] = []
    converged = True

    for number, (train_index, validation_index) in enumerate(outer.split(features, y), start=1):
        # Nothing in this block reads the outer validation fold until the
        # threshold has been fixed. The search sees outer-train rows only.
        train_features = features.iloc[train_index]
        train_target = y[train_index]

        inner_oof, inner_converged = out_of_fold_probabilities(
            estimator, train_features, train_target, inner
        )
        converged = converged and inner_converged
        selection = select_f1_threshold(train_target, inner_oof, default)
        threshold = selection.threshold

        model, outer_converged = _fit(estimator, train_features, train_target)
        converged = converged and outer_converged

        # First and only use of the outer validation fold.
        probability = model.predict_proba(features.iloc[validation_index])[:, 1]
        oof[validation_index] = probability
        validation_target = y[validation_index]

        shape = {
            "fold": number,
            "n_train": int(len(train_index)),
            "n_validation": int(len(validation_index)),
        }
        default_folds.append(
            ThresholdFold(
                **shape,
                threshold=float(default),
                metrics=decision_metrics(validation_target, probability, default),
            )
        )
        selected_folds.append(
            ThresholdFold(
                **shape,
                threshold=threshold,
                metrics=decision_metrics(validation_target, probability, threshold),
            )
        )
        selections.append(
            InnerThresholdSelection(
                fold=number,
                n_outer_train=int(len(train_index)),
                n_inner_splits=int(inner.get_n_splits()),
                selection=selection,
            )
        )
        logger.info(
            "outer fold %d: inner-OOF threshold %.6f from %d candidates, inner F1 %.4f",
            number,
            threshold,
            selection.n_candidates,
            selection.operating_point.f1,
        )

    if np.isnan(oof).any():
        raise RuntimeError(
            f"{int(np.isnan(oof).sum())} row(s) received no outer out-of-fold probability. "
            "Every row must be in exactly one outer validation fold."
        )

    return NestedThresholdEvaluation(
        default_run=ThresholdRun(
            experiment=DEFAULT_EXPERIMENT,
            label=f"Default threshold {default}",
            policy=DEFAULT_POLICY,
            folds=tuple(default_folds),
        ),
        selected_run=ThresholdRun(
            experiment=NESTED_EXPERIMENT,
            label="Nested F1-max threshold",
            policy=F1_POLICY,
            folds=tuple(selected_folds),
        ),
        selections=tuple(selections),
        oof_probability=oof,
        converged=converged,
    )


def verify_ranking_invariance(evaluation: NestedThresholdEvaluation) -> float:
    """Check that both arms report identical Average Precision and ROC-AUC.

    Neither metric depends on a threshold, and both arms read the *same* outer
    probability vector, so the values must be identical — not close, identical.
    A difference would mean the two arms were not evaluated on the same
    probabilities, and every paired delta in this phase would then be measuring
    something other than the cut.

    Returns:
        The largest absolute difference found, which must be ``0.0``.

    Raises:
        RankingInvarianceError: If any fold disagrees.
    """
    largest = 0.0
    offenders: list[str] = []
    for default_fold, selected_fold in zip(
        evaluation.default_run.folds, evaluation.selected_run.folds, strict=True
    ):
        for metric in GUARDRAIL_METRIC_NAMES:
            difference = abs(
                default_fold.metrics.value(metric) - selected_fold.metrics.value(metric)
            )
            largest = max(largest, difference)
            if difference != 0.0:
                offenders.append(f"fold {default_fold.fold} {metric}: {difference:.3e}")

    if offenders:
        raise RankingInvarianceError(
            "D0 and D1 report different threshold-independent metrics. They must read the same "
            "outer probabilities, so this means the arms diverged somewhere other than the "
            "decision rule.\n  " + "\n  ".join(offenders)
        )
    logger.info("Ranking invariance holds: D0 and D1 share AP and ROC-AUC exactly.")
    return largest


@dataclass(frozen=True)
class PairedThresholdDelta:
    """Per-fold difference of one metric between two threshold policies.

    Counts are named by **direction**, not by merit. For F1, precision and recall
    a higher value is better and the report says so; for the predicted positive
    rate "higher" is neither good nor bad — it is how many customers the rule
    would contact — and a field called ``folds_improved`` would smuggle in a
    judgement this phase has no basis for.
    """

    metric: str
    reference: str
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
    def n_higher(self) -> int:
        """Folds in which the candidate scored strictly higher."""
        return int(sum(delta > 0 for delta in self.deltas))

    @property
    def n_lower(self) -> int:
        """Folds in which the candidate scored strictly lower."""
        return int(sum(delta < 0 for delta in self.deltas))

    @property
    def n_tied(self) -> int:
        """Folds in which the two scored identically."""
        return int(sum(delta == 0 for delta in self.deltas))


def compare(
    run: ThresholdRun,
    reference: ThresholdRun,
    metrics: Sequence[str] = DECISION_METRIC_NAMES,
) -> dict[str, PairedThresholdDelta]:
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
        metric: PairedThresholdDelta(
            metric=metric,
            reference=reference.experiment,
            deltas=tuple(
                float(a.metrics.value(metric) - b.metrics.value(metric))
                for a, b in zip(run.folds, reference.folds, strict=True)
            ),
        )
        for metric in metrics
    }


def pair(run: ThresholdRun, reference: ThresholdRun) -> None:
    """Attach paired per-outer-fold deltas of ``run`` against ``reference``."""
    run.deltas[reference.experiment] = compare(run, reference)


def eligibility(delta: PairedThresholdDelta, n_folds: int) -> tuple[str, str]:
    """Decide whether the threshold-selection procedure may replace 0.5.

    The rule, fixed before any Phase 9B number existed: a **positive mean paired
    delta F1** and a **strict** F1 improvement in at least
    :data:`MIN_IMPROVED_FOLDS` of the outer folds. No minimum-gain cutoff exists
    — magnitude is reported separately and weighed by a reader — and this is an
    engineering heuristic, not a significance test.

    A fold in which both arms scored identically is a **tie**, never an
    improvement. That is not pedantry: a tie is exactly what happens when the
    search returns the default, and counting it as a win would let a procedure
    that changed nothing look like one that helped.

    Args:
        delta: Paired per-fold F1 deltas of D1 against D0.
        n_folds: Number of outer folds.

    Returns:
        ``(status, rationale)``.
    """
    summary = (
        f"mean paired delta F1 = {delta.mean:+.5f} with {delta.n_higher} of {n_folds} outer "
        f"folds improving strictly ({delta.n_tied} tied, {delta.n_lower} worse)"
    )
    if delta.mean > 0 and delta.n_higher >= MIN_IMPROVED_FOLDS:
        return ELIGIBLE, (
            f"{summary}: the selection procedure beats the default consistently, on folds whose "
            "rows took no part in choosing their own threshold."
        )
    return NOT_ELIGIBLE, (
        f"{summary}: it does not clear the pre-registered eligibility rule, so the unoptimised "
        "default stands."
    )


def select_threshold_policy(status: str, default: float = DEFAULT_THRESHOLD) -> tuple[str, str]:
    """Return the threshold policy implied by the eligibility status.

    Deliberately trivial and deliberately separate. Keeping the selection out of
    :func:`eligibility` means the "what happens when it fails" branch is a piece
    of code with its own test, rather than an ``else`` nobody exercised.

    Returns:
        ``(policy, rationale)``.
    """
    if status == ELIGIBLE:
        return F1_POLICY, (
            "The nested comparison shows the F1-maximisation procedure beating the default "
            "consistently on rows that took no part in selecting their own threshold, so the "
            "procedure is applied once more to the whole training pool to fix the threshold that "
            "goes to the holdout."
        )
    return DEFAULT_POLICY, (
        f"The F1-maximisation procedure did not clear the pre-registered eligibility rule, so the "
        f"unoptimised default of {default} is frozen. A tuning step has to justify replacing the "
        "default; the fallback when it does not is the simpler rule, not the better-looking one."
    )


@dataclass(frozen=True)
class ThresholdStability:
    """How much the selected threshold moved across the outer folds."""

    thresholds: tuple[float, ...]
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    spread: float
    n_distinct: int


def threshold_stability(selections: Sequence[InnerThresholdSelection]) -> ThresholdStability:
    """Describe the fold-to-fold dispersion of the selected thresholds.

    Description, not a criterion. A wide spread is a limitation to document, and
    the eligibility rule was fixed before these numbers existed: turning
    stability into a post-hoc gate would be choosing the rule after seeing the
    result.
    """
    values = np.array([item.threshold for item in selections], dtype=float)
    return ThresholdStability(
        thresholds=tuple(float(value) for value in values),
        mean=float(values.mean()),
        median=float(np.median(values)),
        std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        minimum=float(values.min()),
        maximum=float(values.max()),
        spread=float(values.max() - values.min()),
        n_distinct=int(np.unique(values).size),
    )


@dataclass(frozen=True)
class RecallScenario:
    """A hypothetical operating point reaching at least a given recall."""

    target_recall: float
    attainable: bool
    operating_point: OperatingPoint | None


def recall_scenarios(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    targets: Sequence[float] = RECALL_TARGETS,
) -> tuple[RecallScenario, ...]:
    """Describe the operating points that reach given recall levels.

    **These are not requirements.** Nobody in this project has stated a recall
    target, and none of these rows can select the frozen threshold: they exist so
    a reader can see what the trade-off looks like elsewhere on the curve.

    Recall is non-increasing in the threshold, so the **largest** threshold
    reaching the target is also the most precise one that does. That is the rule
    applied, and it is deterministic.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Predicted probability of the positive class.
        targets: Recall levels to describe.

    Returns:
        One :class:`RecallScenario` per target, in input order.
    """
    sweep = sweep_thresholds(y_true, y_probability)
    scenarios: list[RecallScenario] = []
    for target in targets:
        reaching = np.flatnonzero(sweep.recall >= float(target))
        if reaching.size == 0:
            scenarios.append(RecallScenario(float(target), False, None))
            continue
        # Thresholds are sorted ascending, so the last index is the largest one.
        scenarios.append(RecallScenario(float(target), True, sweep.at(int(reaching[-1]))))
    return tuple(scenarios)


@dataclass(frozen=True)
class TopKRecord:
    """One hypothetical contact-capacity level, read off the ranking."""

    fraction: float
    n_contacted: int
    churners_captured: int
    recall: float
    precision: float
    lift: float
    #: ``None`` when the budget rounds down to nobody: an empty contact list has
    #: no cut, and reporting ``0.0`` would state a probability that does not exist.
    probability_at_cut: float | None


def top_k_analysis(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    fractions: Sequence[float] = TOP_K_FRACTIONS,
    prevalence: float | None = None,
) -> tuple[TopKRecord, ...]:
    """Describe what a fixed contact budget would capture, by rank.

    A capacity question, not a threshold question: it asks "if we could contact
    the top K% of the ranking, how many churners would that reach", and it uses
    **only** the descending order of the probabilities. It cannot and does not
    select the frozen threshold — the probability at the cut is reported so the
    two views can be compared, never so one can be substituted for the other.

    ``n_contacted`` is ``floor(fraction * n)``, so a budget is never overspent by
    rounding. Ties at the cut are broken by ascending row position — arbitrary,
    but deterministic and stated; with a near-continuous score this affects at
    most a handful of customers.

    ``lift`` is precision divided by the **training-pool prevalence**, the only
    base rate this phase has. It says how much denser in churners the contacted
    group is than an equally sized random draw.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Predicted probability of the positive class.
        fractions: Capacity levels, as shares of the pool.
        prevalence: Base rate for the lift. Defaults to the mean of ``y_true``.

    Returns:
        One :class:`TopKRecord` per fraction, in input order.

    Raises:
        ValueError: If a fraction is outside ``(0, 1]``.
    """
    labels = np.asarray(y_true).astype(int).ravel()
    probability = np.asarray(y_probability, dtype=float).ravel()
    n_rows = int(labels.size)
    n_positives = int(labels.sum())
    base_rate = float(labels.mean()) if prevalence is None else float(prevalence)

    order = np.argsort(-probability, kind="stable")
    ranked_labels = labels[order]
    ranked_probability = probability[order]
    captured = np.concatenate(([0], np.cumsum(ranked_labels)))

    records: list[TopKRecord] = []
    for fraction in fractions:
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError(f"A capacity fraction must be in (0, 1]; got {fraction!r}.")
        n_contacted = int(np.floor(float(fraction) * n_rows))
        if n_contacted == 0:
            records.append(TopKRecord(float(fraction), 0, 0, 0.0, 0.0, 0.0, None))
            continue
        hits = int(captured[n_contacted])
        precision = hits / n_contacted
        records.append(
            TopKRecord(
                fraction=float(fraction),
                n_contacted=n_contacted,
                churners_captured=hits,
                recall=hits / n_positives if n_positives else 0.0,
                precision=precision,
                lift=precision / base_rate if base_rate else 0.0,
                probability_at_cut=float(ranked_probability[n_contacted - 1]),
            )
        )
    return tuple(records)


def operating_curve(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    n_points: int = CURVE_GRID_POINTS,
) -> ThresholdSweep:
    """Evaluate the decision rule on a uniform reporting grid over ``[0, 1]``.

    For the report and the artefact only. The **search** reads every distinct
    probability; this grid exists so a curve can be tabulated at a fixed size,
    and no threshold is ever selected from it.
    """
    return sweep_thresholds(y_true, y_probability, np.linspace(0.0, 1.0, int(n_points)))
