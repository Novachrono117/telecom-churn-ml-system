"""Phase 8B: nested cross-validation of two tuning procedures.

Three things happen here, in this order, and the order is the protocol.

**A compatibility migration.** The frozen logistic regression passes
``penalty="l2"``, deprecated in scikit-learn 1.8 and scheduled for removal in
1.10. The modern spelling is ``l1_ratio=0.0``. The legacy builder stays exactly
as it is — it is what reproduces every historical artefact — and a modern
builder is introduced beside it, gated by a proof that the two produce the same
per-fold metrics *and* the same out-of-fold probabilities. This is a
**compatibility migration, not a model change**.

**Nested cross-validation.** A hyperparameter search that reports its own best
inner score is reporting a number chosen by looking at the data it is scored on.
Nested CV avoids that: the search runs entirely inside each outer training fold,
and the outer validation fold is touched only once, by the estimator the search
already committed to::

    for each outer fold (5, the frozen partition):
        GridSearchCV(inner 4-fold, scoring=average_precision) on the outer TRAIN
        refit the winner on the outer TRAIN
        predict the outer VALIDATION exactly once

What that estimates is **the tuning procedure**, not one configuration. A
per-fold ``best_params_`` that varies is not a defect of the estimate; it is
part of what is being measured.

**A pre-registered selection rule.** :func:`eligibility` and
:func:`select_candidate` are written before any result is read, and they are
tested on synthetic inputs so that the branch taken by the real numbers is not
the only branch anybody ever exercised.

Out of scope by protocol: the threshold stays at the diagnostic default,
``class_weight`` stays ``None`` everywhere, no engineered feature is used, no
probability is calibrated, no random forest is included, and the native
categorical representation closed in Phase 8A is not reopened.
"""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline

from churn.config import get_config
from churn.features.ablation import PairedComparison, compare
from churn.features.pipeline import build_feature_preprocessor
from churn.modeling.evaluation import FoldResult, ModelEvaluation
from churn.modeling.families import HIST_GRADIENT_BOOSTING_PARAMS, build_hist_gradient_boosting
from churn.modeling.metrics import DEFAULT_THRESHOLD, compute_metrics
from churn.modeling.models import (
    CLASSIFIER_STEP,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    PREPROCESSOR_STEP,
)

logger = logging.getLogger(__name__)

#: Experiment identifiers.
FROZEN_LOGISTIC = "T0"
TUNED_LOGISTIC = "T1"
TUNED_HGB = "T2"

#: Model keys, for figure colours and labels.
T0_MODEL = "logistic_frozen"
T1_MODEL = "logistic_tuned"
T2_MODEL = "hgb_tuned"

MODEL_KEYS: dict[str, str] = {
    FROZEN_LOGISTIC: T0_MODEL,
    TUNED_LOGISTIC: T1_MODEL,
    TUNED_HGB: T2_MODEL,
}

#: Inner cross-validation, used **only** inside an outer training fold.
INNER_N_SPLITS = 4
INNER_SHUFFLE = True

#: The search selects on this and nothing else. A multi-metric objective would
#: make the selection depend on a weighting nobody chose deliberately.
SEARCH_SCORING = "average_precision"

#: Regularisation grid for the tuned logistic regression. Log-spaced and
#: **containing 1.0**, so the frozen baseline configuration is a member of the
#: search space rather than a point outside it.
LOGISTIC_C_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

T1_SEARCH_SPACE: dict[str, list[object]] = {
    f"{CLASSIFIER_STEP}__C": list(LOGISTIC_C_GRID),
}

#: Boosting grid. Small and declared up front: 2 x 2 x 2 x 3 x 2 = 48 points.
#: The Phase 7 configuration (0.1 / 100 / 31 / 20 / 0.0) is one of them, so the
#: search can return "the untuned configuration was already the best" instead of
#: being structurally unable to say it.
T2_SEARCH_SPACE: dict[str, list[object]] = {
    f"{CLASSIFIER_STEP}__learning_rate": [0.05, 0.10],
    f"{CLASSIFIER_STEP}__max_iter": [100, 200],
    f"{CLASSIFIER_STEP}__max_leaf_nodes": [15, 31],
    f"{CLASSIFIER_STEP}__min_samples_leaf": [10, 20, 40],
    f"{CLASSIFIER_STEP}__l2_regularization": [0.0, 1.0],
}

#: Hyperparameters frozen for every boosting candidate. Named here so the
#: artefact can state them, and asserted in the tests against what is built.
T2_FIXED_PARAMS: tuple[str, ...] = (
    "loss",
    "max_depth",
    "max_features",
    "early_stopping",
    "class_weight",
    "random_state",
)

#: Outer folds in which a procedure must improve on the baseline to be eligible.
#: As everywhere in this project, a statement about consistency of direction —
#: not about effect size, and not a significance test.
ELIGIBILITY_MIN_POSITIVE_FOLDS = 4

ELIGIBLE = "ELIGIBLE_TO_REPLACE_BASELINE"
NOT_ELIGIBLE = "NOT_ELIGIBLE_TO_REPLACE_BASELINE"


def build_modern_logistic() -> LogisticRegression:
    """Return the frozen logistic configuration in the modern spelling.

    ``l1_ratio=0.0`` is the documented replacement for ``penalty="l2"``. The
    argument is **not** passed: the point of the migration is to stop using the
    deprecated one, and passing it here would keep emitting the warning this
    change exists to remove.

    Every other value — ``C``, ``solver``, ``class_weight``, ``max_iter`` — is
    the frozen Phase 5 configuration, taken from the same constants the legacy
    builder uses.
    """
    return LogisticRegression(
        C=1.0,
        l1_ratio=0.0,
        class_weight=None,
        solver=LOGISTIC_SOLVER,
        max_iter=LOGISTIC_MAX_ITER,
    )


def build_modern_logistic_pipeline() -> Pipeline:
    """Return the complete modern logistic estimator, unfitted."""
    return Pipeline(
        steps=[
            (PREPROCESSOR_STEP, build_feature_preprocessor(())),
            (CLASSIFIER_STEP, build_modern_logistic()),
        ]
    )


def build_hgb_pipeline() -> Pipeline:
    """Return the Phase 7 boosting estimator on the common representation.

    The classifier comes from :func:`churn.modeling.families.build_hist_gradient_boosting`
    — the frozen Phase 7 factory — so every parameter the grid does not vary is
    the Phase 7 value by construction rather than by restatement.
    """
    return Pipeline(
        steps=[
            (PREPROCESSOR_STEP, build_feature_preprocessor(())),
            (CLASSIFIER_STEP, build_hist_gradient_boosting()),
        ]
    )


def n_candidates(search_space: Mapping[str, Sequence[object]]) -> int:
    """Return how many configurations a search space contains."""
    return len(ParameterGrid(dict(search_space)))


def build_inner_splitter(seed: int | None = None) -> StratifiedKFold:
    """Return the inner splitter. It only ever sees an outer training fold."""
    random_state = get_config().seed if seed is None else seed
    return StratifiedKFold(
        n_splits=INNER_N_SPLITS, shuffle=INNER_SHUFFLE, random_state=random_state
    )


def build_search(
    estimator: Pipeline,
    search_space: Mapping[str, Sequence[object]],
    inner: StratifiedKFold,
) -> GridSearchCV:
    """Wrap an **unfitted pipeline** in a grid search over the inner folds.

    The estimator handed over is the complete pipeline, preprocessing included.
    That is what makes the search leakage-safe: every inner fold refits the
    scaler and the encoder on its own training part, so no inner validation row
    contributes to the transformation applied to it.

    ``refit=True`` means the winning configuration is refitted on the whole
    outer training fold before it is returned — which is exactly the estimator
    the outer validation fold must be predicted with.
    """
    return GridSearchCV(
        estimator=clone(estimator),
        param_grid=dict(search_space),
        scoring=SEARCH_SCORING,
        cv=inner,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )


@dataclass(frozen=True)
class OuterFold:
    """One outer fold: its metrics and, when tuned, what the search chose."""

    fold: int
    n_train: int
    n_validation: int
    metrics: object
    best_params: dict[str, object] | None = None
    best_inner_score: float | None = None


@dataclass(frozen=True)
class NestedRun:
    """One procedure evaluated on the outer folds."""

    experiment: str
    label: str
    model: str
    tuned: bool
    folds: tuple[OuterFold, ...]
    oof_probability: np.ndarray
    oof_metrics: object
    converged: bool
    paired_deltas: dict[str, dict[str, PairedComparison]] = field(default_factory=dict)

    def as_evaluation(self) -> ModelEvaluation:
        """Adapt to the shared :class:`ModelEvaluation` so paired code is reused."""
        return ModelEvaluation(
            name=self.experiment,
            folds=tuple(
                FoldResult(
                    fold=fold.fold,
                    n_train=fold.n_train,
                    n_validation=fold.n_validation,
                    metrics=fold.metrics,
                )
                for fold in self.folds
            ),
            oof_probability=self.oof_probability,
            oof_metrics=self.oof_metrics,
            converged=self.converged,
            max_iterations_used=None,
        )

    def mean(self, metric: str) -> float:
        """Mean of a metric across the outer folds."""
        return float(np.mean([getattr(fold.metrics, metric) for fold in self.folds]))


def run_nested_cv(
    experiment: str,
    label: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    outer: StratifiedKFold,
    search_space: Mapping[str, Sequence[object]] | None = None,
    inner: StratifiedKFold | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> NestedRun:
    """Evaluate one procedure on the outer folds, tuning inside each of them.

    With ``search_space=None`` the pipeline is simply fitted on each outer
    training fold — that is T0, the frozen baseline, and it goes through this
    same loop so that T0, T1 and T2 provably share the outer partition.

    Args:
        experiment: Identifier such as ``"T1"``.
        label: Human-readable name.
        pipeline: Unfitted complete pipeline; cloned before every outer fold.
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        outer: The frozen outer splitter.
        search_space: Grid to search inside each outer training fold. ``None``
            runs no search.
        inner: Inner splitter. Required when ``search_space`` is given.
        threshold: Diagnostic decision rule. Never optimised.

    Returns:
        The :class:`NestedRun`.

    Raises:
        ValueError: If a search space is given without an inner splitter.
        RuntimeError: If any row fails to receive an outer out-of-fold prediction.
    """
    if search_space is not None and inner is None:
        raise ValueError("An inner splitter is required whenever a search space is given.")

    y = np.asarray(target)
    oof = np.full(len(y), np.nan, dtype=float)
    folds: list[OuterFold] = []
    converged = True

    for number, (train_index, validation_index) in enumerate(outer.split(features, y), start=1):
        # The outer validation fold is not referenced anywhere in this block.
        train_features = features.iloc[train_index]
        train_target = y[train_index]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            if search_space is None:
                model: BaseEstimator = clone(pipeline).fit(train_features, train_target)
                best_params: dict[str, object] | None = None
                best_inner: float | None = None
            else:
                search = build_search(pipeline, search_space, inner).fit(
                    train_features, train_target
                )
                model = search.best_estimator_
                best_params = dict(search.best_params_)
                best_inner = float(search.best_score_)

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
            OuterFold(
                fold=number,
                n_train=int(len(train_index)),
                n_validation=int(len(validation_index)),
                metrics=compute_metrics(y[validation_index], probability, threshold),
                best_params=best_params,
                best_inner_score=best_inner,
            )
        )
        logger.info(
            "%s: outer fold %d done%s",
            experiment,
            number,
            f", best inner AP {best_inner:.4f}, params {best_params}" if best_inner else "",
        )

    if np.isnan(oof).any():
        raise RuntimeError(
            f"{int(np.isnan(oof).sum())} row(s) received no outer out-of-fold prediction for "
            f"{experiment!r}. Every row must be in exactly one outer validation fold."
        )

    return NestedRun(
        experiment=experiment,
        label=label,
        model=MODEL_KEYS[experiment],
        tuned=search_space is not None,
        folds=tuple(folds),
        oof_probability=oof,
        oof_metrics=compute_metrics(y, oof, threshold),
        converged=converged,
    )


def pair(run: NestedRun, reference: NestedRun, metrics: Sequence[str]) -> None:
    """Attach paired per-outer-fold deltas of ``run`` against ``reference``."""
    run.paired_deltas[reference.experiment] = {
        metric: compare(run.as_evaluation(), reference.as_evaluation(), metric)
        for metric in metrics
    }


def eligibility(comparison: PairedComparison, n_folds: int) -> tuple[str, str]:
    """Decide whether a tuned procedure may replace the frozen baseline.

    The rule, fixed before any Phase 8B number existed: a positive mean paired
    delta on the primary metric **and** improvement in at least
    :data:`ELIGIBILITY_MIN_POSITIVE_FOLDS` of the outer folds. There is no
    minimum-gain cutoff — magnitude is reported separately and weighed by a
    reader — and this is an engineering heuristic, not a significance test.

    "Improvement" means a **strictly positive** delta. A fold in which the two
    procedures scored identically is counted as a tie, never as an improvement,
    and is reported in its own column: ``n_positive`` and ``n_negative`` do not
    have to sum to ``n_folds``, and reporting only those two would let a tie
    read as a loss.

    Returns:
        ``(status, rationale)``.
    """
    summary = (
        f"mean paired delta AP = {comparison.mean:+.5f} with "
        f"{comparison.n_positive} of {n_folds} outer folds improving "
        f"({comparison.n_zero} tied, {comparison.n_negative} worse)"
    )
    if comparison.mean > 0 and comparison.n_positive >= ELIGIBILITY_MIN_POSITIVE_FOLDS:
        return ELIGIBLE, f"{summary}: positive and consistent against the frozen baseline."
    return NOT_ELIGIBLE, f"{summary}: it does not clear the pre-registered eligibility rule."


def select_candidate(
    statuses: Mapping[str, str],
    head_to_head: PairedComparison | None,
    n_folds: int,
) -> tuple[str, str]:
    """Apply the pre-registered selection rule and return the frozen candidate.

    Written and tested before the real numbers were read. Every branch is
    exercised by the test suite on synthetic inputs, so the branch the real
    result happens to take is not the only one that was ever checked.

    Args:
        statuses: Eligibility status keyed by experiment id, for T1 and T2.
        head_to_head: Paired primary-metric deltas of T2 against T1. Only
            consulted when both are eligible.
        n_folds: Number of outer folds.

    Returns:
        ``(selected_experiment, rationale)``.
    """
    t1 = statuses.get(TUNED_LOGISTIC) == ELIGIBLE
    t2 = statuses.get(TUNED_HGB) == ELIGIBLE

    if not t1 and not t2:
        return (
            FROZEN_LOGISTIC,
            "Neither tuned procedure cleared the eligibility rule, so the frozen baseline "
            "stands. Tuning has to justify replacing it; the default when it does not is to "
            "keep the simpler model, not to adopt the best-scoring one.",
        )
    if t1 and not t2:
        return (
            TUNED_LOGISTIC,
            "Only the tuned logistic regression cleared the eligibility rule.",
        )
    if t2 and not t1:
        return (
            TUNED_HGB,
            "Only the tuned boosting procedure cleared the eligibility rule.",
        )

    if head_to_head is None:
        raise ValueError("Both procedures are eligible: the T2-vs-T1 comparison is required.")

    if head_to_head.mean > 0 and head_to_head.n_positive >= ELIGIBILITY_MIN_POSITIVE_FOLDS:
        return (
            TUNED_HGB,
            f"Both cleared the rule and boosting won the direct comparison: mean delta AP "
            f"{head_to_head.mean:+.5f} in {head_to_head.n_positive}/{n_folds} outer folds.",
        )
    if head_to_head.mean < 0 and head_to_head.n_negative >= ELIGIBILITY_MIN_POSITIVE_FOLDS:
        return (
            TUNED_LOGISTIC,
            f"Both cleared the rule and the logistic regression won the direct comparison: "
            f"mean delta AP {head_to_head.mean:+.5f} in {head_to_head.n_negative}/{n_folds} "
            "outer folds against it.",
        )
    return (
        TUNED_LOGISTIC,
        "Both cleared the rule and neither won the direct comparison consistently, so the "
        "pre-registered tie-break applies: prefer the logistic regression for its lower "
        "complexity and higher interpretability.",
    )


def selection_frequency(run: NestedRun) -> OrderedDict[str, OrderedDict[str, int]]:
    """Count how often each hyperparameter value was chosen across outer folds.

    A description of stability, not evidence. Five folds cannot establish that a
    value is "the" right one; they can show whether the search lands in the same
    region or moves with the partition.
    """
    counts: OrderedDict[str, OrderedDict[str, int]] = OrderedDict()
    for fold in run.folds:
        for parameter, value in (fold.best_params or {}).items():
            name = parameter.split("__", 1)[-1]
            bucket = counts.setdefault(name, OrderedDict())
            key = str(value)
            bucket[key] = bucket.get(key, 0) + 1
    return counts


def run_final_search(
    pipeline: Pipeline,
    search_space: Mapping[str, Sequence[object]],
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
) -> tuple[dict[str, object], float]:
    """Search the **whole training pool** to fix the parameters for Phase 9.

    This produces **no generalisation estimate**. Its cross-validated score is
    computed on the same data the selection used, so it is optimistically biased
    by construction — the honest estimate of the procedure is the nested outer
    result, and this search exists only to answer "which values get frozen".

    Returns:
        ``(best_params, best_cv_score)`` — the score being reported for
        transparency, never as a performance figure.
    """
    search = GridSearchCV(
        estimator=clone(pipeline),
        param_grid=dict(search_space),
        scoring=SEARCH_SCORING,
        cv=splitter,
        refit=False,
        n_jobs=1,
        error_score="raise",
    ).fit(features, np.asarray(target))
    return dict(search.best_params_), float(search.best_score_)


def frozen_hgb_configuration() -> dict[str, object]:
    """Return the Phase 7 boosting values for the parameters this phase searches.

    Used to assert that the untuned configuration is a member of the grid.
    """
    return {
        parameter: HIST_GRADIENT_BOOSTING_PARAMS[parameter.split("__", 1)[-1]]
        for parameter in T2_SEARCH_SPACE
    }
