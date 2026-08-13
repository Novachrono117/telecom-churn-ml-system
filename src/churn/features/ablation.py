"""Paired ablation protocol for the candidate features.

Each candidate is compared against the baseline **on the same folds**, fold by
fold::

    delta_AP_fold = AP_candidate_fold - AP_baseline_fold

Why paired rather than a comparison of averages: most of the spread in a
per-fold metric comes from the folds themselves — some partitions are simply
harder — and that component cancels when both models are measured on the same
partition. A candidate can be consistently better than the baseline by less than
the baseline's own between-fold standard deviation, which is why that standard
deviation is **not** used here as a bar to clear. It measures dispersion of one
model across partitions, not the uncertainty of a difference.

Five folds cannot support a formal significance test, and none is run. The
classification below is an engineering heuristic about consistency and
direction, deliberately free of any minimum-gain cutoff: magnitude,
interpretability and complexity are reported separately and judged by a reader.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from churn.features.groups import ABLATION_ORDER
from churn.features.pipeline import build_model_pipeline, transformed_width
from churn.modeling.evaluation import ModelEvaluation, evaluate_model

logger = logging.getLogger(__name__)

PROMISING = "PROMISING"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_SUPPORTED = "NOT_SUPPORTED"

#: Primary metric, inherited from the Phase 5 metric protocol.
PRIMARY_METRIC = "average_precision"
SECONDARY_METRIC = "roc_auc"

#: Folds that must improve for a candidate to be called PROMISING. With five
#: folds this means "improves almost everywhere", which is a statement about
#: consistency of direction, not about effect size.
PROMISING_MIN_POSITIVE_FOLDS = 4


@dataclass(frozen=True)
class PairedComparison:
    """Fold-by-fold difference of one metric between a candidate and the baseline."""

    metric: str
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
    def n_positive(self) -> int:
        """Folds in which the candidate improved on the baseline."""
        return int(sum(delta > 0 for delta in self.deltas))

    @property
    def n_negative(self) -> int:
        """Folds in which the candidate lost to the baseline."""
        return int(sum(delta < 0 for delta in self.deltas))


@dataclass(frozen=True)
class ExperimentResult:
    """One ablation experiment: its evaluation and its paired comparisons."""

    experiment: str
    label: str
    feature_groups: tuple[str, ...]
    n_transformed_features: int
    evaluation: ModelEvaluation
    comparisons: dict[str, PairedComparison]
    classification: str | None
    rationale: str


def compare(
    candidate: ModelEvaluation,
    baseline: ModelEvaluation,
    metric: str,
) -> PairedComparison:
    """Return the per-fold deltas of ``metric``, candidate minus baseline.

    Args:
        candidate: Evaluation of the candidate.
        baseline: Evaluation of the baseline, on the identical folds.
        metric: Metric name.

    Returns:
        The :class:`PairedComparison`.

    Raises:
        ValueError: If the two evaluations do not have matching folds, which
            would make the subtraction meaningless.
    """
    candidate_folds = [fold.fold for fold in candidate.folds]
    baseline_folds = [fold.fold for fold in baseline.folds]
    if candidate_folds != baseline_folds:
        raise ValueError(
            f"Fold identities differ between {candidate.name!r} and {baseline.name!r}: "
            f"{candidate_folds} vs {baseline_folds}. A paired delta requires the same folds."
        )
    sizes = [(fold.n_train, fold.n_validation) for fold in candidate.folds]
    baseline_sizes = [(fold.n_train, fold.n_validation) for fold in baseline.folds]
    if sizes != baseline_sizes:
        raise ValueError(
            f"Fold sizes differ between {candidate.name!r} and {baseline.name!r}. "
            "The comparison would not be paired."
        )

    deltas = tuple(
        float(getattr(c.metrics, metric) - getattr(b.metrics, metric))
        for c, b in zip(candidate.folds, baseline.folds, strict=True)
    )
    return PairedComparison(metric=metric, deltas=deltas)


def classify(primary: PairedComparison, secondary: PairedComparison) -> tuple[str, str]:
    """Classify a candidate from its paired comparisons.

    The rule, in order:

    * **NOT_SUPPORTED** — the mean primary delta is not positive, or the
      candidate loses in a majority of folds. The hypothesis did not show up.
    * **INCONCLUSIVE** — the direction is not consistent enough to act on, or
      the primary metric improves while the secondary ranking metric degrades.
      The latter is a trade-off, and a trade-off is a judgement call, not a
      result.
    * **PROMISING** — the mean primary delta is positive, it improves in at
      least :data:`PROMISING_MIN_POSITIVE_FOLDS` of the folds, and the secondary
      metric does not move against it.

    No minimum-gain threshold appears anywhere: the rule is about direction and
    consistency. Whether a consistent gain is *worth* its complexity is argued
    in the report, not decided here.

    Args:
        primary: Paired comparison on the primary metric.
        secondary: Paired comparison on the secondary ranking metric.

    Returns:
        The classification and a one-sentence rationale.
    """
    n_folds = len(primary.deltas)
    majority_worse = primary.n_negative > n_folds / 2

    if primary.mean <= 0 or majority_worse:
        return NOT_SUPPORTED, (
            f"mean delta {primary.metric} = {primary.mean:+.4f} with "
            f"{primary.n_positive}/{n_folds} folds improving: the hypothesis does not show "
            "up under the frozen protocol."
        )

    if primary.n_positive < PROMISING_MIN_POSITIVE_FOLDS:
        return INCONCLUSIVE, (
            f"mean delta {primary.metric} = {primary.mean:+.4f} but only "
            f"{primary.n_positive}/{n_folds} folds improve: the direction is not consistent "
            "enough to act on with five folds and no significance test."
        )

    if secondary.mean < 0:
        return INCONCLUSIVE, (
            f"{primary.metric} improves in {primary.n_positive}/{n_folds} folds "
            f"({primary.mean:+.4f}) while {secondary.metric} moves against it "
            f"({secondary.mean:+.4f}): a trade-off that judgement has to resolve, not a "
            "clean gain."
        )

    return PROMISING, (
        f"mean delta {primary.metric} = {primary.mean:+.4f}, improving in "
        f"{primary.n_positive}/{n_folds} folds, with {secondary.metric} "
        f"{secondary.mean:+.4f}: consistent in direction under the frozen protocol."
    )


def run_experiment(
    experiment: str,
    label: str,
    feature_groups: Sequence[str],
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
    baseline: ModelEvaluation | None = None,
) -> ExperimentResult:
    """Cross-validate one experiment and, when given a baseline, compare to it.

    Args:
        experiment: Identifier such as ``"E1"``.
        label: Human-readable name.
        feature_groups: Groups added on top of the original features.
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        splitter: The shared splitter — identical folds for every experiment.
        baseline: Evaluation of E0. ``None`` for E0 itself.

    Returns:
        The :class:`ExperimentResult`.
    """
    pipeline = build_model_pipeline(feature_groups)
    evaluation = evaluate_model(experiment, pipeline, features, target, splitter)

    # Output width is a structural property of the feature set, so it is read
    # from a separate pipeline fitted on the training pool. That fitted object is
    # used for nothing else: no metric here comes from a model fitted on all the
    # rows it is scored on.
    width_probe = build_model_pipeline(feature_groups).fit(features, target)

    comparisons: dict[str, PairedComparison] = {}
    classification: str | None = None
    rationale = "Baseline reproduction; nothing to compare against."

    if baseline is not None:
        for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
            comparisons[metric] = compare(evaluation, baseline, metric)
        classification, rationale = classify(
            comparisons[PRIMARY_METRIC], comparisons[SECONDARY_METRIC]
        )
        logger.info(
            "%s (%s): delta %s %+.4f in %d/%d folds -> %s",
            experiment,
            label,
            PRIMARY_METRIC,
            comparisons[PRIMARY_METRIC].mean,
            comparisons[PRIMARY_METRIC].n_positive,
            len(evaluation.folds),
            classification,
        )

    return ExperimentResult(
        experiment=experiment,
        label=label,
        feature_groups=tuple(feature_groups),
        n_transformed_features=transformed_width(width_probe),
        evaluation=evaluation,
        comparisons=comparisons,
        classification=classification,
        rationale=rationale,
    )


def promising_groups(results: Sequence[ExperimentResult]) -> tuple[str, ...]:
    """Return the feature groups of every PROMISING single-group experiment."""
    return tuple(
        group
        for result in results
        if result.classification == PROMISING
        for group in result.feature_groups
    )


def run_ablation(
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
) -> OrderedDict[str, ExperimentResult]:
    """Run E0 through E5, then E6 if more than one candidate is PROMISING.

    Args:
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        splitter: Shared splitter.

    Returns:
        Ordered mapping of experiment id to result.
    """
    results: OrderedDict[str, ExperimentResult] = OrderedDict()
    results["E0"] = run_experiment(
        "E0", "Original features (baseline reproduction)", (), features, target, splitter
    )
    baseline = results["E0"].evaluation

    for index, group_name in enumerate(ABLATION_ORDER, start=1):
        experiment = f"E{index}"
        results[experiment] = run_experiment(
            experiment,
            f"Original + {group_name}",
            (group_name,),
            features,
            target,
            splitter,
            baseline,
        )

    winners = promising_groups(list(results.values()))
    if len(winners) > 1:
        results["E6"] = run_experiment(
            "E6",
            "Original + combined promising features",
            winners,
            features,
            target,
            splitter,
            baseline,
        )
    else:
        logger.info(
            "E6 skipped: %d candidate(s) classified PROMISING; a combination needs at least 2.",
            len(winners),
        )
    return results
