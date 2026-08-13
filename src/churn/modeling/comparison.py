"""Phase 7 protocol: one comparison of families, then one sensitivity analysis.

Two questions are answered separately, and the separation is structural rather
than editorial.

**A — model family.** M0/M1/M2 differ only in the estimator. Each candidate is
paired against M0 fold by fold::

    delta_AP_fold = AP_family_fold - AP_logistic_fold

**B — manual interaction sensitivity.** S0/S1/S2 add ``contract_tenure`` to the
*same* family, and each is paired against that family's own main-comparison run.
The reference of a sensitivity delta is therefore never M0 — otherwise a
family's advantage and a feature's contribution would arrive summed into one
number, and neither could be read.

The paired-difference machinery is imported from the Phase 6 ablation instead of
being rewritten: the delta of Phase 7 must mean exactly what the delta of Phase 6
meant, including the guard that refuses to subtract metrics computed on
different folds. One definition, not two copies.

Nothing here searches, tunes or thresholds. It runs declared configurations on
frozen folds and records what came out.
"""

from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import PurePath

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC, PairedComparison, compare
from churn.features.groups import CONTRACT_TENURE
from churn.features.pipeline import transformed_width
from churn.modeling.evaluation import ModelEvaluation, evaluate_model
from churn.modeling.families import (
    FAMILY_ORDER,
    MAIN_EXPERIMENTS,
    SENSITIVITY_EXPERIMENTS,
    build_family_pipeline,
)

logger = logging.getLogger(__name__)

#: The one Phase 6 candidate carried into Phase 7, as a *retained candidate*.
#: The four candidates that did not pass the Phase 6 heuristic are not reopened
#: here: re-running them against every new family would turn a controlled
#: comparison into a search over feature-set x family cells.
SENSITIVITY_GROUPS: tuple[str, ...] = (CONTRACT_TENURE,)


@dataclass(frozen=True)
class FamilyRun:
    """One experiment: a family, a feature set, its metrics and its deltas."""

    experiment: str
    family: str
    feature_groups: tuple[str, ...]
    n_transformed_features: int
    evaluation: ModelEvaluation
    reference: str | None
    deltas: dict[str, PairedComparison] = field(default_factory=dict)

    @property
    def is_reference(self) -> bool:
        """True when this run is the one others are compared against."""
        return self.reference is None


def run_family(
    experiment: str,
    family: str,
    feature_groups: Sequence[str],
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
    reference: FamilyRun | None = None,
) -> FamilyRun:
    """Cross-validate one family and, when given a reference, pair against it.

    Args:
        experiment: Identifier such as ``"M1"``.
        family: Registered family name.
        feature_groups: Feature groups added to the 19 original features.
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        splitter: The shared splitter — identical folds for every experiment.
        reference: Run whose per-fold metrics the deltas are taken against.

    Returns:
        The :class:`FamilyRun`.
    """
    pipeline = build_family_pipeline(family, feature_groups)
    evaluation = evaluate_model(experiment, pipeline, features, target, splitter)

    # Structural width of the representation, read from a pipeline fitted on the
    # training pool. That fitted object produces no metric: every number in this
    # phase comes from the fold loop above.
    width_probe = build_family_pipeline(family, feature_groups).fit(features, target)

    deltas: dict[str, PairedComparison] = {}
    if reference is not None:
        for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
            deltas[metric] = compare(evaluation, reference.evaluation, metric)
        primary = deltas[PRIMARY_METRIC]
        logger.info(
            "%s (%s): delta %s %+.4f vs %s in %d/%d folds",
            experiment,
            family,
            PRIMARY_METRIC,
            primary.mean,
            reference.experiment,
            primary.n_positive,
            len(evaluation.folds),
        )

    return FamilyRun(
        experiment=experiment,
        family=family,
        feature_groups=tuple(feature_groups),
        n_transformed_features=transformed_width(width_probe),
        evaluation=evaluation,
        reference=reference.experiment if reference is not None else None,
        deltas=deltas,
    )


def run_main_comparison(
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
    gate: Callable[[FamilyRun], None] | None = None,
) -> OrderedDict[str, FamilyRun]:
    """Run M0, M1 and M2 on the 19 original features, paired against M0.

    Args:
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        splitter: Shared splitter.
        gate: Called with M0 **before** any other family runs. It is where the
            baseline-reproduction check belongs: if the reference cannot be
            reproduced, no delta against it means anything, so nothing else
            should be computed — let the callable raise.

    Returns:
        Ordered mapping of experiment id to run, M0 first.
    """
    runs: OrderedDict[str, FamilyRun] = OrderedDict()
    baseline: FamilyRun | None = None

    for family in FAMILY_ORDER:
        experiment = MAIN_EXPERIMENTS[family]
        run = run_family(experiment, family, (), features, target, splitter, baseline)
        runs[experiment] = run
        if baseline is None:
            baseline = run
            if gate is not None:
                gate(run)

    return runs


def run_sensitivity(
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
    main: OrderedDict[str, FamilyRun],
    gate: Callable[[FamilyRun], None] | None = None,
) -> OrderedDict[str, FamilyRun]:
    """Run S0, S1 and S2 — the same families plus ``contract_tenure``.

    Each run is paired against **its own family's** main-comparison run, so the
    only difference inside a comparison is the feature.

    Args:
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        splitter: Shared splitter — the same folds as the main comparison.
        main: Result of :func:`run_main_comparison`.
        gate: Called with S0 before the other families run. It is where the E3
            reproduction check belongs: S0 is the Phase 6 experiment rebuilt
            through the Phase 7 pipeline, so if it does not match E3 the feature
            being tested is not the one Phase 6 measured.

    Returns:
        Ordered mapping of experiment id to run.
    """
    by_family = {run.family: run for run in main.values()}
    runs: OrderedDict[str, FamilyRun] = OrderedDict()

    for family in FAMILY_ORDER:
        experiment = SENSITIVITY_EXPERIMENTS[family]
        run = run_family(
            experiment,
            family,
            SENSITIVITY_GROUPS,
            features,
            target,
            splitter,
            by_family[family],
        )
        runs[experiment] = run
        if gate is not None and len(runs) == 1:
            gate(run)

    return runs


@dataclass(frozen=True)
class CapturedWarning:
    """A warning raised during the run, tallied for the artefact."""

    category: str
    message: str
    source: str
    count: int


def _source(filename: str) -> str:
    """Return a machine-independent origin: the last two path components."""
    parts = PurePath(filename).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else filename


@contextmanager
def collect_warnings() -> Iterator[list[CapturedWarning]]:
    """Record every warning raised inside the block, tallied by identity.

    Warnings are recorded rather than silenced. Nothing is filtered by category:
    the point of writing them into the artefact is that a future reader learns
    which warnings the run actually raised, including ones nobody anticipated.

    The yielded list is **empty until the block exits** — the tally can only be
    computed once every warning has been raised. Read it afterwards.

    Yields:
        The list that receives the deduplicated warnings, ordered by first
        occurrence so the artefact stays deterministic.
    """
    collected: list[CapturedWarning] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield collected

    tally: OrderedDict[tuple[str, str, str], int] = OrderedDict()
    for entry in caught:
        key = (entry.category.__name__, str(entry.message), _source(entry.filename))
        tally[key] = tally.get(key, 0) + 1

    collected.extend(
        CapturedWarning(category=category, message=message, source=source, count=count)
        for (category, message, source), count in tally.items()
    )


def ranked_by(runs: Sequence[FamilyRun], metric: str = PRIMARY_METRIC) -> list[FamilyRun]:
    """Return the runs ordered by mean ``metric``, best first.

    An ordering, not a verdict. Whether the top run is *materially* better is a
    question about the size and the consistency of its paired deltas, and that
    question is answered in the report by a reader, not here by a sort.
    """
    return sorted(runs, key=lambda run: run.evaluation.mean(metric), reverse=True)
