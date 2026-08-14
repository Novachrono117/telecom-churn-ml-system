"""Phase 8A: how much of the HGB-versus-logistic gap is the representation?

Phase 7 held the representation constant so that the estimator was the only
variable, and recorded plainly that a common representation is not necessarily
optimal for every family. This module answers the question that observation
leaves open, and **only** that question::

    R0  logistic regression      + common representation   (Phase 7 M0)
    R1  HistGradientBoosting     + common representation   (Phase 7 M2)
    R2  HistGradientBoosting     + native categorical representation

R0 and R1 are rebuilt through the Phase 7 builder rather than re-declared, so
they are the same objects that produced `model_comparison_results.json` and can
be checked against it fold by fold. R2 differs from R1 in **one** respect: the
16 contracted categorical columns reach the estimator as ordinal codes declared
categorical, instead of as 43 one-hot indicators. Every hyperparameter is the
frozen Phase 7 configuration.

This is a gate, not tuning. Nothing here searches a parameter, moves a
threshold, adds an engineered feature or touches the holdout.

**Why the ordinal codes are not an ordinal assumption.** ``OrdinalEncoder``
assigns integer codes in sorted order, and nothing about those integers is
meaningful. What keeps them nominal is the second half of the arrangement: the
same 16 positions are declared to the estimator through ``categorical_features``,
so the split finder partitions *sets of categories* rather than thresholding a
number line. The encoder alone would be an ordinal assumption; the encoder plus
the declaration is not. That is why the mask is explicit and never inferred from
a dtype — a representation whose nominality depends on an upstream dtype is not
a guarantee of anything.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from churn.config import get_config
from churn.features.ablation import PRIMARY_METRIC, PairedComparison, compare
from churn.modeling.evaluation import ModelEvaluation, evaluate_model
from churn.modeling.families import (
    HIST_GRADIENT_BOOSTING,
    HIST_GRADIENT_BOOSTING_PARAMS,
    LOGISTIC_REGRESSION,
    build_family_pipeline,
)
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn.preprocessing.pipeline import (
    CATEGORICAL_STEP,
    CLEANER_STEP,
    ENCODER_STEP,
    NUMERIC_STEP,
)
from churn.preprocessing.transformers import TotalChargesCleaner

logger = logging.getLogger(__name__)

#: Experiment identifiers of this gate.
REFERENCE_LOGISTIC = "R0"
REFERENCE_HGB_COMMON = "R1"
NATIVE_HGB = "R2"

#: Model key of the new experiment, for figures and labels. R0 and R1 keep the
#: family names they carry in Phase 7 so a model keeps its colour across reports.
HGB_NATIVE = "hgb_native"

#: Name of the cardinality check step inside the R2 preprocessor.
GUARD_STEP = "cardinality_guard"

#: Names of the two representations compared here.
COMMON_REPRESENTATION = "common_ohe"
NATIVE_REPRESENTATION = "native_categorical"

#: Maximum categories ``HistGradientBoostingClassifier`` can bin into a
#: categorical split. Read from the estimator's own ``max_bins`` default rather
#: than written down, so it cannot drift away from the library that enforces it.
MAX_CATEGORICAL_CARDINALITY: int = int(HistGradientBoostingClassifier().max_bins)

#: Ordinal encoder configuration. An unseen category becomes ``NaN`` and is
#: routed to the estimator's own missing-value handling for categorical splits.
#: This is the serving behaviour a *new but valid* category needs. It is not a
#: blank-handling mechanism: blanks, ``None`` and empty strings are rejected
#: earlier by the Phase 4 feature contract, and a genuinely new category must
#: not be confused with invalid input.
ORDINAL_ENCODER_PARAMS: dict[str, object] = {
    "handle_unknown": "use_encoded_value",
    "unknown_value": np.nan,
    "dtype": np.float64,
}

#: Which transformed columns are categorical. Positional and explicit: the
#: ``ColumnTransformer`` emits the numeric block first, then the categorical one,
#: so the mask is 3 False followed by 16 True. Never ``"from_dtype"``.
CATEGORICAL_MASK: tuple[bool, ...] = (False,) * len(NUMERIC_FEATURES) + (True,) * len(
    CATEGORICAL_FEATURES
)

#: Folds that must improve for the representation to be called helpful. As in
#: Phase 6, a statement about consistency of direction — not about effect size,
#: and not a significance test.
HELPFUL_MIN_POSITIVE_FOLDS = 4

REPRESENTATION_HELPFUL = "REPRESENTATION_HELPFUL"
REPRESENTATION_INCONCLUSIVE = "REPRESENTATION_INCONCLUSIVE"
REPRESENTATION_NOT_SUPPORTED = "REPRESENTATION_NOT_SUPPORTED"

BELOW_LOGISTIC = "BELOW_LOGISTIC"
COMPARABLE_TO_LOGISTIC = "COMPARABLE_TO_LOGISTIC"
ABOVE_LOGISTIC = "ABOVE_LOGISTIC"


class CategoricalCardinalityError(ValueError):
    """A training fold carries more categories than the estimator can bin."""


class CategoricalCardinalityGuard(BaseEstimator, TransformerMixin):
    """Fail loudly when a categorical column exceeds the estimator's limit.

    The check runs in ``fit``, so it only ever sees a training fold: the
    cardinality of a validation fold is not measured, not compared and not used.
    ``transform`` returns its input unchanged — this step exists to raise, not to
    modify.

    Categories are **never** grouped, hashed or truncated to fit. Silently
    collapsing rare levels would change the representation being measured while
    the experiment kept claiming it had changed only the encoding.
    """

    def __init__(
        self,
        columns: Sequence[str] = CATEGORICAL_FEATURES,
        max_cardinality: int = MAX_CATEGORICAL_CARDINALITY,
    ) -> None:
        self.columns = columns
        self.max_cardinality = max_cardinality

    def fit(self, X: pd.DataFrame, y: object = None) -> CategoricalCardinalityGuard:
        """Record the cardinality of each categorical column in this fold.

        Raises:
            CategoricalCardinalityError: If any column exceeds the limit.
        """
        self.cardinality_ = {
            column: int(X[column].nunique(dropna=False)) for column in self.columns
        }
        offenders = {
            column: count
            for column, count in self.cardinality_.items()
            if count > self.max_cardinality
        }
        if offenders:
            raise CategoricalCardinalityError(
                f"Categorical column(s) exceed the {self.max_cardinality}-category limit of "
                f"HistGradientBoostingClassifier: {offenders}. No column is grouped, hashed or "
                "truncated to fit: that would silently change the representation under test."
            )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` unchanged."""
        return X

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Return the input feature names unchanged."""
        return np.asarray(list(input_features) if input_features is not None else [], dtype=object)


def observed_cardinality(features: pd.DataFrame) -> dict[str, int]:
    """Return the categories per categorical column, counted on the given frame.

    Called by the orchestration script with the **training pool** so the record
    can state what the representation actually had to encode. It also enforces
    the limit, through the same guard the pipeline uses inside every fold.

    Raises:
        CategoricalCardinalityError: If a column exceeds the estimator's limit.
    """
    guard = CategoricalCardinalityGuard().fit(features)
    return dict(guard.cardinality_)


def categorical_indices() -> tuple[int, ...]:
    """Return the positional indices of the categorical transformed columns."""
    return tuple(index for index, flag in enumerate(CATEGORICAL_MASK) if flag)


def build_native_column_encoder() -> ColumnTransformer:
    """Build the native-categorical encoding block.

    The numeric branch is the Phase 4 ``StandardScaler``, unchanged. Keeping it
    means the numeric handling is identical to R1, so the categorical branch is
    the only difference between the two experiments — which is the whole point
    of the gate. Standardisation does not affect where a tree places a split, so
    keeping it costs nothing and removes a second variable.
    """
    return ColumnTransformer(
        transformers=[
            (NUMERIC_STEP, StandardScaler(), list(NUMERIC_FEATURES)),
            (
                CATEGORICAL_STEP,
                OrdinalEncoder(**ORDINAL_ENCODER_PARAMS),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_native_preprocessor() -> Pipeline:
    """Build the R2 preprocessing block, unfitted.

    ``TotalChargesCleaner`` runs first, exactly as in every earlier phase, then
    the cardinality guard, then the encoder. Every fitted component — the
    scaler's statistics, the encoder's category lists and the guard's counts —
    is estimated inside ``fit`` and therefore inside a training fold.
    """
    return Pipeline(
        steps=[
            (CLEANER_STEP, TotalChargesCleaner()),
            (GUARD_STEP, CategoricalCardinalityGuard()),
            (ENCODER_STEP, build_native_column_encoder()),
        ]
    )


def build_hgb_native(seed: int | None = None) -> HistGradientBoostingClassifier:
    """Return the Phase 7 HGB with the categorical mask, and nothing else changed.

    The configuration is :data:`~churn.modeling.families.HIST_GRADIENT_BOOSTING_PARAMS`
    — the frozen M2 dictionary — with ``categorical_features`` replaced by the
    explicit mask. Deriving it from that dictionary instead of restating it is
    what guarantees the two estimators cannot drift apart on any other
    parameter.
    """
    random_state = get_config().seed if seed is None else seed
    params = {**HIST_GRADIENT_BOOSTING_PARAMS, "categorical_features": list(CATEGORICAL_MASK)}
    return HistGradientBoostingClassifier(random_state=random_state, **params)


def build_native_pipeline() -> Pipeline:
    """Build the complete R2 estimator, unfitted."""
    return Pipeline(
        steps=[
            (PREPROCESSOR_STEP, build_native_preprocessor()),
            (CLASSIFIER_STEP, build_hgb_native()),
        ]
    )


def build_reference_pipeline(family: str) -> Pipeline:
    """Build R0 or R1 through the Phase 7 builder, on the 19 original features."""
    return build_family_pipeline(family, ())


@dataclass(frozen=True)
class RepresentationRun:
    """One experiment of the gate, with its paired deltas keyed by reference."""

    experiment: str
    label: str
    model: str
    representation: str
    n_transformed_features: int
    evaluation: ModelEvaluation
    paired_deltas: dict[str, dict[str, PairedComparison]] = field(default_factory=dict)


def _transformed_width(pipeline: Pipeline) -> int:
    encoder = pipeline.named_steps[PREPROCESSOR_STEP].named_steps[ENCODER_STEP]
    return len(encoder.get_feature_names_out())


def run_experiment(
    experiment: str,
    label: str,
    model: str,
    representation: str,
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
) -> RepresentationRun:
    """Cross-validate one experiment on the shared folds.

    Args:
        experiment: Identifier such as ``"R2"``.
        label: Human-readable name.
        model: Model key, used for colours and labels in figures.
        representation: Which representation this run uses.
        pipeline: The unfitted estimator, cloned fresh inside every fold.
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        splitter: The shared splitter — identical folds for every experiment.

    Returns:
        The :class:`RepresentationRun`, without deltas.
    """
    evaluation = evaluate_model(experiment, pipeline, features, target, splitter)

    # Structural width, read from a pipeline fitted on the training pool. This
    # fitted object produces no metric: every number comes from the fold loop.
    probe = clone(pipeline).fit(features, target)

    return RepresentationRun(
        experiment=experiment,
        label=label,
        model=model,
        representation=representation,
        n_transformed_features=_transformed_width(probe),
        evaluation=evaluation,
    )


def pair(run: RepresentationRun, reference: RepresentationRun, metrics: Sequence[str]) -> None:
    """Attach the paired per-fold deltas of ``run`` against ``reference``.

    Args:
        run: The experiment being compared.
        reference: The experiment it is measured against.
        metrics: Metric names to pair.
    """
    run.paired_deltas[reference.experiment] = {
        metric: compare(run.evaluation, reference.evaluation, metric) for metric in metrics
    }
    primary = run.paired_deltas[reference.experiment][PRIMARY_METRIC]
    logger.info(
        "%s vs %s: delta %s %+.5f in %d/%d folds",
        run.experiment,
        reference.experiment,
        PRIMARY_METRIC,
        primary.mean,
        primary.n_positive,
        len(run.evaluation.folds),
    )


def run_representation_gate(
    features: pd.DataFrame,
    target: pd.Series,
    splitter: StratifiedKFold,
    metrics: Sequence[str],
    gate: Callable[[RepresentationRun], None] | None = None,
) -> OrderedDict[str, RepresentationRun]:
    """Run R0, R1 and R2 on the shared folds and pair R2 against both references.

    Args:
        features: Canonical feature matrix of the training pool.
        target: Encoded target.
        splitter: Shared splitter.
        metrics: Metrics to pair — primary first.
        gate: Called with R0 and then with R1, **before** R2 runs. It is where
            the reproduction checks belong: if a reference cannot be reproduced,
            no delta against it means anything, so let the callable raise.

    Returns:
        Ordered mapping of experiment id to run.
    """
    runs: OrderedDict[str, RepresentationRun] = OrderedDict()

    runs[REFERENCE_LOGISTIC] = run_experiment(
        REFERENCE_LOGISTIC,
        "Logistic regression, common representation",
        LOGISTIC_REGRESSION,
        COMMON_REPRESENTATION,
        build_reference_pipeline(LOGISTIC_REGRESSION),
        features,
        target,
        splitter,
    )
    if gate is not None:
        gate(runs[REFERENCE_LOGISTIC])

    runs[REFERENCE_HGB_COMMON] = run_experiment(
        REFERENCE_HGB_COMMON,
        "HistGradientBoosting, common representation",
        HIST_GRADIENT_BOOSTING,
        COMMON_REPRESENTATION,
        build_reference_pipeline(HIST_GRADIENT_BOOSTING),
        features,
        target,
        splitter,
    )
    if gate is not None:
        gate(runs[REFERENCE_HGB_COMMON])

    runs[NATIVE_HGB] = run_experiment(
        NATIVE_HGB,
        "HistGradientBoosting, native categorical representation",
        HGB_NATIVE,
        NATIVE_REPRESENTATION,
        build_native_pipeline(),
        features,
        target,
        splitter,
    )

    # Question A first, then question B. Two references, never summed.
    pair(runs[NATIVE_HGB], runs[REFERENCE_HGB_COMMON], metrics)
    pair(runs[NATIVE_HGB], runs[REFERENCE_LOGISTIC], metrics)
    return runs


def classify_representation(comparison: PairedComparison, n_folds: int) -> tuple[str, str]:
    """Label the representation effect of R2 against R1.

    An engineering heuristic about direction and consistency, fixed before any
    result was seen. It carries **no minimum-gain cutoff**: magnitude is
    reported separately and weighed by a reader, and with five folds no
    significance test is possible.

    Args:
        comparison: Paired primary-metric deltas of R2 against R1.
        n_folds: Number of folds.

    Returns:
        ``(classification, rationale)``.
    """
    positive, negative, mean = comparison.n_positive, comparison.n_negative, comparison.mean
    summary = f"mean delta {PRIMARY_METRIC} = {mean:+.5f} with {positive}/{n_folds} folds improving"

    if mean <= 0 or negative > n_folds / 2:
        return (
            REPRESENTATION_NOT_SUPPORTED,
            f"{summary}: the native representation does not improve the family under the "
            "frozen protocol.",
        )
    if positive >= HELPFUL_MIN_POSITIVE_FOLDS:
        return (
            REPRESENTATION_HELPFUL,
            f"{summary}: positive and consistent in direction under the frozen protocol.",
        )
    return (
        REPRESENTATION_INCONCLUSIVE,
        f"{summary}: the direction is not consistent enough across folds to read as an "
        "effect with five folds and no significance test.",
    )


def standing_against_logistic(comparison: PairedComparison, n_folds: int) -> tuple[str, str]:
    """Say where R2 sits relative to R0, without deciding anything.

    Deliberately judged by direction and consistency rather than by a magnitude
    threshold: inventing a "close enough to tie" cutoff would smuggle a decision
    into a phase whose decision is explicitly deferred.

    Returns:
        ``(standing, rationale)``.
    """
    positive, negative, mean = comparison.n_positive, comparison.n_negative, comparison.mean
    summary = f"mean delta {PRIMARY_METRIC} = {mean:+.5f} in {positive}/{n_folds} folds"

    if mean < 0 and negative > n_folds / 2:
        return (
            BELOW_LOGISTIC,
            f"{summary}: still below the logistic regression, consistently across folds.",
        )
    if mean > 0 and positive > n_folds / 2:
        return (
            ABOVE_LOGISTIC,
            f"{summary}: above the logistic regression, consistently across folds.",
        )
    return (
        COMPARABLE_TO_LOGISTIC,
        f"{summary}: the direction is not consistent across folds. 'Comparable' describes "
        "that inconsistency; it is not a demonstration of equivalence, which five folds "
        "cannot establish.",
    )
