"""Phase 10: the exact decomposition of the frozen logistic regression.

The model is closed. This phase asks one question — **how does the frozen
pipeline turn the 19 contracted features into an estimated churn score** — and it
answers it by writing down the function the model already is, not by probing it.

**Why there is no SHAP here.** SHAP, LIME, permutation importance and partial
dependence are approximations for models whose response surface is unknown. This
model's is known exactly::

    logit(P) = intercept + sum_j  coefficient_j * transformed_feature_j
    P        = sigmoid(logit)

Every quantity below is an algebraic rearrangement of that identity. Introducing
a sampling-based estimator to approximate a function we can evaluate in closed
form would add a dependency, a random seed and an approximation error in exchange
for nothing.

**Nothing is fitted.** This module transforms with ``.transform`` and reads
learned attributes; it contains no ``fit``, no search and no estimator
construction. It also cannot reach the holdout: it never loads anything, so the
partition it describes is whatever the caller supplies, and the caller for the
global interpretation is the training pool.

**The one-hot parameterisation is redundant, and that governs how a coefficient
may be read.** Every level of every categorical feature has its own column and
the model also has an intercept, so the parameters are not identified in the
textbook sense: adding a constant to every level of one feature and subtracting
it from the intercept produces the same predictions. L2 regularisation picks one
point in that family, which means an individual dummy coefficient is a property
of *this fitted parameterisation*, not a stable quantity. What **is** invariant
under that shift is the **difference between two levels of the same feature**,
which is why :func:`level_contrasts` is the primary categorical output and a
ranking of ``|beta|`` across mixed column types is deliberately not produced.

**Grouping back to the 19 raw features is exact, not approximate.** The
transformed columns partition into 19 disjoint groups, so summing the
per-column terms inside each group and adding the intercept reproduces the logit
exactly. That identity is verified numerically rather than asserted.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.preprocessing.pipeline import CATEGORICAL_STEP, ENCODER_STEP, NUMERIC_STEP

logger = logging.getLogger(__name__)

#: What this phase is, recorded so no reader has to infer it.
ANALYSIS_TYPE = "FROZEN_MODEL_INTERPRETATION"

#: Kinds of raw feature group.
NUMERIC_KIND = "numeric"
CATEGORICAL_KIND = "categorical"

#: Tolerance for the reconstruction gates, on the logit and on the probability.
#:
#: Both quantities are order 1 and are computed as a dot product of 46 float64
#: terms, so the accumulated rounding error is bounded by a few multiples of
#: ``2**-52 ~ 2.2e-16``. ``1e-12`` therefore sits roughly four orders of
#: magnitude above the noise floor and many orders below any error that would
#: indicate the decomposition is describing a different function. It is strict:
#: a genuine mismatch — a wrong column, a missed intercept, a transposed
#: coefficient — fails it by a wide margin rather than by a rounding digit.
RECONSTRUCTION_TOLERANCE = 1e-12

#: The metric the 19 raw features are ranked by, **fixed here before any value
#: was computed**. The alternatives below are recorded as supporting columns.
#:
#: ``std`` is chosen over ``mean_absolute_centered_contribution`` for one reason:
#: it is the standard deviation of that feature's additive term in the logit, so
#: it answers "how much does this feature move the score around, in log-odds, in
#: this population" in the same unit as the logit itself. The mean absolute
#: centered contribution answers the same question with a different robustness
#: profile and is reported beside it.
RANKING_METRIC = "std"
SUPPORTING_METRICS: tuple[str, ...] = ("mean_absolute_centered_contribution", "iqr")

#: What the ranking is called, and what it must never be called. The phrasing is
#: pinned in code because the constraint is on the claim, not on the arithmetic.
RANKING_NAME = "empirical contribution dispersion in the training pool"
RANKING_IS_NOT: tuple[str, ...] = (
    "true feature importance",
    "causal importance",
    "business importance",
)


class InterpretationError(RuntimeError):
    """The frozen pipeline cannot be decomposed as this phase requires."""


class ReconstructionError(RuntimeError):
    """The manual decomposition does not reproduce the pipeline's own output."""


@dataclass(frozen=True)
class LinearTerms:
    """The learned parameters of the frozen linear model, read structurally.

    ``coefficients`` is the single row of ``coef_`` that scores the positive
    class, in transformed-column order; ``names`` labels those columns.
    """

    estimator: str
    classes: tuple[int, ...]
    positive_class_label: int
    positive_class_column: int
    intercept: float
    coefficients: np.ndarray
    names: tuple[str, ...]
    n_iter: tuple[int, ...]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    categories: tuple[tuple[str, ...], ...]

    @property
    def n_transformed_features(self) -> int:
        """Number of columns the classifier receives."""
        return int(self.coefficients.size)


def _steps(pipeline: Pipeline) -> tuple[object, object, object, object]:
    """Return ``(classifier, encoder, scaler, one_hot)`` or fail loudly."""
    try:
        classifier = pipeline.named_steps[CLASSIFIER_STEP]
        encoder = pipeline.named_steps[PREPROCESSOR_STEP].named_steps[ENCODER_STEP]
        scaler = encoder.named_transformers_[NUMERIC_STEP]
        one_hot = encoder.named_transformers_[CATEGORICAL_STEP]
    except (KeyError, AttributeError) as error:
        raise InterpretationError(
            "The pipeline does not have the frozen structure "
            f"({PREPROCESSOR_STEP} -> {ENCODER_STEP} -> {CLASSIFIER_STEP}), or it has not been "
            "fitted. There is no linear function to decompose."
        ) from error
    return classifier, encoder, scaler, one_hot


def positive_class_column(classes: Sequence[int], positive_label: int) -> int:
    """Return the column of ``predict_proba`` holding the positive class.

    Resolved from ``classes_`` rather than assumed. The assumption happens to
    hold for this estimator and is still the wrong thing to write down: reading
    the wrong column inverts every statement in this phase without raising.

    Raises:
        InterpretationError: If the positive label is not uniquely present.
    """
    matches = np.flatnonzero(np.asarray(classes) == positive_label)
    if matches.size != 1:
        raise InterpretationError(
            f"The positive label {positive_label!r} appears {matches.size} time(s) in "
            f"classes_={list(classes)}; the positive column is not identifiable."
        )
    return int(matches[0])


def extract_terms(pipeline: Pipeline, positive_label: int = 1) -> LinearTerms:
    """Read the frozen model's learned parameters out of the fitted pipeline.

    Every attribute is read by name from the step it belongs to. Nothing is
    reconstructed from a factory and nothing is inferred from a string.

    Raises:
        InterpretationError: If the structure is not the frozen one, if the
            classifier is not a single-row linear model, or if the positive
            class does not sit in the column ``decision_function`` scores.
    """
    classifier, encoder, scaler, one_hot = _steps(pipeline)

    coefficients = np.asarray(classifier.coef_, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[0] != 1:
        raise InterpretationError(
            f"coef_ has shape {coefficients.shape}; this decomposition is written for the binary "
            "single-row case and must not silently describe a multinomial model."
        )

    classes = tuple(int(value) for value in classifier.classes_)
    column = positive_class_column(classes, positive_label)

    # ``decision_function`` of a binary linear classifier scores ``classes_[1]``.
    # If the positive class were the other one, the manual logit would need its
    # sign flipped, so the mismatch is refused rather than papered over.
    if column != 1:
        raise InterpretationError(
            f"The positive class sits in column {column}, but decision_function scores "
            f"classes_[1]={classes[1]!r}. The decomposition would carry the wrong sign."
        )

    names = tuple(str(name) for name in encoder.get_feature_names_out())
    if len(names) != coefficients.shape[1]:
        raise InterpretationError(
            f"{len(names)} transformed feature names against {coefficients.shape[1]} coefficients."
        )

    return LinearTerms(
        estimator=type(classifier).__name__,
        classes=classes,
        positive_class_label=int(positive_label),
        positive_class_column=column,
        intercept=float(np.asarray(classifier.intercept_, dtype=float).ravel()[0]),
        coefficients=coefficients[0],
        names=names,
        n_iter=tuple(int(value) for value in np.asarray(classifier.n_iter_).ravel()),
        scaler_mean=np.asarray(scaler.mean_, dtype=float),
        scaler_scale=np.asarray(scaler.scale_, dtype=float),
        numeric_features=tuple(str(name) for name in scaler.feature_names_in_),
        categorical_features=tuple(str(name) for name in one_hot.feature_names_in_),
        categories=tuple(tuple(str(level) for level in levels) for levels in one_hot.categories_),
    )


@dataclass(frozen=True)
class FeatureGroup:
    """One raw feature and the transformed columns it owns.

    ``levels`` is empty for a numeric feature and lists the encoder's categories,
    in column order, for a categorical one.
    """

    feature: str
    kind: str
    columns: tuple[int, ...]
    levels: tuple[str, ...]

    @property
    def n_columns(self) -> int:
        """How many transformed columns this raw feature produced."""
        return len(self.columns)


def feature_groups(terms: LinearTerms) -> tuple[FeatureGroup, ...]:
    """Map the 46 transformed columns onto the 19 raw features they came from.

    **Derived from the encoder's own structure, not from the column names.** The
    fitted ``ColumnTransformer`` emits its numeric block first, one column per
    scaled feature in ``feature_names_in_`` order, then its categorical block,
    ``len(categories_[i])`` consecutive columns for the i-th categorical feature.
    That layout is a property of the fitted object, so the mapping is index
    arithmetic over ``categories_``.

    Splitting ``"categorical__PaymentMethod_Mailed check"`` on ``"_"`` would be
    the fragile alternative: both the separator and the level values contain the
    delimiter, and a feature named after another feature's prefix would silently
    mis-assign. The generated names are used **only** to cross-check the result.

    Raises:
        InterpretationError: If the columns do not partition exactly, or if the
            derived mapping disagrees with the encoder's own generated names.
    """
    groups: list[FeatureGroup] = []
    cursor = 0

    for feature in terms.numeric_features:
        groups.append(
            FeatureGroup(feature=feature, kind=NUMERIC_KIND, columns=(cursor,), levels=())
        )
        cursor += 1

    for feature, levels in zip(terms.categorical_features, terms.categories, strict=True):
        span = tuple(range(cursor, cursor + len(levels)))
        groups.append(
            FeatureGroup(feature=feature, kind=CATEGORICAL_KIND, columns=span, levels=levels)
        )
        cursor += len(levels)

    if cursor != terms.n_transformed_features:
        raise InterpretationError(
            f"The groups cover {cursor} columns against {terms.n_transformed_features} "
            "coefficients; the encoder's layout is not the one this mapping assumes."
        )
    verify_group_partition(groups, terms)
    _cross_check_against_generated_names(groups, terms)
    return tuple(groups)


def verify_group_partition(groups: Sequence[FeatureGroup], terms: LinearTerms) -> None:
    """Check that the groups are a true partition of the transformed columns.

    Three failure modes, each of which would silently corrupt every grouped
    number downstream: an orphan column contributes to no feature, a shared
    column contributes to two, and a wrong total means the layout changed.

    Raises:
        InterpretationError: On any of them.
    """
    assigned: list[int] = [column for group in groups for column in group.columns]
    total = terms.n_transformed_features

    duplicated = sorted({column for column in assigned if assigned.count(column) > 1})
    if duplicated:
        raise InterpretationError(
            f"Transformed column(s) {duplicated} belong to more than one raw feature; a grouped "
            "contribution would count them twice."
        )

    orphaned = sorted(set(range(total)) - set(assigned))
    if orphaned:
        raise InterpretationError(
            f"Transformed column(s) {orphaned} belong to no raw feature; a grouped contribution "
            "would silently drop them."
        )

    if len(assigned) != total:
        raise InterpretationError(f"{len(assigned)} assigned columns against {total} coefficients.")


def _cross_check_against_generated_names(
    groups: Sequence[FeatureGroup],
    terms: LinearTerms,
) -> None:
    """Confirm the structural mapping agrees with the encoder's generated names.

    A **check**, never the derivation. ``verbose_feature_names_out=True`` prefixes
    every column with its transformer name, so ``numeric__`` and ``categorical__``
    are reliable block markers even though the rest of the string is not a
    reliable parse. Each categorical column must also *contain* the raw feature
    name it was assigned to; that is a containment test, not a split.

    Raises:
        InterpretationError: If the structural mapping and the names disagree.
    """
    for group in groups:
        prefix = f"{NUMERIC_STEP}__" if group.kind == NUMERIC_KIND else f"{CATEGORICAL_STEP}__"
        for column in group.columns:
            name = terms.names[column]
            if not name.startswith(prefix):
                raise InterpretationError(
                    f"Column {column} was mapped to the {group.kind} feature {group.feature!r} but "
                    f"the encoder named it {name!r}."
                )
            if group.feature not in name:
                raise InterpretationError(
                    f"Column {column} was mapped to {group.feature!r} but the encoder named it "
                    f"{name!r}, which does not mention that feature."
                )


def transform_features(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return the transformed matrix ``Z`` the classifier actually receives.

    Uses the **fitted** preprocessor's ``transform``. There is no ``fit`` and no
    ``fit_transform`` anywhere in this module: refitting the scaler or the
    encoder on the data being described would change the very representation the
    coefficients were learned for.
    """
    return np.asarray(
        pipeline.named_steps[PREPROCESSOR_STEP].transform(features),
        dtype=float,
    )


def manual_logit(terms: LinearTerms, transformed: np.ndarray) -> np.ndarray:
    """Return ``intercept + Z @ coefficients`` — the model's own linear predictor."""
    return float(terms.intercept) + np.asarray(transformed, dtype=float) @ terms.coefficients


def sigmoid(logit: np.ndarray) -> np.ndarray:
    """Return the logistic function, evaluated without overflow on either tail."""
    values = np.asarray(logit, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def logit_of(probability: float) -> float:
    """Return ``log(p / (1 - p))``.

    Raises:
        InterpretationError: If ``p`` is not strictly inside ``(0, 1)``.
    """
    value = float(probability)
    if not 0.0 < value < 1.0:
        raise InterpretationError(f"{value!r} is not a probability strictly inside (0, 1).")
    return float(np.log(value / (1.0 - value)))


@dataclass(frozen=True)
class Reconstruction:
    """How exactly the manual decomposition reproduces the fitted pipeline."""

    n_rows: int
    max_abs_logit_error: float
    max_abs_probability_error: float
    max_grouped_reconstruction_error: float
    tolerance: float

    @property
    def within_tolerance(self) -> bool:
        """Whether all three errors clear :data:`RECONSTRUCTION_TOLERANCE`."""
        return (
            max(
                self.max_abs_logit_error,
                self.max_abs_probability_error,
                self.max_grouped_reconstruction_error,
            )
            <= self.tolerance
        )

    def as_dict(self) -> dict[str, object]:
        """Return the record as a JSON-encodable mapping."""
        return {
            "n_rows": int(self.n_rows),
            "max_abs_logit_error": float(self.max_abs_logit_error),
            "max_abs_probability_error": float(self.max_abs_probability_error),
            "max_grouped_reconstruction_error": float(self.max_grouped_reconstruction_error),
            "tolerance": float(self.tolerance),
            "within_tolerance": bool(self.within_tolerance),
        }


def group_contributions(
    terms: LinearTerms,
    groups: Sequence[FeatureGroup],
    transformed: np.ndarray,
) -> np.ndarray:
    """Return the ``(n_rows, n_groups)`` matrix of per-raw-feature contributions.

    Each entry is ``sum over that group's columns of coefficient_j * Z_ij``,
    which for a numeric feature is ``beta * standardised_value`` and for a
    categorical feature is the coefficient of the level that is active — the
    other columns of the group are zero. Writing it as the group dot product
    rather than as a lookup keeps the identity

    ``intercept + row.sum() == logit``

    true by construction, including for a row whose category was unseen at fit
    time and which ``handle_unknown="ignore"`` therefore encodes as all zeros.
    """
    matrix = np.asarray(transformed, dtype=float)
    return np.column_stack(
        [
            matrix[:, list(group.columns)] @ terms.coefficients[list(group.columns)]
            for group in groups
        ]
    )


def verify_reconstruction(
    pipeline: Pipeline,
    terms: LinearTerms,
    groups: Sequence[FeatureGroup],
    features: pd.DataFrame,
    tolerance: float = RECONSTRUCTION_TOLERANCE,
) -> Reconstruction:
    """Check the decomposition against the pipeline's own outputs. The gate.

    Three comparisons, in increasing strength:

    1. ``intercept + Z @ coefficients`` against ``decision_function``;
    2. ``sigmoid`` of that against ``predict_proba[:, positive_column]``;
    3. ``intercept + sum of the 19 grouped contributions`` against
       ``decision_function`` — which is what proves the regrouping of the 43
       one-hot columns loses nothing and double-counts nothing.

    An interpretation of a model that does not reproduce its own decision
    function is not an interpretation of that model, so a failure raises rather
    than being reported as a caveat.

    Raises:
        ReconstructionError: If any error exceeds ``tolerance``.
    """
    transformed = transform_features(pipeline, features)
    reference_logit = np.asarray(pipeline.decision_function(features), dtype=float).ravel()
    reference_probability = np.asarray(pipeline.predict_proba(features), dtype=float)[
        :, terms.positive_class_column
    ]

    derived_logit = manual_logit(terms, transformed)
    derived_probability = sigmoid(derived_logit)
    grouped_logit = float(terms.intercept) + group_contributions(terms, groups, transformed).sum(
        axis=1
    )

    reconstruction = Reconstruction(
        n_rows=int(len(features)),
        max_abs_logit_error=float(np.max(np.abs(derived_logit - reference_logit))),
        max_abs_probability_error=float(
            np.max(np.abs(derived_probability - reference_probability))
        ),
        max_grouped_reconstruction_error=float(np.max(np.abs(grouped_logit - reference_logit))),
        tolerance=float(tolerance),
    )

    if not reconstruction.within_tolerance:
        raise ReconstructionError(
            "The manual decomposition does not reproduce the frozen pipeline within "
            f"{tolerance:g}:\n"
            f"  logit       : {reconstruction.max_abs_logit_error:g}\n"
            f"  probability : {reconstruction.max_abs_probability_error:g}\n"
            f"  grouped     : {reconstruction.max_grouped_reconstruction_error:g}\n"
            "An interpretation of a function the decomposition does not reproduce would describe "
            "a different model."
        )

    logger.info(
        "Reconstruction exact within %g over %d rows (logit %.3g, probability %.3g, grouped %.3g).",
        tolerance,
        reconstruction.n_rows,
        reconstruction.max_abs_logit_error,
        reconstruction.max_abs_probability_error,
        reconstruction.max_grouped_reconstruction_error,
    )
    return reconstruction


@dataclass(frozen=True)
class NumericTerm:
    """One standardised numeric feature, and its exact re-expressions."""

    feature: str
    column: int
    standardized_coefficient: float
    scaler_mean: float
    scaler_scale: float

    @property
    def raw_logit_coefficient(self) -> float:
        """Change in model log-odds per **one unit of the raw feature**.

        An exact algebraic re-expression, not a new estimate. The transformed
        value is ``(x - mean) / scale``, so the linear term is
        ``beta * (x - mean) / scale`` and its derivative with respect to ``x`` is
        ``beta / scale``. The intercept absorbs ``-beta * mean / scale``; it is
        not reported per feature because the model has one intercept, not three.
        """
        return float(self.standardized_coefficient) / float(self.scaler_scale)

    @property
    def odds_ratio_per_1_sd(self) -> float:
        """``exp(beta)`` — the modelled multiplicative change in odds per 1 SD."""
        return float(np.exp(self.standardized_coefficient))

    def as_dict(self) -> dict[str, object]:
        """Return the record as a JSON-encodable mapping."""
        return {
            "feature": self.feature,
            "transformed_column": int(self.column),
            "standardized_coefficient": float(self.standardized_coefficient),
            "scaler_mean": float(self.scaler_mean),
            "scaler_scale": float(self.scaler_scale),
            "raw_logit_coefficient": self.raw_logit_coefficient,
            "odds_ratio_per_1_sd": self.odds_ratio_per_1_sd,
            "raw_unit": RAW_UNITS[self.feature],
        }


#: The unit each raw numeric coefficient is expressed per, stated so no reader
#: has to guess and so no impressive-looking increment is invented after the
#: fact. These are the dataset's own units; the dataset states no currency.
RAW_UNITS: dict[str, str] = {
    "tenure": "one month of tenure",
    "MonthlyCharges": "one monetary unit of the dataset's monthly charge scale",
    "TotalCharges": "one monetary unit of the dataset's total charge scale",
}


def numeric_terms(terms: LinearTerms, groups: Sequence[FeatureGroup]) -> tuple[NumericTerm, ...]:
    """Return one :class:`NumericTerm` per standardised numeric feature."""
    records: list[NumericTerm] = []
    for position, group in enumerate(group for group in groups if group.kind == NUMERIC_KIND):
        column = group.columns[0]
        records.append(
            NumericTerm(
                feature=group.feature,
                column=column,
                standardized_coefficient=float(terms.coefficients[column]),
                scaler_mean=float(terms.scaler_mean[position]),
                scaler_scale=float(terms.scaler_scale[position]),
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class LevelContrast:
    """The modelled log-odds difference between two levels of one feature.

    ``delta_log_odds`` is **always** exactly ``beta_a - beta_b``, whatever
    ``structurally_coupled`` says. That flag changes nothing about the
    arithmetic; it changes what a reader may conclude from it. When one of the
    two levels is a sentinel that this dataset ties to another raw feature — a
    customer is ``OnlineSecurity = 'No internet service'`` **exactly when** they
    are ``InternetService = 'No'`` — then "move only this feature from A to B and
    hold everything else fixed" describes a row that does not occur, so the
    contrast is an algebraic property of the function rather than a single-feature
    change a real customer could undergo.
    """

    feature: str
    level_a: str
    level_b: str
    delta_log_odds: float
    structurally_coupled: bool = False

    @property
    def modeled_odds_ratio(self) -> float:
        """``exp(beta_a - beta_b)``, the modelled odds ratio of A against B."""
        return float(np.exp(self.delta_log_odds))

    @property
    def raw_single_feature_counterfactual_supported(self) -> bool:
        """Whether changing only this raw feature between A and B is a valid row.

        ``False`` when either level participates in a structural dependency
        verified on the training pool. It is a statement about what the *data*
        permits, not about whether the contrast was computed correctly.
        """
        return not self.structurally_coupled

    @property
    def reading(self) -> str:
        """The sentence the report is allowed to use for this contrast."""
        if self.structurally_coupled:
            return (
                f"the algebraic contrast between the encoded levels {self.level_a!r} and "
                f"{self.level_b!r} of {self.feature} is {self.delta_log_odds:+.4f} in model "
                "log-odds. At least one of these levels is structurally coupled to another raw "
                "feature in this dataset, so this is NOT a single-feature change a real customer "
                "could undergo: altering only this one-hot while holding the other raw features "
                "fixed would describe a combination that does not occur in the training pool"
            )
        return (
            f"within the frozen model, encoding {self.feature} as {self.level_a!r} instead of "
            f"{self.level_b!r} and holding every other transformed feature fixed changes the "
            f"model log-odds by {self.delta_log_odds:+.4f}"
        )

    def as_dict(self) -> dict[str, object]:
        """Return the record as a JSON-encodable mapping."""
        return {
            "feature": self.feature,
            "level_a": self.level_a,
            "level_b": self.level_b,
            "delta_log_odds": float(self.delta_log_odds),
            "modeled_odds_ratio": self.modeled_odds_ratio,
            "structurally_coupled": bool(self.structurally_coupled),
            "raw_single_feature_counterfactual_supported": (
                self.raw_single_feature_counterfactual_supported
            ),
            "reading": self.reading,
        }


def level_contrasts(
    terms: LinearTerms,
    group: FeatureGroup,
    coupled_levels: Sequence[str] = (),
) -> tuple[LevelContrast, ...]:
    """Return every ordered within-feature contrast for one categorical feature.

    The invariant quantity. A single dummy coefficient depends on where the
    redundant parameterisation happened to place the intercept; a difference
    between two levels of the same feature does not, because the shared shift
    cancels.

    Args:
        terms: The extracted linear terms.
        group: The categorical feature group.
        coupled_levels: Levels of this feature that participate in a structural
            dependency verified on the interpretation population. A contrast
            touching one of them is flagged, never dropped or altered.

    Raises:
        InterpretationError: If the group is not categorical.
    """
    if group.kind != CATEGORICAL_KIND:
        raise InterpretationError(f"{group.feature!r} is not a categorical feature.")

    coupled = frozenset(coupled_levels)
    betas = {
        level: float(terms.coefficients[column])
        for level, column in zip(group.levels, group.columns, strict=True)
    }
    return tuple(
        LevelContrast(
            feature=group.feature,
            level_a=first,
            level_b=second,
            delta_log_odds=betas[first] - betas[second],
            structurally_coupled=first in coupled or second in coupled,
        )
        for first in group.levels
        for second in group.levels
        if first != second
    )


@dataclass(frozen=True)
class CategoricalTerm:
    """One categorical feature: its levels, their coefficients and their spread.

    ``spread`` is deliberately **not** called importance. It is the width of this
    feature's coefficient range, which bounds how much the feature can move the
    logit *if* a customer's level changes — it says nothing about how often that
    happens in any population, which is what the contribution dispersion adds.
    """

    feature: str
    levels: tuple[str, ...]
    columns: tuple[int, ...]
    coefficients: tuple[float, ...]
    coupled_levels: tuple[str, ...] = ()

    @property
    def coefficient_per_level(self) -> dict[str, float]:
        """The fitted coefficient of each level, in encoder order."""
        return dict(zip(self.levels, self.coefficients, strict=True))

    @property
    def lowest_coefficient_level(self) -> str:
        """The level with the smallest coefficient. Ties break on encoder order."""
        return self.levels[int(np.argmin(self.coefficients))]

    @property
    def highest_coefficient_level(self) -> str:
        """The level with the largest coefficient. Ties break on encoder order."""
        return self.levels[int(np.argmax(self.coefficients))]

    @property
    def minimum(self) -> float:
        """The smallest coefficient among the levels."""
        return float(min(self.coefficients))

    @property
    def maximum(self) -> float:
        """The largest coefficient among the levels."""
        return float(max(self.coefficients))

    @property
    def coefficient_spread(self) -> float:
        """``max(beta) - min(beta)`` across this feature's levels."""
        return self.maximum - self.minimum

    @property
    def max_pairwise_modeled_odds_ratio(self) -> float:
        """``exp(spread)`` — the widest modelled odds ratio inside this feature."""
        return float(np.exp(self.coefficient_spread))

    def as_dict(self) -> dict[str, object]:
        """Return the record as a JSON-encodable mapping."""
        return {
            "feature": self.feature,
            "n_levels": len(self.levels),
            "levels": list(self.levels),
            "transformed_columns": [int(column) for column in self.columns],
            "coefficient_per_level": {
                level: float(value) for level, value in self.coefficient_per_level.items()
            },
            "lowest_coefficient_level": self.lowest_coefficient_level,
            "highest_coefficient_level": self.highest_coefficient_level,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "spread": self.coefficient_spread,
            "max_pairwise_modeled_odds_ratio": self.max_pairwise_modeled_odds_ratio,
            "structurally_coupled_levels": list(self.coupled_levels),
            "has_structurally_coupled_level": bool(self.coupled_levels),
            "structurally_coupled_per_level": {
                level: level in self.coupled_levels for level in self.levels
            },
            "strongest_pairwise_contrast": LevelContrast(
                feature=self.feature,
                level_a=self.highest_coefficient_level,
                level_b=self.lowest_coefficient_level,
                delta_log_odds=self.coefficient_spread,
                structurally_coupled=(
                    self.highest_coefficient_level in self.coupled_levels
                    or self.lowest_coefficient_level in self.coupled_levels
                ),
            ).as_dict(),
            "reference_level_removed": False,
            "reference_level": None,
            "encoding_note": (
                "OneHotEncoder(handle_unknown='ignore') with NO dropped baseline: every level has "
                "its own column and the model also has an intercept, so exp(beta) of a single "
                "level is NOT an odds ratio against a reference category. The identified quantity "
                "is the within-feature contrast beta_a - beta_b"
            ),
        }


def categorical_terms(
    terms: LinearTerms,
    groups: Sequence[FeatureGroup],
    coupled_levels: Mapping[str, Sequence[str]] | None = None,
) -> tuple[CategoricalTerm, ...]:
    """Return one :class:`CategoricalTerm` per one-hot encoded feature.

    Args:
        terms: The extracted linear terms.
        groups: The raw feature groups.
        coupled_levels: ``{feature: levels}`` participating in a structural
            dependency verified on the interpretation population. Levels are
            flagged, never removed: every coefficient is still reported.
    """
    coupled = coupled_levels or {}
    return tuple(
        CategoricalTerm(
            feature=group.feature,
            levels=group.levels,
            columns=group.columns,
            coefficients=tuple(float(terms.coefficients[column]) for column in group.columns),
            coupled_levels=tuple(
                level for level in group.levels if level in set(coupled.get(group.feature, ()))
            ),
        )
        for group in groups
        if group.kind == CATEGORICAL_KIND
    )


@dataclass(frozen=True)
class ContributionDispersion:
    """How much one raw feature's additive term varied across a population.

    Every field is a summary of the same per-row quantity: the contribution that
    feature made to the logit. The population is whatever frame was passed in,
    and the value is therefore a joint property of the coefficients, the
    preprocessing **and** that population's composition — not of the feature.
    """

    feature: str
    kind: str
    n_rows: int
    mean: float
    std: float
    median: float
    q1: float
    q3: float
    minimum: float
    maximum: float
    mean_absolute_centered_contribution: float

    @property
    def iqr(self) -> float:
        """``q3 - q1`` of the contribution distribution."""
        return float(self.q3 - self.q1)

    @property
    def range(self) -> float:
        """``max - min`` of the contribution distribution."""
        return float(self.maximum - self.minimum)

    def as_dict(self) -> dict[str, object]:
        """Return the record as a JSON-encodable mapping."""
        return {
            "feature": self.feature,
            "kind": self.kind,
            "n_rows": int(self.n_rows),
            "mean": float(self.mean),
            "std": float(self.std),
            "median": float(self.median),
            "q1": float(self.q1),
            "q3": float(self.q3),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "iqr": self.iqr,
            "range": self.range,
            "mean_absolute_centered_contribution": float(self.mean_absolute_centered_contribution),
        }


#: The exact definition of the secondary dispersion metric, written down because
#: "mean absolute deviation" is ambiguous in the wild — it is taken about the
#: **mean**, not about the median, and it is not divided by anything else.
MEAN_ABSOLUTE_CENTERED_DEFINITION = (
    "mean(abs(contribution - mean(contribution))), taken about the MEAN of that feature's "
    "contribution over the training pool, in log-odds units"
)

#: The exact definition of the primary ranking metric.
STD_DEFINITION = (
    "population standard deviation (ddof=0) of that feature's additive contribution to the logit "
    "over the training pool, in log-odds units"
)


def contribution_dispersion(
    groups: Sequence[FeatureGroup],
    contributions: np.ndarray,
) -> tuple[ContributionDispersion, ...]:
    """Summarise each raw feature's contribution distribution over a population.

    Args:
        groups: The raw feature groups, in column order.
        contributions: The ``(n_rows, n_groups)`` matrix from
            :func:`group_contributions`.

    Returns:
        One record per group, in the same order.

    Raises:
        InterpretationError: If the matrix does not match the groups.
    """
    matrix = np.asarray(contributions, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(groups):
        raise InterpretationError(
            f"Contribution matrix has shape {matrix.shape} against {len(groups)} feature groups."
        )

    records: list[ContributionDispersion] = []
    for position, group in enumerate(groups):
        column = matrix[:, position]
        records.append(
            ContributionDispersion(
                feature=group.feature,
                kind=group.kind,
                n_rows=int(column.size),
                mean=float(column.mean()),
                std=float(column.std(ddof=0)),
                median=float(np.median(column)),
                q1=float(np.percentile(column, 25)),
                q3=float(np.percentile(column, 75)),
                minimum=float(column.min()),
                maximum=float(column.max()),
                mean_absolute_centered_contribution=float(np.mean(np.abs(column - column.mean()))),
            )
        )
    return tuple(records)


def rank_by_dispersion(
    dispersions: Sequence[ContributionDispersion],
    metric: str = RANKING_METRIC,
) -> tuple[dict[str, object], ...]:
    """Rank the raw features by the **pre-declared** dispersion metric.

    The metric is :data:`RANKING_METRIC`, fixed in this module before any value
    was computed, so the ordering is not the one that turned out to look best.
    Ties break on the feature's position in the frozen contract rather than
    arbitrarily, which keeps the output byte-reproducible.

    Raises:
        InterpretationError: If ``metric`` is not the declared one or a declared
            supporting metric.
    """
    allowed = (RANKING_METRIC, *SUPPORTING_METRICS)
    if metric not in allowed:
        raise InterpretationError(
            f"{metric!r} is not a declared dispersion metric; the declared ones are {allowed}."
        )

    order = sorted(
        range(len(dispersions)),
        key=lambda index: (-float(dispersions[index].as_dict()[metric]), index),
    )
    return tuple(
        {
            "rank": position + 1,
            "feature": dispersions[index].feature,
            "kind": dispersions[index].kind,
            "ranking_metric": metric,
            "ranking_value": float(dispersions[index].as_dict()[metric]),
            **{
                supporting: float(dispersions[index].as_dict()[supporting])
                for supporting in SUPPORTING_METRICS
                if supporting != metric
            },
        }
        for position, index in enumerate(order)
    )


# --- structural dependencies between raw features -----------------------------
#
# The contrasts above are exact statements about the linear function. They are
# not automatically statements about a change a customer could undergo, and the
# gap between the two is the subject of everything below.


@dataclass(frozen=True)
class StructuralRule:
    """A candidate equivalence between two encoded states, stated before checking.

    Declared as a *hypothesis*. Whether it holds is decided by
    :func:`check_structural_dependency` against a population, and a rule that
    fails is reported with its real violation count rather than quietly dropped
    or quietly asserted.
    """

    left_feature: str
    left_level: str
    right_feature: str
    right_level: str

    @property
    def relationship(self) -> str:
        """The rule as a readable equivalence."""
        return f"{self.left_feature}={self.left_level} <-> {self.right_feature}={self.right_level}"


#: The equivalences this project checks, taken from the product semantics the
#: Phase 2 data dictionary records — a sentinel category such as "No internet
#: service" exists precisely because the customer has no internet — and **not**
#: from anything observed in the model's coefficients or in the holdout.
#:
#: They are candidates. Nothing below assumes any of them is true.
STRUCTURAL_RULES: tuple[StructuralRule, ...] = (
    *(
        StructuralRule("InternetService", "No", service, "No internet service")
        for service in (
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        )
    ),
    StructuralRule("PhoneService", "No", "MultipleLines", "No phone service"),
)


@dataclass(frozen=True)
class StructuralCheck:
    """What one :class:`StructuralRule` actually did on one population."""

    rule: StructuralRule
    n_rows_checked: int
    n_left: int
    n_right: int
    n_both: int
    left_only: int
    right_only: int

    @property
    def violations(self) -> int:
        """Rows satisfying exactly one side of the equivalence."""
        return int(self.left_only + self.right_only)

    @property
    def deterministic(self) -> bool:
        """Whether the equivalence held on **every** row checked.

        Derived from :attr:`violations`, never set by hand. A rule with even one
        violation cannot report itself as deterministic.
        """
        return self.violations == 0

    def as_dict(self, population: str) -> dict[str, object]:
        """Return the record as a JSON-encodable mapping."""
        return {
            "relationship": self.rule.relationship,
            "features": [self.rule.left_feature, self.rule.right_feature],
            "left": {"feature": self.rule.left_feature, "level": self.rule.left_level},
            "right": {"feature": self.rule.right_feature, "level": self.rule.right_level},
            "observed_rule": (
                f"a row has {self.rule.left_feature}={self.rule.left_level!r} if and only if it "
                f"has {self.rule.right_feature}={self.rule.right_level!r}"
            ),
            "population": population,
            "n_rows_checked": int(self.n_rows_checked),
            "n_left": int(self.n_left),
            "n_right": int(self.n_right),
            "n_both": int(self.n_both),
            "left_only": int(self.left_only),
            "right_only": int(self.right_only),
            "violations": self.violations,
            "deterministic_in_training_pool": self.deterministic,
            "interpretation_consequence": (
                (
                    "the two encoded states are the same state in this population, so a "
                    "counterfactual that moves one of them while holding the other raw feature "
                    "fixed describes a row that never occurs. Contrasts touching either level are "
                    "flagged structurally_coupled and are algebraic contrasts only"
                )
                if self.deterministic
                else (
                    f"the equivalence does NOT hold: {self.violations} row(s) satisfy exactly one "
                    "side. It is reported as an observed tendency, not as a structural constraint, "
                    "and it is not used to flag any contrast"
                )
            ),
        }


def check_structural_dependency(
    frame: pd.DataFrame,
    rule: StructuralRule,
) -> StructuralCheck:
    """Test one equivalence against a population and count what it found.

    Args:
        frame: The raw-valued frame of the interpretation population — **the
            training pool** for this phase. This function loads nothing itself,
            which is what keeps the holdout unreachable from here.
        rule: The candidate equivalence.

    Returns:
        The counts, from which :attr:`StructuralCheck.deterministic` follows.

    Raises:
        InterpretationError: If either feature is absent from the frame.
    """
    missing = [
        feature
        for feature in (rule.left_feature, rule.right_feature)
        if feature not in frame.columns
    ]
    if missing:
        raise InterpretationError(
            f"Cannot check {rule.relationship!r}: column(s) {missing} are not in the frame."
        )

    left = (frame[rule.left_feature] == rule.left_level).to_numpy()
    right = (frame[rule.right_feature] == rule.right_level).to_numpy()

    return StructuralCheck(
        rule=rule,
        n_rows_checked=int(len(frame)),
        n_left=int(left.sum()),
        n_right=int(right.sum()),
        n_both=int((left & right).sum()),
        left_only=int((left & ~right).sum()),
        right_only=int((~left & right).sum()),
    )


def check_structural_dependencies(
    frame: pd.DataFrame,
    rules: Sequence[StructuralRule] = STRUCTURAL_RULES,
) -> tuple[StructuralCheck, ...]:
    """Test every candidate equivalence against one population."""
    checks = tuple(check_structural_dependency(frame, rule) for rule in rules)
    confirmed = sum(check.deterministic for check in checks)
    logger.info(
        "Structural dependencies: %d/%d deterministic over %d rows.",
        confirmed,
        len(checks),
        len(frame),
    )
    return checks


def coupled_levels_from(checks: Sequence[StructuralCheck]) -> dict[str, tuple[str, ...]]:
    """Return ``{feature: coupled levels}`` from the checks that actually held.

    Only a **deterministic** check contributes. A rule with violations describes
    a tendency, and a tendency is not a reason to qualify an exact contrast.
    """
    coupled: dict[str, set[str]] = {}
    for check in checks:
        if not check.deterministic:
            continue
        coupled.setdefault(check.rule.left_feature, set()).add(check.rule.left_level)
        coupled.setdefault(check.rule.right_feature, set()).add(check.rule.right_level)
    return {feature: tuple(sorted(levels)) for feature, levels in sorted(coupled.items())}


@dataclass(frozen=True)
class CoupledBlock:
    """Transformed columns that are the *same vector* on the population.

    Found empirically rather than declared, so the record cannot claim a
    redundancy the data does not show. Columns in one block are perfectly
    collinear by construction, which is what makes the split of their shared
    weight a property of the penalty rather than of the world.
    """

    columns: tuple[int, ...]
    members: tuple[tuple[str, str], ...]
    coefficients: tuple[float, ...]
    n_active_rows: int

    @property
    def aggregate_coefficient(self) -> float:
        """Total log-odds the shared underlying state contributes when it is active."""
        return float(sum(self.coefficients))

    @property
    def equal_split_share(self) -> float:
        """The aggregate divided evenly, which is what an L2 fit converges to."""
        return self.aggregate_coefficient / len(self.columns)

    @property
    def max_pairwise_coefficient_difference(self) -> float:
        """How unevenly the shared weight was split across the block."""
        return float(max(self.coefficients) - min(self.coefficients))

    @property
    def spans_multiple_raw_features(self) -> bool:
        """Whether the block crosses raw feature boundaries."""
        return len({feature for feature, _ in self.members}) > 1

    def as_dict(self) -> dict[str, object]:
        """Return the record as a JSON-encodable mapping."""
        return {
            "n_columns": len(self.columns),
            "transformed_columns": [int(column) for column in self.columns],
            "members": [{"feature": feature, "level": level} for feature, level in self.members],
            "raw_features": sorted({feature for feature, _ in self.members}),
            "spans_multiple_raw_features": self.spans_multiple_raw_features,
            "n_active_rows": int(self.n_active_rows),
            "coefficient_per_column": [float(value) for value in self.coefficients],
            "aggregate_coefficient": self.aggregate_coefficient,
            "equal_split_share": self.equal_split_share,
            "max_pairwise_coefficient_difference": self.max_pairwise_coefficient_difference,
            "note": (
                "these columns are the SAME vector on the interpretation population, so they are "
                "perfectly collinear. The per-column coefficients are one point in a family of "
                "parameterisations that all produce identical predictions; the L2 penalty selects "
                "the one that splits the shared weight evenly. The aggregate is what the model "
                "actually applies when the underlying state is active, and repeated equal "
                "coefficients are NOT evidence of one effect per column"
            ),
        }


def coupled_column_blocks(
    terms: LinearTerms,
    groups: Sequence[FeatureGroup],
    transformed: np.ndarray,
) -> tuple[CoupledBlock, ...]:
    """Find groups of transformed columns that are identical on the population.

    Deliberately small and auditable rather than a general constraint engine: it
    compares the columns to each other, exactly, and groups the ones that match.
    Only blocks spanning more than one raw feature are returned — two levels of
    the *same* feature can never be identical vectors under one-hot encoding, so
    a within-feature block would say nothing about cross-feature redundancy.

    Args:
        terms: The extracted linear terms.
        groups: The raw feature groups.
        transformed: The ``Z`` matrix of the interpretation population.

    Returns:
        The blocks, ordered by their first column.
    """
    matrix = np.asarray(transformed, dtype=float)
    label = {
        column: (group.feature, level)
        for group in groups
        for column, level in zip(group.columns, group.levels or ("",), strict=False)
    }

    remaining = list(range(matrix.shape[1]))
    blocks: list[CoupledBlock] = []
    while remaining:
        head = remaining.pop(0)
        matched = [
            column for column in remaining if np.array_equal(matrix[:, head], matrix[:, column])
        ]
        if not matched:
            continue
        for column in matched:
            remaining.remove(column)
        columns = (head, *matched)
        block = CoupledBlock(
            columns=columns,
            members=tuple(label[column] for column in columns),
            coefficients=tuple(float(terms.coefficients[column]) for column in columns),
            n_active_rows=int(matrix[:, head].sum()),
        )
        if block.spans_multiple_raw_features:
            blocks.append(block)

    logger.info(
        "Coupled column blocks spanning multiple raw features: %d (%s).",
        len(blocks),
        ", ".join(f"{len(block.columns)} columns" for block in blocks) or "none",
    )
    return tuple(blocks)
