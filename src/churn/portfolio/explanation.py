"""The exact local decomposition of one served prediction.

Phase 10 answered "how does this model work" over a population. This module answers
"why did it produce *that* number for *this* input", which is the question a person
looking at a screen actually has — and it answers it with the same identity, applied
to one row::

    logit(P) = intercept + sum over the 19 raw features of contribution_f
    P        = sigmoid(logit)

**This is not an approximation, and the distinction is the point.** SHAP, LIME and
permutation importance exist to probe models whose response surface is unknown. This
one is known in closed form, so a sampling estimator would add a dependency, a random
seed and an approximation error in exchange for a worse answer to a question already
answered exactly. Nothing here samples, and nothing here has a seed.

**This module never scores.** It receives the prepared feature row and the
probability the serving boundary already produced, and decomposes that row. The
alternative — recomputing the probability here — would create a second inference path
whose float could differ from the one the API returned, and ``/api/v1/explain`` would
then be explaining a number that ``/api/v1/predict`` never emitted. The contract is
the same one Phase 12 gave the monitoring collector: results are handed in, never
derived a second time.

**An explanation is published only if the arithmetic closes.** Every call re-derives
``intercept + sum(contributions)``, pushes it through the logistic function, and
compares the result with the probability that was served, at Phase 10's tolerance. A
decomposition that does not reproduce the model it claims to describe is not an
explanation of that model — so a mismatch raises. The prediction path is unaffected
by that failure: it is authoritative and has already answered.

**What a contribution is, and is not.** It is a term of a linear predictor: how many
log-odds this feature's value added to the sum, in *this* fitted parameterisation and
for *this* record. It is not a causal effect, not a counterfactual, and not a
recommendation. Because the one-hot encoding is redundant with the intercept, an
individual level's coefficient is a property of the fitted parameterisation rather
than a stable quantity — which is why the ordering produced here is explicitly local
to the request and is never called feature importance.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.modeling.interpretation import (
    RECONSTRUCTION_TOLERANCE,
    FeatureGroup,
    LinearTerms,
    group_contributions,
    logit_of,
    sigmoid,
    transform_features,
)
from churn.portfolio.coupling import CouplingBlock, active_blocks, partition

logger = logging.getLogger(__name__)

#: Named in the record so nobody has to infer which family of method this is.
EXPLANATION_METHOD = "EXACT_LOGISTIC_DECOMPOSITION"

#: Directions a contribution can carry. Deliberately about the *score*, not about
#: the customer: a term of a linear predictor moves a number, it does not cause a
#: person to leave.
INCREASES = "increases"
DECREASES = "decreases"
NEUTRAL = "neutral"

#: Kinds of raw feature, mirroring Phase 10's group kinds.
NUMERIC_KIND = "numeric"
CATEGORICAL_KIND = "categorical"

#: Shown wherever a contribution list is. Short on purpose — a caveat nobody reads
#: protects nobody.
CAUSAL_NOTE = (
    "Contributions explain the frozen model's calculation for this input. They are "
    "not causal effects or recommendations for changing customer behaviour."
)

#: The score is the model's raw output; Phase 9A froze calibration to NONE.
CALIBRATION_NOTE = (
    "Model-estimated churn probability; no post-hoc calibration was adopted, so the "
    "score is a ranking position rather than a calibrated frequency."
)


class ExplanationError(RuntimeError):
    """The decomposition does not reproduce the prediction it claims to explain."""


@dataclass(frozen=True)
class Contribution:
    """One raw feature's term in the linear predictor, for one record."""

    feature: str
    kind: str
    value_display: str
    active_level: str | None
    is_unseen_level: bool
    contribution_log_odds: float

    @property
    def direction(self) -> str:
        """Whether this term pushed the score up, down, or not at all."""
        if self.contribution_log_odds > 0.0:
            return INCREASES
        if self.contribution_log_odds < 0.0:
            return DECREASES
        return NEUTRAL

    @property
    def magnitude(self) -> float:
        """Absolute size, the quantity the local ordering is by."""
        return abs(self.contribution_log_odds)

    def as_record(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "kind": self.kind,
            "value_display": self.value_display,
            "active_level": self.active_level,
            "is_unseen_level": self.is_unseen_level,
            "contribution_log_odds": self.contribution_log_odds,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class GroupedContribution:
    """A coupled block presented as one row, valued at the exact sum of its members."""

    name: str
    label: str
    value_display: str
    members: tuple[str, ...]
    member_contributions: tuple[Contribution, ...]
    contribution_log_odds: float
    description: str

    @property
    def direction(self) -> str:
        if self.contribution_log_odds > 0.0:
            return INCREASES
        if self.contribution_log_odds < 0.0:
            return DECREASES
        return NEUTRAL

    @property
    def magnitude(self) -> float:
        return abs(self.contribution_log_odds)

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "value_display": self.value_display,
            "members": list(self.members),
            "n_members": len(self.members),
            "contribution_log_odds": self.contribution_log_odds,
            "direction": self.direction,
            "description": self.description,
        }


@dataclass(frozen=True)
class Reconstruction:
    """How exactly the decomposition reproduces the probability that was served."""

    logit_error: float
    probability_error: float
    tolerance: float

    @property
    def within_tolerance(self) -> bool:
        return max(self.logit_error, self.probability_error) <= self.tolerance

    def as_record(self) -> dict[str, object]:
        return {
            "logit_error": self.logit_error,
            "probability_error": self.probability_error,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
        }


@dataclass(frozen=True)
class LocalExplanation:
    """One served prediction, decomposed, with the identity that produced it.

    ``churn_probability`` and ``prediction`` are the values the serving boundary
    produced — copied, never recomputed — so this object can never disagree with the
    prediction endpoint. ``model_logit`` is the decomposition's own sum, which makes
    ``intercept + sum(contributions) == model_logit`` true by construction rather
    than to within a rounding error.
    """

    churn_probability: float
    prediction: int
    decision: str
    threshold: float
    comparison: str
    calibration_policy: str
    intercept: float
    model_logit: float
    threshold_logit: float
    contributions: tuple[Contribution, ...]
    grouped: tuple[GroupedContribution, ...]
    ungrouped: tuple[Contribution, ...]
    reconstruction: Reconstruction

    @property
    def margin_log_odds(self) -> float:
        """How far the record sits from the decision boundary, in log-odds."""
        return self.model_logit - self.threshold_logit

    def ranked(self) -> tuple[GroupedContribution | Contribution, ...]:
        """Blocks and ungrouped features together, ordered by local magnitude.

        The ordering is a property of **this request** — which parts of this input
        moved this score furthest. It is not global feature importance, and calling
        it that would be wrong for a redundantly parameterised one-hot model.
        """
        items: list[GroupedContribution | Contribution] = [*self.grouped, *self.ungrouped]
        return tuple(sorted(items, key=lambda item: (-item.magnitude, _label_of(item))))

    def as_record(self) -> dict[str, object]:
        return {
            "churn_probability": self.churn_probability,
            "prediction": self.prediction,
            "decision": self.decision,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "calibration_policy": self.calibration_policy,
            "explanation_method": EXPLANATION_METHOD,
            "intercept": self.intercept,
            "model_logit": self.model_logit,
            "threshold_logit": self.threshold_logit,
            "margin_log_odds": self.margin_log_odds,
            "contributions": [item.as_record() for item in self.contributions],
            "grouped_contributions": [item.as_record() for item in self.grouped],
            "ungrouped_features": [item.feature for item in self.ungrouped],
            "reconstruction": self.reconstruction.as_record(),
            "causal_note": CAUSAL_NOTE,
            "calibration_note": CALIBRATION_NOTE,
        }


def _label_of(item: GroupedContribution | Contribution) -> str:
    """Stable tiebreaker for the local ordering, so equal magnitudes never reorder."""
    return item.label if isinstance(item, GroupedContribution) else item.feature


def _display(value: object) -> str:
    """Render a raw contracted value the way the model received it.

    ``TotalCharges`` blank at zero tenure is the case worth naming: the frozen
    cleaner treats it as a structural zero, and showing an empty string would leave a
    reader guessing whether the field was lost in transit.
    """
    if value is None:
        return "(blank)"
    text = str(value).strip()
    if text == "":
        return "(blank)"
    return text


def _numeric_display(value: object) -> str:
    """Render a numeric feature, keeping an integral value integral."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "(blank)"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _display(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _contribution_for(
    group: FeatureGroup,
    value: object,
    amount: float,
) -> Contribution:
    """Build one raw feature's contribution row."""
    if group.kind == NUMERIC_KIND:
        return Contribution(
            feature=group.feature,
            kind=NUMERIC_KIND,
            value_display=_numeric_display(value),
            active_level=None,
            is_unseen_level=False,
            contribution_log_odds=amount,
        )

    text = _display(value)
    known = text in group.levels
    return Contribution(
        feature=group.feature,
        kind=CATEGORICAL_KIND,
        value_display=text,
        active_level=text if known else None,
        # An unseen level is encoded as an all-zero block by handle_unknown="ignore",
        # so it contributes exactly nothing. Saying so is more useful than a silent 0.
        is_unseen_level=not known,
        contribution_log_odds=amount,
    )


def explain_prepared_row(
    pipeline: Pipeline,
    terms: LinearTerms,
    groups: Sequence[FeatureGroup],
    features: pd.DataFrame,
    served_probability: float,
    prediction: int,
    decision: str,
    threshold: float,
    comparison: str,
    calibration_policy: str,
    tolerance: float = RECONSTRUCTION_TOLERANCE,
) -> LocalExplanation:
    """Decompose one already-scored row into its exact per-feature terms.

    Args:
        pipeline: The frozen, verified pipeline. Used for ``.transform`` only — the
            representation the coefficients were learned for must not be refitted.
        terms: Learned parameters read out of that pipeline by Phase 10.
        groups: The 19 raw-feature groups over the 46 transformed columns.
        features: **One row**, already through ``prepare_features``: the exact matrix
            the canonical scoring path handed the model.
        served_probability: The probability that path returned. Copied into the
            answer, never recomputed.
        prediction: The decision that path produced under the frozen rule.
        decision: Its human label.
        threshold: The frozen decision threshold.
        comparison: The frozen comparison operator.
        calibration_policy: The frozen calibration policy.
        tolerance: Reconstruction tolerance. Defaults to Phase 10's.

    Returns:
        A :class:`LocalExplanation` whose contributions sum, with the intercept, to
        ``model_logit``.

    Raises:
        ValueError: If ``features`` does not hold exactly one row.
        ExplanationError: If the decomposition does not reproduce the served
            probability within ``tolerance``. Raised rather than reported, because an
            explanation that does not match the prediction is not an explanation of
            it.
    """
    if len(features) != 1:
        raise ValueError(
            f"A local explanation describes one record; {len(features)} row(s) were given."
        )

    transformed = transform_features(pipeline, features)
    amounts = group_contributions(terms, groups, transformed)[0]
    if amounts.size != len(groups):
        raise ExplanationError(
            f"The decomposition produced {amounts.size} term(s) for {len(groups)} raw "
            "feature(s); the group layout is not the one the coefficients assume."
        )

    row: Mapping[str, object] = features.iloc[0].to_dict()
    contributions = tuple(
        _contribution_for(group, row.get(group.feature), float(amount))
        for group, amount in zip(groups, amounts, strict=True)
    )

    intercept = float(terms.intercept)
    model_logit = intercept + float(np.sum(amounts))
    derived_probability = float(sigmoid(np.asarray([model_logit], dtype=float))[0])

    reconstruction = Reconstruction(
        logit_error=abs(model_logit - logit_of(served_probability)),
        probability_error=abs(derived_probability - float(served_probability)),
        tolerance=float(tolerance),
    )
    if not reconstruction.within_tolerance:
        raise ExplanationError(
            "The local decomposition does not reproduce the probability that was "
            f"served: logit error {reconstruction.logit_error:.3e}, probability error "
            f"{reconstruction.probability_error:.3e}, tolerance "
            f"{reconstruction.tolerance:.3e}. No explanation is returned; the "
            "prediction itself is unaffected and remains authoritative."
        )

    by_feature = {item.feature: item for item in contributions}
    blocks = active_blocks(row)
    grouped = tuple(_grouped_for(block, by_feature) for block in blocks)
    ordered = [item.feature for item in contributions]
    ungrouped = tuple(by_feature[feature] for feature in partition(ordered, blocks))

    return LocalExplanation(
        churn_probability=float(served_probability),
        prediction=int(prediction),
        decision=decision,
        threshold=float(threshold),
        comparison=comparison,
        calibration_policy=calibration_policy,
        intercept=intercept,
        model_logit=model_logit,
        threshold_logit=logit_of(threshold),
        contributions=contributions,
        grouped=grouped,
        ungrouped=ungrouped,
        reconstruction=reconstruction,
    )


def _grouped_for(
    block: CouplingBlock,
    by_feature: Mapping[str, Contribution],
) -> GroupedContribution:
    """Sum a block's members into one presented row. Exact, never reweighted."""
    members = tuple(by_feature[feature] for feature in block.members)
    return GroupedContribution(
        name=block.name,
        label=block.label,
        value_display=f"{block.trigger_feature} = {block.trigger_level}",
        members=block.members,
        member_contributions=members,
        contribution_log_odds=float(sum(item.contribution_log_odds for item in members)),
        description=block.description,
    )


__all__ = [
    "CALIBRATION_NOTE",
    "CATEGORICAL_KIND",
    "CAUSAL_NOTE",
    "DECREASES",
    "EXPLANATION_METHOD",
    "INCREASES",
    "NEUTRAL",
    "NUMERIC_KIND",
    "Contribution",
    "ExplanationError",
    "GroupedContribution",
    "LocalExplanation",
    "Reconstruction",
    "explain_prepared_row",
]
