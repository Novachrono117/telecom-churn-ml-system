"""Declarative registry of the Phase 6 candidate feature groups.

A group bundles one hypothesis with the transformer that expresses it and the
columns it contributes, so the ablation can vary *what the model sees* without
any pipeline being written twice.

Every derived column is routed to the **numeric** block. The counts and flags
here are ordinal or binary quantities whose hypotheses are monotone — "more
protections, less churn"; "automatic payment, less churn" — so a single
coefficient states exactly the claim being tested. One-hot encoding them would
spend several coefficients on a shape nobody hypothesised, and for the binary
flag it would add an exactly collinear dummy pair.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sklearn.base import TransformerMixin

from churn.features.deterministic import (
    AUTOMATIC_PAYMENT,
    CONTRACT_INTERACTIONS,
    CONTRACT_REFERENCE,
    HISTORICAL_AVERAGE_CHARGE,
    PROTECTIVE_COUNT,
    AutomaticPaymentFlag,
    ContractTenureInteraction,
    HistoricalAverageCharge,
    ProtectiveServiceCount,
)
from churn.features.intensity import CHARGE_INTENSITY, ChargeIntensityByTier

PROTECTIVE_SERVICES = "protective_services"
AUTOMATIC_PAYMENT_GROUP = "automatic_payment"
CONTRACT_TENURE = "contract_tenure"
HISTORICAL_CHARGE = "historical_average_charge"
CHARGE_INTENSITY_GROUP = "charge_intensity"


@dataclass(frozen=True)
class FeatureGroup:
    """One hypothesis, the transformer that expresses it, and what it adds."""

    name: str
    hypothesis: str
    factory: Callable[[], TransformerMixin]
    numeric: tuple[str, ...]
    stateful: bool
    notes: str = ""
    categorical: tuple[str, ...] = field(default_factory=tuple)

    def build(self) -> TransformerMixin:
        """Return a fresh, unfitted transformer for this group."""
        return self.factory()


FEATURE_GROUPS: dict[str, FeatureGroup] = {
    PROTECTIVE_SERVICES: FeatureGroup(
        name=PROTECTIVE_SERVICES,
        hypothesis=(
            "Churn falls monotonically as protective services accumulate, so the count "
            "carries signal the four independent dummies cannot express as one slope."
        ),
        factory=ProtectiveServiceCount,
        numeric=(PROTECTIVE_COUNT,),
        stateful=False,
        notes=(
            "Counts only explicit 'Yes'. 'No' and the sentinel 'No internet service' both "
            "count as zero. The four source columns are kept, so the count is a "
            "deterministic sum of indicators the baseline already receives one-hot: it "
            "adds no information, only a coarser parametrisation of information the "
            "model already had."
        ),
    ),
    AUTOMATIC_PAYMENT_GROUP: FeatureGroup(
        name=AUTOMATIC_PAYMENT_GROUP,
        hypothesis=(
            "The manual/automatic split is the axis of the payment-method signal, and one "
            "pooled coefficient estimates it more stably than four separate dummies."
        ),
        factory=AutomaticPaymentFlag,
        numeric=(AUTOMATIC_PAYMENT,),
        stateful=False,
        notes=(
            "Positive for 'Bank transfer (automatic)' and 'Credit card (automatic)'. "
            "PaymentMethod is kept, so the flag is a deterministic aggregation of two of "
            "the four categories the encoder already emits: it adds no information, only "
            "a coarser parametrisation of information the model already had."
        ),
    ),
    CONTRACT_TENURE: FeatureGroup(
        name=CONTRACT_TENURE,
        hypothesis=(
            "The tenure gradient differs by contract type, which the purely additive "
            "baseline cannot express with a single shared tenure slope."
        ),
        factory=ContractTenureInteraction,
        numeric=tuple(CONTRACT_INTERACTIONS.values()),
        stateful=False,
        notes=(
            f"'{CONTRACT_REFERENCE}' is the implicit reference. A third interaction would "
            "sum with the other two to exactly 'tenure', which is already in the matrix."
        ),
    ),
    HISTORICAL_CHARGE: FeatureGroup(
        name=HISTORICAL_CHARGE,
        hypothesis=(
            "Accumulated charge per month of tenure exposes the price history that the "
            "current MonthlyCharges hides, and the model would otherwise have to recover "
            "it from three separate columns."
        ),
        factory=HistoricalAverageCharge,
        numeric=(HISTORICAL_AVERAGE_CHARGE,),
        stateful=False,
        notes=(
            "A descriptive ratio, not an exact average contractual price. Undefined at "
            "tenure 0 and represented there as 0.0, coherent with the structural zero of "
            "TotalCharges; the model still receives tenure and can tell those rows apart."
        ),
    ),
    CHARGE_INTENSITY_GROUP: FeatureGroup(
        name=CHARGE_INTENSITY_GROUP,
        hypothesis=(
            "Paying above the going rate for one's own internet tier relates to churn, "
            "whereas the raw charge mostly encodes which tier was bought — the EDA showed "
            "the marginal charge gap reversing once the tier is held fixed."
        ),
        factory=ChargeIntensityByTier,
        numeric=(CHARGE_INTENSITY,),
        stateful=True,
        notes=(
            "The only stateful candidate. Tier medians are learned in fit — inside each "
            "training fold — and an unseen tier falls back to the training fold's global "
            "median. No median from the full-dataset EDA is reused."
        ),
    ),
}

#: Order in which the individual ablation experiments run (E1..E5).
ABLATION_ORDER: tuple[str, ...] = (
    PROTECTIVE_SERVICES,
    AUTOMATIC_PAYMENT_GROUP,
    CONTRACT_TENURE,
    HISTORICAL_CHARGE,
    CHARGE_INTENSITY_GROUP,
)


def resolve(names: Sequence[str]) -> tuple[FeatureGroup, ...]:
    """Return the groups for ``names``, in registry order.

    Args:
        names: Group names to select.

    Returns:
        The corresponding groups, ordered as in :data:`ABLATION_ORDER`.

    Raises:
        KeyError: If a name is not registered.
    """
    unknown = [name for name in names if name not in FEATURE_GROUPS]
    if unknown:
        raise KeyError(f"Unknown feature group(s): {unknown}. Known: {sorted(FEATURE_GROUPS)}.")
    selected = set(names)
    return tuple(FEATURE_GROUPS[name] for name in ABLATION_ORDER if name in selected)


def added_numeric(groups: Sequence[FeatureGroup]) -> tuple[str, ...]:
    """Return every numeric column contributed by ``groups``, in order."""
    return tuple(column for group in groups for column in group.numeric)


def added_categorical(groups: Sequence[FeatureGroup]) -> tuple[str, ...]:
    """Return every categorical column contributed by ``groups``, in order."""
    return tuple(column for group in groups for column in group.categorical)
