"""Which raw features move together, so the UI does not present one fact seven times.

Phase 10 established that six internet add-ons carry the level
``"No internet service"`` **exactly when** ``InternetService`` is ``"No"``, and that
``MultipleLines`` carries ``"No phone service"`` exactly when ``PhoneService`` is
``"No"``. Phase 12 turned those equivalences into data-quality rules. Phase 13 uses
the same declaration for a third purpose, and it is worth being precise about which.

**The problem this solves is a reading error, not a computation error.** A customer
without internet produces seven separate contributions — ``InternetService = No``
plus six add-ons at ``No internet service`` — and every one of them is a correct term
of the linear predictor. Listed as seven rows, they look like seven independent
reasons the score moved. They are one fact about the customer, recorded in seven
columns because that is how the product encodes it. A reader who counts them as
seven pieces of evidence has been misled by the presentation, not by the model.

So the block is presented as one row whose value is the **exact sum** of its members'
contributions. Nothing is dropped, nothing is reweighted, and the grand total still
reconstructs the logit — grouping is a change of layout, never of arithmetic.

**A block is grouped only when the equivalence actually holds.** The serving contract
accepts a record that breaks it (``InternetService = "No"`` with
``OnlineSecurity = "No"``), Phase 12 counts that as a structural violation, and here
it means the seven columns are *not* saying one thing. Such a record gets the
individual contributions, ungrouped, because presenting them as a block would assert
a relationship the input just contradicted.

The blocks are derived from :data:`churn.monitoring.structural.STRUCTURAL_RULES`
rather than restated, so a rule added there appears here without a second edit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from churn.monitoring.structural import STRUCTURAL_RULES

logger = logging.getLogger(__name__)

#: Human labels for the blocks the equivalences imply. Keyed by
#: ``(trigger_feature, trigger_level)`` so a new rule with a new trigger is a missing
#: key rather than a silently unlabelled block.
_BLOCK_LABELS: dict[tuple[str, str], tuple[str, str]] = {
    ("InternetService", "No"): (
        "no_internet_service",
        "No internet service",
    ),
    ("PhoneService", "No"): (
        "no_phone_service",
        "No phone service",
    ),
}


@dataclass(frozen=True)
class CouplingBlock:
    """A trigger feature and the features that are pinned by it.

    ``members`` always includes the trigger itself: the block's value is the sum over
    every raw feature the situation determines, and leaving the trigger out would
    make the grouped total no longer partition the logit.
    """

    name: str
    label: str
    trigger_feature: str
    trigger_level: str
    dependents: tuple[str, ...]
    dependent_level: str

    @property
    def members(self) -> tuple[str, ...]:
        """Every raw feature this block accounts for, trigger first."""
        return (self.trigger_feature, *self.dependents)

    @property
    def description(self) -> str:
        """Why these features are one row and not several."""
        return (
            f"{self.trigger_feature} = {self.trigger_level} pins "
            f"{len(self.dependents)} related feature(s) to "
            f"{self.dependent_level!r}. They are one fact about the customer, "
            "encoded in several columns, so their contributions are summed rather "
            "than listed as independent factors."
        )


def _derive_blocks() -> tuple[CouplingBlock, ...]:
    """Build the blocks from the frozen equivalences, in declaration order."""
    order: list[tuple[str, str]] = []
    dependents: dict[tuple[str, str], list[str]] = {}
    levels: dict[tuple[str, str], set[str]] = {}

    for rule in STRUCTURAL_RULES:
        key = (rule.left_feature, rule.left_level)
        if key not in dependents:
            order.append(key)
            dependents[key] = []
            levels[key] = set()
        dependents[key].append(rule.right_feature)
        levels[key].add(rule.right_level)

    blocks: list[CouplingBlock] = []
    for key in order:
        trigger_feature, trigger_level = key
        if len(levels[key]) != 1:
            # Two different coupled levels under one trigger would make a single
            # `dependent_level` a lie. Better to refuse the block than to pick one.
            raise ValueError(
                f"{trigger_feature}={trigger_level} pins more than one level "
                f"({sorted(levels[key])}); the block cannot be described by one."
            )
        name, label = _BLOCK_LABELS[key]
        blocks.append(
            CouplingBlock(
                name=name,
                label=label,
                trigger_feature=trigger_feature,
                trigger_level=trigger_level,
                dependents=tuple(dependents[key]),
                dependent_level=next(iter(levels[key])),
            )
        )
    return tuple(blocks)


#: The coupled blocks, derived once from the Phase 10 equivalences.
COUPLING_BLOCKS: tuple[CouplingBlock, ...] = _derive_blocks()


def coupled_features() -> frozenset[str]:
    """Return every raw feature that belongs to some block."""
    return frozenset(feature for block in COUPLING_BLOCKS for feature in block.members)


def is_block_consistent(block: CouplingBlock, values: Mapping[str, object]) -> bool:
    """Return whether ``values`` satisfies the equivalence this block describes.

    Both directions are required, exactly as Phase 12 defines a violation: the
    trigger at its level with every dependent pinned, or the trigger away from its
    level with no dependent pinned. A record in between is inconsistent, and an
    inconsistent record is not grouped.
    """
    triggered = str(values.get(block.trigger_feature)) == block.trigger_level
    pinned = [str(values.get(feature)) == block.dependent_level for feature in block.dependents]
    return all(pinned) if triggered else not any(pinned)


def active_blocks(values: Mapping[str, object]) -> tuple[CouplingBlock, ...]:
    """Return the blocks that are both triggered and internally consistent.

    Args:
        values: One record's raw feature values, as the frozen contract holds them.

    Returns:
        The blocks whose trigger is at its level and whose dependents all agree. A
        triggered-but-inconsistent block is deliberately absent: the record broke the
        equivalence, so its columns are not one fact and must not be summed into one.
    """
    return tuple(
        block
        for block in COUPLING_BLOCKS
        if str(values.get(block.trigger_feature)) == block.trigger_level
        and is_block_consistent(block, values)
    )


def block_records() -> list[dict[str, object]]:
    """Return the blocks as plain data, for the machine-readable record."""
    return [
        {
            "name": block.name,
            "label": block.label,
            "trigger": f"{block.trigger_feature}={block.trigger_level}",
            "dependent_level": block.dependent_level,
            "members": list(block.members),
            "n_members": len(block.members),
        }
        for block in COUPLING_BLOCKS
    ]


def partition(features: Sequence[str], blocks: Sequence[CouplingBlock]) -> tuple[str, ...]:
    """Return the features that no active block accounts for, in the given order.

    Together with the blocks' members this is a partition of ``features``, which is
    what keeps the grouped presentation summing to the same logit as the flat one.
    """
    grouped = {member for block in blocks for member in block.members}
    return tuple(feature for feature in features if feature not in grouped)


__all__ = [
    "COUPLING_BLOCKS",
    "CouplingBlock",
    "active_blocks",
    "block_records",
    "coupled_features",
    "is_block_consistent",
    "partition",
]
