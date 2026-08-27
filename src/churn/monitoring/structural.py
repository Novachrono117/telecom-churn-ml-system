"""The seven product equivalences, watched rather than enforced.

Phase 10 verified these on the frozen training pool: a customer is
``OnlineSecurity = "No internet service"`` **exactly when** they are
``InternetService = "No"``, and so on for the other five internet add-ons, plus the
phone equivalent. They are facts about the product's encoding, not statistical
findings, which is what makes them usable as data-quality checks: an upstream
system that starts emitting ``InternetService = "No"`` alongside
``OnlineSecurity = "No"`` has changed its serialisation, and that is worth a signal
long before any distribution moves.

**Watched, not enforced.** Phase 11's request schema accepts these combinations,
and Phase 12 does not retroactively make them invalid. A record that violates an
equivalence is still scored, still returned, and still counted — the violation is
reported as a rate, and nothing about the serving contract changes. Tightening
input validity after an API has been published is a breaking change dressed up as
a bug fix.

The rules are restated here rather than imported from
:mod:`churn.modeling.interpretation`. That module reaches the analysis half of the
project and transitively imports the dataset loaders; monitoring runs *inside the
serving process*, where the import surface is deliberately narrow. The duplication
is seven four-string tuples and it cannot drift: ``tests/test_monitoring_structural.py``
asserts this declaration equals Phase 10's, rule for rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StructuralRule:
    """An equivalence between two encoded states, checked as a data-quality rule."""

    left_feature: str
    left_level: str
    right_feature: str
    right_level: str

    @property
    def name(self) -> str:
        """A stable identifier for the rule, usable as a JSON key."""
        return f"{self.left_feature}={self.left_level}<->{self.right_feature}={self.right_level}"

    @property
    def relationship(self) -> str:
        """The rule as a readable equivalence."""
        return f"{self.left_feature}={self.left_level} <-> {self.right_feature}={self.right_level}"


#: The seven equivalences Phase 10 verified on the training pool.
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


def rule_violations(frame: pd.DataFrame, rule: StructuralRule) -> pd.Series:
    """Return a boolean mask of rows that break ``rule``.

    An equivalence breaks in either direction: the left state without the right, or
    the right state without the left. Both are counted, because both mean the two
    columns stopped agreeing.

    Args:
        frame: Records carrying the two features, in the canonical feature contract.
        rule: The equivalence to check.

    Returns:
        A boolean Series aligned with ``frame``.
    """
    left = frame[rule.left_feature].astype("string") == rule.left_level
    right = frame[rule.right_feature].astype("string") == rule.right_level
    return (left & ~right) | (right & ~left)


def count_violations(frame: pd.DataFrame) -> dict[str, int]:
    """Return ``{rule name: violating rows}`` for every rule.

    Raises:
        KeyError: If a feature the rules need is absent. Monitoring observes records
            that already passed the frozen feature contract, so a missing column
            means the caller bypassed it.
    """
    return {rule.name: int(rule_violations(frame, rule).sum()) for rule in STRUCTURAL_RULES}


def count_records_with_any_violation(frame: pd.DataFrame) -> int:
    """Return how many *records* break at least one rule.

    Distinct from summing the per-rule counts: one malformed record typically
    breaks six internet rules at once, and reporting six would overstate how much of
    the traffic is affected.
    """
    if frame.empty:
        return 0
    any_violation = pd.Series(False, index=frame.index)
    for rule in STRUCTURAL_RULES:
        any_violation = any_violation | rule_violations(frame, rule)
    return int(any_violation.sum())


__all__ = [
    "STRUCTURAL_RULES",
    "StructuralRule",
    "count_records_with_any_violation",
    "count_violations",
    "rule_violations",
]
