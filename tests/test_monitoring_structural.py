"""The seven equivalences: monitoring's copy must equal Phase 10's, rule for rule.

The rules are declared twice — once in :mod:`churn.modeling.interpretation`, which
verified them, and once in :mod:`churn.monitoring.structural`, which watches them.
The duplication is deliberate: monitoring runs inside the serving process, where the
import surface is kept narrow, and the analysis module transitively imports the
dataset loaders.

Duplication is only acceptable when it cannot drift. This file is what makes it so.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.modeling.interpretation import STRUCTURAL_RULES as PHASE_10_RULES
from churn.monitoring.structural import (
    STRUCTURAL_RULES,
    count_records_with_any_violation,
    count_violations,
    rule_violations,
)

CLEAN = {
    "InternetService": "No",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "PhoneService": "Yes",
    "MultipleLines": "No",
}


def test_the_monitoring_rules_are_the_phase_10_rules() -> None:
    """Same rules, same order, same levels. A divergence fails here, loudly."""
    assert len(STRUCTURAL_RULES) == len(PHASE_10_RULES) == 7

    mine = [
        (rule.left_feature, rule.left_level, rule.right_feature, rule.right_level)
        for rule in STRUCTURAL_RULES
    ]
    theirs = [
        (rule.left_feature, rule.left_level, rule.right_feature, rule.right_level)
        for rule in PHASE_10_RULES
    ]

    assert mine == theirs


def test_the_relationship_strings_agree() -> None:
    assert [rule.relationship for rule in STRUCTURAL_RULES] == [
        rule.relationship for rule in PHASE_10_RULES
    ]


def test_a_consistent_record_violates_nothing() -> None:
    frame = pd.DataFrame([CLEAN])

    assert count_records_with_any_violation(frame) == 0
    assert all(count == 0 for count in count_violations(frame).values())


def test_an_equivalence_breaks_in_both_directions() -> None:
    """Left without right, and right without left, are both violations."""
    left_only = pd.DataFrame([{**CLEAN, "OnlineSecurity": "No"}])
    right_only = pd.DataFrame([{**CLEAN, "InternetService": "DSL"}])

    assert count_records_with_any_violation(left_only) == 1
    assert count_records_with_any_violation(right_only) == 1


def test_one_malformed_record_is_counted_once_not_six_times() -> None:
    """A single bad record breaks six internet rules; the record count says one."""
    frame = pd.DataFrame([{**CLEAN, "InternetService": "DSL"}])

    per_rule = count_violations(frame)

    assert sum(per_rule.values()) == 6
    assert count_records_with_any_violation(frame) == 1


def test_the_phone_rule_is_independent_of_the_internet_rules() -> None:
    frame = pd.DataFrame([{**CLEAN, "PhoneService": "No", "MultipleLines": "No"}])

    per_rule = count_violations(frame)
    phone_rule = "PhoneService=No<->MultipleLines=No phone service"

    assert per_rule[phone_rule] == 1
    assert sum(count for name, count in per_rule.items() if name != phone_rule) == 0


def test_rule_violations_returns_an_aligned_mask() -> None:
    frame = pd.DataFrame([CLEAN, {**CLEAN, "InternetService": "DSL"}, CLEAN])

    mask = rule_violations(frame, STRUCTURAL_RULES[0])

    assert list(mask) == [False, True, False]
    assert list(mask.index) == list(frame.index)


def test_an_empty_frame_has_no_violations() -> None:
    assert count_records_with_any_violation(pd.DataFrame(columns=list(CLEAN))) == 0


def test_a_missing_feature_raises_rather_than_reporting_zero() -> None:
    """Silently reporting "no violations" for a column that is not there would lie."""
    with pytest.raises(KeyError):
        count_violations(pd.DataFrame([{"InternetService": "No"}]))
