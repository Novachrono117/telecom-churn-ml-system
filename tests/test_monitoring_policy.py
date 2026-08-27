"""The operational policy: what it may configure, and what it must never reach.

The separation between "when should a human look" and "what does the model predict"
is the point of having a second configuration file at all, so it is asserted rather
than described — including the negative half: the policy has no field through which
a threshold, a calibration decision or a feature could travel.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from churn.config import PROJECT_ROOT, get_config
from churn.modeling.freeze_results import load_decision_policy
from churn.monitoring.settings import (
    DEFAULT_MONITORING_CONFIG_PATH,
    SEVERITY,
    STATUS_CRITICAL,
    STATUS_INSUFFICIENT_DATA,
    STATUS_OK,
    STATUS_WARNING,
    MonitoringPolicy,
    get_monitoring_policy,
    load_monitoring_policy,
    worst,
)

FROZEN_THRESHOLD = 0.3272694566222328


@pytest.fixture(scope="module")
def policy() -> MonitoringPolicy:
    return load_monitoring_policy()


def test_the_policy_is_not_in_the_frozen_configuration() -> None:
    """configs/base.toml belongs to the Phase 9C provenance and must stay untouched."""
    assert DEFAULT_MONITORING_CONFIG_PATH.name == "monitoring.toml"
    assert DEFAULT_MONITORING_CONFIG_PATH.is_file()

    frozen = (PROJECT_ROOT / "configs" / "base.toml").read_text(encoding="utf-8")
    for forbidden in ("psi", "tvd", "drift", "monitoring", "window"):
        assert forbidden not in frozen.lower(), forbidden


def test_the_monitoring_policy_has_no_model_parameter(policy: MonitoringPolicy) -> None:
    """There is no field to set, so there is nothing to forget to guard."""
    fields = set(MonitoringPolicy.model_fields)

    assert not fields & {
        "threshold",
        "final_threshold",
        "calibration",
        "calibration_policy",
        "positive_class",
        "features",
        "feature_columns",
        "model_path",
    }
    assert fields == {
        "minimum_window_size",
        "max_distinct_unseen_tracked_per_feature",
        "numeric_psi",
        "numeric_out_of_range",
        "categorical_tvd",
        "categorical_unseen_rate",
        "score_psi",
        "score_positive_rate_delta",
        "structural_violation_rate",
    }


def test_the_monitoring_file_declares_no_model_parameter() -> None:
    """Checked on the file too, not only on the model that parses it."""
    raw = tomllib.loads(DEFAULT_MONITORING_CONFIG_PATH.read_text(encoding="utf-8"))
    flattened = {key for section in raw.values() for key in section}

    assert not flattened & {"threshold", "calibration", "positive_class", "comparison"}


def test_the_monitoring_policy_cannot_move_the_frozen_threshold(policy: MonitoringPolicy) -> None:
    """Loading it changes nothing about the decision policy."""
    before = load_decision_policy()

    get_monitoring_policy()
    after = load_decision_policy()

    assert after.threshold.final_threshold == before.threshold.final_threshold
    assert after.threshold.final_threshold == FROZEN_THRESHOLD
    assert after.decision_rule.comparison == ">="


def test_the_monitoring_policy_cannot_change_calibration() -> None:
    before = load_decision_policy().calibration.calibration_policy

    get_monitoring_policy()

    assert load_decision_policy().calibration.calibration_policy == before == "NONE"


def test_the_monitoring_policy_does_not_touch_the_project_configuration() -> None:
    """The seed, the target and the split policy are unaffected."""
    config = get_config()

    get_monitoring_policy()

    assert get_config().seed == config.seed == 42
    assert get_config().target.column == "Churn"


def test_the_policy_is_labelled_as_heuristic(policy: MonitoringPolicy) -> None:
    """The record must not let a cutoff be mistaken for a significance level."""
    record = policy.as_record()

    assert record["classification"] == "OPERATIONAL_MONITORING_POLICY"
    assert record["is_statistical_significance"] is False
    assert record["estimated_from_data"] is False
    assert "not test statistics" in record["note"]


def test_the_policy_is_immutable(policy: MonitoringPolicy) -> None:
    """It is read while comparing a window and is never rebuilt per record."""
    with pytest.raises(ValidationError):
        policy.minimum_window_size = 1


def test_the_cutoffs_are_ordered(policy: MonitoringPolicy) -> None:
    """A critical cutoff below its warning would make CRITICAL unreachable."""
    for name in (
        "numeric_psi",
        "numeric_out_of_range",
        "categorical_tvd",
        "categorical_unseen_rate",
        "score_psi",
        "score_positive_rate_delta",
        "structural_violation_rate",
    ):
        cutoffs = getattr(policy, name)
        assert cutoffs.warning <= cutoffs.critical, name


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, STATUS_OK), (0.05, STATUS_OK), (0.10, STATUS_WARNING), (0.30, STATUS_CRITICAL)],
)
def test_a_cutoff_maps_a_value_to_a_status(
    policy: MonitoringPolicy, value: float, expected: str
) -> None:
    assert policy.numeric_psi.status(value) == expected


def test_the_minimum_window_size_is_documented_and_positive(policy: MonitoringPolicy) -> None:
    assert policy.minimum_window_size == 100
    assert "minimum_window_size" in DEFAULT_MONITORING_CONFIG_PATH.read_text(encoding="utf-8")


def test_insufficient_data_is_not_a_rung_on_the_severity_ladder() -> None:
    """It is the absence of a verdict, not a mild one."""
    assert STATUS_INSUFFICIENT_DATA not in SEVERITY
    assert set(SEVERITY) == {STATUS_OK, STATUS_WARNING, STATUS_CRITICAL}


def test_worst_ignores_the_absence_of_a_verdict() -> None:
    assert worst([STATUS_OK, STATUS_WARNING]) == STATUS_WARNING
    assert worst([STATUS_WARNING, STATUS_CRITICAL]) == STATUS_CRITICAL
    assert worst([STATUS_INSUFFICIENT_DATA]) == STATUS_OK
    assert worst([]) == STATUS_OK


def test_a_missing_section_fails_loudly(tmp_path: Path) -> None:
    """A silently defaulted cutoff is a cutoff nobody chose."""
    broken = tmp_path / "monitoring.toml"
    broken.write_text("[window]\nminimum_window_size = 10\n", encoding="utf-8")

    with pytest.raises(KeyError):
        load_monitoring_policy(broken)
