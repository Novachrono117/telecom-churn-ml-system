"""The reference, compared with itself, must not drift.

This is the strongest single test in the phase. It rebuilds the reference
population, pushes it through the *production* path — the collector, not the
builder — and asserts that every distance comes out exactly zero.

It is strong because the two sides are computed by different code. The reference
histogram was produced by :mod:`churn.monitoring.build` using
:func:`numpy.searchsorted` over a whole column; the window histogram is produced by
:mod:`churn.monitoring.collector` accumulating one batch at a time into a running
counter. If the two disagreed about an edge, an inclusivity convention, a dtype or
the treatment of ``TotalCharges``, this test would show it — and a monitoring system
whose reference does not match itself would report drift that is not there, which is
worse than reporting none at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from churn.modeling.freeze import apply_decision_rule, load_pipeline, positive_probability
from churn.monitoring.collector import MonitoringCollector
from churn.monitoring.reference import load_reference_profile, profile_digest
from churn.monitoring.service import compare
from churn.monitoring.settings import STATUS_OK
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features
from churn.preprocessing.splitting import load_training_pool


@pytest.fixture(scope="module")
def self_report():
    """Score the reference population through the production path and compare."""
    reference = load_reference_profile()
    pipeline = load_pipeline()
    features = prepare_features(load_training_pool().loc[:, list(FEATURE_COLUMNS)])
    scores = np.asarray(positive_probability(pipeline, features), dtype=float)
    decisions = apply_decision_rule(scores, reference.score.threshold)

    collector = MonitoringCollector(reference)
    collector.record_request(len(features))
    collector.observe(features, scores, decisions)
    return compare(collector.snapshot(), reference, reference_digest=profile_digest(reference))


def test_the_reference_does_not_drift_from_itself(self_report) -> None:
    """Every distance is exactly zero, not merely small."""
    assert self_report.n_records == 5634
    assert self_report.status == STATUS_OK

    for feature, entry in self_report.numeric.items():
        assert entry["psi"]["value"] == 0.0, f"numeric PSI moved for {feature}"
        assert entry["out_of_reference_range_rate"]["value"] == 0.0, feature
        assert entry["mean_shift_in_reference_sd"] == pytest.approx(0.0, abs=1e-12), feature

    for feature, entry in self_report.categorical.items():
        assert entry["tvd"]["value"] == 0.0, f"categorical TVD moved for {feature}"
        assert entry["unseen_rate"]["value"] == 0.0, feature
        assert entry["n_distinct_unseen_observed"] == 0, feature

    assert self_report.score["psi"]["value"] == 0.0
    assert self_report.score["predicted_positive_rate_delta"]["value"] == 0.0
    assert self_report.score["mean_delta"] == pytest.approx(0.0, abs=1e-12)

    assert self_report.structural["records_with_any_violation"] == 0
    assert self_report.structural["violation_rate"] == 0.0


def test_the_collector_histogram_equals_the_reference_histogram(self_report) -> None:
    """Bin for bin, produced by two different implementations."""
    reference = load_reference_profile()

    for feature, entry in self_report.numeric.items():
        assert list(entry["bin_counts"]) == list(reference.numeric[feature].bin_counts), feature

    assert list(self_report.score["bin_counts"]) == list(reference.score.bin_counts)


def test_the_collector_level_counts_equal_the_reference_counts(self_report) -> None:
    reference = load_reference_profile()

    for feature, entry in self_report.categorical.items():
        assert dict(entry["count_by_level"]) == dict(
            reference.categorical[feature].count_by_level
        ), feature


def test_the_predicted_positive_rate_matches_the_reference(self_report) -> None:
    reference = load_reference_profile()

    assert self_report.score["predicted_positive_rate"] == (reference.score.predicted_positive_rate)


def test_the_self_check_reports_no_performance_verdict(self_report) -> None:
    """Even a perfect self-comparison says nothing about performance."""
    record = self_report.as_record()

    assert record["performance_degradation"]["evaluated"] is False
    assert self_report.section_status["performance_degradation"] == "INSUFFICIENT_DATA"
