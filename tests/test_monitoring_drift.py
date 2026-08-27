"""Synthetic drift: does each metric move, in the right direction, for the right reason?

Every window here is invented. None is drawn from the holdout — demonstrating drift
with the held-out partition would reopen it for a purpose it was never reserved for,
and a synthetic shift proves the metric works far more directly anyway: the shift is
known, so the expected direction is known.

The windows are built from the reference population's *own* level vocabulary so that
each test isolates one change. A window that shifted ``tenure`` and introduced a new
``Contract`` at the same time would move two metrics and prove neither.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.modeling.freeze import apply_decision_rule, load_pipeline, positive_probability
from churn.monitoring.collector import MonitoringCollector
from churn.monitoring.distributions import bin_counts, quantile_bin_edges
from churn.monitoring.drift import (
    EPSILON,
    population_stability_index,
    total_variation_distance,
)
from churn.monitoring.reference import load_reference_profile
from churn.monitoring.service import compare
from churn.monitoring.settings import (
    STATUS_CRITICAL,
    STATUS_INSUFFICIENT_DATA,
    STATUS_WARNING,
    get_monitoring_policy,
)
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features

#: A synthetic record built from levels the reference knows. Invented, not sampled.
BASE_RECORD: dict[str, object] = {
    "tenure": 24,
    "MonthlyCharges": 70.0,
    "TotalCharges": "1680.00",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

WINDOW_SIZE = 400


@pytest.fixture(scope="module")
def reference():
    return load_reference_profile()


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


def make_window(records: list[dict[str, object]]) -> pd.DataFrame:
    """Turn invented records into a canonical feature matrix."""
    return prepare_features(pd.DataFrame(records).loc[:, list(FEATURE_COLUMNS)])


def observe(reference, pipeline, records: list[dict[str, object]]):
    """Score a synthetic window through the production path and compare it."""
    features = make_window(records)
    scores = np.asarray(positive_probability(pipeline, features), dtype=float)
    decisions = apply_decision_rule(scores, reference.score.threshold)
    collector = MonitoringCollector(reference)
    collector.record_request(len(features))
    collector.observe(features, scores, decisions)
    return compare(collector.snapshot(), reference)


def spread(values: list[object], size: int) -> list[object]:
    """Repeat a small vocabulary across a window without randomness."""
    return [values[index % len(values)] for index in range(size)]


# --------------------------------------------------------------------------- #
# The metrics themselves.
# --------------------------------------------------------------------------- #


def test_psi_is_zero_for_identical_histograms() -> None:
    counts = [10, 20, 30, 40]

    assert population_stability_index(counts, counts) == 0.0


def test_psi_is_symmetric() -> None:
    left, right = [10, 20, 30, 40], [40, 30, 20, 10]

    assert population_stability_index(left, right) == pytest.approx(
        population_stability_index(right, left)
    )


def test_psi_grows_as_a_histogram_moves_away() -> None:
    reference_counts = [25, 25, 25, 25]
    near = population_stability_index([30, 25, 25, 20], reference_counts)
    far = population_stability_index([90, 5, 3, 2], reference_counts)

    assert 0.0 < near < far


def test_psi_needs_the_same_bins_on_both_sides() -> None:
    """Comparing different partitions would measure the binning, not the data."""
    with pytest.raises(ValueError, match="same bins"):
        population_stability_index([1, 2, 3], [1, 2, 3, 4])


def test_psi_handles_an_empty_bin_with_the_documented_epsilon() -> None:
    """Empty bins are normal; the formula is undefined at zero, so a floor is applied."""
    value = population_stability_index([0, 50, 50], [33, 33, 34])

    assert np.isfinite(value)
    assert value > 1.0  # an emptied bin dominates, as documented
    assert EPSILON == 1e-6


def test_tvd_is_zero_for_identical_distributions() -> None:
    distribution = {"a": 0.5, "b": 0.5}

    assert total_variation_distance(distribution, distribution) == 0.0


def test_tvd_is_one_for_disjoint_support() -> None:
    assert total_variation_distance({"a": 1.0}, {"b": 1.0}) == 1.0


def test_tvd_stays_inside_the_unit_interval() -> None:
    assert 0.0 <= total_variation_distance({"a": 0.7, "b": 0.3}, {"a": 0.2, "c": 0.8}) <= 1.0


def test_bins_come_from_the_reference_only(reference) -> None:
    """A window never recomputes edges; it counts into the reference's."""
    edges = tuple(reference.numeric["tenure"].bin_edges)
    window_values = [1.0, 1.0, 1.0, 2.0]

    counts = bin_counts(window_values, edges)
    recomputed = quantile_bin_edges(window_values, n_bins=10)

    assert len(counts) == len(edges) + 1
    assert sum(counts) == len(window_values)
    assert recomputed != edges  # the window's own quantiles differ, and are not used


# --------------------------------------------------------------------------- #
# Synthetic windows.
# --------------------------------------------------------------------------- #


def test_an_unshifted_window_reports_low_drift(reference, pipeline) -> None:
    """The control. Records drawn from the reference vocabulary should look normal."""
    records = [
        {
            **BASE_RECORD,
            "tenure": value,
            "MonthlyCharges": charge,
            "TotalCharges": f"{value * charge:.2f}",
            "Contract": contract,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": security,
            "DeviceProtection": security,
            "TechSupport": security,
            "StreamingTV": security,
            "StreamingMovies": security,
        }
        for value, charge, contract, internet, security in zip(
            spread([1, 8, 20, 33, 47, 60, 72], WINDOW_SIZE),
            spread([19.5, 35.0, 55.0, 70.0, 85.0, 100.0, 115.0], WINDOW_SIZE),
            spread(["Month-to-month", "One year", "Two year"], WINDOW_SIZE),
            spread(["DSL", "Fiber optic"], WINDOW_SIZE),
            spread(["No", "Yes"], WINDOW_SIZE),
            strict=False,
        )
    ]

    report = observe(reference, pipeline, records)

    assert report.n_records == WINDOW_SIZE
    assert report.structural["records_with_any_violation"] == 0
    assert max(entry["unseen_rate"]["value"] for entry in report.categorical.values()) == 0.0


def test_a_shifted_numeric_feature_raises_psi(reference, pipeline) -> None:
    """Every customer in the window has a tenure the reference rarely saw."""
    long_tenure = [{**BASE_RECORD, "tenure": 71, "TotalCharges": "5000.00"}] * WINDOW_SIZE
    mixed = [
        {**BASE_RECORD, "tenure": value, "TotalCharges": f"{value * 70.0:.2f}"}
        for value in spread([1, 12, 24, 36, 48, 60, 72], WINDOW_SIZE)
    ]

    shifted = observe(reference, pipeline, long_tenure)
    spread_out = observe(reference, pipeline, mixed)

    assert shifted.numeric["tenure"]["psi"]["value"] > spread_out.numeric["tenure"]["psi"]["value"]
    assert shifted.numeric["tenure"]["psi"]["status"] == STATUS_CRITICAL
    assert shifted.numeric["tenure"]["mean_shift_in_reference_sd"] > 1.0
    assert shifted.status in {STATUS_WARNING, STATUS_CRITICAL}


def test_a_value_beyond_the_reference_range_is_counted(reference, pipeline) -> None:
    """Out-of-range is its own signal, separate from the histogram."""
    beyond = reference.numeric["MonthlyCharges"].max + 500.0
    records = [{**BASE_RECORD, "MonthlyCharges": beyond}] * WINDOW_SIZE

    report = observe(reference, pipeline, records)

    assert report.numeric["MonthlyCharges"]["out_of_reference_range_rate"]["value"] == 1.0
    assert report.numeric["MonthlyCharges"]["out_of_reference_range_rate"]["status"] == (
        STATUS_CRITICAL
    )


def test_a_concentrated_category_raises_tvd(reference, pipeline) -> None:
    """Every record on one contract term, where the reference had three."""
    concentrated = [{**BASE_RECORD, "Contract": "Two year"}] * WINDOW_SIZE
    balanced = [
        {**BASE_RECORD, "Contract": value}
        for value in spread(["Month-to-month", "One year", "Two year"], WINDOW_SIZE)
    ]

    focused = observe(reference, pipeline, concentrated)
    even = observe(reference, pipeline, balanced)

    assert (
        focused.categorical["Contract"]["tvd"]["value"]
        > (even.categorical["Contract"]["tvd"]["value"])
    )
    assert focused.categorical["Contract"]["tvd"]["status"] == STATUS_CRITICAL
    assert focused.categorical["Contract"]["unseen_rate"]["value"] == 0.0


def test_an_unseen_category_is_counted(reference, pipeline) -> None:
    """The API accepts it, the encoder ignores it, and monitoring makes it visible."""
    records = [{**BASE_RECORD, "PaymentMethod": "Instant transfer (PIX)"}] * WINDOW_SIZE

    report = observe(reference, pipeline, records)
    entry = report.categorical["PaymentMethod"]

    assert entry["unseen_rate"]["value"] == 1.0
    assert entry["unseen_count"] == WINDOW_SIZE
    assert entry["n_distinct_unseen_observed"] == 1
    assert entry["unseen_rate"]["status"] == STATUS_CRITICAL
    assert entry["count_by_level"]["__UNSEEN__"] == WINDOW_SIZE
    assert report.quality["unseen_category_records"] == WINDOW_SIZE


def test_distinct_unseen_categories_are_counted_not_stored(reference, pipeline) -> None:
    new_methods = ["PIX", "Crypto", "Voucher", "Direct debit v2"]
    records = [
        {**BASE_RECORD, "PaymentMethod": method} for method in spread(new_methods, WINDOW_SIZE)
    ]

    report = observe(reference, pipeline, records)
    entry = report.categorical["PaymentMethod"]

    assert entry["n_distinct_unseen_observed"] == len(new_methods)
    assert entry["distinct_unseen_tracking_saturated"] is False
    assert entry["distinct_unseen_is_lower_bound"] is False
    rendered = str(report.as_record())
    for method in new_methods:
        assert method not in rendered


def test_a_broken_equivalence_is_counted(reference, pipeline) -> None:
    """InternetService=No alongside OnlineSecurity=No — impossible in the reference."""
    records = [
        {
            **BASE_RECORD,
            "InternetService": "No",
            "OnlineSecurity": "No",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
        }
    ] * WINDOW_SIZE

    report = observe(reference, pipeline, records)

    assert report.structural["records_with_any_violation"] == WINDOW_SIZE
    assert report.structural["violation_rate"] == 1.0
    assert report.structural["status"] == STATUS_CRITICAL
    assert report.quality["structural_violation_records"] == WINDOW_SIZE
    broken_rule = "InternetService=No<->OnlineSecurity=No internet service"
    assert report.structural["violations_by_rule"][broken_rule] == WINDOW_SIZE


def test_a_violating_record_is_still_scored(reference, pipeline) -> None:
    """Watched, not enforced: Phase 12 does not retroactively reject Phase 11 input."""
    records = [{**BASE_RECORD, "InternetService": "No", "OnlineSecurity": "No"}] * WINDOW_SIZE

    report = observe(reference, pipeline, records)

    assert report.score["count"] == WINDOW_SIZE
    assert report.quality["successful_records"] == WINDOW_SIZE


def test_a_shifted_population_moves_the_score(reference, pipeline) -> None:
    """Prediction drift: the system marks a different share of requests positive."""
    low_risk = [
        {
            **BASE_RECORD,
            "tenure": 70,
            "Contract": "Two year",
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
            "MonthlyCharges": 20.0,
            "TotalCharges": "1400.00",
            "PaymentMethod": "Bank transfer (automatic)",
            "PaperlessBilling": "No",
        }
    ] * WINDOW_SIZE

    report = observe(reference, pipeline, low_risk)

    assert report.score["psi"]["value"] > 1.0
    assert report.score["psi"]["status"] == STATUS_CRITICAL
    assert report.score["predicted_positive_rate"] == 0.0
    assert report.score["predicted_positive_rate_delta"]["value"] < -0.3
    assert report.score["predicted_positive_rate_delta"]["status"] == STATUS_CRITICAL
    # The threshold and the comparison are reported, and unchanged.
    assert report.score["threshold"] == reference.score.threshold
    assert report.score["comparison"] == ">="


def test_score_drift_is_never_reported_as_performance(reference, pipeline) -> None:
    records = [{**BASE_RECORD, "Contract": "Two year", "tenure": 70}] * WINDOW_SIZE

    record = observe(reference, pipeline, records).as_record()

    assert record["performance_degradation"]["evaluated"] is False
    assert "labels" in record["performance_degradation"]["reason"]
    assert "NOT evidence that the model degraded" in record["interpretation"]


# --------------------------------------------------------------------------- #
# Window size.
# --------------------------------------------------------------------------- #


def test_a_small_window_claims_no_drift(reference, pipeline) -> None:
    """No verdict, and no distribution either — the window is too small for both.

    The two are the same decision seen from two sides. A PSI over five records is
    dominated by the epsilon rather than by the data, so the number would not mean
    what a reader thinks; and a histogram over five records is close enough to the
    records to identify them. Neither is reported.
    """
    policy = get_monitoring_policy()
    records = [{**BASE_RECORD, "tenure": 71, "Contract": "Two year"}] * 5

    report = observe(reference, pipeline, records)

    assert report.n_records < policy.minimum_window_size
    assert report.status == STATUS_INSUFFICIENT_DATA
    assert report.has_verdict is False
    assert report.details_suppressed is True
    # The global counters are all there: a suppressed window is not a dead process.
    assert report.n_records == 5
    assert report.quality["successful_records"] == 5
    # Nothing per-feature, per-bin or per-rule was computed.
    assert report.numeric == {}
    assert report.categorical == {}
    assert report.score == {}
    assert report.structural == {}


def test_the_same_window_at_full_size_does_claim_drift(reference, pipeline) -> None:
    """The contrast that shows the floor is about size, not about the data."""
    policy = get_monitoring_policy()
    records = [{**BASE_RECORD, "tenure": 71, "Contract": "Two year"}]

    small = observe(reference, pipeline, records * 5)
    large = observe(reference, pipeline, records * policy.minimum_window_size)

    assert small.status == STATUS_INSUFFICIENT_DATA
    assert large.status == STATUS_CRITICAL
    assert large.has_verdict is True
