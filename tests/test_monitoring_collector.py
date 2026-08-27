"""The collector: what it counts, what it refuses to keep, and whether it is safe.

Privacy is asserted structurally rather than by inspecting an output. The strongest
version of "no payload is persisted" is that there is nowhere to persist one: the
collector's entire state is enumerated here and checked to contain only integers,
floats and count tables.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from churn.monitoring.collector import QUALITY_COUNTERS, MonitoringCollector
from churn.monitoring.reference import load_reference_profile
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features

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


@pytest.fixture(scope="module")
def reference():
    return load_reference_profile()


@pytest.fixture
def collector(reference) -> MonitoringCollector:
    return MonitoringCollector(reference)


def features_for(records: list[dict[str, object]]) -> pd.DataFrame:
    return prepare_features(pd.DataFrame(records).loc[:, list(FEATURE_COLUMNS)])


# --------------------------------------------------------------------------- #
# Counting.
# --------------------------------------------------------------------------- #


def test_an_empty_window_reports_zeros(collector) -> None:
    snapshot = collector.snapshot()

    assert snapshot.n_records == 0
    assert set(snapshot.quality) == set(QUALITY_COUNTERS)
    assert all(value == 0 for value in snapshot.quality.values())
    assert snapshot.score["predicted_positive_rate"] == 0.0


def test_observation_updates_every_aggregate(collector) -> None:
    features = features_for([BASE_RECORD] * 10)
    scores = np.full(10, 0.6)
    decisions = np.ones(10, dtype=int)

    collector.record_request(10)
    collector.observe(features, scores, decisions)
    snapshot = collector.snapshot()

    assert snapshot.n_records == 10
    assert snapshot.quality["requests_total"] == 1
    assert snapshot.quality["records_total"] == 10
    assert snapshot.quality["successful_records"] == 10
    assert snapshot.score["predicted_positive_count"] == 10
    assert snapshot.score["predicted_positive_rate"] == 1.0
    assert sum(snapshot.numeric["tenure"]["bin_counts"]) == 10
    assert sum(snapshot.categorical["Contract"]["count_by_level"].values()) == 10


def test_the_request_and_record_counts_are_separate(collector) -> None:
    """A batch is one request and many records; conflating them hides load."""
    collector.record_request(1)
    collector.record_request(50)

    counters = collector.counters()

    assert counters["requests_total"] == 2
    assert counters["records_total"] == 51


def test_rejections_are_counted_by_class(collector) -> None:
    collector.record_invalid_schema(3)
    collector.record_invalid_feature(2)

    counters = collector.counters()

    assert counters["invalid_schema_records"] == 3
    assert counters["invalid_feature_records"] == 2
    assert counters["successful_records"] == 0


def test_mismatched_arguments_raise_rather_than_truncate(collector) -> None:
    """Silently dropping half a batch would corrupt every rate downstream."""
    features = features_for([BASE_RECORD] * 3)

    with pytest.raises(ValueError):
        collector.observe(features, [0.5, 0.5], [1, 1])


def test_a_window_reset_returns_the_closing_snapshot(collector) -> None:
    """No observation is lost between two windows."""
    collector.record_request(4)
    collector.observe(features_for([BASE_RECORD] * 4), np.full(4, 0.5), np.ones(4, dtype=int))

    closed = collector.reset()
    fresh = collector.snapshot()

    assert closed.n_records == 4
    assert fresh.n_records == 0
    assert all(value == 0 for value in fresh.quality.values())


def test_a_window_reset_changes_no_model_state(collector, reference) -> None:
    """It is a counter operation and nothing else."""
    before = collector.reference

    collector.reset()

    assert collector.reference is before
    assert collector.reference.score.threshold == reference.score.threshold
    assert collector.reference.score.calibration_policy == "NONE"


# --------------------------------------------------------------------------- #
# Privacy.
# --------------------------------------------------------------------------- #


def test_the_collector_keeps_no_record(collector) -> None:
    """There is no buffer of records, so there is nothing to leak.

    "No payload is persisted" is asserted as a property of the data structure, not of
    a redaction step that could be forgotten: the collector's state is a fixed set of
    counters, count tables and running moments whose size does not grow with traffic.

    Note what this does *not* claim. A window min and max ARE reported — they are
    order statistics, the same ones the reference profile records for the training
    pool, and they are what makes an out-of-range excursion diagnosable. In a window
    of one record they necessarily equal that record's value; that is a property of
    tiny windows rather than of this aggregate, and it is one more reason the report
    withholds a verdict below the minimum window size.
    """
    features = features_for([{**BASE_RECORD, "MonthlyCharges": 99.99}] * 5)
    collector.observe(features, np.full(5, 0.42), np.ones(5, dtype=int))
    snapshot = collector.snapshot()

    # No per-record score survives: the score state is moments and a histogram, and
    # nothing that could hold five individual values. (A substring search would be
    # the wrong check here — the mean of five identical scores IS 0.42, and that is
    # an aggregate, not a retained record.)
    assert set(snapshot.score) == {
        "count",
        "mean",
        "std",
        "bin_counts",
        "predicted_positive_count",
        "predicted_positive_rate",
    }

    # State size is independent of how many records were seen.
    before = len(str(snapshot.as_record()))
    collector.observe(features_for([BASE_RECORD] * 200), np.full(200, 0.3), np.zeros(200, int))
    assert len(str(collector.snapshot().as_record())) == pytest.approx(before, rel=0.35)

    # The categorical table names KNOWN levels, and that is not customer data: the
    # vocabulary came from the reference profile, which is already committed. What
    # matters is that the table carries no level the profile did not already list.
    known = set(collector.reference.categorical["PaymentMethod"].known_levels) | {"__UNSEEN__"}
    assert set(snapshot.categorical["PaymentMethod"]["count_by_level"]) == known


def test_the_collector_state_is_only_numbers(collector) -> None:
    features = features_for([BASE_RECORD] * 3)
    collector.observe(features, np.full(3, 0.5), np.ones(3, dtype=int))
    snapshot = collector.snapshot()

    for entry in snapshot.numeric.values():
        for key, value in entry.items():
            assert isinstance(value, int | float | list), key
    for value in snapshot.score.values():
        assert isinstance(value, int | float | list)


def test_the_collector_keeps_no_unseen_category_text(collector) -> None:
    """Cardinality is tracked with digests; the strings themselves are discarded."""
    secrets = ["SuperSecretPlan-A", "SuperSecretPlan-B", "SuperSecretPlan-A"]
    features = features_for([{**BASE_RECORD, "Contract": value} for value in secrets])

    collector.observe(features, np.full(3, 0.5), np.ones(3, dtype=int))
    snapshot = collector.snapshot()
    rendered = str(snapshot.as_record())

    assert snapshot.categorical["Contract"]["unseen_count"] == 3
    assert snapshot.categorical["Contract"]["n_distinct_unseen_observed"] == 2
    for secret in set(secrets):
        assert secret not in rendered


def test_no_identifier_can_reach_the_collector(collector) -> None:
    """It only ever sees the canonical feature matrix, which has no identifier."""
    features = features_for([BASE_RECORD] * 2)

    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert tuple(features.columns) == FEATURE_COLUMNS


# --------------------------------------------------------------------------- #
# Concurrency.
# --------------------------------------------------------------------------- #


def test_concurrent_observation_loses_no_update(collector) -> None:
    """Eight threads, a hundred observations each: every count must land.

    A histogram update is a read-modify-write, so an unlocked collector would lose
    increments here — not deterministically, but often enough that the totals stop
    adding up.
    """
    threads_count, per_thread = 8, 100
    features = features_for([BASE_RECORD])
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(per_thread):
                collector.record_request(1)
                collector.observe(features, [0.6], [1])
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(threads_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    expected = threads_count * per_thread
    snapshot = collector.snapshot()

    assert not errors
    assert snapshot.n_records == expected
    assert snapshot.quality["requests_total"] == expected
    assert snapshot.quality["successful_records"] == expected
    assert sum(snapshot.numeric["tenure"]["bin_counts"]) == expected
    assert sum(snapshot.score["bin_counts"]) == expected
    assert sum(snapshot.categorical["Contract"]["count_by_level"].values()) == expected


def test_a_snapshot_taken_during_writes_is_consistent(collector) -> None:
    """A snapshot must be one window, not a mixture of two moments."""
    features = features_for([BASE_RECORD])
    stop = threading.Event()
    inconsistent: list[tuple[int, int]] = []

    def writer() -> None:
        while not stop.is_set():
            collector.observe(features, [0.6], [1])

    def reader() -> None:
        for _ in range(200):
            snapshot = collector.snapshot()
            binned = sum(snapshot.numeric["tenure"]["bin_counts"])
            if binned != snapshot.n_records:
                inconsistent.append((binned, snapshot.n_records))

    writers = [threading.Thread(target=writer) for _ in range(4)]
    for thread in writers:
        thread.start()
    reader()
    stop.set()
    for thread in writers:
        thread.join()

    assert not inconsistent
