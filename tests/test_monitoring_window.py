"""Windows are counter generations, and closing one is a single atomic operation.

Phase 12 asked for a window semantics that is not merely "everything since the
process started". The primitive that provides it is
:meth:`~churn.monitoring.collector.MonitoringCollector.reset`: it returns the
snapshot of the window it closes and opens an empty one, both inside one acquisition
of the collector's lock.

Two properties are worth testing and neither is obvious from reading the method.
**The halves must not come apart** — snapshot-then-clear as two operations would
silently drop every update that arrived between them, which is data loss inside the
component whose whole job is to notice things. And **a concurrent update must land
in exactly one window** — not both, not neither.

This is single-process consistency. There is no distributed claim here, and none is
needed: one collector, one address space, one lock.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from churn.modeling.freeze import apply_decision_rule, load_pipeline, positive_probability
from churn.monitoring.collector import QUALITY_COUNTERS, MonitoringCollector, WindowSnapshot
from churn.monitoring.reference import ReferenceProfile, load_reference_profile
from churn.monitoring.service import MonitoringService, compare
from churn.monitoring.settings import STATUS_INSUFFICIENT_DATA, get_monitoring_policy
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features

#: A single well-formed record. Its values do not matter here — only that observing
#: it moves every accumulator, so a reset that missed one would be visible.
RECORD: dict[str, object] = {
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

MINIMUM = get_monitoring_policy().minimum_window_size


@pytest.fixture(scope="module")
def reference() -> ReferenceProfile:
    return load_reference_profile()


@pytest.fixture(scope="module")
def scored(reference: ReferenceProfile):
    """One record scored once, replayed as many times as a window needs."""
    pipeline = load_pipeline()
    features = prepare_features(pd.DataFrame([RECORD]).loc[:, list(FEATURE_COLUMNS)])
    scores = np.asarray(positive_probability(pipeline, features), dtype=float)
    return features, scores, apply_decision_rule(scores, reference.score.threshold)


def _observe(collector: MonitoringCollector, scored, times: int) -> None:
    features, scores, decisions = scored
    for _ in range(times):
        collector.record_request(1)
        collector.observe(features, scores, decisions)


# --------------------------------------------------------------------------- #
# The primitive.
# --------------------------------------------------------------------------- #


def test_the_collector_exposes_a_snapshot_and_reset_primitive(reference) -> None:
    """``reset`` is snapshot-and-reset, not a bare clear: it returns the window."""
    collector = MonitoringCollector(reference)

    closing = collector.reset()

    assert isinstance(closing, WindowSnapshot)
    assert collector.snapshot().n_records == 0


def test_a_reset_closes_one_window_and_opens_an_empty_one(reference, scored) -> None:
    """Window A of 100, then a fresh window B of 25 — the sequence Phase 12 asked for."""
    collector = MonitoringCollector(reference)

    _observe(collector, scored, MINIMUM)
    window_a = collector.reset()

    assert window_a.n_records == MINIMUM
    assert window_a.quality["records_total"] == MINIMUM
    # B starts empty, before anything is observed into it.
    assert collector.snapshot().n_records == 0
    assert collector.snapshot().quality == dict.fromkeys(QUALITY_COUNTERS, 0)

    _observe(collector, scored, 25)
    window_b = collector.snapshot()

    assert window_b.n_records == 25
    report_a = compare(window_a, reference, get_monitoring_policy())
    report_b = compare(window_b, reference, get_monitoring_policy())
    assert report_a.status != STATUS_INSUFFICIENT_DATA
    assert report_b.status == STATUS_INSUFFICIENT_DATA
    assert report_b.details_suppressed is True


def test_a_reset_clears_every_accumulator_not_only_the_record_count(reference, scored) -> None:
    """A partially cleared window would report window B's count over A's histogram."""
    collector = MonitoringCollector(reference)
    _observe(collector, scored, MINIMUM)
    collector.reset()

    empty = collector.snapshot()

    assert all(sum(entry["bin_counts"]) == 0 for entry in empty.numeric.values())
    assert all(entry["count"] == 0 for entry in empty.categorical.values())
    assert all(entry["n_distinct_unseen_observed"] == 0 for entry in empty.categorical.values())
    assert all(
        entry["distinct_unseen_tracking_saturated"] is False for entry in empty.categorical.values()
    )
    assert sum(empty.score["bin_counts"]) == 0
    assert empty.score["predicted_positive_count"] == 0
    assert empty.structural["records_with_any_violation"] == 0
    assert all(count == 0 for count in empty.structural["violations_by_rule"].values())


def test_closing_a_window_changes_nothing_the_model_owns(reference, scored) -> None:
    """The blast radius of a reset is the integers this module owns, and that is all."""
    service = MonitoringService.from_reference(reference, "unused-digest")
    before = (
        service.reference.score.threshold,
        service.reference.score.calibration_policy,
        service.reference.provenance.model_fingerprint_sha256,
        service.reference.provenance.pipeline_sha256,
        service.policy,
        service.reference_sha256,
    )

    _observe(service.collector, scored, 5)
    service.close_window()

    after = (
        service.reference.score.threshold,
        service.reference.score.calibration_policy,
        service.reference.provenance.model_fingerprint_sha256,
        service.reference.provenance.pipeline_sha256,
        service.policy,
        service.reference_sha256,
    )
    assert before == after
    assert service.collector.reference is reference


def test_close_window_reports_on_the_window_it_closed(reference, scored) -> None:
    """The report describes A; the collector is already serving B."""
    service = MonitoringService.from_reference(reference, "unused-digest")
    _observe(service.collector, scored, MINIMUM)

    report = service.close_window()

    assert report.n_records == MINIMUM
    assert service.report().n_records == 0


# --------------------------------------------------------------------------- #
# Atomicity, single process.
# --------------------------------------------------------------------------- #


def test_concurrent_observation_and_reset_lose_and_duplicate_nothing(reference, scored) -> None:
    """Eight writers, one resetter, and the arithmetic has to close exactly.

    Every observation must be counted once: in the window that was open when it took
    the lock, or in the next one, never in both and never in neither. Summing the
    closed windows and the final open one has to give back exactly what was written.
    """
    collector = MonitoringCollector(reference)
    features, scores, decisions = scored
    writers, per_writer = 8, 40
    closed: list[WindowSnapshot] = []
    closed_lock = threading.Lock()
    failures: list[BaseException] = []
    start = threading.Barrier(writers + 1)
    stop = threading.Event()

    def write() -> None:
        try:
            start.wait()
            for _ in range(per_writer):
                collector.record_request(1)
                collector.observe(features, scores, decisions)
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            failures.append(error)

    def rotate() -> None:
        try:
            start.wait()
            while not stop.is_set():
                snapshot = collector.reset()
                with closed_lock:
                    closed.append(snapshot)
        except BaseException as error:  # noqa: BLE001
            failures.append(error)

    threads = [threading.Thread(target=write) for _ in range(writers)]
    rotator = threading.Thread(target=rotate)
    rotator.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stop.set()
    rotator.join()
    final = collector.snapshot()

    assert not failures, f"a thread raised: {failures}"
    every = [*closed, final]
    assert all(snapshot.n_records >= 0 for snapshot in every)
    assert all(count >= 0 for snapshot in every for count in snapshot.quality.values()), (
        "a counter went negative"
    )
    assert sum(snapshot.n_records for snapshot in every) == writers * per_writer
    assert sum(snapshot.quality["successful_records"] for snapshot in every) == writers * per_writer


def test_a_window_is_internally_consistent_under_concurrent_rotation(reference, scored) -> None:
    """Within one closed window, the counters must agree with each other.

    A snapshot torn across a reset would show a record count that its own histograms
    do not add up to — the failure mode a bare clear-then-read produces.
    """
    collector = MonitoringCollector(reference)
    features, scores, decisions = scored
    closed: list[WindowSnapshot] = []
    closed_lock = threading.Lock()
    stop = threading.Event()

    def rotate() -> None:
        while not stop.is_set():
            snapshot = collector.reset()
            with closed_lock:
                closed.append(snapshot)

    rotator = threading.Thread(target=rotate)
    rotator.start()
    for _ in range(200):
        collector.observe(features, scores, decisions)
    stop.set()
    rotator.join()

    for snapshot in [*closed, collector.snapshot()]:
        assert sum(snapshot.score["bin_counts"]) == snapshot.n_records
        for entry in snapshot.numeric.values():
            assert sum(entry["bin_counts"]) == snapshot.n_records
            assert entry["count"] == snapshot.n_records
        for entry in snapshot.categorical.values():
            assert sum(entry["count_by_level"].values()) == snapshot.n_records
