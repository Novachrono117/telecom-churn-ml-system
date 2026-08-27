"""The unseen-cardinality tracker must be bounded, and honest about being bounded.

Two properties that are easy to confuse live here, and the file exists to keep them
apart.

**Privacy** is containment: unseen values are never retained and their digests never
leave the process. That is established in ``test_monitoring_privacy.py`` and is
unaffected by anything below.

**Memory safety** is boundedness: *how many* digests may be retained at once. The
serving contract accepts an unseen category on purpose — the frozen encoder absorbs
it and the record is still scored — so a caller sending ``value_000001 …
value_1000000`` would have grown an unbounded set inside a long-lived process whose
window has no automatic rotation and whose monitoring endpoint is unauthenticated.
A component whose job is to observe must not be the thing that exhausts what it
observes, however private its contents are.

The cap costs exactly one thing, and the tests below pin down which: the *distinct*
count saturates and is then reported as a lower bound. The *occurrence* count and
the unseen *rate* — the quantity the alert policy actually reads — stay exact under
any traffic, because they are integers rather than a set.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from churn.modeling.freeze import apply_decision_rule, load_pipeline, positive_probability
from churn.monitoring.collector import MonitoringCollector
from churn.monitoring.reference import ReferenceProfile, load_reference_profile
from churn.monitoring.service import MonitoringService, compare
from churn.monitoring.settings import get_monitoring_policy, load_monitoring_policy
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, prepare_features

#: Small enough to saturate quickly in a test, large enough that the arithmetic is
#: not trivially satisfied. The production cap is read from configs/monitoring.toml.
TEST_CAP = 32

#: How many distinct unseen values the cardinality-attack tests send.
ATTACK_SIZE = 1000

BASE: dict[str, object] = {
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
def reference() -> ReferenceProfile:
    return load_reference_profile()


@pytest.fixture(scope="module")
def pipeline():
    return load_pipeline()


def _scored(reference: ReferenceProfile, pipeline, records: list[dict[str, object]]):
    features = prepare_features(pd.DataFrame(records).loc[:, list(FEATURE_COLUMNS)])
    scores = np.asarray(positive_probability(pipeline, features), dtype=float)
    return features, scores, apply_decision_rule(scores, reference.score.threshold)


def _attack(feature: str = "Contract", size: int = ATTACK_SIZE) -> list[dict[str, object]]:
    """One record per distinct never-seen value — the shape of the abuse case."""
    return [{**BASE, feature: f"value_{index:06d}"} for index in range(size)]


# --------------------------------------------------------------------------- #
# The bound.
# --------------------------------------------------------------------------- #


def test_a_thousand_distinct_unseen_values_do_not_grow_the_collector(reference, pipeline) -> None:
    """The cardinality attack, and the structural property that defeats it.

    A thousand distinct new categories arrive; the retained set holds thirty-two.
    The assertion is on the collector's own state rather than on process memory:
    what matters is that the state does not grow with the traffic, and RSS would
    measure the interpreter as much as the accumulator.
    """
    records = _attack()
    features, scores, decisions = _scored(reference, pipeline, records)
    collector = MonitoringCollector(reference, TEST_CAP)

    collector.record_request(len(records))
    collector.observe(features, scores, decisions)

    digests = collector._categorical["Contract"].unseen_digests
    entry = collector.snapshot().categorical["Contract"]

    assert len(digests) == TEST_CAP
    assert entry["n_distinct_unseen_observed"] == TEST_CAP
    assert entry["distinct_unseen_tracking_cap"] == TEST_CAP
    assert entry["distinct_unseen_tracking_saturated"] is True
    assert entry["distinct_unseen_is_lower_bound"] is True
    # The count of occurrences is untouched by the bound on distinct values.
    assert entry["unseen_count"] == ATTACK_SIZE


def test_saturation_costs_no_occurrence(reference, pipeline) -> None:
    """Every unseen record is counted, including the 968 that were never tracked."""
    records = _attack()
    features, scores, decisions = _scored(reference, pipeline, records)
    collector = MonitoringCollector(reference, TEST_CAP)
    collector.record_request(len(records))
    collector.observe(features, scores, decisions)

    snapshot = collector.snapshot()
    entry = snapshot.categorical["Contract"]

    assert entry["unseen_count"] == ATTACK_SIZE
    assert entry["unseen_rate"] == 1.0
    assert entry["count"] == ATTACK_SIZE
    assert entry["count_by_level"]["__UNSEEN__"] == ATTACK_SIZE
    assert snapshot.quality["unseen_category_records"] == ATTACK_SIZE


def test_the_bound_is_per_feature_not_per_collector(reference, pipeline) -> None:
    """Saturating one feature must not blind the tracker to another."""
    records = [
        {**BASE, "Contract": f"contract_{index:06d}", "PaymentMethod": "PIX"}
        for index in range(ATTACK_SIZE)
    ]
    features, scores, decisions = _scored(reference, pipeline, records)
    collector = MonitoringCollector(reference, TEST_CAP)
    collector.observe(features, scores, decisions)

    categorical = collector.snapshot().categorical

    assert categorical["Contract"]["distinct_unseen_tracking_saturated"] is True
    assert categorical["PaymentMethod"]["distinct_unseen_tracking_saturated"] is False
    assert categorical["PaymentMethod"]["n_distinct_unseen_observed"] == 1
    assert categorical["PaymentMethod"]["unseen_count"] == ATTACK_SIZE


def test_nothing_is_evicted_to_make_room(reference, pipeline) -> None:
    """The first cap distinct values are the ones retained, and they stay retained.

    Eviction would produce a number that is neither the cardinality nor a bound on
    it, and ``1024`` would mean something different in every window.
    """
    first = _attack(size=TEST_CAP)
    features, scores, decisions = _scored(reference, pipeline, first)
    collector = MonitoringCollector(reference, TEST_CAP)
    collector.observe(features, scores, decisions)
    retained = set(collector._categorical["Contract"].unseen_digests)

    later = [{**BASE, "Contract": f"later_{index:06d}"} for index in range(200)]
    features, scores, decisions = _scored(reference, pipeline, later)
    collector.observe(features, scores, decisions)

    assert collector._categorical["Contract"].unseen_digests == retained
    assert len(retained) == TEST_CAP


# --------------------------------------------------------------------------- #
# Semantics before and after saturation.
# --------------------------------------------------------------------------- #


def test_below_the_cap_the_count_is_exact_and_says_so(reference, pipeline) -> None:
    distinct = TEST_CAP - 10
    records = _attack(size=distinct)
    features, scores, decisions = _scored(reference, pipeline, records)
    collector = MonitoringCollector(reference, TEST_CAP)
    collector.observe(features, scores, decisions)

    entry = collector.snapshot().categorical["Contract"]

    assert entry["n_distinct_unseen_observed"] == distinct
    assert entry["distinct_unseen_tracking_saturated"] is False
    assert entry["distinct_unseen_is_lower_bound"] is False


def test_the_report_carries_the_qualifiers_and_explains_them(reference, pipeline) -> None:
    """A number a reader could misread must not travel without its caveat."""
    minimum = get_monitoring_policy().minimum_window_size
    records = _attack(size=max(ATTACK_SIZE, minimum))
    features, scores, decisions = _scored(reference, pipeline, records)
    collector = MonitoringCollector(reference, TEST_CAP)
    collector.record_request(len(records))
    collector.observe(features, scores, decisions)

    record = compare(collector.snapshot(), reference, get_monitoring_policy()).as_record()
    entry = record["feature_drift"]["categorical"]["Contract"]

    assert entry["distinct_unseen_is_lower_bound"] is True
    assert entry["n_distinct_unseen_observed"] == TEST_CAP
    assert "lower bound" in record["unseen_cardinality"]
    assert "remain exact" in record["unseen_cardinality"]
    # And the old, unqualified name is gone, so no reader can pick it up by habit.
    assert "n_distinct_unseen" not in entry


# --------------------------------------------------------------------------- #
# The cap must not reach the alert policy.
# --------------------------------------------------------------------------- #


def test_saturation_never_moves_an_alert_status(reference, pipeline) -> None:
    """Same traffic, two caps, identical verdicts.

    The unseen signal the policy grades is ``unseen_rate``, which counts occurrences.
    If a status could move with the tracking cap, an operational memory limit would
    be silently changing what the system alerts on.
    """
    minimum = get_monitoring_policy().minimum_window_size
    records = _attack(size=max(ATTACK_SIZE, minimum))
    features, scores, decisions = _scored(reference, pipeline, records)

    reports = []
    for cap in (4, 100_000):
        collector = MonitoringCollector(reference, cap)
        collector.record_request(len(records))
        collector.observe(features, scores, decisions)
        reports.append(compare(collector.snapshot(), reference, get_monitoring_policy()))

    tight, loose = reports

    assert tight.categorical["Contract"]["distinct_unseen_tracking_saturated"] is True
    assert loose.categorical["Contract"]["distinct_unseen_tracking_saturated"] is False
    assert tight.status == loose.status
    assert tight.section_status == loose.section_status
    for feature in CATEGORICAL_FEATURES:
        assert (
            tight.categorical[feature]["unseen_rate"] == loose.categorical[feature]["unseen_rate"]
        ), feature
        assert tight.categorical[feature]["status"] == loose.categorical[feature]["status"]
        assert (
            tight.categorical[feature]["unseen_count"] == loose.categorical[feature]["unseen_count"]
        )


def test_the_policy_has_the_cap_and_derives_no_status_from_it() -> None:
    """It lives with the cutoffs because operators tune it, not because it grades."""
    policy = get_monitoring_policy()

    assert policy.max_distinct_unseen_tracked_per_feature == 1024
    assert policy.as_record()["max_distinct_unseen_tracked_per_feature"] == 1024
    # It is an int, not a warning/critical pair: there is no `.status()` to call.
    assert isinstance(policy.max_distinct_unseen_tracked_per_feature, int)
    assert not hasattr(policy.max_distinct_unseen_tracked_per_feature, "status")


def test_the_cap_comes_from_the_operational_config_file() -> None:
    """Tunable where the other operational knobs are, and nowhere near the freeze.

    ``configs/monitoring.toml`` is deliberately not ``configs/base.toml``, whose
    digest a model freeze depends on. A memory limit has no business invalidating a
    frozen model, and a frozen model has no business pinning a memory limit.
    """
    from churn.monitoring.settings import DEFAULT_MONITORING_CONFIG_PATH

    text = DEFAULT_MONITORING_CONFIG_PATH.read_text(encoding="utf-8")
    policy = load_monitoring_policy()

    assert "max_distinct_unseen_tracked_per_feature" in text
    assert policy.max_distinct_unseen_tracked_per_feature == 1024
    assert DEFAULT_MONITORING_CONFIG_PATH.name == "monitoring.toml"
    assert "max_distinct_unseen" not in (
        DEFAULT_MONITORING_CONFIG_PATH.parent / "base.toml"
    ).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Reset.
# --------------------------------------------------------------------------- #


def test_a_reset_clears_saturation_and_the_digest_set(reference, pipeline) -> None:
    """Saturation is a property of one window, never of the process."""
    records = _attack()
    features, scores, decisions = _scored(reference, pipeline, records)
    collector = MonitoringCollector(reference, TEST_CAP)
    collector.record_request(len(records))
    collector.observe(features, scores, decisions)

    closing = collector.reset()
    fresh = collector.snapshot()

    assert closing.categorical["Contract"]["distinct_unseen_tracking_saturated"] is True
    assert closing.categorical["Contract"]["n_distinct_unseen_observed"] == TEST_CAP

    for feature in CATEGORICAL_FEATURES:
        entry = fresh.categorical[feature]
        assert entry["n_distinct_unseen_observed"] == 0, feature
        assert entry["distinct_unseen_tracking_saturated"] is False, feature
        assert entry["distinct_unseen_is_lower_bound"] is False, feature
        assert entry["unseen_count"] == 0, feature
        assert collector._categorical[feature].unseen_digests == set(), feature


def test_the_new_window_tracks_exactly_again_after_a_saturated_one(reference, pipeline) -> None:
    """Window B is not penalised for window A having been attacked."""
    features, scores, decisions = _scored(reference, pipeline, _attack())
    collector = MonitoringCollector(reference, TEST_CAP)
    collector.observe(features, scores, decisions)
    collector.reset()

    modest = [{**BASE, "Contract": f"new_{index}"} for index in range(3)]
    features, scores, decisions = _scored(reference, pipeline, modest)
    collector.observe(features, scores, decisions)

    entry = collector.snapshot().categorical["Contract"]

    assert entry["n_distinct_unseen_observed"] == 3
    assert entry["distinct_unseen_tracking_saturated"] is False
    assert entry["distinct_unseen_is_lower_bound"] is False


# --------------------------------------------------------------------------- #
# Thread safety of the bounded path.
# --------------------------------------------------------------------------- #


def test_concurrent_distinct_unseen_traffic_stays_bounded_and_exact(reference, pipeline) -> None:
    """Eight threads, all sending distinct new categories, all racing the cap.

    The check-then-insert around the cap is a read-modify-write like any histogram
    update, so it runs under the same ``RLock``. Two properties have to survive: the
    set never exceeds the cap, and no occurrence is lost while threads contend for
    the boundary.
    """
    workers, per_worker = 8, 60
    batches = [
        _scored(
            reference,
            pipeline,
            [{**BASE, "Contract": f"w{worker}_v{index:04d}"} for index in range(per_worker)],
        )
        for worker in range(workers)
    ]
    collector = MonitoringCollector(reference, TEST_CAP)
    failures: list[BaseException] = []
    start = threading.Barrier(workers)

    def write(batch) -> None:
        try:
            features, scores, decisions = batch
            start.wait()
            collector.record_request(len(features))
            collector.observe(features, scores, decisions)
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            failures.append(error)

    threads = [threading.Thread(target=write, args=(batch,)) for batch in batches]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    snapshot = collector.snapshot()
    entry = snapshot.categorical["Contract"]
    total = workers * per_worker

    assert not failures, f"a thread raised: {failures}"
    assert len(collector._categorical["Contract"].unseen_digests) <= TEST_CAP
    assert entry["n_distinct_unseen_observed"] == TEST_CAP
    assert entry["distinct_unseen_tracking_saturated"] is True
    # Nothing lost at the boundary: every occurrence counted exactly once.
    assert entry["unseen_count"] == total
    assert entry["count"] == total
    assert snapshot.n_records == total
    assert snapshot.quality["unseen_category_records"] == total
    assert all(count >= 0 for count in snapshot.quality.values())


def test_the_default_collector_uses_the_configured_cap(reference) -> None:
    """No caller has to remember to pass it, and no caller can silently drop it."""
    collector = MonitoringCollector(reference)
    expected = get_monitoring_policy().max_distinct_unseen_tracked_per_feature

    assert collector.max_distinct_unseen_tracked_per_feature == expected
    assert all(
        accumulator.max_distinct_unseen == expected
        for accumulator in collector._categorical.values()
    )


def test_the_service_collects_under_the_same_policy_it_reports_against(reference) -> None:
    """One policy governs the cutoffs and the bound, so the two cannot diverge."""
    policy = load_monitoring_policy()
    service = MonitoringService.from_reference(reference, "unused-digest", policy)

    assert (
        service.collector.max_distinct_unseen_tracked_per_feature
        == policy.max_distinct_unseen_tracked_per_feature
    )
    assert service.policy is policy
