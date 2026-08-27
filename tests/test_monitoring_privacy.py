"""What a monitoring report must not let a reader recover about one record.

The threat this file exists for is not a payload being stored — the collector has no
buffer to leak. It is subtler: **a distribution over one record is that record.**
``count_by_level = {"Month-to-month": 1}`` names the contract. ``bin_counts =
[0, 1, 0, ...]`` places the tenure in a decile. ``predicted_positive_rate = 1.0``
over one record is that customer's decision. Each of those is an aggregate by type
and a row by content, and a low-traffic deployment whose monitoring endpoint is
polled after every request would hand them back one customer at a time.

So the tests here do not check individual fields for individual leaks. They take a
window built from a record with values chosen to be unmistakable, serialise the
whole report, and walk it — every key, every value, every list — asserting that
nothing in it identifies the record. A field-by-field allow-list would pass the day
someone adds a new distribution and forgets to suppress it; a recursive sweep fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest

from churn.modeling.freeze import apply_decision_rule, load_pipeline, positive_probability
from churn.monitoring.collector import QUALITY_COUNTERS, MonitoringCollector
from churn.monitoring.reference import ReferenceProfile, load_reference_profile
from churn.monitoring.service import SUPPRESSED_SECTIONS, compare
from churn.monitoring.settings import STATUS_INSUFFICIENT_DATA, get_monitoring_policy
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, prepare_features

#: One record whose values are recognisable on sight, so a leak is unambiguous
#: rather than a coincidence of the reference population.
MARKED_RECORD: dict[str, object] = {
    "tenure": 3,
    "MonthlyCharges": 97.31,
    "TotalCharges": "291.93",
    "gender": "Female",
    "SeniorCitizen": 1,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

#: An unseen level that could not plausibly come from anywhere but this test.
SECRET_LEVEL = "SECRET_CONTRACT_VALUE_ZZZ999"


@pytest.fixture(scope="module")
def reference() -> ReferenceProfile:
    return load_reference_profile()


def _score(reference: ReferenceProfile, records: list[dict[str, object]]):
    """Run records through the frozen pipeline exactly as serving would."""
    pipeline = load_pipeline()
    features = prepare_features(pd.DataFrame(records).loc[:, list(FEATURE_COLUMNS)])
    scores = np.asarray(positive_probability(pipeline, features), dtype=float)
    return features, scores, apply_decision_rule(scores, reference.score.threshold)


def _window(reference: ReferenceProfile, records: list[dict[str, object]]) -> dict[str, Any]:
    """Observe a window and return the report it produces, as plain data."""
    features, scores, decisions = _score(reference, records)
    collector = MonitoringCollector(reference)
    collector.record_request(len(records))
    collector.observe(features, scores, decisions)
    return compare(collector.snapshot(), reference, get_monitoring_policy()).as_record()


def _walk(node: Any, path: str = "") -> list[tuple[str, Any]]:
    """Return every ``(path, leaf)`` in a nested structure, keys included as leaves."""
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.append((f"{path}.{key}", key))
            found.extend(_walk(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk(value, f"{path}[{index}]"))
    else:
        found.append((path, node))
    return found


# --------------------------------------------------------------------------- #
# n = 1.
# --------------------------------------------------------------------------- #


def test_a_one_record_window_cannot_be_read_back_out_of_the_report(reference) -> None:
    """The central claim: one record in, nothing about that record out.

    Every categorical value the record carried is checked against every key and every
    value anywhere in the report, at any depth. ``Fiber optic`` appearing as a key
    with the count ``1`` beside it would be exactly as much of a leak as the string
    appearing in a field called ``last_value``.
    """
    record = _window(reference, [MARKED_RECORD])
    leaves = _walk(record, "report")

    marked_values = {str(MARKED_RECORD[feature]) for feature in CATEGORICAL_FEATURES}
    offenders = [
        (path, leaf) for path, leaf in leaves if isinstance(leaf, str) and leaf in marked_values
    ]

    assert record["n_records"] == 1
    assert record["status"] == STATUS_INSUFFICIENT_DATA
    assert record["details_suppressed"] is True
    assert not offenders, f"the report names the record's own values at {offenders}"


def test_no_counter_in_a_one_record_window_can_be_narrowed_to_a_feature(reference) -> None:
    """A count of 1 is fine; a count of 1 *attached to a level or a bin* is not.

    With one record every non-zero detailed count is that record. So the assertion is
    structural: outside the global data-quality counters, no numeric leaf may be
    non-zero anywhere in the report.
    """
    record = _window(reference, [MARKED_RECORD])
    detail = {key: value for key, value in record.items() if key != "data_quality"}

    numeric_leaves = [
        (path, leaf)
        for path, leaf in _walk(detail, "report")
        if isinstance(leaf, (int, float)) and not isinstance(leaf, bool)
    ]
    suspicious = [
        (path, leaf)
        for path, leaf in numeric_leaves
        if leaf not in (0, 0.0) and not path.endswith((".n_records", ".minimum_window_size"))
    ]

    assert not suspicious, f"a per-record quantity survived: {suspicious}"


def test_a_small_window_exposes_no_distribution_at_all(reference) -> None:
    """Not "fewer statistics" — none. The sections are absent, not thinned out."""
    record = _window(reference, [MARKED_RECORD])

    for section in SUPPRESSED_SECTIONS:
        assert record[section] is None, section
    assert record["small_window_note"] is not None
    assert "suppressed" in record["small_window_note"]


def test_an_unseen_level_in_a_one_record_window_is_invisible(reference) -> None:
    """Not the text, not a count of it, not a distinct-cardinality of one."""
    record = _window(reference, [{**MARKED_RECORD, "Contract": SECRET_LEVEL}])
    rendered = json.dumps(record, ensure_ascii=False)
    keys = {leaf for path, leaf in _walk(record, "report") if path.endswith(f".{leaf}")}

    assert SECRET_LEVEL not in rendered
    assert "n_distinct_unseen_observed" not in keys
    assert "distinct_unseen_tracking_saturated" not in keys
    assert "unseen_rate" not in keys
    assert "unseen_count" not in keys
    # The deployment still learns that *something* unknown arrived.
    assert record["data_quality"]["unseen_category_records"] == 1


def test_a_structural_violation_in_a_one_record_window_names_no_rule(reference) -> None:
    """Which equivalence broke is a property of the record. The count is not."""
    broken = {
        **MARKED_RECORD,
        "InternetService": "No",
        "OnlineSecurity": "No",
        "StreamingTV": "Yes",
    }
    record = _window(reference, [broken])

    keys = {leaf for path, leaf in _walk(record, "report") if path.endswith(f".{leaf}")}

    assert record["structural_consistency"] is None
    assert "violations_by_rule" not in keys
    assert record["data_quality"]["structural_violation_records"] == 1


def test_the_decision_of_a_single_record_is_not_reported(reference) -> None:
    """``predicted_positive_rate`` over one record is that customer's decision."""
    record = _window(reference, [MARKED_RECORD])
    # The prose fields name these quantities to explain them; only structure counts.
    structure = {
        key: value
        for key, value in record.items()
        if key not in ("interpretation", "calibration", "small_window_note")
    }
    keys = {leaf for path, leaf in _walk(structure, "report") if path.endswith(f".{leaf}")}

    assert record["prediction_drift"] is None
    for leaked in ("predicted_positive_rate", "predicted_positive_count", "bin_counts"):
        assert leaked not in keys


# --------------------------------------------------------------------------- #
# The boundary: 99, then 100.
# --------------------------------------------------------------------------- #


def test_the_window_just_below_the_minimum_is_suppressed_too(reference) -> None:
    """99 is not "nearly enough". The policy is a threshold, not a gradient."""
    minimum = get_monitoring_policy().minimum_window_size
    record = _window(reference, [MARKED_RECORD] * (minimum - 1))

    assert record["n_records"] == minimum - 1
    assert record["status"] == STATUS_INSUFFICIENT_DATA
    assert record["details_suppressed"] is True
    assert all(record[section] is None for section in SUPPRESSED_SECTIONS)


def test_at_the_minimum_the_distributions_are_reported(reference) -> None:
    """The contrast, and the reason the cut exists where it does."""
    minimum = get_monitoring_policy().minimum_window_size
    record = _window(reference, [MARKED_RECORD] * minimum)

    assert record["n_records"] == minimum
    assert record["status"] != STATUS_INSUFFICIENT_DATA
    assert record["details_suppressed"] is False
    assert record["small_window_note"] is None
    numeric = record["feature_drift"]["numeric"]["tenure"]
    categorical = record["feature_drift"]["categorical"]["Contract"]
    assert numeric["mean"] is not None
    assert sum(numeric["bin_counts"]) == minimum
    assert categorical["count_by_level"]["Month-to-month"] == minimum
    assert record["prediction_drift"]["predicted_positive_rate"] is not None
    assert record["structural_consistency"]["violations_by_rule"] is not None


def test_the_suppressed_report_still_says_the_deployment_is_alive(reference) -> None:
    """Suppression must not be indistinguishable from a dead process."""
    record = _window(reference, [MARKED_RECORD])

    assert set(record["data_quality"]) == set(QUALITY_COUNTERS)
    assert record["data_quality"]["requests_total"] == 1
    assert record["data_quality"]["records_total"] == 1
    assert record["data_quality"]["successful_records"] == 1
    assert record["minimum_window_size"] == get_monitoring_policy().minimum_window_size
    assert len(record["reference_profile_sha256"]) == 64
    assert set(record["section_status"]) == {
        "data_quality",
        "feature_drift",
        "prediction_drift",
        "structural_consistency",
        "performance_degradation",
    }


# --------------------------------------------------------------------------- #
# Unseen-value digests.
# --------------------------------------------------------------------------- #


def test_no_unseen_digest_reaches_a_snapshot_a_report_or_a_log(
    reference, caplog: pytest.LogCaptureFixture
) -> None:
    """The digest is an in-memory bookkeeping detail, and must stay one.

    SHA-256 of a categorical value is *not* anonymisation — the domain is small
    enough to hash exhaustively — so the property that matters is that no reader
    exists. Here that property is checked on all four surfaces at once: the
    snapshot, the report, the serialised response and the log.
    """
    digest = hashlib.sha256(SECRET_LEVEL.encode("utf-8")).hexdigest()
    minimum = get_monitoring_policy().minimum_window_size
    records = [{**MARKED_RECORD, "Contract": SECRET_LEVEL}] * minimum

    features, scores, decisions = _score(reference, records)
    collector = MonitoringCollector(reference)
    with caplog.at_level(logging.DEBUG, logger="churn"):
        collector.observe(features, scores, decisions)
        snapshot = collector.reset()
    report = compare(snapshot, reference, get_monitoring_policy())

    rendered = json.dumps(snapshot.as_record()) + json.dumps(report.as_record())
    logged = "\n".join(entry.getMessage() for entry in caplog.records)

    assert digest not in rendered
    assert digest not in logged
    assert SECRET_LEVEL not in rendered
    assert SECRET_LEVEL not in logged
    # The cardinality — and only the cardinality — survives.
    assert report.categorical["Contract"]["n_distinct_unseen_observed"] == 1


def test_the_digest_set_is_process_local_and_not_a_field_of_any_snapshot(reference) -> None:
    """A snapshot is the only thing that leaves the collector, and it has no set."""
    features, scores, decisions = _score(reference, [{**MARKED_RECORD, "Contract": SECRET_LEVEL}])
    collector = MonitoringCollector(reference)
    collector.observe(features, scores, decisions)

    snapshot = collector.snapshot()
    entry = snapshot.categorical["Contract"]

    assert set(entry) == {
        "count",
        "count_by_level",
        "unseen_count",
        "unseen_rate",
        "n_distinct_unseen_observed",
        "distinct_unseen_tracking_cap",
        "distinct_unseen_tracking_saturated",
        "distinct_unseen_is_lower_bound",
    }
    assert all(not isinstance(value, (set, frozenset)) for value in entry.values())
    assert entry["n_distinct_unseen_observed"] == 1


def test_every_digest_in_a_committed_artefact_is_a_named_artefact_fingerprint() -> None:
    """Persistence, checked by shape rather than by hunting for one known digest.

    A value digest that leaked into a file would appear as a 64-character hex string
    somewhere no fingerprint belongs. So every such string in the reference profile
    and in the Phase 12 record must sit under a key that declares what it hashes —
    ``*_sha256`` or ``model_fingerprint``. Anything else is an unexplained digest,
    which is exactly what an escaped unseen-value hash would look like.
    """
    from churn.monitoring.results import RECORD_PATH
    from churn.monitoring.settings import default_reference_profile_path

    unexplained: list[str] = []
    for path in (default_reference_profile_path(), RECORD_PATH):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for location, leaf in _walk(payload, path.name):
            if not (isinstance(leaf, str) and len(leaf) == 64):
                continue
            try:
                int(leaf, 16)
            except ValueError:
                continue
            key = location.rsplit(".", 1)[-1]
            if not (key.endswith("sha256") or key == "model_fingerprint"):
                unexplained.append(location)

    assert not unexplained, f"unexplained digest(s) in a committed artefact: {unexplained}"
