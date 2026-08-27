"""Incremental, privacy-safe aggregation of what the boundary is being asked.

The collector never stores a record. It receives one, updates a handful of
counters, and forgets it. That is a privacy property first — there is no buffer of
customer data to leak, subpoena or accidentally log — and a memory property second:
every accumulator here has a size fixed by the reference profile, so the state a
process holds after a month of traffic has the same *shape* it had on its first
request. The one structure that could have grown with traffic is the unseen-digest
set, and it is explicitly bounded — see below. No heap figure is claimed: actual
Python memory use is implementation-dependent and was not benchmarked in this phase.

What it keeps, in total:

* per numeric feature — a bin count vector over the **reference** edges, a running
  sum and sum of squares, a min and a max, and an out-of-range count;
* per categorical feature — a count per known level plus ``__UNSEEN__``, and a
  **bounded** count of distinct unseen values;
* the model score — a bin count vector over the reference score edges, running
  moments, and a count of records at or above the frozen threshold;
* structural violations — per rule, and per record;
* data-quality counters — requests, records, and each rejection class.

What it never keeps: a payload, a feature value, a probability, a decision, an
identifier, or an unseen category's text.

**Distinct unseen values are counted, not stored.** ``n_distinct_unseen`` is useful
("eleven new contract terms appeared") and the eleven strings are not needed to act
on it. Keeping them would be unbounded in cardinality and would be customer-supplied
free text — the two properties that make a monitoring store into a data-protection
incident. The count is maintained with a hash-only set, which lets a repeat of the
same new category be recognised without the category being retained.

**The SHA-256 here is not anonymisation, and must not be described as such.** A
categorical feature has a small, guessable domain, so an unsalted digest of one of
its values is trivially reversed by hashing the candidates — a rainbow table with a
few dozen entries. What the digest buys is narrower and still worth having: the
cleartext never sits in process memory as a retained string, so it cannot reach a
heap dump, a debugger's locals or a log line by accident.

The property this design actually relies on is containment, not the hash: the digest
set is process-local, it is never serialised, never written to a report or a
snapshot, never logged, and never leaves :meth:`_CategoricalAccumulator.snapshot` —
which emits a count derived from it and nothing else. Because the digests never
cross a boundary, a keyed or salted construction would harden nothing that is
exposed; it would add a key to manage in exchange for a threat model that has no
reader in it. If a future phase ever needed to publish or persist unseen-value
identity, the unsalted digest would be inadequate and the design would have to
change with it.

**The digest set is also bounded, which is a different problem from privacy.**
Containment answers "can anyone read these"; it says nothing about "how many can
there be". The serving contract accepts an unseen category on purpose, so a caller
sending ``value_000001 … value_1000000`` would grow an unbounded set inside a
long-lived process whose window has no automatic rotation and whose monitoring
endpoint is unauthenticated. An observability component that can be made to exhaust
the process it observes is a defect regardless of how private its contents are.

So the set stops growing at ``max_distinct_unseen_tracked_per_feature``
(:mod:`churn.monitoring.settings`, from ``configs/monitoring.toml``). The guarantee
is stated in the unit it is actually enforced in: **at most that many retained
digests per feature per window**, so collector state is a function of
(features × cap) rather than of traffic. It is not a statement about bytes — the
heap cost of a Python ``set`` of ``str`` is implementation-dependent and larger than
the digests themselves, and nothing here measured it.

What the cap does **not** touch is the part anyone alerts on: ``unseen_count`` and
``unseen_rate`` are integers and a ratio, not a set, and stay exact at any cap and
any traffic. Only the distinct-cardinality diagnostic saturates, and a saturated
window says so rather than continuing to report the cap as though it were the true
cardinality — see :meth:`_CategoricalAccumulator.snapshot`.

Entries are never evicted to make room. A set that dropped old digests to admit new
ones would report a number that is neither the true cardinality nor a bound on it,
and "1024" would silently mean something different in every window.

**A window is a counter generation, not "since the process started".** The counters
here are cumulative *within a window*, and :meth:`MonitoringCollector.reset` closes
one: it returns the snapshot of the window it is closing and opens an empty one, both
under the same lock, so it is a snapshot-and-reset rather than two operations a
concurrent request could slip between. That is the whole windowing primitive, and it
is deliberately the whole of it — this is a counter generation, not a time-series
store, and there is no history, no retention and no rollup here to get wrong.

The primitive is not reachable over HTTP. Closing a window destroys the evidence a
drift investigation runs on, this service has no authentication, and an anonymous
caller able to do that is a worse trade than asking an operator to restart the
process. It changes no model, no policy, no threshold, no prediction and no
reference profile: every field it touches is an integer this module owns.

**Thread safety.** One collector is shared by every request in the process, and a
histogram update is a read-modify-write. Every mutating path takes an ``RLock``.
Reads take it too, so a snapshot is a consistent picture rather than a mixture of
two windows, and :meth:`~MonitoringCollector.observe` applies all of one call's
updates inside a single acquisition — so an observation lands entirely in the window
that was open when it got the lock, never split across a reset.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from churn.monitoring.distributions import UNSEEN_LEVEL
from churn.monitoring.reference import ReferenceProfile
from churn.monitoring.settings import get_monitoring_policy
from churn.monitoring.structural import STRUCTURAL_RULES, rule_violations
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn.preprocessing.transformers import clean_total_charges

logger = logging.getLogger(__name__)


@dataclass
class _NumericAccumulator:
    """Running aggregates for one numeric feature. Never holds a value."""

    edges: tuple[float, ...]
    reference_min: float
    reference_max: float
    counts: list[int]
    n: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    out_of_range: int = 0

    def observe(self, values: np.ndarray) -> None:
        boundaries = np.asarray(self.edges, dtype=float)
        indices = np.searchsorted(boundaries, values, side="left")
        for index in indices:
            self.counts[int(index)] += 1
        self.n += int(values.size)
        self.total += float(values.sum())
        self.total_squares += float(np.square(values).sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        outside = (values < self.reference_min) | (values > self.reference_max)
        self.out_of_range += int(outside.sum())

    def snapshot(self) -> dict[str, object]:
        mean = self.total / self.n if self.n else 0.0
        variance = max(0.0, (self.total_squares / self.n) - mean * mean) if self.n else 0.0
        return {
            "count": self.n,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum if self.n else 0.0,
            "max": self.maximum if self.n else 0.0,
            "bin_counts": list(self.counts),
            "out_of_reference_range_count": self.out_of_range,
            "out_of_reference_range_rate": (self.out_of_range / self.n) if self.n else 0.0,
        }


@dataclass
class _CategoricalAccumulator:
    """Level counts for one categorical feature, plus a bounded unseen-cardinality counter.

    Two quantities with deliberately different guarantees live here. The **occurrence**
    counters — every entry of ``counts``, and therefore ``unseen_count`` and
    ``unseen_rate`` — are exact under all conditions, because they are integers. The
    **distinct** count is exact only while the digest set is below
    ``max_distinct_unseen``; past that it is a lower bound, and the snapshot says which
    of the two it is instead of leaving a reader to assume.
    """

    known: tuple[str, ...]
    counts: dict[str, int]
    max_distinct_unseen: int
    unseen_digests: set[str] = field(default_factory=set)

    @property
    def saturated(self) -> bool:
        """True once the tracking cap has been reached and the count became a bound."""
        return len(self.unseen_digests) >= self.max_distinct_unseen

    def observe(self, values: pd.Series) -> None:
        text = values.astype("string")
        known = set(self.known)
        for raw in text:
            value = str(raw)
            if value in known:
                self.counts[value] += 1
                continue
            # The occurrence count is exact and is what the alert policy reads. It
            # is incremented before the cap is consulted, so saturation can never
            # cost an occurrence.
            self.counts[UNSEEN_LEVEL] += 1
            if self.saturated:
                # Bounded, and bounded by refusing new entries rather than evicting
                # old ones: an evicting set would report a number that is neither the
                # cardinality nor a bound on it.
                continue
            # A digest, never the value: enough to recognise a repeat, and it keeps
            # the cleartext from being retained anywhere. It is NOT anonymisation —
            # a categorical domain is small enough to hash exhaustively — which is
            # why the set is process-local and never serialised, logged or reported.
            # See the module docstring.
            self.unseen_digests.add(hashlib.sha256(value.encode("utf-8")).hexdigest())

    def snapshot(self) -> dict[str, object]:
        total = sum(self.counts.values())
        unseen = self.counts[UNSEEN_LEVEL]
        saturated = self.saturated
        return {
            "count": total,
            "count_by_level": dict(self.counts),
            # Exact, always.
            "unseen_count": unseen,
            "unseen_rate": (unseen / total) if total else 0.0,
            # Exact below the cap, a lower bound at or above it. The two booleans
            # exist so the number is never ambiguous: reporting the cap as though it
            # were the cardinality is the failure mode this replaces.
            "n_distinct_unseen_observed": len(self.unseen_digests),
            "distinct_unseen_tracking_cap": self.max_distinct_unseen,
            "distinct_unseen_tracking_saturated": saturated,
            "distinct_unseen_is_lower_bound": saturated,
        }


@dataclass
class _ScoreAccumulator:
    """Running aggregates for the model score."""

    edges: tuple[float, ...]
    threshold: float
    counts: list[int]
    n: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    positives: int = 0

    def observe(self, scores: np.ndarray, predictions: np.ndarray) -> None:
        boundaries = np.asarray(self.edges, dtype=float)
        indices = np.searchsorted(boundaries, scores, side="left")
        for index in indices:
            self.counts[int(index)] += 1
        self.n += int(scores.size)
        self.total += float(scores.sum())
        self.total_squares += float(np.square(scores).sum())
        self.positives += int(np.asarray(predictions, dtype=int).sum())

    def snapshot(self) -> dict[str, object]:
        mean = self.total / self.n if self.n else 0.0
        variance = max(0.0, (self.total_squares / self.n) - mean * mean) if self.n else 0.0
        return {
            "count": self.n,
            "mean": mean,
            "std": math.sqrt(variance),
            "bin_counts": list(self.counts),
            "predicted_positive_count": self.positives,
            "predicted_positive_rate": (self.positives / self.n) if self.n else 0.0,
        }


@dataclass(frozen=True)
class WindowSnapshot:
    """An immutable picture of one monitoring window.

    Aggregates only. Nothing here can be traced to a person: there is no
    identifier, no payload, no row-level score and no unseen category text.
    """

    n_records: int
    quality: dict[str, int]
    numeric: dict[str, dict[str, object]]
    categorical: dict[str, dict[str, object]]
    score: dict[str, object]
    structural: dict[str, object]

    def as_record(self) -> dict[str, object]:
        return {
            "n_records": self.n_records,
            "quality": dict(self.quality),
            "numeric": {name: dict(entry) for name, entry in self.numeric.items()},
            "categorical": {name: dict(entry) for name, entry in self.categorical.items()},
            "score": dict(self.score),
            "structural": dict(self.structural),
        }


#: The data-quality counters every window carries.
QUALITY_COUNTERS: tuple[str, ...] = (
    "requests_total",
    "records_total",
    "successful_records",
    "invalid_schema_records",
    "invalid_feature_records",
    "unseen_category_records",
    "structural_violation_records",
)


class MonitoringCollector:
    """Thread-safe incremental aggregation over a window of served records.

    One instance per process, shared by every request. It observes; it never
    decides. Nothing it does can reach the prediction path — see
    :meth:`observe`'s contract — and a failure inside it is the caller's to isolate.
    """

    def __init__(
        self,
        reference: ReferenceProfile,
        max_distinct_unseen_tracked_per_feature: int | None = None,
    ) -> None:
        """Build a collector over one reference profile.

        Args:
            reference: The frozen baseline whose bins and levels every accumulator
                is shaped by. Never modified.
            max_distinct_unseen_tracked_per_feature: Memory-safety bound on the
                per-feature distinct-unseen set. Defaults to the configured
                operational policy. It bounds *state*, never a status: no alert reads
                it, and the occurrence counters stay exact whatever it is.
        """
        self._reference = reference
        self._max_distinct_unseen = (
            max_distinct_unseen_tracked_per_feature
            if max_distinct_unseen_tracked_per_feature is not None
            else get_monitoring_policy().max_distinct_unseen_tracked_per_feature
        )
        self._lock = threading.RLock()
        self._reset_unlocked()

    # -- window lifecycle ---------------------------------------------------

    def _reset_unlocked(self) -> None:
        self._quality: dict[str, int] = dict.fromkeys(QUALITY_COUNTERS, 0)
        self._numeric = {
            feature: _NumericAccumulator(
                edges=tuple(self._reference.numeric[feature].bin_edges),
                reference_min=self._reference.numeric[feature].min,
                reference_max=self._reference.numeric[feature].max,
                counts=[0] * (len(self._reference.numeric[feature].bin_edges) + 1),
            )
            for feature in NUMERIC_FEATURES
        }
        self._categorical = {
            feature: _CategoricalAccumulator(
                known=tuple(self._reference.categorical[feature].known_levels),
                counts={
                    **dict.fromkeys(self._reference.categorical[feature].known_levels, 0),
                    UNSEEN_LEVEL: 0,
                },
                max_distinct_unseen=self._max_distinct_unseen,
            )
            for feature in CATEGORICAL_FEATURES
        }
        self._score = _ScoreAccumulator(
            edges=tuple(self._reference.score.bin_edges),
            threshold=self._reference.score.threshold,
            counts=[0] * (len(self._reference.score.bin_edges) + 1),
        )
        self._violations: dict[str, int] = {rule.name: 0 for rule in STRUCTURAL_RULES}
        self._records_with_violation = 0

    def reset(self) -> WindowSnapshot:
        """Close the current window and start an empty one — snapshot *and* reset.

        The two halves are one operation under one lock acquisition, which is the
        point: taking a snapshot and then clearing would drop every update that
        arrived in between, and on a busy process that is silent data loss in the
        component whose job is to notice things.

        Returns:
            The snapshot of the window that just closed. Nothing observed before the
            call is lost, and nothing observed after it is attributed to the old
            window.

        Every accumulator is rebuilt from the reference rather than cleared in place,
        so a new window starts with fresh histograms, fresh level tables, an **empty
        unseen-digest set and an unsaturated tracking flag**. Saturation is a property
        of one window, not of the process: a burst of new categories cannot leave the
        next window reporting a bound it never hit.

        Note:
            Changes no model, no policy, no threshold, no prediction and no reference
            profile. It is a counter operation and nothing else, and it is not
            exposed over HTTP.
        """
        with self._lock:
            closing = self._snapshot_unlocked()
            self._reset_unlocked()
        logger.info("Monitoring window reset after %d record(s).", closing.n_records)
        return closing

    def snapshot(self) -> WindowSnapshot:
        """Return the current window without closing it."""
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> WindowSnapshot:
        n_records = self._score.n
        return WindowSnapshot(
            n_records=n_records,
            quality=dict(self._quality),
            numeric={name: acc.snapshot() for name, acc in self._numeric.items()},
            categorical={name: acc.snapshot() for name, acc in self._categorical.items()},
            score=self._score.snapshot(),
            structural={
                "violations_by_rule": dict(self._violations),
                "records_with_any_violation": self._records_with_violation,
                "violation_rate": (self._records_with_violation / n_records) if n_records else 0.0,
            },
        )

    # -- observation --------------------------------------------------------

    def observe(
        self,
        features: pd.DataFrame,
        probabilities: Sequence[float],
        predictions: Sequence[int],
    ) -> None:
        """Fold one or more already-scored records into the current window.

        The contract is one-directional: this method reads its arguments and
        mutates counters. It returns nothing, it cannot influence a probability,
        and the caller is expected to have produced the prediction already.

        Args:
            features: The canonical feature matrix the pipeline was given —
                already through ``prepare_features``, so column order is
                canonical and the values are the ones the model saw.
            probabilities: One model score per row.
            predictions: One binary decision per row, from the frozen rule.

        Raises:
            ValueError: If the three arguments disagree in length. Raised rather
                than silently truncated; the caller isolates the failure.
        """
        scores = np.asarray(probabilities, dtype=float)
        decisions = np.asarray(predictions, dtype=int)
        if len(features) != scores.size or scores.size != decisions.size:
            raise ValueError(
                f"observe() got {len(features)} record(s), {scores.size} probability/ies "
                f"and {decisions.size} prediction(s)."
            )
        if scores.size == 0:
            return

        numeric_frame = clean_total_charges(features)
        unseen_mask = np.zeros(len(features), dtype=bool)
        violation_mask = np.zeros(len(features), dtype=bool)

        for feature in CATEGORICAL_FEATURES:
            column = features[feature].astype("string")
            known = set(self._reference.categorical[feature].known_levels)
            unseen_mask |= ~column.isin(known).to_numpy(dtype=bool)

        for rule in STRUCTURAL_RULES:
            broken = rule_violations(features, rule).to_numpy(dtype=bool)
            violation_mask |= broken

        with self._lock:
            for feature in NUMERIC_FEATURES:
                self._numeric[feature].observe(numeric_frame[feature].astype(float).to_numpy())
            for feature in CATEGORICAL_FEATURES:
                self._categorical[feature].observe(features[feature])
            self._score.observe(scores, decisions)
            for rule in STRUCTURAL_RULES:
                self._violations[rule.name] += int(
                    rule_violations(features, rule).to_numpy(dtype=bool).sum()
                )
            self._records_with_violation += int(violation_mask.sum())
            self._quality["successful_records"] += int(scores.size)
            self._quality["unseen_category_records"] += int(unseen_mask.sum())
            self._quality["structural_violation_records"] += int(violation_mask.sum())

    # -- data-quality counters ---------------------------------------------

    def record_request(self, n_records: int) -> None:
        """Count one request and the records it carried, whatever its outcome."""
        with self._lock:
            self._quality["requests_total"] += 1
            self._quality["records_total"] += int(n_records)

    def record_invalid_schema(self, n_records: int = 1) -> None:
        """Count records the request schema rejected before the contract saw them."""
        with self._lock:
            self._quality["invalid_schema_records"] += int(n_records)

    def record_invalid_feature(self, n_records: int = 1) -> None:
        """Count records the frozen feature contract rejected."""
        with self._lock:
            self._quality["invalid_feature_records"] += int(n_records)

    # -- introspection ------------------------------------------------------

    @property
    def reference(self) -> ReferenceProfile:
        """The reference profile this collector's bins came from."""
        return self._reference

    @property
    def max_distinct_unseen_tracked_per_feature(self) -> int:
        """The per-feature bound on retained unseen digests."""
        return self._max_distinct_unseen

    def counters(self) -> Mapping[str, int]:
        """Return the data-quality counters alone."""
        with self._lock:
            return dict(self._quality)


__all__ = ["QUALITY_COUNTERS", "MonitoringCollector", "WindowSnapshot"]
