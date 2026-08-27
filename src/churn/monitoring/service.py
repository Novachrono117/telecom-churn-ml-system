"""Comparing a window with the reference, and saying only what that supports.

Four different questions get four different answers here, and the report keeps them
apart because conflating them is the classic monitoring failure:

* **data quality** — is the input well formed? Counters, no distributions.
* **data drift** — has ``P(X)`` moved? PSI per numeric feature, TVD per categorical.
* **prediction drift** — has the score distribution moved? PSI, plus the change in
  the share of records the frozen rule marks positive.
* **performance degradation** — has the model got worse? **Not answered.** It cannot
  be, without labels. No field in this module reports accuracy, recall, precision,
  AP, ROC-AUC or a confusion matrix, and none is computed anywhere.

The last point is the one worth being pedantic about. A window can be CRITICAL on
every drift signal and the model can still be making exactly the decisions the
business wants; a window can be OK on all of them while the model quietly rots. The
statuses here mean "someone should look", not "the model is wrong".

**A window too small to judge is not a quiet OK.** Below the configured minimum, the
report is :data:`~churn.monitoring.settings.STATUS_INSUFFICIENT_DATA`: the counts
are all still there, and no drift verdict is claimed. That is deliberately not the
lowest rung of the OK/WARNING/CRITICAL ladder — it is the absence of a rung.

**A small window also has every detailed distribution suppressed, and that is a
privacy control rather than a cosmetic one.** The obvious leak is an order
statistic: a mean over one record *is* that record's value, and so are its min and
its max. The subtler leak is that *any* per-feature breakdown of a one-record window
is that record. ``{"Month-to-month": 1}`` names the customer's contract exactly.
``bin_counts = [0, 1, 0, ...]`` places their tenure in a decile. A per-rule
structural detail says which product equivalence they broke, and a
``predicted_positive_rate`` of ``1.0`` over one record *is* that customer's decision.
None of those is an aggregate; each is a single record wearing a histogram's
clothing. On a low-traffic deployment an unauthenticated endpoint polled after every
request would reconstruct customers one at a time.

So the rule is a policy, not a per-field judgement call: below
``minimum_window_size`` the status is ``INSUFFICIENT_DATA`` and **no detailed
distribution is computed at all** — not suppressed after the fact, never built. What
survives is the window's size, the operational counters the deployment needs to know
it is alive, and the statuses. Above the minimum every distribution is reported
normally: a histogram over a hundred records is an aggregate.

The suppressed report is not a degraded one. ``details_suppressed`` says plainly why
the sections are absent, so an operator reads "too small to expose" rather than
"nothing happened".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from churn.monitoring.collector import MonitoringCollector, WindowSnapshot
from churn.monitoring.distributions import (
    UNSEEN_LEVEL,
    NumericSummary,
    frequencies,
    mean_shift_in_reference_sd,
)
from churn.monitoring.drift import delta, population_stability_index, total_variation_distance
from churn.monitoring.reference import ReferenceProfile, profile_digest
from churn.monitoring.settings import (
    STATUS_INSUFFICIENT_DATA,
    STATUS_OK,
    MonitoringPolicy,
    get_monitoring_policy,
    worst,
)
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger(__name__)

#: What this phase can and cannot conclude, carried in every report so the
#: distinction survives being copied into a ticket.
INTERPRETATION_NOTE = (
    "Drift means the population stopped resembling the reference. It is NOT evidence "
    "that the model degraded: that is a claim about outcomes and requires ground-truth "
    "labels, which this phase does not have. A change in predicted_positive_rate means "
    "the system is assigning positive decisions to a different share of requests, not "
    "that a different share of customers is churning."
)

#: Why a small window reports no distribution at all, not just a softer verdict.
SMALL_WINDOW_NOTE = (
    "This window holds fewer records than the operational minimum, so every detailed "
    "distribution is suppressed: per-feature histograms, category counts and "
    "frequencies, unseen counts and rates, the score histogram and its summaries, the "
    "predicted-positive rate, per-feature drift metrics and per-rule structural "
    "details are all withheld. Over a handful of records those are not aggregates — a "
    "single count against a single level identifies the record that produced it. The "
    "window size, the global operational counters and the statuses are reported; the "
    "distributions become available once the window reaches the minimum, which is also "
    "the point at which they become interpretable."
)

#: The keys a suppressed report answers with in place of a distribution section.
SUPPRESSED_SECTIONS: tuple[str, ...] = (
    "feature_drift",
    "prediction_drift",
    "structural_consistency",
)

#: What the distinct-unseen diagnostic means, carried in every report so the
#: distinction survives being copied into a ticket alongside the number.
UNSEEN_CARDINALITY_NOTE = (
    "Distinct unseen cardinality is exact until the configured tracking cap is reached. "
    "After saturation, the reported value is a lower bound; occurrence counts and unseen "
    "rates remain exact. The cap bounds memory, not privacy, and no alert status is "
    "derived from the distinct count -- unseen_rate is the monitored signal."
)

#: The score is the frozen model's raw output; Phase 9A froze calibration to NONE.
CALIBRATION_NOTE = (
    "calibration_policy is NONE, so a score of 0.70 is a ranking position, not a "
    "70 % chance of churn. Read the score distribution accordingly."
)


@dataclass(frozen=True)
class Signal:
    """One monitored quantity, its value, and the status the policy assigns it."""

    name: str
    value: float
    status: str
    detail: str = ""

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"value": self.value, "status": self.status}
        if self.detail:
            record["detail"] = self.detail
        return record


@dataclass(frozen=True)
class MonitoringReport:
    """A window, compared with the reference, with the verdicts kept separate."""

    status: str
    n_records: int
    minimum_window_size: int
    reference_profile_sha256: str
    quality: dict[str, int]
    numeric: dict[str, dict[str, Any]]
    categorical: dict[str, dict[str, Any]]
    score: dict[str, Any]
    structural: dict[str, Any]
    section_status: dict[str, str]
    details_suppressed: bool = False

    @property
    def has_verdict(self) -> bool:
        """False when the window was too small for any drift claim."""
        return self.status != STATUS_INSUFFICIENT_DATA

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "status": self.status,
            "n_records": self.n_records,
            "minimum_window_size": self.minimum_window_size,
            "reference_profile_sha256": self.reference_profile_sha256,
            "details_suppressed": self.details_suppressed,
            "section_status": dict(self.section_status),
            # Global operational counters. They describe the deployment — how many
            # requests arrived, how many records were rejected and in which class —
            # and never a feature value, a level or a score, so they stay visible at
            # every window size. Without them a small window would be indistinguishable
            # from a dead process.
            "data_quality": dict(self.quality),
            "performance_degradation": {
                "evaluated": False,
                "reason": "no production ground-truth labels are available in this phase",
            },
            "interpretation": INTERPRETATION_NOTE,
            "calibration": CALIBRATION_NOTE,
            "small_window_note": SMALL_WINDOW_NOTE if self.details_suppressed else None,
        }
        if self.details_suppressed:
            # Explicitly null rather than an empty or partial structure: there is no
            # shape here to be probed, and a reader cannot mistake absence for zero.
            record.update(dict.fromkeys(SUPPRESSED_SECTIONS))
            return record

        record["feature_drift"] = {
            "numeric": {name: dict(entry) for name, entry in self.numeric.items()},
            "categorical": {name: dict(entry) for name, entry in self.categorical.items()},
        }
        record["prediction_drift"] = dict(self.score)
        record["structural_consistency"] = dict(self.structural)
        # Static prose, but it explains a field that only exists in this branch, so
        # it appears only where that field does. A suppressed window says nothing
        # about unseen cardinality at all — not even what the number would mean.
        record["unseen_cardinality"] = UNSEEN_CARDINALITY_NOTE
        return record


def _numeric_reference_summary(profile: ReferenceProfile, feature: str) -> NumericSummary:
    entry = profile.numeric[feature]
    return NumericSummary(
        count=entry.count,
        mean=entry.mean,
        std=entry.std,
        minimum=entry.min,
        q01=entry.q01,
        q05=entry.q05,
        q25=entry.q25,
        median=entry.median,
        q75=entry.q75,
        q95=entry.q95,
        q99=entry.q99,
        maximum=entry.max,
    )


def _suppressed_report(
    snapshot: WindowSnapshot,
    policy: MonitoringPolicy,
    digest: str,
) -> MonitoringReport:
    """Report a window too small to expose, without computing what it would expose.

    Nothing per-feature, per-level, per-bin or per-rule is calculated here — not
    computed and then dropped, but never built, so there is no code path on which a
    detail could survive into the response by accident.

    What is kept is deliberately the set that describes the *deployment* rather than
    a customer: how many records the window holds, how large it must be before
    distributions are shown, and the global data-quality counters. Those say
    "traffic is arriving and n of it was rejected"; none of them can be narrowed to
    a feature, a level, a bin or a score.
    """
    return MonitoringReport(
        status=STATUS_INSUFFICIENT_DATA,
        n_records=snapshot.n_records,
        minimum_window_size=policy.minimum_window_size,
        reference_profile_sha256=digest,
        quality=dict(snapshot.quality),
        numeric={},
        categorical={},
        score={},
        structural={},
        section_status=dict.fromkeys(
            (
                "data_quality",
                "feature_drift",
                "prediction_drift",
                "structural_consistency",
                "performance_degradation",
            ),
            STATUS_INSUFFICIENT_DATA,
        ),
        details_suppressed=True,
    )


def compare(
    snapshot: WindowSnapshot,
    reference: ReferenceProfile,
    policy: MonitoringPolicy | None = None,
    reference_digest: str | None = None,
) -> MonitoringReport:
    """Compare one window snapshot with the reference profile.

    Args:
        snapshot: Aggregates of the window. Never contains a record.
        reference: The frozen reference profile.
        policy: Operational cutoffs. Defaults to the configured policy.
        reference_digest: Digest of ``reference``; recomputed when omitted.

    Returns:
        A :class:`MonitoringReport`. When the window holds fewer records than the
        policy's minimum, the status is ``INSUFFICIENT_DATA``, the global
        operational counters are still reported, and every detailed distribution is
        suppressed — see :func:`_suppressed_report`.
    """
    resolved = policy or get_monitoring_policy()
    digest = reference_digest or profile_digest(reference)
    n = snapshot.n_records
    if n < resolved.minimum_window_size:
        return _suppressed_report(snapshot, resolved, digest)

    numeric: dict[str, dict[str, Any]] = {}
    for feature in NUMERIC_FEATURES:
        window = snapshot.numeric[feature]
        entry = reference.numeric[feature]
        summary = _numeric_reference_summary(reference, feature)
        psi = population_stability_index(list(window["bin_counts"]), list(entry.bin_counts))
        out_of_range = float(window["out_of_reference_range_rate"])
        signals = {
            "psi": Signal("psi", psi, resolved.numeric_psi.status(psi)),
            "out_of_reference_range_rate": Signal(
                "out_of_reference_range_rate",
                out_of_range,
                resolved.numeric_out_of_range.status(out_of_range),
            ),
        }
        numeric[feature] = {
            "count": window["count"],
            "mean": float(window["mean"]),
            "std": float(window["std"]),
            "min": float(window["min"]),
            "max": float(window["max"]),
            "reference_mean": entry.mean,
            "reference_std": entry.std,
            "mean_shift_in_reference_sd": mean_shift_in_reference_sd(
                float(window["mean"]), summary
            ),
            "psi": signals["psi"].as_record(),
            "out_of_reference_range_rate": signals["out_of_reference_range_rate"].as_record(),
            "bin_counts": list(window["bin_counts"]),
            "status": worst([signal.status for signal in signals.values()]),
        }

    categorical: dict[str, dict[str, Any]] = {}
    for feature in CATEGORICAL_FEATURES:
        window = snapshot.categorical[feature]
        entry = reference.categorical[feature]
        window_freq = frequencies(dict(window["count_by_level"]))
        reference_freq = {**entry.frequency_by_level, UNSEEN_LEVEL: 0.0}
        tvd = total_variation_distance(window_freq, reference_freq)
        unseen_rate = float(window["unseen_rate"])
        signals = {
            "tvd": Signal("tvd", tvd, resolved.categorical_tvd.status(tvd)),
            "unseen_rate": Signal(
                "unseen_rate",
                unseen_rate,
                resolved.categorical_unseen_rate.status(unseen_rate),
            ),
        }
        categorical[feature] = {
            "count": window["count"],
            "tvd": signals["tvd"].as_record(),
            # The alert reads this, and only this, for unseen categories. It is
            # exact at every tracking cap because it counts occurrences, not
            # distinct values — so saturating the diagnostic below cannot move a
            # status.
            "unseen_rate": signals["unseen_rate"].as_record(),
            "unseen_count": window["unseen_count"],
            # Diagnostic, and explicitly qualified. Exact below the cap; a lower
            # bound at or above it. No status is derived from any of the three.
            "n_distinct_unseen_observed": window["n_distinct_unseen_observed"],
            "distinct_unseen_tracking_cap": window["distinct_unseen_tracking_cap"],
            "distinct_unseen_tracking_saturated": window["distinct_unseen_tracking_saturated"],
            "distinct_unseen_is_lower_bound": window["distinct_unseen_is_lower_bound"],
            "count_by_level": dict(window["count_by_level"]),
            "status": worst([signal.status for signal in signals.values()]),
        }

    window_score = snapshot.score
    score_psi = population_stability_index(
        list(window_score["bin_counts"]), list(reference.score.bin_counts)
    )
    positive_rate = float(window_score["predicted_positive_rate"])
    rate_delta = delta(positive_rate, reference.score.predicted_positive_rate)
    score_signals = {
        "psi": Signal("psi", score_psi, resolved.score_psi.status(score_psi)),
        "predicted_positive_rate_delta": Signal(
            "predicted_positive_rate_delta",
            rate_delta,
            resolved.score_positive_rate_delta.status(abs(rate_delta)),
        ),
    }
    score = {
        "count": window_score["count"],
        "mean": float(window_score["mean"]),
        "std": float(window_score["std"]),
        "reference_mean": reference.score.mean,
        "reference_median": reference.score.median,
        "mean_delta": delta(float(window_score["mean"]), reference.score.mean),
        "psi": score_signals["psi"].as_record(),
        "predicted_positive_rate": positive_rate,
        "reference_predicted_positive_rate": reference.score.predicted_positive_rate,
        "predicted_positive_rate_delta": score_signals["predicted_positive_rate_delta"].as_record(),
        "threshold": reference.score.threshold,
        "comparison": reference.score.comparison,
        "calibration_policy": reference.score.calibration_policy,
        "bin_counts": list(window_score["bin_counts"]),
        "status": worst([signal.status for signal in score_signals.values()]),
    }

    violation_rate = float(snapshot.structural["violation_rate"])
    structural = {
        "violations_by_rule": dict(snapshot.structural["violations_by_rule"]),
        "records_with_any_violation": snapshot.structural["records_with_any_violation"],
        "violation_rate": violation_rate,
        "status": resolved.structural_violation_rate.status(violation_rate),
        "note": (
            "These equivalences are watched, not enforced. The serving contract still "
            "accepts a violating record and still scores it."
        ),
    }

    section_status = {
        "data_quality": STATUS_OK,
        "feature_drift": worst(
            [entry["status"] for entry in numeric.values()]
            + [entry["status"] for entry in categorical.values()]
        ),
        "prediction_drift": score["status"],
        "structural_consistency": structural["status"],
        "performance_degradation": STATUS_INSUFFICIENT_DATA,
    }
    overall = worst(
        [
            section_status["feature_drift"],
            section_status["prediction_drift"],
            section_status["structural_consistency"],
        ]
    )

    return MonitoringReport(
        status=overall,
        n_records=n,
        minimum_window_size=resolved.minimum_window_size,
        reference_profile_sha256=digest,
        quality=dict(snapshot.quality),
        numeric=numeric,
        categorical=categorical,
        score=score,
        structural=structural,
        section_status=section_status,
    )


@dataclass(frozen=True)
class MonitoringService:
    """Holds the verified reference, the collector and the operational policy.

    Shared by every request in the process. It observes and reports; it decides
    nothing about a prediction, and it has no access to a model, a threshold or a
    dataset — the numbers it compares against all come out of the reference profile
    it was handed at startup.
    """

    reference: ReferenceProfile
    reference_sha256: str
    collector: MonitoringCollector
    policy: MonitoringPolicy

    @classmethod
    def from_reference(
        cls,
        reference: ReferenceProfile,
        reference_sha256: str,
        policy: MonitoringPolicy | None = None,
    ) -> MonitoringService:
        """Build a service around an already verified reference profile."""
        resolved = policy or get_monitoring_policy()
        return cls(
            reference=reference,
            reference_sha256=reference_sha256,
            # One policy governs both the cutoffs and the collector's memory bound,
            # so a deployment cannot end up comparing against one policy while
            # collecting under another.
            collector=MonitoringCollector(
                reference, resolved.max_distinct_unseen_tracked_per_feature
            ),
            policy=resolved,
        )

    def report(self) -> MonitoringReport:
        """Compare the current window with the reference, without closing it."""
        return compare(
            self.collector.snapshot(),
            self.reference,
            self.policy,
            self.reference_sha256,
        )

    def close_window(self) -> MonitoringReport:
        """Close the current window and report on it.

        Changes no model, no policy and no threshold: this resets counters.
        """
        return compare(
            self.collector.reset(),
            self.reference,
            self.policy,
            self.reference_sha256,
        )


__all__ = [
    "CALIBRATION_NOTE",
    "INTERPRETATION_NOTE",
    "SMALL_WINDOW_NOTE",
    "SUPPRESSED_SECTIONS",
    "UNSEEN_CARDINALITY_NOTE",
    "MonitoringReport",
    "MonitoringService",
    "Signal",
    "compare",
]
