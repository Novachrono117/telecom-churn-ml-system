"""Wiring the monitoring collector into the serving process, safely.

This module is the seam between two things that must stay separable: a prediction
boundary that is authoritative, and an observer that is not. It owns three
responsibilities and refuses a fourth.

**It verifies the reference against a pinned expectation before anything observes
with it.** The profile is loaded, validated against its schema and its invariants,
and its digest is compared with
:data:`~churn.monitoring.settings.EXPECTED_REFERENCE_PROFILE_SHA256` — a constant in
source, recorded independently of the file being checked. That comparison is what
makes the digest a gate: hashing whatever was loaded and logging the result answers
"what did I load", never "did I load the right thing". A profile that does not match
stops the startup, so monitoring can never reach ``HEALTHY`` on an unpinned
baseline. Same fail-closed posture as the frozen model, for the same reason: a drift
number computed against an unknown baseline is worse than no number, because it
looks like one.

**It guarantees the observer cannot raise into the prediction path.** Every
:meth:`ServingMonitor.observe` call is wrapped; a failure increments a counter, is
logged, and returns normally. The prediction path guards again on its own side —
two catches, and they are not redundant: the inner one exists to *count* the failure
so it can be reported, the outer one exists to *guarantee* the prediction, and
neither can be removed without losing something.

**The failure log carries a type and nothing else.** An exception raised while
observing has been handed feature values, and ``str(error)`` — or a traceback frame,
which renders the same message — is text the request's own data can reach into.
``ValueError: 'Instant transfer (PIX)' is not a known level`` written to a log file
is the payload leaking through the observability layer, which is the one place
nobody audits for it. So the failure is logged as an event name, an exception type
and a counter: enough to know the collector broke and roughly where, with no field
whose contents the caller controls.

**It degrades visibly.** Failures do not vanish. The monitoring status turns
``DEGRADED`` and stays there for the life of the window, which is what makes a
silently broken collector impossible to mistake for a quiet one.

What it refuses: it never makes the service NOT READY. Readiness is a claim about
the *artefact* — is the frozen model loaded and verified. Drift is a claim about the
*population*, and a shifted population is not a corrupt model. Conflating them would
mean a load balancer pulling a perfectly functioning instance out of rotation
because customers changed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from churn.monitoring.reference import (
    ReferenceProfileError,
    load_reference_profile,
    profile_digest,
    raise_on_failure,
    verify_reference_profile,
)
from churn.monitoring.service import MonitoringReport, MonitoringService
from churn.monitoring.settings import (
    EXPECTED_REFERENCE_PROFILE_SHA256,
    REFERENCE_TRUST_ANCHOR,
    STATUS_INSUFFICIENT_DATA,
    MonitoringPolicy,
    default_reference_profile_path,
    get_monitoring_policy,
)
from churn.serving.errors import ServingStartupError
from churn.serving.settings import ServingSettings

logger = logging.getLogger(__name__)

#: Reported when the collector has raised at least once in this window.
HEALTH_DEGRADED = "DEGRADED"
#: Reported when the collector has been working.
HEALTH_HEALTHY = "HEALTHY"
#: Reported when monitoring is not enabled on this process.
HEALTH_DISABLED = "DISABLED"


class MonitoringStartupError(ServingStartupError):
    """The monitoring reference is absent, malformed, or not the one expected."""


#: Logged instead of a message. The event name is a constant, the only variable part
#: is an exception *type*, and neither can carry a value the request supplied.
OBSERVER_FAILURE_EVENT = "monitoring_observer_failure"


@dataclass
class _Failures:
    """A mutable failure tally, held by an otherwise immutable monitor.

    ``last_error`` holds an exception **type name** — ``"ValueError"`` — never a
    message. The message of an exception raised inside the collector is derived from
    the record being observed, so it is the one thing that must not be kept.
    """

    count: int = 0
    last_error: str = ""


@dataclass(frozen=True)
class ServingMonitor:
    """The process-wide observer. Implements ``PredictionObserver``.

    Frozen, shared by every request, and incapable of raising into a caller.
    """

    service: MonitoringService
    failures: _Failures = field(default_factory=_Failures)

    @classmethod
    def from_settings(
        cls,
        settings: ServingSettings,
        policy: MonitoringPolicy | None = None,
    ) -> ServingMonitor:
        """Load and verify the reference profile, then build the observer.

        Raises:
            MonitoringStartupError: If the profile is absent, unparseable, or fails
                an invariant. Monitoring does not start on an unverified baseline.
        """
        path = settings.reference_profile_path or default_reference_profile_path()
        try:
            reference = load_reference_profile(path)
        except FileNotFoundError as error:
            raise MonitoringStartupError(
                "The monitoring reference profile is absent, so there is no baseline to "
                "compare production against. Monitoring does not start and does not "
                "build one: a reference invented at serving time would describe "
                "production, not the population the model was built for."
            ) from error
        except Exception as error:  # noqa: BLE001 - any parse failure is a failed gate
            raise MonitoringStartupError(
                f"The monitoring reference profile does not parse "
                f"({type(error).__name__}). A file that is not a reference profile "
                "cannot be monitored against."
            ) from error

        # The pinned digest is passed in, so `profile_digest_matches` is checked
        # alongside every structural invariant rather than after them. Without an
        # expectation to compare against, computing the digest would establish
        # nothing: it is the pin that turns a hash into a gate.
        try:
            raise_on_failure(verify_reference_profile(reference, EXPECTED_REFERENCE_PROFILE_SHA256))
        except ReferenceProfileError as error:
            raise MonitoringStartupError(str(error)) from error

        digest = profile_digest(reference)
        logger.info(
            "Monitoring reference verified against %s: sha256=%s n=%d population=%s",
            REFERENCE_TRUST_ANCHOR,
            digest[:16],
            reference.provenance.n_reference,
            reference.provenance.reference_population,
        )
        return cls(
            service=MonitoringService.from_reference(
                reference, digest, policy or get_monitoring_policy()
            )
        )

    # -- observation --------------------------------------------------------

    def observe(
        self,
        features: pd.DataFrame,
        probabilities: Sequence[float],
        predictions: Sequence[int],
    ) -> None:
        """Fold scored records into the window. Never raises."""
        try:
            self.service.collector.observe(features, probabilities, predictions)
        except Exception as error:  # noqa: BLE001 - counted, logged, never propagated
            self._note_failure("observe", error)

    def record_request(self, n_records: int) -> None:
        """Count one request and the records it carried. Never raises."""
        try:
            self.service.collector.record_request(n_records)
        except Exception as error:  # noqa: BLE001
            self._note_failure("record_request", error)

    def record_rejection(self, kind: str, n_records: int = 1) -> None:
        """Count a rejected record. Never raises.

        Args:
            kind: ``"schema"`` when the request schema refused the payload,
                ``"feature"`` when the frozen feature contract did.
            n_records: How many records the rejection covered. A body that failed to
                parse yields no record count, so it is counted as one; that
                under-counts a rejected batch, and the report says so.
        """
        try:
            collector = self.service.collector
            collector.record_request(n_records)
            if kind == "schema":
                collector.record_invalid_schema(n_records)
            else:
                collector.record_invalid_feature(n_records)
        except Exception as error:  # noqa: BLE001
            self._note_failure("record_rejection", error)

    def _note_failure(self, stage: str, error: BaseException) -> None:
        """Count and log one collector failure without echoing anything it was given.

        Deliberately **not** ``logger.exception``. A traceback renders
        ``str(error)``, and an exception raised inside the collector was raised
        while holding feature values, so its message is attacker- and
        customer-controlled text. What is logged instead is a fixed event name, the
        stage, and the exception's *type*: enough to tell a broken histogram from a
        broken counter and to find the code path, with no field the request can
        write into.

        The stack trace is not lost so much as traded: the failure is counted, the
        monitoring status degrades visibly, and a developer reproducing it locally
        gets the full traceback from the collector's own unit tests, where the
        records are synthetic.
        """
        self.failures.count += 1
        self.failures.last_error = type(error).__name__
        logger.error(
            "%s stage=%s exception_type=%s failures=%d",
            OBSERVER_FAILURE_EVENT,
            stage,
            type(error).__name__,
            self.failures.count,
        )

    # -- reporting ----------------------------------------------------------

    @property
    def reference_sha256(self) -> str:
        """Digest of the reference profile this monitor compares against."""
        return self.service.reference_sha256

    @property
    def health(self) -> str:
        """``HEALTHY`` or ``DEGRADED`` — about the collector, never about drift."""
        return HEALTH_DEGRADED if self.failures.count else HEALTH_HEALTHY

    def report(self) -> MonitoringReport:
        """Compare the current window with the reference, without closing it."""
        return self.service.report()

    def as_response(self) -> dict[str, object]:
        """Return the aggregate payload the monitoring endpoint answers with.

        Aggregates only: counts, histograms, rates and statuses. No payload, no
        feature value, no row-level score, no identifier, and no unseen category
        text — nor any digest of one — because the collector never had them to give.

        Below the operational minimum the distribution sections are absent entirely:
        see :func:`churn.monitoring.service._suppressed_report`.
        """
        report = self.report()
        record = report.as_record()
        record["collector_health"] = self.health
        record["collector_failures"] = self.failures.count
        record["monitoring_enabled"] = True
        record["reference_population"] = self.service.reference.provenance.reference_population
        record["reference_n"] = self.service.reference.provenance.n_reference
        record["window_has_verdict"] = report.has_verdict
        # Both sides of the startup gate, named. A reader should never have to guess
        # what the reported digest was compared against, or whether it was compared
        # at all.
        record["expected_reference_profile_sha256"] = EXPECTED_REFERENCE_PROFILE_SHA256
        record["reference_profile_sha256_source"] = REFERENCE_TRUST_ANCHOR
        record["rejection_counting_note"] = (
            "A payload rejected by the request schema is counted as one record, because "
            "a body that failed to parse carries no record count. This under-counts a "
            "rejected batch."
        )
        return record


def disabled_response() -> dict[str, object]:
    """Return the payload for a process where monitoring is not enabled."""
    return {
        "monitoring_enabled": False,
        "collector_health": HEALTH_DISABLED,
        "status": STATUS_INSUFFICIENT_DATA,
        "detail": (
            "Monitoring is not enabled on this process. Set CHURN_SERVING_MONITORING=1 "
            "to collect aggregates. Predictions are identical either way."
        ),
    }


__all__ = [
    "HEALTH_DEGRADED",
    "HEALTH_DISABLED",
    "HEALTH_HEALTHY",
    "OBSERVER_FAILURE_EVENT",
    "MonitoringStartupError",
    "ServingMonitor",
    "disabled_response",
]
