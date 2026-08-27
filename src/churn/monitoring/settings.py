"""Operational monitoring policy — cutoffs, window size, and what they are not.

Everything in this module answers "when should a human look at this". Nothing in it
answers "what does the model predict". The separation is enforced rather than
described: :class:`MonitoringPolicy` has no field for a decision threshold, a
calibration policy, a positive class or a feature list, so there is no setting to
forget to guard.

One field in it is not a cutoff at all:
``max_distinct_unseen_tracked_per_feature`` bounds how much state the collector may
retain per feature. It is here because it is operational and belongs beside the other
things an operator tunes, but nothing derives a *status* from it — see
:class:`MonitoringPolicy` for what that separation buys.

**The cutoffs are heuristics, and the code says so.** PSI and TVD are descriptive
distances between two histograms. They are not test statistics, they have no null
distribution, and no p-value is being computed anywhere in this package. A window
crossing ``psi_critical`` means "this window stopped resembling the reference
population by this much"; it does not mean the model degraded, and it cannot,
because degradation is a statement about labels and this phase has none.

The 0.10 / 0.25 pair is the value most often quoted in credit-scoring practice. It
is used because operators recognise it, not because it was validated here. It is
loaded from ``configs/monitoring.toml`` — deliberately not from
``configs/base.toml``, which belongs to the frozen provenance.
"""

from __future__ import annotations

import logging
import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Operational policy file. Not part of the frozen provenance.
DEFAULT_MONITORING_CONFIG_PATH = PROJECT_ROOT / "configs" / "monitoring.toml"

#: Where the reference profile is written and read from.
REFERENCE_PROFILE_RELATIVE_PATH = "reports/monitoring/reference_profile.json"

#: **The trust anchor for the monitoring reference profile.**
#:
#: Hashing the file that was just loaded and then logging the result is not integrity
#: validation: it answers "what did I load", never "did I load the right thing". A
#: digest is only a gate when it is compared against an expectation recorded
#: *independently of the file being checked*. This constant is that expectation,
#: pinned here in source, in the same repository and under the same review as the
#: code that enforces it, exactly as ``FREEZE_COMMIT`` and the model fingerprint are
#: pinned for the frozen artefacts.
#:
#: It is enforced at monitoring startup by
#: :meth:`churn.serving.monitoring.ServingMonitor.from_settings`, which passes it to
#: :func:`churn.monitoring.reference.verify_reference_profile` as ``expected_digest``.
#: A mismatch fails the ``profile_digest_matches`` check, raises
#: ``MonitoringStartupError``, and the process does not come up — so monitoring can
#: never reach ``HEALTHY`` on a profile nobody pinned.
#:
#: Rebuilding the reference profile is therefore a deliberate two-line change: the
#: artefact and this constant move together, in one commit, or startup fails.
EXPECTED_REFERENCE_PROFILE_SHA256 = (
    "6fef9f81a6dd03c94d4f2b7ebbd143b24c4e225eb2c6001ee1be68c4db0c531b"
)

#: Where the expected digest comes from, named in the monitoring response so an
#: operator reading it never has to guess what the comparison was against.
REFERENCE_TRUST_ANCHOR = "churn.monitoring.settings.EXPECTED_REFERENCE_PROFILE_SHA256"

#: Route the serving layer publishes when monitoring is enabled. Declared here so
#: the Phase 12 record can name it without importing a web framework.
MONITORING_ENDPOINT = "/api/v1/monitoring"

#: Statuses a monitoring signal can carry. Ordered from least to most severe;
#: ``INSUFFICIENT_DATA`` is deliberately *not* on that scale — it is the absence of
#: a verdict, not a mild one.
STATUS_OK = "OK"
STATUS_WARNING = "WARNING"
STATUS_CRITICAL = "CRITICAL"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

SEVERITY: dict[str, int] = {STATUS_OK: 0, STATUS_WARNING: 1, STATUS_CRITICAL: 2}


class _Cutoffs(BaseModel):
    """A warning/critical pair. Both are heuristics; neither is a significance level."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    warning: float = Field(ge=0.0)
    critical: float = Field(ge=0.0)

    def status(self, value: float) -> str:
        """Return the operational status this value falls into."""
        if value >= self.critical:
            return STATUS_CRITICAL
        if value >= self.warning:
            return STATUS_WARNING
        return STATUS_OK


class MonitoringPolicy(BaseModel):
    """Every operational knob monitoring has, and nothing the model has.

    Frozen after construction: it is read while comparing a window and is never
    rebuilt per record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum_window_size: int = Field(ge=1)
    max_distinct_unseen_tracked_per_feature: int = Field(
        ge=1,
        description=(
            "Memory-safety bound on the per-feature distinct-unseen set. Not an alert "
            "cutoff: no status is derived from it, and unseen_count and unseen_rate "
            "stay exact regardless of it."
        ),
    )
    numeric_psi: _Cutoffs
    numeric_out_of_range: _Cutoffs
    categorical_tvd: _Cutoffs
    categorical_unseen_rate: _Cutoffs
    score_psi: _Cutoffs
    score_positive_rate_delta: _Cutoffs
    structural_violation_rate: _Cutoffs

    def as_record(self) -> dict[str, object]:
        """Return the policy as plain data, labelled for the machine-readable record."""
        return {
            "classification": "OPERATIONAL_MONITORING_POLICY",
            "is_statistical_significance": False,
            "estimated_from_data": False,
            "note": (
                "heuristic cutoffs chosen before any production data existed; PSI and TVD "
                "are descriptive distances, not test statistics, and crossing a cutoff is "
                "never evidence of model degradation"
            ),
            "minimum_window_size": self.minimum_window_size,
            "max_distinct_unseen_tracked_per_feature": (
                self.max_distinct_unseen_tracked_per_feature
            ),
            "numeric_psi": self.numeric_psi.model_dump(),
            "numeric_out_of_range": self.numeric_out_of_range.model_dump(),
            "categorical_tvd": self.categorical_tvd.model_dump(),
            "categorical_unseen_rate": self.categorical_unseen_rate.model_dump(),
            "score_psi": self.score_psi.model_dump(),
            "score_positive_rate_delta": self.score_positive_rate_delta.model_dump(),
            "structural_violation_rate": self.structural_violation_rate.model_dump(),
        }


def load_monitoring_policy(path: Path | None = None) -> MonitoringPolicy:
    """Read and validate the operational monitoring policy.

    Raises:
        FileNotFoundError: If the configuration file is absent.
        KeyError: If a required section or key is missing.
        pydantic.ValidationError: If a value violates the contract.
    """
    source = path or DEFAULT_MONITORING_CONFIG_PATH
    with source.open("rb") as handle:
        raw = tomllib.load(handle)

    return MonitoringPolicy(
        minimum_window_size=raw["window"]["minimum_window_size"],
        max_distinct_unseen_tracked_per_feature=raw["categorical"][
            "max_distinct_unseen_tracked_per_feature"
        ],
        numeric_psi=_Cutoffs(
            warning=raw["numeric"]["psi_warning"], critical=raw["numeric"]["psi_critical"]
        ),
        numeric_out_of_range=_Cutoffs(
            warning=raw["numeric"]["out_of_range_warning"],
            critical=raw["numeric"]["out_of_range_critical"],
        ),
        categorical_tvd=_Cutoffs(
            warning=raw["categorical"]["tvd_warning"],
            critical=raw["categorical"]["tvd_critical"],
        ),
        categorical_unseen_rate=_Cutoffs(
            warning=raw["categorical"]["unseen_rate_warning"],
            critical=raw["categorical"]["unseen_rate_critical"],
        ),
        score_psi=_Cutoffs(
            warning=raw["score"]["psi_warning"], critical=raw["score"]["psi_critical"]
        ),
        score_positive_rate_delta=_Cutoffs(
            warning=raw["score"]["positive_rate_delta_warning"],
            critical=raw["score"]["positive_rate_delta_critical"],
        ),
        structural_violation_rate=_Cutoffs(
            warning=raw["structural"]["violation_rate_warning"],
            critical=raw["structural"]["violation_rate_critical"],
        ),
    )


@lru_cache(maxsize=1)
def get_monitoring_policy() -> MonitoringPolicy:
    """Return the default operational policy, loaded once per process."""
    return load_monitoring_policy()


def worst(statuses: list[str]) -> str:
    """Return the most severe status in a list, ignoring ``INSUFFICIENT_DATA``.

    Absence of a verdict does not outrank a verdict, and it does not soften one:
    a window with too few records reports :data:`STATUS_INSUFFICIENT_DATA` at the
    top level, and this helper is only reached once that is settled.
    """
    ranked = [status for status in statuses if status in SEVERITY]
    if not ranked:
        return STATUS_OK
    return max(ranked, key=lambda status: SEVERITY[status])


def default_reference_profile_path(root: Path | None = None) -> Path:
    """Return the repository's reference-profile location."""
    return (root or PROJECT_ROOT) / REFERENCE_PROFILE_RELATIVE_PATH


__all__ = [
    "DEFAULT_MONITORING_CONFIG_PATH",
    "EXPECTED_REFERENCE_PROFILE_SHA256",
    "MONITORING_ENDPOINT",
    "REFERENCE_TRUST_ANCHOR",
    "REFERENCE_PROFILE_RELATIVE_PATH",
    "SEVERITY",
    "STATUS_CRITICAL",
    "STATUS_INSUFFICIENT_DATA",
    "STATUS_OK",
    "STATUS_WARNING",
    "MonitoringPolicy",
    "default_reference_profile_path",
    "get_monitoring_policy",
    "load_monitoring_policy",
    "worst",
]
