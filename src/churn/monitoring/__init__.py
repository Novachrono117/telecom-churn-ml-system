"""Phase 12 — monitoring, data quality and drift detection for the frozen model.

The package answers one question: **do the records arriving in production still look
like the population this system was built for?** It answers it in four separate
registers, and never lets them blur:

* data quality — is the input well formed;
* data drift — has ``P(X)`` moved;
* prediction drift — has the score distribution moved;
* performance degradation — **not answered**, and cannot be, without labels.

It trains nothing, changes no threshold, recalibrates nothing and never touches the
holdout. Its only inputs are the frozen reference profile and the records the
serving boundary already scored.

Layering::

    build        churn.monitoring.build         offline, data handed in, never loaded
       ↓
    reference    churn.monitoring.reference     the frozen aggregate profile
       ↓
    collector    churn.monitoring.collector     incremental, privacy-safe, thread-safe
       ↓
    service      churn.monitoring.service       window vs. reference, four registers

Importing this package pulls in no web framework and no dataset loader.
"""

from churn.monitoring.collector import MonitoringCollector, WindowSnapshot
from churn.monitoring.reference import (
    ReferenceProfile,
    ReferenceProfileError,
    load_reference_profile,
    profile_digest,
    verify_reference_profile,
)
from churn.monitoring.service import MonitoringReport, MonitoringService, compare
from churn.monitoring.settings import (
    MONITORING_ENDPOINT,
    STATUS_CRITICAL,
    STATUS_INSUFFICIENT_DATA,
    STATUS_OK,
    STATUS_WARNING,
    MonitoringPolicy,
    default_reference_profile_path,
    get_monitoring_policy,
)
from churn.monitoring.structural import STRUCTURAL_RULES

__all__ = [
    "MONITORING_ENDPOINT",
    "STATUS_CRITICAL",
    "STATUS_INSUFFICIENT_DATA",
    "STATUS_OK",
    "STATUS_WARNING",
    "STRUCTURAL_RULES",
    "MonitoringCollector",
    "MonitoringPolicy",
    "MonitoringReport",
    "MonitoringService",
    "ReferenceProfile",
    "ReferenceProfileError",
    "WindowSnapshot",
    "compare",
    "default_reference_profile_path",
    "get_monitoring_policy",
    "load_reference_profile",
    "profile_digest",
    "verify_reference_profile",
]
