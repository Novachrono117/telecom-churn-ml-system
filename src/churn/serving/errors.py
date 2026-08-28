"""Error taxonomy of the serving boundary, split by *when* the error can happen.

Two families, and the split is the whole point:

* :class:`ServingStartupError` — the process cannot honestly serve. It is raised
  while the frozen artefacts are being located, verified and loaded, and it is
  **never** turned into an HTTP response: a process that cannot prove it holds
  the frozen model must not start, rather than start and answer 503 forever.
* :class:`ServingRequestError` — one request cannot be answered. It carries a
  stable ``code`` and an HTTP ``status`` so the API layer can map it without
  inspecting messages, and its message is written to be shown to a client:
  no path, no traceback, no payload value.

The preprocessing errors raised by the frozen feature contract
(:class:`~churn.preprocessing.exceptions.FeatureContractError` and
:class:`~churn.preprocessing.exceptions.DataQualityError`) are deliberately
**not** re-raised as members of this hierarchy. They are the authority on what a
valid feature record is, and wrapping them here would create a second, drifting
definition of validity. The API maps them to codes at the edge instead.
"""

from __future__ import annotations

#: Stable machine-readable error codes. Clients may branch on these; they are
#: part of the API contract and change only with the API version.
CODE_SERVICE_NOT_READY = "SERVICE_NOT_READY"
CODE_INVALID_REQUEST_SCHEMA = "INVALID_REQUEST_SCHEMA"
CODE_INVALID_FEATURE_PAYLOAD = "INVALID_FEATURE_PAYLOAD"
CODE_INVALID_FEATURE_VALUE = "INVALID_FEATURE_VALUE"
CODE_BATCH_TOO_LARGE = "BATCH_TOO_LARGE"
CODE_INTERNAL_ERROR = "INTERNAL_ERROR"

#: Every code the **Phase 11 API** can emit. Used by the tests to keep the set closed,
#: and recorded in `reports/experiments/serving_results.json`, which is why a later
#: phase must not extend it: adding a member here would change a published contract
#: that verifies byte for byte.
ERROR_CODES: frozenset[str] = frozenset(
    {
        CODE_SERVICE_NOT_READY,
        CODE_INVALID_REQUEST_SCHEMA,
        CODE_INVALID_FEATURE_PAYLOAD,
        CODE_INVALID_FEATURE_VALUE,
        CODE_BATCH_TOO_LARGE,
        CODE_INTERNAL_ERROR,
    }
)

#: Phase 13. The local decomposition did not reproduce the probability that was
#: served, so no explanation is returned. Distinct from INTERNAL_ERROR because it is
#: specific and actionable: only the explanation failed, and the prediction endpoints
#: are unaffected.
#:
#: Deliberately **not** a member of :data:`ERROR_CODES`. It belongs to a route that
#: exists only when the portfolio demo is enabled, so on a Phase 11 deployment it
#: genuinely cannot be emitted and the recorded set stays exactly true.
CODE_EXPLANATION_UNAVAILABLE = "EXPLANATION_UNAVAILABLE"

#: The Phase 11 codes plus the ones the optional demo adds.
PORTFOLIO_ERROR_CODES: frozenset[str] = frozenset({CODE_EXPLANATION_UNAVAILABLE})
ALL_ERROR_CODES: frozenset[str] = ERROR_CODES | PORTFOLIO_ERROR_CODES


class ServingStartupError(RuntimeError):
    """The process cannot serve the frozen model and must not start.

    Fail-closed by construction: there is no repair path attached to any
    subclass. In particular, a missing artefact is an error, never a trigger to
    rebuild one — the serving boundary does not train.
    """


class ServingConfigurationError(ServingStartupError):
    """The serving configuration is invalid or tries to override a frozen decision."""


class ArtifactNotFoundError(ServingStartupError):
    """A required frozen artefact is absent."""


class ArtifactIntegrityError(ServingStartupError):
    """A frozen artefact is present but is not the one the decision policy describes."""


class RuntimeCompatibilityError(ServingStartupError):
    """The runtime differs from the environment the frozen model was produced in."""


class ServingRequestError(RuntimeError):
    """One request cannot be answered. Carries a stable code and an HTTP status."""

    code: str = CODE_INTERNAL_ERROR
    status: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ServiceNotReadyError(ServingRequestError):
    """The inference service is not available on this process."""

    code = CODE_SERVICE_NOT_READY
    status = 503


class BatchTooLargeError(ServingRequestError):
    """The batch exceeds the operational limit configured for this deployment."""

    code = CODE_BATCH_TOO_LARGE
    status = 413
