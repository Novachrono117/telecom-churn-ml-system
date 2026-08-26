"""Operational configuration of the serving layer — and nothing else.

This module exists because ``configs/base.toml`` must not grow web settings. That
file is part of the Phase 9C provenance: its digest is recorded in
``reports/decision_policy.json`` and ``scripts/freeze_model.py --verify`` fails if
it moves. A host name has no business invalidating a model freeze.

**What may be configured here is deliberately narrow.** Everything in this module
answers "where does this process run and how much load does it accept". Nothing
in it answers "what does the model predict". The line is enforced, not merely
documented: :data:`FORBIDDEN_ENV_VARS` names the environment variables that
would, if they existed, let a deployment redefine a frozen decision, and
:func:`load_settings` refuses to start when one of them is set.

The refusal is active rather than passive on purpose. Silently ignoring
``CHURN_THRESHOLD=0.5`` is worse than rejecting it: the operator would believe
the threshold had changed, and every decision would be attributed to a rule
nobody applied. The threshold and the calibration policy come from
``reports/decision_policy.json``, which is the only source there is.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT
from churn.serving.errors import ServingConfigurationError

logger = logging.getLogger(__name__)

#: Prefix of every environment variable this layer reads.
ENV_PREFIX = "CHURN_SERVING_"

#: Default location of the frozen decision policy, relative to the repository root.
DEFAULT_POLICY_RELATIVE_PATH = "reports/decision_policy.json"

#: Maximum records accepted by the batch endpoint, by default.
#:
#: 500 is an operational bound, not a statistical one. It is chosen so that one
#: request stays a few hundred kilobytes of JSON and one synchronous
#: ``predict_proba`` call over a 46-column dense matrix stays in the low
#: milliseconds — small enough that a single blocking request cannot monopolise
#: the event loop, large enough that a retention campaign is not forced into
#: hundreds of round trips. It bounds the *request*, never the model: the
#: probability a record receives is identical whether it arrives alone or in a
#: batch of 500.
DEFAULT_MAX_BATCH_SIZE = 500

#: Hard ceiling on the configured batch size. A deployment may lower the limit;
#: raising it without bound would turn the endpoint into an unmetered work queue.
MAX_ALLOWED_BATCH_SIZE = 5_000

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000

#: Environment variables that would let a deployment override a frozen decision.
#: None of them is read. All of them are refused, loudly.
FORBIDDEN_ENV_VARS: tuple[str, ...] = (
    "CHURN_THRESHOLD",
    "CHURN_SERVING_THRESHOLD",
    "CHURN_DECISION_THRESHOLD",
    "CHURN_SERVING_DECISION_THRESHOLD",
    "CHURN_CALIBRATION",
    "CHURN_SERVING_CALIBRATION",
    "CHURN_CALIBRATION_POLICY",
    "CHURN_SERVING_CALIBRATION_POLICY",
    "CHURN_POSITIVE_CLASS",
    "CHURN_SERVING_POSITIVE_CLASS",
    "CHURN_SERVING_COMPARISON",
)


class ServingSettings(BaseModel):
    """Where the process finds its artefacts, and how much load it accepts.

    Frozen after construction: the settings object is read by every request and
    is never rebuilt, so it must not be mutable state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_path: Path
    pipeline_path: Path | None = Field(
        default=None,
        description=(
            "Overrides the pipeline location. When None, the path recorded in the "
            "decision policy is used, resolved against the repository root."
        ),
    )
    max_batch_size: int = Field(default=DEFAULT_MAX_BATCH_SIZE, ge=1, le=MAX_ALLOWED_BATCH_SIZE)
    host: str = Field(default=DEFAULT_HOST, min_length=1)
    port: int = Field(default=DEFAULT_PORT, ge=1, le=65535)


def default_policy_path(root: Path | None = None) -> Path:
    """Return the repository's decision-policy location."""
    return (root or PROJECT_ROOT) / DEFAULT_POLICY_RELATIVE_PATH


def _reject_frozen_overrides(env: Mapping[str, str]) -> None:
    """Refuse to start if the environment tries to redefine a frozen decision."""
    present = [name for name in FORBIDDEN_ENV_VARS if name in env]
    if present:
        raise ServingConfigurationError(
            f"Environment variable(s) {present} are not settings of this service. "
            "The decision threshold, the calibration policy and the positive class are "
            "part of the frozen model contract and are read only from the decision "
            "policy. Unset them; a deployment cannot move a decision boundary."
        )


def _read_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as error:
        raise ServingConfigurationError(f"{name} must be an integer, got {raw!r}.") from error


def _read_path(env: Mapping[str, str], name: str) -> Path | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return Path(raw.strip()).expanduser()


def load_settings(
    env: Mapping[str, str] | None = None,
    root: Path | None = None,
) -> ServingSettings:
    """Build the serving settings from the process environment.

    Args:
        env: Environment mapping. Defaults to ``os.environ``.
        root: Repository root used for the default artefact locations.

    Returns:
        The validated, frozen settings.

    Raises:
        ServingConfigurationError: If a forbidden override is present or a value
            cannot be read.
    """
    environment = os.environ if env is None else env
    _reject_frozen_overrides(environment)

    policy_path = _read_path(environment, f"{ENV_PREFIX}POLICY_PATH") or default_policy_path(root)
    settings = ServingSettings(
        policy_path=policy_path,
        pipeline_path=_read_path(environment, f"{ENV_PREFIX}MODEL_PATH"),
        max_batch_size=_read_int(
            environment, f"{ENV_PREFIX}MAX_BATCH_SIZE", DEFAULT_MAX_BATCH_SIZE
        ),
        host=environment.get(f"{ENV_PREFIX}HOST", DEFAULT_HOST),
        port=_read_int(environment, f"{ENV_PREFIX}PORT", DEFAULT_PORT),
    )
    # The paths are operational configuration and are logged; no request data is.
    logger.info(
        "Serving settings: max_batch_size=%d host=%s port=%d",
        settings.max_batch_size,
        settings.host,
        settings.port,
    )
    return settings
