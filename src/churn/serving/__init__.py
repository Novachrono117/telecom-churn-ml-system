"""Phase 11 — production inference boundary for the frozen churn model.

This package turns the object frozen in Phase 9C into something that can be
called: it validates a payload, runs the frozen feature contract, applies the
persisted pipeline, reads P(Churn = 1) from the column ``classes_`` identifies
and applies the frozen ``>=`` threshold. It trains nothing, selects nothing,
calibrates nothing and reads no dataset.

The layers are separate on purpose::

    HTTP (api)  ->  inference service (service)  ->  frozen artefacts (artifacts)

Only the first needs FastAPI. Importing this package does **not** import it: the
predictive path is usable, and testable, without a web framework — which is why
``api`` is not re-exported here.
"""

from churn.serving.artifacts import (
    FREEZE_COMMIT,
    FREEZE_COMMIT_SHORT,
    SERVING_VERSION,
    FrozenArtifacts,
    load_frozen_artifacts,
)
from churn.serving.inference import Prediction
from churn.serving.service import ChurnInferenceService, ModelIdentity
from churn.serving.settings import ServingSettings, load_settings

__all__ = [
    "FREEZE_COMMIT",
    "FREEZE_COMMIT_SHORT",
    "SERVING_VERSION",
    "ChurnInferenceService",
    "FrozenArtifacts",
    "ModelIdentity",
    "Prediction",
    "ServingSettings",
    "load_frozen_artifacts",
    "load_settings",
]
