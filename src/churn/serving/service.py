"""``ChurnInferenceService``: the primitive the HTTP layer is a thin skin over.

It knows the pipeline, the threshold, the positive class and the policy
metadata. It does not know what a request, a status code, a training set or a
holdout is — it cannot load data at all, because nothing in it reads a file
after startup.

That is the point of the split: :meth:`ChurnInferenceService.predict_one` is
callable, and assertable, without a server.

**Endpoint invariance is exact.** The same record scored through ``/predict`` and
through ``/predict/batch`` returns the *same float*, not a float within a
tolerance. Both endpoints call :meth:`ChurnInferenceService._score`, once per
record, and that method scores a one-row frame — always the same matrix shape,
therefore always the same arithmetic.

That shape is a deliberate choice and it cost something. The first implementation
scored a batch as one N-row matrix, which is the efficient thing to do and was not
wrong: the preprocessing output was bit-identical at any batch size. The classifier
was not. ``decision_function`` is a matrix product, and BLAS dispatches a 1x46
product to a different kernel (GEMV) than an Nx46 one (GEMM); the same 46 terms
accumulate in a different order, floating-point addition is not associative, and
the last bit differed — measured at up to 1 ULP, 1.11e-16.

That was a legitimate implementation, and the difference was never going to change
a decision except for a record sitting within one ULP of the threshold. It was
still the wrong contract to publish: a caller must not have to know how a request
was batched in order to reproduce a probability. Rounding the score would have
hidden it, and Phase 9C already settled why that is unacceptable — it stored the
threshold unrounded because a value rounded in the sixth decimal can move customers
across the boundary. So the arithmetic was made canonical rather than the output
made blurry.

**The trade-off, stated plainly.** Batch scoring no longer amortises one matrix
product over N rows, so its potential throughput is lower than a vectorised
implementation's. No benchmark is claimed here. The cost was accepted because this
model is small — 46 transformed columns, a logistic regression — and the batch
endpoint is bounded at 500 records, so the ceiling on what is given up is known and
modest. Semantic consistency at the API boundary was judged worth more than
unmeasured throughput.

The pipeline is still loaded **once per process**. What repeats per record is
arithmetic, not I/O: nothing is unpickled, re-read or re-verified per row.

The instance is frozen and shared by every request in the process. It holds no
per-request state, mutates nothing, and never re-reads an artefact: the frozen
pipeline is read-only during serving, which is what makes one shared instance
safe under concurrency.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from churn.preprocessing.exceptions import DataQualityError, FeatureContractError
from churn.serving.artifacts import (
    FREEZE_COMMIT,
    FREEZE_COMMIT_SHORT,
    SERVING_VERSION,
    FrozenArtifacts,
    load_frozen_artifacts,
)
from churn.serving.errors import BatchTooLargeError
from churn.serving.inference import (
    Prediction,
    canonical_features_and_probability,
    decide,
    decision_label,
)
from churn.serving.settings import DEFAULT_MAX_BATCH_SIZE, ServingSettings

logger = logging.getLogger(__name__)


@runtime_checkable
class PredictionObserver(Protocol):
    """Something that watches predictions without participating in them.

    The contract is one-directional and deliberately tiny: an observer is handed
    the canonical feature matrix, the scores and the decisions **after** they have
    been produced, and whatever it returns is discarded. There is no way for an
    implementation to influence a probability, because nothing it does is read.

    Implementations are expected not to raise. The prediction path guards against
    it anyway — see :meth:`ChurnInferenceService._observe` — because "expected not
    to" is not a guarantee, and a monitoring bug must never cost a prediction.
    """

    def observe(
        self,
        features: pd.DataFrame,
        probabilities: Sequence[float],
        predictions: Sequence[int],
    ) -> None:
        """Fold already-scored records into whatever aggregates are being kept."""
        ...


@dataclass(frozen=True)
class ModelIdentity:
    """Non-sensitive metadata about what this process is serving.

    Everything here is already public in ``reports/decision_policy.json``. No
    filesystem path, no environment secret and no dataset statistic appears: a
    metadata endpoint exists to identify a model, not to describe the machine it
    runs on.

    ``freeze_commit`` and ``serving_version`` are separate fields on purpose.
    Phase 11 versions the *application*; the model it serves is the one frozen in
    Phase 9C and is unchanged by any release of this code.
    """

    estimator: str
    freeze_commit: str
    freeze_commit_short: str
    model_fingerprint: str
    pipeline_sha256: str
    decision_policy_sha256: str
    n_features: int
    n_transformed_features: int
    feature_names: list[str]
    calibration_policy: str
    threshold_policy: str
    threshold: float
    comparison: str
    positive_class_label: int
    positive_class_column: int
    positive_class_meaning: str
    serving_version: str
    runtime: dict[str, str]


@dataclass(frozen=True)
class ChurnInferenceService:
    """Scores records with the frozen model under the frozen decision policy."""

    artifacts: FrozenArtifacts
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    observer: PredictionObserver | None = None

    @classmethod
    def from_settings(cls, settings: ServingSettings | None = None) -> ChurnInferenceService:
        """Run the startup gates and build the service, or fail closed.

        Raises:
            ServingStartupError: If any artefact, integrity or runtime gate fails.
        """
        from churn.serving.settings import load_settings

        resolved = settings or load_settings()
        artifacts = load_frozen_artifacts(resolved)
        return cls(artifacts=artifacts, max_batch_size=resolved.max_batch_size)

    @property
    def threshold(self) -> float:
        """The frozen decision threshold. Read from the policy, never configurable."""
        return self.artifacts.threshold

    @property
    def model_fingerprint(self) -> str:
        """Authoritative identity of the frozen model."""
        return self.artifacts.model_fingerprint

    def predict_batch(self, records: Sequence[Mapping[str, Any]]) -> list[Prediction]:
        """Score every record, in the order given.

        Args:
            records: Validated feature records. Order is preserved: result ``i``
                belongs to record ``i``.

        Returns:
            One :class:`~churn.serving.inference.Prediction` per record.

        Raises:
            BatchTooLargeError: If the batch exceeds the configured operational
                limit. The limit bounds the request, never the model.
            FeatureContractError: If the schema does not satisfy the frozen contract.
            DataQualityError: If a value is blank or unreadable where no rule
                justifies it.
        """
        if not records:
            raise ValueError("At least one record is required.")
        if len(records) > self.max_batch_size:
            raise BatchTooLargeError(
                f"The batch carries {len(records)} records; this deployment accepts at "
                f"most {self.max_batch_size} per request. Split it and retry. The limit "
                "is operational and does not change any prediction."
            )

        scored: list[Prediction] = []
        for index, record in enumerate(records):
            try:
                scored.append(self._score(record))
            except (FeatureContractError, DataQualityError) as error:
                # Re-raised as the SAME type, so the error code the API maps it to
                # is unchanged; the index is added because the frozen contract's
                # message describes row 0 of a one-row frame and a caller sending
                # four hundred records needs to know which one it was.
                raise type(error)(f"Record {index} of the batch: {error}") from error
        return scored

    def predict_one(self, record: Mapping[str, Any]) -> Prediction:
        """Score one record.

        Calls the same primitive :meth:`predict_batch` calls, once. Not a second
        implementation of the same steps, and not a batch of one either — the
        message a rejected record produces should not mention a batch that the
        caller never sent.
        """
        return self._score(record)

    def _score(self, record: Mapping[str, Any]) -> Prediction:
        """The single scoring path. Both endpoints go through here, per record.

        Everything below the probability is metadata copied from the verified
        frozen artefacts, so two calls with the same record and the same process
        return equal values in every field.

        The observer, when there is one, is called **after** the answer exists and
        cannot alter it: the returned :class:`Prediction` is already fully
        determined by the line above the call.
        """
        features, probability = canonical_features_and_probability(self.artifacts.pipeline, record)
        prediction = decide(probability, self.threshold)
        answer = Prediction(
            churn_probability=probability,
            prediction=prediction,
            decision=decision_label(prediction),
            threshold=self.threshold,
            comparison=self.artifacts.comparison,
            calibration_policy=self.artifacts.calibration_policy,
            model_fingerprint=self.artifacts.model_fingerprint,
        )
        self._observe(features, probability, prediction)
        return answer

    def _observe(self, features: pd.DataFrame, probability: float, prediction: int) -> None:
        """Hand one scored record to the observer, and never let that cost a prediction.

        The prediction path is authoritative. A monitoring failure is logged and
        counted by the observer, and the request continues: the alternative —
        failing a prediction because a histogram could not be updated — trades a
        correct answer for an observation, which is backwards.

        The failure is never swallowed silently: it is logged, and the monitoring
        status the service reports degrades, so the loss of observability is visible
        even though the prediction was not affected.

        What is logged is an event name and the exception's **type** — never
        ``str(error)`` and never a traceback. This guard runs holding ``features``
        and ``probability``, so an exception raised beneath it can carry either into
        its message; writing that message to a log would move the payload into the
        one place nobody audits for it.
        """
        observer = self.observer
        if observer is None:
            return
        try:
            observer.observe(features, [probability], [prediction])
        except Exception as error:  # noqa: BLE001 - monitoring must never break a prediction
            logger.error(
                "monitoring_observation_failure exception_type=%s; the prediction is unaffected.",
                type(error).__name__,
            )

    def identity(self) -> ModelIdentity:
        """Return the non-sensitive metadata of the frozen model."""
        artifacts = self.artifacts
        rule = artifacts.policy.decision_rule
        return ModelIdentity(
            estimator=artifacts.estimator,
            freeze_commit=FREEZE_COMMIT,
            freeze_commit_short=FREEZE_COMMIT_SHORT,
            model_fingerprint=artifacts.model_fingerprint,
            pipeline_sha256=artifacts.pipeline_sha256,
            decision_policy_sha256=artifacts.policy_sha256,
            n_features=artifacts.n_features,
            n_transformed_features=artifacts.n_transformed_features,
            feature_names=list(artifacts.feature_columns),
            calibration_policy=artifacts.calibration_policy,
            threshold_policy=artifacts.threshold_policy,
            threshold=artifacts.threshold,
            comparison=artifacts.comparison,
            positive_class_label=artifacts.positive_class_label,
            positive_class_column=artifacts.positive_class_column,
            positive_class_meaning=rule.positive_class_meaning,
            serving_version=SERVING_VERSION,
            runtime=dict(artifacts.runtime),
        )


__all__ = ["ChurnInferenceService", "ModelIdentity", "PredictionObserver"]
