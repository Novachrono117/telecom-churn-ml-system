"""The predictive function of the serving boundary, with no HTTP in sight.

The chain implemented here is the one the freeze wrote down, in this order and
with no shortcut::

    raw records
      -> DataFrame
      -> churn.preprocessing.contracts.prepare_features(...)
      -> frozen pipeline.predict_proba(...)
      -> column resolved from classes_
      -> P(Churn = 1)
      -> probability >= frozen threshold
      -> decision

Two design rules hold this module together.

**Nothing frozen is reimplemented.** The positive-column resolution and the
``>=`` comparison come from :mod:`churn.modeling.freeze`, whose source is hashed
into ``code_provenance`` in the decision policy. A second implementation of
``probability >= threshold`` living here would be a second decision rule, free to
drift from the frozen one while every fingerprint kept matching — exactly the
class of silent divergence the freeze exists to prevent.

**``prepare_features`` is not optional.** The persisted pipeline begins *after*
it: the object on disk expects a validated matrix in canonical column order and
validates nothing itself. Handing it a raw payload would work — it would produce
a number — and the number would be wrong in ways nothing raises: an unvalidated
blank would ride through ``handle_unknown="ignore"`` as an all-zero block, and a
JSON object's arbitrary key order would silently permute the columns.
:func:`prepared_features` is therefore the only way into the pipeline in this
package, and a test asserts the pipeline is never called with anything else.

``pipeline.predict`` is never used, here or anywhere in this package. It applies
scikit-learn's own 0.5 cut, which is not the frozen policy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.modeling.freeze import apply_decision_rule, positive_probability
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features

logger = logging.getLogger(__name__)

#: What each decision label means to a reader. ``1`` is ``Churn = Yes``.
RETAINED = "retained"
CHURN = "churn"
DECISION_LABELS: dict[int, str] = {0: RETAINED, 1: CHURN}


@dataclass(frozen=True)
class Prediction:
    """One scored record. A plain value, deliberately not a pydantic model.

    The service returns these, and the HTTP layer maps them to response schemas.
    Keeping the type free of pydantic and of FastAPI is what lets the predictive
    path be exercised — and asserted on — without a server.

    Every field is a pure function of the payload and the frozen artefacts: no
    timestamp, no request id, nothing that would make two identical requests
    produce two different answers.
    """

    churn_probability: float
    prediction: int
    decision: str
    threshold: float
    comparison: str
    calibration_policy: str
    model_fingerprint: str


def build_feature_frame(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Turn validated payload records into a frame, preserving input order.

    The frame keeps whatever column order the payload happened to use; JSON has
    no order semantics and :func:`prepare_features` normalises it. The index is a
    plain ``RangeIndex`` so that row ``i`` of the output is record ``i`` of the
    request.

    Args:
        records: One mapping per record, each carrying the contracted features.

    Returns:
        A frame with one row per record.

    Raises:
        ValueError: If no record was given.
    """
    if not records:
        raise ValueError("At least one record is required to build a feature frame.")
    return pd.DataFrame(list(records))


def prepared_features(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return the canonical feature matrix for ``records``.

    The single entry point into the frozen pipeline. Raises through the frozen
    contract rather than reinterpreting it:

    Raises:
        FeatureContractError: If the schema does not satisfy the contract.
        DataQualityError: If a value is blank where no rule justifies it, or if
            ``TotalCharges`` can be neither read as a number nor treated as a
            structural zero.
    """
    return prepare_features(build_feature_frame(records))


def churn_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return P(Churn = 1) per row, read from the column ``classes_`` names.

    Delegates to :func:`churn.modeling.freeze.positive_probability`, which
    resolves the column from ``classes_`` instead of assuming ``[:, 1]``. The
    assumption happens to hold for this estimator; reading the wrong column would
    invert every decision without raising anything, so it is resolved rather than
    assumed.

    Args:
        pipeline: The frozen, verified pipeline.
        features: Output of :func:`prepared_features`.

    Returns:
        A float array of probabilities, one per row.

    Raises:
        ValueError: If a probability falls outside ``[0, 1]``, which would mean
            the object being served is not a probabilistic classifier.
    """
    probabilities = np.asarray(positive_probability(pipeline, features), dtype=float)
    if probabilities.shape[0] != len(features):
        raise ValueError(
            f"The model returned {probabilities.shape[0]} probabilities for {len(features)} rows."
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned a non-finite probability.")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("The model returned a value outside [0, 1]; it is not a probability.")
    return probabilities


def decide(probability: float, threshold: float) -> int:
    """Apply the frozen rule to one probability: ``1`` if ``probability >= threshold``.

    The comparison is ``>=``, not ``>``. The two differ on exactly the customers
    whose probability equals the threshold, and a record sitting on the boundary
    is predicted **positive**.

    This is a thin scalar view over
    :func:`churn.modeling.freeze.apply_decision_rule` so that both the scalar and
    the vector path are the same comparison, in the same frozen module.
    """
    return int(apply_decision_rule(np.asarray([probability], dtype=float), threshold)[0])


def canonical_probability(pipeline: Pipeline, record: Mapping[str, Any]) -> float:
    """Score exactly one record. **The** scoring primitive of this boundary.

    Every probability the API ever returns comes out of this function, whether the
    record arrived alone or inside a batch of five hundred. That is not a
    convenience — it is what makes the two endpoints numerically identical.

    Why it has to be one row at a time
    ----------------------------------
    The first implementation scored a batch as one N-row matrix, which is the
    obvious and efficient thing to do. Every step of the frozen pipeline is
    row-wise, so the results *ought* to have been identical, and the preprocessing
    output was in fact bit-identical at any batch size — verified with
    ``array_equal``.

    The classifier was not. ``decision_function`` is a matrix product, and BLAS
    dispatches a 1x46 product to a different kernel (GEMV) than an Nx46 one (GEMM).
    The two accumulate the same 46 terms in a different order, floating-point
    addition is not associative, and the last bit can differ — measured here at up
    to 1 ULP, 1.11e-16. Tiny, but *observable through the API*: the same customer
    could come back with two different probabilities depending on how the caller
    happened to batch the request, and, at the very edge, with two different
    decisions.

    Rounding the score would have hidden it and would have been the wrong fix:
    Phase 9C stored the threshold unrounded precisely because a value rounded in
    the sixth decimal can move customers across the boundary. So the arithmetic is
    made canonical instead. One row is the shape the single endpoint has always
    used, so it is the shape everything uses.

    The pipeline is **not** reloaded per record — it is loaded once per process and
    handed in. What repeats per record is arithmetic, not I/O.

    Raises:
        FeatureContractError: If the schema does not satisfy the contract.
        DataQualityError: If a value is blank or unreadable where no rule
            justifies it.
    """
    return float(churn_probabilities(pipeline, prepared_features([record]))[0])


def decision_label(prediction: int) -> str:
    """Map ``0``/``1`` to the words the API answers with.

    Raises:
        ValueError: If the prediction is not a binary label.
    """
    try:
        return DECISION_LABELS[int(prediction)]
    except KeyError as error:
        raise ValueError(f"{prediction!r} is not a binary decision label.") from error


__all__ = [
    "CHURN",
    "DECISION_LABELS",
    "FEATURE_COLUMNS",
    "RETAINED",
    "Prediction",
    "build_feature_frame",
    "churn_probabilities",
    "canonical_probability",
    "decide",
    "decision_label",
    "prepared_features",
]
