"""Reusable preprocessing pipeline.

Structure::

    TotalChargesCleaner        deterministic, learns nothing
    └── ColumnTransformer
        ├── numeric      -> StandardScaler        (mean/std fitted on train only)
        └── categorical  -> OneHotEncoder         (categories fitted on train only)

Only two components hold fitted state — the scaler's mean/standard deviation
and the encoder's category lists — and both are estimated inside ``fit``. As
long as ``fit`` is called on the training pool (or on a cross-validation
training fold), no information from the holdout can reach the model.

No statistical imputation is configured. After the structural ``TotalCharges``
rule the approved dataset carries no genuine numeric missingness, so an imputer
would have nothing to do except hide an unexpected gap. An unexpected gap must
fail through :class:`~churn.preprocessing.exceptions.DataQualityError` instead.

.. warning::

   This pipeline does **not** validate its own input. It expects the canonical,
   contract-checked frame produced by
   :func:`churn.preprocessing.contracts.prepare_features`. Requirement carried
   into Phase 11: the inference interface must call ``prepare_features()``
   before the persisted pipeline, and ``predict`` / ``predict_proba`` must not
   be exposed as a public interface for unvalidated raw payloads. Keeping
   validation outside the estimator is what makes it safe to refit inside every
   cross-validation fold in Phase 7.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.preprocessing.contracts import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn.preprocessing.transformers import TotalChargesCleaner

logger = logging.getLogger(__name__)

NUMERIC_STEP = "numeric"
CATEGORICAL_STEP = "categorical"
CLEANER_STEP = "total_charges"
ENCODER_STEP = "encode"


def build_column_encoder(
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> ColumnTransformer:
    """Build the scaling/encoding block for a given set of columns.

    Extracted so that later phases can encode a *wider* frame — the Phase 6
    ablation appends derived columns — without duplicating the encoder's
    configuration. The Phase 4 pipeline calls it with the contracted 19
    features and is unaffected.

    Args:
        numeric: Columns to standardize.
        categorical: Columns to one-hot encode.

    Returns:
        The unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    return ColumnTransformer(
        transformers=[
            (NUMERIC_STEP, StandardScaler(), list(numeric)),
            (
                CATEGORICAL_STEP,
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
                list(categorical),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessor() -> Pipeline:
    """Build the preprocessing pipeline, unfitted.

    Dense output is requested from the encoder: the matrix is roughly 5.6k x 45
    for this dataset, so sparsity buys nothing while a dense array keeps every
    downstream estimator, explainer and serialization path simple.

    ``handle_unknown="ignore"`` makes a category never seen during ``fit``
    encode as all-zeros instead of raising at prediction time — the behaviour a
    deployed pipeline needs. Monitoring for unseen categories is designed in
    Phase 12; silently ignoring them is a serving decision, not a substitute for
    detecting them.

    Returns:
        The unfitted :class:`~sklearn.pipeline.Pipeline`.
    """
    return Pipeline(
        steps=[
            (CLEANER_STEP, TotalChargesCleaner()),
            (ENCODER_STEP, build_column_encoder(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        ]
    )


def feature_names_out(preprocessor: Pipeline) -> list[str]:
    """Return the transformed feature names of a fitted preprocessor.

    Args:
        preprocessor: A pipeline returned by :func:`build_preprocessor`, fitted.

    Returns:
        One name per output column, in output order.

    Raises:
        sklearn.exceptions.NotFittedError: If the pipeline has not been fitted.
    """
    return [str(name) for name in preprocessor.named_steps[ENCODER_STEP].get_feature_names_out()]
