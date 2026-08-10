"""Feature contract: which raw columns may reach a model, and in which order.

The contract is deliberately explicit rather than inferred from the file. A
schema that is derived from whatever happens to be on disk cannot detect that
the file changed; a declared schema can.

Two guarantees matter here and are enforced by tests:

* ``customerID`` never becomes a feature — it is unique per row and can only be
  memorized;
* ``Churn`` never becomes a feature — that is the target.

``SeniorCitizen`` is listed as **categorical** even though the CSV stores it as
``0``/``1``. It is a binary flag, not a quantity: routing it through the
one-hot encoder keeps any notion of magnitude or ordering out of the model.

Column order is **not** part of the contract. A caller may present the features
in any order — a JSON payload has no inherent order — and
:func:`prepare_features` normalizes to :data:`FEATURE_COLUMNS` after validating.
Everything downstream receives that canonical representation, so the positional
output array always means the same thing.
"""

from __future__ import annotations

import logging

import pandas as pd

from churn.config import get_config
from churn.preprocessing.exceptions import DataQualityError, FeatureContractError

logger = logging.getLogger(__name__)

#: Row identifier. Used for split integrity and fingerprints, never as a feature.
ID_COLUMN = "customerID"

#: Continuous features, in the order the feature matrix must present them.
NUMERIC_FEATURES: tuple[str, ...] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
)

#: Categorical features, in the order the feature matrix must present them.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)

#: Canonical column order of the feature matrix.
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Numeric features that must already be numeric in the *raw* frame.
#: ``TotalCharges`` is excluded on purpose: it arrives as text and is made
#: numeric by :class:`churn.preprocessing.transformers.TotalChargesCleaner`.
_RAW_NUMERIC_FEATURES: tuple[str, ...] = ("tenure", "MonthlyCharges")

#: The single feature allowed to arrive blank. Its blanks are resolved by the
#: structural ``tenure == 0`` rule in
#: :func:`churn.preprocessing.transformers.clean_total_charges`, which is also
#: where a blank that the rule cannot justify is rejected.
STRUCTURAL_BLANK_FEATURE = "TotalCharges"

#: How many offending rows are named in an error message before it is truncated.
_MAX_REPORTED_ROWS = 10


def target_column() -> str:
    """Return the configured target column name."""
    return get_config().target.column


def validate_source_frame(frame: pd.DataFrame) -> None:
    """Check that a raw-shaped frame can produce the contracted feature matrix.

    Args:
        frame: Frame carrying the original columns (target and identifier may be
            present; they are simply not selected).

    Raises:
        FeatureContractError: If a contracted feature is absent.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise FeatureContractError(
            f"Required feature column(s) missing from the source frame: {missing}. "
            f"The contract declares {len(FEATURE_COLUMNS)} features; "
            f"{len(FEATURE_COLUMNS) - len(missing)} were found."
        )


def _blank_mask(values: pd.Series) -> pd.Series:
    """Mark cells that carry no usable value.

    Blank means missing, empty, or whitespace-only. It does **not** mean a
    sentinel product state: ``No internet service`` and ``No phone service`` are
    real categories with their own churn behaviour and pass untouched.
    """
    missing = values.isna()
    if pd.api.types.is_numeric_dtype(values):
        return missing
    empty = values.astype("string").str.strip().eq("").fillna(False)
    return (missing | empty).astype(bool)


def _describe_rows(mask: pd.Series) -> str:
    labels = [str(label) for label in mask.index[mask][:_MAX_REPORTED_ROWS]]
    suffix = ", ..." if int(mask.sum()) > _MAX_REPORTED_ROWS else ""
    return ", ".join(labels) + suffix


def validate_no_unexpected_blanks(matrix: pd.DataFrame) -> None:
    """Reject missing, empty or whitespace-only values in the features.

    Every feature except :data:`STRUCTURAL_BLANK_FEATURE` must carry a usable
    value. There is no imputation anywhere in this pipeline, so a blank that
    reached this point would either propagate as ``NaN`` through the scaler or
    be silently absorbed by ``handle_unknown="ignore"`` — both of which turn a
    data problem into a quiet degradation of the prediction.

    Args:
        matrix: Candidate feature matrix.

    Raises:
        DataQualityError: If any feature other than the structural exception
            carries a blank.
    """
    offenders: list[str] = []
    for column in FEATURE_COLUMNS:
        if column == STRUCTURAL_BLANK_FEATURE:
            continue
        mask = _blank_mask(matrix[column])
        if bool(mask.any()):
            offenders.append(f"{column} ({int(mask.sum())} row(s): {_describe_rows(mask)})")

    if offenders:
        raise DataQualityError(
            "Missing, empty or whitespace-only value(s) in feature(s): "
            + "; ".join(offenders)
            + ". No feature is imputed in this pipeline, so a blank cannot be repaired "
            f"without inventing information. Only {STRUCTURAL_BLANK_FEATURE!r} may arrive "
            "blank, and only where `tenure == 0` justifies a structural zero. Sentinel "
            "categories such as 'No internet service' are real product states, not blanks."
        )


def validate_feature_matrix(matrix: pd.DataFrame) -> None:
    """Check that a frame *is* a valid feature matrix, in any column order.

    Args:
        matrix: Candidate feature matrix.

    Raises:
        FeatureContractError: If the identifier or the target leaked into the
            matrix, if a feature is missing, if an unexpected column is present,
            or if a column that must be numeric is not.
        DataQualityError: If a feature carries an unexpected blank.
    """
    columns = list(matrix.columns)

    if ID_COLUMN in columns:
        raise FeatureContractError(
            f"{ID_COLUMN!r} is an identifier and must never be part of the feature "
            "matrix: it is unique per row, so a model can only memorize it."
        )

    target = target_column()
    if target in columns:
        raise FeatureContractError(
            f"{target!r} is the target and must never be part of the feature matrix. "
            "Its presence would be total leakage."
        )

    missing = [column for column in FEATURE_COLUMNS if column not in columns]
    if missing:
        raise FeatureContractError(f"Feature matrix is missing contracted column(s): {missing}.")

    unexpected = [column for column in columns if column not in FEATURE_COLUMNS]
    if unexpected:
        raise FeatureContractError(
            f"Feature matrix carries column(s) outside the contract: {unexpected}. "
            "Undeclared columns are rejected because they would silently change what "
            "the model sees between training and inference."
        )

    not_numeric = [
        column
        for column in _RAW_NUMERIC_FEATURES
        if not pd.api.types.is_numeric_dtype(matrix[column])
    ]
    if not_numeric:
        raise FeatureContractError(
            f"Column(s) declared numeric are not numeric: {not_numeric}. "
            "This signals an incompatible schema, not a value to be repaired here."
        )

    validate_no_unexpected_blanks(matrix)


def prepare_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Validate a feature matrix and return it in canonical column order.

    This is the entry point everything downstream must go through — training,
    cross-validation and inference alike. It accepts the features in **any**
    order, because a caller (a JSON payload, a rebuilt dataframe) has no reason
    to preserve one, and returns them in :data:`FEATURE_COLUMNS` order, because
    the transformed output is a positional array whose meaning must not depend
    on how the caller happened to assemble the input.

    Args:
        matrix: Candidate feature matrix, in any column order.

    Returns:
        A new frame with the contracted columns in canonical order. The input is
        never modified.

    Raises:
        FeatureContractError: If the schema does not satisfy the contract.
        DataQualityError: If a feature carries an unexpected blank.
    """
    validate_feature_matrix(matrix)
    return matrix.loc[:, list(FEATURE_COLUMNS)].copy()


def build_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the contracted features from a raw-shaped frame.

    The identifier, the target and anything else the source frame carries are
    dropped by the selection. The input frame is never modified.

    Args:
        frame: Frame carrying the original columns.

    Returns:
        The validated feature matrix, in canonical order.

    Raises:
        FeatureContractError: If the contract is not satisfied.
        DataQualityError: If a feature carries an unexpected blank.
    """
    validate_source_frame(frame)
    matrix = prepare_features(frame.loc[:, list(FEATURE_COLUMNS)])
    logger.debug("Feature matrix built: %d rows x %d columns", *matrix.shape)
    return matrix
