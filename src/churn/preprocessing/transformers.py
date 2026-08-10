"""Deterministic, stateless cleaning of ``TotalCharges``.

Phase 2 and the EDA established the fact this rule rests on: the 11 unreadable
``TotalCharges`` cells are **exactly** the 11 customers with ``tenure == 0``.
The two sets coincide; there is no unreadable cell at positive tenure.

Rule applied here
-----------------
1. surrounding whitespace is stripped;
2. the value is coerced to a number;
3. blank **and** ``tenure == 0`` becomes ``0.0``;
4. anything else that cannot be read as a number raises.

Why ``0.0`` and not an imputed statistic: ``TotalCharges`` is an accumulated
amount. A customer who has not completed a billing cycle has accumulated
nothing, so zero is the value the quantity actually takes — no distribution is
estimated and nothing is learned from the data. A median would assert a billing
history that does not exist.

**This is a domain decision grounded in the pattern observed in this dataset,
not a universal rule for telecom data.** In a file where blanks appeared at
positive tenure, the blank would mean "unknown", not "never billed", and this
rule would be wrong. That case is exactly what step 4 refuses to paper over.

Nothing is fitted: ``fit`` learns no statistic, so the transformer behaves
identically on training data, on cross-validation folds and at inference time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from churn.preprocessing.exceptions import DataQualityError

logger = logging.getLogger(__name__)

#: How many offending rows are named in an error message before it is truncated.
_MAX_REPORTED_ROWS = 10


def _coerce(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return the numeric coercion of ``values`` and the mask of blank cells.

    "Blank" means missing or whitespace-only. It is kept apart from
    "unparseable" because only the former can be a structural zero.
    """
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce"), values.isna()

    text = values.astype("string").str.strip()
    blank = text.isna() | (text == "")
    return pd.to_numeric(text, errors="coerce"), blank


def _describe_rows(mask: pd.Series) -> str:
    labels = [str(label) for label in mask.index[mask][:_MAX_REPORTED_ROWS]]
    suffix = ", ..." if int(mask.sum()) > _MAX_REPORTED_ROWS else ""
    return ", ".join(labels) + suffix


def clean_total_charges(
    frame: pd.DataFrame,
    column: str = "TotalCharges",
    tenure_column: str = "tenure",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``column`` as a valid ``float64`` Series.

    Args:
        frame: Frame carrying ``column`` and ``tenure_column``.
        column: Accumulated-charges column to clean.
        tenure_column: Relationship-length column that justifies a structural zero.

    Returns:
        A new frame. The input is never modified.

    Raises:
        KeyError: If either column is absent.
        DataQualityError: If ``tenure`` itself is unreadable, or if ``column``
            holds a value that is neither numeric nor a blank at ``tenure == 0``.
    """
    for required in (column, tenure_column):
        if required not in frame.columns:
            raise KeyError(f"Column {required!r} is required to clean {column!r}.")

    tenure = pd.to_numeric(frame[tenure_column], errors="coerce")
    unreadable_tenure = tenure.isna()
    if bool(unreadable_tenure.any()):
        raise DataQualityError(
            f"{int(unreadable_tenure.sum())} row(s) have an unreadable "
            f"{tenure_column!r}, so no {column!r} decision can be justified. "
            f"Rows: {_describe_rows(unreadable_tenure)}."
        )

    coerced, blank = _coerce(frame[column])
    unparseable = coerced.isna() & ~blank
    blank_with_history = blank & (tenure != 0)
    invalid = unparseable | blank_with_history

    if bool(invalid.any()):
        raise DataQualityError(
            f"{int(invalid.sum())} row(s) have a {column!r} value that cannot be read "
            f"as a number and cannot be treated as a structural zero "
            f"({int(unparseable.sum())} non-numeric, "
            f"{int(blank_with_history.sum())} blank with {tenure_column} > 0). "
            f"Rows: {_describe_rows(invalid)}. "
            "Blanks are only accepted where the customer has not completed a billing "
            "cycle; imputing anything else here would invent a billing history."
        )

    result = frame.copy()
    result[column] = coerced.where(~blank, 0.0).astype("float64")

    # Logged at DEBUG on purpose: a count of structural zeros in one partition,
    # combined with the total published in Phase 3, would reveal the count in the
    # other by subtraction. The rule that produced them is what matters, and it is
    # documented; the tally is a debugging detail, not a normal output.
    if bool(blank.any()):
        logger.debug(
            "%s: %d blank cell(s) set to 0.0 as structural zeros (%s == 0)",
            column,
            int(blank.sum()),
            tenure_column,
        )
    return result


class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """Scikit-learn wrapper around :func:`clean_total_charges`.

    Stateless by construction: ``fit`` records only the input column names, so
    that the step can sit inside a ``Pipeline`` and be persisted with it without
    ever carrying a statistic estimated from data.
    """

    def __init__(self, column: str = "TotalCharges", tenure_column: str = "tenure") -> None:
        self.column = column
        self.tenure_column = tenure_column

    def fit(self, X: pd.DataFrame, y: object = None) -> TotalChargesCleaner:  # noqa: N803
        """Record the input schema. No statistic is estimated."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return a cleaned copy of ``X``."""
        check_is_fitted(self, "feature_names_in_")
        return clean_total_charges(X, self.column, self.tenure_column)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the unchanged column names; the step adds and drops nothing."""
        check_is_fitted(self, "feature_names_in_")
        return np.asarray(self.feature_names_in_, dtype=object)
