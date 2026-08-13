"""Deterministic derived features for the Phase 6 ablation.

Every transformer here is **stateless**: ``fit`` records the input schema and
learns nothing. The derived value of a row depends only on that row, so the
transformation is identical on a training fold, a validation fold and, later, a
single record at inference time. None of them reads the target.

Each one *appends* columns. The original features are never removed or
overwritten — Phase 6 asks whether an added representation helps, which is a
different question from feature selection.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

#: Services whose presence the EDA associated with markedly lower churn.
PROTECTIVE_SERVICES: tuple[str, ...] = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
)

#: The only value that counts as holding a protective service. ``No`` and the
#: sentinel ``No internet service`` both count as zero: a customer without
#: internet cannot hold an internet-dependent protection, and the EDA treats
#: that sentinel as a real product state rather than a missing value.
HELD = "Yes"

PROTECTIVE_COUNT = "protective_service_count"

#: Payment methods that debit the customer without a manual action each cycle.
AUTOMATIC_PAYMENT_METHODS: tuple[str, ...] = (
    "Bank transfer (automatic)",
    "Credit card (automatic)",
)

AUTOMATIC_PAYMENT = "automatic_payment"

#: Contract levels that receive an explicit tenure interaction. ``Two year`` is
#: the implicit reference: including all three would make the interactions sum
#: exactly to ``tenure``, which is already in the matrix, so the third column
#: would be an exact linear combination of columns already present.
CONTRACT_REFERENCE = "Two year"
CONTRACT_INTERACTIONS: dict[str, str] = {
    "Month-to-month": "tenure_x_month_to_month",
    "One year": "tenure_x_one_year",
}

HISTORICAL_AVERAGE_CHARGE = "historical_average_charge"


class _AppendingTransformer(BaseEstimator, TransformerMixin):
    """Base for stateless transformers that append columns to a frame."""

    #: Columns this transformer adds, in output order.
    added: tuple[str, ...] = ()

    def fit(self, X: pd.DataFrame, y: object = None) -> _AppendingTransformer:  # noqa: N803
        """Record the input schema. Nothing is estimated, ``y`` is ignored."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return a copy of ``X`` with the derived columns appended."""
        check_is_fitted(self, "feature_names_in_")
        return self._derive(X.copy())

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the input columns followed by the appended ones."""
        check_is_fitted(self, "feature_names_in_")
        return np.asarray([*self.feature_names_in_, *self.added], dtype=object)

    def _derive(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class ProtectiveServiceCount(_AppendingTransformer):
    """Candidate A — how many protective services the customer holds (0..4).

    **Hypothesis.** The EDA found churn falling monotonically as protective
    services accumulate among internet subscribers. The baseline sees the four
    services only as independent dummies, so it can express "holds TechSupport"
    but not "holds three protections". This adds the count as one ordinal
    quantity, which a linear model can turn into a single monotone coefficient.

    It is an *aggregation*, not a replacement: the four source columns stay.
    """

    added = (PROTECTIVE_COUNT,)

    def _derive(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in PROTECTIVE_SERVICES if column not in frame.columns]
        if missing:
            raise KeyError(f"Columns required for {PROTECTIVE_COUNT!r} are absent: {missing}.")
        held = (
            frame[column].astype("string").str.strip().eq(HELD).fillna(False).astype(int)
            for column in PROTECTIVE_SERVICES
        )
        frame[PROTECTIVE_COUNT] = sum(held)
        return frame


class AutomaticPaymentFlag(_AppendingTransformer):
    """Candidate B — does the customer pay automatically?

    **Hypothesis.** The EDA found electronic check churning far above the two
    automatic methods, and the gap survived conditioning on contract type. The
    baseline already one-hot encodes all four methods, so this feature can only
    help if the *manual vs automatic* split is the axis that carries the signal
    and a single pooled coefficient estimates it more stably than four separate
    ones. It is a semantic aggregation of information already present, which is
    exactly what makes it a real test rather than new information.

    ``PaymentMethod`` is kept.
    """

    added = (AUTOMATIC_PAYMENT,)

    def _derive(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "PaymentMethod" not in frame.columns:
            raise KeyError(f"Column 'PaymentMethod' is required for {AUTOMATIC_PAYMENT!r}.")
        method = frame["PaymentMethod"].astype("string").str.strip()
        frame[AUTOMATIC_PAYMENT] = method.isin(AUTOMATIC_PAYMENT_METHODS).fillna(False).astype(int)
        return frame


class ContractTenureInteraction(_AppendingTransformer):
    """Candidate C — let tenure act differently inside each contract type.

    **Hypothesis.** The baseline logistic regression is purely additive: it
    fits one tenure slope shared by every contract. The EDA's contract × tenure
    table showed the churn gradient differing sharply by contract, which an
    additive model cannot express. These two columns give month-to-month and
    one-year customers their own tenure slope.

    ``Two year`` is the implicit reference. A third interaction would satisfy
    ``tenure_x_month_to_month + tenure_x_one_year + tenure_x_two_year == tenure``,
    making it an exact linear combination of columns already in the matrix —
    redundant, and needlessly hard to interpret under L2 regularisation.
    """

    added = tuple(CONTRACT_INTERACTIONS.values())

    def _derive(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("tenure", "Contract"):
            if column not in frame.columns:
                raise KeyError(f"Column {column!r} is required for the contract interactions.")
        tenure = pd.to_numeric(frame["tenure"], errors="coerce")
        contract = frame["Contract"].astype("string").str.strip()
        for level, name in CONTRACT_INTERACTIONS.items():
            frame[name] = tenure.where(contract.eq(level), 0.0).astype("float64")
        return frame


class HistoricalAverageCharge(_AppendingTransformer):
    """Candidate D — accumulated charges per month of relationship.

    **Hypothesis.** ``MonthlyCharges`` is the *current* price while
    ``TotalCharges`` is the accumulated past. The EDA showed the ratio of
    ``TotalCharges`` to ``tenure × MonthlyCharges`` spanning 0.69 to 1.57, so
    the price a customer has actually been paying over the relationship differs
    from the price they pay today. This ratio exposes that history directly
    instead of leaving the model to recover it from three separate columns.

    **What it is not.** It is a descriptive ratio between accumulated billing
    and recorded tenure, not an exact average contractual price: the dataset
    records no billing history, no plan changes and no promotional periods.

    **Tenure zero.** The ratio is mathematically undefined there. ``0.0`` is
    used as the operational representation, coherent with the structural zero
    already assigned to ``TotalCharges`` for those customers: no billing cycle
    has completed, so no history has accumulated. The model still receives
    ``tenure``, so it can tell those rows apart from a genuine average of zero.

    Runs **after** the deterministic ``TotalCharges`` cleanup, so the column is
    already numeric and its blanks already resolved. The target is never read.
    """

    added = (HISTORICAL_AVERAGE_CHARGE,)

    def _derive(self, frame: pd.DataFrame) -> pd.DataFrame:
        for column in ("tenure", "TotalCharges"):
            if column not in frame.columns:
                raise KeyError(f"Column {column!r} is required for {HISTORICAL_AVERAGE_CHARGE!r}.")
        tenure = pd.to_numeric(frame["tenure"], errors="coerce").to_numpy(dtype="float64")
        total = pd.to_numeric(frame["TotalCharges"], errors="coerce").to_numpy(dtype="float64")
        ratio = np.divide(total, tenure, out=np.zeros_like(total), where=tenure > 0)
        frame[HISTORICAL_AVERAGE_CHARGE] = ratio
        return frame
