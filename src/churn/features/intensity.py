"""Candidate E — charge intensity relative to the customer's own internet tier.

This is the **only stateful** feature in Phase 6, and therefore the only one
that can leak. Its whole point is a group statistic — the median monthly charge
of each ``InternetService`` tier — and a group statistic computed over data the
model should not have seen is a textbook leak.

So the median lookup is learned in ``fit`` and nowhere else:

* ``fit`` reads only the frame it is handed. Inside cross-validation that is the
  training fold, so a validation row cannot influence the medians it is later
  divided by.
* ``transform`` consults only the state stored at fit time. It never recomputes
  anything from the frame being transformed.
* No median from the Phase 3 EDA — which ran on the complete dataset — is
  reused. Those numbers exist in the report and are deliberately not imported.

**Hypothesis.** The EDA found that the marginal ``MonthlyCharges`` gap between
churners and non-churners *reverses* once the internet tier is held fixed:
within DSL and within fiber, churners do not pay more. That makes the raw price
uninterpretable on its own — it mostly encodes which tier the customer bought.
This feature asks the question the raw column cannot: does paying *above the
going rate for your own tier* relate to churn?

**Unseen tier at transform time.** A valid ``InternetService`` value absent from
the training fold has no learned median. Such a row is divided by the **global
median of the training fold**, recorded in ``fit`` for exactly this case. The
fallback is a learned constant, not a recomputation, so it cannot leak either.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from churn.preprocessing.exceptions import DataQualityError

logger = logging.getLogger(__name__)

CHARGE_INTENSITY = "charge_intensity"


class ChargeIntensityByTier(BaseEstimator, TransformerMixin):
    """Divide ``MonthlyCharges`` by the median charge of the customer's tier.

    Args:
        value_column: Charge column to normalise.
        group_column: Column defining the peer group.
    """

    added = (CHARGE_INTENSITY,)

    def __init__(
        self,
        value_column: str = "MonthlyCharges",
        group_column: str = "InternetService",
    ) -> None:
        self.value_column = value_column
        self.group_column = group_column

    def fit(self, X: pd.DataFrame, y: object = None) -> ChargeIntensityByTier:  # noqa: N803
        """Learn the per-group and global medians from ``X`` alone.

        ``y`` is accepted for API compatibility and never read: this feature is
        derived from the features only, so it cannot encode the target.

        Args:
            X: Frame to learn from — a training fold, never the full dataset.
            y: Ignored.

        Returns:
            ``self``.

        Raises:
            KeyError: If either column is absent.
            DataQualityError: If the global median is not strictly positive,
                which would make the ratio meaningless.
        """
        for column in (self.value_column, self.group_column):
            if column not in X.columns:
                raise KeyError(f"Column {column!r} is required for {CHARGE_INTENSITY!r}.")

        values = pd.to_numeric(X[self.value_column], errors="coerce")
        groups = X[self.group_column].astype("string").str.strip()

        global_median = float(values.median())
        if not global_median > 0:
            raise DataQualityError(
                f"The global median of {self.value_column!r} is {global_median}, which cannot "
                f"normalise anything. {CHARGE_INTENSITY!r} assumes strictly positive charges."
            )

        medians = values.groupby(groups, observed=True).median()
        # A non-positive group median would produce an infinite ratio; fall back
        # to the global median, which is already validated above.
        self.group_medians_ = {
            str(name): (float(value) if value > 0 else global_median)
            for name, value in medians.items()
            if pd.notna(value)
        }
        self.global_median_ = global_median
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        logger.debug(
            "%s: learned %d group median(s) from %d rows",
            CHARGE_INTENSITY,
            len(self.group_medians_),
            len(X),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Append the intensity ratio, using only the state learned in ``fit``."""
        check_is_fitted(self, "group_medians_")
        frame = X.copy()
        values = pd.to_numeric(frame[self.value_column], errors="coerce").to_numpy(dtype="float64")
        groups = frame[self.group_column].astype("string").str.strip()
        reference = groups.map(self.group_medians_).fillna(self.global_median_)
        frame[CHARGE_INTENSITY] = values / reference.to_numpy(dtype="float64")
        return frame

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the input columns followed by the appended one."""
        check_is_fitted(self, "feature_names_in_")
        return np.asarray([*self.feature_names_in_, *self.added], dtype=object)
