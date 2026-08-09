"""EDA-only views of the raw dataset.

The transformations here exist **for exploration only**. They are deliberately
not a preprocessing pipeline: nothing is persisted, no value is imputed and the
raw file is never modified. Phase 4 decides how the dataset is really prepared,
after the train/test split.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from churn.config import get_config
from churn.data.loader import load_raw_typed

logger = logging.getLogger(__name__)

CHURN_FLAG = "churn_flag"
TENURE_BAND = "tenure_band"

#: Descriptive tenure bands, in months. The cuts follow the contract cycles the
#: dataset itself offers (month-to-month, one year, two years) rather than equal
#: widths: 0-6 isolates the very first months, 7-12 the rest of the first year,
#: 13-24 the second year, 25-48 the two-to-four-year range and 49-72 the tail up
#: to the observed maximum. They are a *reading aid for charts and tables only*
#: and carry no modeling commitment.
TENURE_BAND_EDGES = (0, 6, 12, 24, 48, 72)
TENURE_BAND_LABELS = ("0-6", "7-12", "13-24", "25-48", "49-72")


def coerce_total_charges(frame: pd.DataFrame, column: str = "TotalCharges") -> pd.Series:
    """Return ``TotalCharges`` as a numeric Series, blanks becoming ``NaN``.

    The 11 whitespace-only cells documented in Phase 2 cannot be read as numbers.
    Coercing them to ``NaN`` is an **analysis convenience**, not the imputation
    decision — that belongs to Phase 4.
    """
    text = frame[column].astype("string").str.strip()
    return pd.to_numeric(text, errors="coerce")


def add_churn_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a 0/1 churn indicator derived from the configured target."""
    target = get_config().target
    result = frame.copy()
    result[CHURN_FLAG] = (result[target.column] == target.positive_label).astype(int)
    return result


def add_tenure_band(frame: pd.DataFrame, column: str = "tenure") -> pd.DataFrame:
    """Return a copy with the descriptive tenure band as an ordered category."""
    result = frame.copy()
    result[TENURE_BAND] = pd.cut(
        result[column],
        bins=list(TENURE_BAND_EDGES),
        labels=list(TENURE_BAND_LABELS),
        include_lowest=True,
        ordered=True,
    )
    return result


def load_eda_frame(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CSV and attach the EDA-only helper columns.

    ``TotalCharges`` is replaced by its numeric coercion, and ``churn_flag`` and
    ``tenure_band`` are added. Nothing is written to disk.
    """
    frame = load_raw_typed(path)
    frame = frame.assign(TotalCharges=coerce_total_charges(frame))
    frame = add_tenure_band(add_churn_flag(frame))
    logger.info("EDA frame ready: %d rows x %d columns", *frame.shape)
    return frame
