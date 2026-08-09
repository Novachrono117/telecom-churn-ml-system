"""Loading of the raw Telco Customer Churn CSV.

Two read-only views of the same untouched file are exposed:

* :func:`load_raw_typed` — pandas' own dtype inference, i.e. what a downstream
  step would see by default.
* :func:`load_raw_text` — every column as text with NA conversion disabled, so
  empty strings and whitespace-only cells stay distinguishable from genuinely
  missing values during data-quality inspection.

Neither function modifies the file or its content.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from churn.config import get_config

logger = logging.getLogger(__name__)

#: File name distributed by the Kaggle dataset ``blastchar/telco-customer-churn``.
RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

_MISSING_FILE_HINT = (
    "Raw dataset not found at {path}.\n"
    "It is intentionally not versioned. Acquire it as documented in data/README.md "
    "(Kaggle dataset: blastchar/telco-customer-churn) and place the file there."
)


def raw_csv_path() -> Path:
    """Return the configured absolute path of the raw CSV."""
    return get_config().data.raw_dir / RAW_FILENAME


def _resolve(path: Path | None) -> Path:
    resolved = path or raw_csv_path()
    if not resolved.is_file():
        raise FileNotFoundError(_MISSING_FILE_HINT.format(path=resolved))
    return resolved


def load_raw_typed(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV letting pandas infer dtypes.

    Args:
        path: Override for the CSV location. Defaults to the configured path.

    Returns:
        The dataset exactly as pandas would load it by default.

    Raises:
        FileNotFoundError: If the CSV is not available locally.
    """
    resolved = _resolve(path)
    logger.info("Reading raw CSV with dtype inference: %s", resolved)
    return pd.read_csv(resolved)


def load_raw_text(path: Path | None = None) -> pd.DataFrame:
    """Read the raw CSV as pure text, preserving blanks verbatim.

    ``na_filter=False`` keeps ``""`` and ``" "`` as literal strings instead of
    collapsing them into ``NaN``, which is required to tell "missing" apart from
    "blank" during data-quality checks.

    Args:
        path: Override for the CSV location. Defaults to the configured path.

    Returns:
        The dataset with every column typed as ``object`` (str).

    Raises:
        FileNotFoundError: If the CSV is not available locally.
    """
    resolved = _resolve(path)
    logger.info("Reading raw CSV as text: %s", resolved)
    return pd.read_csv(resolved, dtype=str, keep_default_na=False, na_filter=False)
