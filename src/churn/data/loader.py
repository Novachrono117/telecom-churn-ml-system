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
from churn.data.provenance import compute_sha256

logger = logging.getLogger(__name__)

#: File name distributed by the Kaggle dataset ``blastchar/telco-customer-churn``.
RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

#: SHA-256 of the exact file approved in Phase 2 (7043 x 21, 977501 bytes).
#: Every result recorded in this repository was produced from these bytes; any
#: other content invalidates the frozen split and the reported numbers.
EXPECTED_RAW_SHA256 = "88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"

_MISSING_FILE_HINT = (
    "Raw dataset not found at {path}.\n"
    "It is intentionally not versioned. Acquire it as documented in data/README.md "
    "(Kaggle dataset: blastchar/telco-customer-churn) and place the file there."
)


class RawDatasetMismatchError(ValueError):
    """The raw file on disk is not the dataset approved in Phase 2."""


def raw_csv_path() -> Path:
    """Return the configured absolute path of the raw CSV."""
    return get_config().data.raw_dir / RAW_FILENAME


def verify_raw_dataset(path: Path | None = None) -> str:
    """Check that the raw CSV is byte-identical to the dataset approved in Phase 2.

    Args:
        path: Override for the CSV location. Defaults to the configured path.

    Returns:
        The verified SHA-256 digest, so callers can record it in a manifest.

    Raises:
        FileNotFoundError: If the CSV is not available locally.
        RawDatasetMismatchError: If the digest differs from
            :data:`EXPECTED_RAW_SHA256`.
    """
    resolved = _resolve(path)
    digest = compute_sha256(resolved)
    if digest != EXPECTED_RAW_SHA256:
        raise RawDatasetMismatchError(
            f"Raw dataset at {resolved} does not match the approved Phase 2 file.\n"
            f"expected SHA-256: {EXPECTED_RAW_SHA256}\n"
            f"observed SHA-256: {digest}\n"
            "Re-acquire the dataset as documented in data/README.md; results produced "
            "from a different file are not comparable with the recorded ones."
        )
    logger.info("Raw dataset verified against the approved SHA-256: %s", resolved)
    return digest


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
