"""Frozen train/holdout separation.

The split is the **first** operation applied to the raw file: nothing is
coerced, imputed, encoded or scaled before it. Every parameter comes from
``configs/base.toml``, so the partition is a pure function of the raw bytes plus
the configuration and can be regenerated at any time instead of being stored.

Accidental use of the holdout is discouraged structurally rather than by
convention: :func:`load_training_pool` is the entry point for every phase up to
Phase 8 and returns the training rows only. :func:`load_holdout` exists because
Phase 9 needs it, is not exported by the package, and logs a warning when it
runs.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import get_config
from churn.data.loader import load_raw_typed, verify_raw_dataset
from churn.preprocessing.contracts import ID_COLUMN

logger = logging.getLogger(__name__)

#: Fully-qualified name of the split routine, recorded in the manifest.
SPLIT_FUNCTION = "sklearn.model_selection.train_test_split"


class SplitIntegrityError(ValueError):
    """The generated partition violates a structural guarantee."""


@dataclass(frozen=True)
class DatasetSplit:
    """The two partitions of the raw dataset, still completely untransformed."""

    training: pd.DataFrame
    holdout: pd.DataFrame


def fingerprint_ids(ids: pd.Series) -> str:
    """Return a deterministic SHA-256 over a set of identifiers.

    The identifiers are sorted before hashing, so the digest depends on the
    *set* of rows and not on the order ``train_test_split`` happened to emit.
    This is what lets the manifest prove the partition is identical without
    storing thousands of identifiers.

    Args:
        ids: Identifier column of one partition.

    Returns:
        The lowercase hexadecimal digest.
    """
    payload = "\n".join(sorted(str(value) for value in ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_integrity(frame: pd.DataFrame, split: DatasetSplit) -> None:
    """Structural checks only — no distribution of either partition is computed."""
    training_ids = set(split.training[ID_COLUMN])
    holdout_ids = set(split.holdout[ID_COLUMN])

    if len(training_ids) != len(split.training):
        raise SplitIntegrityError("Training partition contains duplicated identifiers.")
    if len(holdout_ids) != len(split.holdout):
        raise SplitIntegrityError("Holdout partition contains duplicated identifiers.")

    overlap = training_ids & holdout_ids
    if overlap:
        raise SplitIntegrityError(
            f"{len(overlap)} identifier(s) appear in both partitions, e.g. {sorted(overlap)[:5]}."
        )

    if training_ids | holdout_ids != set(frame[ID_COLUMN]):
        raise SplitIntegrityError(
            "The union of both partitions does not reproduce the source dataset."
        )

    total = len(split.training) + len(split.holdout)
    if total != len(frame):
        raise SplitIntegrityError(
            f"Partitions hold {total} rows against {len(frame)} in the source."
        )


def split_dataset(frame: pd.DataFrame) -> DatasetSplit:
    """Split a raw frame into the training pool and the frozen holdout.

    Args:
        frame: The raw dataset, untransformed.

    Returns:
        The two partitions, still untransformed.

    Raises:
        KeyError: If the identifier or target column is absent.
        SplitIntegrityError: If the partition violates a structural guarantee.
    """
    config = get_config()
    target = config.target.column

    for required in (ID_COLUMN, target):
        if required not in frame.columns:
            raise KeyError(f"Column {required!r} is required to split the dataset.")

    stratify = frame[target] if config.split.stratify else None
    training, holdout = train_test_split(
        frame,
        test_size=config.split.test_size,
        random_state=config.seed,
        stratify=stratify,
        shuffle=True,
    )
    split = DatasetSplit(training=training, holdout=holdout)
    _check_integrity(frame, split)

    logger.info(
        "Split frozen: %d training rows, %d holdout rows (test_size=%s, seed=%s)",
        len(split.training),
        len(split.holdout),
        config.split.test_size,
        config.seed,
    )
    return split


def load_split(path: Path | None = None) -> DatasetSplit:
    """Verify the raw file and regenerate both partitions.

    Args:
        path: Override for the raw CSV location.

    Returns:
        The regenerated :class:`DatasetSplit`.

    Raises:
        FileNotFoundError: If the raw CSV is absent.
        churn.data.loader.RawDatasetMismatchError: If the raw file is not the
            one approved in Phase 2.
    """
    verify_raw_dataset(path)
    return split_dataset(load_raw_typed(path))


def load_training_pool(path: Path | None = None) -> pd.DataFrame:
    """Return the training rows only — the entry point for Phases 4 to 8.

    The holdout is discarded before returning, so a caller that uses this
    function cannot read it by accident.

    Args:
        path: Override for the raw CSV location.

    Returns:
        The untransformed training partition.
    """
    training = load_split(path).training
    logger.info("Training pool ready: %d rows x %d columns", *training.shape)
    return training


def load_holdout(path: Path | None = None) -> pd.DataFrame:
    """Return the holdout rows. **Phase 9 only.**

    Deliberately absent from ``churn.preprocessing``'s public exports: reaching
    it requires importing this module by name, which makes any use of the
    holdout visible in a diff.

    Args:
        path: Override for the raw CSV location.

    Returns:
        The untransformed holdout partition.
    """
    logger.warning(
        "Holdout partition loaded. It is reserved for the final evaluation (Phase 9); "
        "consulting it earlier invalidates the estimate it is meant to provide."
    )
    return load_split(path).holdout
