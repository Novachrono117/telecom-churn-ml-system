"""Target encoding: ``Churn`` labels to a 0/1 integer vector.

The mapping is **declared**, not learned. An encoder that derives its mapping
from the data (``LabelEncoder`` and friends) would silently accept a relabeled
file and flip the meaning of every metric that follows. Here the labels come
from the validated configuration and anything else is an error.
"""

from __future__ import annotations

import logging

import pandas as pd

from churn.config import get_config

logger = logging.getLogger(__name__)


class UnknownTargetLabelError(ValueError):
    """The target column carries a label outside the configured pair."""


def encode_target(values: pd.Series) -> pd.Series:
    """Encode the target as ``negative -> 0`` and ``positive -> 1``.

    The configured labels (``No``/``Yes``, verified in Phase 2 against the real
    file) are the only accepted values. Missing values are rejected too: a
    silent ``NaN`` would be encoded as ``0`` and quietly invent a negative case.

    Args:
        values: Raw target column.

    Returns:
        A new ``int8`` Series with the same index and name. The input is not
        modified.

    Raises:
        UnknownTargetLabelError: If any value is outside the configured labels.
    """
    target = get_config().target
    allowed = (target.negative_label, target.positive_label)

    observed = pd.unique(values)
    unexpected = [value for value in observed if value not in allowed]
    if unexpected:
        raise UnknownTargetLabelError(
            f"Target column {values.name!r} carries label(s) outside the configured "
            f"pair {list(allowed)}: {unexpected}. The mapping is declared in "
            "configs/base.toml and is never inferred from the data."
        )

    encoded = (values == target.positive_label).astype("int8")
    logger.debug("Target encoded: %d rows, %d positive", len(encoded), int(encoded.sum()))
    return encoded
