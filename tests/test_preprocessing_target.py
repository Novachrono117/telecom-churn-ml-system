"""Tests for the declared target encoding."""

from __future__ import annotations

import pandas as pd
import pytest

from churn.config import get_config
from churn.preprocessing.target import UnknownTargetLabelError, encode_target


def test_positive_label_maps_to_one() -> None:
    target = get_config().target
    encoded = encode_target(pd.Series([target.positive_label]))

    assert encoded.tolist() == [1]


def test_negative_label_maps_to_zero() -> None:
    target = get_config().target
    encoded = encode_target(pd.Series([target.negative_label]))

    assert encoded.tolist() == [0]


def test_mixed_labels_keep_index_and_are_integers(contract_frame: pd.DataFrame) -> None:
    encoded = encode_target(contract_frame["Churn"])

    assert pd.api.types.is_integer_dtype(encoded)
    assert encoded.index.equals(contract_frame.index)
    assert set(encoded.unique()) == {0, 1}


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(UnknownTargetLabelError, match="Maybe"):
        encode_target(pd.Series(["Yes", "No", "Maybe"]))


def test_missing_value_is_rejected_instead_of_becoming_zero() -> None:
    with pytest.raises(UnknownTargetLabelError):
        encode_target(pd.Series(["Yes", None, "No"]))


def test_input_series_is_not_modified() -> None:
    original = pd.Series(["Yes", "No"], name="Churn")
    before = original.copy()

    encode_target(original)

    pd.testing.assert_series_equal(original, before)
