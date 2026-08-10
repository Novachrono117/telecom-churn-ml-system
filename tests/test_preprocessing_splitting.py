"""Tests for the frozen split and its fingerprints.

These run on the synthetic contract frame, so they hold on a fresh clone. The
checks against the real dataset live in ``test_split_manifest_contract.py``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn.data.loader import RawDatasetMismatchError, verify_raw_dataset
from churn.preprocessing.contracts import ID_COLUMN
from churn.preprocessing.splitting import (
    SplitIntegrityError,
    fingerprint_ids,
    split_dataset,
)


def test_split_is_deterministic(contract_frame: pd.DataFrame) -> None:
    first = split_dataset(contract_frame)
    second = split_dataset(contract_frame)

    pd.testing.assert_frame_equal(first.training, second.training)
    pd.testing.assert_frame_equal(first.holdout, second.holdout)


def test_partitions_do_not_overlap(contract_frame: pd.DataFrame) -> None:
    split = split_dataset(contract_frame)

    assert set(split.training[ID_COLUMN]) & set(split.holdout[ID_COLUMN]) == set()


def test_partitions_cover_the_whole_dataset(contract_frame: pd.DataFrame) -> None:
    split = split_dataset(contract_frame)
    union = set(split.training[ID_COLUMN]) | set(split.holdout[ID_COLUMN])

    assert union == set(contract_frame[ID_COLUMN])
    assert len(split.training) + len(split.holdout) == len(contract_frame)


def test_identifiers_stay_unique_inside_each_partition(contract_frame: pd.DataFrame) -> None:
    split = split_dataset(contract_frame)

    assert split.training[ID_COLUMN].is_unique
    assert split.holdout[ID_COLUMN].is_unique


def test_holdout_receives_the_configured_share(contract_frame: pd.DataFrame) -> None:
    split = split_dataset(contract_frame)

    assert len(split.holdout) == 8  # 20% of 40 rows


def test_split_keeps_the_data_untransformed(contract_frame: pd.DataFrame) -> None:
    """The split must precede every transformation, TotalCharges included."""
    split = split_dataset(contract_frame)

    assert split.training["TotalCharges"].dtype == contract_frame["TotalCharges"].dtype
    assert list(split.training.columns) == list(contract_frame.columns)


def test_split_requires_the_identifier(contract_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match=ID_COLUMN):
        split_dataset(contract_frame.drop(columns=[ID_COLUMN]))


def test_duplicated_identifiers_are_detected(contract_frame: pd.DataFrame) -> None:
    duplicated = contract_frame.copy()
    duplicated[ID_COLUMN] = "same-id"

    with pytest.raises(SplitIntegrityError):
        split_dataset(duplicated)


def test_fingerprint_is_reproducible(contract_frame: pd.DataFrame) -> None:
    first = split_dataset(contract_frame)
    second = split_dataset(contract_frame)

    assert fingerprint_ids(first.training[ID_COLUMN]) == fingerprint_ids(second.training[ID_COLUMN])
    assert fingerprint_ids(first.holdout[ID_COLUMN]) == fingerprint_ids(second.holdout[ID_COLUMN])


def test_fingerprint_ignores_order_but_not_membership() -> None:
    ids = pd.Series(["a", "b", "c"])

    assert fingerprint_ids(ids) == fingerprint_ids(ids.iloc[::-1])
    assert fingerprint_ids(ids) != fingerprint_ids(pd.Series(["a", "b", "d"]))


def test_fingerprint_differs_between_the_two_partitions(contract_frame: pd.DataFrame) -> None:
    split = split_dataset(contract_frame)

    assert fingerprint_ids(split.training[ID_COLUMN]) != fingerprint_ids(split.holdout[ID_COLUMN])


def test_a_changed_raw_file_is_rejected(tmp_path: Path) -> None:
    tampered = tmp_path / "tampered.csv"
    tampered.write_text("customerID,Churn\n0001-AAAA,No\n", encoding="utf-8")

    with pytest.raises(RawDatasetMismatchError, match="does not match the approved"):
        verify_raw_dataset(tampered)
