"""Tests for the raw dataset loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn.config import get_config
from churn.data.loader import RAW_FILENAME, load_raw_text, load_raw_typed, raw_csv_path


def test_raw_csv_path_comes_from_config_and_is_absolute() -> None:
    path = raw_csv_path()

    assert path.is_absolute()
    assert path.name == RAW_FILENAME
    assert path.parent == get_config().data.raw_dir


def test_load_raw_text_keeps_every_column_as_text(sample_csv: Path) -> None:
    frame = load_raw_text(sample_csv)

    assert list(frame.columns)[0] == "customerID"
    assert all(pd.api.types.is_string_dtype(dtype) for dtype in frame.dtypes)


def test_load_raw_text_preserves_whitespace_instead_of_nan(sample_csv: Path) -> None:
    frame = load_raw_text(sample_csv)

    assert frame["TotalCharges"].iloc[3] == " "
    assert frame["TotalCharges"].isna().sum() == 0


def test_load_raw_typed_infers_numeric_columns(sample_csv: Path) -> None:
    frame = load_raw_typed(sample_csv)

    assert pd.api.types.is_numeric_dtype(frame["MonthlyCharges"])
    assert pd.api.types.is_integer_dtype(frame["tenure"])


def test_load_raw_typed_leaves_blank_polluted_column_non_numeric(sample_csv: Path) -> None:
    frame = load_raw_typed(sample_csv)

    assert not pd.api.types.is_numeric_dtype(frame["TotalCharges"])


def test_both_views_describe_the_same_table(sample_csv: Path) -> None:
    text = load_raw_text(sample_csv)
    typed = load_raw_typed(sample_csv)

    assert list(text.columns) == list(typed.columns)
    assert len(text) == len(typed)


def test_missing_file_raises_with_acquisition_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_raw_text(tmp_path / "absent.csv")
