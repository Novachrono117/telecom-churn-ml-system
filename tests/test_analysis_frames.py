"""Tests for the EDA-only frame helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn.analysis.frames import (
    CHURN_FLAG,
    TENURE_BAND,
    TENURE_BAND_EDGES,
    TENURE_BAND_LABELS,
    add_churn_flag,
    add_tenure_band,
    coerce_total_charges,
    load_eda_frame,
)
from churn.data.loader import load_raw_typed, raw_csv_path


def test_coerce_total_charges_turns_blanks_into_nan(sample_csv: Path) -> None:
    frame = load_raw_typed(sample_csv)

    coerced = coerce_total_charges(frame)

    assert coerced.isna().sum() == 1
    assert coerced.iloc[0] == pytest.approx(29.85)
    assert coerced.iloc[1] == pytest.approx(1889.50)


def test_coerce_total_charges_does_not_mutate_the_input(sample_csv: Path) -> None:
    frame = load_raw_typed(sample_csv)
    before = frame["TotalCharges"].tolist()

    coerce_total_charges(frame)

    assert frame["TotalCharges"].tolist() == before


def test_add_churn_flag_uses_the_configured_positive_label() -> None:
    frame = pd.DataFrame({"Churn": ["Yes", "No", "Yes"]})

    result = add_churn_flag(frame)

    assert result[CHURN_FLAG].tolist() == [1, 0, 1]
    assert CHURN_FLAG not in frame.columns


@pytest.mark.parametrize(
    ("tenure", "expected"),
    [(0, "0-6"), (1, "0-6"), (6, "0-6"), (7, "7-12"), (12, "7-12"), (13, "13-24"), (72, "49-72")],
)
def test_tenure_band_edges_are_inclusive_on_the_right(tenure: int, expected: str) -> None:
    frame = pd.DataFrame({"tenure": [tenure]})

    result = add_tenure_band(frame)

    assert str(result[TENURE_BAND].iloc[0]) == expected


def test_tenure_bands_cover_the_documented_edges() -> None:
    assert len(TENURE_BAND_LABELS) == len(TENURE_BAND_EDGES) - 1
    assert TENURE_BAND_EDGES[0] == 0
    assert TENURE_BAND_EDGES[-1] == 72


@pytest.mark.skipif(not raw_csv_path().is_file(), reason="raw dataset not acquired yet")
def test_load_eda_frame_exposes_numeric_total_charges_and_helpers() -> None:
    frame = load_eda_frame()

    assert pd.api.types.is_numeric_dtype(frame["TotalCharges"])
    assert frame["TotalCharges"].isna().sum() == 11
    assert set(frame[CHURN_FLAG].unique()) == {0, 1}
    assert TENURE_BAND in frame.columns
