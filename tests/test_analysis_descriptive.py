"""Tests for the descriptive churn statistics."""

from __future__ import annotations

import pandas as pd
import pytest

from churn.analysis.descriptive import (
    churn_rate_by,
    churn_rate_matrix,
    count_yes,
    numeric_summary_by_target,
    overall_churn_rate,
    value_share,
)
from churn.analysis.frames import CHURN_FLAG


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Contract": ["M", "M", "M", "M", "Y", "Y", "Y", "Y"],
            "Support": ["No", "No", "Yes", "Yes", "No", "Yes", "Yes", "Yes"],
            "charges": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, None],
            CHURN_FLAG: [1, 1, 1, 0, 1, 0, 0, 0],
        }
    )


def test_overall_churn_rate(frame: pd.DataFrame) -> None:
    assert overall_churn_rate(frame) == pytest.approx(0.5)


def test_churn_rate_by_reports_size_rate_and_share(frame: pd.DataFrame) -> None:
    table = churn_rate_by(frame, "Contract")

    assert table.loc["M", "n"] == 4
    assert table.loc["M", "churned"] == 3
    assert table.loc["M", "churn_rate"] == pytest.approx(0.75)
    assert table.loc["M", "share"] == pytest.approx(0.5)


def test_churn_rate_by_sorts_by_rate_when_asked(frame: pd.DataFrame) -> None:
    assert churn_rate_by(frame, "Contract").index.tolist() == ["M", "Y"]
    assert churn_rate_by(frame, "Contract", sort=False).index.tolist() == ["M", "Y"]


def test_churn_rate_matrix_returns_aligned_rates_and_counts(frame: pd.DataFrame) -> None:
    rates, counts = churn_rate_matrix(frame, "Contract", "Support")

    assert rates.shape == counts.shape
    assert rates.loc["M", "No"] == pytest.approx(1.0)
    assert counts.loc["M", "No"] == 2
    assert rates.loc["Y", "Yes"] == pytest.approx(0.0)
    assert counts.loc["Y", "Yes"] == 3


def test_numeric_summary_splits_by_class_and_counts_missing(frame: pd.DataFrame) -> None:
    summary = numeric_summary_by_target(frame, "charges")

    assert summary.index.tolist() == ["retained", "churned"]
    assert summary.loc["churned", "n"] == 4
    assert summary.loc["retained", "missing"] == 1
    assert summary.loc["churned", "median"] == pytest.approx(25.0)


def test_count_yes_counts_matching_columns() -> None:
    frame = pd.DataFrame(
        {"a": ["Yes", "No", "Yes"], "b": ["Yes", "Yes", "No"], "c": ["No", "No", "No"]}
    )

    assert count_yes(frame, ["a", "b", "c"]).tolist() == [2, 1, 1]


def test_value_share_can_be_restricted_to_churners(frame: pd.DataFrame) -> None:
    overall = value_share(frame, "Contract")
    churners = value_share(frame, "Contract", among_churners=True)

    assert overall["M"] == pytest.approx(0.5)
    assert churners["M"] == pytest.approx(0.75)
