"""Tests for the structural and data-quality inspection."""

from __future__ import annotations

from pathlib import Path

import pytest

from churn.data.inspection import (
    DatasetInspection,
    check_target_assumptions,
    inspect_dataset,
    leakage_suspects,
    target_distribution,
)
from churn.data.loader import load_raw_text, load_raw_typed


@pytest.fixture
def inspection(sample_csv: Path) -> DatasetInspection:
    return inspect_dataset(load_raw_text(sample_csv), load_raw_typed(sample_csv), "Churn")


def _profile(inspection: DatasetInspection, name: str):
    return next(p for p in inspection.profiles if p.name == name)


def test_shape_and_column_order(inspection: DatasetInspection) -> None:
    assert inspection.n_rows == 5
    assert inspection.n_columns == 8
    assert inspection.columns[0] == "customerID"
    assert inspection.columns[-1] == "Churn"


def test_whitespace_cell_is_counted_and_not_confused_with_missing(
    inspection: DatasetInspection,
) -> None:
    profile = _profile(inspection, "TotalCharges")

    assert profile.whitespace_count == 1
    assert profile.blank_count == 0
    assert profile.missing_count == 0


def test_blank_polluted_column_is_still_numeric_apart_from_blanks(
    inspection: DatasetInspection,
) -> None:
    profile = _profile(inspection, "TotalCharges")

    assert profile.is_numeric_coercible
    assert profile.non_numeric_examples == ()
    assert profile.minimum == pytest.approx(29.85)
    assert profile.maximum == pytest.approx(1889.50)


def test_identifier_and_constant_columns_are_detected(inspection: DatasetInspection) -> None:
    # Uniqueness is the observable fact, so on a 5-row sample a continuous column
    # can also be unique. The check is therefore membership, not an exact set.
    assert "customerID" in inspection.identifier_candidates
    assert inspection.constant_columns == ("Country",)


def test_high_cardinality_detects_the_identifier(inspection: DatasetInspection) -> None:
    assert "customerID" in inspection.high_cardinality_columns


def test_no_duplicates_in_the_clean_sample(inspection: DatasetInspection) -> None:
    assert inspection.duplicate_rows == 0


def test_duplicate_rows_are_counted(tmp_path: Path) -> None:
    path = tmp_path / "dupes.csv"
    path.write_text("a,Churn\n1,No\n1,No\n2,Yes\n", encoding="utf-8")

    result = inspect_dataset(load_raw_text(path), load_raw_typed(path), "Churn")

    assert result.duplicate_rows == 1


def test_target_distribution_counts_observed_labels(inspection: DatasetInspection) -> None:
    assert inspection.target_counts == {"No": 3, "Yes": 2}


def test_target_distribution_is_empty_when_column_absent(sample_csv: Path) -> None:
    assert target_distribution(load_raw_text(sample_csv), "Missing") == {}


def test_assumptions_are_confirmed_against_matching_data(inspection: DatasetInspection) -> None:
    checks = check_target_assumptions(inspection, "Churn", "Yes", "No")

    assert all(check.confirmed for check in checks)


def test_assumptions_are_contradicted_when_labels_differ(inspection: DatasetInspection) -> None:
    checks = check_target_assumptions(inspection, "Churn", "1", "0")
    contradicted = [check.description for check in checks if not check.confirmed]

    assert "Positive class label" in contradicted
    assert "Target column name" not in contradicted


def test_leakage_screening_flags_outcome_like_names_but_not_the_target() -> None:
    columns = ("customerID", "tenure", "Churn", "Churn Score", "CLTV")

    assert leakage_suspects(columns, "Churn") == ("Churn Score", "CLTV")


def test_mismatched_views_are_rejected(sample_csv: Path, tmp_path: Path) -> None:
    other = tmp_path / "other.csv"
    other.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="same file"):
        inspect_dataset(load_raw_text(sample_csv), load_raw_typed(other), "Churn")
