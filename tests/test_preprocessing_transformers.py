"""Tests for the deterministic ``TotalCharges`` rule.

The rule is the one place in the phase where a domain judgement was encoded, so
both branches are pinned: the structural zero it accepts, and everything it
refuses to repair silently.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.preprocessing.transformers import (
    DataQualityError,
    TotalChargesCleaner,
    clean_total_charges,
)


def _frame(total: list[object], tenure: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": [50.0] * len(tenure),
            "TotalCharges": total,
        }
    )


def test_whitespace_with_zero_tenure_becomes_structural_zero() -> None:
    cleaned = clean_total_charges(_frame([" "], [0]))

    assert cleaned["TotalCharges"].tolist() == [0.0]


def test_missing_with_zero_tenure_becomes_structural_zero() -> None:
    cleaned = clean_total_charges(_frame([None], [0]))

    assert cleaned["TotalCharges"].tolist() == [0.0]


def test_empty_string_with_zero_tenure_becomes_structural_zero() -> None:
    cleaned = clean_total_charges(_frame([""], [0]))

    assert cleaned["TotalCharges"].tolist() == [0.0]


def test_valid_numeric_text_is_preserved() -> None:
    cleaned = clean_total_charges(_frame(["1889.50", " 108.15 "], [34, 2]))

    assert cleaned["TotalCharges"].tolist() == [1889.50, 108.15]


def test_already_numeric_column_is_preserved() -> None:
    cleaned = clean_total_charges(_frame([1889.5, 108.15], [34, 2]))

    assert cleaned["TotalCharges"].tolist() == [1889.50, 108.15]


def test_output_column_is_float() -> None:
    cleaned = clean_total_charges(_frame([" ", "108.15"], [0, 2]))

    assert cleaned["TotalCharges"].dtype == "float64"


def test_whitespace_with_positive_tenure_raises() -> None:
    with pytest.raises(DataQualityError, match="1 blank with tenure > 0"):
        clean_total_charges(_frame([" "], [5]))


def test_missing_with_positive_tenure_raises() -> None:
    with pytest.raises(DataQualityError, match="1 blank with tenure > 0"):
        clean_total_charges(_frame([None], [5]))


def test_non_numeric_text_with_positive_tenure_raises() -> None:
    with pytest.raises(DataQualityError, match="1 non-numeric"):
        clean_total_charges(_frame(["unknown"], [5]))


def test_non_numeric_text_with_zero_tenure_also_raises() -> None:
    """A structural zero is justified by a *blank*, not by any unreadable value."""
    with pytest.raises(DataQualityError, match="1 non-numeric"):
        clean_total_charges(_frame(["unknown"], [0]))


def test_unreadable_tenure_raises_before_any_decision() -> None:
    with pytest.raises(DataQualityError, match="unreadable 'tenure'"):
        clean_total_charges(_frame([" "], ["n/a"]))


def test_error_message_names_the_offending_rows() -> None:
    frame = _frame(["100.0", " ", "200.0"], [2, 5, 4])
    frame.index = ["a", "b", "c"]

    with pytest.raises(DataQualityError, match="Rows: b"):
        clean_total_charges(frame)


def test_input_frame_is_not_modified() -> None:
    frame = _frame([" ", "108.15"], [0, 2])
    before = frame.copy()

    clean_total_charges(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_missing_column_raises_key_error() -> None:
    frame = pd.DataFrame({"tenure": [1]})

    with pytest.raises(KeyError, match="TotalCharges"):
        clean_total_charges(frame)


def test_transformer_matches_the_function(contract_frame: pd.DataFrame) -> None:
    cleaner = TotalChargesCleaner().fit(contract_frame)

    pd.testing.assert_frame_equal(
        cleaner.transform(contract_frame), clean_total_charges(contract_frame)
    )


def test_transformer_output_does_not_depend_on_what_it_was_fitted_on(
    contract_frame: pd.DataFrame,
) -> None:
    """No statistic is learned, so the fitting sample cannot change the output."""
    fitted_on_all = TotalChargesCleaner().fit(contract_frame)
    fitted_on_three_rows = TotalChargesCleaner().fit(contract_frame.head(3))

    pd.testing.assert_frame_equal(
        fitted_on_all.transform(contract_frame),
        fitted_on_three_rows.transform(contract_frame),
    )


def test_transformer_keeps_the_column_names(contract_frame: pd.DataFrame) -> None:
    cleaner = TotalChargesCleaner().fit(contract_frame)

    assert list(cleaner.get_feature_names_out()) == list(contract_frame.columns)
