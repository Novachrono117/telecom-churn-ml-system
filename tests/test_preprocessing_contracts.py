"""Tests for the feature contract.

The identifier and target exclusions are regression guards: if a future change
lets either into the feature matrix, these tests fail rather than the leakage
being discovered through a suspiciously good score.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    build_feature_matrix,
    prepare_features,
    target_column,
    validate_feature_matrix,
    validate_source_frame,
)
from churn.preprocessing.exceptions import DataQualityError, FeatureContractError


def test_contract_declares_three_numeric_and_sixteen_categorical_features() -> None:
    assert NUMERIC_FEATURES == ("tenure", "MonthlyCharges", "TotalCharges")
    assert len(CATEGORICAL_FEATURES) == 16
    assert len(FEATURE_COLUMNS) == 19


def test_senior_citizen_is_contracted_as_categorical() -> None:
    assert "SeniorCitizen" in CATEGORICAL_FEATURES
    assert "SeniorCitizen" not in NUMERIC_FEATURES


def test_identifier_is_never_a_feature() -> None:
    assert ID_COLUMN not in FEATURE_COLUMNS


def test_target_is_never_a_feature() -> None:
    assert target_column() not in FEATURE_COLUMNS


def test_build_feature_matrix_excludes_identifier_and_target(
    contract_frame: pd.DataFrame,
) -> None:
    matrix = build_feature_matrix(contract_frame)

    assert ID_COLUMN not in matrix.columns
    assert target_column() not in matrix.columns
    assert list(matrix.columns) == list(FEATURE_COLUMNS)


def test_build_feature_matrix_does_not_modify_the_input(contract_frame: pd.DataFrame) -> None:
    before = contract_frame.copy()

    matrix = build_feature_matrix(contract_frame)
    matrix.loc[matrix.index[0], "tenure"] = 999

    pd.testing.assert_frame_equal(contract_frame, before)


def test_missing_required_feature_is_rejected(contract_frame: pd.DataFrame) -> None:
    incomplete = contract_frame.drop(columns=["Contract"])

    with pytest.raises(FeatureContractError, match="Contract"):
        validate_source_frame(incomplete)


def test_build_feature_matrix_rejects_a_frame_missing_a_feature(
    contract_frame: pd.DataFrame,
) -> None:
    with pytest.raises(FeatureContractError, match="missing"):
        build_feature_matrix(contract_frame.drop(columns=["PaymentMethod"]))


def test_identifier_inside_the_feature_matrix_is_rejected(contract_frame: pd.DataFrame) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix[ID_COLUMN] = contract_frame[ID_COLUMN]

    with pytest.raises(FeatureContractError, match="identifier"):
        validate_feature_matrix(matrix)


def test_target_inside_the_feature_matrix_is_rejected(contract_frame: pd.DataFrame) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix[target_column()] = contract_frame[target_column()]

    with pytest.raises(FeatureContractError, match="leakage"):
        validate_feature_matrix(matrix)


def test_undeclared_column_in_the_feature_matrix_is_rejected(
    contract_frame: pd.DataFrame,
) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix["Country"] = "US"

    with pytest.raises(FeatureContractError, match="outside the contract"):
        validate_feature_matrix(matrix)


def test_incompatible_dtype_for_a_numeric_feature_is_rejected(
    contract_frame: pd.DataFrame,
) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix["tenure"] = matrix["tenure"].astype(str)

    with pytest.raises(FeatureContractError, match="not numeric"):
        validate_feature_matrix(matrix)


# --- column order is not part of the contract ---------------------------------


def test_shuffled_columns_are_accepted(contract_frame: pd.DataFrame) -> None:
    shuffled = build_feature_matrix(contract_frame).loc[:, list(reversed(FEATURE_COLUMNS))]

    validate_feature_matrix(shuffled)


def test_prepare_features_restores_the_canonical_order(contract_frame: pd.DataFrame) -> None:
    shuffled = build_feature_matrix(contract_frame).loc[:, list(reversed(FEATURE_COLUMNS))]

    prepared = prepare_features(shuffled)

    assert list(prepared.columns) == list(FEATURE_COLUMNS)


def test_prepare_features_preserves_every_value(contract_frame: pd.DataFrame) -> None:
    matrix = build_feature_matrix(contract_frame)
    shuffled = matrix.loc[:, list(reversed(FEATURE_COLUMNS))]

    pd.testing.assert_frame_equal(prepare_features(shuffled), matrix)


def test_prepare_features_does_not_modify_its_input(contract_frame: pd.DataFrame) -> None:
    shuffled = build_feature_matrix(contract_frame).loc[:, list(reversed(FEATURE_COLUMNS))]
    before = shuffled.copy()

    prepared = prepare_features(shuffled)
    prepared.loc[prepared.index[0], "tenure"] = 999

    pd.testing.assert_frame_equal(shuffled, before)


def test_prepare_features_still_rejects_an_invalid_matrix(contract_frame: pd.DataFrame) -> None:
    shuffled = build_feature_matrix(contract_frame).loc[:, list(reversed(FEATURE_COLUMNS))]
    shuffled[ID_COLUMN] = contract_frame[ID_COLUMN]

    with pytest.raises(FeatureContractError, match="identifier"):
        prepare_features(shuffled)


# --- unexpected blanks are rejected, sentinel categories are not --------------


@pytest.mark.parametrize("column", ["tenure", "MonthlyCharges"])
def test_missing_numeric_feature_value_is_rejected(
    contract_frame: pd.DataFrame, column: str
) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix.loc[matrix.index[0], column] = float("nan")

    with pytest.raises(DataQualityError, match=column):
        validate_feature_matrix(matrix)


def test_none_in_a_categorical_feature_is_rejected(contract_frame: pd.DataFrame) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix.loc[matrix.index[0], "Contract"] = None

    with pytest.raises(DataQualityError, match="Contract"):
        validate_feature_matrix(matrix)


def test_empty_string_in_a_categorical_feature_is_rejected(contract_frame: pd.DataFrame) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix.loc[matrix.index[0], "PaymentMethod"] = ""

    with pytest.raises(DataQualityError, match="PaymentMethod"):
        validate_feature_matrix(matrix)


def test_whitespace_only_categorical_feature_is_rejected(contract_frame: pd.DataFrame) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix.loc[matrix.index[0], "InternetService"] = "   "

    with pytest.raises(DataQualityError, match="InternetService"):
        validate_feature_matrix(matrix)


def test_error_message_names_the_offending_rows(contract_frame: pd.DataFrame) -> None:
    matrix = build_feature_matrix(contract_frame)
    matrix.index = [f"row-{position}" for position in range(len(matrix))]
    matrix.loc["row-3", "Contract"] = " "

    with pytest.raises(DataQualityError, match="row-3"):
        validate_feature_matrix(matrix)


@pytest.mark.parametrize(
    ("column", "sentinel"),
    [
        ("MultipleLines", "No phone service"),
        ("OnlineSecurity", "No internet service"),
        ("TechSupport", "No internet service"),
        ("StreamingTV", "No internet service"),
    ],
)
def test_sentinel_product_states_are_not_treated_as_blank(
    contract_frame: pd.DataFrame, column: str, sentinel: str
) -> None:
    """`No internet service` is a real category with its own churn rate."""
    matrix = build_feature_matrix(contract_frame)
    matrix[column] = sentinel

    validate_feature_matrix(matrix)


def test_blank_total_charges_is_the_only_accepted_blank(contract_frame: pd.DataFrame) -> None:
    """The structural rule owns `TotalCharges`; the contract must not pre-empt it."""
    matrix = build_feature_matrix(contract_frame)

    assert (matrix["TotalCharges"].astype("string").str.strip() == "").any()
    validate_feature_matrix(matrix)
