"""Tests for the preprocessing pipeline.

The leakage tests are the important ones: they pin that the fitted state is a
function of the fitting sample and nothing else, which is what makes "fit on the
training pool only" a guarantee rather than a comment. They do this by comparing
the fitted parameters against statistics recomputed from the *training* rows —
never by fitting anything on data reserved for evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_feature_matrix,
    prepare_features,
)
from churn.preprocessing.pipeline import (
    CATEGORICAL_STEP,
    ENCODER_STEP,
    NUMERIC_STEP,
    build_preprocessor,
    feature_names_out,
)
from churn.preprocessing.splitting import split_dataset
from churn.preprocessing.transformers import clean_total_charges


@pytest.fixture
def training_matrix(contract_frame: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix of the training pool — the holdout is discarded here."""
    return build_feature_matrix(split_dataset(contract_frame).training)


def test_fit_and_transform_produce_a_numeric_matrix(training_matrix: pd.DataFrame) -> None:
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(training_matrix)

    assert transformed.shape[0] == len(training_matrix)
    assert np.issubdtype(transformed.dtype, np.floating)


def test_transformed_matrix_has_no_missing_values(training_matrix: pd.DataFrame) -> None:
    transformed = build_preprocessor().fit_transform(training_matrix)

    assert not np.isnan(transformed).any()


def test_output_is_dense(training_matrix: pd.DataFrame) -> None:
    transformed = build_preprocessor().fit_transform(training_matrix)

    assert isinstance(transformed, np.ndarray)


def test_feature_names_match_the_number_of_columns(training_matrix: pd.DataFrame) -> None:
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(training_matrix)
    names = feature_names_out(preprocessor)

    assert len(names) == transformed.shape[1]
    assert all(name.startswith(("numeric__", "categorical__")) for name in names)


def test_numeric_features_keep_their_own_columns(training_matrix: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(training_matrix)
    names = feature_names_out(preprocessor)

    for column in NUMERIC_FEATURES:
        assert f"numeric__{column}" in names


def test_every_categorical_feature_is_expanded(training_matrix: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(training_matrix)
    names = feature_names_out(preprocessor)

    for column in CATEGORICAL_FEATURES:
        assert any(name.startswith(f"categorical__{column}_") for name in names)


def test_shuffled_input_columns_produce_identical_output(training_matrix: pd.DataFrame) -> None:
    """A caller's column order must not change what the model sees."""
    preprocessor = build_preprocessor().fit(training_matrix)
    shuffled = training_matrix.loc[:, list(reversed(FEATURE_COLUMNS))]

    np.testing.assert_allclose(
        preprocessor.transform(prepare_features(shuffled)),
        preprocessor.transform(training_matrix),
    )


def test_fitting_on_shuffled_columns_produces_identical_output(
    training_matrix: pd.DataFrame,
) -> None:
    shuffled = prepare_features(training_matrix.loc[:, list(reversed(FEATURE_COLUMNS))])

    np.testing.assert_allclose(
        build_preprocessor().fit_transform(shuffled),
        build_preprocessor().fit_transform(training_matrix),
    )
    assert feature_names_out(build_preprocessor().fit(shuffled)) == feature_names_out(
        build_preprocessor().fit(training_matrix)
    )


def test_unseen_category_at_transform_time_does_not_raise(training_matrix: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(training_matrix)
    unseen = training_matrix.head(1).copy()
    unseen.loc[unseen.index[0], "Contract"] = "Three year"

    transformed = preprocessor.transform(unseen)

    assert not np.isnan(transformed).any()


def test_unseen_category_is_encoded_as_all_zeros(training_matrix: pd.DataFrame) -> None:
    preprocessor = build_preprocessor().fit(training_matrix)
    names = feature_names_out(preprocessor)
    contract_columns = [
        index for index, name in enumerate(names) if name.startswith("categorical__Contract_")
    ]
    unseen = training_matrix.head(1).copy()
    unseen.loc[unseen.index[0], "Contract"] = "Three year"

    transformed = preprocessor.transform(unseen)

    assert transformed[0, contract_columns].sum() == 0.0


def test_scaler_parameters_come_only_from_the_rows_passed_to_fit(
    contract_frame: pd.DataFrame,
) -> None:
    """The fitted mean and scale must equal the training rows' own statistics.

    Nothing is fitted on, or measured from, the rows reserved for evaluation:
    the expected values are recomputed from the training partition alone.
    """
    training = build_feature_matrix(split_dataset(contract_frame).training)
    preprocessor = build_preprocessor().fit(training)
    scaler = preprocessor.named_steps[ENCODER_STEP].named_transformers_[NUMERIC_STEP]

    expected = clean_total_charges(training)[list(NUMERIC_FEATURES)].to_numpy(dtype=float)

    np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, expected.std(axis=0))


def test_transformed_training_rows_are_standardized_by_their_own_statistics(
    contract_frame: pd.DataFrame,
) -> None:
    """Equivalent statement, checked on the output rather than the parameters."""
    training = build_feature_matrix(split_dataset(contract_frame).training)
    transformed = build_preprocessor().fit_transform(training)
    numeric_block = transformed[:, : len(NUMERIC_FEATURES)]

    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(numeric_block.std(axis=0), 1.0, atol=1e-9)


def test_categories_come_only_from_the_fitting_sample() -> None:
    frame = pd.DataFrame(
        {
            "tenure": [1, 2],
            "MonthlyCharges": [10.0, 20.0],
            "TotalCharges": ["10.00", "40.00"],
            **{column: ["A", "A"] for column in CATEGORICAL_FEATURES},
        }
    )[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]

    preprocessor = build_preprocessor().fit(frame)
    encoder = preprocessor.named_steps[ENCODER_STEP].named_transformers_[CATEGORICAL_STEP]

    assert all(list(categories) == ["A"] for categories in encoder.categories_)


def test_feature_names_require_a_fitted_pipeline() -> None:
    with pytest.raises(NotFittedError):
        feature_names_out(build_preprocessor())
