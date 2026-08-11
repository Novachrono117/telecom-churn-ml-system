"""Cross-validation protocol: fold isolation, shared folds, OOF coverage.

The fold-isolation tests use spy estimators on the synthetic contract frame
rather than the real dataset. A leakage guarantee has to be proven by observing
*what the estimator was handed*, which the real data cannot show, and Phase 4's
holdout is off limits to tests as much as to scripts.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from churn.config import get_config
from churn.modeling.evaluation import (
    N_SPLITS,
    build_splitter,
    evaluate_model,
    evaluate_models,
)
from churn.modeling.models import build_baselines
from churn.preprocessing.contracts import NUMERIC_FEATURES, build_feature_matrix
from churn.preprocessing.pipeline import build_preprocessor
from churn.preprocessing.target import encode_target


class IndexSpy(ClassifierMixin, BaseEstimator):
    """Records the row labels it is fitted on and asked to predict."""

    records: ClassVar[list[dict[str, set]]] = []

    def fit(self, X, y):  # noqa: N803
        self.classes_ = np.unique(y)
        IndexSpy.records.append({"fit": set(X.index), "predict": set()})
        return self

    def predict_proba(self, X):  # noqa: N803
        IndexSpy.records[-1]["predict"] = set(X.index)
        return np.tile([0.4, 0.6], (len(X), 1))


class ArraySpy(ClassifierMixin, BaseEstimator):
    """Records the transformed arrays handed to it, per fold."""

    records: ClassVar[list[dict[str, np.ndarray]]] = []

    def fit(self, X, y):  # noqa: N803
        self.classes_ = np.unique(y)
        ArraySpy.records.append({"fit": np.asarray(X), "predict": np.empty((0, 0))})
        return self

    def predict_proba(self, X):  # noqa: N803
        ArraySpy.records[-1]["predict"] = np.asarray(X)
        return np.tile([0.4, 0.6], (len(X), 1))


@pytest.fixture
def training_xy(contract_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = build_feature_matrix(contract_frame)
    target = encode_target(contract_frame["Churn"])
    return features, target


@pytest.fixture
def index_spy() -> type[IndexSpy]:
    IndexSpy.records = []
    return IndexSpy


@pytest.fixture
def array_spy() -> type[ArraySpy]:
    ArraySpy.records = []
    return ArraySpy


# --- splitter -----------------------------------------------------------------


def test_splitter_uses_the_configured_policy() -> None:
    splitter = build_splitter()

    assert splitter.n_splits == N_SPLITS
    assert splitter.shuffle is True
    assert splitter.random_state == get_config().seed


def test_splitter_is_deterministic(training_xy) -> None:
    features, target = training_xy
    splitter = build_splitter()

    first = [(tuple(a), tuple(b)) for a, b in splitter.split(features, target)]
    second = [(tuple(a), tuple(b)) for a, b in splitter.split(features, target)]

    assert first == second


def test_every_model_sees_exactly_the_same_folds(training_xy, index_spy) -> None:
    """The comparison must be paired: one model cannot draw an easier split."""
    features, target = training_xy
    splitter = build_splitter()

    evaluate_model("a", index_spy(), features, target, splitter)
    first = [record["predict"] for record in index_spy.records]
    index_spy.records = []
    evaluate_model("b", index_spy(), features, target, splitter)
    second = [record["predict"] for record in index_spy.records]

    assert first == second


# --- fold isolation -----------------------------------------------------------


def test_no_row_is_predicted_by_a_model_trained_on_it(training_xy, index_spy) -> None:
    features, target = training_xy

    evaluate_model("spy", index_spy(), features, target, build_splitter())

    assert len(index_spy.records) == N_SPLITS
    for number, record in enumerate(index_spy.records, start=1):
        assert record["fit"] & record["predict"] == set(), f"fold {number} leaked rows"


def test_validation_folds_partition_the_input_exactly_once(training_xy, index_spy) -> None:
    features, target = training_xy

    evaluate_model("spy", index_spy(), features, target, build_splitter())

    predicted = [label for record in index_spy.records for label in record["predict"]]

    assert len(predicted) == len(features)
    assert set(predicted) == set(features.index)


def test_preprocessing_is_refitted_inside_each_fold(training_xy, array_spy) -> None:
    """The scaler's statistics must come from the training fold, not the pool."""
    features, target = training_xy
    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", array_spy())])

    evaluate_model("spy", pipeline, features, target, build_splitter())

    numeric = slice(0, len(NUMERIC_FEATURES))
    for number, record in enumerate(array_spy.records, start=1):
        # Standardized by its own statistics: the training fold is centred.
        assert np.allclose(record["fit"][:, numeric].mean(axis=0), 0.0, atol=1e-9), number
        # The validation fold was transformed with the training fold's statistics,
        # so it is not centred on itself.
        assert not np.allclose(record["predict"][:, numeric].mean(axis=0), 0.0, atol=1e-9), number


def test_folds_do_not_share_learned_state(training_xy, array_spy) -> None:
    """Each fold starts from a fresh clone, so the scalers cannot agree exactly."""
    features, target = training_xy
    pipeline = Pipeline([("preprocessor", build_preprocessor()), ("classifier", array_spy())])

    evaluate_model("spy", pipeline, features, target, build_splitter())

    first, second = array_spy.records[0]["predict"], array_spy.records[1]["predict"]

    assert not np.array_equal(first, second)


# --- out-of-fold predictions --------------------------------------------------


def test_out_of_fold_has_one_prediction_per_row(training_xy) -> None:
    features, target = training_xy

    evaluation = evaluate_model(
        "logistic", build_baselines()["logistic_regression"], features, target, build_splitter()
    )

    assert evaluation.oof_probability.shape == (len(features),)
    assert not np.isnan(evaluation.oof_probability).any()


def test_out_of_fold_probabilities_are_valid(training_xy) -> None:
    features, target = training_xy

    evaluation = evaluate_model(
        "logistic", build_baselines()["logistic_regression"], features, target, build_splitter()
    )

    assert evaluation.oof_probability.min() >= 0.0
    assert evaluation.oof_probability.max() <= 1.0


def test_evaluation_records_one_result_per_fold(training_xy) -> None:
    features, target = training_xy

    evaluation = evaluate_model(
        "majority", build_baselines()["majority"], features, target, build_splitter()
    )

    assert len(evaluation.folds) == N_SPLITS
    assert [fold.fold for fold in evaluation.folds] == list(range(1, N_SPLITS + 1))
    assert all(fold.n_train + fold.n_validation == len(features) for fold in evaluation.folds)


# --- aggregation --------------------------------------------------------------


def test_mean_and_std_match_the_per_fold_values(training_xy) -> None:
    features, target = training_xy

    evaluation = evaluate_model(
        "logistic", build_baselines()["logistic_regression"], features, target, build_splitter()
    )
    values = evaluation.metric_values("roc_auc")

    assert evaluation.mean("roc_auc") == pytest.approx(values.mean())
    assert evaluation.std("roc_auc") == pytest.approx(values.std(ddof=1))


def test_summary_covers_every_metric_with_mean_and_std(training_xy) -> None:
    features, target = training_xy

    summary = evaluate_model(
        "majority", build_baselines()["majority"], features, target, build_splitter()
    ).summary()

    assert set(summary) == {"roc_auc", "average_precision", "precision", "recall", "f1", "accuracy"}
    assert all(set(entry) == {"mean", "std"} for entry in summary.values())


def test_evaluate_models_preserves_input_order(training_xy) -> None:
    features, target = training_xy
    models = build_baselines()

    results = evaluate_models(models, features, target, build_splitter())

    assert list(results) == list(models)
