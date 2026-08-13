"""The only stateful candidate feature, and therefore the only one that can leak.

Every test here answers one question: can anything outside the frame passed to
``fit`` influence the learned medians or the transformed output? The real
holdout is never touched — these use controlled synthetic frames, which is the
only way to observe the state directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from churn.features.intensity import CHARGE_INTENSITY, ChargeIntensityByTier
from churn.preprocessing.exceptions import DataQualityError


def _frame(charges: list[float], tiers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"MonthlyCharges": charges, "InternetService": tiers})


def test_intensity_is_the_ratio_to_the_tier_median() -> None:
    frame = _frame([50.0, 100.0, 150.0, 20.0], ["DSL", "DSL", "DSL", "No"])

    result = ChargeIntensityByTier().fit(frame).transform(frame)

    # DSL median is 100.0; the single 'No' row is its own median.
    assert result[CHARGE_INTENSITY].tolist() == [0.5, 1.0, 1.5, 1.0]


def test_medians_are_learned_per_group() -> None:
    frame = _frame([10.0, 30.0, 100.0, 300.0], ["DSL", "DSL", "Fiber optic", "Fiber optic"])

    transformer = ChargeIntensityByTier().fit(frame)

    assert transformer.group_medians_ == {"DSL": 20.0, "Fiber optic": 200.0}


def test_fit_learns_only_from_the_frame_it_receives() -> None:
    """The core leakage guarantee: rows outside `fit` cannot move the medians."""
    train = _frame([10.0, 30.0], ["DSL", "DSL"])
    everything = _frame([10.0, 30.0, 1000.0, 5000.0], ["DSL", "DSL", "DSL", "DSL"])

    on_train = ChargeIntensityByTier().fit(train)
    on_everything = ChargeIntensityByTier().fit(everything)

    assert on_train.group_medians_["DSL"] == 20.0
    assert on_everything.group_medians_["DSL"] != on_train.group_medians_["DSL"]


def test_transform_does_not_change_the_learned_state() -> None:
    train = _frame([10.0, 30.0], ["DSL", "DSL"])
    validation = _frame([900.0, 1100.0], ["DSL", "DSL"])
    transformer = ChargeIntensityByTier().fit(train)
    before = dict(transformer.group_medians_)

    transformer.transform(validation)
    transformer.transform(validation)

    assert transformer.group_medians_ == before
    assert transformer.global_median_ == 20.0


def test_transform_reuses_the_fitted_medians_not_its_own() -> None:
    """A validation fold is divided by the training fold's median."""
    train = _frame([10.0, 30.0], ["DSL", "DSL"])
    validation = _frame([40.0], ["DSL"])

    result = ChargeIntensityByTier().fit(train).transform(validation)

    assert result[CHARGE_INTENSITY].tolist() == [2.0]  # 40 / 20, not 40 / 40


def test_unseen_group_falls_back_to_the_fitted_global_median() -> None:
    train = _frame([10.0, 30.0, 100.0, 300.0], ["DSL", "DSL", "Fiber optic", "Fiber optic"])
    validation = _frame([65.0], ["Satellite"])

    transformer = ChargeIntensityByTier().fit(train)
    result = transformer.transform(validation)

    assert transformer.global_median_ == 65.0
    assert result[CHARGE_INTENSITY].tolist() == [1.0]


def test_fallback_is_learned_state_not_a_recomputation() -> None:
    """The unseen-group value cannot depend on the frame being transformed."""
    train = _frame([10.0, 30.0], ["DSL", "DSL"])
    transformer = ChargeIntensityByTier().fit(train)

    small = transformer.transform(_frame([20.0], ["Satellite"]))
    large = transformer.transform(_frame([20.0, 9999.0], ["Satellite", "Satellite"]))

    assert small[CHARGE_INTENSITY].iloc[0] == large[CHARGE_INTENSITY].iloc[0]


def test_target_is_ignored() -> None:
    frame = _frame([10.0, 30.0, 50.0, 70.0], ["DSL"] * 4)
    zeros = np.zeros(4, dtype=int)
    ones = np.ones(4, dtype=int)

    with_zeros = ChargeIntensityByTier().fit(frame, zeros)
    with_ones = ChargeIntensityByTier().fit(frame, ones)

    assert with_zeros.group_medians_ == with_ones.group_medians_


def test_input_frame_is_not_mutated() -> None:
    frame = _frame([10.0, 30.0], ["DSL", "DSL"])
    before = frame.copy()

    ChargeIntensityByTier().fit(frame).transform(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_transform_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        ChargeIntensityByTier().transform(_frame([10.0], ["DSL"]))


def test_missing_columns_raise() -> None:
    with pytest.raises(KeyError, match="InternetService"):
        ChargeIntensityByTier().fit(pd.DataFrame({"MonthlyCharges": [10.0]}))


def test_non_positive_global_median_is_a_data_quality_failure() -> None:
    frame = _frame([0.0, 0.0], ["DSL", "DSL"])

    with pytest.raises(DataQualityError, match="normalise"):
        ChargeIntensityByTier().fit(frame)


def test_non_positive_group_median_falls_back_to_the_global_one() -> None:
    frame = _frame([0.0, 0.0, 10.0, 30.0, 50.0], ["Free", "Free", "DSL", "DSL", "DSL"])

    transformer = ChargeIntensityByTier().fit(frame)

    assert transformer.group_medians_["Free"] == transformer.global_median_
    assert np.isfinite(transformer.transform(frame)[CHARGE_INTENSITY]).all()


def test_appended_column_is_the_only_addition() -> None:
    frame = _frame([10.0, 30.0], ["DSL", "DSL"])

    result = ChargeIntensityByTier().fit(frame).transform(frame)

    assert list(result.columns) == [*frame.columns, CHARGE_INTENSITY]
