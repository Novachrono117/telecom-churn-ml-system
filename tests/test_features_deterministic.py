"""The four deterministic candidate features.

Hand-checkable synthetic rows, not the real dataset: a feature test that depends
on the data cannot distinguish a formula bug from a data change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.features.deterministic import (
    AUTOMATIC_PAYMENT,
    CONTRACT_INTERACTIONS,
    HISTORICAL_AVERAGE_CHARGE,
    PROTECTIVE_COUNT,
    AutomaticPaymentFlag,
    ContractTenureInteraction,
    HistoricalAverageCharge,
    ProtectiveServiceCount,
)

MONTH_TO_MONTH = CONTRACT_INTERACTIONS["Month-to-month"]
ONE_YEAR = CONTRACT_INTERACTIONS["One year"]


def _fit_transform(transformer, frame: pd.DataFrame) -> pd.DataFrame:
    return transformer.fit(frame).transform(frame)


# --- protective service count -------------------------------------------------


def _services(**values: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values)


def test_protective_count_counts_only_explicit_yes() -> None:
    frame = _services(
        OnlineSecurity=["Yes", "No", "No internet service", "Yes"],
        OnlineBackup=["Yes", "No", "No internet service", "No"],
        DeviceProtection=["Yes", "Yes", "No internet service", "No"],
        TechSupport=["Yes", "No", "No internet service", "No"],
    )

    result = _fit_transform(ProtectiveServiceCount(), frame)

    assert result[PROTECTIVE_COUNT].tolist() == [4, 1, 0, 1]


def test_protective_count_treats_sentinel_as_zero() -> None:
    """`No internet service` is a real product state, and it is not a protection."""
    frame = _services(
        OnlineSecurity=["No internet service"],
        OnlineBackup=["No internet service"],
        DeviceProtection=["No internet service"],
        TechSupport=["No internet service"],
    )

    assert _fit_transform(ProtectiveServiceCount(), frame)[PROTECTIVE_COUNT].tolist() == [0]


def test_protective_count_stays_within_zero_and_four(contract_frame: pd.DataFrame) -> None:
    counts = _fit_transform(ProtectiveServiceCount(), contract_frame)[PROTECTIVE_COUNT]

    assert counts.min() >= 0
    assert counts.max() <= 4


def test_protective_count_requires_its_source_columns() -> None:
    with pytest.raises(KeyError, match="TechSupport"):
        _fit_transform(ProtectiveServiceCount(), _services(OnlineSecurity=["Yes"]))


# --- automatic payment --------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("Bank transfer (automatic)", 1),
        ("Credit card (automatic)", 1),
        ("Electronic check", 0),
        ("Mailed check", 0),
    ],
)
def test_automatic_payment_flag(method: str, expected: int) -> None:
    frame = pd.DataFrame({"PaymentMethod": [method]})

    result = _fit_transform(AutomaticPaymentFlag(), frame)

    assert result[AUTOMATIC_PAYMENT].tolist() == [expected]


def test_automatic_payment_is_binary(contract_frame: pd.DataFrame) -> None:
    values = _fit_transform(AutomaticPaymentFlag(), contract_frame)[AUTOMATIC_PAYMENT]

    assert set(values.unique()) <= {0, 1}


def test_automatic_payment_requires_payment_method() -> None:
    with pytest.raises(KeyError, match="PaymentMethod"):
        _fit_transform(AutomaticPaymentFlag(), pd.DataFrame({"other": [1]}))


# --- contract x tenure --------------------------------------------------------


def test_contract_interactions_carry_tenure_only_in_their_own_level() -> None:
    frame = pd.DataFrame(
        {
            "tenure": [10, 20, 30],
            "Contract": ["Month-to-month", "One year", "Two year"],
        }
    )

    result = _fit_transform(ContractTenureInteraction(), frame)

    assert result[MONTH_TO_MONTH].tolist() == [10.0, 0.0, 0.0]
    assert result[ONE_YEAR].tolist() == [0.0, 20.0, 0.0]


def test_two_year_is_the_implicit_reference() -> None:
    """Both interactions are zero for the reference level, by construction."""
    frame = pd.DataFrame({"tenure": [72], "Contract": ["Two year"]})

    result = _fit_transform(ContractTenureInteraction(), frame)

    assert result[MONTH_TO_MONTH].tolist() == [0.0]
    assert result[ONE_YEAR].tolist() == [0.0]


def test_interactions_never_sum_beyond_tenure(contract_frame: pd.DataFrame) -> None:
    """The omitted third interaction is what keeps them from summing to tenure."""
    result = _fit_transform(ContractTenureInteraction(), contract_frame)
    total = result[MONTH_TO_MONTH] + result[ONE_YEAR]

    assert (total <= result["tenure"]).all()
    assert (total < result["tenure"]).any()


def test_contract_interactions_require_their_columns() -> None:
    with pytest.raises(KeyError, match="Contract"):
        _fit_transform(ContractTenureInteraction(), pd.DataFrame({"tenure": [1]}))


# --- historical average charge ------------------------------------------------


def test_historical_average_is_total_over_tenure() -> None:
    frame = pd.DataFrame({"tenure": [10, 4], "TotalCharges": [500.0, 100.0]})

    result = _fit_transform(HistoricalAverageCharge(), frame)

    assert result[HISTORICAL_AVERAGE_CHARGE].tolist() == [50.0, 25.0]


def test_historical_average_is_zero_at_zero_tenure() -> None:
    """Undefined ratio; 0.0 is the operational representation of no history."""
    frame = pd.DataFrame({"tenure": [0], "TotalCharges": [0.0]})

    result = _fit_transform(HistoricalAverageCharge(), frame)

    assert result[HISTORICAL_AVERAGE_CHARGE].tolist() == [0.0]
    assert np.isfinite(result[HISTORICAL_AVERAGE_CHARGE]).all()


def test_historical_average_never_divides_by_zero(contract_frame: pd.DataFrame) -> None:
    from churn.preprocessing.transformers import clean_total_charges

    cleaned = clean_total_charges(contract_frame)

    result = _fit_transform(HistoricalAverageCharge(), cleaned)

    assert np.isfinite(result[HISTORICAL_AVERAGE_CHARGE]).all()


def test_historical_average_requires_its_columns() -> None:
    with pytest.raises(KeyError, match="TotalCharges"):
        _fit_transform(HistoricalAverageCharge(), pd.DataFrame({"tenure": [1]}))


# --- shared behaviour ---------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [ProtectiveServiceCount, AutomaticPaymentFlag, ContractTenureInteraction],
    ids=lambda f: f.__name__,
)
def test_transformers_do_not_mutate_their_input(
    contract_frame: pd.DataFrame, factory: type
) -> None:
    before = contract_frame.copy()

    _fit_transform(factory(), contract_frame)

    pd.testing.assert_frame_equal(contract_frame, before)


@pytest.mark.parametrize(
    "factory",
    [ProtectiveServiceCount, AutomaticPaymentFlag, ContractTenureInteraction],
    ids=lambda f: f.__name__,
)
def test_transformers_keep_every_original_column(
    contract_frame: pd.DataFrame, factory: type
) -> None:
    """This phase adds representation; it never removes an original feature."""
    result = _fit_transform(factory(), contract_frame)

    assert set(contract_frame.columns) <= set(result.columns)


@pytest.mark.parametrize(
    "factory",
    [ProtectiveServiceCount, AutomaticPaymentFlag, ContractTenureInteraction],
    ids=lambda f: f.__name__,
)
def test_transformers_learn_nothing_from_the_fitting_sample(
    contract_frame: pd.DataFrame, factory: type
) -> None:
    """Stateless: the fitting sample cannot change the output."""
    fitted_on_all = factory().fit(contract_frame)
    fitted_on_two_rows = factory().fit(contract_frame.head(2))

    pd.testing.assert_frame_equal(
        fitted_on_all.transform(contract_frame),
        fitted_on_two_rows.transform(contract_frame),
    )


@pytest.mark.parametrize(
    "factory",
    [ProtectiveServiceCount, AutomaticPaymentFlag, ContractTenureInteraction],
    ids=lambda f: f.__name__,
)
def test_transformers_ignore_the_target(contract_frame: pd.DataFrame, factory: type) -> None:
    """No candidate reads Churn; a different y cannot change the output."""
    zeros = np.zeros(len(contract_frame), dtype=int)
    ones = np.ones(len(contract_frame), dtype=int)

    pd.testing.assert_frame_equal(
        factory().fit(contract_frame, zeros).transform(contract_frame),
        factory().fit(contract_frame, ones).transform(contract_frame),
    )
