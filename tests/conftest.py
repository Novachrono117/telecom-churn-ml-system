"""Shared fixtures.

The tests deliberately use a tiny synthetic CSV that *mimics the shape* of the
Telco dataset (identifier, numeric charges, a whitespace-only cell, a constant
column) instead of duplicating the real dataset. Contract tests against the real
file live in ``test_raw_dataset_contract.py`` and skip when it is absent.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pandas as pd
import pytest

SAMPLE_CSV = (
    "customerID,gender,SeniorCitizen,tenure,MonthlyCharges,TotalCharges,Country,Churn\n"
    "0001-AAAA,Female,0,1,29.85,29.85,US,No\n"
    "0002-BBBB,Male,0,34,56.95,1889.50,US,No\n"
    "0003-CCCC,Male,1,2,53.85,108.15,US,Yes\n"
    "0004-DDDD,Female,0,0,45.30, ,US,No\n"
    "0005-EEEE,Male,0,45,42.30,1840.75,US,Yes\n"
)


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write the synthetic dataset to a temporary file and return its path."""
    path = tmp_path / "sample.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return path


#: Row count of :func:`contract_frame`. Large enough that a stratified 20% draw
#: keeps several rows of each class, small enough to stay readable.
CONTRACT_ROWS = 40

_CATEGORY_VALUES: dict[str, tuple[object, ...]] = {
    "gender": ("Female", "Male"),
    "SeniorCitizen": (0, 1),
    "Partner": ("Yes", "No"),
    "Dependents": ("No", "Yes"),
    "PhoneService": ("Yes", "No"),
    "MultipleLines": ("No", "Yes", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": ("No", "Yes", "No internet service"),
    "OnlineBackup": ("Yes", "No", "No internet service"),
    "DeviceProtection": ("No", "Yes", "No internet service"),
    "TechSupport": ("Yes", "No", "No internet service"),
    "StreamingTV": ("No", "Yes", "No internet service"),
    "StreamingMovies": ("Yes", "No", "No internet service"),
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": ("Yes", "No"),
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}

#: Tenure pattern. The zeros are what the structural `TotalCharges` rule needs;
#: they sit at positions 0, 8, 16, 24 and 32.
_TENURE_PATTERN = (0, 1, 12, 24, 36, 48, 60, 72)
_MONTHLY_PATTERN = (19.70, 45.30, 70.35, 89.10, 105.50)

#: 3 retained for each churner, so the fixture exercises stratification.
_CHURN_PATTERN = ("No", "No", "No", "Yes")


def _cycle(values: tuple[object, ...], size: int) -> list[object]:
    return list(itertools.islice(itertools.cycle(values), size))


def make_contract_frame(rows: int = CONTRACT_ROWS) -> pd.DataFrame:
    """Build a synthetic frame with the exact Phase 4 schema.

    Deterministic by construction — no random generator — so every test that
    depends on it is reproducible. ``TotalCharges`` is text, as it is in the raw
    CSV, and is whitespace-only exactly where ``tenure == 0``.
    """
    tenure = _cycle(_TENURE_PATTERN, rows)
    monthly = _cycle(_MONTHLY_PATTERN, rows)
    total = [" " if t == 0 else f"{t * m:.2f}" for t, m in zip(tenure, monthly, strict=True)]

    data: dict[str, object] = {
        "customerID": [f"{index:04d}-TEST" for index in range(rows)],
        "gender": _cycle(_CATEGORY_VALUES["gender"], rows),
        "SeniorCitizen": _cycle(_CATEGORY_VALUES["SeniorCitizen"], rows),
        "Partner": _cycle(_CATEGORY_VALUES["Partner"], rows),
        "Dependents": _cycle(_CATEGORY_VALUES["Dependents"], rows),
        "tenure": tenure,
        "PhoneService": _cycle(_CATEGORY_VALUES["PhoneService"], rows),
        "MultipleLines": _cycle(_CATEGORY_VALUES["MultipleLines"], rows),
        "InternetService": _cycle(_CATEGORY_VALUES["InternetService"], rows),
        "OnlineSecurity": _cycle(_CATEGORY_VALUES["OnlineSecurity"], rows),
        "OnlineBackup": _cycle(_CATEGORY_VALUES["OnlineBackup"], rows),
        "DeviceProtection": _cycle(_CATEGORY_VALUES["DeviceProtection"], rows),
        "TechSupport": _cycle(_CATEGORY_VALUES["TechSupport"], rows),
        "StreamingTV": _cycle(_CATEGORY_VALUES["StreamingTV"], rows),
        "StreamingMovies": _cycle(_CATEGORY_VALUES["StreamingMovies"], rows),
        "Contract": _cycle(_CATEGORY_VALUES["Contract"], rows),
        "PaperlessBilling": _cycle(_CATEGORY_VALUES["PaperlessBilling"], rows),
        "PaymentMethod": _cycle(_CATEGORY_VALUES["PaymentMethod"], rows),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": _cycle(_CHURN_PATTERN, rows),
    }
    return pd.DataFrame(data)


@pytest.fixture
def contract_frame() -> pd.DataFrame:
    """A synthetic frame carrying the full Phase 4 schema."""
    return make_contract_frame()


#: Rows of :func:`comparison_frame`. Twice :data:`CONTRACT_ROWS`, so a stratified
#: 5-fold split still leaves several churners in every validation fold — a tree
#: ensemble fitted on two positives says nothing about anything.
COMPARISON_ROWS = 80


@pytest.fixture(scope="module")
def comparison_frame() -> pd.DataFrame:
    """A larger synthetic frame, module-scoped for cross-validation tests."""
    return make_contract_frame(COMPARISON_ROWS)
