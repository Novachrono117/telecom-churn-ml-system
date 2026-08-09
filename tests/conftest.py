"""Shared fixtures.

The tests deliberately use a tiny synthetic CSV that *mimics the shape* of the
Telco dataset (identifier, numeric charges, a whitespace-only cell, a constant
column) instead of duplicating the real dataset. Contract tests against the real
file live in ``test_raw_dataset_contract.py`` and skip when it is absent.
"""

from __future__ import annotations

from pathlib import Path

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
