"""Structural contract of the real raw dataset.

These tests are skipped while the CSV has not been acquired, so the suite stays
green on a fresh clone. Once the file is present they guard against two real
risks: silently analyzing a different dataset variant, and a schema change
between dataset versions.
"""

from __future__ import annotations

import pytest

from churn.config import get_config
from churn.data.inspection import check_target_assumptions, inspect_dataset
from churn.data.loader import load_raw_text, load_raw_typed, raw_csv_path

pytestmark = pytest.mark.skipif(
    not raw_csv_path().is_file(),
    reason="raw dataset not acquired yet (see data/README.md)",
)

#: Columns exclusive to the enriched IBM Cognos variant, which this project rejects
#: because they are recorded after the churn outcome.
FORBIDDEN_COLUMNS = ("Churn Score", "CLTV", "Churn Reason", "Churn Label", "Customer Status")


@pytest.fixture(scope="module")
def inspection():
    path = raw_csv_path()
    target = get_config().target.column
    return inspect_dataset(load_raw_text(path), load_raw_typed(path), target)


def test_target_column_is_present(inspection) -> None:
    assert get_config().target.column in inspection.columns


def test_target_labels_match_the_configuration(inspection) -> None:
    target = get_config().target
    checks = check_target_assumptions(
        inspection, target.column, target.positive_label, target.negative_label
    )
    contradicted = [check.description for check in checks if not check.confirmed]

    assert contradicted == []


def test_dataset_is_the_single_table_variant(inspection) -> None:
    present = [column for column in FORBIDDEN_COLUMNS if column in inspection.columns]

    assert present == [], f"Enriched IBM Cognos variant detected: {present}"


def test_dataset_has_a_unique_identifier(inspection) -> None:
    assert inspection.identifier_candidates, "no column is unique across all rows"


def test_dataset_has_no_fully_duplicated_rows(inspection) -> None:
    assert inspection.duplicate_rows == 0
