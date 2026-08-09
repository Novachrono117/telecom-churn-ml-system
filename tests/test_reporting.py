"""Tests for the Markdown artifacts of the Data Understanding phase."""

from __future__ import annotations

from pathlib import Path

import pytest

from churn.config import PROJECT_ROOT
from churn.data.inspection import DatasetInspection, check_target_assumptions, inspect_dataset
from churn.data.loader import load_raw_text, load_raw_typed
from churn.data.provenance import FileProvenance
from churn.data.reporting import (
    PROVENANCE_END,
    PROVENANCE_START,
    render_data_dictionary,
    render_provenance_block,
    render_report,
    replace_provenance_section,
)

PROVENANCE = FileProvenance(filename="sample.csv", size_bytes=123, sha256="deadbeef")


@pytest.fixture
def inspection(sample_csv: Path) -> DatasetInspection:
    return inspect_dataset(load_raw_text(sample_csv), load_raw_typed(sample_csv), "Churn")


def test_dictionary_lists_every_observed_column(inspection: DatasetInspection) -> None:
    document = render_data_dictionary(inspection, "Churn", PROVENANCE)

    for column in inspection.columns:
        assert f"`{column}`" in document


def test_dictionary_marks_roles_and_source_of_descriptions(inspection: DatasetInspection) -> None:
    document = render_data_dictionary(inspection, "Churn", PROVENANCE)

    assert "identifier" in document
    assert "target" in document
    assert "source doc" in document


def test_dictionary_reports_undocumented_columns_without_guessing(
    inspection: DatasetInspection,
) -> None:
    document = render_data_dictionary(inspection, "Churn", PROVENANCE)

    assert "Not documented by the source" in document


def test_report_contains_the_four_required_sections(inspection: DatasetInspection) -> None:
    checks = check_target_assumptions(inspection, "Churn", "Yes", "No")

    document = render_report(inspection, checks, PROVENANCE, "Churn", "2026-01-01")

    for heading in (
        "## Findings",
        "## Data quality issues",
        "## Leakage review",
        "## Decisions deferred",
    ):
        assert heading in document


def test_report_flags_contradicted_assumptions(inspection: DatasetInspection) -> None:
    checks = check_target_assumptions(inspection, "Churn", "1", "0")

    document = render_report(inspection, checks, PROVENANCE, "Churn", "2026-01-01")

    assert "CONTRADICTED" in document


def test_report_states_the_whitespace_problem(inspection: DatasetInspection) -> None:
    checks = check_target_assumptions(inspection, "Churn", "Yes", "No")

    document = render_report(inspection, checks, PROVENANCE, "Churn", "2026-01-01")

    assert "whitespace-only cell" in document


def test_report_does_not_blame_the_identifier_for_being_textual(
    inspection: DatasetInspection,
) -> None:
    checks = check_target_assumptions(inspection, "Churn", "Yes", "No")

    document = render_report(inspection, checks, PROVENANCE, "Churn", "2026-01-01")
    issues = document.split("## Data quality issues")[1].split("## Leakage review")[0]

    assert "not numerically coercible" not in issues


def test_provenance_section_is_replaced_in_place() -> None:
    document = f"intro\n\n{PROVENANCE_START}\nold\n{PROVENANCE_END}\n\noutro\n"
    block = render_provenance_block(PROVENANCE, "http://example", "owner/dataset", "2026-01-01")

    updated = replace_provenance_section(document, block)

    assert "old" not in updated
    assert updated.startswith("intro")
    assert updated.endswith("outro\n")
    assert "deadbeef" in updated


def test_replacing_provenance_without_markers_raises() -> None:
    with pytest.raises(ValueError, match="markers"):
        replace_provenance_section("no markers here", "block")


def test_data_readme_declares_the_provenance_markers() -> None:
    document = (PROJECT_ROOT / "data" / "README.md").read_text(encoding="utf-8")

    assert PROVENANCE_START in document
    assert PROVENANCE_END in document
