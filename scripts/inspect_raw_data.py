"""Run the Phase 2 inspection over the raw dataset and regenerate its artifacts.

Usage (from the repository root):

    uv run python scripts/inspect_raw_data.py

Writes:
    reports/data_understanding.md
    reports/data_dictionary.md
    data/README.md  (provenance section only)

The raw file is opened read-only and never modified.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

from churn.config import get_config
from churn.data.inspection import check_target_assumptions, inspect_dataset
from churn.data.loader import load_raw_text, load_raw_typed, raw_csv_path
from churn.data.provenance import describe_file
from churn.data.reporting import (
    render_data_dictionary,
    render_provenance_block,
    render_report,
    replace_provenance_section,
)

logger = logging.getLogger("inspect_raw_data")

SOURCE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
DATASET_ID = "blastchar/telco-customer-churn"


def main() -> int:
    """Generate the Data Understanding artifacts. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()
    csv_path = raw_csv_path()

    try:
        text_frame = load_raw_text(csv_path)
        typed_frame = load_raw_typed(csv_path)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    provenance = describe_file(csv_path)
    logger.info("SHA-256 %s (%d bytes)", provenance.sha256, provenance.size_bytes)

    inspection = inspect_dataset(text_frame, typed_frame, config.target.column)
    assumptions = check_target_assumptions(
        inspection,
        config.target.column,
        config.target.positive_label,
        config.target.negative_label,
    )
    for check in assumptions:
        if not check.confirmed:
            logger.warning(
                "Assumption contradicted: %s (expected %r, observed %r)",
                check.description,
                check.expected,
                check.observed,
            )

    inspected_on = date.today().isoformat()
    root = Path(config.data.raw_dir).parents[1]
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "data_understanding.md"
    report_path.write_text(
        render_report(inspection, assumptions, provenance, config.target.column, inspected_on),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", report_path)

    dictionary_path = reports_dir / "data_dictionary.md"
    dictionary_path.write_text(
        render_data_dictionary(inspection, config.target.column, provenance),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", dictionary_path)

    data_readme = Path(config.data.raw_dir).parent / "README.md"
    block = render_provenance_block(provenance, SOURCE_URL, DATASET_ID, inspected_on)
    data_readme.write_text(
        replace_provenance_section(data_readme.read_text(encoding="utf-8"), block),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Updated provenance in %s", data_readme)

    return 0


if __name__ == "__main__":
    sys.exit(main())
