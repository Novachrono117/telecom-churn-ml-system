"""Phase 14A: write (or verify) the machine-readable record of the academic delivery.

Usage
-----
::

    uv run python scripts/build_academic_record.py
    uv run python scripts/build_academic_record.py --verify

What it records, and what it refuses to record
----------------------------------------------
It reads the academic deliverables that exist in the repository — the notebook, the
report, the checklist, the scripts — and writes what is **observably true** about
them: which sections they carry, which frozen anchors they pin, whether the notebook
reaches for a local path or a credential.

It does **not** record anything about work performed outside this process. Whether the
notebook ran on Google Colab, whether a share link exists, whether a video was
published: none of that is observable from here, so all of it stays at its honest
default and the invariants below fail if anything flips it.

``--verify`` rebuilds the record in memory from the deliverables that actually exist,
compares it with the file **byte for byte**, and checks every invariant. It writes
nothing.
"""

from __future__ import annotations

import argparse
import logging

from churn.academic.results import (
    RECORD_PATH,
    build_record,
    canonical_json,
    read_record,
    record_invariants,
    write_record,
)

logger = logging.getLogger("build_academic_record")


def build() -> int:
    record = build_record()
    for name, holds, detail in record_invariants(record):
        if not holds:
            logger.error("Invariant %s does not hold (%s); nothing was written.", name, detail)
            return 1
    write_record(record)
    logger.info(
        "ACADEMIC_DELIVERY: notebook=%d cells (%d code) report_sections=%d/11 "
        "markers=%d colab_executed=%s submission_ready=%s",
        record["notebook"]["n_cells"],
        record["notebook"]["n_code_cells"],
        len(record["report"]["sections_present"]),
        record["report"]["n_external_input_markers"],
        record["actual_google_colab_execution"],
        record["academic_submission_ready"],
    )
    for action in record["external_actions_remaining"]:
        logger.info("EXTERNAL_ACTION_REQUIRED: %s", action)
    return 0


def verify() -> int:
    rebuilt = build_record()
    try:
        committed = read_record()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    failures: list[str] = []
    if canonical_json(committed) != RECORD_PATH.read_text(encoding="utf-8"):
        failures.append("the record on disk is not in the canonical serialisation")
    if committed != rebuilt:
        differing = sorted(
            key for key in set(committed) | set(rebuilt) if committed.get(key) != rebuilt.get(key)
        )
        failures.append(f"the record no longer matches the repository: {differing}")

    for name, holds, detail in record_invariants(rebuilt):
        logger.info("%s %s: %s", "PASS" if holds else "FAIL", name, detail)
        if not holds:
            failures.append(f"invariant {name} does not hold ({detail})")

    if failures:
        for failure in failures:
            logger.error("%s", failure)
        return 1

    logger.info("Academic delivery record verified: %s", RECORD_PATH)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild in memory and compare byte for byte. Writes nothing.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return verify() if arguments.verify else build()


if __name__ == "__main__":
    raise SystemExit(main())
