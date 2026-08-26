"""Phase 11: write (or verify) the machine-readable record of the serving boundary.

Usage
-----
Build the record::

    uv run --project serving python scripts/build_serving_record.py

Verify that the committed record still describes the running boundary::

    uv run --project serving python scripts/build_serving_record.py --verify

``--verify`` rebuilds the record in memory from the real frozen artefacts and the
real application, compares it with the file **byte for byte**, and checks every
record invariant. A byte comparison is only meaningful because the record carries
no clock: a file with a generation timestamp would differ on every run and the
check would have to be weakened to something that proves less.

This script must run in the serving environment (``--project serving``): it builds
the FastAPI application in order to read the published routes from it rather than
from a list maintained by hand, and FastAPI is deliberately absent from the root
project.

It trains nothing, loads no dataset, and writes nothing but the record.
"""

from __future__ import annotations

import argparse
import json
import logging

from churn.serving.record import (
    RECORD_PATH,
    build_record,
    read_record,
    record_invariants,
    write_record,
)
from churn.serving.service import ChurnInferenceService

logger = logging.getLogger("build_serving_record")


def _serialise(record: dict[str, object]) -> str:
    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"


def verify() -> int:
    """Compare the committed record with a fresh build. Returns a process exit code."""
    service = ChurnInferenceService.from_settings()
    rebuilt = build_record(service)

    try:
        committed = read_record()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    failures: list[str] = []

    if _serialise(committed) != RECORD_PATH.read_text(encoding="utf-8"):
        failures.append("the record on disk is not in the canonical serialisation")
    if committed != rebuilt:
        differing = sorted(
            key for key in set(committed) | set(rebuilt) if committed.get(key) != rebuilt.get(key)
        )
        failures.append(f"the record no longer matches the running boundary: {differing}")

    for name, holds, detail in record_invariants(rebuilt):
        status = "PASS" if holds else "FAIL"
        logger.info("%s %s: %s", status, name, detail)
        if not holds:
            failures.append(f"invariant {name} does not hold ({detail})")

    if failures:
        for failure in failures:
            logger.error("%s", failure)
        return 1

    logger.info("Record at %s is reproducible and every invariant holds.", RECORD_PATH)
    logger.info(
        "PRODUCTION_INFERENCE_API: model %s, threshold %r %s, calibration %s, %d startup gates",
        rebuilt["frozen_model"]["model_fingerprint"][:16],
        rebuilt["decision_policy"]["threshold"],
        rebuilt["decision_policy"]["comparison"],
        rebuilt["decision_policy"]["calibration_policy"],
        rebuilt["startup"]["n_gates"],
    )
    return 0


def build() -> int:
    """Write the record. Returns a process exit code."""
    service = ChurnInferenceService.from_settings()
    record = build_record(service)

    for name, holds, detail in record_invariants(record):
        if not holds:
            logger.error("Invariant %s does not hold (%s); nothing was written.", name, detail)
            return 1

    write_record(record)
    logger.info(
        "PRODUCTION_INFERENCE_API: model %s, threshold %r %s, calibration %s, %d startup gates",
        record["frozen_model"]["model_fingerprint"][:16],
        record["decision_policy"]["threshold"],
        record["decision_policy"]["comparison"],
        record["decision_policy"]["calibration_policy"],
        record["startup"]["n_gates"],
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild the record in memory and compare it with the committed file.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return verify() if arguments.verify else build()


if __name__ == "__main__":
    raise SystemExit(main())
