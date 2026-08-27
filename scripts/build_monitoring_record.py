"""Phase 12: write (or verify) the machine-readable record of the monitoring layer.

Usage
-----
::

    uv run python scripts/build_monitoring_record.py
    uv run python scripts/build_monitoring_record.py --verify

``--verify`` rebuilds the record in memory from the reference profile and the
operational policy that actually exist, compares it with the file **byte for byte**,
and checks every record invariant. It writes nothing.

Runs in the root environment: the monitoring core imports no web framework, which is
what lets this script — and the whole monitoring test suite — run without one.
"""

from __future__ import annotations

import argparse
import logging

from churn.monitoring.reference import load_reference_profile
from churn.monitoring.results import (
    RECORD_PATH,
    build_record,
    canonical_json,
    read_record,
    record_invariants,
    write_record,
)
from churn.monitoring.settings import load_monitoring_policy

logger = logging.getLogger("build_monitoring_record")


def _build() -> dict[str, object]:
    return build_record(load_reference_profile(), load_monitoring_policy())


def build() -> int:
    record = _build()
    for name, holds, detail in record_invariants(record):
        if not holds:
            logger.error("Invariant %s does not hold (%s); nothing was written.", name, detail)
            return 1
    write_record(record)
    logger.info(
        "MONITORING_AND_DRIFT: reference=%s n=%d minimum_window=%d structural_checks=%d",
        record["reference"]["reference_profile_sha256"][:16],
        record["reference"]["reference_n"],
        record["window"]["minimum_window_size"],
        record["monitoring_scope"]["structural_checks"],
    )
    return 0


def verify() -> int:
    rebuilt = _build()
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
        failures.append(f"the record no longer matches the running system: {differing}")

    for name, holds, detail in record_invariants(rebuilt):
        logger.info("%s %s: %s", "PASS" if holds else "FAIL", name, detail)
        if not holds:
            failures.append(f"invariant {name} does not hold ({detail})")

    if failures:
        for failure in failures:
            logger.error("%s", failure)
        return 1

    logger.info("Record at %s is reproducible and every invariant holds.", RECORD_PATH)
    logger.info(
        "MONITORING_AND_DRIFT: reference=%s n=%d performance_without_labels=%s",
        rebuilt["reference"]["reference_profile_sha256"][:16],
        rebuilt["reference"]["reference_n"],
        rebuilt["monitoring_scope"]["performance_monitoring_without_labels"],
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
