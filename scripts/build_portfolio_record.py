"""Phase 13: write (or verify) the machine-readable record of the portfolio product.

Usage
-----
::

    uv run python scripts/build_portfolio_record.py
    uv run python scripts/build_portfolio_record.py --verify

``--verify`` rebuilds the record in memory from the metadata and the coupling blocks
that actually exist, compares it with the file **byte for byte**, and checks every
record invariant. It writes nothing.

Runs in the root environment: the portfolio core imports no web framework, which is
what lets this script — and the whole portfolio test suite — run without one.
"""

from __future__ import annotations

import argparse
import logging

from churn.portfolio.metadata import load_portfolio_metadata
from churn.portfolio.results import (
    RECORD_PATH,
    build_record,
    canonical_json,
    read_record,
    record_invariants,
    write_record,
)

logger = logging.getLogger("build_portfolio_record")

#: The files the demo is made of. Named here rather than imported from the serving
#: package so this script stays runnable without a web framework installed.
STATIC_ASSETS: tuple[str, ...] = ("index.html", "css/app.css", "js/app.js")


def _build() -> dict[str, object]:
    return build_record(load_portfolio_metadata(), STATIC_ASSETS)


def build() -> int:
    record = _build()
    for name, holds, detail in record_invariants(record):
        if not holds:
            logger.error("Invariant %s does not hold (%s); nothing was written.", name, detail)
            return 1
    write_record(record)
    logger.info(
        "PORTFOLIO_PRODUCT: method=%s ui_default=%s blocks=%d headline=%s",
        record["explanation"]["local_explanation_method"],
        record["product"]["ui_enabled_by_default"],
        record["structural_coupling"]["n_blocks"],
        record["evaluation_shown"]["headline_metrics"],
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

    logger.info("Portfolio record verified: %s", RECORD_PATH)
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
