"""Phase 13: write (or verify) the versioned metadata the demo is allowed to display.

Usage
-----
::

    uv run python scripts/build_portfolio_metadata.py
    uv run python scripts/build_portfolio_metadata.py --verify

What it does, and what it deliberately does not
-----------------------------------------------
It reads two artefacts that are already committed — ``holdout_results.json`` and
``decision_policy.json`` — and copies the numbers a portfolio page may show into one
small, validated, deterministic file.

It does **not** open the holdout, load any partition, score anything or recompute a
metric. Phase 9D evaluated the frozen model once, under the frozen protocol, and this
script's whole reason to exist is that the count stays at one: the serving process
reads this summary instead of Phase 9D's record, so nothing in the runtime is even in
a position to re-evaluate.

The analyst-exposure limitation is a field of the artefact rather than a line in the
HTML, which means the metrics cannot be rendered without it.

``--verify`` rebuilds the summary in memory from the artefacts that actually exist,
compares it with the file **byte for byte**, checks every invariant, and checks the
digest against ``EXPECTED_PORTFOLIO_METADATA_SHA256`` — the constant the serving layer
enforces at startup. It writes nothing.

The pin is deliberately checked here as well as at startup so that rebuilding the
summary and forgetting to move the constant fails offline, in the same command that
produced the drift, rather than at the next deployment.
"""

from __future__ import annotations

import argparse
import logging

from churn.portfolio.metadata import (
    EXPECTED_PORTFOLIO_METADATA_SHA256,
    PORTFOLIO_METADATA_TRUST_ANCHOR,
    build_metadata,
    canonical_json,
    default_metadata_path,
    load_portfolio_metadata,
    metadata_digest,
    verify_portfolio_metadata,
    write_portfolio_metadata,
)

logger = logging.getLogger("build_portfolio_metadata")


def build() -> int:
    metadata = build_metadata()
    for name, holds, detail in verify_portfolio_metadata(metadata):
        if not holds:
            logger.error("Invariant %s does not hold (%s); nothing was written.", name, detail)
            return 1
    write_portfolio_metadata(metadata)
    evaluation = metadata.evaluation
    digest = metadata_digest(metadata)
    logger.info(
        "PORTFOLIO_METADATA: n=%d evaluations=%d headline=%s holdout_reopened=%s sha256=%s",
        evaluation.n_samples,
        evaluation.evaluations_performed,
        [entry.key for entry in evaluation.headline],
        metadata.provenance.holdout_reopened,
        digest,
    )
    # Building is allowed to move the digest; the pin is not updated here, because a
    # script that rewrote its own expectation would be no expectation at all. The
    # operator moves the constant deliberately, and --verify is what refuses until
    # they do.
    if digest != EXPECTED_PORTFOLIO_METADATA_SHA256:
        logger.warning(
            "The summary was written but its digest is not the pinned one. Update %s "
            "to %s, or the serving layer will refuse to start with the demo enabled.",
            PORTFOLIO_METADATA_TRUST_ANCHOR,
            digest,
        )
    return 0


def verify() -> int:
    rebuilt = build_metadata()
    path = default_metadata_path()
    try:
        committed = load_portfolio_metadata(path)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    failures: list[str] = []
    if canonical_json(committed) != path.read_text(encoding="utf-8"):
        failures.append("the file on disk is not in the canonical serialisation")
    if committed != rebuilt:
        failures.append("the summary no longer matches the artefacts it was copied from")

    for name, holds, detail in verify_portfolio_metadata(
        rebuilt, EXPECTED_PORTFOLIO_METADATA_SHA256
    ):
        logger.info("%s %s: %s", "PASS" if holds else "FAIL", name, detail)
        if not holds:
            failures.append(f"invariant {name} does not hold ({detail})")

    committed_digest = metadata_digest(committed)
    if committed_digest != EXPECTED_PORTFOLIO_METADATA_SHA256:
        failures.append(
            f"the committed summary hashes to {committed_digest}, but "
            f"{PORTFOLIO_METADATA_TRUST_ANCHOR} pins "
            f"{EXPECTED_PORTFOLIO_METADATA_SHA256}"
        )

    if failures:
        for failure in failures:
            logger.error("%s", failure)
        return 1

    logger.info(
        "Portfolio metadata verified: %s (sha256=%s, pinned by %s)",
        path,
        committed_digest,
        PORTFOLIO_METADATA_TRUST_ANCHOR,
    )
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
