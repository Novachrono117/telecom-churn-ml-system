"""Freeze the train/holdout split and record it in a versioned manifest.

Usage (from the repository root):

    uv run python scripts/build_split.py            # generate and write
    uv run python scripts/build_split.py --verify   # check the recorded manifest

Writes:
    reports/split_manifest.json

The partition itself is never written to disk: it is regenerated from the raw
file plus the configuration whenever it is needed. Only structural facts are
logged — row counts and fingerprints. No distribution of either partition is
computed here.
"""

from __future__ import annotations

import argparse
import logging
import sys

from churn.data.loader import RawDatasetMismatchError
from churn.preprocessing.manifest import (
    MANIFEST_PATH,
    SplitManifestMismatchError,
    build_split_manifest,
    load_split_manifest,
    verify_split_manifest,
    write_split_manifest,
)

logger = logging.getLogger("build_split")


def _log_structure(manifest) -> None:
    logger.info("raw SHA-256          : %s", manifest.raw_sha256)
    logger.info("rows total           : %d", manifest.n_rows_total)
    logger.info("rows training pool   : %d", manifest.n_rows_training)
    logger.info("rows holdout         : %d", manifest.n_rows_holdout)
    logger.info("training IDs SHA-256 : %s", manifest.training_ids_sha256)
    logger.info("holdout  IDs SHA-256 : %s", manifest.holdout_ids_sha256)


def main() -> int:
    """Generate or verify the split manifest. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check that the recorded manifest is still reproducible, writing nothing",
    )
    arguments = parser.parse_args()

    try:
        if arguments.verify:
            manifest = verify_split_manifest(load_split_manifest())
            logger.info("Manifest at %s is reproducible.", MANIFEST_PATH)
        else:
            manifest = build_split_manifest()
            write_split_manifest(manifest)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1
    except RawDatasetMismatchError as error:
        logger.error("%s", error)
        return 2
    except SplitManifestMismatchError as error:
        logger.error("%s", error)
        return 3

    _log_structure(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
