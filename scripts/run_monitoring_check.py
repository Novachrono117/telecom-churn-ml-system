"""Phase 12: score a window of records offline and report drift against the reference.

Usage
-----
::

    uv run python scripts/run_monitoring_check.py --input <path-to-records.csv>
    uv run python scripts/run_monitoring_check.py --input <path> --json

Why this exists alongside the serving integration
-------------------------------------------------
The serving endpoint reports on traffic a *running process* has seen. This script
reports on a window somebody has in a file: a batch export from an upstream system,
a replay of a period under investigation, a candidate feed being qualified before it
is pointed at production. Neither replaces the other, and this one needs no server.

There is deliberately **no default input.** A monitoring tool that reads
``data/production/latest.csv`` when nobody asked will one day read something that is
not what its operator thought, and the number it prints will be believed anyway.
The path is always explicit.

Security boundary
-----------------
The only thing the ``--input`` file supplies is **records**. The model, the decision
policy and the reference profile are located by the process's own configuration and
can never be named by the input. The file is read as records, run through the frozen
feature contract, scored with the frozen pipeline, and folded into a collector that
keeps aggregates only.

The output contains no row. It is the same aggregate report the endpoint returns, and
the file it was computed from is never echoed into it.

This script does not train, does not load the holdout, and does not touch the target:
a ``Churn`` column in the input would be dropped by the feature contract along with
everything else outside the 19 features.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from churn.modeling.freeze import apply_decision_rule, load_pipeline, positive_probability
from churn.monitoring.collector import MonitoringCollector
from churn.monitoring.reference import load_reference_profile, profile_digest
from churn.monitoring.service import compare
from churn.monitoring.settings import get_monitoring_policy
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features

logger = logging.getLogger("run_monitoring_check")

#: Input formats understood. Both carry records and nothing else.
SUPPORTED_SUFFIXES = (".csv", ".json")


def read_records(path: Path) -> pd.DataFrame:
    """Read a window of records from a file.

    Raises:
        SystemExit: If the file is absent or its format is not supported. Told
            plainly rather than guessed at: silently misreading a monitoring input
            produces a number that looks fine.
    """
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise SystemExit(
            f"Unsupported input format {path.suffix!r}. Supported: {list(SUPPORTED_SUFFIXES)}."
        )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=object)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    return pd.DataFrame(records)


def _coerce_contract_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Give the contracted numeric columns numeric dtypes.

    A CSV arrives as text. ``tenure`` and ``MonthlyCharges`` must be numeric for the
    frozen contract to accept them, and ``TotalCharges`` must stay exactly as
    written so the structural-zero rule still sees a blank as a blank. Nothing here
    repairs a value: an unreadable number stays unreadable and is rejected
    downstream by the frozen preprocessing, which is the authority.
    """
    result = frame.copy()
    for column in ("tenure", "MonthlyCharges"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["SeniorCitizen"] = pd.to_numeric(result["SeniorCitizen"], errors="coerce").astype(
        "Int64"
    )
    return result


def run(input_path: Path, as_json: bool) -> int:
    reference = load_reference_profile()
    digest = profile_digest(reference)
    policy = get_monitoring_policy()
    pipeline = load_pipeline()

    frame = read_records(input_path)
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"Input is missing contracted feature column(s): {missing}")

    features = prepare_features(_coerce_contract_dtypes(frame.loc[:, list(FEATURE_COLUMNS)]))
    scores = np.asarray(positive_probability(pipeline, features), dtype=float)
    decisions = apply_decision_rule(scores, reference.score.threshold)

    collector = MonitoringCollector(reference)
    collector.record_request(len(features))
    collector.observe(features, scores, decisions)
    report = compare(collector.snapshot(), reference, policy, digest)

    if as_json:
        sys.stdout.write(json.dumps(report.as_record(), indent=2, ensure_ascii=False) + "\n")
        return 0

    logger.info("MONITORING_WINDOW: %d record(s) from %s", report.n_records, input_path.name)
    logger.info("reference_profile_sha256 : %s", report.reference_profile_sha256)
    logger.info("overall status           : %s", report.status)
    if report.details_suppressed:
        # The same policy the endpoint applies, and for the same reason: a
        # per-feature breakdown of a handful of records is those records. A file on
        # an operator's disk is not a safer place to print them than an HTTP body.
        logger.info(
            "  below the operational minimum of %d records: every detailed "
            "distribution is suppressed and no drift verdict is claimed",
            report.minimum_window_size,
        )
        logger.info("data quality             : %s", report.quality)
        logger.info("performance degradation  : NOT EVALUATED (no production labels)")
        return 0
    for section, status in report.section_status.items():
        logger.info("  %-24s %s", section, status)
    logger.info("data quality             : %s", report.quality)
    for feature, entry in report.numeric.items():
        logger.info(
            "  numeric %-15s psi=%.4f  mean_shift_sd=%+.3f  out_of_range=%.4f  [%s]",
            feature,
            entry["psi"]["value"],
            entry["mean_shift_in_reference_sd"],
            entry["out_of_reference_range_rate"]["value"],
            entry["status"],
        )
    for feature, entry in report.categorical.items():
        logger.info(
            "  categorical %-19s tvd=%.4f  unseen=%.4f (%s%d distinct)  [%s]",
            feature,
            entry["tvd"]["value"],
            entry["unseen_rate"]["value"],
            # ">=" rather than a bare number once tracking saturated: the value is a
            # lower bound then, and printing it unqualified would misreport it.
            ">=" if entry["distinct_unseen_is_lower_bound"] else "",
            entry["n_distinct_unseen_observed"],
            entry["status"],
        )
    logger.info(
        "  score psi=%.4f  positive_rate=%.4f (reference %.4f, delta %+.4f)  [%s]",
        report.score["psi"]["value"],
        report.score["predicted_positive_rate"],
        report.score["reference_predicted_positive_rate"],
        report.score["predicted_positive_rate_delta"]["value"],
        report.score["status"],
    )
    logger.info(
        "  structural violations=%d rate=%.4f  [%s]",
        report.structural["records_with_any_violation"],
        report.structural["violation_rate"],
        report.structural["status"],
    )
    logger.info("performance degradation  : NOT EVALUATED (no production labels)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Records to check. CSV or JSON. Required: there is no default input.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the aggregate report as JSON on stdout instead of a log summary.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run(arguments.input, arguments.as_json)


if __name__ == "__main__":
    raise SystemExit(main())
