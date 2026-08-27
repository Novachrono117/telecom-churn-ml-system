"""Phase 12: build (or verify) the monitoring reference profile.

Usage
-----
Build::

    uv run python scripts/build_monitoring_reference.py

Verify — read-only, regenerates nothing::

    uv run python scripts/build_monitoring_reference.py --verify

``--verify`` reads the committed profile, checks every invariant, confirms the
provenance still matches the frozen artefacts, and rebuilds the profile in memory to
compare it **byte for byte** with the file. It writes nothing. A byte comparison is
only meaningful because the profile carries no clock.

Why this script may load the training pool
------------------------------------------
It is **offline**. The prohibition the serving boundary lives under — no dataset,
ever — exists because a request handler that can read a partition can read the wrong
one. A build step has no requests and no such risk, and the reference has to come
from somewhere.

The holdout stays unreachable even here. This script imports ``load_training_pool``
and nothing else from :mod:`churn.preprocessing.splitting`, and a test asserts that
``load_holdout`` appears nowhere in the Phase 12 source.

The pool is loaded here and *handed* to :func:`churn.monitoring.build.build_reference_profile`,
which has no way to load anything itself. The target is never selected: the builder
takes the frame and immediately reduces it to the 19 contracted features.
"""

from __future__ import annotations

import argparse
import logging

from churn.config import PROJECT_ROOT, get_config
from churn.modeling.freeze import file_digest, load_pipeline, model_fingerprint
from churn.modeling.freeze_results import POLICY_PATH, load_decision_policy
from churn.monitoring.build import build_reference_profile
from churn.monitoring.reference import (
    ReferenceProfile,
    canonical_json,
    load_reference_profile,
    profile_digest,
    raise_on_failure,
    verify_reference_profile,
    write_reference_profile,
)
from churn.monitoring.settings import default_reference_profile_path
from churn.preprocessing.splitting import load_training_pool

logger = logging.getLogger("build_monitoring_reference")

#: The commit that published the serving boundary this reference is paired with.
SERVING_COMMIT = "71a7f3a785c13a052e4f8ab3a6780842de60110a"


def _freeze_commit() -> str:
    """Return the Phase 9C freeze commit, from the serving layer's own constant."""
    from churn.serving.artifacts import FREEZE_COMMIT

    return FREEZE_COMMIT


def _build() -> ReferenceProfile:
    """Build the profile from the frozen artefacts and the frozen training pool."""
    config = get_config()
    policy = load_decision_policy()
    pipeline = load_pipeline()

    actual_fingerprint = model_fingerprint(pipeline)
    if actual_fingerprint != policy.artifacts.model_fingerprint_sha256:
        raise SystemExit(
            "The pipeline on disk is not the frozen model. Refusing to build a "
            "monitoring reference from an unverified artefact."
        )

    training = load_training_pool()
    logger.info("Reference population: training pool, %d rows.", len(training))

    return build_reference_profile(
        training,
        pipeline,
        threshold=policy.threshold.final_threshold,
        comparison=policy.decision_rule.comparison,
        calibration_policy=policy.calibration.calibration_policy,
        raw_sha256=policy.raw_sha256,
        training_ids_sha256=policy.training_ids_sha256,
        model_fingerprint_sha256=actual_fingerprint,
        pipeline_sha256=file_digest(PROJECT_ROOT / policy.artifacts.pipeline_path),
        decision_policy_sha256=file_digest(POLICY_PATH),
        freeze_commit=_freeze_commit(),
        serving_commit=SERVING_COMMIT,
        random_seed=config.seed,
    )


def build() -> int:
    profile = _build()
    raise_on_failure(verify_reference_profile(profile))
    destination = write_reference_profile(profile)
    logger.info("MONITORING_REFERENCE: %s", destination)
    logger.info("reference_profile_sha256 : %s", profile_digest(profile))
    logger.info("n_reference              : %d", profile.provenance.n_reference)
    logger.info("holdout_used             : %s", profile.provenance.holdout_used)
    logger.info("target_used              : %s", profile.provenance.target_used)
    logger.info(
        "score reference          : mean %.6f, positive rate %.6f at %r %s",
        profile.score.mean,
        profile.score.predicted_positive_rate,
        profile.score.threshold,
        profile.score.comparison,
    )
    return 0


def verify() -> int:
    path = default_reference_profile_path()
    try:
        committed = load_reference_profile(path)
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    failures: list[str] = []
    for name, passed, detail in verify_reference_profile(committed):
        logger.info("%s %s: %s", "PASS" if passed else "FAIL", name, detail)
        if not passed:
            failures.append(f"{name}: {detail}")

    if canonical_json(committed) != path.read_text(encoding="utf-8"):
        failures.append("the profile on disk is not in the canonical serialisation")

    rebuilt = _build()
    if canonical_json(rebuilt) != canonical_json(committed):
        failures.append(
            "the committed profile is not reproducible from the frozen artefacts and "
            "the frozen training pool"
        )
    else:
        logger.info("PASS profile_is_reproducible: rebuilt profile is byte-identical")

    if failures:
        for failure in failures:
            logger.error("%s", failure)
        return 1

    logger.info(
        "MONITORING_REFERENCE verified: sha256=%s n=%d holdout_used=%s target_used=%s",
        profile_digest(committed),
        committed.provenance.n_reference,
        committed.provenance.holdout_used,
        committed.provenance.target_used,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Read-only: check the committed profile and rebuild it in memory to compare.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return verify() if arguments.verify else build()


if __name__ == "__main__":
    raise SystemExit(main())
