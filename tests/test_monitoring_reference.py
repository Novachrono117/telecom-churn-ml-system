"""The reference profile: where it comes from, what it refuses to look at.

The population question is the one that decides whether the whole phase is
methodologically sound, so it is tested from three angles: the profile *says* it
used the training pool, the builder *cannot* reach any partition, and the recorded
size is the frozen pool's.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from churn.monitoring import build as build_module
from churn.monitoring.build import build_reference_profile
from churn.monitoring.reference import (
    PROFILE_NAME,
    REFERENCE_POPULATION,
    ReferenceProfileError,
    canonical_json,
    load_reference_profile,
    profile_digest,
    raise_on_failure,
    verify_reference_profile,
    write_reference_profile,
)
from churn.monitoring.settings import default_reference_profile_path
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, NUMERIC_FEATURES

#: The frozen training pool size, fixed by the Phase 9C split.
POOL_ROWS = 5634


def _keys(node: object) -> Iterator[str]:
    """Yield every key in a nested mapping, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keys(item)


@pytest.fixture(scope="module")
def profile():
    return load_reference_profile()


# --------------------------------------------------------------------------- #
# Population.
# --------------------------------------------------------------------------- #


def test_the_reference_is_the_training_pool(profile) -> None:
    assert profile.provenance.reference_population == REFERENCE_POPULATION == "training_pool"
    assert profile.profile == PROFILE_NAME


def test_the_reference_has_the_frozen_pool_size(profile) -> None:
    assert profile.provenance.n_reference == POOL_ROWS
    assert profile.score.count == POOL_ROWS
    for entry in profile.numeric.values():
        assert entry.count == POOL_ROWS
    for entry in profile.categorical.values():
        assert entry.count == POOL_ROWS


def test_the_target_is_never_read(profile) -> None:
    """Recorded as false, and structurally impossible to have been true.

    The builder's signature has no parameter through which a label could arrive,
    and its body never names the target column.
    """
    assert profile.provenance.target_used is False

    signature = inspect.signature(build_reference_profile)
    assert "target" not in signature.parameters
    assert "y" not in signature.parameters

    source = inspect.getsource(build_reference_profile)
    assert "Churn" not in source
    assert "encode_target" not in source


def test_the_holdout_is_never_reachable(profile) -> None:
    """Recorded as false, and no monitoring module imports a holdout loader."""
    assert profile.provenance.holdout_used is False

    package = Path(build_module.__file__).parent
    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert "load_holdout" not in names, module.name
        assert "load_split" not in names, module.name
        assert "load_training_pool" not in names, module.name


def test_the_builder_cannot_load_anything(profile) -> None:
    """The population is handed in, exactly as Phase 9C's freeze modules required."""
    source = Path(build_module.__file__).read_text(encoding="utf-8")

    assert "read_csv" not in source
    assert "load_raw" not in source
    assert "frame: pd.DataFrame" in source


def test_the_provenance_names_the_frozen_artefacts(profile) -> None:
    provenance = profile.provenance
    policy = json.loads(
        (default_reference_profile_path().parents[1] / "decision_policy.json").read_text(
            encoding="utf-8"
        )
    )

    assert provenance.raw_sha256 == policy["raw_sha256"]
    assert provenance.training_ids_sha256 == policy["training_ids_sha256"]
    assert provenance.model_fingerprint_sha256 == policy["artifacts"]["model_fingerprint_sha256"]
    assert provenance.pipeline_sha256 == policy["artifacts"]["pipeline_sha256"]
    assert len(provenance.freeze_commit) == 40
    assert len(provenance.serving_commit) == 40


# --------------------------------------------------------------------------- #
# Content.
# --------------------------------------------------------------------------- #


def test_every_contracted_feature_is_profiled(profile) -> None:
    assert set(profile.numeric) == set(NUMERIC_FEATURES)
    assert set(profile.categorical) == set(CATEGORICAL_FEATURES)
    assert len(profile.numeric) + len(profile.categorical) == 19


def test_the_numeric_histograms_are_total(profile) -> None:
    """Bins extend to +/- infinity, so no mass is ever lost."""
    for feature, entry in profile.numeric.items():
        assert sum(entry.bin_counts) == entry.count, feature
        assert len(entry.bin_counts) == len(entry.bin_edges) + 1, feature


def test_the_numeric_bin_edges_are_strictly_increasing(profile) -> None:
    """Duplicate quantile edges are dropped, not kept as zero-width bins."""
    for feature, entry in profile.numeric.items():
        assert list(entry.bin_edges) == sorted(set(entry.bin_edges)), feature


def test_the_unseen_bucket_exists_and_is_empty_in_the_reference(profile) -> None:
    """It is the level the reference itself never had, kept so a window shares a support."""
    for feature, entry in profile.categorical.items():
        assert "__UNSEEN__" in entry.count_by_level, feature
        assert entry.count_by_level["__UNSEEN__"] == 0, feature
        assert entry.frequency_by_level["__UNSEEN__"] == 0.0, feature


def test_the_score_reference_is_a_distribution_not_a_metric(profile) -> None:
    """Scores, quantiles and a positive rate. No accuracy, no recall, no AP."""
    score = profile.score

    assert 0.0 <= score.min <= score.max <= 1.0
    assert score.comparison == ">="
    assert score.calibration_policy == "NONE"
    assert 0.0 < score.threshold < 1.0
    assert sum(score.bin_counts) == score.count

    # Checked on the KEYS, not on a substring of the rendered file: a SHA-256 hex
    # digest contains arbitrary two-character sequences, and "af1c..." would make a
    # naive search for "f1" fail for a reason that has nothing to do with metrics.
    forbidden = {
        "accuracy",
        "recall",
        "precision",
        "f1",
        "f1_score",
        "roc_auc",
        "average_precision",
        "confusion_matrix",
        "log_loss",
        "brier",
        "y_true",
        "labels",
    }
    assert not forbidden & set(_keys(profile.model_dump()))


def test_the_seven_structural_rules_hold_on_the_reference(profile) -> None:
    """Phase 10's finding, re-established on the profile that will be monitored against."""
    assert len(profile.structural.rules) == 7
    assert profile.structural.records_with_any_violation == 0
    assert profile.structural.violation_rate == 0.0
    assert all(count == 0 for count in profile.structural.violations_by_rule.values())


def test_the_profile_holds_no_row_level_data(profile) -> None:
    """Aggregates only: no identifier, and nothing of the size of a population."""
    rendered = json.dumps(profile.model_dump())

    assert "customerID" not in rendered
    # The largest list in the profile is a histogram, not a column of 5 634 values.
    largest = max(len(entry.bin_counts) for entry in (*profile.numeric.values(), profile.score))
    assert largest < 50


# --------------------------------------------------------------------------- #
# Integrity.
# --------------------------------------------------------------------------- #


def test_the_profile_is_deterministic(profile) -> None:
    """The same value serialises to the same bytes, so --verify can compare them."""
    assert canonical_json(profile) == canonical_json(profile)
    assert profile_digest(profile) == profile_digest(load_reference_profile())
    assert canonical_json(profile) == default_reference_profile_path().read_text(encoding="utf-8")


def test_the_profile_carries_no_clock(profile) -> None:
    rendered = json.dumps(profile.model_dump()).lower()

    for forbidden in ("generated_at", "timestamp", "created_at", "hostname", "run_id"):
        assert forbidden not in rendered, forbidden


def test_every_invariant_passes_on_the_real_profile(profile) -> None:
    checks = verify_reference_profile(profile)

    assert len(checks) >= 15
    assert all(passed for _, passed, _ in checks)


def test_a_tampered_reference_fails_verification(profile) -> None:
    """The digest gate: a profile read from other bytes is not this profile."""
    checks = dict((name, passed) for name, passed, _ in verify_reference_profile(profile, "0" * 64))

    assert checks["profile_digest_matches"] is False


def test_a_reference_claiming_the_holdout_is_refused(profile) -> None:
    broken = profile.model_copy(
        update={"provenance": profile.provenance.model_copy(update={"holdout_used": True})}
    )

    checks = {name: passed for name, passed, _ in verify_reference_profile(broken)}

    assert checks["holdout_not_used"] is False
    with pytest.raises(ReferenceProfileError):
        raise_on_failure(verify_reference_profile(broken))


def test_a_reference_claiming_the_target_is_refused(profile) -> None:
    broken = profile.model_copy(
        update={"provenance": profile.provenance.model_copy(update={"target_used": True})}
    )

    checks = {name: passed for name, passed, _ in verify_reference_profile(broken)}

    assert checks["target_not_used"] is False


def test_a_reference_from_another_population_is_refused(profile) -> None:
    broken = profile.model_copy(
        update={
            "provenance": profile.provenance.model_copy(update={"reference_population": "holdout"})
        }
    )

    checks = {name: passed for name, passed, _ in verify_reference_profile(broken)}

    assert checks["reference_population_is_the_training_pool"] is False


def test_writing_the_profile_is_idempotent(profile, tmp_path: Path) -> None:
    destination = tmp_path / "reference_profile.json"

    first = write_reference_profile(profile, destination).read_bytes()
    second = write_reference_profile(profile, destination).read_bytes()

    assert first == second
    assert first.endswith(b"\n")
    assert b"\r\n" not in first
