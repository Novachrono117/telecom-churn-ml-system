"""The frozen split, checked against the real dataset and the recorded manifest.

Skipped while the CSV has not been acquired, so the suite stays green on a fresh
clone. When the file is present these are the tests that prove the partition
carrying the Phase 9 estimate is the one that was frozen in Phase 4.

Only structural facts about the holdout are asserted: counts, uniqueness and
fingerprints. No distribution of the holdout is computed anywhere here.
"""

from __future__ import annotations

import pytest

from churn.data.loader import raw_csv_path
from churn.preprocessing.contracts import ID_COLUMN, build_feature_matrix
from churn.preprocessing.manifest import (
    MANIFEST_PATH,
    build_split_manifest,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import fingerprint_ids, load_split
from churn.preprocessing.target import encode_target

pytestmark = pytest.mark.skipif(
    not raw_csv_path().is_file(),
    reason="raw dataset not acquired yet (see data/README.md)",
)


@pytest.fixture(scope="module")
def manifest():
    return build_split_manifest()


def test_manifest_is_reproducible_from_the_raw_file(manifest) -> None:
    regenerated = build_split_manifest()

    assert regenerated.training_ids_sha256 == manifest.training_ids_sha256
    assert regenerated.holdout_ids_sha256 == manifest.holdout_ids_sha256


def test_recorded_manifest_still_matches_the_regenerated_split() -> None:
    if not MANIFEST_PATH.is_file():
        pytest.skip("split manifest not generated yet (uv run python scripts/build_split.py)")

    verify_split_manifest(load_split_manifest())


def test_partitions_are_disjoint_and_complete(manifest) -> None:
    split = load_split()
    training_ids = set(split.training[ID_COLUMN])
    holdout_ids = set(split.holdout[ID_COLUMN])

    assert training_ids & holdout_ids == set()
    assert len(training_ids) == manifest.n_rows_training
    assert len(holdout_ids) == manifest.n_rows_holdout
    assert len(training_ids | holdout_ids) == manifest.n_rows_total


def test_fingerprints_describe_the_regenerated_partitions(manifest) -> None:
    split = load_split()

    assert fingerprint_ids(split.training[ID_COLUMN]) == manifest.training_ids_sha256
    assert fingerprint_ids(split.holdout[ID_COLUMN]) == manifest.holdout_ids_sha256


def test_training_pool_satisfies_the_feature_contract() -> None:
    training = load_split().training

    matrix = build_feature_matrix(training)

    assert len(matrix) == len(training)
    assert ID_COLUMN not in matrix.columns


def test_training_target_encodes_without_unknown_labels() -> None:
    training = load_split().training

    encoded = encode_target(training["Churn"])

    assert set(encoded.unique()) == {0, 1}
