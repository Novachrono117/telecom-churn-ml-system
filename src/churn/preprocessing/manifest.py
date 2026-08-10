"""Machine-readable manifest that freezes the train/holdout partition.

The partition itself is **not** versioned — neither the two files nor the
identifier lists. What is versioned is enough information to regenerate it and
to prove the regeneration is byte-for-byte the same partition:

* the SHA-256 of the raw file it was derived from;
* every parameter that determines the split;
* the row counts;
* a SHA-256 over the sorted identifiers of each partition.

If any of those drift, :func:`verify_split_manifest` fails loudly instead of
letting a silently different holdout carry the final numbers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import sklearn
from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT, get_config
from churn.data.loader import RAW_FILENAME, verify_raw_dataset
from churn.preprocessing.contracts import ID_COLUMN
from churn.preprocessing.splitting import SPLIT_FUNCTION, fingerprint_ids, load_split

logger = logging.getLogger(__name__)

#: Default location of the versioned manifest.
MANIFEST_PATH = PROJECT_ROOT / "reports" / "split_manifest.json"

#: Bumped when the meaning of a field changes, so an old manifest is not
#: silently reinterpreted under new semantics.
MANIFEST_VERSION = 1


class SplitManifestMismatchError(ValueError):
    """A regenerated split does not match the recorded manifest."""


class SplitManifest(BaseModel):
    """Everything needed to regenerate and verify the frozen partition."""

    model_config = ConfigDict(frozen=True)

    manifest_version: int = Field(ge=1)
    raw_filename: str
    raw_sha256: str = Field(min_length=64, max_length=64)
    id_column: str
    target_column: str
    positive_label: str
    negative_label: str
    test_size: float = Field(gt=0.0, lt=1.0)
    random_state: int = Field(ge=0)
    stratified_by: str | None
    shuffle: bool
    split_function: str
    scikit_learn_version: str
    n_rows_total: int = Field(ge=1)
    n_rows_training: int = Field(ge=1)
    n_rows_holdout: int = Field(ge=1)
    training_ids_sha256: str = Field(min_length=64, max_length=64)
    holdout_ids_sha256: str = Field(min_length=64, max_length=64)


def build_split_manifest(path: Path | None = None) -> SplitManifest:
    """Regenerate the split and describe it.

    Args:
        path: Override for the raw CSV location.

    Returns:
        The manifest describing the partition just regenerated.

    Raises:
        churn.data.loader.RawDatasetMismatchError: If the raw file is not the
            one approved in Phase 2.
    """
    config = get_config()
    raw_sha256 = verify_raw_dataset(path)
    split = load_split(path)

    return SplitManifest(
        manifest_version=MANIFEST_VERSION,
        raw_filename=RAW_FILENAME,
        raw_sha256=raw_sha256,
        id_column=ID_COLUMN,
        target_column=config.target.column,
        positive_label=config.target.positive_label,
        negative_label=config.target.negative_label,
        test_size=config.split.test_size,
        random_state=config.seed,
        stratified_by=config.target.column if config.split.stratify else None,
        shuffle=True,
        split_function=SPLIT_FUNCTION,
        scikit_learn_version=sklearn.__version__,
        n_rows_total=len(split.training) + len(split.holdout),
        n_rows_training=len(split.training),
        n_rows_holdout=len(split.holdout),
        training_ids_sha256=fingerprint_ids(split.training[ID_COLUMN]),
        holdout_ids_sha256=fingerprint_ids(split.holdout[ID_COLUMN]),
    )


def write_split_manifest(manifest: SplitManifest, path: Path | None = None) -> Path:
    """Write the manifest as indented JSON.

    Args:
        manifest: Manifest to persist.
        path: Destination. Defaults to :data:`MANIFEST_PATH`.

    Returns:
        The path written.
    """
    destination = path or MANIFEST_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote split manifest: %s", destination)
    return destination


def load_split_manifest(path: Path | None = None) -> SplitManifest:
    """Read and validate a manifest from disk.

    Args:
        path: Manifest location. Defaults to :data:`MANIFEST_PATH`.

    Returns:
        The validated manifest.

    Raises:
        FileNotFoundError: If the manifest has not been generated yet.
        pydantic.ValidationError: If a field violates the contract.
    """
    source = path or MANIFEST_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at {source}. Generate it with "
            "`uv run python scripts/build_split.py`."
        )
    return SplitManifest.model_validate_json(source.read_text(encoding="utf-8"))


def verify_split_manifest(
    manifest: SplitManifest | None = None,
    path: Path | None = None,
) -> SplitManifest:
    """Regenerate the split and confirm it reproduces the recorded manifest.

    Args:
        manifest: Manifest to check against. Read from disk when omitted.
        path: Override for the raw CSV location.

    Returns:
        The manifest that was verified.

    Raises:
        SplitManifestMismatchError: If any recorded field differs from the
            regenerated split.
    """
    expected = manifest if manifest is not None else load_split_manifest()
    observed = build_split_manifest(path)

    differences = [
        f"{field}: manifest={getattr(expected, field)!r} regenerated={getattr(observed, field)!r}"
        for field in SplitManifest.model_fields
        if field != "scikit_learn_version" and getattr(expected, field) != getattr(observed, field)
    ]
    if differences:
        raise SplitManifestMismatchError(
            "The regenerated split does not reproduce the recorded manifest:\n  "
            + "\n  ".join(differences)
        )

    if expected.scikit_learn_version != observed.scikit_learn_version:
        logger.warning(
            "Split reproduced with scikit-learn %s, manifest recorded %s. "
            "The partition is identical; the version is recorded for provenance.",
            observed.scikit_learn_version,
            expected.scikit_learn_version,
        )
    logger.info("Split manifest verified: the partition is reproducible.")
    return expected
