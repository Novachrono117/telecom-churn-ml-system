"""The reference profile: what "normal" means, written down once.

This artefact is to monitoring what ``decision_policy.json`` is to serving — the
frozen thing every later comparison is made against. It holds **aggregates only**:
counts, order statistics, fixed bin edges, level frequencies and a score histogram.
No row of the training pool survives into it, and none could be reconstructed from
it.

Three properties it must have, and why

**It comes from the training pool, and only from it.** The holdout was consumed by
Phase 9D and offers nothing here: an operational reference describes the population
the system was built for, and 5 634 rows describe it better than 1 409 do. Nothing
in this module can reach either partition — the population is handed in by the
offline build script, exactly as Phase 9C's freeze modules never loaded data
themselves.

**It does not use the target.** A reference distribution is a statement about
inputs and scores, not about outcomes. Touching ``Churn`` here would blur the line
this phase exists to hold: monitoring detects that the population changed, and it
must never be mistaken for an estimate of how well the model performs.
``target_used`` is recorded as ``false`` and asserted by a test.

**It is byte-deterministic.** No timestamp, no hostname, no run id. Rebuilding it on
an unchanged repository reproduces the file exactly, which is what lets
``--verify`` compare bytes instead of tolerances.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from churn.monitoring.settings import default_reference_profile_path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
PROFILE_NAME = "phase12-monitoring-reference-profile"

#: Where the reference population comes from. There is one legal value.
REFERENCE_POPULATION = "training_pool"


class NumericReference(BaseModel):
    """Order statistics and fixed histogram of one numeric feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(ge=1)
    mean: float
    std: float = Field(ge=0.0)
    min: float
    q01: float
    q05: float
    q25: float
    median: float
    q75: float
    q95: float
    q99: float
    max: float
    bin_edges: list[float] = Field(
        description=(
            "Interior edges only. The partition is (-inf, e1], (e1, e2], ..., "
            "(e_last, +inf), so it is total and no value can fall outside it."
        )
    )
    bin_counts: list[int]

    @property
    def n_bins(self) -> int:
        return len(self.bin_counts)


class CategoricalReference(BaseModel):
    """Known levels and their frequencies for one categorical feature."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(ge=1)
    known_levels: list[str]
    count_by_level: dict[str, int]
    frequency_by_level: dict[str, float]


class ScoreReference(BaseModel):
    """The model's own score distribution on the reference population.

    A **monitoring reference distribution**, not a performance estimate. It says
    what scores the frozen model assigns to the population it was fitted on; it says
    nothing about whether those scores are right, and it was computed without ever
    reading a label.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = Field(ge=1)
    mean: float
    std: float = Field(ge=0.0)
    min: float
    q01: float
    q05: float
    q25: float
    median: float
    q75: float
    q95: float
    q99: float
    max: float
    bin_edges: list[float]
    bin_counts: list[int]
    threshold: float
    comparison: str
    predicted_positive_count: int = Field(ge=0)
    predicted_positive_rate: float = Field(ge=0.0, le=1.0)
    calibration_policy: str


class StructuralReference(BaseModel):
    """How each product equivalence behaved on the reference population."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: dict[str, str]
    violations_by_rule: dict[str, int]
    records_with_any_violation: int = Field(ge=0)
    violation_rate: float = Field(ge=0.0, le=1.0)


class ReferenceProvenance(BaseModel):
    """Which bytes produced this profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reference_population: str
    n_reference: int = Field(ge=1)
    holdout_used: bool
    target_used: bool
    raw_sha256: str = Field(min_length=64, max_length=64)
    training_ids_sha256: str = Field(min_length=64, max_length=64)
    model_fingerprint_sha256: str = Field(min_length=64, max_length=64)
    pipeline_sha256: str = Field(min_length=64, max_length=64)
    decision_policy_sha256: str = Field(min_length=64, max_length=64)
    freeze_commit: str
    serving_commit: str
    random_seed: int


class ProfileConfiguration(BaseModel):
    """How this profile was computed, so a reader can reproduce every number."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_bins_requested: int = Field(ge=2)
    binning_strategy: str
    bin_interval_convention: str
    quantiles_recorded: list[float]
    unseen_level: str
    numeric_features: list[str]
    categorical_features: list[str]
    psi_epsilon: float = Field(gt=0.0)
    numeric_drift_metric: str
    categorical_drift_metric: str
    score_drift_metric: str


class ReferenceProfile(BaseModel):
    """The complete Phase 12 monitoring reference."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    schema_version: int = Field(ge=1)
    profile: str
    built_in_phase: str
    provenance: ReferenceProvenance
    configuration: ProfileConfiguration
    numeric: dict[str, NumericReference]
    categorical: dict[str, CategoricalReference]
    score: ScoreReference
    structural: StructuralReference
    notes: list[str]


def canonical_json(profile: ReferenceProfile) -> str:
    """Return the one serialisation this project writes and hashes.

    Indented for review, ``ensure_ascii=False`` so the text stays readable, key
    order preserved from the model rather than sorted — the recorded orderings
    (feature order, bin order, level order) are part of what is being fingerprinted,
    and sorting them would hide a reordering.
    """
    return json.dumps(profile.model_dump(), indent=2, ensure_ascii=False) + "\n"


def profile_digest(profile: ReferenceProfile) -> str:
    """Return the SHA-256 of a profile's canonical serialisation.

    Computed from the *value*, not from whatever bytes happen to be on disk, so a
    profile built in memory and one read back from a file hash identically.
    """
    return hashlib.sha256(canonical_json(profile).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reference_profile(profile: ReferenceProfile, path: Path | None = None) -> Path:
    """Write the profile with LF endings and a trailing newline."""
    destination = path or default_reference_profile_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(profile), encoding="utf-8", newline="\n")
    logger.info("Wrote monitoring reference profile: %s", destination)
    return destination


def load_reference_profile(path: Path | None = None) -> ReferenceProfile:
    """Read and validate a reference profile.

    Raises:
        FileNotFoundError: If it has not been built yet.
        pydantic.ValidationError: If the file is not a reference profile.
    """
    source = path or default_reference_profile_path()
    if not source.is_file():
        raise FileNotFoundError(
            f"Monitoring reference profile not found at {source}. Build it with "
            "`uv run python scripts/build_monitoring_reference.py`."
        )
    return ReferenceProfile.model_validate_json(source.read_text(encoding="utf-8"))


class ReferenceProfileError(RuntimeError):
    """The reference profile is absent, malformed, or not the one expected."""


def verify_reference_profile(
    profile: ReferenceProfile,
    expected_digest: str | None = None,
) -> list[tuple[str, bool, str]]:
    """Return ``(name, passed, detail)`` for every invariant a profile must satisfy.

    Returned rather than raised so a caller can report all of them at once.
    """
    provenance = profile.provenance
    configuration = profile.configuration
    actual_digest = profile_digest(profile)

    checks: list[tuple[str, bool, str]] = [
        (
            "profile_is_the_phase12_reference",
            profile.profile == PROFILE_NAME and profile.schema_version == SCHEMA_VERSION,
            f"profile={profile.profile!r} v{profile.schema_version}",
        ),
        (
            "reference_population_is_the_training_pool",
            provenance.reference_population == REFERENCE_POPULATION,
            provenance.reference_population,
        ),
        ("holdout_not_used", provenance.holdout_used is False, "holdout_used=false"),
        ("target_not_used", provenance.target_used is False, "target_used=false"),
        (
            "every_numeric_feature_profiled",
            set(profile.numeric) == set(configuration.numeric_features),
            f"{len(profile.numeric)} numeric feature(s)",
        ),
        (
            "every_categorical_feature_profiled",
            set(profile.categorical) == set(configuration.categorical_features),
            f"{len(profile.categorical)} categorical feature(s)",
        ),
        (
            "numeric_histograms_are_total",
            all(
                sum(entry.bin_counts) == entry.count
                and len(entry.bin_counts) == len(entry.bin_edges) + 1
                for entry in profile.numeric.values()
            ),
            "counts sum to n and bins = edges + 1",
        ),
        (
            "numeric_bin_edges_strictly_increasing",
            all(
                list(entry.bin_edges) == sorted(set(entry.bin_edges))
                for entry in profile.numeric.values()
            ),
            "no duplicate or unordered edge",
        ),
        (
            "categorical_counts_sum_to_n",
            all(
                sum(entry.count_by_level.values()) == entry.count
                for entry in profile.categorical.values()
            ),
            "level counts sum to n",
        ),
        (
            "score_histogram_is_total",
            sum(profile.score.bin_counts) == profile.score.count
            and len(profile.score.bin_counts) == len(profile.score.bin_edges) + 1,
            f"{profile.score.count} scored records",
        ),
        (
            "score_is_a_probability",
            0.0 <= profile.score.min and profile.score.max <= 1.0,
            f"[{profile.score.min:.6f}, {profile.score.max:.6f}]",
        ),
        (
            "score_uses_the_frozen_decision_rule",
            profile.score.comparison == ">=" and 0.0 < profile.score.threshold < 1.0,
            f"{profile.score.comparison} {profile.score.threshold!r}",
        ),
        (
            "calibration_policy_is_recorded_as_none",
            profile.score.calibration_policy == "NONE",
            profile.score.calibration_policy,
        ),
        (
            "seven_structural_rules_checked",
            len(profile.structural.rules) == 7
            and set(profile.structural.rules) == set(profile.structural.violations_by_rule),
            f"{len(profile.structural.rules)} rules",
        ),
        (
            "reference_population_satisfies_every_structural_rule",
            profile.structural.records_with_any_violation == 0
            and all(count == 0 for count in profile.structural.violations_by_rule.values()),
            f"{profile.structural.records_with_any_violation} violating record(s)",
        ),
        (
            "counts_agree_with_the_declared_size",
            all(entry.count == provenance.n_reference for entry in profile.numeric.values())
            and all(entry.count == provenance.n_reference for entry in profile.categorical.values())
            and profile.score.count == provenance.n_reference,
            f"n_reference={provenance.n_reference}",
        ),
    ]

    if expected_digest is not None:
        checks.append(
            (
                "profile_digest_matches",
                actual_digest == expected_digest,
                f"actual={actual_digest[:16]}… expected={expected_digest[:16]}…",
            )
        )
    return checks


def raise_on_failure(checks: list[tuple[str, bool, str]]) -> list[tuple[str, bool, str]]:
    """Return ``checks`` unchanged, or raise listing every one that failed."""
    failed = [(name, detail) for name, passed, detail in checks if not passed]
    if failed:
        raise ReferenceProfileError(
            "The monitoring reference profile did not pass its checks:\n  "
            + "\n  ".join(f"{name}: {detail}" for name, detail in failed)
        )
    return checks


__all__ = [
    "PROFILE_NAME",
    "REFERENCE_POPULATION",
    "SCHEMA_VERSION",
    "CategoricalReference",
    "NumericReference",
    "ProfileConfiguration",
    "ReferenceProfile",
    "ReferenceProfileError",
    "ReferenceProvenance",
    "ScoreReference",
    "StructuralReference",
    "canonical_json",
    "file_digest",
    "load_reference_profile",
    "profile_digest",
    "raise_on_failure",
    "verify_reference_profile",
    "write_reference_profile",
]
