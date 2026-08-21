"""Phase 9C: materialise every earlier decision as one frozen, verifiable object.

Nothing is *decided* here. Phases 5 to 8B chose the estimator, Phase 9A settled
that its probabilities are used raw, Phase 9B settled where they are cut. This
phase does the only thing left before the holdout can be opened: fit that
estimator on the whole frozen training pool, persist it, and write down — with
fingerprints — exactly what will later be applied to the holdout.

**The fit is a materialisation, not a selection.** Every earlier phase fitted the
same pipeline on four fifths of the pool at a time, because it needed a
validation fold. That need is over: the configuration is frozen, so the final
object is fitted on all of it. No metric is computed from this fit, and none
could change anything if it were.

**Two fingerprints, and the difference matters.**

* :func:`model_fingerprint` hashes the *learned parameters* — transformed feature
  names, coefficients, intercept, scaler statistics, encoder categories — under a
  canonical JSON encoding that round-trips every double exactly. This is the
  **identity of the model**: two objects with the same value make the same
  predictions, whatever pickle protocol, library build or file compression
  produced them.
* :func:`file_digest` hashes the serialised bytes. That catches a corrupted or
  swapped file, but it is a property of the *file*, not of the model, and it is
  not guaranteed to survive a change of environment.

The authoritative check is the first one. The second is reported alongside it and
never on its own.

**The persisted pipeline starts after ``prepare_features``.** That boundary was
fixed in Phase 4 and is not moved here: the object stored on disk expects a
validated, canonically ordered feature matrix, and whoever calls it is
responsible for producing one. The public prediction interface is Phase 11's
subject; what lives here are persistence and integrity primitives, deliberately
small.

Out of scope by protocol: no model comparison, no tuning, no feature, no
calibration, no threshold search, no serving layer, and no holdout — the training
pool is the only partition this module can reach, because it never loads
anything.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.config import PROJECT_ROOT
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.modeling.tuning import build_modern_logistic_pipeline
from churn.preprocessing.contracts import prepare_features
from churn.preprocessing.pipeline import (
    CATEGORICAL_STEP,
    CLEANER_STEP,
    ENCODER_STEP,
    NUMERIC_STEP,
)

logger = logging.getLogger(__name__)

#: Where the fitted pipeline is written. Relative to the repository root, and
#: stored in the policy as a POSIX path so the record is platform-neutral.
PIPELINE_RELATIVE_PATH = "artifacts/model/churn_pipeline.joblib"
PIPELINE_PATH = PROJECT_ROOT / PIPELINE_RELATIVE_PATH

#: Serialisation settings, pinned rather than left at their defaults.
#:
#: ``protocol`` is pinned because a future interpreter raising
#: ``pickle.DEFAULT_PROTOCOL`` would silently change every byte of the file and
#: therefore its digest, without anything about the model having changed.
#: ``compress=0`` keeps the stream free of a compressor's own version-dependent
#: framing; the file is a few kilobytes, so compression buys nothing worth that.
SERIALIZATION_PROTOCOL = 5
SERIALIZATION_COMPRESS = 0

#: The two policy vocabularies. A value outside them is a corrupted or
#: hand-edited record, not an unusual configuration.
KNOWN_CALIBRATION_POLICIES: frozenset[str] = frozenset({"NONE", "C1", "C2"})
KNOWN_THRESHOLD_POLICIES: frozenset[str] = frozenset({"DEFAULT_0_5", "F1_MAXIMIZATION"})

#: The comparison in the decision rule. Written down because ``>`` and ``>=``
#: differ on exactly the customers sitting at the threshold, and a rule that
#: leaves the boundary implicit is not a frozen rule.
DECISION_COMPARISON = ">="

#: Label of the positive class in the encoded target, and what it means.
POSITIVE_CLASS_LABEL = 1
POSITIVE_CLASS_MEANING = "Churn = Yes"
NEGATIVE_CLASS_LABEL = 0


#: The entry point of the inference function that sits **outside** the persisted
#: pipeline, and the module it lives in.
PREPARE_FEATURES = prepare_features
PREPARE_FEATURES_QUALNAME = "churn.preprocessing.contracts.prepare_features"

#: Source files that materially determine the frozen inference function.
#:
#: This list closes a hole the two model fingerprints cannot see. A ``.joblib``
#: stores fitted *state* — arrays, hyperparameters — but every class in it is
#: pickled **by reference**: the stream carries ``churn.preprocessing.transformers``
#: and unpickling imports whatever that module currently defines. So editing
#: ``TotalChargesCleaner.transform`` changes what the loaded pipeline computes
#: while ``pipeline_sha256`` and ``model_fingerprint_sha256`` stay identical. The
#: same is true of ``prepare_features``, which runs before the pipeline is even
#: reached, and of the decision rule in this module, which runs after it.
#:
#: Each entry is here because a change to it can change a prediction:
INFERENCE_SOURCE_FILES: tuple[str, ...] = (
    # prepare_features, the feature contract and the canonical column order.
    "src/churn/preprocessing/contracts.py",
    # TotalChargesCleaner — pickled BY REFERENCE inside the persisted pipeline.
    "src/churn/preprocessing/transformers.py",
    # The ColumnTransformer's configuration and the step names it is read by.
    "src/churn/preprocessing/pipeline.py",
    # build_feature_preprocessor, which assembles the preprocessing pipeline.
    "src/churn/features/pipeline.py",
    # The estimator builders: build_modern_logistic and its pipeline wrapper.
    "src/churn/modeling/tuning.py",
    # The solver, max_iter and step-name constants those builders read.
    "src/churn/modeling/models.py",
    # encode_target, which fixes which label is the positive class.
    "src/churn/preprocessing/target.py",
    # This module: the decision rule, the positive-column resolution and the
    # comparison itself. Changing `>=` to `>` here would change every decision.
    "src/churn/modeling/freeze.py",
)

#: Configuration and dependency pins. The first fixes the seed and the target
#: definition; the last two fix the library versions whose behaviour the fitted
#: object and the transformations depend on.
CONFIGURATION_FILES: tuple[str, ...] = (
    "configs/base.toml",
    "pyproject.toml",
    "uv.lock",
)


class FrozenModelError(RuntimeError):
    """The persisted model is not the one the decision policy describes."""


class IntegrityError(RuntimeError):
    """One or more integrity checks on the frozen policy failed."""


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in binary so line endings cannot alter it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(path: Path) -> str:
    """Return the SHA-256 of a text file with its newlines normalised to ``\\n``.

    Deliberately **not** :func:`file_digest`. A raw byte hash of a source file
    changes when the checkout's line endings change, which would fail the freeze
    for a reason that has nothing to do with the code. ``.gitattributes`` already
    pins this repository to LF, so the two agree today; normalising makes the
    digest survive a checkout that ignores it — a zip download, or a Colab
    upload, both of which this project explicitly supports.

    Everything else is hashed exactly: no whitespace stripping, no comment
    removal, no parsing. A change of one character in a docstring changes the
    digest, and that is the intended, fail-closed behaviour.
    """
    text = path.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_digests(paths: tuple[str, ...], root: Path | None = None) -> dict[str, str]:
    """Return ``{repo-relative path: source digest}`` for each file, in order."""
    base = root or PROJECT_ROOT
    return {path: source_digest(base / path) for path in paths}


def function_source_digest(function: Callable[..., object]) -> str:
    """Return the SHA-256 of one function's source text.

    A **diagnostic** fingerprint, narrower than the module digest that gates
    alongside it. When a freeze fails, the pair localises the change: both
    moving means the function itself was edited, only the module moving means
    something else in the same file was.

    It is not sufficient on its own and is never used on its own.
    ``prepare_features`` delegates to ``validate_feature_matrix`` and reads
    ``FEATURE_COLUMNS``, neither of which appears in its own source, so a hash of
    the function alone would miss a change that alters what it returns. The
    module digest is what actually closes that gap.

    Uses :func:`inspect.getsource`, which reads the file through ``linecache`` in
    text mode: the result is newline-normalised for the same reason
    :func:`source_digest` normalises, and there is no source parsing involved.
    """
    text = inspect.getsource(function).replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_digest(values: object) -> str:
    """Return the SHA-256 of a canonical JSON encoding of ``values``.

    ``sort_keys`` is deliberately **not** used: the orderings that matter here —
    feature order, class order, coefficient order — are part of what is being
    fingerprinted, and sorting them would hide a reordering that changes what the
    model does.
    """
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def fit_final_pipeline(features: pd.DataFrame, target: np.ndarray) -> Pipeline:
    """Fit the frozen candidate on the data it is given, and return it.

    The estimator comes from the Phase 8B factory rather than being restated, so
    "the frozen model" is a fact about the code. The caller supplies the data;
    this function has no way to load anything, which is what keeps the holdout
    unreachable from here.

    Args:
        features: Feature matrix, already through ``prepare_features``.
        target: Encoded target aligned with ``features``.

    Returns:
        The fitted pipeline.
    """
    pipeline = build_modern_logistic_pipeline().fit(features, np.asarray(target))
    logger.info("Final pipeline fitted on %d rows.", len(features))
    return pipeline


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Return the names of the columns the classifier actually receives."""
    encoder = pipeline.named_steps[PREPROCESSOR_STEP].named_steps[ENCODER_STEP]
    return [str(name) for name in encoder.get_feature_names_out()]


def model_state(pipeline: Pipeline) -> dict[str, object]:
    """Return every learned parameter of a fitted pipeline, losslessly.

    Read explicitly, step by step, rather than by walking the object graph for
    attributes ending in an underscore. A generic walk would silently keep
    working if the pipeline's structure changed underneath it, and the whole
    point of a freeze is to break loudly instead.

    Floats survive the round trip exactly: ``ndarray.tolist()`` yields Python
    floats and :func:`json.dumps` writes the shortest decimal string that reads
    back as the same double.

    Args:
        pipeline: A fitted pipeline built by :func:`fit_final_pipeline`.

    Returns:
        A JSON-encodable mapping of the fitted state.

    Raises:
        FrozenModelError: If the pipeline does not have the expected structure or
            has not been fitted.
    """
    try:
        preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]
        classifier = pipeline.named_steps[CLASSIFIER_STEP]
        encoder = preprocessor.named_steps[ENCODER_STEP]
        scaler = encoder.named_transformers_[NUMERIC_STEP]
        one_hot = encoder.named_transformers_[CATEGORICAL_STEP]
        coefficients = classifier.coef_
    except (KeyError, AttributeError) as error:
        raise FrozenModelError(
            "The pipeline does not have the frozen structure "
            f"({PREPROCESSOR_STEP} -> [{CLEANER_STEP}, {ENCODER_STEP}] -> {CLASSIFIER_STEP}), "
            "or it has not been fitted. A fingerprint of it would describe nothing."
        ) from error

    return {
        "transformed_feature_names": transformed_feature_names(pipeline),
        "classifier": {
            "type": type(classifier).__name__,
            "classes": classifier.classes_.tolist(),
            "coefficients": coefficients.tolist(),
            "intercept": classifier.intercept_.tolist(),
            "n_iter": classifier.n_iter_.tolist(),
        },
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "variance": scaler.var_.tolist(),
            "n_samples_seen": int(scaler.n_samples_seen_),
        },
        "one_hot_categories": [category.tolist() for category in one_hot.categories_],
    }


def model_fingerprint(pipeline: Pipeline) -> str:
    """Return the SHA-256 of the learned parameters of a fitted pipeline.

    This is the **identity of the model**, and the check that decides whether an
    object loaded later is the frozen one. It is invariant to how the object was
    serialised and depends only on what the model does.
    """
    return text_digest(model_state(pipeline))


def dump_pipeline(pipeline: Pipeline, path: Path | None = None) -> Path:
    """Serialise a fitted pipeline with the pinned settings.

    Args:
        pipeline: The fitted pipeline.
        path: Destination. Defaults to :data:`PIPELINE_PATH`.

    Returns:
        The path written.
    """
    destination = path or PIPELINE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        pipeline,
        destination,
        compress=SERIALIZATION_COMPRESS,
        protocol=SERIALIZATION_PROTOCOL,
    )
    logger.info("Wrote frozen pipeline: %s (%d bytes)", destination, destination.stat().st_size)
    return destination


def load_pipeline(path: Path | None = None) -> Pipeline:
    """Read a serialised pipeline back.

    Raises:
        FileNotFoundError: If the artefact is absent, with the command that
            regenerates it — the file is deliberately not versioned.
    """
    source = path or PIPELINE_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Frozen pipeline not found at {source}. It is a regenerable artefact and is not "
            "committed; rebuild it with `uv run python scripts/freeze_model.py`."
        )
    return joblib.load(source)


def positive_class_column(pipeline: Pipeline) -> int:
    """Return the ``predict_proba`` column holding P(Churn = Yes).

    Resolved from ``classes_`` rather than assumed to be ``1``. The assumption
    happens to hold for this estimator, and it is still the wrong thing to write
    down: the column is a property of the fitted object, and a rule that silently
    reads the wrong one inverts every decision without raising anything.

    Raises:
        FrozenModelError: If the positive label is not among the fitted classes.
    """
    classes = pipeline.named_steps[CLASSIFIER_STEP].classes_
    matches = np.flatnonzero(np.asarray(classes) == POSITIVE_CLASS_LABEL)
    if matches.size != 1:
        raise FrozenModelError(
            f"The positive label {POSITIVE_CLASS_LABEL!r} appears {matches.size} time(s) in "
            f"classes_={list(classes)}. The probability column of the positive class is not "
            "identifiable, so no decision rule can be applied."
        )
    return int(matches[0])


def positive_probability(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return P(Churn = Yes) for each row, from the column ``classes_`` says.

    An integrity primitive, not a prediction interface: it exists so the checks
    in this module and the tests around them read the same column the frozen rule
    reads. The public inference boundary is Phase 11's subject.
    """
    return pipeline.predict_proba(features)[:, positive_class_column(pipeline)]


def apply_decision_rule(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the frozen rule ``positive = probability >= threshold``.

    One implementation, so the comparison cannot drift. ``>=`` and ``>`` differ
    on exactly the customers whose probability equals the threshold; with a
    threshold read from a stored decimal that set is usually empty and
    occasionally is not, and "usually empty" is not a specification.
    """
    return (np.asarray(probability, dtype=float) >= float(threshold)).astype(int)


@dataclass(frozen=True)
class IntegrityCheck:
    """One named integrity check and what it observed."""

    name: str
    passed: bool
    detail: str


def _check(name: str, passed: bool, detail: str) -> IntegrityCheck:
    return IntegrityCheck(name=name, passed=bool(passed), detail=detail)


def check_threshold(threshold: object, expected: float) -> list[IntegrityCheck]:
    """Check that a stored threshold is present, numeric, usable and unchanged."""
    checks = [
        _check("threshold_present", threshold is not None, f"value={threshold!r}"),
        _check(
            "threshold_is_numeric",
            isinstance(threshold, float | int) and not isinstance(threshold, bool),
            f"type={type(threshold).__name__}",
        ),
    ]
    numeric = isinstance(threshold, float | int) and not isinstance(threshold, bool)
    value = float(threshold) if numeric else float("nan")
    checks.append(
        _check(
            "threshold_is_finite_and_a_probability",
            numeric and np.isfinite(value) and 0.0 < value < 1.0,
            f"value={value!r}",
        )
    )
    checks.append(
        _check(
            "threshold_matches_phase9b_exactly",
            numeric and value == float(expected),
            f"policy={value!r} phase9b={float(expected)!r}",
        )
    )
    return checks


def check_policy_vocabulary(calibration_policy: str, threshold_policy: str) -> list[IntegrityCheck]:
    """Check that both policy names are recognised values, not free text."""
    return [
        _check(
            "calibration_policy_recognised",
            calibration_policy in KNOWN_CALIBRATION_POLICIES,
            f"{calibration_policy!r} in {sorted(KNOWN_CALIBRATION_POLICIES)}",
        ),
        _check(
            "threshold_policy_recognised",
            threshold_policy in KNOWN_THRESHOLD_POLICIES,
            f"{threshold_policy!r} in {sorted(KNOWN_THRESHOLD_POLICIES)}",
        ),
    ]


def check_decision_rule(
    comparison: str,
    positive_column: int,
    classes: list[int],
) -> list[IntegrityCheck]:
    """Check that the decision rule leaves nothing to interpretation."""
    return [
        _check(
            "decision_comparison_is_greater_or_equal",
            comparison == DECISION_COMPARISON,
            f"comparison={comparison!r}",
        ),
        _check(
            "positive_class_column_is_unambiguous",
            0 <= positive_column < len(classes)
            and classes[positive_column] == POSITIVE_CLASS_LABEL,
            f"column={positive_column} classes={classes}",
        ),
        _check(
            "classes_are_the_encoded_binary_target",
            classes == [NEGATIVE_CLASS_LABEL, POSITIVE_CLASS_LABEL],
            f"classes={classes}",
        ),
    ]


def _compare_digests(recorded: Mapping[str, str], root: Path) -> list[str]:
    """Return a description of every path whose digest no longer matches."""
    divergent: list[str] = []
    for path, expected in recorded.items():
        target = root / path
        if not target.is_file():
            divergent.append(f"{path}: missing")
            continue
        actual = source_digest(target)
        if actual != expected:
            divergent.append(f"{path}: {actual[:12]}… != {expected[:12]}…")
    return divergent


def check_source_provenance(
    recorded_sources: Mapping[str, str],
    recorded_configuration: Mapping[str, str],
    recorded_prepare_features: str,
    root: Path | None = None,
) -> list[IntegrityCheck]:
    """Check that nothing determining the inference function has moved.

    This is what makes the freeze cover the **whole** predictive function rather
    than only the serialised object. Three checks, each naming what diverged:

    1. the source of ``prepare_features`` itself, the diagnostic fingerprint;
    2. every module in :data:`INFERENCE_SOURCE_FILES`, the gate;
    3. the configuration and the dependency pins.

    Any divergence fails. There is no tolerance and no partial credit: a freeze
    that still verifies after its preprocessing changed is not a freeze.
    """
    base = root or PROJECT_ROOT

    actual_prepare = function_source_digest(PREPARE_FEATURES)
    checks = [
        _check(
            "prepare_features_source_unchanged",
            actual_prepare == recorded_prepare_features,
            f"actual={actual_prepare[:16]}… recorded={recorded_prepare_features[:16]}…",
        )
    ]

    divergent_sources = _compare_digests(recorded_sources, base)
    checks.append(
        _check(
            "inference_source_modules_unchanged",
            not divergent_sources,
            f"{len(recorded_sources) - len(divergent_sources)}/{len(recorded_sources)} match"
            + (f"; diverged: {', '.join(divergent_sources)}" if divergent_sources else ""),
        )
    )

    divergent_configuration = _compare_digests(recorded_configuration, base)
    checks.append(
        _check(
            "configuration_and_dependency_pins_unchanged",
            not divergent_configuration,
            f"{len(recorded_configuration) - len(divergent_configuration)}/"
            f"{len(recorded_configuration)} match"
            + (
                f"; diverged: {', '.join(divergent_configuration)}"
                if divergent_configuration
                else ""
            ),
        )
    )
    return checks


def check_pipeline_artefact(
    path: Path,
    expected_file_digest: str,
    expected_model_fingerprint: str,
) -> tuple[list[IntegrityCheck], Pipeline | None]:
    """Check the serialised pipeline against both recorded fingerprints.

    Returns:
        ``(checks, pipeline)``; the pipeline is ``None`` when it could not be
        loaded, so a caller can report every other check rather than stopping.
    """
    if not path.is_file():
        return [_check("pipeline_file_present", False, f"missing: {path}")], None

    checks = [_check("pipeline_file_present", True, str(path))]
    actual_file = file_digest(path)
    checks.append(
        _check(
            "pipeline_file_digest_matches",
            actual_file == expected_file_digest,
            f"actual={actual_file[:16]}… expected={expected_file_digest[:16]}…",
        )
    )

    try:
        pipeline = load_pipeline(path)
    except Exception as error:  # noqa: BLE001 - any failure here is a failed check
        checks.append(_check("pipeline_loads", False, f"{type(error).__name__}: {error}"))
        return checks, None

    checks.append(_check("pipeline_loads", True, type(pipeline).__name__))
    try:
        actual_model = model_fingerprint(pipeline)
    except FrozenModelError as error:
        checks.append(_check("pipeline_has_the_frozen_structure", False, str(error)))
        return checks, pipeline

    checks.append(_check("pipeline_has_the_frozen_structure", True, "preprocessor -> classifier"))
    checks.append(
        _check(
            "model_fingerprint_matches",
            actual_model == expected_model_fingerprint,
            f"actual={actual_model[:16]}… expected={expected_model_fingerprint[:16]}…",
        )
    )
    return checks, pipeline


def raise_on_failure(checks: list[IntegrityCheck]) -> list[IntegrityCheck]:
    """Return ``checks`` unchanged, or raise listing every one that failed.

    Every check is reported, not only the first: a freeze that fails in three
    ways should say so once rather than three runs later.

    Raises:
        IntegrityError: If any check failed.
    """
    failed = [check for check in checks if not check.passed]
    if failed:
        raise IntegrityError(
            "The frozen decision policy did not pass its integrity checks:\n  "
            + "\n  ".join(f"{check.name}: {check.detail}" for check in failed)
        )
    return checks
