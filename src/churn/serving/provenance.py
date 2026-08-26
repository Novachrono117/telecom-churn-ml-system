"""Runtime source provenance: the executable half of the freeze.

Why this exists
---------------
A ``.joblib`` protects *serialised state*. It does not protect *code*. Every class
inside the stream is pickled **by reference**: the file carries the string
``churn.preprocessing.transformers`` and unpickling imports whatever that module
currently defines. So an edit to ``TotalChargesCleaner.transform`` changes what the
loaded pipeline computes while

* ``pipeline_sha256`` — a hash of the file's bytes — stays identical,
* ``model_fingerprint_sha256`` — a hash of the learned parameters — stays identical,
* the transformed feature names, the classes, the threshold — all stay identical.

Nothing in the model's own fingerprints can see that change. Phase 9C anticipated it
and recorded a whole-file digest of every source file that materially determines the
frozen inference function, under ``code_provenance.source_digests``. This module is
what finally *acts* on those digests at serving time.

Stated plainly, and repeated in the report:

    pickle integrity protects serialized state; source provenance additionally
    protects the behavior of custom Python code resolved at runtime.

What is gated, and what deliberately is not
-------------------------------------------
Gating all eight recorded files would be the easy answer and the wrong one. Three of
them are never executed by this process, and pinning them would cost availability —
a serving instance refusing to start over an edit to a training builder it never
calls — while protecting no behaviour at all.

So each recorded file carries an explicit role, and the roles were **traced, not
assumed**. The trace ran three ways against the real running boundary:

1. a walk of the unpickled pipeline's object graph, collecting the defining module
   of every object in it — this is what identifies pickle-by-reference exposure;
2. a call-level profile (``sys.setprofile``) over a full startup *and* a real
   prediction, with every module already imported, so that "imported" and "called"
   are distinguished;
3. an AST scan of every module the profile showed executing, for ``from <recorded
   module> import ...`` bindings — because a module can determine behaviour through a
   constant that another module reads, without any function of its own being called.

``serving/tests/test_provenance.py`` re-derives all three at test time and asserts
they still agree with the declaration below. The classification is therefore
evidence, not commentary: moving a function between modules breaks the test.

A file is **runtime-effective** when changing it could change any of

* (i) the feature matrix handed to the pipeline,
* (ii) what the unpickled pipeline computes,
* (iii) which probability column is read,
* (iv) the decision rule applied,
* (v) the outcome of a startup integrity gate — that is, what this boundary is
  willing to accept as the frozen model.

and **training-only** when it is reachable in this process only as an artefact of the
Phases 1-9 import graph, is never called, has none of its values read, and therefore
cannot reach (i)-(v).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from churn.modeling.freeze import source_digest

logger = logging.getLogger(__name__)

#: Prefix the decision policy uses for repository-relative source paths.
_SOURCE_ROOT = "src/"


@dataclass(frozen=True)
class SourceRole:
    """One recorded source file, its role at serving, and why."""

    path: str
    runtime_effective: bool
    evidence: str
    reason: str

    @property
    def module(self) -> str:
        """The importable module name this path denotes."""
        return module_for(self.path)


def module_for(path: str) -> str:
    """Map a policy-recorded repository path to an importable module name.

    ``src/churn/preprocessing/contracts.py`` becomes
    ``churn.preprocessing.contracts``. The *path* is only ever used as a key into
    the policy; what is actually hashed is the file the imported module reports.
    """
    relative = path[len(_SOURCE_ROOT) :] if path.startswith(_SOURCE_ROOT) else path
    return relative.removesuffix(".py").replace("/", ".").replace("\\", ".")


#: The role of every file in ``code_provenance.source_digests``.
#:
#: ``evidence`` names which arm of the trace established the role; ``reason`` says
#: what would break. Both are asserted against a fresh trace by the tests.
SOURCE_ROLES: tuple[SourceRole, ...] = (
    SourceRole(
        path="src/churn/preprocessing/contracts.py",
        runtime_effective=True,
        evidence="called: prepare_features, validate_feature_matrix and _blank_mask all "
        "execute on every request",
        reason="(i) it IS the input side of the inference function. It decides which "
        "columns reach the model, in which order, and which values are rejected as "
        "blank. An edit here changes the matrix the pipeline is handed while every "
        "model fingerprint stays identical",
    ),
    SourceRole(
        path="src/churn/preprocessing/transformers.py",
        runtime_effective=True,
        evidence="called on every request, and the only module owning a class inside "
        "the unpickled pipeline (TotalChargesCleaner)",
        reason="(ii) the canonical pickle-by-reference exposure. The stream stores the "
        "module path, not the code; unpickling binds whatever transform() currently "
        "is. Editing it changes what the loaded pipeline computes with pipeline_sha256 "
        "and model_fingerprint_sha256 both unmoved",
    ),
    SourceRole(
        path="src/churn/modeling/freeze.py",
        runtime_effective=True,
        evidence="called on every request (positive_probability, positive_class_column, "
        "apply_decision_rule) and throughout startup (load_pipeline, model_fingerprint, "
        "transformed_feature_names, file_digest)",
        reason="(iii)(iv)(v) it implements the decision rule itself. Changing >= to > "
        "here moves every customer sitting on the threshold; changing the positive-column "
        "resolution inverts every decision; changing the fingerprint computation decides "
        "what the startup gates accept",
    ),
    SourceRole(
        path="src/churn/modeling/models.py",
        runtime_effective=True,
        evidence="not called, but CLASSIFIER_STEP and PREPROCESSOR_STEP are bound by "
        "churn.modeling.freeze and by churn.serving.artifacts",
        reason="(iii) CLASSIFIER_STEP is dereferenced on the request path, inside "
        "positive_class_column: pipeline.named_steps[CLASSIFIER_STEP].classes_. A "
        "different value resolves a different step, so the probability column read "
        "would no longer be the one classes_ describes",
    ),
    SourceRole(
        path="src/churn/preprocessing/pipeline.py",
        runtime_effective=True,
        evidence="not called, but CATEGORICAL_STEP, CLEANER_STEP, ENCODER_STEP and "
        "NUMERIC_STEP are bound by churn.modeling.freeze",
        reason="(v) those constants select which sub-transformer model_state() and "
        "transformed_feature_names() read, and both run as startup gates. A changed "
        "value changes what the model fingerprint is computed over, so a model that is "
        "not the frozen one could pass verification",
    ),
    SourceRole(
        path="src/churn/modeling/tuning.py",
        runtime_effective=False,
        evidence="not called; build_modern_logistic_pipeline is bound by "
        "churn.modeling.freeze but is only ever invoked from fit_final_pipeline, which "
        "this process never reaches",
        reason="it builds an UNFITTED estimator. The object being served was fitted in "
        "Phase 9C and is read from disk; no builder runs. Gating on it would refuse to "
        "start over an edit that cannot reach a prediction",
    ),
    SourceRole(
        path="src/churn/features/pipeline.py",
        runtime_effective=False,
        evidence="not called, and not bound by any module the profile showed executing; "
        "reachable only through churn.modeling.tuning, which is itself never called",
        reason="build_feature_preprocessor assembles a preprocessor at fit time. At "
        "serving the assembled object is unpickled, already configured, so the assembler "
        "never runs",
    ),
    SourceRole(
        path="src/churn/preprocessing/target.py",
        runtime_effective=False,
        evidence="not called, and not imported by any module the profile showed executing",
        reason="encode_target maps 'Yes'/'No' onto 1/0 during training. Serving never "
        "sees a label: the positive class it acts on is read from the fitted classes_ "
        "and cross-checked against the policy, never recomputed from this module",
    ),
)


def runtime_effective_roles() -> tuple[SourceRole, ...]:
    """Return the roles whose source is gated at startup."""
    return tuple(role for role in SOURCE_ROLES if role.runtime_effective)


def training_only_roles() -> tuple[SourceRole, ...]:
    """Return the roles that are recorded by the freeze but not gated here."""
    return tuple(role for role in SOURCE_ROLES if not role.runtime_effective)


def installed_source_path(module: str) -> Path:
    """Return the file the **already-imported** module came from.

    Two deliberate choices here.

    *Resolved through the module object, never by joining a repository root with a
    relative path.* The digest has to describe the code this process will actually
    execute, wherever it happens to be installed. Nothing in this module compares,
    stores or logs an absolute path — the path is a means of reading the source,
    and the source text is what is compared.

    *Read from* ``sys.modules``, *not imported on demand.* The runtime-effective
    modules are all pulled in by ``churn.serving.artifacts`` before any gate runs,
    so they are already loaded, and looking them up rather than importing them has
    two benefits: the digest is taken from exactly the code the interpreter has
    bound — not from a file that a fresh import might resolve differently — and the
    serving package keeps zero dynamic-import surface, which its static audit
    enforces.

    Raises:
        ModuleNotFoundError: If the module is not loaded. Unverifiable code is a
            failed gate, never something to skip.
        FileNotFoundError: If it has no source file — a namespace package, a frozen
            module or an extension.
    """
    loaded = sys.modules.get(module)
    if loaded is None:
        raise ModuleNotFoundError(
            f"Module {module!r} is not loaded in this process, so the code that would "
            "run cannot be compared with the freeze."
        )
    origin = getattr(loaded, "__file__", None)
    if not origin:
        raise FileNotFoundError(
            f"Module {module!r} reports no source file, so the code this process "
            "would execute cannot be verified against the freeze."
        )
    return Path(origin).resolve()


def installed_source_digest(module: str) -> str:
    """Return the SHA-256 of the imported module's source.

    Uses :func:`churn.modeling.freeze.source_digest`, newline-normalised, which is
    the same function and the same convention the policy's digests were produced
    with. A raw byte hash would fail on a CRLF checkout for a reason that has
    nothing to do with the code.
    """
    return source_digest(installed_source_path(module))


#: How a caller supplies digests. Overridden in the tests so that divergence can be
#: simulated without ever editing a real source file.
DigestReader = Callable[[str], str]


def runtime_source_checks(
    recorded: Mapping[str, str],
    digest_of: DigestReader | None = None,
) -> list[tuple[str, bool, str]]:
    """Compare every runtime-effective source with the digest the policy recorded.

    Returns ``(name, passed, detail)`` triples rather than a richer type so that
    this module stays free of any dependency on the startup layer that consumes it.

    Args:
        recorded: ``code_provenance.source_digests`` from the decision policy.
        digest_of: Resolves a module name to the digest of its installed source.
            Defaults to :func:`installed_source_digest`.

    Returns:
        One check per runtime-effective file, plus one check that the declaration
        below covers every file the policy recorded — an unclassified entry must
        fail loudly rather than be silently skipped.
    """
    read = digest_of or installed_source_digest
    checks: list[tuple[str, bool, str]] = []

    classified = {role.path for role in SOURCE_ROLES}
    unclassified = sorted(set(recorded) - classified)
    missing_from_policy = sorted(classified - set(recorded))
    checks.append(
        (
            "runtime_source_classification_covers_the_policy",
            not unclassified and not missing_from_policy,
            f"{len(classified & set(recorded))}/{len(recorded)} recorded file(s) classified"
            + (f"; unclassified: {unclassified}" if unclassified else "")
            + (
                f"; declared but absent from the policy: {missing_from_policy}"
                if missing_from_policy
                else ""
            ),
        )
    )

    for role in runtime_effective_roles():
        name = f"runtime_source_{role.module.replace('.', '_')}_unchanged"
        expected = recorded.get(role.path)
        if expected is None:
            checks.append((name, False, f"the policy records no digest for {role.path}"))
            continue
        try:
            actual = read(role.module)
        except (ModuleNotFoundError, FileNotFoundError, OSError) as error:
            checks.append((name, False, f"{type(error).__name__}: {error}"))
            continue
        checks.append(
            (
                name,
                actual == expected,
                f"actual={actual[:16]}… recorded={expected[:16]}…",
            )
        )

    return checks


__all__ = [
    "SOURCE_ROLES",
    "DigestReader",
    "SourceRole",
    "installed_source_digest",
    "installed_source_path",
    "module_for",
    "runtime_effective_roles",
    "runtime_source_checks",
    "training_only_roles",
]
