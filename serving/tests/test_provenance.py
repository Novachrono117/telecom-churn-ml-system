"""Runtime source provenance: the classification, and the gate it drives.

Two things are proved here, and they are different claims.

**The classification is right.** It is not taken on trust from the declaration in
:mod:`churn.serving.provenance`; it is re-derived at test time from the running
boundary — an object-graph walk of the unpickled pipeline, a call-level profile of a
real startup and a real prediction, and an AST scan for constant bindings — and the
declaration must agree with what the trace found. Move a function between modules and
this file fails.

**The gate works.** A divergent runtime-effective source stops the startup, one file
at a time, and a divergent training-only source does not — because it cannot reach a
prediction, and refusing to serve over it would cost availability for nothing.

No test here edits a real source file. Divergence is simulated through an injected
digest reader or a monkeypatched resolver, and the repository is asserted to be
untouched afterwards.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

from churn.config import PROJECT_ROOT
from churn.modeling.freeze import INFERENCE_SOURCE_FILES, source_digest
from churn.modeling.freeze_results import DecisionPolicy, load_decision_policy
from churn.serving.artifacts import load_frozen_artifacts
from churn.serving.errors import ArtifactIntegrityError
from churn.serving.provenance import (
    SOURCE_ROLES,
    installed_source_digest,
    installed_source_path,
    module_for,
    runtime_effective_roles,
    runtime_source_checks,
    training_only_roles,
)
from churn.serving.service import ChurnInferenceService
from churn.serving.settings import ServingSettings

DIVERGENT = "0" * 64


@pytest.fixture(scope="module")
def policy(policy_path: Path) -> DecisionPolicy:
    return load_decision_policy(policy_path)


@pytest.fixture(scope="module")
def recorded(policy: DecisionPolicy) -> dict[str, str]:
    return dict(policy.code_provenance.source_digests)


# --------------------------------------------------------------------------- #
# The classification, re-derived from the running boundary.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def trace(settings: ServingSettings) -> dict[str, Any]:
    """Run the three arms of the trace once and share the result.

    Every module is already imported by the time the profiler is installed, so a
    'call' event means a function actually ran — not merely that the module was
    pulled in by the Phases 1-9 import graph.
    """
    from churn.serving.schemas import EXAMPLE_RECORD

    # (1) which modules own an object inside the unpickled pipeline
    warm = ChurnInferenceService.from_settings(settings)
    owners: set[str] = set()
    seen: set[int] = set()

    def walk(obj: Any, depth: int = 0) -> None:
        if depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        owners.add(type(obj).__module__ or "")
        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                walk(value, depth + 1)
        if isinstance(obj, list | tuple):
            for value in obj:
                walk(value, depth + 1)
        if isinstance(obj, dict):
            for value in obj.values():
                walk(value, depth + 1)

    walk(warm.artifacts.pipeline)

    # (2) which files have a function actually called during startup + a request
    called: set[Path] = set()

    def profiler(frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            called.add(Path(frame.f_code.co_filename).resolve())

    sys.setprofile(profiler)
    try:
        service = ChurnInferenceService.from_settings(settings)
        service.predict_one(dict(EXAMPLE_RECORD))
    finally:
        sys.setprofile(None)

    # (3) which recorded modules are `from ... import`-ed by the files that ran
    policy_modules = {module_for(path) for path in INFERENCE_SOURCE_FILES}
    source_root = (PROJECT_ROOT / "src").resolve()
    bound: set[str] = set()
    for executed in called:
        if not executed.is_relative_to(source_root):
            continue
        tree = ast.parse(executed.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in policy_modules:
                bound.add(node.module)

    return {"owners": owners, "called": called, "bound": bound}


def test_the_declaration_covers_exactly_the_files_the_policy_records(
    recorded: dict[str, str],
) -> None:
    assert {role.path for role in SOURCE_ROLES} == set(recorded)
    assert set(recorded) == set(INFERENCE_SOURCE_FILES)
    assert len(SOURCE_ROLES) == 8


def test_the_trace_actually_observed_something(trace: dict[str, Any]) -> None:
    """A trace that saw nothing would make every assertion below vacuous."""
    assert trace["called"]
    assert trace["owners"]
    assert trace["bound"]


def test_every_runtime_effective_file_is_called_or_bound(trace: dict[str, Any]) -> None:
    """The positive half of the classification, checked against the trace."""
    for role in runtime_effective_roles():
        path = (PROJECT_ROOT / role.path).resolve()
        called = path in trace["called"]
        bound = role.module in trace["bound"]
        owns = role.module in trace["owners"]
        assert called or bound or owns, (
            f"{role.path} is declared runtime-effective but the trace found it "
            "neither called, nor bound by a module that ran, nor owning an object "
            "inside the pipeline"
        )


def test_no_training_only_file_is_called_or_owns_a_pickled_object(
    trace: dict[str, Any],
) -> None:
    """The negative half. A training-only file must not be running.

    Being *bound* is not enough to disqualify it: `churn.modeling.freeze` imports
    `build_modern_logistic_pipeline` from `tuning`, and never calls it. What must
    hold is that nothing in it executes and nothing of it is inside the pipeline.
    """
    for role in training_only_roles():
        path = (PROJECT_ROOT / role.path).resolve()
        assert path not in trace["called"], f"{role.path} is declared training-only but ran"
        assert role.module not in trace["owners"], (
            f"{role.path} is declared training-only but owns an object inside the pipeline"
        )


def test_transformers_is_the_module_pickled_by_reference(trace: dict[str, Any]) -> None:
    """The exposure this whole gate exists for, confirmed on the real artefact."""
    churn_owners = {name for name in trace["owners"] if name.startswith("churn")}

    assert churn_owners == {"churn.preprocessing.transformers"}


def test_every_role_states_evidence_and_a_reason() -> None:
    """A classification without a justification is an assertion."""
    for role in SOURCE_ROLES:
        assert len(role.evidence) > 30, role.path
        assert len(role.reason) > 30, role.path


def test_the_split_is_five_and_three() -> None:
    assert len(runtime_effective_roles()) == 5
    assert len(training_only_roles()) == 3


# --------------------------------------------------------------------------- #
# Resolution: content, not paths.
# --------------------------------------------------------------------------- #


def test_the_source_is_resolved_through_the_loaded_module() -> None:
    for role in SOURCE_ROLES:
        if not role.runtime_effective:
            continue
        resolved = installed_source_path(role.module)
        assert resolved.is_file()
        assert resolved == Path(sys.modules[role.module].__file__).resolve()


def test_the_digest_matches_the_policy_for_every_runtime_effective_file(
    recorded: dict[str, str],
) -> None:
    for role in runtime_effective_roles():
        assert installed_source_digest(role.module) == recorded[role.path], role.path


def test_the_digest_is_newline_normalised_not_a_byte_hash() -> None:
    """The policy's convention, reused. A CRLF checkout must not fail the gate."""
    for role in runtime_effective_roles():
        path = installed_source_path(role.module)
        assert installed_source_digest(role.module) == source_digest(path)


def test_a_module_that_is_not_loaded_is_a_failure_not_a_skip() -> None:
    with pytest.raises(ModuleNotFoundError):
        installed_source_path("churn.serving.definitely_not_a_module")


def test_no_absolute_path_is_baked_into_the_declaration() -> None:
    """The gate compares source content; it must not depend on where the repo lives."""
    for role in SOURCE_ROLES:
        assert not role.path.startswith("/")
        assert ":" not in role.path
        assert role.path.startswith("src/churn/")


# --------------------------------------------------------------------------- #
# The gate, provoked one file at a time.
# --------------------------------------------------------------------------- #


def test_an_intact_runtime_source_passes_every_gate(recorded: dict[str, str]) -> None:
    checks = runtime_source_checks(recorded)

    assert len(checks) == len(runtime_effective_roles()) + 1
    assert all(passed for _, passed, _ in checks)


@pytest.mark.parametrize(
    "module",
    [role.module for role in runtime_effective_roles()],
    ids=lambda name: name.rsplit(".", 1)[-1],
)
def test_a_divergent_runtime_effective_source_fails_its_gate(
    recorded: dict[str, str], module: str
) -> None:
    """One file at a time, so the gate that fires identifies what changed."""
    results = dict(
        (name, passed)
        for name, passed, _ in runtime_source_checks(
            recorded,
            digest_of=lambda asked, target=module: (
                DIVERGENT if asked == target else installed_source_digest(asked)
            ),
        )
    )

    failing = {name for name, passed in results.items() if not passed}
    assert failing == {f"runtime_source_{module.replace('.', '_')}_unchanged"}


def test_a_policy_missing_a_recorded_digest_fails(recorded: dict[str, str]) -> None:
    incomplete = {k: v for k, v in recorded.items() if k != "src/churn/modeling/freeze.py"}

    results = {name: passed for name, passed, _ in runtime_source_checks(incomplete)}

    assert results["runtime_source_churn_modeling_freeze_unchanged"] is False
    assert results["runtime_source_classification_covers_the_policy"] is False


def test_an_unclassified_file_in_the_policy_fails(recorded: dict[str, str]) -> None:
    """A new provenance entry must be classified, not silently ignored."""
    extended = {**recorded, "src/churn/modeling/something_new.py": DIVERGENT}

    results = {name: passed for name, passed, _ in runtime_source_checks(extended)}

    assert results["runtime_source_classification_covers_the_policy"] is False


def test_a_training_only_file_is_not_gated(recorded: dict[str, str]) -> None:
    """The other half of #11: availability is not spent on code that cannot run."""
    gated_modules = {
        name.removeprefix("runtime_source_").removesuffix("_unchanged")
        for name, _, _ in runtime_source_checks(recorded)
        if name != "runtime_source_classification_covers_the_policy"
    }

    for role in training_only_roles():
        assert role.module.replace(".", "_") not in gated_modules, role.path


def test_a_divergent_training_only_source_does_not_fail_anything(
    recorded: dict[str, str],
) -> None:
    """Simulated divergence in tuning.py, features/pipeline.py and target.py."""
    training_modules = {role.module for role in training_only_roles()}

    def reader(module: str) -> str:
        return DIVERGENT if module in training_modules else installed_source_digest(module)

    assert all(passed for _, passed, _ in runtime_source_checks(recorded, digest_of=reader))


# --------------------------------------------------------------------------- #
# End to end: a divergent source must stop the process, not warn.
# --------------------------------------------------------------------------- #


def _fail_startup_for(
    monkeypatch: pytest.MonkeyPatch, module: str, settings: ServingSettings
) -> ArtifactIntegrityError:
    """Simulate one divergent module and return the startup failure it caused."""
    from churn.serving import provenance as provenance_module

    real = provenance_module.installed_source_digest

    def fake(asked: str) -> str:
        return DIVERGENT if asked == module else real(asked)

    monkeypatch.setattr(provenance_module, "installed_source_digest", fake)

    with pytest.raises(ArtifactIntegrityError) as error:
        load_frozen_artifacts(settings)
    return error.value


@pytest.mark.parametrize(
    "module",
    ["churn.preprocessing.contracts", "churn.preprocessing.transformers", "churn.modeling.freeze"],
    ids=["contracts", "transformers", "freeze"],
)
def test_a_divergent_runtime_source_fails_the_startup(
    monkeypatch: pytest.MonkeyPatch, settings: ServingSettings, module: str
) -> None:
    """Fail-closed, end to end: the service is never constructed."""
    failure = _fail_startup_for(monkeypatch, module, settings)

    assert f"runtime_source_{module.replace('.', '_')}_unchanged" in str(failure)
    assert "does not start" in str(failure)


def test_a_changed_total_charges_cleaner_stops_the_service_before_any_request(
    monkeypatch: pytest.MonkeyPatch, settings: ServingSettings
) -> None:
    """The conceptual risk this gate exists for, stated as a test.

    ``TotalChargesCleaner`` is pickled BY REFERENCE inside the frozen pipeline: the
    stream carries ``churn.preprocessing.transformers`` and unpickling binds
    whatever ``transform`` currently is. An edit there changes the value every
    customer's ``TotalCharges`` becomes, and therefore the probability served —
    while ``pipeline_sha256`` and ``model_fingerprint_sha256`` both stay identical,
    because neither describes code.

    So the source digest is what catches it, and it must catch it at startup,
    before a single request has been answered.
    """
    from churn.preprocessing.transformers import TotalChargesCleaner

    # The class really is inside the artefact — the premise, not an assumption.
    intact = ChurnInferenceService.from_settings(settings)
    cleaner = intact.artifacts.pipeline.named_steps["preprocessor"].named_steps["total_charges"]
    assert isinstance(cleaner, TotalChargesCleaner)
    assert type(cleaner).__module__ == "churn.preprocessing.transformers"

    failure = _fail_startup_for(monkeypatch, "churn.preprocessing.transformers", settings)

    assert "runtime_source_churn_preprocessing_transformers_unchanged" in str(failure)
    assert "does not start" in str(failure)
    assert "No artefact is rebuilt" in str(failure)


def test_a_divergent_training_only_source_still_starts(
    monkeypatch: pytest.MonkeyPatch, settings: ServingSettings
) -> None:
    """Availability is not spent on a builder the process never calls."""
    from churn.serving import provenance as provenance_module

    real = provenance_module.installed_source_digest
    monkeypatch.setattr(
        provenance_module,
        "installed_source_digest",
        lambda asked: DIVERGENT if asked == "churn.modeling.tuning" else real(asked),
    )

    service = ChurnInferenceService.from_settings(settings)

    assert all(check.passed for check in service.artifacts.checks)


# --------------------------------------------------------------------------- #
# The tests must not have touched the repository.
# --------------------------------------------------------------------------- #


def test_no_real_source_file_was_modified(recorded: dict[str, str]) -> None:
    """Every simulation above went through an injected reader, never the disk."""
    for role in SOURCE_ROLES:
        path = installed_source_path(role.module)
        assert source_digest(path) == recorded[role.path], role.path
