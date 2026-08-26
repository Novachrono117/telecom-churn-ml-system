"""Static audit of the serving package: what is never *written*, not just never run.

The behavioural suite in ``test_isolation.py`` proves the forbidden operations do
not happen on the paths the tests exercise. This module proves they are not in
the source at all, on any path — including one nobody thought to test.

Both are needed. A behavioural test cannot cover a branch that only fires in
production; a static audit cannot see through an indirection. Together they close
each other's gap.

Only ``src/churn/serving`` is audited. The modules it imports from Phases 1-10 do
legitimately contain ``fit`` and dataset loaders — that is what they are for. The
question here is whether the *serving boundary* reaches for them.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import churn.serving

SERVING_PACKAGE = Path(churn.serving.__file__).parent

#: Attribute or function names that must never appear in the serving source.
#:
#: Matched as whole identifiers on ``Name``/``Attribute`` nodes, so ``predict``
#: does not match ``predict_proba``, ``predict_one`` or ``predict_batch``: those
#: are different names, and one of them is the point of the module.
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        # Training.
        "fit",
        "fit_transform",
        "partial_fit",
        "GridSearchCV",
        "RandomizedSearchCV",
        "CalibratedClassifierCV",
        "cross_val_score",
        "cross_validate",
        # Historical partitions.
        "load_holdout",
        "load_training_pool",
        "load_split",
        "split_dataset",
        "load_raw_typed",
        "load_raw_text",
        "read_csv",
        "read_parquet",
        # Arbitrary execution.
        "eval",
        "exec",
        "compile",
        "__import__",
        "import_module",
        "system",
        "popen",
        # scikit-learn's own 0.5 cut, which is not the frozen policy.
        "predict",
    }
)

#: Modules the serving package must not import.
#:
#: Deserialisation is where the import ban does the real work. A bare ``loads``
#: cannot be banned by name — ``json.loads`` is how the record and the policy are
#: read — so the guarantee is made one level up: ``pickle``, ``dill`` and
#: ``marshal`` are unreachable, so no ``loads`` in this package can execute
#: arbitrary code. The one pickle stream the process ever opens is the frozen
#: pipeline, loaded by joblib from a path that comes from trusted process
#: configuration and whose bytes were checked against the policy first.
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "pickle",
        "dill",
        "marshal",
        "subprocess",
        "os.system",
        "churn.data.loader",
        "churn.preprocessing.splitting",
        "churn.modeling.tuning",
        "churn.modeling.holdout",
        "churn.modeling.calibration",
        "churn.modeling.threshold",
    }
)


def serving_modules() -> list[Path]:
    return sorted(SERVING_PACKAGE.glob("*.py"))


def test_the_audit_actually_found_the_package() -> None:
    """An audit over zero files passes vacuously; this makes that impossible."""
    modules = serving_modules()

    assert len(modules) >= 8
    assert {path.name for path in modules} >= {
        "__init__.py",
        "api.py",
        "artifacts.py",
        "errors.py",
        "inference.py",
        "runtime.py",
        "schemas.py",
        "service.py",
        "settings.py",
    }


@pytest.mark.parametrize("module", serving_modules(), ids=lambda path: path.name)
def test_no_forbidden_call_is_written_in_the_serving_source(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    offenders: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            offenders.append(f"line {node.lineno}: .{node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            offenders.append(f"line {node.lineno}: {node.id}")

    assert not offenders, f"{module.name} reaches for {offenders}"


@pytest.mark.parametrize("module", serving_modules(), ids=lambda path: path.name)
def test_no_forbidden_module_is_imported(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not imported & FORBIDDEN_IMPORTS, f"{module.name} imports {imported & FORBIDDEN_IMPORTS}"


@pytest.mark.parametrize("module", serving_modules(), ids=lambda path: path.name)
def test_no_dataset_path_is_referenced(module: Path) -> None:
    """No string in the serving source points at a data partition."""
    source = module.read_text(encoding="utf-8")

    for needle in ("data/raw", "data/interim", "data/processed", ".csv", "split_manifest"):
        assert needle not in source, f"{module.name} mentions {needle!r}"


def test_predict_proba_is_reached_through_the_frozen_primitive() -> None:
    """The serving source does not even name predict_proba: it delegates.

    The positive-column resolution and the ``predict_proba`` call both live in
    ``churn.modeling.freeze``, whose source is hashed into the decision policy.
    Calling them from there rather than re-writing them here is what makes the
    served probability the same expression the freeze recorded.
    """
    import churn.serving.inference as inference_module
    from churn.modeling.freeze import positive_probability

    assert inference_module.positive_probability is positive_probability

    tree = ast.parse((SERVING_PACKAGE / "inference.py").read_text(encoding="utf-8"))
    called = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "predict_proba" not in called


def test_the_threshold_is_never_written_as_a_literal_in_the_serving_source() -> None:
    """It comes from the decision policy, so it must not be duplicated in code."""
    for module in serving_modules():
        source = module.read_text(encoding="utf-8")
        assert "0.3272694566222328" not in source, module.name


def test_the_calibration_policy_is_never_applied_only_asserted() -> None:
    """`NONE` is a fact to verify, not a transformation to perform."""
    for module in serving_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"predict_proba_calibrated", "calibrate"}, module.name
