"""What the portfolio package must never do, proved behaviourally and then statically.

Phase 13 is the phase where a project quietly starts modelling again: an explanation
is one refit away from being an explanation of a *different* model, and an
approximate attribution library is one import away from replacing an identity that
was already exact. Neither is possible here, and the tests say so rather than the
docstrings.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import loader as loader_module
from churn.modeling.freeze import load_pipeline, positive_probability
from churn.modeling.interpretation import extract_terms, feature_groups
from churn.portfolio import coupling as coupling_module
from churn.portfolio.explanation import explain_prepared_row
from churn.preprocessing import splitting as splitting_module
from churn.preprocessing import transformers as transformers_module
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features

PORTFOLIO_PACKAGE = Path(coupling_module.__file__).parent

BASE_RECORD: dict[str, object] = {
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "TotalCharges": "845.50",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

_FIT_METHODS = (
    (Pipeline, "fit"),
    (Pipeline, "fit_transform"),
    (ColumnTransformer, "fit"),
    (ColumnTransformer, "fit_transform"),
    (StandardScaler, "fit"),
    (StandardScaler, "partial_fit"),
    (OneHotEncoder, "fit"),
    (LogisticRegression, "fit"),
    (GridSearchCV, "fit"),
    (RandomizedSearchCV, "fit"),
    (CalibratedClassifierCV, "fit"),
    (transformers_module.TotalChargesCleaner, "fit"),
)

_DATASET_ENTRY_POINTS = (
    (loader_module, "load_raw_typed"),
    (loader_module, "load_raw_text"),
    (splitting_module, "split_dataset"),
    (splitting_module, "load_split"),
    (splitting_module, "load_training_pool"),
    (splitting_module, "load_holdout"),
)


def _explode(name: str) -> Any:
    def guard(*_: object, **__: object) -> None:
        raise AssertionError(f"{name} was called inside the portfolio path.")

    return guard


def portfolio_modules() -> list[Path]:
    return sorted(PORTFOLIO_PACKAGE.glob("*.py"))


@pytest.fixture(scope="module")
def scored():
    """One record, scored once, for the behavioural guards to replay."""
    pipeline = load_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    features = prepare_features(pd.DataFrame([BASE_RECORD]).loc[:, list(FEATURE_COLUMNS)])
    probability = float(np.asarray(positive_probability(pipeline, features), dtype=float)[0])
    return pipeline, terms, groups, features, probability


def _explain(scored):
    pipeline, terms, groups, features, probability = scored
    return explain_prepared_row(
        pipeline=pipeline,
        terms=terms,
        groups=groups,
        features=features,
        served_probability=probability,
        prediction=1,
        decision="churn",
        threshold=0.3272694566222328,
        comparison=">=",
        calibration_policy="NONE",
    )


# --------------------------------------------------------------------------- #
# Behavioural.
# --------------------------------------------------------------------------- #


def test_the_explanation_path_calls_no_fit(scored, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every training entry point armed, then a real explanation produced.

    Refitting the scaler or the encoder on the record being described would change
    the representation the coefficients were learned for, so the numbers would
    explain a model that does not exist.
    """
    for owner, method in _FIT_METHODS:
        monkeypatch.setattr(owner, method, _explode(f"{owner.__name__}.{method}"), raising=False)

    explanation = _explain(scored)

    assert len(explanation.contributions) == 19
    assert explanation.reconstruction.within_tolerance


def test_the_holdout_is_never_reachable(scored, monkeypatch: pytest.MonkeyPatch) -> None:
    """No loader, no read_csv. The explanation describes the payload, nothing else."""
    for module, function in _DATASET_ENTRY_POINTS:
        monkeypatch.setattr(module, function, _explode(function), raising=False)
    monkeypatch.setattr(pd, "read_csv", _explode("pandas.read_csv"))

    explanation = _explain(scored)

    assert explanation.churn_probability == scored[4]


def test_the_explanation_never_reaches_the_decision_rule(
    scored, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is handed a decision; it does not make one.

    A second place that applied a threshold would be a second place that could apply
    a different one.
    """
    from churn.modeling import freeze as freeze_module

    monkeypatch.setattr(freeze_module, "apply_decision_rule", _explode("apply_decision_rule"))

    explanation = _explain(scored)

    assert explanation.prediction == 1
    assert explanation.threshold == 0.3272694566222328


# --------------------------------------------------------------------------- #
# No approximate attribution, anywhere.
# --------------------------------------------------------------------------- #


def test_no_approximate_attribution_library_is_used() -> None:
    """The methodological claim of the phase, enforced statically.

    The identity is exact. Reaching for SHAP, LIME or permutation importance would
    approximate a function that can be evaluated in closed form, and would introduce
    a dependency, a random seed and an error term to do it.
    """
    forbidden_imports = {"shap", "lime", "eli5", "interpret", "captum", "alibi"}
    forbidden_names = {
        "KernelExplainer",
        "TreeExplainer",
        "DeepExplainer",
        "LimeTabularExplainer",
        "permutation_importance",
        "partial_dependence",
        "PartialDependenceDisplay",
    }

    for module in portfolio_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        offending = imported & forbidden_imports
        assert not offending, f"{module.name} imports {offending}"

        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert not names & forbidden_names, f"{module.name} reaches for {names & forbidden_names}"


def test_nothing_random_is_used() -> None:
    """No sampling means no seed to manage and no run-to-run variation to explain.

    Checked over the parsed source rather than the raw text: the prose deliberately
    says "no random seed", and a substring search would flag the very sentence that
    documents the property.
    """
    forbidden = {"random", "shuffle", "choice", "seed", "default_rng", "RandomState"}

    for module in portfolio_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert not used & forbidden, f"{module.name} calls {used & forbidden}"
        assert "random" not in imported, module.name


# --------------------------------------------------------------------------- #
# Static audit.
# --------------------------------------------------------------------------- #


def test_the_audit_found_the_package() -> None:
    """An audit over zero files passes vacuously."""
    modules = {path.name for path in portfolio_modules()}

    assert modules >= {
        "__init__.py",
        "coupling.py",
        "explanation.py",
        "metadata.py",
        "results.py",
    }


@pytest.mark.parametrize("module", portfolio_modules(), ids=lambda path: path.name)
def test_no_forbidden_call_is_written_in_the_portfolio_source(module: Path) -> None:
    forbidden = {
        "fit",
        "fit_transform",
        "partial_fit",
        "GridSearchCV",
        "RandomizedSearchCV",
        "CalibratedClassifierCV",
        "load_holdout",
        "load_training_pool",
        "load_split",
        "split_dataset",
        "load_raw_typed",
        "load_raw_text",
        "read_csv",
        "eval",
        "exec",
        "__import__",
        "predict",
        "predict_proba",
        "decision_function",
    }
    tree = ast.parse(module.read_text(encoding="utf-8"))
    offenders = [
        f"line {node.lineno}: {getattr(node, 'attr', None) or getattr(node, 'id', None)}"
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr in forbidden)
        or (isinstance(node, ast.Name) and node.id in forbidden)
    ]

    assert not offenders, f"{module.name} reaches for {offenders}"


@pytest.mark.parametrize("module", portfolio_modules(), ids=lambda path: path.name)
def test_no_portfolio_module_names_a_dataset_path(module: Path) -> None:
    source = module.read_text(encoding="utf-8")

    for needle in ("data/raw", "data/interim", "data/processed", ".csv"):
        assert needle not in source, f"{module.name} mentions {needle!r}"


def test_no_portfolio_module_imports_a_web_framework() -> None:
    """The core must stay testable, and runnable, without a server."""
    for module in portfolio_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"fastapi", "starlette", "uvicorn"}, module.name


def test_the_frozen_threshold_is_never_written_as_a_literal() -> None:
    """It arrives from the decision policy, through the serving boundary."""
    for module in portfolio_modules():
        assert "0.3272694566222328" not in module.read_text(encoding="utf-8"), module.name


def test_the_phase_10_module_is_reused_and_not_reimplemented() -> None:
    """One implementation of one identity. Two would be two chances to drift."""
    source = (PORTFOLIO_PACKAGE / "explanation.py").read_text(encoding="utf-8")

    assert "from churn.modeling.interpretation import" in source
    for primitive in ("group_contributions", "transform_features", "sigmoid", "logit_of"):
        assert primitive in source, primitive
    # And the identity itself is not spelled out a second time.
    assert "np.exp" not in source
    assert "1.0 / (1.0 +" not in source
