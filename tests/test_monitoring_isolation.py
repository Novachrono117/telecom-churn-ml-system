"""What monitoring must never do, proved behaviourally and then statically.

The prohibitions are the same family Phase 11 established for serving, with one
addition that is specific to this phase and is the most important of all: **no
metric may be computed against a label.** Monitoring detects that the population
changed. It does not detect that the model got worse, and a single accuracy number
computed from an invented label would erase that distinction for every reader who
came after.
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
from churn.modeling.freeze import apply_decision_rule, load_pipeline, positive_probability
from churn.monitoring import collector as collector_module
from churn.monitoring.collector import MonitoringCollector
from churn.monitoring.reference import load_reference_profile
from churn.monitoring.service import compare
from churn.monitoring.settings import get_monitoring_policy
from churn.preprocessing import splitting as splitting_module
from churn.preprocessing import transformers as transformers_module
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features

MONITORING_PACKAGE = Path(collector_module.__file__).parent

BASE_RECORD: dict[str, object] = {
    "tenure": 24,
    "MonthlyCharges": 70.0,
    "TotalCharges": "1680.00",
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
        raise AssertionError(f"{name} was called inside the monitoring path.")

    return guard


def monitoring_modules() -> list[Path]:
    return sorted(MONITORING_PACKAGE.glob("*.py"))


@pytest.fixture(scope="module")
def scored():
    """A synthetic window, scored once, for the behavioural guards to replay."""
    reference = load_reference_profile()
    pipeline = load_pipeline()
    features = prepare_features(pd.DataFrame([BASE_RECORD] * 20).loc[:, list(FEATURE_COLUMNS)])
    scores = np.asarray(positive_probability(pipeline, features), dtype=float)
    decisions = apply_decision_rule(scores, reference.score.threshold)
    return reference, features, scores, decisions


# --------------------------------------------------------------------------- #
# Behavioural.
# --------------------------------------------------------------------------- #


def test_monitoring_calls_no_fit_and_loads_no_dataset(
    scored, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every training entry point and every loader armed, then a real window observed."""
    reference, features, scores, decisions = scored

    for owner, method in _FIT_METHODS:
        monkeypatch.setattr(owner, method, _explode(f"{owner.__name__}.{method}"), raising=False)
    for module, function in _DATASET_ENTRY_POINTS:
        monkeypatch.setattr(module, function, _explode(function), raising=False)
    monkeypatch.setattr(pd, "read_csv", _explode("pandas.read_csv"))

    collector = MonitoringCollector(reference)
    collector.record_request(len(features))
    collector.observe(features, scores, decisions)
    report = compare(collector.snapshot(), reference)

    assert report.n_records == 20


def test_monitoring_never_calls_pipeline_predict(scored, monkeypatch: pytest.MonkeyPatch) -> None:
    """scikit-learn's own 0.5 cut is not the frozen policy, here or anywhere."""
    reference, features, scores, decisions = scored
    monkeypatch.setattr(Pipeline, "predict", _explode("Pipeline.predict"))

    collector = MonitoringCollector(reference)
    collector.observe(features, scores, decisions)

    assert collector.snapshot().n_records == 20


def test_the_collector_never_scores_anything_itself(
    scored, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It receives probabilities; it does not produce them.

    If the collector could score, monitoring would have a second inference path and
    the two could disagree about what the model said.
    """
    reference, features, scores, decisions = scored
    monkeypatch.setattr(Pipeline, "predict_proba", _explode("Pipeline.predict_proba"))

    collector = MonitoringCollector(reference)
    collector.observe(features, scores, decisions)

    assert collector.snapshot().score["count"] == 20


def test_comparing_a_window_touches_no_model(scored, monkeypatch: pytest.MonkeyPatch) -> None:
    """The comparison reads a snapshot and a profile. Nothing else exists to it."""
    reference, features, scores, decisions = scored
    collector = MonitoringCollector(reference)
    collector.observe(features, scores, decisions)
    snapshot = collector.snapshot()

    import joblib

    monkeypatch.setattr(joblib, "load", _explode("joblib.load"))
    report = compare(snapshot, reference)

    assert report.n_records == 20


# --------------------------------------------------------------------------- #
# No label-based metric, anywhere.
# --------------------------------------------------------------------------- #


def test_no_label_based_metric_is_computed() -> None:
    """The methodological boundary of the phase, enforced statically.

    Without production labels, a performance claim is not merely unsupported — it is
    unmakeable. So no metric function is imported, no metric name is written, and the
    report says ``evaluated: false`` rather than a number nobody could justify.
    """
    forbidden_names = {
        "accuracy_score",
        "recall_score",
        "precision_score",
        "f1_score",
        "roc_auc_score",
        "average_precision_score",
        "confusion_matrix",
        "log_loss",
        "brier_score_loss",
        "classification_report",
        "precision_recall_curve",
        "roc_curve",
        "encode_target",
    }
    forbidden_imports = {"sklearn.metrics", "churn.modeling.metrics", "churn.modeling.evaluation"}

    for module in monitoring_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        }
        assert not names & forbidden_names, f"{module.name} reaches for {names & forbidden_names}"

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not imported & forbidden_imports, module.name


def test_the_report_declares_performance_unevaluated(scored) -> None:
    reference, features, scores, decisions = scored
    collector = MonitoringCollector(reference)
    collector.observe(features, scores, decisions)

    record = compare(collector.snapshot(), reference).as_record()

    assert record["performance_degradation"] == {
        "evaluated": False,
        "reason": "no production ground-truth labels are available in this phase",
    }
    assert "requires ground-truth" in record["interpretation"]


def test_the_report_does_not_promise_a_calibrated_probability(scored) -> None:
    """Phase 9A froze calibration to NONE; monitoring must not quietly forget it.

    Reported over a window large enough to publish a score distribution: below the
    operational minimum ``prediction_drift`` is absent entirely, which is a privacy
    control and not a statement about calibration.
    """
    reference, features, scores, decisions = scored
    minimum = get_monitoring_policy().minimum_window_size
    collector = MonitoringCollector(reference)
    for _ in range(minimum):
        collector.observe(features, scores, decisions)

    record = compare(collector.snapshot(), reference).as_record()

    assert "calibration_policy is NONE" in record["calibration"]
    assert record["prediction_drift"]["calibration_policy"] == "NONE"


# --------------------------------------------------------------------------- #
# Static audit.
# --------------------------------------------------------------------------- #


def test_the_audit_found_the_package() -> None:
    """An audit over zero files passes vacuously."""
    modules = {path.name for path in monitoring_modules()}

    assert modules >= {
        "__init__.py",
        "build.py",
        "collector.py",
        "distributions.py",
        "drift.py",
        "reference.py",
        "results.py",
        "service.py",
        "settings.py",
        "structural.py",
    }


@pytest.mark.parametrize("module", monitoring_modules(), ids=lambda path: path.name)
def test_no_forbidden_call_is_written_in_the_monitoring_source(module: Path) -> None:
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
    }
    tree = ast.parse(module.read_text(encoding="utf-8"))
    offenders = [
        f"line {node.lineno}: {getattr(node, 'attr', None) or getattr(node, 'id', None)}"
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr in forbidden)
        or (isinstance(node, ast.Name) and node.id in forbidden)
    ]

    assert not offenders, f"{module.name} reaches for {offenders}"


@pytest.mark.parametrize("module", monitoring_modules(), ids=lambda path: path.name)
def test_no_monitoring_module_names_a_dataset_path(module: Path) -> None:
    source = module.read_text(encoding="utf-8")

    for needle in ("data/raw", "data/interim", "data/processed", "data/production", ".csv"):
        assert needle not in source, f"{module.name} mentions {needle!r}"


def test_no_monitoring_module_imports_a_web_framework() -> None:
    """The core must stay testable, and runnable, without a server."""
    for module in monitoring_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert not imported & {"fastapi", "starlette", "uvicorn"}, module.name


def test_the_frozen_threshold_is_never_written_as_a_literal() -> None:
    """It comes from the reference profile, which got it from the decision policy."""
    for module in monitoring_modules():
        assert "0.3272694566222328" not in module.read_text(encoding="utf-8"), module.name
