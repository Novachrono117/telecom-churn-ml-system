"""What the serving boundary must never do, proved by making it impossible.

These are behavioural tests, not static ones. Every forbidden operation is
monkeypatched to raise or to count, and then **real** requests are driven through
the **real** application against the **real** frozen pipeline. If any of them
were reachable, the request would fail rather than the assertion.

The static counterpart lives in ``test_static_audit.py``; the two answer
different questions and neither replaces the other. A grep proves the call is not
written; this proves it does not happen.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from starlette.testclient import TestClient

from churn.data import loader as loader_module
from churn.modeling.freeze import file_digest
from churn.preprocessing import splitting as splitting_module
from churn.preprocessing import transformers as transformers_module
from churn.serving.api import create_app
from churn.serving.service import ChurnInferenceService
from churn.serving.settings import ServingSettings

PREDICT = "/api/v1/predict"
BATCH = "/api/v1/predict/batch"

#: Every training entry point reachable from the installed package.
_FIT_METHODS = (
    (Pipeline, "fit"),
    (Pipeline, "fit_transform"),
    (ColumnTransformer, "fit"),
    (ColumnTransformer, "fit_transform"),
    (StandardScaler, "fit"),
    (StandardScaler, "fit_transform"),
    (StandardScaler, "partial_fit"),
    (OneHotEncoder, "fit"),
    (OneHotEncoder, "fit_transform"),
    (LogisticRegression, "fit"),
    (GridSearchCV, "fit"),
    (RandomizedSearchCV, "fit"),
    (CalibratedClassifierCV, "fit"),
    (transformers_module.TotalChargesCleaner, "fit"),
)

#: Every way a historical partition could be reached.
_DATASET_ENTRY_POINTS = (
    (loader_module, "load_raw_typed"),
    (loader_module, "load_raw_text"),
    (loader_module, "verify_raw_dataset"),
    (splitting_module, "split_dataset"),
    (splitting_module, "load_split"),
    (splitting_module, "load_training_pool"),
    (splitting_module, "load_holdout"),
)


def _explode(name: str) -> Any:
    def guard(*_: object, **__: object) -> None:
        raise AssertionError(f"{name} was called inside the serving boundary.")

    return guard


@pytest.fixture
def guarded_client(
    service: ChurnInferenceService, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A real client with every forbidden operation armed to explode."""
    for owner, method in _FIT_METHODS:
        monkeypatch.setattr(owner, method, _explode(f"{owner.__name__}.{method}"), raising=False)
    for module, function in _DATASET_ENTRY_POINTS:
        monkeypatch.setattr(module, function, _explode(function), raising=False)
    monkeypatch.setattr(pd, "read_csv", _explode("pandas.read_csv"))
    monkeypatch.setattr(Pipeline, "predict", _explode("Pipeline.predict"))

    with TestClient(create_app(service=service)) as client:
        yield client


def test_serving_a_request_calls_no_fit_and_loads_no_dataset(
    guarded_client: TestClient, record: dict[str, Any]
) -> None:
    """The whole prohibition, in one real round trip."""
    assert guarded_client.get("/health/live").status_code == 200
    assert guarded_client.get("/health/ready").status_code == 200
    assert guarded_client.get("/api/v1/model").status_code == 200
    assert guarded_client.post(PREDICT, json=record).status_code == 200
    assert guarded_client.post(BATCH, json={"records": [record] * 5}).status_code == 200


def test_a_rejected_payload_also_calls_nothing_forbidden(
    guarded_client: TestClient, record: dict[str, Any]
) -> None:
    """The error paths must be as clean as the happy one."""
    assert guarded_client.post(PREDICT, json={**record, "Contract": ""}).status_code == 422
    assert guarded_client.post(PREDICT, json={**record, "customerID": "x"}).status_code == 422
    assert guarded_client.post(BATCH, json={"records": []}).status_code == 422


def test_the_holdout_is_never_loaded(
    service: ChurnInferenceService, monkeypatch: pytest.MonkeyPatch, record: dict[str, Any]
) -> None:
    """Named on its own because it is the one partition that must never be reachable."""
    monkeypatch.setattr(splitting_module, "load_holdout", _explode("load_holdout"))

    with TestClient(create_app(service=service)) as client:
        assert client.post(PREDICT, json=record).status_code == 200


def test_predict_proba_is_what_produces_the_answer(
    service: ChurnInferenceService, monkeypatch: pytest.MonkeyPatch, record: dict[str, Any]
) -> None:
    calls: list[int] = []
    original = Pipeline.predict_proba

    def counted(self: Pipeline, X: Any) -> Any:  # noqa: ANN401, N803
        calls.append(1)
        return original(self, X)

    monkeypatch.setattr(Pipeline, "predict_proba", counted)
    monkeypatch.setattr(Pipeline, "predict", _explode("Pipeline.predict"))

    with TestClient(create_app(service=service)) as client:
        assert client.post(PREDICT, json=record).status_code == 200

    assert calls, "predict_proba was never called; the answer came from somewhere else."


def test_prepare_features_is_on_the_path_of_every_prediction(
    service: ChurnInferenceService, monkeypatch: pytest.MonkeyPatch, record: dict[str, Any]
) -> None:
    """The pipeline must never be handed anything that did not go through it."""
    import churn.serving.inference as inference_module

    seen: list[pd.DataFrame] = []
    original = inference_module.prepare_features

    def spy(matrix: pd.DataFrame) -> pd.DataFrame:
        prepared = original(matrix)
        seen.append(prepared)
        return prepared

    monkeypatch.setattr(inference_module, "prepare_features", spy)

    with TestClient(create_app(service=service)) as client:
        assert client.post(PREDICT, json=record).status_code == 200

    assert len(seen) == 1
    assert tuple(seen[0].columns) == service.artifacts.feature_columns


def test_the_pipeline_only_ever_receives_prepared_features(
    service: ChurnInferenceService, monkeypatch: pytest.MonkeyPatch, record: dict[str, Any]
) -> None:
    """Asserted on the argument, not on the call order."""
    received: list[tuple[str, ...]] = []
    original = Pipeline.predict_proba

    def capture(self: Pipeline, X: Any) -> Any:  # noqa: ANN401, N803
        received.append(tuple(X.columns))
        return original(self, X)

    monkeypatch.setattr(Pipeline, "predict_proba", capture)

    with TestClient(create_app(service=service)) as client:
        client.post(PREDICT, json=dict(reversed(list(record.items()))))

    assert received == [service.artifacts.feature_columns]


# --------------------------------------------------------------------------- #
# One load per process, and no writes.
# --------------------------------------------------------------------------- #


def test_the_model_is_loaded_once_per_process_not_once_per_request(
    settings: ServingSettings, monkeypatch: pytest.MonkeyPatch, record: dict[str, Any]
) -> None:
    import churn.serving.api as api_module

    loads: list[int] = []
    original = api_module.ChurnInferenceService.from_settings

    def counted(cls_settings: Any = None) -> ChurnInferenceService:  # noqa: ANN401
        loads.append(1)
        return original(cls_settings)

    monkeypatch.setattr(api_module.ChurnInferenceService, "from_settings", staticmethod(counted))

    with TestClient(create_app(settings=settings)) as client:
        for _ in range(5):
            assert client.post(PREDICT, json=record).status_code == 200

    assert len(loads) == 1


def test_every_request_reuses_the_same_service_instance(
    service: ChurnInferenceService, record: dict[str, Any]
) -> None:
    app = create_app(service=service)
    identities: set[int] = set()

    with TestClient(app) as client:
        for _ in range(3):
            client.post(PREDICT, json=record)
            identities.add(id(app.state.service))

    assert identities == {id(service)}


def test_serving_writes_nothing_to_the_repository(
    client: TestClient, policy_path: Path, pipeline_path: Path, record: dict[str, Any]
) -> None:
    """Inference is read-only with respect to the project."""
    before = {path: file_digest(path) for path in (policy_path, pipeline_path)}

    client.get("/health/ready")
    client.get("/api/v1/model")
    client.post(PREDICT, json=record)
    client.post(BATCH, json={"records": [record] * 10})
    client.post(PREDICT, json={**record, "Contract": ""})

    assert {path: file_digest(path) for path in before} == before


def test_serving_creates_no_file_anywhere_in_reports_or_artifacts(
    client: TestClient, repo_root: Path, record: dict[str, Any]
) -> None:
    watched = [repo_root / "reports", repo_root / "artifacts", repo_root / "data"]
    before = {
        path: sorted(str(item.relative_to(path)) for item in path.rglob("*"))
        for path in watched
        if path.is_dir()
    }

    client.post(PREDICT, json=record)
    client.post(BATCH, json={"records": [record] * 3})

    after = {
        path: sorted(str(item.relative_to(path)) for item in path.rglob("*"))
        for path in watched
        if path.is_dir()
    }
    assert after == before


def test_no_payload_and_no_probability_is_logged(
    client: TestClient, record: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Operational logs carry shape, never a person."""
    with caplog.at_level("DEBUG", logger="churn"):
        body = client.post(PREDICT, json=record).json()

    logged = "\n".join(entry.getMessage() for entry in caplog.records)
    assert "Electronic check" not in logged
    assert "70.35" not in logged
    assert str(body["churn_probability"]) not in logged
    assert "customerID" not in logged
