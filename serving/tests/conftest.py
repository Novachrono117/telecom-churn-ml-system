"""Shared fixtures for the serving suite.

Two kinds of test live here and they are deliberately not mixed:

* **unit** tests, which inject a double or call a primitive directly;
* **integration** tests, which load the *real* frozen artefacts and drive the
  *real* application. A boundary proved only against mocks is a boundary proved
  against itself, so at least one flow in this suite always exercises the actual
  pipeline on disk.

No fixture reads a dataset. The payloads here are invented; none is copied from
the training pool or the holdout, neither of which this package can reach.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import pytest
from starlette.testclient import TestClient

from churn.config import PROJECT_ROOT
from churn.serving.api import create_app
from churn.serving.schemas import EXAMPLE_RECORD
from churn.serving.service import ChurnInferenceService
from churn.serving.settings import ServingSettings, default_policy_path

#: Repository-relative location of the frozen pipeline, per the freeze.
PIPELINE_RELATIVE_PATH = "artifacts/model/churn_pipeline.joblib"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def policy_path(repo_root: Path) -> Path:
    return default_policy_path(repo_root)


@pytest.fixture(scope="session")
def pipeline_path(repo_root: Path) -> Path:
    return repo_root / PIPELINE_RELATIVE_PATH


@pytest.fixture(scope="session")
def settings(policy_path: Path) -> ServingSettings:
    """Real settings pointing at the repository's frozen artefacts."""
    return ServingSettings(policy_path=policy_path)


@pytest.fixture(scope="session")
def service(settings: ServingSettings) -> ChurnInferenceService:
    """The real inference service, built once — as a process would build it."""
    return ChurnInferenceService.from_settings(settings)


@pytest.fixture
def client(service: ChurnInferenceService) -> Iterator[TestClient]:
    """A client over the real frozen model, with the lifespan actually run."""
    with TestClient(create_app(service=service)) as test_client:
        yield test_client


@pytest.fixture
def record() -> dict[str, Any]:
    """A fresh copy of the synthetic example record."""
    return dict(EXAMPLE_RECORD)


@pytest.fixture
def frozen_copy(tmp_path: Path, policy_path: Path, pipeline_path: Path) -> tuple[Path, Path]:
    """Byte-identical copies of both artefacts in a temporary directory.

    The starting point for the fail-closed tests: each one corrupts exactly one
    thing, so the gate that fires identifies what it detected.
    """
    policy_copy = tmp_path / "decision_policy.json"
    pipeline_copy = tmp_path / "churn_pipeline.joblib"
    shutil.copyfile(policy_path, policy_copy)
    shutil.copyfile(pipeline_path, pipeline_copy)
    return policy_copy, pipeline_copy


def read_policy_json(path: Path) -> dict[str, Any]:
    """Read a policy file as plain JSON, for tests that need to corrupt it."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_policy_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a policy file the way Phase 9C writes it."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def dump_pipeline_copy(pipeline: Any, path: Path) -> Path:
    """Serialise a pipeline with the freeze's pinned joblib settings."""
    joblib.dump(pipeline, path, compress=0, protocol=5)
    return path


# --------------------------------------------------------------------------- #
# Comparing a single prediction with the same record inside a batch.
# --------------------------------------------------------------------------- #

#: Every field of a PredictionResponse. The equivalence contract covers all of
#: them, with no tolerance anywhere — including ``churn_probability``.
#:
#: An earlier revision of this suite allowed the probability to differ by a few
#: ULP between the two endpoints, because scoring a batch as one Nx46 matrix let
#: BLAS pick a different kernel than the 1x46 product the single endpoint used.
#: That leniency is gone: the service now scores every record as one row, so the
#: arithmetic is identical and the assertion can be `==`. If this ever needs a
#: tolerance again, something has silently stopped being canonical.
PREDICTION_FIELDS = (
    "churn_probability",
    "prediction",
    "decision",
    "threshold",
    "comparison",
    "calibration_policy",
    "model_fingerprint",
)


def ulp_distance(first: float, second: float) -> int:
    """Return how many representable doubles separate two positive floats.

    Kept for diagnostics: when an equality assertion fails, "3 ULP apart" and
    "a different number entirely" call for very different investigations.
    """
    import numpy as np

    def as_int(value: float) -> int:
        return int(np.frombuffer(np.float64(value).tobytes(), dtype=np.int64)[0])

    return abs(as_int(first) - as_int(second))


def prediction_fields(prediction: Any) -> dict[str, Any]:
    """Normalise a Prediction object or a decoded JSON body to a plain mapping."""
    return dict(prediction) if isinstance(prediction, dict) else dict(vars(prediction))


def assert_predictions_agree(left: Any, right: Any) -> None:
    """Assert two scorings of the same record are identical in every field.

    Exact equality, probability included. No `isclose`, no ULP budget.
    """
    first, second = prediction_fields(left), prediction_fields(right)

    assert set(first) == set(second) == set(PREDICTION_FIELDS)
    for field in PREDICTION_FIELDS:
        if field == "churn_probability" and first[field] != second[field]:
            raise AssertionError(
                f"probabilities differ by {ulp_distance(first[field], second[field])} ULP: "
                f"{first[field]!r} vs {second[field]!r}. The endpoints are no longer "
                "scoring through the same canonical one-row path."
            )
        assert first[field] == second[field], field
