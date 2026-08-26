"""End-to-end smoke test: one process, one startup, every endpoint, no dataset.

Deliberately runnable on a machine that has the frozen artefacts and nothing
else. The payload is the synthetic fixture published in the OpenAPI example; no
holdout row is used, and none is reachable from here.
"""

from __future__ import annotations

from typing import Any

from starlette.testclient import TestClient

from churn.serving.api import create_app
from churn.serving.schemas import EXAMPLE_RECORD
from churn.serving.settings import ServingSettings
from conftest import assert_predictions_agree

#: A second synthetic customer, invented to contrast with the example: long
#: tenure, two-year contract, automatic payment. Not a row of any partition.
SECOND_FIXTURE: dict[str, Any] = {
    **EXAMPLE_RECORD,
    "tenure": 66,
    "MonthlyCharges": 24.10,
    "TotalCharges": "1590.60",
    "InternetService": "No",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
}


def test_the_whole_boundary_comes_up_and_answers(settings: ServingSettings) -> None:
    """A real startup: the artefacts are located, verified and loaded here."""
    with TestClient(create_app(settings=settings)) as client:
        assert client.get("/health/live").json()["status"] == "alive"

        ready = client.get("/health/ready").json()
        assert ready["status"] == "ready"
        assert ready["startup_gates_passed"] > 0

        metadata = client.get("/api/v1/model").json()
        assert metadata["estimator"] == "LogisticRegression"
        assert metadata["calibration_policy"] == "NONE"
        assert metadata["comparison"] == ">="

        single = client.post("/api/v1/predict", json=EXAMPLE_RECORD)
        assert single.status_code == 200
        assert single.json()["model_fingerprint"] == metadata["model_fingerprint"]

        batch = client.post(
            "/api/v1/predict/batch", json={"records": [EXAMPLE_RECORD, SECOND_FIXTURE]}
        )
        assert batch.status_code == 200
        assert batch.json()["count"] == 2
        assert batch.json()["predictions"][0] == single.json()
        assert_predictions_agree(batch.json()["predictions"][0], single.json())

        invalid = client.post("/api/v1/predict", json={"tenure": 1})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "INVALID_REQUEST_SCHEMA"


def test_the_two_fixtures_are_scored_differently(settings: ServingSettings) -> None:
    """A smoke test that returned the same number for both would prove nothing."""
    with TestClient(create_app(settings=settings)) as client:
        body = client.post(
            "/api/v1/predict/batch", json={"records": [EXAMPLE_RECORD, SECOND_FIXTURE]}
        ).json()

    month_to_month, two_year = body["predictions"]
    assert month_to_month["churn_probability"] != two_year["churn_probability"]
    # Direction only, as a sanity check on wiring — not a claim about the model.
    assert month_to_month["churn_probability"] > two_year["churn_probability"]
