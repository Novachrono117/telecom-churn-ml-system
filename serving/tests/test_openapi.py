"""The published contract. What OpenAPI says must be what the boundary does."""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from churn.preprocessing.contracts import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from churn.serving.schemas import EXAMPLE_RECORD


@pytest.fixture
def schema(client: TestClient) -> dict[str, Any]:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


@pytest.fixture
def request_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return schema["components"]["schemas"]["PredictionRequest"]


def test_the_document_is_a_valid_openapi_schema(schema: dict[str, Any]) -> None:
    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"] == "Churn Prediction API"
    assert "paths" in schema and "components" in schema


def test_every_endpoint_is_published(schema: dict[str, Any]) -> None:
    assert {
        "/health/live",
        "/health/ready",
        "/api/v1/model",
        "/api/v1/predict",
        "/api/v1/predict/batch",
    } <= set(schema["paths"])


def test_the_request_declares_exactly_the_nineteen_contracted_features(
    request_schema: dict[str, Any],
) -> None:
    properties = set(request_schema["properties"])

    assert properties == set(FEATURE_COLUMNS)
    assert len(properties) == 19
    assert set(request_schema["required"]) == set(FEATURE_COLUMNS)


def test_the_identifier_and_the_target_are_not_accepted_fields(
    request_schema: dict[str, Any],
) -> None:
    assert "customerID" not in request_schema["properties"]
    assert "Churn" not in request_schema["properties"]


def test_undeclared_fields_are_forbidden_by_the_published_schema(
    request_schema: dict[str, Any],
) -> None:
    assert request_schema["additionalProperties"] is False


def test_no_category_is_published_as_a_closed_enum(request_schema: dict[str, Any]) -> None:
    """The frozen encoder handles unknown categories; the schema must not pre-empt it."""
    for feature in CATEGORICAL_FEATURES:
        if feature == "SeniorCitizen":
            continue
        assert "enum" not in request_schema["properties"][feature], feature


def test_no_range_constraint_is_published_for_the_numeric_features(
    request_schema: dict[str, Any],
) -> None:
    """Training range is not input validity."""
    for feature in NUMERIC_FEATURES:
        published = request_schema["properties"][feature]
        assert not {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"} & set(published)


def test_total_charges_may_arrive_blank(request_schema: dict[str, Any]) -> None:
    """The structural-zero rule needs the raw value to survive the schema."""
    published = request_schema["properties"]["TotalCharges"]

    assert "anyOf" in published
    assert {"string", "number", "null"} <= {member.get("type") for member in published["anyOf"]}


def test_senior_citizen_is_published_as_an_integer(request_schema: dict[str, Any]) -> None:
    assert request_schema["properties"]["SeniorCitizen"]["type"] == "integer"


def test_the_probability_is_documented_as_an_uncalibrated_score(
    schema: dict[str, Any],
) -> None:
    """Phase 9A froze calibration to NONE; the docs must not promise otherwise."""
    response_schema = schema["components"]["schemas"]["PredictionResponse"]
    probability = response_schema["properties"]["churn_probability"]
    description = schema["info"]["description"]

    assert "Not calibrated" in probability["description"]
    assert "**not** a calibrated probability" in description
    assert "P(Churn = 1)" in description


def test_the_decision_semantics_are_documented(schema: dict[str, Any]) -> None:
    description = schema["info"]["description"]

    assert "churn_probability >= threshold" in description
    assert "prepare_features" in description
    assert "classes_" in description


def test_the_documented_decision_field_is_closed(schema: dict[str, Any]) -> None:
    response_schema = schema["components"]["schemas"]["PredictionResponse"]

    assert response_schema["properties"]["decision"]["enum"] == ["retained", "churn"]
    assert response_schema["properties"]["prediction"]["enum"] == [0, 1]
    comparison = response_schema["properties"]["comparison"]
    assert comparison.get("const") == ">=" or comparison.get("enum") == [">="]


def test_the_error_envelope_is_published(schema: dict[str, Any]) -> None:
    assert "ErrorResponse" in schema["components"]["schemas"]
    responses = schema["paths"]["/api/v1/predict"]["post"]["responses"]
    assert {"413", "422", "503"} <= set(responses)


def test_the_published_example_is_a_payload_the_service_accepts(
    client: TestClient, request_schema: dict[str, Any]
) -> None:
    """Documentation that does not work is worse than none."""
    example = request_schema["example"]

    assert example == EXAMPLE_RECORD
    assert client.post("/api/v1/predict", json=example).status_code == 200
