"""HTTP layer. A thin, replaceable skin over :class:`ChurnInferenceService`.

Everything predictive lives one module down. What is here is transport: routes,
status codes, an error envelope, and a startup that refuses to hand out a service
it could not verify.

**The app is built by a factory, never at import.** ``create_app()`` returns a new
application with its own state, so a test can start one against a temporary,
deliberately corrupted artefact directory without poisoning the next test, and a
startup *failure* is itself testable. A module-level ``app = FastAPI()`` that
loaded a model as a side effect of import would make both impossible.

**The model is loaded once, in the lifespan.** Not per request: a request handler
that unpickles an 8 KB pipeline would pay for it on every call and, worse, could
observe a *different* file mid-flight. If the lifespan cannot build a verified
service, it raises and the process does not come up — a server that never starts
is a better failure than a server that starts and answers 503 forever, because
the first one is visible to whoever deployed it.

**Nothing about a request is logged beyond its shape.** Method, path, status,
latency, batch size, model fingerprint. No payload, no feature value, no
probability attached to a person. ``customerID`` is not accepted by the schema in
the first place, so there is no identifier in the process to leak.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn.preprocessing.exceptions import DataQualityError, FeatureContractError
from churn.serving.artifacts import FREEZE_COMMIT_SHORT, SERVING_VERSION
from churn.serving.errors import (
    CODE_INTERNAL_ERROR,
    CODE_INVALID_FEATURE_PAYLOAD,
    CODE_INVALID_FEATURE_VALUE,
    CODE_INVALID_REQUEST_SCHEMA,
    ServiceNotReadyError,
    ServingRequestError,
)
from churn.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from churn.serving.service import ChurnInferenceService
from churn.serving.settings import ServingSettings, load_settings

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"

#: How many field-level notes a validation error reports before truncating.
_MAX_VALIDATION_DETAILS = 20

DESCRIPTION = f"""
Serving boundary for the frozen telecom churn model.

**What it does.** Validates a payload of the 19 raw features, runs it through the
frozen feature contract (`prepare_features`), applies the persisted scikit-learn
pipeline, reads P(Churn = 1) from the `predict_proba` column that `classes_`
identifies, and compares it with the frozen threshold using `>=`.

**What `churn_probability` is.** The model's raw score for `Churn = Yes`. The
frozen calibration policy is `NONE` — decided in Phase 9A, because neither
sigmoid nor isotonic calibration cleared the pre-registered eligibility rule — so
the score is **not** a calibrated probability. It orders customers by risk on a
0-1 scale; it does not promise that a score of 0.30 means 30 % of such customers
churn.

**What `decision` is.** The frozen policy, and only that:
`churn_probability >= threshold`. The threshold was selected in Phase 9B by
F1 maximisation on out-of-fold probabilities of the training pool. It is part of
the model contract, not a deployment setting: no environment variable can move
it.

**What this service does not do.** It does not train, tune, calibrate, or read
any dataset. It does not persist requests, predictions, or identifiers.
`customerID` and `Churn` are rejected, not ignored. Drift and performance
monitoring are out of scope here.

Model version (freeze commit `{FREEZE_COMMIT_SHORT}`) and serving version
(`{SERVING_VERSION}`) are independent: releasing this API does not produce a new
model.
"""

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "Batch larger than the deployment's operational limit.",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "The payload is not a valid feature record.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "The frozen model is not loaded on this process.",
    },
}


def _error(status_code: int, code: str, message: str, details: list[str] | None = None) -> Response:
    """Build the single error envelope. Never carries a traceback or a path."""
    payload: dict[str, Any] = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return JSONResponse(status_code=status_code, content={"error": payload})


def get_service(request: Request) -> ChurnInferenceService:
    """Return the process-wide inference service.

    A FastAPI dependency, so a test can override it with a double — while the
    integration tests deliberately do not, and exercise the real frozen pipeline.

    Raises:
        ServiceNotReadyError: If the lifespan has not produced a verified service.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise ServiceNotReadyError(
            "The frozen model is not loaded on this process, so no prediction can be "
            "made. The service does not load or rebuild it on demand."
        )
    return service


#: The injected inference service, declared with ``Annotated`` rather than as a
#: ``Depends(...)`` argument default. Both work; only this one keeps the call out
#: of a mutable default position, and it is the idiom current FastAPI documents.
ServiceDependency = Annotated[ChurnInferenceService, Depends(get_service)]


def _build_lifespan(
    service: ChurnInferenceService | None,
    settings: ServingSettings | None,
) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            app.state.service = service
        else:
            # Fail-closed: any startup gate that fails raises here, and the
            # process does not come up. Nothing is rebuilt on the way.
            app.state.service = ChurnInferenceService.from_settings(settings or load_settings())
        logger.info(
            "Serving ready: freeze_commit=%s serving_version=%s fingerprint=%s",
            FREEZE_COMMIT_SHORT,
            SERVING_VERSION,
            app.state.service.model_fingerprint[:16],
        )
        yield
        app.state.service = None
        logger.info("Serving stopped.")

    return lifespan


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServingRequestError)
    async def _serving_error(_: Request, error: ServingRequestError) -> Response:
        return _error(error.status, error.code, error.message)

    @app.exception_handler(FeatureContractError)
    async def _contract_error(_: Request, error: FeatureContractError) -> Response:
        return _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT, CODE_INVALID_FEATURE_PAYLOAD, str(error)
        )

    @app.exception_handler(DataQualityError)
    async def _quality_error(_: Request, error: DataQualityError) -> Response:
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, CODE_INVALID_FEATURE_VALUE, str(error))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, error: RequestValidationError) -> Response:
        # Only `loc` and `msg` are echoed. Pydantic also reports the offending
        # `input`, which would put submitted feature values into the response and
        # from there into any client-side log.
        details = [
            f"{'.'.join(str(part) for part in item.get('loc', ()))}: {item.get('msg', '')}"
            for item in error.errors()[:_MAX_VALIDATION_DETAILS]
        ]
        return _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            CODE_INVALID_REQUEST_SCHEMA,
            "The request body is not a valid feature payload. The 19 contracted features "
            "are required and no other field is accepted.",
            details,
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, error: Exception) -> Response:
        # Logged without the payload; answered without the traceback.
        logger.exception("Unhandled error while serving a request: %s", type(error).__name__)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            CODE_INTERNAL_ERROR,
            "The service failed to complete the request.",
        )


def _register_routes(app: FastAPI) -> None:
    @app.get(
        "/health/live",
        response_model=HealthResponse,
        tags=["health"],
        summary="Liveness: is the HTTP process running?",
    )
    async def live() -> HealthResponse:
        """Answer without touching the model. Liveness is about the process only."""
        return HealthResponse(status="alive", serving_version=SERVING_VERSION)

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        tags=["health"],
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: _ERROR_RESPONSES[503]},
        summary="Readiness: is a verified frozen model loaded?",
    )
    async def ready(request: Request) -> Response:
        """READY only if every startup gate passed and the service exists."""
        service = getattr(request.app.state, "service", None)
        if service is None:
            return _error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                ServiceNotReadyError.code,
                "The frozen model is not loaded on this process.",
            )
        body = HealthResponse(
            status="ready",
            serving_version=SERVING_VERSION,
            model_fingerprint=service.model_fingerprint,
            startup_gates_passed=len(service.artifacts.checks),
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump())

    @app.get(
        f"{API_PREFIX}/model",
        response_model=ModelMetadataResponse,
        tags=["model"],
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: _ERROR_RESPONSES[503]},
        summary="Identity and policy of the frozen model being served",
    )
    async def model_metadata(service: ServiceDependency) -> ModelMetadataResponse:
        """Non-sensitive, already-versioned metadata. No path, no secret, no dataset."""
        return ModelMetadataResponse(**vars(service.identity()))

    @app.post(
        f"{API_PREFIX}/predict",
        response_model=PredictionResponse,
        tags=["prediction"],
        responses=_ERROR_RESPONSES,
        summary="Score one customer",
    )
    async def predict(
        payload: PredictionRequest,
        service: ServiceDependency,
    ) -> PredictionResponse:
        """Score one record through the frozen contract, pipeline and threshold."""
        prediction = service.predict_one(payload.to_record())
        return PredictionResponse(**vars(prediction))

    @app.post(
        f"{API_PREFIX}/predict/batch",
        response_model=BatchPredictionResponse,
        tags=["prediction"],
        responses=_ERROR_RESPONSES,
        summary="Score several customers, answering in input order",
    )
    async def predict_batch(
        payload: BatchPredictionRequest,
        service: ServiceDependency,
    ) -> BatchPredictionResponse:
        """Score a batch. Order is preserved and nothing is persisted."""
        predictions = service.predict_batch([record.to_record() for record in payload.records])
        logger.info("Scored a batch of %d record(s).", len(predictions))
        return BatchPredictionResponse(
            count=len(predictions),
            predictions=[PredictionResponse(**vars(prediction)) for prediction in predictions],
        )


def _register_access_log(app: FastAPI) -> None:
    @app.middleware("http")
    async def access_log(request: Request, call_next: Callable[[Request], Any]) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        # Shape only. No payload, no feature value, no probability, no identifier.
        logger.info(
            "%s %s -> %d in %.1f ms",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000.0,
        )
        return response


def create_app(
    *,
    service: ChurnInferenceService | None = None,
    settings: ServingSettings | None = None,
) -> FastAPI:
    """Build the application.

    Args:
        service: A ready inference service to serve. When given, the lifespan
            installs it instead of loading the artefacts — the injection point
            the unit tests use. When ``None``, the frozen artefacts are located,
            verified and loaded during startup, and a failed gate prevents the
            process from coming up.
        settings: Trusted process configuration used when ``service`` is ``None``.

    Returns:
        A new :class:`fastapi.FastAPI` application with its own state.
    """
    app = FastAPI(
        title="Churn Prediction API",
        version=SERVING_VERSION,
        description=DESCRIPTION,
        lifespan=_build_lifespan(service, settings),
    )
    app.state.service = None
    _register_access_log(app)
    _register_exception_handlers(app)
    _register_routes(app)
    return app


__all__ = ["API_PREFIX", "create_app", "get_service"]
