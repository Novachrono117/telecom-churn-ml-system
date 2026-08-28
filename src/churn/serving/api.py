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
from dataclasses import replace
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn.monitoring.settings import MONITORING_ENDPOINT
from churn.portfolio.explanation import ExplanationError
from churn.preprocessing.exceptions import DataQualityError, FeatureContractError
from churn.serving.artifacts import FREEZE_COMMIT_SHORT, SERVING_VERSION
from churn.serving.errors import (
    CODE_EXPLANATION_UNAVAILABLE,
    CODE_INTERNAL_ERROR,
    CODE_INVALID_FEATURE_PAYLOAD,
    CODE_INVALID_FEATURE_VALUE,
    CODE_INVALID_REQUEST_SCHEMA,
    ServiceNotReadyError,
    ServingRequestError,
)
from churn.serving.monitoring import HEALTH_DISABLED, ServingMonitor, disabled_response
from churn.serving.portfolio import (
    DEMO_ROUTE,
    STATIC_MOUNT,
    PortfolioService,
    default_static_path,
)
from churn.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    ExplanationResponse,
    HealthResponse,
    ModelMetadataResponse,
    MonitoringResponse,
    PortfolioMetadataResponse,
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


def get_portfolio(request: Request) -> PortfolioService:
    """Return the process-wide portfolio service, or refuse the request.

    Only reachable on a process where the demo is enabled, because the routes that
    depend on it are not registered otherwise.
    """
    portfolio = getattr(request.app.state, "portfolio", None)
    if portfolio is None:
        raise ServiceNotReadyError("The portfolio demo is not available on this process.")
    return portfolio


def get_monitor(request: Request) -> ServingMonitor | None:
    """Return the process-wide monitor, or ``None`` when monitoring is off.

    Never raises and never blocks a request: monitoring is observational, so its
    absence is a normal state rather than an error.
    """
    return getattr(request.app.state, "monitor", None)


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

#: The process-wide monitor, or None when monitoring is off. Observational: no
#: endpoint's answer depends on it.
MonitorDependency = Annotated["ServingMonitor | None", Depends(get_monitor)]

#: The process-wide portfolio service. Presentational: no prediction depends on it.
PortfolioDependency = Annotated["PortfolioService", Depends(get_portfolio)]


def _build_lifespan(
    service: ChurnInferenceService | None,
    settings: ServingSettings,
) -> Callable[[FastAPI], Any]:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings
        if service is not None:
            app.state.service = service
        else:
            # Fail-closed: any startup gate that fails raises here, and the
            # process does not come up. Nothing is rebuilt on the way.
            app.state.service = ChurnInferenceService.from_settings(resolved)

        if resolved.monitoring_enabled:
            # Fail-closed too, and for the same reason: a drift number computed
            # against an unverified baseline looks like a signal and is not one.
            monitor = ServingMonitor.from_settings(resolved)
            app.state.monitor = monitor
            # The observer is attached by REPLACING the frozen service, not by
            # mutating it. One shared, immutable instance stays the rule.
            app.state.service = replace(app.state.service, observer=monitor)
            logger.info("Monitoring enabled: reference=%s", monitor.reference_sha256[:16])

        if resolved.portfolio_ui_enabled:
            # Fail-closed as well: a demo that cannot verify its own decomposition,
            # or whose metrics nobody validated, should not come up at all.
            app.state.portfolio = PortfolioService.from_service(app.state.service, resolved)

        logger.info(
            "Serving ready: freeze_commit=%s serving_version=%s fingerprint=%s monitoring=%s",
            FREEZE_COMMIT_SHORT,
            SERVING_VERSION,
            app.state.service.model_fingerprint[:16],
            resolved.monitoring_enabled,
        )
        if resolved.portfolio_ui_enabled:
            logger.info("Portfolio demo published at %s", DEMO_ROUTE)
        yield
        app.state.service = None
        app.state.monitor = None
        app.state.portfolio = None
        logger.info("Serving stopped.")

    return lifespan


def _count_rejection(request: Request, kind: str) -> None:
    """Count a rejected record when monitoring is on. Never affects the response."""
    monitor = getattr(request.app.state, "monitor", None)
    if monitor is not None:
        monitor.record_rejection(kind)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServingRequestError)
    async def _serving_error(_: Request, error: ServingRequestError) -> Response:
        return _error(error.status, error.code, error.message)

    @app.exception_handler(ExplanationError)
    async def _explanation_error(_: Request, error: ExplanationError) -> Response:
        # A decomposition that does not reproduce the model is not an explanation of
        # it, so nothing is returned rather than something plausible. Only this
        # endpoint fails; the prediction path is authoritative and unaffected.
        logger.error("explanation_reconstruction_failure exception_type=%s", type(error).__name__)
        return _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            CODE_EXPLANATION_UNAVAILABLE,
            "The local explanation could not be verified against the served "
            "prediction and was withheld. The prediction endpoints are unaffected.",
        )

    @app.exception_handler(FeatureContractError)
    async def _contract_error(request: Request, error: FeatureContractError) -> Response:
        _count_rejection(request, "feature")
        return _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT, CODE_INVALID_FEATURE_PAYLOAD, str(error)
        )

    @app.exception_handler(DataQualityError)
    async def _quality_error(request: Request, error: DataQualityError) -> Response:
        _count_rejection(request, "feature")
        return _error(status.HTTP_422_UNPROCESSABLE_CONTENT, CODE_INVALID_FEATURE_VALUE, str(error))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, error: RequestValidationError) -> Response:
        _count_rejection(request, "schema")
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
        monitor = getattr(request.app.state, "monitor", None)
        body = HealthResponse(
            status="ready",
            serving_version=SERVING_VERSION,
            model_fingerprint=service.model_fingerprint,
            startup_gates_passed=len(service.artifacts.checks),
            # Reported, never gating. Readiness is a claim about the ARTEFACT — is
            # the frozen model loaded and verified. Drift is a claim about the
            # POPULATION, and a shifted population is not a corrupt model; letting
            # it mark the process NOT READY would pull a healthy instance out of
            # rotation because customers changed.
            monitoring=HEALTH_DISABLED if monitor is None else monitor.health,
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
        monitor: MonitorDependency,
    ) -> PredictionResponse:
        """Score one record through the frozen contract, pipeline and threshold."""
        if monitor is not None:
            monitor.record_request(1)
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
        monitor: MonitorDependency,
    ) -> BatchPredictionResponse:
        """Score a batch. Order is preserved and nothing is persisted."""
        if monitor is not None:
            monitor.record_request(len(payload.records))
        predictions = service.predict_batch([record.to_record() for record in payload.records])
        logger.info("Scored a batch of %d record(s).", len(predictions))
        return BatchPredictionResponse(
            count=len(predictions),
            predictions=[PredictionResponse(**vars(prediction)) for prediction in predictions],
        )


def _register_portfolio_routes(app: FastAPI) -> None:
    """Publish the demo: two read-only endpoints, one page, one static mount.

    Registered only when the demo is enabled, which is what keeps the Phase 11 route
    list byte-identical on a deployment that wants only the API.
    """
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    static_root = (
        app.state.portfolio_static_path
        if getattr(app.state, "portfolio_static_path", None) is not None
        else None
    )

    @app.post(
        f"{API_PREFIX}/explain",
        response_model=ExplanationResponse,
        tags=["explanation"],
        responses=_ERROR_RESPONSES,
        summary="Score one customer and decompose that score exactly",
    )
    async def explain(
        payload: PredictionRequest,
        service: ServiceDependency,
        portfolio: PortfolioDependency,
        monitor: MonitorDependency,
    ) -> ExplanationResponse:
        """Return the same prediction ``/predict`` returns, plus its exact decomposition.

        The predictive fields are copied from the one canonical scoring path, so this
        endpoint cannot report a probability the prediction endpoint would not.

        The decomposition is an algebraic identity, not an approximation: no SHAP, no
        LIME, no sampling, no seed. If it fails to reproduce the served probability,
        this endpoint fails — the prediction endpoints are unaffected and remain
        authoritative.
        """
        if monitor is not None:
            monitor.record_request(1)
        explanation = portfolio.explain(service, payload.to_record())
        return ExplanationResponse(**explanation.as_record())

    @app.get(
        f"{API_PREFIX}/portfolio",
        response_model=PortfolioMetadataResponse,
        tags=["portfolio"],
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: _ERROR_RESPONSES[503]},
        summary="Versioned metadata the demo displays",
    )
    async def portfolio_metadata(portfolio: PortfolioDependency) -> PortfolioMetadataResponse:
        """Already-versioned metadata. Nothing is computed and no dataset is read."""
        return PortfolioMetadataResponse(**portfolio.metadata_response())

    @app.get(
        DEMO_ROUTE,
        response_class=FileResponse,
        include_in_schema=False,
        tags=["portfolio"],
    )
    async def demo(portfolio: PortfolioDependency) -> FileResponse:
        """Serve the demo page from the configured directory. No parameter, no path."""
        return FileResponse(portfolio.static_path / "index.html", media_type="text/html")

    if static_root is not None:
        # Mounted from a directory resolved at startup. StaticFiles refuses to serve
        # anything outside its root, and no route here accepts a filename.
        app.mount(STATIC_MOUNT, StaticFiles(directory=static_root), name="portfolio-static")


def _register_monitoring_route(app: FastAPI) -> None:
    @app.get(
        MONITORING_ENDPOINT,
        response_model=MonitoringResponse,
        tags=["monitoring"],
        summary="Aggregate drift and data-quality state of the current window",
    )
    async def monitoring(monitor: MonitorDependency) -> MonitoringResponse:
        """Return aggregates only — never a payload, a score or an identifier.

        Read-only. There is deliberately **no** HTTP reset: closing a window is an
        administrative operation, this service has no authentication, and an
        unauthenticated endpoint that erases the evidence a drift investigation
        depends on is a worse trade than asking an operator to restart the process
        or call the collector primitive directly.

        ``status`` says whether this window still resembles the reference. It does
        not say the model degraded — that needs labels, and this phase has none.
        """
        if monitor is None:
            return MonitoringResponse(**disabled_response())
        return MonitoringResponse(**monitor.as_response())


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
    resolved = settings or load_settings()
    app = FastAPI(
        title="Churn Prediction API",
        version=SERVING_VERSION,
        description=DESCRIPTION,
        lifespan=_build_lifespan(service, resolved),
    )
    app.state.service = None
    app.state.monitor = None
    app.state.portfolio = None
    app.state.portfolio_static_path = None
    _register_access_log(app)
    _register_exception_handlers(app)
    _register_routes(app)
    if resolved.monitoring_enabled:
        _register_monitoring_route(app)
    if resolved.portfolio_ui_enabled:
        # Resolved here rather than inside the route so the mount is created once,
        # from operator configuration, and never from anything a request carries.
        app.state.portfolio_static_path = (
            resolved.portfolio_static_path or default_static_path()
        ).resolve()
        _register_portfolio_routes(app)
    return app


__all__ = ["API_PREFIX", "create_app", "get_monitor", "get_portfolio", "get_service"]
