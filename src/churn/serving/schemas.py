"""Request and response schemas — the typed edge of the serving boundary.

Three decisions in here are worth stating, because each one could easily have
been made the other way and been wrong.

**Extra fields are forbidden, so ``customerID`` and ``Churn`` are rejected.**
``customerID`` is unique per row and can only be memorised; ``Churn`` is the
target and its presence would be leakage. Neither is silently dropped: dropping
an unexpected field is how a client comes to believe the model reads something
it never sees.

**Categories are typed, not enumerated.** The frozen encoder was fitted with
``handle_unknown="ignore"`` precisely so that a category which did not exist when
the model was trained — a new contract term, a new payment method — can still be
scored. Freezing today's observed values into an ``Enum`` here would revoke that
by the back door: the request would be rejected at the schema, and the design
decision taken in Phase 4 would silently stop applying. So a category is a
string, and a *blank* string is rejected — by the feature contract, one layer
down, which is the authority on what blank means.

**No range constraints are declared on the numeric features.** The training data
happens to span some interval of ``MonthlyCharges``; that is a fact about a
sample, not a definition of a valid input. Writing ``le=118.75`` here would
confuse the training distribution with input validity and would reject a real,
scoreable customer. Whether such a record is *far from the training
distribution* is a monitoring question — Phase 12 — not a schema question.

The one exception to "type only, no semantics" is ``SeniorCitizen``, and it is a
type decision rather than a domain one. The frozen encoder learned it as the
integers ``0``/``1``. A JSON string ``"0"`` would not match the integer ``0``, so
``handle_unknown="ignore"`` would encode it as an all-zero block and the record
would be scored as if the field had never been sent — no error, a quietly
different prediction. Typing the field as an integer is what prevents that.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from churn.preprocessing.contracts import FEATURE_COLUMNS

#: A synthetic record used as the OpenAPI example and by the smoke test.
#:
#: It is invented for documentation. It is not a row of the training pool and not
#: a row of the holdout: no partition is readable from this package, and copying
#: a real customer into public documentation would publish that customer.
EXAMPLE_RECORD: dict[str, Any] = {
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

_CATEGORY_NOTE = (
    "Free-text category. Values unseen during training are accepted and handled by the "
    "frozen encoder's handle_unknown='ignore'; blank, empty or whitespace-only values "
    "are rejected by the feature contract."
)

Category = Annotated[str, Field(description=_CATEGORY_NOTE)]


class PredictionRequest(BaseModel):
    """The 19 raw features of one customer. Nothing else is accepted.

    The identifier and the target are not fields of this model and are rejected
    as extra properties.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"example": EXAMPLE_RECORD},
    )

    tenure: float = Field(description="Months the customer has been with the company.")
    MonthlyCharges: float = Field(description="Amount charged to the customer this month.")
    TotalCharges: float | str | None = Field(
        description=(
            "Amount charged in total. Accepts a number or the raw string the source "
            "system emits. A blank value is admissible only where tenure == 0, which "
            "is a customer who has not completed a billing cycle; a blank at positive "
            "tenure and any unreadable value are rejected. That rule lives in the "
            "frozen preprocessing, not here."
        )
    )

    gender: Category
    SeniorCitizen: int = Field(
        description=(
            "Binary flag, encoded as the integer 0 or 1. Typed as an integer because "
            "the frozen encoder learned it as one; a string would be treated as an "
            "unknown category and silently ignored."
        )
    )
    Partner: Category
    Dependents: Category
    PhoneService: Category
    MultipleLines: Category
    InternetService: Category
    OnlineSecurity: Category
    OnlineBackup: Category
    DeviceProtection: Category
    TechSupport: Category
    StreamingTV: Category
    StreamingMovies: Category
    Contract: Category
    PaperlessBilling: Category
    PaymentMethod: Category

    def to_record(self) -> dict[str, Any]:
        """Return the features as a plain mapping for the inference service."""
        return self.model_dump()


class BatchPredictionRequest(BaseModel):
    """A list of records, scored independently and returned in the same order."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={"example": {"records": [EXAMPLE_RECORD]}},
    )

    records: list[PredictionRequest] = Field(
        min_length=1,
        description=(
            "One entry per customer. The response preserves this order exactly. The "
            "maximum size is an operational limit of the deployment; exceeding it is "
            "refused rather than truncated, and one invalid record fails the whole "
            "batch rather than returning a partial answer. Nothing is persisted. A "
            "record's decision is identical to what the single endpoint would return; "
            "its probability can differ in the final bit, because BLAS sums a matrix "
            "product differently depending on its shape."
        ),
    )


class PredictionResponse(BaseModel):
    """One scored record.

    ``churn_probability`` is the model's score for ``Churn = Yes``, read from the
    ``predict_proba`` column that ``classes_`` identifies. It is **not** a
    calibrated probability: the frozen calibration policy is ``NONE``, decided in
    Phase 9A because neither sigmoid nor isotonic calibration cleared the
    pre-registered eligibility rule. Read it as a ranking score with a
    probability's scale, not as a frequency guarantee.

    ``prediction`` and ``decision`` are the same fact twice: ``1`` is ``churn``,
    ``0`` is ``retained``. Both follow from ``churn_probability >= threshold`` and
    from nothing else — scikit-learn's own ``predict`` and its implicit 0.5 cut
    are not used anywhere in this service.

    Every field is a pure function of the payload and the frozen artefacts. Two
    identical requests produce two byte-identical responses.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    churn_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Model score for Churn = Yes. Not calibrated; see the model metadata.",
    )
    prediction: Literal[0, 1] = Field(description="1 = churn, 0 = retained.")
    decision: Literal["retained", "churn"] = Field(
        description="The prediction in words. Identical information, no extra judgement."
    )
    threshold: float = Field(
        description="The frozen decision threshold this answer was produced with."
    )
    comparison: Literal[">="] = Field(
        description=(
            "The frozen comparison. A record whose probability equals the threshold "
            "exactly is predicted positive."
        )
    )
    calibration_policy: str = Field(
        description="Frozen calibration policy. 'NONE' means the score is the model's raw output."
    )
    model_fingerprint: str = Field(
        description="Authoritative identity of the frozen model that produced this answer."
    )


class BatchPredictionResponse(BaseModel):
    """Scored records, in the order they were sent."""

    model_config = ConfigDict(frozen=True)

    count: int = Field(ge=1, description="Number of records scored. Equals the number sent.")
    predictions: list[PredictionResponse] = Field(
        description="One entry per input record, in input order."
    )


class ModelMetadataResponse(BaseModel):
    """What this process is serving, and under which policy.

    ``freeze_commit`` is the model's version; ``serving_version`` is this
    application's. They move independently, and a release of the serving code
    does not produce a new model.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    estimator: str
    freeze_commit: str = Field(description="Commit that froze the model. The model's version.")
    freeze_commit_short: str
    model_fingerprint: str
    pipeline_sha256: str
    decision_policy_sha256: str
    n_features: int = Field(description="Raw features accepted by the boundary.")
    n_transformed_features: int = Field(description="Columns the classifier actually receives.")
    feature_names: list[str]
    calibration_policy: str
    threshold_policy: str
    threshold: float
    comparison: Literal[">="]
    positive_class_label: int
    positive_class_column: int = Field(description="predict_proba column, resolved from classes_.")
    positive_class_meaning: str
    serving_version: str = Field(
        description="Version of this API. Independent of the model version."
    )
    runtime: dict[str, str] = Field(description="Library versions this process is running.")


class HealthResponse(BaseModel):
    """Liveness or readiness of the process."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    status: Literal["alive", "ready", "not_ready"]
    serving_version: str
    monitoring: str | None = Field(
        default=None,
        description=(
            "Collector health: HEALTHY, DEGRADED or DISABLED. Reported here for "
            "convenience and deliberately NOT part of readiness: a drifting "
            "population is not a corrupt artefact, so drift never makes this "
            "process NOT READY."
        ),
    )
    model_fingerprint: str | None = Field(
        default=None,
        description="Present once the frozen model is loaded and verified.",
    )
    startup_gates_passed: int | None = Field(
        default=None,
        description="Number of startup integrity gates that passed.",
    )


class MonitoringResponse(BaseModel):
    """Aggregate monitoring state of the current window.

    Everything here is a count, a rate, a histogram or a status. There is no
    payload, no feature value, no row-level score and no identifier: the collector
    never retained one.

    ``status`` answers "does this window still resemble the reference population",
    and nothing else. It is **not** a statement about model performance — that would
    need production labels, which this phase does not have, and
    ``performance_degradation.evaluated`` is ``false`` for exactly that reason.

    **Below ``minimum_window_size`` the distribution sections are ``null``.** A
    per-feature breakdown of a window holding one record is that record: a level
    count names their contract, a histogram bin places their tenure, a
    predicted-positive rate of 1.0 is their decision. ``details_suppressed`` is then
    ``true``, and what remains — the window size, the global operational counters and
    the statuses — describes the deployment rather than a customer.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    monitoring_enabled: bool
    status: str = Field(
        description=(
            "OK, WARNING, CRITICAL, or INSUFFICIENT_DATA when the window is too small "
            "for any drift claim. Cutoffs are operational policy, not significance levels."
        )
    )
    collector_health: str = Field(
        description="HEALTHY, DEGRADED or DISABLED. About the collector, never about drift."
    )
    detail: str | None = None
    n_records: int | None = None
    minimum_window_size: int | None = None
    window_has_verdict: bool | None = None
    details_suppressed: bool | None = Field(
        default=None,
        description=(
            "True when the window is below the operational minimum, in which case "
            "feature_drift, prediction_drift and structural_consistency are null: over "
            "a handful of records a per-feature breakdown identifies the records "
            "themselves."
        ),
    )
    collector_failures: int | None = None
    reference_profile_sha256: str | None = Field(
        default=None,
        description="Digest of the reference profile actually loaded at startup.",
    )
    expected_reference_profile_sha256: str | None = Field(
        default=None,
        description=(
            "The digest startup required. Pinned in source, independently of the file "
            "checked; a mismatch prevents the process from coming up."
        ),
    )
    reference_profile_sha256_source: str | None = Field(
        default=None,
        description="Where the expected digest comes from — the trust anchor.",
    )
    reference_population: str | None = None
    reference_n: int | None = None
    section_status: dict[str, str] | None = None
    data_quality: dict[str, int] | None = None
    feature_drift: dict[str, Any] | None = None
    prediction_drift: dict[str, Any] | None = None
    structural_consistency: dict[str, Any] | None = None
    performance_degradation: dict[str, Any] | None = Field(
        default=None,
        description="Always {evaluated: false} in this phase: no production labels exist.",
    )
    interpretation: str | None = None
    calibration: str | None = None
    unseen_cardinality: str | None = Field(
        default=None,
        description=(
            "How to read n_distinct_unseen_observed: exact below the tracking cap, a "
            "lower bound at or above it. Present only when the distributions are."
        ),
    )
    rejection_counting_note: str | None = None
    small_window_note: str | None = Field(
        default=None,
        description=(
            "Present when the window is below the operational minimum, explaining "
            "which distributions are withheld and why."
        ),
    )


class ContributionItem(BaseModel):
    """One raw feature's term in the linear predictor, for one request.

    ``contribution_log_odds`` is a term of a sum, not a causal effect and not a
    counterfactual. The one-hot parameterisation is redundant with the intercept, so
    an individual level's coefficient is a property of this fitted parameterisation
    rather than a transferable quantity.
    """

    model_config = ConfigDict(frozen=True)

    feature: str
    kind: Literal["numeric", "categorical"]
    value_display: str = Field(description="The value as the model received it.")
    active_level: str | None = Field(
        default=None,
        description="The encoder level that was active, or null for a numeric feature.",
    )
    is_unseen_level: bool = Field(
        description=(
            "True when the category was not in the training contract. The frozen "
            "encoder ignores it, so the contribution is exactly zero."
        )
    )
    contribution_log_odds: float
    direction: Literal["increases", "decreases", "neutral"] = Field(
        description="Whether this term pushed the model score higher or lower."
    )


class GroupedContributionItem(BaseModel):
    """Structurally coupled features, presented as one row.

    A customer without internet produces seven correct terms that are one fact about
    the product's encoding, not seven independent reasons. The value is the **exact
    sum** of the members' contributions, so grouping changes the layout and never the
    arithmetic.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    value_display: str
    members: list[str]
    n_members: int
    contribution_log_odds: float
    direction: Literal["increases", "decreases", "neutral"]
    description: str


class ReconstructionReport(BaseModel):
    """Evidence that the decomposition reproduces the prediction it explains."""

    model_config = ConfigDict(frozen=True)

    logit_error: float
    probability_error: float
    tolerance: float
    within_tolerance: bool = Field(
        description="Always true in a returned explanation; a failure raises instead."
    )


class ExplanationResponse(BaseModel):
    """One scored customer, decomposed exactly.

    The predictive fields are **the same values** ``/api/v1/predict`` returns for the
    same payload — copied from the one canonical scoring path, not recomputed — so the
    two endpoints cannot disagree about a probability or a decision.

    ``model_logit`` is the decomposition's own sum, which makes
    ``intercept + sum(contributions) == model_logit`` exact rather than approximate.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    churn_probability: float = Field(ge=0.0, le=1.0)
    prediction: Literal[0, 1]
    decision: Literal["retained", "churn"] = Field(
        description="The same label PredictionResponse carries for this payload."
    )
    threshold: float
    comparison: Literal[">="]
    calibration_policy: str
    explanation_method: Literal["EXACT_LOGISTIC_DECOMPOSITION"]
    intercept: float
    model_logit: float
    threshold_logit: float = Field(
        description="The threshold expressed in log-odds, so the margin is readable."
    )
    margin_log_odds: float
    contributions: list[ContributionItem] = Field(
        description="All 19 raw features. Ordering is local to this request."
    )
    grouped_contributions: list[GroupedContributionItem]
    ungrouped_features: list[str]
    reconstruction: ReconstructionReport
    causal_note: str
    calibration_note: str


class PortfolioMetadataResponse(BaseModel):
    """Versioned metadata the demo displays. Nothing here is computed per request.

    The evaluation numbers are copied from Phase 9D's committed artefact by an offline
    build step. The holdout is not reopened by this process and no metric is
    recomputed — both facts are asserted in ``provenance``.

    They are also the *pinned* numbers: the summary's digest was compared at startup
    against a constant in source, and a mismatch would have stopped the process rather
    than produced this response. ``integrity`` reports that comparison.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    schema_version: int
    built_in_phase: str
    integrity: dict[str, Any] = Field(
        description=(
            "Which summary was served and what it was checked against: the digest "
            "computed at startup, the expectation pinned in source, and the name of "
            "the constant holding it. Startup fails on a mismatch, so a client that "
            "receives this block is reading numbers that matched their pin."
        )
    )
    policy: dict[str, Any]
    evaluation: dict[str, Any] = Field(
        description="The single frozen holdout evaluation, with its analyst-exposure caveat."
    )
    known_levels: dict[str, list[str]] = Field(
        description=(
            "Levels the frozen encoder learned, per categorical feature. A UI "
            "convenience only: the API still accepts any non-blank category."
        )
    )
    numeric_features: list[str]
    categorical_features: list[str]
    provenance: dict[str, Any]


class ErrorDetail(BaseModel):
    """A stable code and a message safe to show a client."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable summary. Never a traceback or a file path.")
    details: list[str] | None = Field(
        default=None,
        description="Field-level validation notes, when the error is a schema violation.",
    )


class ErrorResponse(BaseModel):
    """The single error envelope every failing endpoint answers with."""

    model_config = ConfigDict(frozen=True)

    error: ErrorDetail


#: Feature names the request schema must declare, in contract order. Imported by
#: the tests so that a change to the contract breaks the schema loudly.
REQUEST_FEATURE_NAMES: tuple[str, ...] = FEATURE_COLUMNS


__all__ = [
    "EXAMPLE_RECORD",
    "REQUEST_FEATURE_NAMES",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ModelMetadataResponse",
    "ContributionItem",
    "ExplanationResponse",
    "GroupedContributionItem",
    "MonitoringResponse",
    "PortfolioMetadataResponse",
    "PredictionRequest",
    "PredictionResponse",
    "ReconstructionReport",
]
