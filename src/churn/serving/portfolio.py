"""Wiring the portfolio demo into the serving process without giving it authority.

This module is the seam between a boundary that decides things and a page that shows
them. It owns three responsibilities and refuses several more.

**It scores through the existing path, never beside it.**
:meth:`PortfolioService.explain` calls the same
:func:`~churn.serving.inference.canonical_features_and_probability` the prediction
endpoints call, then hands the resulting row and probability to the Phase 13
decomposition. There is one inference implementation in this process, so
``/api/v1/explain`` cannot return a probability that ``/api/v1/predict`` would not
have returned for the same payload — they are the same float, not two floats that
agree.

**It refuses metadata nobody pinned.** The metrics the page presents as measured
come from one file, and a setting names which file. So the summary is not merely
parsed and structurally validated at startup — it is hashed, and the digest is
compared against
:data:`~churn.portfolio.metadata.EXPECTED_PORTFOLIO_METADATA_SHA256`, an expectation
recorded in source rather than in the file being checked. A mismatch stops the
process: it does not fall back to empty metadata and it does not show the numbers
unverified.

**It reads its learned parameters once, at startup.** ``extract_terms`` and
``feature_groups`` walk the fitted pipeline's internals and verify that the 46
transformed columns partition into the 19 raw features. Doing that per request would
repeat work whose answer cannot change while the process lives; doing it at startup
also means a pipeline whose layout is not the expected one stops the process instead
of failing on a customer's request.

**It serves static files from a directory the operator names, and from nowhere
else.** The path comes from settings, is resolved once, and is handed to Starlette's
``StaticFiles``, which refuses traversal outside its root. No request parameter ever
reaches a filesystem call, because no route takes a filename.

What it refuses: it never affects readiness, never touches a threshold, never writes
anything, and never persists a payload. Enabling the demo adds routes to this process
and changes nothing about what the model computes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from churn.config import PROJECT_ROOT
from churn.modeling.interpretation import (
    FeatureGroup,
    LinearTerms,
    extract_terms,
    feature_groups,
)
from churn.portfolio.explanation import LocalExplanation, explain_prepared_row
from churn.portfolio.metadata import (
    EXPECTED_PORTFOLIO_METADATA_SHA256,
    PORTFOLIO_METADATA_TRUST_ANCHOR,
    PortfolioMetadata,
    PortfolioMetadataError,
    default_metadata_path,
    load_portfolio_metadata,
    metadata_digest,
    raise_on_failure,
    verify_portfolio_metadata,
)
from churn.serving.errors import ServingStartupError
from churn.serving.service import ChurnInferenceService
from churn.serving.settings import ServingSettings

logger = logging.getLogger(__name__)

#: Where the demo is published. Not under ``/api/v1``: it is a page, not an API.
DEMO_ROUTE = "/demo"

#: Where the demo's static assets are mounted.
STATIC_MOUNT = "/demo/static"

#: Files the demo is made of. Enumerated so a missing one fails at startup rather
#: than as a blank page.
REQUIRED_ASSETS: tuple[str, ...] = ("index.html", "css/app.css", "js/app.js")

#: Default location of those files, relative to the repository root.
PORTFOLIO_STATIC_RELATIVE_PATH = "portfolio"


class PortfolioStartupError(ServingStartupError):
    """The demo's assets or metadata are absent, malformed, or not the expected ones."""


def default_static_path(root: Path | None = None) -> Path:
    """Return the repository's portfolio asset directory."""
    return (root or PROJECT_ROOT) / PORTFOLIO_STATIC_RELATIVE_PATH


@dataclass(frozen=True)
class PortfolioService:
    """Explains served predictions and publishes the demo's versioned metadata.

    Frozen and shared by every request. It holds the model's learned parameters and
    the column partition, both read once from the pipeline the serving boundary
    already verified — it never loads an artefact of its own.
    """

    terms: LinearTerms
    groups: tuple[FeatureGroup, ...]
    metadata: PortfolioMetadata
    metadata_sha256: str
    static_path: Path

    @classmethod
    def from_service(
        cls,
        service: ChurnInferenceService,
        settings: ServingSettings,
    ) -> PortfolioService:
        """Build the demo's service around an already verified inference service.

        Raises:
            PortfolioStartupError: If the pipeline cannot be decomposed, if the
                versioned metadata is absent, fails an invariant, or is not the summary
                whose digest is pinned in source, or if a required static asset is
                missing. Fail-closed: a demo that renders half of itself, or that shows
                metrics nobody verified, is worse than one that refuses to start.
        """
        try:
            terms = extract_terms(service.artifacts.pipeline)
            groups = feature_groups(terms)
        except Exception as error:  # noqa: BLE001 - any decomposition failure is a failed gate
            raise PortfolioStartupError(
                f"The frozen pipeline cannot be decomposed for local explanation "
                f"({type(error).__name__}). The demo does not start: an explanation "
                "endpoint that cannot verify its own arithmetic should not exist."
            ) from error

        metadata_path = settings.portfolio_metadata_path or default_metadata_path()
        try:
            metadata = load_portfolio_metadata(metadata_path)
        except FileNotFoundError as error:
            raise PortfolioStartupError(
                "The portfolio metadata is absent, so the demo has no verified numbers "
                "to display. It does not start and it does not compute them: metrics "
                "invented at serving time would not be the frozen evaluation."
            ) from error
        except Exception as error:  # noqa: BLE001 - any parse failure is a failed gate
            raise PortfolioStartupError(
                f"The portfolio metadata does not parse ({type(error).__name__})."
            ) from error

        # The digest is compared against a constant in source, not against anything
        # in the file. Structure alone cannot separate the summary from a file merely
        # shaped like it: every invariant below would still hold on one whose average
        # precision had been edited.
        actual_sha256 = metadata_digest(metadata)
        try:
            raise_on_failure(
                verify_portfolio_metadata(metadata, EXPECTED_PORTFOLIO_METADATA_SHA256)
            )
        except PortfolioMetadataError as error:
            raise PortfolioStartupError(
                f"{error}\n"
                f"  expected_sha256: {EXPECTED_PORTFOLIO_METADATA_SHA256}\n"
                f"  actual_sha256:   {actual_sha256}\n"
                f"  source_of_expected_sha256: {PORTFOLIO_METADATA_TRUST_ANCHOR}\n"
                f"  loaded_from: {metadata_path}\n"
                "The demo does not start on metadata nobody pinned, and it does not "
                "degrade to empty or unverified metrics."
            ) from error

        static_path = (settings.portfolio_static_path or default_static_path()).resolve()
        missing = [name for name in REQUIRED_ASSETS if not (static_path / name).is_file()]
        if missing:
            raise PortfolioStartupError(
                f"The portfolio assets are incomplete at {static_path}: missing {missing}."
            )

        logger.info(
            "Portfolio demo enabled: %d raw feature group(s), assets=%s; "
            "metadata integrity verified (actual=%s expected=%s source=%s)",
            len(groups),
            static_path.name,
            actual_sha256[:16],
            EXPECTED_PORTFOLIO_METADATA_SHA256[:16],
            PORTFOLIO_METADATA_TRUST_ANCHOR,
        )
        return cls(
            terms=terms,
            groups=groups,
            metadata=metadata,
            metadata_sha256=actual_sha256,
            static_path=static_path,
        )

    # -- explanation --------------------------------------------------------

    def explain(
        self,
        service: ChurnInferenceService,
        record: dict[str, object],
    ) -> LocalExplanation:
        """Score one record through the canonical path, then decompose that row.

        The two steps are deliberately in this order and in this process. Scoring
        first means the probability being explained is the probability the API
        serves; decomposing the *returned* matrix means the explanation describes the
        row the model actually saw, not a second preparation of the same payload.

        It goes through :meth:`~churn.serving.service.ChurnInferenceService.score_with_features`
        rather than reaching for the pipeline itself, so an explained record is scored
        and **observed** exactly as a predicted one is: monitoring sees the same
        traffic whichever endpoint asked for it.

        Raises:
            FeatureContractError: If the schema does not satisfy the frozen contract.
            DataQualityError: If a value is blank or unreadable where no rule allows.
            ExplanationError: If the decomposition does not reproduce the served
                probability. The prediction endpoints are unaffected.
        """
        artifacts = service.artifacts
        prediction, features = service.score_with_features(record)
        return explain_prepared_row(
            pipeline=artifacts.pipeline,
            terms=self.terms,
            groups=self.groups,
            features=features,
            served_probability=prediction.churn_probability,
            prediction=prediction.prediction,
            decision=prediction.decision,
            threshold=prediction.threshold,
            comparison=prediction.comparison,
            calibration_policy=prediction.calibration_policy,
        )

    # -- metadata -----------------------------------------------------------

    def known_levels(self) -> dict[str, list[str]]:
        """Return the levels the frozen encoder learned, per categorical feature.

        Read from the fitted encoder rather than from a dataset or a hand-written
        list, so the dropdowns the page offers are exactly the training contract.

        These are a **convenience layer for the UI only.** The API still accepts any
        non-blank category, because the frozen encoder was fitted with
        ``handle_unknown="ignore"`` precisely so a new contract term or payment method
        can still be scored. Nothing here narrows that.
        """
        return {group.feature: list(group.levels) for group in self.groups if group.levels}

    def metadata_response(self) -> dict[str, object]:
        """Return the versioned metadata payload the demo consumes.

        Aggregates and policy only. No dataset row, no filesystem path, no
        environment value, and nothing computed at request time.

        It carries the integrity block startup produced — the digest that was
        verified, the expectation it was compared against, and where that expectation
        came from — so a client can confirm the numbers it is being shown are the
        pinned ones rather than take the word of the process serving them. The
        constant is a digest, not a secret, and the file it names is committed.
        """
        metadata = self.metadata
        evaluation = metadata.evaluation
        return {
            "schema_version": metadata.schema_version,
            "built_in_phase": metadata.built_in_phase,
            "integrity": {
                "verified": True,
                "actual_sha256": self.metadata_sha256,
                "expected_sha256": EXPECTED_PORTFOLIO_METADATA_SHA256,
                "source_of_expected_sha256": PORTFOLIO_METADATA_TRUST_ANCHOR,
                "digest_scope": "canonical serialisation of the summary",
            },
            "policy": metadata.policy.model_dump(),
            "evaluation": evaluation.model_dump(),
            "known_levels": self.known_levels(),
            "numeric_features": list(self.terms.numeric_features),
            "categorical_features": list(self.terms.categorical_features),
            "provenance": {
                "freeze_commit": metadata.provenance.freeze_commit,
                "model_fingerprint_sha256": metadata.provenance.model_fingerprint_sha256,
                "pipeline_sha256": metadata.provenance.pipeline_sha256,
                "holdout_results_sha256": metadata.provenance.holdout_results_sha256,
                "holdout_reopened": metadata.provenance.holdout_reopened,
                "metrics_recomputed": metadata.provenance.metrics_recomputed,
            },
        }


__all__ = [
    "DEMO_ROUTE",
    "PORTFOLIO_STATIC_RELATIVE_PATH",
    "REQUIRED_ASSETS",
    "STATIC_MOUNT",
    "PortfolioService",
    "PortfolioStartupError",
    "default_static_path",
]
