"""The versioned summary the demo is allowed to display, built once and offline.

The demo shows the model's final held-out numbers. Those numbers exist already, in
``reports/experiments/holdout_results.json``, and this module's whole purpose is to
make sure they reach a browser **without the serving process ever going near the
evaluation that produced them**.

Two rules follow from that, and both are structural rather than advisory.

**The holdout is never reopened.** Nothing here scores anything, loads a partition or
recomputes a metric. :func:`build_metadata` reads an already-committed JSON artefact
and copies numbers out of it; the runtime reads only this file. A metric that appears
on the screen is a metric Phase 9D measured once, under the frozen protocol, and the
count of holdout evaluations stays at one.

**The runtime does not read Phase 9D's artefact per request.** It reads this small,
validated, versioned summary instead. That is not a performance argument — it is a
blast-radius one: the serving process should not hold a file handle to the
evaluation record, because a process that can read it is a process that could one day
be made to recompute against it.

**The caveat travels with the numbers.** The analyst-exposure limitation has been
carried since Phase 3, and a portfolio page is exactly where it would be convenient
to drop. It is a field of this artefact, so it cannot be displayed without it.

Deterministic: no timestamp, no hostname, no run id, so ``--verify`` compares bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
METADATA_NAME = "phase13-portfolio-metadata"

#: Where the runtime reads its metadata from.
PORTFOLIO_METADATA_RELATIVE_PATH = "reports/portfolio/portfolio_metadata.json"

#: **The trust anchor for the portfolio metadata.**
#:
#: Everything the product presents as measured — the held-out metrics, their
#: intervals, the confusion matrix, the analyst-exposure caveat — arrives from one
#: file, and ``CHURN_SERVING_PORTFOLIO_METADATA_PATH`` lets an operator say which
#: one. Structural validation alone does not close that: a file can satisfy every
#: invariant in :func:`verify_portfolio_metadata` — one evaluation, accuracy kept out
#: of the headline, a caveat containing the words "optimistic bias" — while carrying
#: an average precision nobody measured. Those checks answer "is this shaped like the
#: summary"; they cannot answer "is this the summary".
#:
#: So the expectation is recorded **independently of the file being checked**: pinned
#: here, in source, in the same repository and under the same review as the code that
#: enforces it — exactly as
#: :data:`churn.monitoring.settings.EXPECTED_REFERENCE_PROFILE_SHA256` is pinned for
#: the Phase 12 reference profile. Hashing a file and logging the result answers "what
#: did I load"; comparing it with this constant answers "did I load the right thing".
#:
#: It is enforced at startup by
#: :meth:`churn.serving.portfolio.PortfolioService.from_service`, which passes it as
#: ``expected_digest``. A mismatch fails the ``metadata_digest_matches`` check, raises
#: ``PortfolioStartupError``, and the process does not come up — so the demo can never
#: display a metric nobody pinned. It does not degrade to empty metadata and it does
#: not show the numbers unverified: a portfolio page whose figures cannot be traced is
#: worse than one that is briefly unavailable.
#:
#: Rebuilding the summary is therefore a deliberate two-line change: the artefact and
#: this constant move together, in one commit, or ``--verify`` and startup both fail.
EXPECTED_PORTFOLIO_METADATA_SHA256 = (
    "e535e67d8038624f2e09ccdeb35ba9265ac0c5a53adf5355ca805e2228dd16cc"
)

#: Where the expected digest comes from, named in the metadata response so an operator
#: reading it never has to guess what the comparison was against.
PORTFOLIO_METADATA_TRUST_ANCHOR = "churn.portfolio.metadata.EXPECTED_PORTFOLIO_METADATA_SHA256"

#: The two artefacts this summary is copied out of. Read offline, never at runtime.
HOLDOUT_RESULTS_RELATIVE_PATH = "reports/experiments/holdout_results.json"
DECISION_POLICY_RELATIVE_PATH = "reports/decision_policy.json"

#: Carried since Phase 3 and repeated here because a portfolio page is precisely
#: where an inconvenient limitation goes missing.
ANALYST_EXPOSURE_CAVEAT = (
    "The final evaluation used the frozen holdout protocol. Earlier exploratory "
    "analysis had already exposed the analyst to all rows, so the estimate may carry "
    "optimistic bias that cannot be quantified here."
)

#: What the headline metrics are, and what they are not.
METRIC_READING_NOTE = (
    "Average precision and ROC-AUC are ranking quality, not accuracy and not "
    "certainty. Precision, recall and F1 describe the single frozen operating point. "
    "Accuracy is reported for completeness only: on a population where 73.5 % of "
    "customers do not churn, a model that predicts nobody churns would score close to "
    "it while finding no one."
)

#: Headline metrics, in the order the product shows them. Accuracy is deliberately
#: not in this list — see :data:`AUXILIARY_METRICS`.
HEADLINE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("average_precision", "Average precision", "discrimination_primary"),
    ("roc_auc", "ROC-AUC", "discrimination_primary"),
    ("recall", "Recall", "operating_point"),
    ("precision", "Precision", "operating_point"),
    ("f1", "F1", "operating_point"),
)

#: Reported, never highlighted.
AUXILIARY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("accuracy", "Accuracy", "auxiliary"),
    ("specificity", "Specificity", "operating_point"),
    ("balanced_accuracy", "Balanced accuracy", "operating_point"),
)


class MetricEntry(BaseModel):
    """One evaluated metric, with its interval when Phase 9D computed one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    label: str
    value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence_level: float | None = None


class EvaluationSummary(BaseModel):
    """Phase 9D's single held-out evaluation, copied verbatim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    evaluation_type: str
    evaluations_performed: int = Field(ge=1)
    n_samples: int = Field(ge=1)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)
    prevalence: float = Field(ge=0.0, le=1.0)
    headline: list[MetricEntry]
    auxiliary: list[MetricEntry]
    confusion_matrix: dict[str, int]
    no_skill_average_precision: float
    no_skill_roc_auc: float
    majority_class_accuracy: float
    bootstrap_method: str
    bootstrap_replications: int = Field(ge=0)
    caveat: str
    metric_reading_note: str


class PolicySummary(BaseModel):
    """The frozen decision policy, as the demo displays it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator: str
    calibration_policy: str
    threshold_policy: str
    threshold: float
    comparison: str
    n_features: int = Field(ge=1)
    n_transformed_features: int = Field(ge=1)
    positive_class_meaning: str


class PortfolioProvenance(BaseModel):
    """Which committed bytes this summary was copied out of."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    holdout_results_path: str
    holdout_results_sha256: str = Field(min_length=64, max_length=64)
    decision_policy_path: str
    decision_policy_sha256: str = Field(min_length=64, max_length=64)
    pipeline_sha256: str = Field(min_length=64, max_length=64)
    model_fingerprint_sha256: str = Field(min_length=64, max_length=64)
    freeze_commit: str
    holdout_reopened: bool
    metrics_recomputed: bool


class PortfolioMetadata(BaseModel):
    """Everything the demo may display that is not live model output."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    schema_version: int = Field(ge=1)
    metadata: str
    built_in_phase: str
    provenance: PortfolioProvenance
    policy: PolicySummary
    evaluation: EvaluationSummary


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(
    holdout: dict[str, Any],
    key: str,
    label: str,
    section: str,
) -> MetricEntry:
    """Copy one metric and its interval out of the Phase 9D record."""
    value = holdout["metrics"][section][key]
    interval = holdout.get("confidence_intervals", {}).get(key)
    if interval is None:
        return MetricEntry(key=key, label=label, value=float(value))
    return MetricEntry(
        key=key,
        label=label,
        value=float(value),
        ci_lower=float(interval["ci_lower"]),
        ci_upper=float(interval["ci_upper"]),
        confidence_level=float(interval["confidence_level"]),
    )


def build_metadata(root: Path | None = None) -> PortfolioMetadata:
    """Read the committed artefacts and produce the demo's versioned summary.

    Offline only. Nothing is scored, no partition is loaded and no metric is
    recomputed: every number is copied from ``holdout_results.json``, whose digest is
    recorded so a reader can confirm which evaluation produced it.

    Raises:
        FileNotFoundError: If either upstream artefact is absent.
        KeyError: If one of them does not carry a field this summary needs.
    """
    base = root or PROJECT_ROOT
    holdout_path = base / HOLDOUT_RESULTS_RELATIVE_PATH
    policy_path = base / DECISION_POLICY_RELATIVE_PATH
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    frozen = holdout["frozen_model_provenance"]
    matrix = holdout["confusion_matrix"]
    baselines = holdout["metrics"]["reference_baselines"]

    return PortfolioMetadata(
        schema_version=SCHEMA_VERSION,
        metadata=METADATA_NAME,
        built_in_phase="13",
        provenance=PortfolioProvenance(
            holdout_results_path=HOLDOUT_RESULTS_RELATIVE_PATH,
            holdout_results_sha256=_digest(holdout_path),
            decision_policy_path=DECISION_POLICY_RELATIVE_PATH,
            decision_policy_sha256=_digest(policy_path),
            pipeline_sha256=frozen["pipeline_sha256_recorded"],
            model_fingerprint_sha256=frozen["model_fingerprint_sha256_recorded"],
            freeze_commit=frozen["freeze_commit"],
            holdout_reopened=False,
            metrics_recomputed=False,
        ),
        policy=PolicySummary(
            estimator=policy["model"]["estimator"],
            calibration_policy=policy["calibration"]["calibration_policy"],
            threshold_policy=policy["threshold"]["threshold_policy"],
            threshold=float(policy["threshold"]["final_threshold"]),
            comparison=policy["decision_rule"]["comparison"],
            n_features=int(policy["feature_contract"]["n_features"]),
            n_transformed_features=int(policy["feature_contract"]["n_transformed_features"]),
            positive_class_meaning="Churn = Yes",
        ),
        evaluation=EvaluationSummary(
            source=HOLDOUT_RESULTS_RELATIVE_PATH,
            evaluation_type=holdout["evaluation_type"],
            evaluations_performed=int(holdout["evaluations_performed"]),
            n_samples=int(holdout["holdout"]["n_samples"]),
            n_positive=int(holdout["holdout"]["n_positive"]),
            n_negative=int(holdout["holdout"]["n_negative"]),
            prevalence=float(holdout["holdout"]["prevalence"]),
            headline=[
                _metric(holdout, key, label, section) for key, label, section in HEADLINE_METRICS
            ],
            auxiliary=[
                _metric(holdout, key, label, section) for key, label, section in AUXILIARY_METRICS
            ],
            confusion_matrix={
                "true_negatives": int(matrix["true_negatives"]),
                "false_positives": int(matrix["false_positives"]),
                "false_negatives": int(matrix["false_negatives"]),
                "true_positives": int(matrix["true_positives"]),
            },
            no_skill_average_precision=float(baselines["no_skill_average_precision"]),
            no_skill_roc_auc=float(baselines["no_skill_roc_auc"]),
            majority_class_accuracy=float(baselines["majority_class_accuracy"]),
            bootstrap_method=holdout["bootstrap"]["method"],
            bootstrap_replications=int(holdout["bootstrap"]["n_bootstrap"]),
            caveat=ANALYST_EXPOSURE_CAVEAT,
            metric_reading_note=METRIC_READING_NOTE,
        ),
    )


def canonical_json(metadata: PortfolioMetadata) -> str:
    """Return the one serialisation this project writes and hashes."""
    return json.dumps(metadata.model_dump(), indent=2, ensure_ascii=False) + "\n"


def metadata_digest(metadata: PortfolioMetadata) -> str:
    """Return the SHA-256 of a summary's canonical serialisation.

    Computed from the **value**, not from whatever bytes happen to be on disk, so a
    summary built in memory and one read back from a file hash identically. The model
    forbids extra fields, so the parsed value determines the content completely: any
    change to any displayed number, interval or caveat moves this digest, while a
    reformatting that changes no field does not. Byte identity of the committed file
    is a separate and stricter guarantee, enforced offline by
    ``scripts/build_portfolio_metadata.py --verify``.

    Same construction as :func:`churn.monitoring.reference.profile_digest`, for the
    same reason: one way to hash an artefact in this repository.
    """
    return hashlib.sha256(canonical_json(metadata).encode("utf-8")).hexdigest()


def default_metadata_path(root: Path | None = None) -> Path:
    """Return the repository's portfolio-metadata location."""
    return (root or PROJECT_ROOT) / PORTFOLIO_METADATA_RELATIVE_PATH


def write_portfolio_metadata(metadata: PortfolioMetadata, path: Path | None = None) -> Path:
    """Write the summary with LF endings and a trailing newline."""
    destination = path or default_metadata_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical_json(metadata), encoding="utf-8", newline="\n")
    logger.info("Wrote portfolio metadata: %s", destination)
    return destination


def load_portfolio_metadata(path: Path | None = None) -> PortfolioMetadata:
    """Read and validate the versioned summary.

    Raises:
        FileNotFoundError: If it has not been built yet.
        pydantic.ValidationError: If the file is not a portfolio metadata artefact.
    """
    source = path or default_metadata_path()
    if not source.is_file():
        raise FileNotFoundError(
            f"Portfolio metadata not found at {source}. Build it with "
            "`uv run python scripts/build_portfolio_metadata.py`."
        )
    return PortfolioMetadata.model_validate_json(source.read_text(encoding="utf-8"))


class PortfolioMetadataError(RuntimeError):
    """The portfolio metadata is absent, malformed, or not the one expected."""


def verify_portfolio_metadata(
    metadata: PortfolioMetadata,
    expected_digest: str | None = None,
) -> list[tuple[str, bool, str]]:
    """Return ``(name, passed, detail)`` for every invariant the summary must satisfy.

    Returned rather than raised so a caller can report all of them at once.

    Args:
        metadata: The summary to check.
        expected_digest: When given, the digest this summary must have. This is the
            only check here that distinguishes *the* summary from one merely shaped
            like it; the others describe structure, and structure is exactly what a
            tampered file keeps. See
            :data:`EXPECTED_PORTFOLIO_METADATA_SHA256`.
    """
    provenance = metadata.provenance
    evaluation = metadata.evaluation
    matrix = evaluation.confusion_matrix
    headline = {entry.key for entry in evaluation.headline}

    checks: list[tuple[str, bool, str]] = [
        (
            "metadata_is_the_phase13_summary",
            metadata.metadata == METADATA_NAME and metadata.schema_version == SCHEMA_VERSION,
            f"{metadata.metadata} v{metadata.schema_version}",
        ),
        ("holdout_not_reopened", provenance.holdout_reopened is False, "holdout_reopened=false"),
        (
            "metrics_not_recomputed",
            provenance.metrics_recomputed is False,
            "copied from the Phase 9D record",
        ),
        (
            "one_holdout_evaluation",
            evaluation.evaluations_performed == 1,
            f"{evaluation.evaluations_performed} evaluation(s)",
        ),
        (
            "accuracy_is_not_headline",
            "accuracy" not in headline and "accuracy" in {e.key for e in evaluation.auxiliary},
            "accuracy reported as auxiliary",
        ),
        (
            "ranking_metrics_are_headline",
            {"average_precision", "roc_auc"} <= headline,
            "average precision and ROC-AUC lead",
        ),
        (
            "confusion_matrix_sums_to_n",
            sum(matrix.values()) == evaluation.n_samples,
            f"{sum(matrix.values())} cells over {evaluation.n_samples} rows",
        ),
        (
            "class_counts_agree",
            evaluation.n_positive + evaluation.n_negative == evaluation.n_samples,
            f"{evaluation.n_positive} + {evaluation.n_negative}",
        ),
        (
            "analyst_exposure_caveat_present",
            "optimistic bias" in evaluation.caveat,
            "carried since Phase 3",
        ),
        (
            "calibration_is_recorded_as_none",
            metadata.policy.calibration_policy == "NONE",
            metadata.policy.calibration_policy,
        ),
        (
            "threshold_is_the_frozen_one",
            metadata.policy.comparison == ">=" and 0.0 < metadata.policy.threshold < 1.0,
            f"{metadata.policy.comparison} {metadata.policy.threshold!r}",
        ),
        (
            "every_headline_metric_has_an_interval",
            all(entry.ci_lower is not None for entry in evaluation.headline),
            f"{len(evaluation.headline)} metric(s)",
        ),
    ]

    if expected_digest is not None:
        actual = metadata_digest(metadata)
        checks.append(
            (
                "metadata_digest_matches",
                actual == expected_digest,
                f"actual={actual[:16]}… expected={expected_digest[:16]}…",
            )
        )
    return checks


def raise_on_failure(checks: list[tuple[str, bool, str]]) -> list[tuple[str, bool, str]]:
    """Return ``checks`` unchanged, or raise listing every one that failed."""
    failed = [(name, detail) for name, passed, detail in checks if not passed]
    if failed:
        raise PortfolioMetadataError(
            "The portfolio metadata did not pass its checks:\n  "
            + "\n  ".join(f"{name}: {detail}" for name, detail in failed)
        )
    return checks


__all__ = [
    "ANALYST_EXPOSURE_CAVEAT",
    "AUXILIARY_METRICS",
    "EXPECTED_PORTFOLIO_METADATA_SHA256",
    "HEADLINE_METRICS",
    "METRIC_READING_NOTE",
    "PORTFOLIO_METADATA_RELATIVE_PATH",
    "PORTFOLIO_METADATA_TRUST_ANCHOR",
    "SCHEMA_VERSION",
    "EvaluationSummary",
    "MetricEntry",
    "PolicySummary",
    "PortfolioMetadata",
    "PortfolioMetadataError",
    "PortfolioProvenance",
    "build_metadata",
    "canonical_json",
    "default_metadata_path",
    "load_portfolio_metadata",
    "metadata_digest",
    "raise_on_failure",
    "verify_portfolio_metadata",
    "write_portfolio_metadata",
]
