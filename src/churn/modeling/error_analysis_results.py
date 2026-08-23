"""Machine-readable record of the Phase 9E post-hoc error analysis.

A **description** of a result that already exists, never a revision of it. The
final estimate is in `reports/experiments/holdout_results.json` and is closed;
this artefact says where the errors that estimate counted actually fell.

Three properties are written into the record itself rather than left to a
reader's charity, because they are what makes a post-hoc analysis legitimate
instead of self-serving:

* the confusion matrix here **reproduces** the Phase 9D one exactly, so the
  description is of the same classification the estimate came from;
* nothing was selected, changed or reconsidered afterwards
  (``selection_after_analysis: false``);
* the slice axes and the guardrails were **fixed before the analysis ran**, so
  the groups reported are not the groups that turned out to look interesting.

No individual is persisted. Row-level outcomes exist only inside the process; the
file contains counts, rates and distribution summaries. The identifier column
never reaches it.

Same deliberate absence as every earlier phase: **no timestamp**, so a rerun on
an unchanged repository reproduces the file byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT
from churn.modeling.error_analysis import (
    ANALYSIS_TYPE,
    AUDIT_AXES,
    INTERACTION_AXES,
    MARGIN_BAND_EDGES,
    MARGIN_BAND_LABELS,
    MARGINAL_ERROR_MARGIN,
    MIN_ACTUAL_NEGATIVES,
    MIN_ACTUAL_POSITIVES,
    MIN_GROUP_SIZE,
    MIN_PREDICTED_POSITIVES,
    OUTCOME_CLASSES,
    SLICE_AXES,
    HighConfidenceErrors,
    ProbabilitySummary,
    SliceMetrics,
)

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase9e-post-hoc-error-analysis"

#: The commit that produced the final estimate this phase describes. Recorded so
#: the ordering — estimate first, description second — is auditable from the
#: artefact alone.
EVALUATION_COMMIT = "8c8e2663a2756a0fbb95188e96d7771bd9816265"
EVALUATION_COMMIT_SHORT = "8c8e266"
EVALUATION_COMMIT_SUBJECT = "feat: evaluate frozen churn model on final holdout"

#: The commit that froze the model and the decision policy, one step earlier.
FREEZE_COMMIT_SHORT = "9d1db49"

#: Artefacts this analysis reads and must not modify.
UPSTREAM_ARTEFACTS: tuple[str, ...] = (
    "reports/split_manifest.json",
    "reports/decision_policy.json",
    "reports/model_freeze_report.md",
    "reports/holdout_evaluation_report.md",
    "reports/experiments/baseline_results.json",
    "reports/experiments/feature_engineering_results.json",
    "reports/experiments/model_comparison_results.json",
    "reports/experiments/hgb_representation_results.json",
    "reports/experiments/tuning_results.json",
    "reports/experiments/calibration_results.json",
    "reports/experiments/threshold_results.json",
    "reports/experiments/holdout_results.json",
)

#: Observations recorded in the Phase 3 EDA, before the evaluation sample was
#: ever opened. Quoted here so the report can contextualise an error rate
#: without inventing a narrative — and so the distinction between "associated
#: with churn" and "where the model errs" stays visible.
EDA_CONTEXT: tuple[dict[str, str], ...] = (
    {
        "axis": "Contract",
        "prior_observation": (
            "month-to-month contracts showed a far higher churn rate than one-year or two-year "
            "contracts"
        ),
        "source": "reports/eda_report.md, reports/figures/eda/04_churn_rate_by_contract.png",
    },
    {
        "axis": "tenure_band",
        "prior_observation": "low tenure was associated with higher churn",
        "source": "reports/eda_report.md, reports/figures/eda/03_churn_rate_by_tenure_band.png",
    },
    {
        "axis": "InternetService",
        "prior_observation": "fiber optic showed a higher churn rate than DSL or no internet",
        "source": "reports/eda_report.md, reports/figures/eda/06_service_churn_rates.png",
    },
    {
        "axis": "PaymentMethod",
        "prior_observation": "electronic check was associated with higher churn",
        "source": "reports/eda_report.md, reports/figures/eda/09_churn_rate_by_payment_method.png",
    },
    {
        "axis": "SeniorCitizen",
        "prior_observation": "a descriptive difference in churn rate was recorded",
        "source": "reports/eda_report.md",
    },
    {
        "axis": "gender",
        "prior_observation": "association with churn was essentially null",
        "source": "reports/eda_report.md, reports/figures/eda/15_association_ranking.png",
    },
)

_PRECISION = 6


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return None if not np.isfinite(number) else round(number, _PRECISION)


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in binary so line endings cannot alter it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ErrorAnalysisResults(BaseModel):
    """The complete Phase 9E record."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    experiment: str
    phase: str
    analysis_type: str
    selection_allowed: bool
    model_change_allowed: bool
    holdout_already_consumed: bool
    post_hoc: bool
    descriptive_only: bool
    selection_after_analysis: bool
    provenance: dict[str, object]
    upstream_artefact_digests: dict[str, str]
    frozen_policy: dict[str, object]
    confusion_reproduction: dict[str, object]
    evaluation_sample: dict[str, object]
    # ``int | float`` rather than ``float``: a count is an integer and rendering
    # it as ``803.0`` in the artefact would be a small lie about its type.
    probability_summaries: dict[str, dict[str, int | float]]
    margin_analysis: dict[str, object]
    high_confidence_errors: dict[str, object]
    error_composition: dict[str, dict[str, list[dict[str, object]]]]
    slice_analyses: dict[str, list[dict[str, object]]]
    demographic_audit: dict[str, object]
    interactions: dict[str, object]
    guardrails: dict[str, object]
    insufficient_cells: list[dict[str, object]]
    prior_eda_observations: list[dict[str, str]]
    methodology: dict[str, object]
    environment: dict[str, str]
    methodological_notes: list[str]


METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "This analysis is POST HOC and DESCRIPTIVE. It runs after the final estimate was produced and "
    "committed, and it describes where the errors that estimate counted fell. It is not part of "
    "the estimate and does not revise it.",
    "NOTHING WAS SELECTED, CHANGED OR RECONSIDERED AS A RESULT OF THIS ANALYSIS. No model, "
    "hyperparameter, feature, preprocessing step, calibration policy or threshold was modified, "
    "and none may be modified on the basis of anything recorded here: the Phase 9D figures "
    "describe the system as it was frozen, and this project has no remaining held-out sample on "
    "which a modified system could be measured.",
    "The Phase 9D result is the DESIGNATED FINAL HELD-OUT EVALUATION under the frozen protocol: "
    "the estimate produced after the model and the decision policy were frozen and committed, on "
    "rows that took no part in any fit, fold, feature decision, calibration gate or threshold "
    "search. It is NOT described as an unbiased estimate. The Phase 3 exploratory analysis ran on "
    "all 7,043 rows, including those that later formed the evaluation sample; the sample was "
    "protected against fitting, tuning and selection from Phase 4 onwards, but that earlier "
    "ANALYST EXPOSURE preceded the protection and can carry an optimistic bias this project "
    "cannot quantify. The protection and the limitation are both real and both stated.",
    "NOTHING WAS FITTED. The pipeline was loaded from the frozen artefact, its model fingerprint "
    "was verified before use, and the file is byte-identical afterwards.",
    "The confusion matrix rederived here reproduces the Phase 9D record EXACTLY. That is an "
    "integrity gate, not a coincidence: an analysis of a different classification than the one "
    "the final estimate came from would describe nothing relevant, so the run aborts on any "
    "disagreement.",
    "The threshold is the single frozen value and is never re-derived, re-optimised or varied. No "
    "alternative threshold was scored, including thresholds that would have reduced false "
    "negatives.",
    "The extreme-score error cut-offs are DIAGNOSTIC ONLY and are explicitly not decision-policy "
    "candidates. They answer whether the model ever assigns an extreme probability and still gets "
    "the case wrong. 'High confidence' in the field names is descriptive shorthand for an extreme "
    "model probability under the frozen score, NOT a validated per-case confidence guarantee: the "
    "probabilities are uncalibrated by the Phase 9A decision. Nothing in this project may use "
    "these cut-offs to move the decision boundary or to reopen the calibration policy.",
    "MARGINAL means BOUNDARY-PROXIMATE under the frozen operating point: the score lies within a "
    "fixed distance of the frozen decision boundary. It does NOT mean the estimated churn "
    "probability is close to 50%. Both boundary-proximate and distant errors remain errors of the "
    "frozen decision; the margin analysis distinguishes proximity to the boundary, not real "
    "errors from incidental ones.",
    "NO ALTERNATIVE THRESHOLD WAS SCORED, SIMULATED OR DESCRIBED QUANTITATIVELY. Boundary "
    "proximity is descriptive only: moving the threshold would change the precision/recall "
    "trade-off, but evaluating such alternatives on this already consumed sample would reopen "
    "decision selection and is outside the scope of this analysis.",
    "The slice axes were FIXED BEFORE the analysis ran, and each carries a justification that "
    "predates the opening of the evaluation sample. They are not the axes that turned out to look "
    "interesting: no sweep over the 19 features was performed and no automatic search over their "
    "combinations was performed.",
    "Only ONE two-way interaction is reported, the pair the Phase 3 EDA already crossed, and it is "
    "secondary. A three-way breakdown would multiply the cells past the point where any of them "
    "supports a rate, and the exercise would become mining.",
    "Every rate is conditioned on the population that could have produced it: false-negative rate "
    "on the actual churners, false-positive rate on the actual non-churners. A raw error count per "
    "category is not used to call a group problematic, because a larger group produces more errors "
    "for reasons unrelated to the model.",
    "Guardrails were fixed before interpretation and applied uniformly. A rate whose own "
    "denominator falls below its minimum is reported as null, never as a number and never as "
    "zero, and categories were NOT merged to clear a threshold. Every insufficient cell is listed "
    "rather than silently omitted.",
    "NO SIGNIFICANCE TEST OF ANY KIND was run: no chi-square, t-test, Mann-Whitney, ANOVA, Fisher "
    "or p-value. With this many descriptive cells examined after the outcome was known, a search "
    "for significant subgroups would find some whether or not anything is there.",
    "The subgroup tables are a DESCRIPTIVE AUDIT. They are not a fairness assessment, they do not "
    "establish bias or discrimination, and a difference between two groups' rates is reported as "
    "an observation about this sample, not as a property of the model or of the population.",
    "Association with churn and error behaviour are DIFFERENT THINGS and are kept separate. A "
    "feature strongly associated with churn does not thereby have a higher error rate, and the "
    "prior EDA observations are quoted as context, never as an explanation of an error rate.",
    "NO CAUSAL CLAIM is made anywhere. The analysis describes where a frozen model's decisions "
    "disagreed with recorded outcomes; it does not establish why.",
    "NO EXPLAINABILITY. No SHAP value, no coefficient interpretation, no permutation importance "
    "and no feature-importance ranking appears here. How the features influence the predictions is "
    "Phase 10's subject; this phase is only about where the decisions were wrong.",
    "NO FINANCIAL QUANTITY. No retention cost, customer lifetime value, lost revenue, campaign ROI "
    "or contact capacity. A false negative is a churner the system did not identify and a false "
    "positive is a flagged customer who stayed; neither is assigned a monetary value.",
    "NO IDENTIFIER IS PERSISTED. Row-level outcomes, probabilities and margins exist only inside "
    "the process. This artefact contains aggregates: counts, guarded rates and distribution "
    "summaries.",
    "The evaluation sample was already consumed by Phase 9D. It cannot provide a fresh "
    "independent estimate for any decision made after it was examined, which is precisely why "
    "analysing it descriptively is permitted and acting on it is not.",
)


def _summary(summary: ProbabilitySummary) -> dict[str, float]:
    return {
        key: value if key == "count" else _round(value) for key, value in summary.as_dict().items()
    }


def _slice_rows(table: Sequence[SliceMetrics]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in table:
        record = entry.as_dict()
        rows.append(
            {
                key: _round(value) if isinstance(value, float) else value
                for key, value in record.items()
            }
        )
    return rows


def _composition_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {key: _round(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in rows
    ]


def build_results(
    policy: object,
    evaluation: object,
    reproduction: Mapping[str, object],
    probability_summaries: Mapping[str, ProbabilitySummary],
    margin_bands: Mapping[str, Mapping[str, int]],
    outcome_margins: Mapping[str, Mapping[str, float]],
    marginal_counts: Mapping[str, object],
    confident: HighConfidenceErrors,
    composition_tables: Mapping[str, Mapping[str, list[dict[str, object]]]],
    slices: Mapping[str, Sequence[SliceMetrics]],
    audit: Mapping[str, Sequence[SliceMetrics]],
    interactions: Mapping[str, Sequence[SliceMetrics]],
    flagged: Sequence[Mapping[str, object]],
    digests: Mapping[str, str],
    loaded_model_fingerprint: str,
    pipeline_digest_before: str,
    pipeline_digest_after: str,
) -> ErrorAnalysisResults:
    """Assemble the Phase 9E record."""
    cells = evaluation.confusion_matrix

    return ErrorAnalysisResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        phase="9E",
        analysis_type=ANALYSIS_TYPE,
        selection_allowed=False,
        model_change_allowed=False,
        holdout_already_consumed=True,
        post_hoc=True,
        descriptive_only=True,
        selection_after_analysis=False,
        provenance={
            "evaluation_commit": EVALUATION_COMMIT,
            "evaluation_commit_short": EVALUATION_COMMIT_SHORT,
            "evaluation_commit_subject": EVALUATION_COMMIT_SUBJECT,
            "evaluation_preceded_this_analysis": True,
            "evaluation_record": "reports/experiments/holdout_results.json",
            "evaluation_record_sha256": digests["reports/experiments/holdout_results.json"],
            "decision_policy": "reports/decision_policy.json",
            "decision_policy_sha256": digests["reports/decision_policy.json"],
            "split_manifest_sha256": digests["reports/split_manifest.json"],
            "freeze_commit_short": FREEZE_COMMIT_SHORT,
            "model_fingerprint_sha256_recorded": policy.artifacts.model_fingerprint_sha256,
            "model_fingerprint_sha256_after_load": loaded_model_fingerprint,
            "model_fingerprint_verified": (
                loaded_model_fingerprint == policy.artifacts.model_fingerprint_sha256
            ),
            "pipeline_sha256_before_analysis": pipeline_digest_before,
            "pipeline_sha256_after_analysis": pipeline_digest_after,
            "pipeline_unchanged_by_analysis": pipeline_digest_before == pipeline_digest_after,
            "refitted_here": False,
        },
        upstream_artefact_digests=dict(digests),
        frozen_policy={
            "source": "reports/decision_policy.json",
            "final_threshold": policy.threshold.final_threshold,
            "comparison": policy.decision_rule.comparison,
            "positive_class_label": policy.decision_rule.positive_class_label,
            "positive_class_column": policy.decision_rule.positive_class_column,
            "calibration_policy": policy.calibration.calibration_policy,
            "threshold_policy": policy.threshold.threshold_policy,
            "estimator": policy.model.estimator,
            "reopened_here": False,
            "alternative_thresholds_scored": 0,
        },
        confusion_reproduction=dict(reproduction),
        evaluation_sample={
            "n_samples": int(cells["total"]),
            "n_positive": int(cells["actual_positives"]),
            "n_negative": int(cells["actual_negatives"]),
            # Derived from the recorded cells rather than read from the Phase 9D
            # record, so this module needs no field of it beyond the confusion
            # matrix — and stays free of the identifier the Phase 5 audit guards.
            "prevalence": _round(int(cells["actual_positives"]) / int(cells["total"])),
            "status": (
                "already consumed as the final evaluation set in Phase 9D. Described here, never "
                "used to choose anything"
            ),
            "identifiers_persisted": False,
        },
        probability_summaries={
            name: _summary(probability_summaries[name]) for name in OUTCOME_CLASSES
        },
        margin_analysis={
            "definition": "margin = probability - final_threshold",
            "sign_meaning": {
                "positive": "flagged side of the frozen cut",
                "negative": "unflagged side of the frozen cut",
            },
            "threshold_used": policy.threshold.final_threshold,
            "threshold_re_derived": False,
            "band_edges": list(MARGIN_BAND_EDGES),
            "band_labels": list(MARGIN_BAND_LABELS),
            "bands_fixed_before_analysis": True,
            "counts_by_band": {band: dict(counts) for band, counts in margin_bands.items()},
            "absolute_margin_by_outcome": {
                name: {
                    key: _round(value) if key != "count" else value
                    for key, value in outcome_margins[name].items()
                }
                for name in OUTCOME_CLASSES
            },
            "marginal_errors": {
                key: _round(value) if isinstance(value, float) else value
                for key, value in marginal_counts.items()
            },
        },
        high_confidence_errors={
            "false_positive_probability_minimum": confident.false_positive_minimum,
            "false_negative_probability_maximum": confident.false_negative_maximum,
            "n_false_positives": confident.n_false_positives,
            "n_high_confidence_false_positives": confident.n_high_confidence_false_positives,
            "share_of_false_positives": _round(confident.share_of_false_positives),
            "n_false_negatives": confident.n_false_negatives,
            "n_high_confidence_false_negatives": confident.n_high_confidence_false_negatives,
            "share_of_false_negatives": _round(confident.share_of_false_negatives),
            "diagnostic_only": True,
            "decision_policy_candidate": False,
            "terminology": (
                "'high confidence' is descriptive shorthand for an EXTREME MODEL PROBABILITY "
                "under the frozen score. It is NOT a validated per-case confidence guarantee: the "
                "probabilities are uncalibrated by the Phase 9A decision, so an extreme score is "
                "an extreme output of the model, not a demonstrated likelihood for the individual "
                "customer. The field names keep this wording for compatibility; the report says "
                "'extreme-score errors'"
            ),
            "note": (
                "these cut-offs answer whether the model ever assigns an extreme probability and "
                "still gets the case wrong. They are not candidate thresholds and must not be "
                "used to move the decision boundary or to reopen the calibration policy"
            ),
        },
        error_composition={
            error_class: {axis: _composition_rows(rows) for axis, rows in axes.items()}
            for error_class, axes in composition_tables.items()
        },
        slice_analyses={axis: _slice_rows(slices[axis]) for axis in slices},
        demographic_audit={
            "status": "descriptive subgroup audit",
            "is_fairness_assessment": False,
            "establishes_bias": False,
            "inferential_tests_run": 0,
            "note": (
                "reported as observations about this sample. A difference between two groups' "
                "rates is not evidence of bias or discrimination, no inferential test was run, "
                "and nothing here may be used to modify the model"
            ),
            "axes": {axis: _slice_rows(audit[axis]) for axis in audit},
        },
        interactions={
            "status": "secondary and descriptive",
            "pairs_examined": [list(pair) for pair in INTERACTION_AXES],
            "automatic_combination_search": False,
            "note": (
                "only the pair the Phase 3 EDA already crossed. Cells are small by construction, "
                "so most rates fail their guardrails and are reported as unavailable"
            ),
            "cells": {name: _slice_rows(interactions[name]) for name in interactions},
        },
        guardrails={
            "fixed_before_interpretation": True,
            "applied_uniformly": True,
            "categories_merged_to_clear_a_threshold": False,
            "minimum_group_size": MIN_GROUP_SIZE,
            "minimum_actual_positives_for_recall_and_fnr": MIN_ACTUAL_POSITIVES,
            "minimum_actual_negatives_for_specificity_and_fpr": MIN_ACTUAL_NEGATIVES,
            "minimum_predicted_positives_for_precision": MIN_PREDICTED_POSITIVES,
            "insufficient_rate_representation": "null, never zero and never a number",
            "marginal_error_absolute_margin": MARGINAL_ERROR_MARGIN,
        },
        insufficient_cells=[dict(entry) for entry in flagged],
        prior_eda_observations=[dict(entry) for entry in EDA_CONTEXT],
        methodology={
            "post_hoc": True,
            "descriptive_only": True,
            "selection_after_analysis": False,
            "fit_calls": 0,
            "models_loaded": 1,
            "alternative_models_evaluated": 0,
            "thresholds_scored": 1,
            "alternative_thresholds_scored": 0,
            "calibrations_performed": 0,
            "hyperparameter_searches": 0,
            "feature_selections": 0,
            "significance_tests_run": 0,
            "slice_axes_fixed_before_analysis": True,
            "slice_axes": list(SLICE_AXES),
            "audit_axes": list(AUDIT_AXES),
            "features_swept": 0,
            "explainability_computed": False,
            "financial_costs_assumed": False,
            "causal_claims": False,
            "identifiers_persisted": False,
            "statement": (
                "The analysis describes the errors of the frozen configuration on the already "
                "consumed evaluation sample. No selection, fit, calibration, tuning, threshold "
                "choice or model change followed it"
            ),
        },
        environment={
            "python": ".".join(str(part) for part in sys.version_info[:2]),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        methodological_notes=list(METHODOLOGICAL_NOTES),
    )


def write_results(results: ErrorAnalysisResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote error analysis results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> ErrorAnalysisResults:
    """Read and validate the Phase 9E record.

    Raises:
        FileNotFoundError: If the analysis has not been run yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Error analysis results not found at {source}. Produce them with "
            "`uv run python scripts/run_error_analysis.py`."
        )
    return ErrorAnalysisResults.model_validate_json(source.read_text(encoding="utf-8"))


def invariant_failures(results: ErrorAnalysisResults) -> list[str]:
    """Return every violated invariant of a loaded record. Read-only.

    What ``--verify`` checks: that the description still matches the estimate it
    describes, that every slice table accounts for the whole sample, that no rate
    was reported without its denominator, and that the protocol flags still say
    nothing was selected.
    """
    failures: list[str] = []
    reproduction = results.confusion_reproduction
    sample = results.evaluation_sample
    total = int(sample["n_samples"])

    if not reproduction["reproduced"]:
        failures.append("the confusion matrix does not reproduce the Phase 9D record")
    if int(reproduction["total"]) != total:
        failures.append("the reproduced confusion total disagrees with the sample size")
    derived = reproduction["derived"]
    if sum(int(derived[name]) for name in OUTCOME_CLASSES) != total:
        failures.append("the four outcome classes do not partition the sample")

    counted = sum(summary["count"] for summary in results.probability_summaries.values())
    if counted != total:
        failures.append(f"probability summaries cover {counted} rows, not {total}")

    banded = sum(
        sum(counts.values()) for counts in results.margin_analysis["counts_by_band"].values()
    )
    if banded != total:
        failures.append(f"margin bands cover {banded} rows, not {total}")

    for axis, rows in results.slice_analyses.items():
        if sum(int(row["n"]) for row in rows) != total:
            failures.append(f"slice {axis} does not cover the whole sample")
        for row in rows:
            expected = (
                int(row["true_positives"])
                + int(row["false_positives"])
                + int(row["false_negatives"])
                + int(row["true_negatives"])
            )
            if expected != int(row["n"]):
                failures.append(f"slice {axis}/{row['group']} cells do not sum to n")
            if not row["sufficient_for_recall"] and row["recall"] is not None:
                failures.append(f"slice {axis}/{row['group']} reports recall without support")
            if not row["sufficient_for_specificity"] and row["specificity"] is not None:
                failures.append(f"slice {axis}/{row['group']} reports specificity without support")
            if not row["sufficient_for_precision"] and row["precision"] is not None:
                failures.append(f"slice {axis}/{row['group']} reports precision without support")

    confident = results.high_confidence_errors
    if confident["n_high_confidence_false_positives"] > confident["n_false_positives"]:
        failures.append("more confident false positives than false positives")
    if confident["n_high_confidence_false_negatives"] > confident["n_false_negatives"]:
        failures.append("more confident false negatives than false negatives")
    if not confident["diagnostic_only"] or confident["decision_policy_candidate"]:
        failures.append("the high-confidence cut-offs are not marked diagnostic-only")

    provenance = results.provenance
    if not provenance["model_fingerprint_verified"]:
        failures.append("the loaded model fingerprint did not match the decision policy")
    if not provenance["pipeline_unchanged_by_analysis"]:
        failures.append("the pipeline file changed during the analysis")
    if provenance["refitted_here"]:
        failures.append("the record claims a refit occurred")
    if provenance["evaluation_commit"] != EVALUATION_COMMIT:
        failures.append("the evaluation commit is not the expected one")

    methodology = results.methodology
    for flag, expected in (
        ("post_hoc", True),
        ("descriptive_only", True),
        ("selection_after_analysis", False),
        ("explainability_computed", False),
        ("financial_costs_assumed", False),
        ("causal_claims", False),
        ("identifiers_persisted", False),
        ("slice_axes_fixed_before_analysis", True),
    ):
        if methodology[flag] is not expected:
            failures.append(f"methodology.{flag} is not {expected}")
    for counter in (
        "fit_calls",
        "alternative_models_evaluated",
        "alternative_thresholds_scored",
        "calibrations_performed",
        "hyperparameter_searches",
        "feature_selections",
        "significance_tests_run",
        "features_swept",
    ):
        if methodology[counter] != 0:
            failures.append(f"methodology.{counter} is not zero")

    if results.selection_allowed or results.selection_after_analysis:
        failures.append("the record permits selection after the analysis")
    if results.model_change_allowed:
        failures.append("the record permits a model change")
    if not results.holdout_already_consumed:
        failures.append("the record does not acknowledge the sample was already consumed")
    if list(results.slice_analyses) != list(SLICE_AXES):
        failures.append("the slice axes are not the declared ones")

    return failures
