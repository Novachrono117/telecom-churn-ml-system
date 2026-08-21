"""Machine-readable record of the Phase 9D final holdout evaluation.

This is the authoritative record of the one result this project exists to
produce, and the first artefact in the repository that contains a test-set
number. Everything about it is written for a reader who arrives later and has to
decide whether to believe it, so it carries three things beyond the metrics:

* **the provenance of what was evaluated** — the freeze commit, both model
  fingerprints, the decision-policy digest and the partition fingerprints, so the
  object measured is provably the object frozen *before* the holdout was opened;
* **the grouping of the metrics** — discrimination, operating point, auxiliary —
  because a reader who cannot tell which number was allowed to decide something
  cannot audit the protocol;
* **the explicit statement that nothing was selected afterwards**
  (``selection_after_holdout: false``), which is the property that makes the rest
  of the file meaningful.

The holdout is **no longer untouched**: ``holdout_touched`` is ``true`` from this
phase onwards. The Phase 9C freeze record keeps ``holdout_touched: false``, and
that stays accurate — it is a statement about the freeze, which happened first.

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

from churn.config import PROJECT_ROOT, get_config
from churn.modeling.holdout import (
    AUXILIARY_METRICS,
    BOOTSTRAP_METHOD,
    CONFUSION_NAMES,
    DISCRIMINATION_METRICS,
    EVALUATION_TYPE,
    HOLDOUT_PURPOSE,
    OPERATING_POINT_METRICS,
    PROBABILITY_DIAGNOSTICS,
    BootstrapInterval,
    GeneralizationCheck,
    HoldoutMetrics,
)

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase9d-final-holdout-evaluation"

#: The commit that froze the model and the decision policy, **before** the
#: holdout was opened. Recorded so the ordering is auditable from the artefact
#: alone rather than only from the git history.
FREEZE_COMMIT = "9d1db4962769093a617f1db2db66354536acffb3"
FREEZE_COMMIT_SHORT = "9d1db49"
FREEZE_COMMIT_SUBJECT = "feat: freeze final churn model and decision policy"

#: Artefacts this evaluation is anchored to. Their digests go into the record.
UPSTREAM_ARTEFACTS: tuple[str, ...] = (
    "reports/split_manifest.json",
    "reports/decision_policy.json",
    "reports/model_freeze_report.md",
    "reports/experiments/baseline_results.json",
    "reports/experiments/feature_engineering_results.json",
    "reports/experiments/model_comparison_results.json",
    "reports/experiments/hgb_representation_results.json",
    "reports/experiments/tuning_results.json",
    "reports/experiments/calibration_results.json",
    "reports/experiments/threshold_results.json",
)

_PRECISION = 6


def _round(value: float) -> float:
    return round(float(value), _PRECISION)


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in binary so line endings cannot alter it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HoldoutMismatchError(RuntimeError):
    """The loaded holdout is not the partition the frozen manifest describes."""


def verify_holdout_partition(
    holdout: pd.DataFrame,
    manifest: object,
    id_column: str,
    fingerprint: object,
) -> dict[str, object]:
    """Check the opened holdout against the frozen split manifest.

    The holdout is regenerated from the raw file rather than stored, so before a
    single metric is computed it has to be shown to be the *same* partition the
    manifest froze. Two independent checks: the row count, and a SHA-256 over the
    sorted identifiers, which depends on the set of rows and not on their order.

    Args:
        holdout: The loaded holdout partition.
        manifest: The loaded split manifest.
        id_column: Name of the identifier column.
        fingerprint: The identifier-hashing function.

    Returns:
        A record of what was checked.

    Raises:
        HoldoutMismatchError: On any disagreement.
    """
    observed_rows = int(len(holdout))
    observed_digest = fingerprint(holdout[id_column])

    problems: list[str] = []
    if observed_rows != manifest.n_rows_holdout:
        problems.append(f"{observed_rows} rows against {manifest.n_rows_holdout} in the manifest")
    if observed_digest != manifest.holdout_ids_sha256:
        problems.append(
            f"identifier digest {observed_digest[:16]}… against "
            f"{manifest.holdout_ids_sha256[:16]}… in the manifest"
        )

    if problems:
        raise HoldoutMismatchError(
            "The opened holdout is not the frozen partition, so no result computed on it would "
            "be the result this protocol promised:\n  " + "\n  ".join(problems)
        )

    logger.info("Holdout partition verified against the frozen manifest: %d rows.", observed_rows)
    return {
        "n_rows_expected": int(manifest.n_rows_holdout),
        "n_rows_observed": observed_rows,
        "holdout_ids_sha256_expected": manifest.holdout_ids_sha256,
        "holdout_ids_sha256_observed": observed_digest,
        "matches_frozen_manifest": True,
        "regenerated_not_stored": True,
        "note": (
            "the partition is a pure function of the raw bytes plus configs/base.toml, so it is "
            "regenerated rather than stored; the identifier digest depends on the set of rows, "
            "not on their order"
        ),
    }


class ConfidenceIntervalRecord(BaseModel):
    """A percentile bootstrap interval. Not a hypothesis test."""

    model_config = ConfigDict(frozen=True)

    metric: str
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: str
    n_bootstrap: int = Field(ge=1)
    n_valid_replications: int = Field(ge=1)
    n_degenerate_replications: int = Field(ge=0)


class GeneralizationCheckRecord(BaseModel):
    """A descriptive holdout-versus-development difference."""

    model_config = ConfigDict(frozen=True)

    metric: str
    holdout: float
    development: float
    development_source: str
    difference: float
    directly_comparable: bool
    caveat: str


class HoldoutResults(BaseModel):
    """The complete Phase 9D record."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    experiment: str
    evaluation_type: str
    holdout_touched: bool
    holdout_purpose: str
    holdout_opened_after_model_freeze: bool
    selection_allowed: bool
    selection_after_holdout: bool
    evaluations_performed: int = Field(ge=1)
    frozen_model_provenance: dict[str, object]
    upstream_artefact_digests: dict[str, str]
    holdout: dict[str, object]
    policy: dict[str, object]
    metrics: dict[str, dict[str, float]]
    confusion_matrix: dict[str, object]
    confidence_intervals: dict[str, ConfidenceIntervalRecord]
    bootstrap: dict[str, object]
    comparison_with_development: dict[str, object]
    methodology: dict[str, object]
    random_seed: int = Field(ge=0)
    scikit_learn_version: str
    environment: dict[str, str]
    methodological_notes: list[str]


METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "This is the FINAL evaluation. Exactly ONE frozen model-and-decision-policy configuration was "
    "evaluated on the holdout, after that configuration had been frozen and committed, and the "
    "holdout was used for measurement only. This is a statement about how many configurations "
    "were evaluated, NOT a claim that the file was physically read once: deterministic "
    "verification and the automated test suite re-run the identical frozen evaluation, which "
    "reproduces byte-identical artefacts and cannot select anything.",
    "NO SELECTION OF ANY KIND FOLLOWED THE OPENING OF THE HOLDOUT. No model, no hyperparameter, "
    "no feature, no calibrator, no threshold and no preprocessing step was chosen, changed or "
    "reconsidered on the basis of anything in this artefact.",
    "NOTHING WAS FITTED IN THIS PHASE. The pipeline was loaded from the frozen .joblib, its model "
    "fingerprint was verified against the decision policy before use, and the same file is "
    "byte-identical after the evaluation.",
    "Exactly one model, one policy and one threshold were evaluated. No alternative estimator was "
    "tried 'for comparison', no second threshold was scored, and no configuration was searched.",
    "The threshold is the single value frozen in Phase 9B and is held constant everywhere, "
    "including inside every bootstrap replication. Re-selecting it per replication would measure "
    "a procedure rather than this operating point.",
    "pipeline.predict() was deliberately not used. It applies scikit-learn's internal 0.5, not the "
    "frozen threshold, so it would have measured a different operating point than the one frozen.",
    "The positive-class probability column is resolved from classes_ rather than assumed, so the "
    "reported metrics provably describe P(Churn = Yes) and not its complement.",
    "Metrics are grouped by what they may claim. Average Precision and ROC-AUC are the PRIMARY "
    "results and are threshold-independent. The operating-point metrics describe what the frozen "
    "decision rule does. Accuracy is AUXILIARY: it is dominated by the majority class, so a large "
    "value is reachable with no skill at all, and it collapses false positives and false "
    "negatives into one number that cannot show which way the decision rule leans. The "
    "majority-class baseline is recorded next to it so the comparison can be made without "
    "recomputing anything; the comparison is reported and decides nothing.",
    "Brier score and log loss are reported as probabilistic DIAGNOSTICS. They were never selection "
    "criteria; the calibration policy was decided in Phase 9A under cross-validation and is NONE. "
    "Nothing in this artefact reopens that decision.",
    "The confidence intervals are PERCENTILE BOOTSTRAP intervals over the holdout observations "
    "with a fixed, recorded seed. They estimate how far each number would move if the holdout had "
    "been a different sample of the same size. They are not significance tests, they contain no "
    "null hypothesis, and they select nothing.",
    "No model was refitted inside the bootstrap. The existing probabilities are re-indexed, so "
    "the intervals describe the uncertainty of the EVALUATION, not of the training.",
    "A bootstrap replication in which a metric is mathematically undefined is counted as "
    "degenerate and excluded from that metric's percentiles rather than filled with a substituted "
    "value, which would drag the interval.",
    "The comparison with development estimates is a GENERALIZATION CHECK, not a gate. No cutoff "
    "exists that turns a difference into a pass or a fail, and none was introduced after seeing "
    "the numbers.",
    "For the threshold-dependent metrics the comparison is NOT like-for-like and is marked as "
    "such. Phase 9B measured a nested PROCEDURE in which each outer fold received a threshold "
    "chosen inside its own training fold; this phase applies one fixed threshold to one sample. "
    "The two answer different questions and a difference between them is not evidence of decay.",
    "No financial quantity appears anywhere. The confusion matrix is not converted into money: "
    "this dataset carries no cost of losing a customer and no cost of a retention contact, so any "
    "such number would be an assumption wearing the clothes of a result.",
    "No causal claim is made. The model describes association between a customer's recorded "
    "attributes and whether that customer had churned; it does not establish that changing an "
    "attribute would change the outcome.",
    "No subgroup battery was run. A slice analysis improvised after opening the holdout invites "
    "opportunistic narratives; fairness and monitoring slices belong to the later phases that can "
    "pre-specify them.",
    "The holdout is now CONSUMED. It has served its single purpose and can no longer provide an "
    "unbiased estimate for any decision taken from here on.",
)


def _interval(interval: BootstrapInterval) -> ConfidenceIntervalRecord:
    return ConfidenceIntervalRecord(
        metric=interval.metric,
        estimate=_round(interval.estimate),
        ci_lower=_round(interval.ci_lower),
        ci_upper=_round(interval.ci_upper),
        confidence_level=interval.confidence_level,
        method=interval.method,
        n_bootstrap=interval.n_bootstrap,
        n_valid_replications=interval.n_valid,
        n_degenerate_replications=interval.n_degenerate,
    )


def _comparison(check: GeneralizationCheck) -> GeneralizationCheckRecord:
    return GeneralizationCheckRecord(
        metric=check.metric,
        holdout=_round(check.holdout),
        development=_round(check.development),
        development_source=check.development_source,
        difference=_round(check.difference),
        directly_comparable=check.comparable,
        caveat=check.caveat,
    )


def build_results(
    metrics: HoldoutMetrics,
    intervals: Mapping[str, BootstrapInterval],
    comparisons: Sequence[GeneralizationCheck],
    policy: object,
    manifest: object,
    partition: Mapping[str, object],
    digests: Mapping[str, str],
    loaded_model_fingerprint: str,
    pipeline_digest_before: str,
    pipeline_digest_after: str,
    bootstrap_seed: int,
) -> HoldoutResults:
    """Assemble the Phase 9D record."""
    config = get_config()
    cells = metrics.confusion

    return HoldoutResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        evaluation_type=EVALUATION_TYPE,
        holdout_touched=True,
        holdout_purpose=HOLDOUT_PURPOSE,
        holdout_opened_after_model_freeze=True,
        selection_allowed=False,
        selection_after_holdout=False,
        evaluations_performed=1,
        frozen_model_provenance={
            "freeze_commit": FREEZE_COMMIT,
            "freeze_commit_short": FREEZE_COMMIT_SHORT,
            "freeze_commit_subject": FREEZE_COMMIT_SUBJECT,
            "freeze_preceded_holdout_opening": True,
            "decision_policy_artefact": "reports/decision_policy.json",
            "decision_policy_sha256": digests["reports/decision_policy.json"],
            "pipeline_artefact": policy.artifacts.pipeline_path,
            "pipeline_sha256_recorded": policy.artifacts.pipeline_sha256,
            "pipeline_sha256_before_evaluation": pipeline_digest_before,
            "pipeline_sha256_after_evaluation": pipeline_digest_after,
            "pipeline_unchanged_by_evaluation": pipeline_digest_before == pipeline_digest_after,
            "model_fingerprint_sha256_recorded": policy.artifacts.model_fingerprint_sha256,
            "model_fingerprint_sha256_after_load": loaded_model_fingerprint,
            "model_fingerprint_verified": (
                loaded_model_fingerprint == policy.artifacts.model_fingerprint_sha256
            ),
            "prepare_features_source_sha256": (
                policy.inference_contract.prepare_features_source_sha256
            ),
            "inference_source_digests": dict(policy.code_provenance.source_digests),
            "configuration_digests": dict(policy.code_provenance.configuration_digests),
            "split_manifest_sha256": digests["reports/split_manifest.json"],
            "training_ids_sha256": manifest.training_ids_sha256,
            "holdout_ids_sha256": manifest.holdout_ids_sha256,
            "raw_sha256": manifest.raw_sha256,
            "refitted_here": False,
        },
        upstream_artefact_digests=dict(digests),
        holdout={
            "loader": "churn.preprocessing.splitting.load_holdout",
            "n_samples": cells.total,
            "n_positive": cells.actual_positives,
            "n_negative": cells.actual_negatives,
            "prevalence": _round(metrics.actual_positive_rate),
            "positive_label_meaning": "Churn = Yes",
            "verification": dict(partition),
            "status": (
                "OPENED for the final evaluation in this phase. It is no longer untouched and can "
                "no longer provide an unbiased estimate for any decision taken from here on"
            ),
        },
        policy={
            "source": "reports/decision_policy.json",
            "estimator": policy.model.estimator,
            "hyperparameters": dict(policy.model.hyperparameters),
            "n_features": policy.feature_contract.n_features,
            "engineered_features": list(policy.feature_contract.engineered),
            "calibration_policy": policy.calibration.calibration_policy,
            "threshold_policy": policy.threshold.threshold_policy,
            "final_threshold": policy.threshold.final_threshold,
            "comparison": policy.decision_rule.comparison,
            "positive_class_label": policy.decision_rule.positive_class_label,
            "positive_class_column": policy.decision_rule.positive_class_column,
            "decision_rule": policy.decision_rule.expression,
            "feature_entry_point": policy.feature_contract.entry_point,
            "reopened_here": False,
        },
        metrics={
            "discrimination_primary": {
                name: _round(metrics.value(name)) for name in DISCRIMINATION_METRICS
            },
            "operating_point": {
                name: _round(metrics.value(name)) for name in OPERATING_POINT_METRICS
            },
            "auxiliary": {name: _round(metrics.value(name)) for name in AUXILIARY_METRICS},
            "probability_diagnostics": {
                name: _round(metrics.value(name)) for name in PROBABILITY_DIAGNOSTICS
            },
            # The trivial references the three headline metrics must be read
            # against. All derived from quantities already recorded above, so a
            # reader can make the comparison without recomputing anything — and
            # none of them is a result or a criterion.
            "reference_baselines": {
                "majority_class_accuracy": _round(cells.actual_negatives / cells.total),
                "no_skill_average_precision": _round(metrics.actual_positive_rate),
                "no_skill_roc_auc": 0.5,
            },
        },
        confusion_matrix={
            **cells.as_dict(),
            "total": cells.total,
            "orientation": (
                "rows are the true class, columns the predicted class: "
                "[[TN, FP], [FN, TP]] with labels pinned to [0, 1]"
            ),
            "actual_positives": cells.actual_positives,
            "actual_negatives": cells.actual_negatives,
            "predicted_positives": cells.predicted_positives,
            "actual_positive_rate": _round(metrics.actual_positive_rate),
            "predicted_positive_rate": _round(metrics.predicted_positive_rate),
            "cells_sum_to_n_samples": cells.total
            == cells.true_negatives
            + cells.false_positives
            + cells.false_negatives
            + cells.true_positives,
            "rate_note": (
                "actual_positive_rate is the share of customers who had churned; "
                "predicted_positive_rate is the share the frozen rule flags. They are different "
                "quantities and a gap between them is a property of the operating point, not an "
                "error: a rule tuned to recall necessarily flags more customers than churn"
            ),
        },
        confidence_intervals={name: _interval(interval) for name, interval in intervals.items()},
        bootstrap={
            "method": BOOTSTRAP_METHOD,
            "resampling_unit": "holdout observation, with replacement",
            "n_bootstrap": next(iter(intervals.values())).n_bootstrap if intervals else 0,
            "confidence_level": next(iter(intervals.values())).confidence_level
            if intervals
            else 0.0,
            "seed": int(bootstrap_seed),
            "percentiles": ["2.5", "97.5"],
            "model_refitted_per_replication": False,
            "threshold_reselected_per_replication": False,
            "point_estimate_source": (
                "the metric on the real holdout, never the mean of the replications"
            ),
            "degenerate_replication_policy": (
                "a replication in which a metric is mathematically undefined is excluded from "
                "that metric's percentiles and counted, never filled with a substituted value"
            ),
            "purpose": "uncertainty quantification only; the bootstrap selects nothing",
            "not_a_significance_test": True,
        },
        comparison_with_development={
            "purpose": "generalization check, descriptive only",
            "selection": False,
            "gate": None,
            "cutoff": None,
            "checks": [_comparison(check).model_dump() for check in comparisons],
            "note": (
                "Average Precision and ROC-AUC are directly comparable: both phases computed them "
                "on probabilities, with no threshold involved. The threshold-dependent metrics are "
                "NOT directly comparable, because Phase 9B measured a nested procedure in which "
                "each outer fold used a threshold chosen inside its own training fold, while this "
                "phase applies one fixed threshold to one sample"
            ),
        },
        methodology={
            # The distinction that matters. `evaluated_configurations` is the
            # scientific claim; the number of times the code physically ran is a
            # different quantity and is not being claimed to be one.
            "evaluated_configurations": 1,
            "physical_executions": (
                "may exceed one. Deterministic verification and the automated test suite re-run "
                "the identical frozen evaluation, which reproduces byte-identical artefacts. Those "
                "re-runs are not additional attempts at selection: there is nothing to select "
                "between, and no decision varies across them"
            ),
            "fit_calls_after_holdout_opened": 0,
            "models_evaluated": 1,
            "thresholds_evaluated": 1,
            "calibrations_performed": 0,
            "hyperparameter_searches": 0,
            "feature_selections": 0,
            "subgroup_slices_evaluated": 0,
            "holdout_evaluations": 1,
            "predict_used": False,
            "predict_proba_used": True,
            "financial_costs_assumed": False,
            "causal_claims": False,
            "sequence": [
                "verify the split manifest",
                "verify the Phase 9C freeze (29 integrity checks) BEFORE opening the holdout",
                "load the frozen pipeline from disk and re-verify its model fingerprint",
                "open the holdout and verify it against the frozen manifest",
                "prepare_features on the holdout",
                "predict_proba, positive column resolved from classes_",
                "apply the frozen threshold with >=",
                "compute metrics, bootstrap the uncertainty, write the record",
            ],
            "statement": (
                "No selection, fit, calibration, tuning or threshold choice occurred after the "
                "holdout was opened"
            ),
        },
        random_seed=config.seed,
        scikit_learn_version=sklearn.__version__,
        environment={
            "python": ".".join(str(part) for part in sys.version_info[:2]),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        methodological_notes=list(METHODOLOGICAL_NOTES),
    )


def write_results(results: HoldoutResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote holdout results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> HoldoutResults:
    """Read and validate the Phase 9D record.

    Raises:
        FileNotFoundError: If the evaluation has not been run yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Holdout results not found at {source}. Produce them with "
            "`uv run python scripts/evaluate_holdout.py`."
        )
    return HoldoutResults.model_validate_json(source.read_text(encoding="utf-8"))


def invariant_failures(results: HoldoutResults) -> list[str]:
    """Return every violated invariant of a loaded record. Read-only.

    What ``--verify`` checks. These are properties the record must satisfy for
    the result to mean what it says: the confusion matrix has to account for
    every row, the rates have to agree with the cells, every metric has to lie in
    its domain, each interval has to contain its own estimate, and the protocol
    flags have to say that nothing was selected after the holdout was opened.
    """
    failures: list[str] = []
    cells = results.confusion_matrix
    total = int(cells["total"])
    counted = sum(int(cells[name]) for name in CONFUSION_NAMES)

    if counted != total:
        failures.append(f"confusion cells sum to {counted}, not {total}")
    if total != int(results.holdout["n_samples"]):
        failures.append(f"confusion total {total} != n_samples {results.holdout['n_samples']}")
    if int(cells["actual_positives"]) != int(results.holdout["n_positive"]):
        failures.append("actual_positives disagrees with n_positive")
    if int(cells["actual_negatives"]) != int(results.holdout["n_negative"]):
        failures.append("actual_negatives disagrees with n_negative")

    expected_rate = round(int(cells["predicted_positives"]) / total, _PRECISION) if total else 0.0
    if abs(float(cells["predicted_positive_rate"]) - expected_rate) > 1e-6:
        failures.append("predicted_positive_rate disagrees with the confusion cells")

    for group, values in results.metrics.items():
        for name, value in values.items():
            if group == "probability_diagnostics":
                if not np.isfinite(value) or value < 0.0:
                    failures.append(f"{group}.{name} = {value} is not a valid loss")
                continue
            if not 0.0 <= value <= 1.0:
                failures.append(f"{group}.{name} = {value} is outside [0, 1]")

    for name, interval in results.confidence_intervals.items():
        if not interval.ci_lower <= interval.estimate <= interval.ci_upper:
            failures.append(
                f"{name}: estimate {interval.estimate} outside "
                f"[{interval.ci_lower}, {interval.ci_upper}]"
            )
        if interval.method != BOOTSTRAP_METHOD:
            failures.append(f"{name}: unexpected interval method {interval.method!r}")
        if interval.n_valid_replications + interval.n_degenerate_replications != (
            interval.n_bootstrap
        ):
            failures.append(f"{name}: valid + degenerate replications != n_bootstrap")

    provenance = results.frozen_model_provenance
    if not provenance["model_fingerprint_verified"]:
        failures.append("the loaded model fingerprint did not match the decision policy")
    if not provenance["pipeline_unchanged_by_evaluation"]:
        failures.append("the pipeline file changed during the evaluation")
    if provenance["refitted_here"]:
        failures.append("the record claims a refit occurred")
    if provenance["freeze_commit"] != FREEZE_COMMIT:
        failures.append("the freeze commit is not the expected one")

    if results.selection_after_holdout or results.selection_allowed:
        failures.append("the record permits selection after the holdout was opened")
    if not results.holdout_touched:
        failures.append("holdout_touched must be true from Phase 9D onwards")
    if results.methodology["fit_calls_after_holdout_opened"] != 0:
        failures.append("a fit call is recorded after the holdout was opened")
    if results.methodology["models_evaluated"] != 1:
        failures.append("more than one model was evaluated")
    if results.methodology["thresholds_evaluated"] != 1:
        failures.append("more than one threshold was evaluated")

    return failures
