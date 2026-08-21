"""Phase 9D: the final evaluation, and the proof that it only evaluated.

The property this phase lives or dies on is negative: **nothing was learned from
the holdout**. That cannot be shown by reading the code, so it is shown
behaviourally — every fit entry point is monkeypatched to raise while the real
evaluation runs to completion, the frozen pipeline is byte-compared before and
after, and the recorded protocol counters are asserted to be zero.

The arithmetic is checked against scikit-learn rather than against itself, the
bootstrap is checked for reproducibility under its recorded seed and for correct
handling of replications where a metric is undefined, and the record is checked
against its own invariants.

Synthetic fixtures are used wherever the question does not require the real
partition. Where it does — the row count, the identifier digest, the recorded
metrics — the real artefacts are read and the tests skip if the evaluation has
not been run.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import re

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from churn.config import PROJECT_ROOT
from churn.modeling.freeze import file_digest, load_pipeline, model_fingerprint
from churn.modeling.freeze_results import load_decision_policy
from churn.modeling.holdout import (
    AUXILIARY_METRICS,
    BOOTSTRAP_METHOD,
    BOOTSTRAP_METRICS,
    CONFIDENCE_LEVEL,
    CONFUSION_NAMES,
    DISCRIMINATION_METRICS,
    EVALUATION_TYPE,
    HOLDOUT_PURPOSE,
    N_BOOTSTRAP,
    OPERATING_POINT_METRICS,
    PROBABILITY_DIAGNOSTICS,
    HoldoutEvaluationError,
    bootstrap_intervals,
    compute_holdout_metrics,
    confusion_of,
    generalization_check,
    score_holdout,
)
from churn.modeling.holdout_results import (
    FREEZE_COMMIT,
    SCHEMA_VERSION,
    UPSTREAM_ARTEFACTS,
    HoldoutMismatchError,
    invariant_failures,
    load_results,
    verify_holdout_partition,
)
from churn.modeling.threshold_results import load_results as load_threshold_results
from churn.preprocessing.manifest import load_split_manifest

#: The frozen threshold. Pinned here as the human-approved value, so a drift in
#: the artefact fails a test rather than silently changing what was evaluated.
FROZEN_THRESHOLD = 0.3272694566222328

#: The frozen holdout size, from the Phase 4 manifest.
HOLDOUT_ROWS = 1409


def _load_script(name: str):
    """Import a file in ``scripts/`` as a module. It is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate_holdout = _load_script("evaluate_holdout")


def _scores(n: int = 400, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Labels correlated with a continuous score, for arithmetic checks."""
    generator = np.random.default_rng(seed)
    probability = generator.random(n)
    labels = (generator.random(n) < probability * 0.8 + 0.05).astype(int)
    return labels, probability


def _results():
    path = PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json"
    if not path.is_file():
        pytest.skip("the final evaluation has not been run yet")
    return load_results()


# --- the evaluation requires a valid freeze -----------------------------------


def test_the_evaluation_reads_the_policy_rather_than_assuming_it() -> None:
    """Nothing about the model is hard-coded in the evaluation path."""
    policy = load_decision_policy()

    assert policy.threshold.final_threshold == FROZEN_THRESHOLD
    assert policy.calibration.calibration_policy == "NONE"
    assert policy.threshold.threshold_policy == "F1_MAXIMIZATION"
    assert policy.decision_rule.comparison == ">="
    assert policy.decision_rule.positive_class_label == 1
    assert policy.model.hyperparameters == {
        "C": 1.0,
        "class_weight": None,
        "l1_ratio": 0.0,
        "max_iter": 100,
        "solver": "lbfgs",
    }
    assert policy.feature_contract.n_features == 19
    assert policy.feature_contract.engineered == []


def test_the_pipeline_is_loaded_and_its_fingerprint_verified() -> None:
    policy = load_decision_policy()
    path = PROJECT_ROOT / policy.artifacts.pipeline_path
    if not path.is_file():
        pytest.skip("the frozen pipeline has not been produced yet")

    pipeline = load_pipeline(path)

    assert model_fingerprint(pipeline) == policy.artifacts.model_fingerprint_sha256
    assert file_digest(path) == policy.artifacts.pipeline_sha256
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_the_record_names_the_freeze_commit_that_preceded_the_opening() -> None:
    results = _results()
    provenance = results.frozen_model_provenance

    assert provenance["freeze_commit"] == FREEZE_COMMIT
    assert provenance["freeze_preceded_holdout_opening"] is True
    assert provenance["model_fingerprint_verified"] is True
    assert provenance["refitted_here"] is False
    assert results.holdout_opened_after_model_freeze is True


# --- the holdout --------------------------------------------------------------


def test_the_holdout_has_exactly_the_frozen_number_of_rows() -> None:
    results = _results()
    manifest = load_split_manifest()

    assert results.holdout["n_samples"] == HOLDOUT_ROWS
    assert manifest.n_rows_holdout == HOLDOUT_ROWS
    assert results.holdout["n_positive"] + results.holdout["n_negative"] == HOLDOUT_ROWS


def test_the_holdout_matches_the_frozen_manifest() -> None:
    results = _results()
    verification = results.holdout["verification"]
    manifest = load_split_manifest()

    assert verification["matches_frozen_manifest"] is True
    assert verification["n_rows_observed"] == verification["n_rows_expected"]
    assert verification["holdout_ids_sha256_observed"] == manifest.holdout_ids_sha256


def test_the_partition_gate_rejects_a_different_row_count() -> None:
    import pandas as pd

    class _Manifest:
        n_rows_holdout = 3
        holdout_ids_sha256 = "a" * 64

    frame = pd.DataFrame({"customerID": ["a", "b"]})

    with pytest.raises(HoldoutMismatchError, match="rows against"):
        verify_holdout_partition(frame, _Manifest(), "customerID", lambda ids: "a" * 64)


def test_the_partition_gate_rejects_a_different_identifier_set() -> None:
    import pandas as pd

    class _Manifest:
        n_rows_holdout = 2
        holdout_ids_sha256 = "a" * 64

    frame = pd.DataFrame({"customerID": ["a", "b"]})

    with pytest.raises(HoldoutMismatchError, match="identifier digest"):
        verify_holdout_partition(frame, _Manifest(), "customerID", lambda ids: "b" * 64)


# --- the decision rule --------------------------------------------------------


def test_the_frozen_threshold_is_preserved_exactly() -> None:
    results = _results()
    raw = (PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json").read_text("utf-8")

    assert results.policy["final_threshold"] == FROZEN_THRESHOLD
    assert '"final_threshold": 0.3272694566222328' in raw
    assert results.metrics["operating_point"]["f1"] > 0.0


def test_the_comparison_is_greater_or_equal() -> None:
    results = _results()

    assert results.policy["comparison"] == ">="
    assert ">=" in results.policy["decision_rule"]


def test_predict_proba_is_used_and_predict_is_not() -> None:
    results = _results()

    assert results.methodology["predict_proba_used"] is True
    assert results.methodology["predict_used"] is False


def test_the_positive_class_column_is_resolved_not_assumed() -> None:
    policy = load_decision_policy()
    path = PROJECT_ROOT / policy.artifacts.pipeline_path
    if not path.is_file():
        pytest.skip("the frozen pipeline has not been produced yet")
    pipeline = load_pipeline(path)
    classes = pipeline.named_steps["classifier"].classes_

    column = policy.decision_rule.positive_class_column

    assert classes[column] == 1
    assert _results().policy["positive_class_column"] == column


def test_scoring_returns_the_positive_class_column(contract_frame) -> None:
    """``score_holdout`` reads the column ``classes_`` names, not column 1 by faith."""
    from churn.preprocessing.contracts import build_feature_matrix
    from churn.preprocessing.target import encode_target

    policy = load_decision_policy()
    path = PROJECT_ROOT / policy.artifacts.pipeline_path
    if not path.is_file():
        pytest.skip("the frozen pipeline has not been produced yet")
    pipeline = load_pipeline(path)
    features = build_feature_matrix(contract_frame)
    encode_target(contract_frame["Churn"])  # exercised for schema parity only

    probability = score_holdout(pipeline, features)
    column = policy.decision_rule.positive_class_column

    assert np.array_equal(probability, pipeline.predict_proba(features)[:, column])
    assert probability.min() >= 0.0
    assert probability.max() <= 1.0


# --- metric arithmetic --------------------------------------------------------


def test_the_confusion_matrix_orientation_is_pinned() -> None:
    labels = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 0, 1])

    cells = confusion_of(labels, predicted)

    assert (cells.true_negatives, cells.false_positives) == (1, 1)
    assert (cells.false_negatives, cells.true_positives) == (1, 1)
    assert cells.total == 4


def test_a_single_class_vector_still_produces_four_cells() -> None:
    """``labels=[0, 1]`` is pinned, so a degenerate vector cannot mis-assign cells."""
    cells = confusion_of(np.array([1, 1]), np.array([1, 1]))

    assert cells.true_positives == 2
    assert cells.true_negatives == 0
    assert cells.total == 2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_metrics_agree_with_scikit_learn(seed: int) -> None:
    labels, probability = _scores(seed=seed)
    threshold = 0.4
    predicted = (probability >= threshold).astype(int)

    metrics = compute_holdout_metrics(labels, probability, threshold)

    assert metrics.average_precision == pytest.approx(average_precision_score(labels, probability))
    assert metrics.roc_auc == pytest.approx(roc_auc_score(labels, probability))
    assert metrics.f1 == pytest.approx(f1_score(labels, predicted, zero_division=0))
    assert metrics.precision == pytest.approx(precision_score(labels, predicted, zero_division=0))
    assert metrics.recall == pytest.approx(recall_score(labels, predicted, zero_division=0))
    assert metrics.balanced_accuracy == pytest.approx(balanced_accuracy_score(labels, predicted))


def test_the_metrics_are_consistent_with_their_own_confusion_matrix() -> None:
    labels, probability = _scores()

    metrics = compute_holdout_metrics(labels, probability, FROZEN_THRESHOLD)
    cells = metrics.confusion

    assert cells.total == labels.size
    assert metrics.recall == pytest.approx(cells.true_positives / cells.actual_positives)
    assert metrics.precision == pytest.approx(cells.true_positives / cells.predicted_positives)
    assert metrics.specificity == pytest.approx(cells.true_negatives / cells.actual_negatives)
    assert metrics.false_positive_rate == pytest.approx(1.0 - metrics.specificity)
    assert metrics.false_negative_rate == pytest.approx(1.0 - metrics.recall)
    assert metrics.predicted_positive_rate == pytest.approx(cells.predicted_positives / cells.total)
    assert metrics.balanced_accuracy == pytest.approx((metrics.recall + metrics.specificity) / 2.0)


def test_the_metrics_reject_misaligned_input() -> None:
    with pytest.raises(HoldoutEvaluationError, match="one probability per label"):
        compute_holdout_metrics(np.array([0, 1, 1]), np.array([0.2, 0.8]), 0.5)
    with pytest.raises(HoldoutEvaluationError, match="at least one row"):
        compute_holdout_metrics(np.array([]), np.array([]), 0.5)


def test_the_recorded_metrics_lie_in_their_domains() -> None:
    results = _results()

    for group in ("discrimination_primary", "operating_point", "auxiliary"):
        for name, value in results.metrics[group].items():
            assert 0.0 <= value <= 1.0, f"{group}.{name} = {value}"
    for name, value in results.metrics["probability_diagnostics"].items():
        assert value >= 0.0 and np.isfinite(value), f"{name} = {value}"


def test_the_recorded_confusion_matrix_sums_to_the_holdout() -> None:
    results = _results()
    cells = results.confusion_matrix

    assert sum(int(cells[name]) for name in CONFUSION_NAMES) == HOLDOUT_ROWS
    assert int(cells["total"]) == HOLDOUT_ROWS
    assert cells["cells_sum_to_n_samples"] is True
    assert int(cells["actual_positives"]) == results.holdout["n_positive"]
    assert int(cells["actual_negatives"]) == results.holdout["n_negative"]


def test_the_predicted_positive_rate_matches_the_predictions() -> None:
    results = _results()
    cells = results.confusion_matrix

    expected = int(cells["predicted_positives"]) / HOLDOUT_ROWS

    assert float(cells["predicted_positive_rate"]) == pytest.approx(expected, abs=1e-6)
    assert results.metrics["operating_point"]["predicted_positive_rate"] == pytest.approx(
        expected, abs=1e-6
    )


def test_the_metric_groups_are_the_declared_ones() -> None:
    results = _results()

    assert set(results.metrics["discrimination_primary"]) == set(DISCRIMINATION_METRICS)
    assert set(results.metrics["operating_point"]) == set(OPERATING_POINT_METRICS)
    assert set(results.metrics["auxiliary"]) == set(AUXILIARY_METRICS)
    assert set(results.metrics["probability_diagnostics"]) == set(PROBABILITY_DIAGNOSTICS)
    assert set(results.metrics["reference_baselines"]) == {
        "majority_class_accuracy",
        "no_skill_average_precision",
        "no_skill_roc_auc",
    }
    # Accuracy is auxiliary, never a primary result.
    assert "accuracy" in AUXILIARY_METRICS
    assert "accuracy" not in DISCRIMINATION_METRICS


def test_the_recorded_reference_baselines_are_the_trivial_ones() -> None:
    """Each baseline is derived from a quantity already in the record."""
    results = _results()
    baselines = results.metrics["reference_baselines"]
    cells = results.confusion_matrix

    assert baselines["majority_class_accuracy"] == pytest.approx(
        int(cells["actual_negatives"]) / int(cells["total"]), abs=1e-6
    )
    assert baselines["no_skill_average_precision"] == pytest.approx(
        results.holdout["prevalence"], abs=1e-6
    )
    assert baselines["no_skill_roc_auc"] == 0.5


def test_accuracy_is_above_the_majority_class_baseline() -> None:
    """The arithmetic, asserted so no report can state the direction backwards.

    0.7615 against a 0.7346 majority-class baseline. Reported, not used to
    promote or reject anything: accuracy is auxiliary either way.
    """
    results = _results()
    accuracy = results.metrics["auxiliary"]["accuracy"]
    baseline = results.metrics["reference_baselines"]["majority_class_accuracy"]

    assert accuracy > baseline
    assert 100 * (accuracy - baseline) == pytest.approx(2.7, abs=0.1)


def test_the_report_states_the_accuracy_comparison_in_the_right_direction() -> None:
    report = PROJECT_ROOT / "reports" / "holdout_evaluation_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")
    text = report.read_text(encoding="utf-8")
    results = _results()
    accuracy = results.metrics["auxiliary"]["accuracy"]
    baseline = results.metrics["reference_baselines"]["majority_class_accuracy"]

    # The substance, not a particular phrasing: both values present, the
    # difference stated with its sign, and the direction never reversed.
    assert "Majority class" in text
    assert f"{baseline:.4f}" in text
    assert f"{accuracy:.4f}" in text
    assert f"{100 * (accuracy - baseline):+.1f} percentage points" in text
    assert "decides nothing" in text
    for reversed_claim in ("lower than the silent rule", "*lower* than", "below the majority"):
        assert reversed_claim not in text, reversed_claim


# --- bootstrap ----------------------------------------------------------------


def test_the_bootstrap_is_deterministic_under_its_seed() -> None:
    labels, probability = _scores()
    point = compute_holdout_metrics(labels, probability, FROZEN_THRESHOLD)

    first = bootstrap_intervals(
        labels, probability, FROZEN_THRESHOLD, point, ("f1", "roc_auc"), 200, 0.95, 42
    )
    second = bootstrap_intervals(
        labels, probability, FROZEN_THRESHOLD, point, ("f1", "roc_auc"), 200, 0.95, 42
    )

    for name in ("f1", "roc_auc"):
        assert first[name].ci_lower == second[name].ci_lower
        assert first[name].ci_upper == second[name].ci_upper


def test_a_different_seed_moves_the_interval() -> None:
    """Proof the seed is actually used rather than decorative."""
    labels, probability = _scores()
    point = compute_holdout_metrics(labels, probability, FROZEN_THRESHOLD)

    a = bootstrap_intervals(labels, probability, FROZEN_THRESHOLD, point, ("f1",), 200, 0.95, 1)
    b = bootstrap_intervals(labels, probability, FROZEN_THRESHOLD, point, ("f1",), 200, 0.95, 2)

    assert a["f1"].ci_lower != b["f1"].ci_lower


def test_the_interval_contains_its_point_estimate() -> None:
    labels, probability = _scores()
    point = compute_holdout_metrics(labels, probability, FROZEN_THRESHOLD)

    intervals = bootstrap_intervals(
        labels, probability, FROZEN_THRESHOLD, point, BOOTSTRAP_METRICS, 300, 0.95, 42
    )

    for name, interval in intervals.items():
        assert interval.ci_lower <= interval.estimate <= interval.ci_upper, name
        assert interval.contains_estimate
        assert interval.estimate == point.value(name)


def test_the_point_estimate_is_the_real_holdout_not_the_bootstrap_mean() -> None:
    labels, probability = _scores()
    point = compute_holdout_metrics(labels, probability, FROZEN_THRESHOLD)

    intervals = bootstrap_intervals(
        labels, probability, FROZEN_THRESHOLD, point, ("roc_auc",), 200, 0.95, 42
    )

    assert intervals["roc_auc"].estimate == point.roc_auc


def test_a_degenerate_replication_is_excluded_and_counted() -> None:
    """A single-class sample makes ROC-AUC undefined; it must not be filled in."""
    labels = np.zeros(20, dtype=int)
    labels[0] = 1
    probability = np.linspace(0.05, 0.95, 20)
    point = compute_holdout_metrics(labels, probability, 0.5)

    intervals = bootstrap_intervals(labels, probability, 0.5, point, ("roc_auc",), 200, 0.95, 7)
    interval = intervals["roc_auc"]

    assert interval.n_degenerate > 0, "no replication lacked the positive class"
    assert interval.n_valid + interval.n_degenerate == interval.n_bootstrap
    assert np.isfinite(interval.ci_lower) and np.isfinite(interval.ci_upper)


def test_an_all_degenerate_metric_raises_instead_of_inventing_an_interval() -> None:
    labels = np.zeros(10, dtype=int)
    probability = np.linspace(0.1, 0.9, 10)
    point = compute_holdout_metrics(labels, probability, 0.5)

    with pytest.raises(HoldoutEvaluationError, match="degenerate"):
        bootstrap_intervals(labels, probability, 0.5, point, ("roc_auc",), 50, 0.95, 3)


def test_an_unknown_bootstrap_metric_is_refused() -> None:
    labels, probability = _scores()
    point = compute_holdout_metrics(labels, probability, 0.5)

    with pytest.raises(HoldoutEvaluationError, match="No bootstrap implementation"):
        bootstrap_intervals(labels, probability, 0.5, point, ("mystery",), 10, 0.95, 1)


def test_the_bootstrap_does_not_touch_the_model() -> None:
    """It resamples probabilities that already exist; no estimator is involved."""
    policy = load_decision_policy()
    path = PROJECT_ROOT / policy.artifacts.pipeline_path
    if not path.is_file():
        pytest.skip("the frozen pipeline has not been produced yet")

    before = file_digest(path)
    fingerprint_before = model_fingerprint(load_pipeline(path))
    labels, probability = _scores()
    point = compute_holdout_metrics(labels, probability, FROZEN_THRESHOLD)

    bootstrap_intervals(labels, probability, FROZEN_THRESHOLD, point, ("f1",), 100, 0.95, 42)

    assert file_digest(path) == before
    assert model_fingerprint(load_pipeline(path)) == fingerprint_before


def test_the_recorded_bootstrap_configuration_is_the_declared_one() -> None:
    results = _results()
    bootstrap = results.bootstrap

    assert bootstrap["method"] == BOOTSTRAP_METHOD == "bootstrap_percentile"
    assert bootstrap["n_bootstrap"] == N_BOOTSTRAP >= 2000
    assert bootstrap["confidence_level"] == CONFIDENCE_LEVEL == 0.95
    assert bootstrap["model_refitted_per_replication"] is False
    assert bootstrap["threshold_reselected_per_replication"] is False
    assert bootstrap["not_a_significance_test"] is True
    assert set(results.confidence_intervals) == set(BOOTSTRAP_METRICS)


def test_every_recorded_interval_brackets_its_estimate() -> None:
    results = _results()

    for name, interval in results.confidence_intervals.items():
        assert interval.ci_lower <= interval.estimate <= interval.ci_upper, name
        assert interval.confidence_level == 0.95
        assert interval.n_valid_replications + interval.n_degenerate_replications == (
            interval.n_bootstrap
        )


# --- no learning happened -----------------------------------------------------


def test_the_evaluation_calls_no_fit(monkeypatch) -> None:
    """Every fit entry point raises, and the real evaluation still completes.

    The strongest available evidence: it does not observe that no fit happened,
    it makes fitting impossible and shows the evaluation never attempted it.
    """
    if not (PROJECT_ROOT / "reports" / "decision_policy.json").is_file():
        pytest.skip("the freeze has not been produced yet")

    attempted: list[str] = []

    def forbid(name: str):
        def guard(*args, **kwargs):
            attempted.append(name)
            raise AssertionError(f"the holdout evaluation called {name}")

        return guard

    monkeypatch.setattr(Pipeline, "fit", forbid("Pipeline.fit"))
    monkeypatch.setattr(Pipeline, "fit_transform", forbid("Pipeline.fit_transform"))
    monkeypatch.setattr(LogisticRegression, "fit", forbid("LogisticRegression.fit"))

    results, metrics, target, probability = evaluate_holdout.evaluate()

    assert attempted == [], f"the evaluation attempted: {attempted}"
    assert target.size == HOLDOUT_ROWS
    assert probability.size == HOLDOUT_ROWS
    assert results.methodology["fit_calls_after_holdout_opened"] == 0
    assert metrics.confusion.total == HOLDOUT_ROWS


def test_the_frozen_pipeline_is_byte_identical_after_the_evaluation() -> None:
    results = _results()
    provenance = results.frozen_model_provenance

    assert (
        provenance["pipeline_sha256_before_evaluation"]
        == provenance["pipeline_sha256_after_evaluation"]
    )
    assert provenance["pipeline_unchanged_by_evaluation"] is True
    assert provenance["pipeline_sha256_recorded"] == provenance["pipeline_sha256_after_evaluation"]

    policy = load_decision_policy()
    path = PROJECT_ROOT / policy.artifacts.pipeline_path
    if path.is_file():
        assert file_digest(path) == provenance["pipeline_sha256_after_evaluation"]


def test_the_record_distinguishes_configurations_from_physical_executions() -> None:
    """ "One evaluation" must mean one configuration, not one physical file read.

    The re-runs are real: deterministic verification and the test suite both
    execute the identical frozen evaluation. The record has to say what it is
    counting, and it must not claim the file was read exactly once.
    """
    results = _results()
    methodology = results.methodology
    raw = (PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json").read_text("utf-8")

    assert methodology["evaluated_configurations"] == 1
    assert "may exceed one" in methodology["physical_executions"]
    assert "opened exactly once" not in raw
    assert "read exactly once" not in raw
    assert "ONE frozen model-and-decision-policy configuration" in raw


def test_the_report_does_not_claim_a_single_physical_read() -> None:
    report = PROJECT_ROOT / "reports" / "holdout_evaluation_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")
    # Line breaks and emphasis markers are artefacts of the template, not of the
    # claim, so both are normalised away before the phrases are checked.
    raw = re.sub(r"\s+", " ", report.read_text(encoding="utf-8"))
    text = raw.replace("*", "").lower()

    # The scientific claim is present and correctly scoped...
    assert "one frozen configuration was evaluated" in text
    assert "one frozen model-and-decision-policy configuration was evaluated" in text
    # ...and the claim about physical reads is explicitly disclaimed.
    assert "is not a claim that the holdout file was physically read exactly once" in text
    # No unqualified claim of a single read survives anywhere: every mention of
    # reading once must be inside that disclaimer.
    for match in re.finditer(r"(read|opened) exactly once", text):
        preceding = text[max(0, match.start() - 70) : match.start()]
        assert "not a claim" in preceding, (
            f"unqualified single-read claim at offset {match.start()}"
        )


def test_the_report_lists_upstream_artefacts_explicitly() -> None:
    """The evidence of immutability is a named list, never a directory."""
    report = PROJECT_ROOT / "reports" / "holdout_evaluation_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")
    text = report.read_text(encoding="utf-8")

    assert "Upstream artefacts, listed explicitly" in text
    assert "a directory-level check would be no" in text
    for artefact in UPSTREAM_ARTEFACTS:
        assert f"`{artefact}`" in text, artefact


def test_the_freeze_report_is_among_the_anchored_artefacts() -> None:
    """Every pre-9D artefact the freeze produced is digest-anchored here."""
    results = _results()

    assert "reports/model_freeze_report.md" in results.upstream_artefact_digests
    assert "reports/decision_policy.json" in results.upstream_artefact_digests
    assert "reports/split_manifest.json" in results.upstream_artefact_digests
    # The .joblib is covered more strongly, by the before/after pair.
    assert "artifacts/model/churn_pipeline.joblib" not in results.upstream_artefact_digests
    assert results.frozen_model_provenance["pipeline_unchanged_by_evaluation"] is True


def test_the_new_record_is_not_treated_as_a_pre_phase9d_artefact() -> None:
    """Its own output cannot be part of the set it claims to have left unchanged."""
    results = _results()

    assert "reports/experiments/holdout_results.json" not in results.upstream_artefact_digests


def test_exactly_one_model_one_threshold_and_no_selection() -> None:
    results = _results()
    methodology = results.methodology

    assert methodology["models_evaluated"] == 1
    assert methodology["thresholds_evaluated"] == 1
    assert methodology["calibrations_performed"] == 0
    assert methodology["hyperparameter_searches"] == 0
    assert methodology["feature_selections"] == 0
    assert methodology["holdout_evaluations"] == 1
    assert methodology["subgroup_slices_evaluated"] == 0
    assert results.evaluations_performed == 1
    assert results.selection_allowed is False
    assert results.selection_after_holdout is False


def test_no_financial_quantity_or_causal_claim_is_recorded() -> None:
    results = _results()
    raw = (PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json").read_text("utf-8")

    assert results.methodology["financial_costs_assumed"] is False
    assert results.methodology["causal_claims"] is False
    for forbidden in ("cost_fp", "cost_fn", "revenue", "profit", "roi_", "lifetime_value"):
        assert forbidden not in raw.lower(), forbidden


# --- the record ---------------------------------------------------------------


def test_the_record_declares_what_it_is() -> None:
    results = _results()

    assert results.schema_version == SCHEMA_VERSION
    assert results.evaluation_type == EVALUATION_TYPE == "FINAL_HOLDOUT"
    assert results.holdout_touched is True
    assert results.holdout_purpose == HOLDOUT_PURPOSE == "FINAL_EVALUATION"


def test_the_record_passes_its_own_invariants() -> None:
    results = _results()

    assert invariant_failures(results) == []


def test_the_verify_command_is_read_only_and_passes(monkeypatch) -> None:
    """`--verify` re-checks the record without re-evaluating or writing."""
    if not (PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json").is_file():
        pytest.skip("the final evaluation has not been run yet")

    watched = [
        PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json",
        PROJECT_ROOT / "reports" / "holdout_evaluation_report.md",
    ]
    before = {path: (file_digest(path), path.stat().st_mtime_ns) for path in watched}

    def forbid(*args, **kwargs):
        raise AssertionError("--verify wrote a file")

    monkeypatch.setattr(pathlib.Path, "write_text", forbid)
    monkeypatch.setattr(pathlib.Path, "write_bytes", forbid)
    monkeypatch.setattr(Pipeline, "fit", forbid)

    assert evaluate_holdout.main(["--verify"]) == 0

    for path in watched:
        assert (file_digest(path), path.stat().st_mtime_ns) == before[path]


def test_the_invariant_checker_catches_a_broken_confusion_matrix() -> None:
    results = _results()
    payload = json.loads(results.model_dump_json())
    payload["confusion_matrix"]["true_positives"] += 1

    from churn.modeling.holdout_results import HoldoutResults

    failures = invariant_failures(HoldoutResults.model_validate(payload))

    assert any("confusion cells sum" in failure for failure in failures)


def test_the_invariant_checker_catches_an_interval_not_containing_its_estimate() -> None:
    results = _results()
    payload = json.loads(results.model_dump_json())
    payload["confidence_intervals"]["f1"]["ci_lower"] = 0.99
    payload["confidence_intervals"]["f1"]["ci_upper"] = 0.999

    from churn.modeling.holdout_results import HoldoutResults

    failures = invariant_failures(HoldoutResults.model_validate(payload))

    assert any("outside" in failure for failure in failures)


def test_the_invariant_checker_catches_a_claimed_refit() -> None:
    results = _results()
    payload = json.loads(results.model_dump_json())
    payload["frozen_model_provenance"]["refitted_here"] = True
    payload["selection_after_holdout"] = True

    from churn.modeling.holdout_results import HoldoutResults

    failures = invariant_failures(HoldoutResults.model_validate(payload))

    assert any("refit" in failure for failure in failures)
    assert any("selection" in failure for failure in failures)


def test_the_record_carries_no_timestamp() -> None:
    path = PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json"
    if not path.is_file():
        pytest.skip("the final evaluation has not been run yet")
    raw = path.read_text(encoding="utf-8")

    assert not [key for key in json.loads(raw) if re.search("time|stamp|generated", key, re.I)]
    assert not re.search(r"\d{4}-\d{2}-\d{2}", raw)


# --- comparison with development ---------------------------------------------


def test_the_comparison_reads_the_development_values_from_artefacts() -> None:
    results = _results()
    threshold_reference = load_threshold_results()
    development = {record.experiment: record for record in threshold_reference.policies}
    checks = {
        (check["metric"], check["development_source"]): check
        for check in results.comparison_with_development["checks"]
    }

    outer = ("phase9b outer-fold mean (threshold-independent)",)
    assert checks[("average_precision", *outer)]["development"] == pytest.approx(
        development["D0"].mean["average_precision"], abs=1e-6
    )
    assert checks[("roc_auc", *outer)]["development"] == pytest.approx(
        development["D0"].mean["roc_auc"], abs=1e-6
    )


def test_threshold_dependent_comparisons_are_flagged_not_comparable() -> None:
    results = _results()

    for check in results.comparison_with_development["checks"]:
        if check["metric"] in ("average_precision", "roc_auc"):
            assert check["directly_comparable"] is True
        else:
            assert check["directly_comparable"] is False
            assert "nested" in check["caveat"].lower()


def test_the_comparison_carries_no_gate_or_cutoff() -> None:
    results = _results()
    comparison = results.comparison_with_development

    assert comparison["selection"] is False
    assert comparison["gate"] is None
    assert comparison["cutoff"] is None


def test_the_difference_is_holdout_minus_development() -> None:
    results = _results()

    for check in results.comparison_with_development["checks"]:
        assert check["difference"] == pytest.approx(
            check["holdout"] - check["development"], abs=1e-6
        )


def test_a_generalization_check_is_descriptive() -> None:
    check = generalization_check("roc_auc", 0.84, 0.85, "source", True, "caveat")

    assert check.difference == pytest.approx(-0.01)
    assert check.comparable is True


# --- upstream artefacts are untouched ----------------------------------------


def test_the_frozen_and_earlier_artefacts_are_unchanged() -> None:
    """Every digest the record anchored to still matches the file on disk."""
    results = _results()
    recorded = results.upstream_artefact_digests

    assert set(recorded) == set(UPSTREAM_ARTEFACTS)
    for artefact, digest in recorded.items():
        actual = file_digest(PROJECT_ROOT / artefact)
        assert actual == digest, f"{artefact} changed since the evaluation"


def test_the_decision_policy_still_records_the_freeze_as_untouched() -> None:
    """9C's `holdout_touched: false` is a statement about the freeze and stays true."""
    policy = load_decision_policy()

    assert policy.holdout_touched is False
    assert _results().holdout_touched is True


# --- audits -------------------------------------------------------------------

_PHASE9D_SOURCES = [
    PROJECT_ROOT / "src" / "churn" / "modeling" / "holdout.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "holdout_results.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "holdout_plots.py",
    PROJECT_ROOT / "scripts" / "evaluate_holdout.py",
]

FORBIDDEN_IDENTIFIERS = (
    "fit",
    "fit_transform",
    "fit_predict",
    "GridSearchCV",
    "RandomizedSearchCV",
    "CalibratedClassifierCV",
    "TunedThresholdClassifierCV",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "select_f1_threshold",
    "run_threshold_nested_cv",
    "fit_final_pipeline",
    "dump_pipeline",
    "build_modern_logistic",
    "cross_val_score",
    "cross_validate",
    "XGBClassifier",
    "optuna",
)


def _identifiers(source) -> set[str]:
    used: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.alias):
            used.add(node.name.rsplit(".", 1)[-1])
            if node.asname:
                used.add(node.asname)
    return used


@pytest.mark.parametrize("source", _PHASE9D_SOURCES, ids=lambda path: path.name)
def test_phase9d_code_never_learns(source) -> None:
    """No fit, no search, no calibrator, no threshold selection anywhere."""
    used = _identifiers(source)

    found = sorted(name for name in FORBIDDEN_IDENTIFIERS if name in used)
    assert not found, (
        f"{source.name} references {found}: this phase evaluates a frozen system and must not "
        "fit, search, calibrate or select anything"
    )


@pytest.mark.parametrize("source", _PHASE9D_SOURCES, ids=lambda path: path.name)
def test_only_the_script_opens_the_holdout(source) -> None:
    """The holdout is authorised in this phase, and only at the orchestration layer."""
    used = _identifiers(source)

    if source.name == "evaluate_holdout.py":
        assert "load_holdout" in used
        return
    assert "load_holdout" not in used, f"{source.name} loads the holdout itself"
    assert "load_training_pool" not in used


def test_the_metric_modules_load_no_data() -> None:
    """They receive arrays; they cannot reach a partition even by accident."""
    import churn.modeling.holdout as module

    for name in ("load_holdout", "load_training_pool", "load_split", "load_raw_typed"):
        assert not hasattr(module, name)


def test_the_report_generator_reads_no_clock() -> None:
    used = _identifiers(PROJECT_ROOT / "scripts" / "evaluate_holdout.py")

    found = sorted(
        name
        for name in ("datetime", "date", "today", "now", "utcnow", "monotonic", "perf_counter")
        if name in used
    )
    assert not found, f"evaluate_holdout.py reads wall-clock state via {found}"


def test_the_report_is_labelled_as_the_final_test_result() -> None:
    report = PROJECT_ROOT / "reports" / "holdout_evaluation_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")
    text = report.read_text(encoding="utf-8")

    assert "final test" in text.lower()
    assert "The final holdout has now been used for evaluation." in text
    assert "No model-selection decision was made from the holdout results." in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    # The holdout may no longer be called untouched as a present state. Every
    # occurrence must be qualified — historical ("previously untouched") or
    # negated ("no longer untouched") — and the qualifier must sit immediately
    # before the word rather than somewhere in the surrounding paragraph.
    qualifiers = ("previously", "no longer")
    for match in re.finditer(r"untouched", text):
        preceding = text[max(0, match.start() - 12) : match.start()].strip().lower()
        assert any(preceding.endswith(qualifier) for qualifier in qualifiers), (
            f"unqualified 'untouched' at offset {match.start()}: {preceding!r}"
        )
