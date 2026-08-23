"""Phase 9E: the post-hoc analysis, and the proof that it only described.

Two negative properties carry this phase and neither can be read off the source,
so both are asserted behaviourally: **nothing was learned** — every fit entry
point raises while the real analysis runs to completion — and **nothing was
selected** — the frozen artefacts are byte-compared before and after and the
protocol counters are asserted to be zero.

The rest is arithmetic that has to be right or the description is worthless: the
four outcome classes must partition the sample exactly, every rate must use the
denominator it claims, and a rate whose denominator is too small must come back
as ``None`` rather than as a plausible-looking number.

Synthetic fixtures are used wherever the question does not need the real sample.
Where it does, the committed artefacts are read and the tests skip if the
analysis has not been run.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.analysis.frames import TENURE_BAND, TENURE_BAND_EDGES, TENURE_BAND_LABELS
from churn.config import PROJECT_ROOT
from churn.modeling.error_analysis import (
    ANALYSIS_TYPE,
    AUDIT_AXES,
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    HIGH_CONFIDENCE_FALSE_NEGATIVE_MAX,
    HIGH_CONFIDENCE_FALSE_POSITIVE_MIN,
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
    TRUE_NEGATIVE,
    TRUE_POSITIVE,
    ConfusionReproductionError,
    ErrorAnalysisError,
    classify_outcomes,
    composition,
    high_confidence_errors,
    insufficient_cells,
    margin_band_labels,
    margin_table,
    marginal_error_counts,
    margins,
    outcome_counts,
    slice_metrics,
    slice_table,
    summarise_probabilities,
    verify_confusion_reproduction,
)
from churn.modeling.error_analysis_results import (
    EVALUATION_COMMIT,
    SCHEMA_VERSION,
    UPSTREAM_ARTEFACTS,
    invariant_failures,
    load_results,
)
from churn.modeling.freeze import file_digest, load_pipeline, model_fingerprint
from churn.modeling.freeze_results import load_decision_policy
from churn.modeling.holdout_results import load_results as load_evaluation_results

#: The frozen values, pinned as the human-approved ones so a drift in an artefact
#: fails a test rather than silently changing what is being described.
FROZEN_THRESHOLD = 0.3272694566222328
SAMPLE_ROWS = 1409
RECORDED_CONFUSION = {"TN": 803, "FP": 232, "FN": 104, "TP": 270}


def _load_script(name: str):
    """Import a file in ``scripts/`` as a module. It is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_error_analysis = _load_script("run_error_analysis")


def _results():
    path = PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json"
    if not path.is_file():
        pytest.skip("the post-hoc analysis has not been run yet")
    return load_results()


def _scores(n: int = 400, seed: int = 5) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    probability = generator.random(n)
    labels = (generator.random(n) < probability * 0.8 + 0.05).astype(int)
    return labels, probability


# --- the analysis requires a valid freeze and a valid evaluation ---------------


def test_the_frozen_policy_is_unchanged() -> None:
    policy = load_decision_policy()

    assert policy.threshold.final_threshold == FROZEN_THRESHOLD
    assert policy.decision_rule.comparison == ">="
    assert policy.decision_rule.positive_class_label == 1
    assert policy.calibration.calibration_policy == "NONE"
    assert policy.threshold.threshold_policy == "F1_MAXIMIZATION"


def test_the_phase9d_evaluation_is_present_and_unchanged() -> None:
    evaluation = load_evaluation_results()
    cells = evaluation.confusion_matrix

    assert {
        name: int(cells[key])
        for name, key in (
            ("TN", "true_negatives"),
            ("FP", "false_positives"),
            ("FN", "false_negatives"),
            ("TP", "true_positives"),
        )
    } == RECORDED_CONFUSION
    assert evaluation.holdout["n_samples"] == SAMPLE_ROWS


def test_the_record_names_the_evaluation_it_describes() -> None:
    results = _results()
    provenance = results.provenance

    assert provenance["evaluation_commit"] == EVALUATION_COMMIT
    assert provenance["evaluation_preceded_this_analysis"] is True
    assert provenance["model_fingerprint_verified"] is True
    assert provenance["refitted_here"] is False


# --- the integrity gate: the same classification ------------------------------


def test_the_recorded_confusion_matrix_reproduces_phase9d_exactly() -> None:
    results = _results()
    reproduction = results.confusion_reproduction

    assert reproduction["reproduced"] is True
    assert {k: int(v) for k, v in reproduction["derived"].items()} == RECORDED_CONFUSION
    assert {k: int(v) for k, v in reproduction["recorded"].items()} == RECORDED_CONFUSION
    assert int(reproduction["total"]) == SAMPLE_ROWS


def test_the_reproduction_gate_rejects_a_different_classification() -> None:
    derived = {"TN": 800, "FP": 235, "FN": 104, "TP": 270}

    with pytest.raises(ConfusionReproductionError, match="does not reproduce"):
        verify_confusion_reproduction(derived, 803, 232, 104, 270)


def test_the_reproduction_gate_accepts_the_matching_one() -> None:
    record = verify_confusion_reproduction(dict(RECORDED_CONFUSION), 803, 232, 104, 270)

    assert record["reproduced"] is True
    assert record["total"] == SAMPLE_ROWS


# --- the outcome partition ----------------------------------------------------


def test_the_four_classes_partition_every_row() -> None:
    labels = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 0, 1])

    outcomes = classify_outcomes(labels, predicted)

    assert outcomes.tolist() == [TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE, TRUE_POSITIVE]
    assert sum(outcome_counts(outcomes).values()) == labels.size


def test_the_recorded_classes_partition_the_whole_sample() -> None:
    results = _results()
    derived = results.confusion_reproduction["derived"]

    assert sum(int(derived[name]) for name in OUTCOME_CLASSES) == SAMPLE_ROWS
    assert (
        sum(int(results.probability_summaries[name]["count"]) for name in OUTCOME_CLASSES)
        == SAMPLE_ROWS
    )


def test_classification_rejects_misaligned_or_non_binary_input() -> None:
    with pytest.raises(ErrorAnalysisError, match="both for every row"):
        classify_outcomes(np.array([0, 1, 1]), np.array([0, 1]))
    with pytest.raises(ErrorAnalysisError, match="binary"):
        classify_outcomes(np.array([0, 2]), np.array([0, 1]))


def test_the_decision_rule_is_the_frozen_one() -> None:
    """The analysis cuts at the frozen threshold with ``>=``, never its own."""
    results = _results()

    assert results.frozen_policy["final_threshold"] == FROZEN_THRESHOLD
    assert results.frozen_policy["comparison"] == ">="
    assert results.frozen_policy["alternative_thresholds_scored"] == 0
    assert results.margin_analysis["threshold_used"] == FROZEN_THRESHOLD
    assert results.margin_analysis["threshold_re_derived"] is False


# --- conditional rates use the right denominators -----------------------------


def test_slice_rates_use_their_own_denominators() -> None:
    labels = np.array([1] * 40 + [0] * 60)
    predicted = np.array([1] * 30 + [0] * 10 + [1] * 20 + [0] * 40)

    entry = slice_metrics("axis", "group", labels, predicted)

    assert entry.n == 100
    assert (entry.positives, entry.negatives) == (40, 60)
    assert (entry.true_positives, entry.false_negatives) == (30, 10)
    assert (entry.false_positives, entry.true_negatives) == (20, 40)
    assert entry.recall == pytest.approx(30 / 40)
    assert entry.false_negative_rate == pytest.approx(10 / 40)
    assert entry.specificity == pytest.approx(40 / 60)
    assert entry.false_positive_rate == pytest.approx(20 / 60)
    assert entry.precision == pytest.approx(30 / 50)
    assert entry.recall + entry.false_negative_rate == pytest.approx(1.0)
    assert entry.specificity + entry.false_positive_rate == pytest.approx(1.0)


def test_a_rate_is_withheld_when_its_denominator_is_too_small() -> None:
    """Not zero, not a number: ``None``. The guardrail is per-denominator."""
    labels = np.array([1] * 5 + [0] * 45)
    predicted = np.array([1] * 5 + [0] * 45)

    entry = slice_metrics("axis", "group", labels, predicted)

    assert entry.n == 50 >= MIN_GROUP_SIZE
    assert entry.positives == 5 < MIN_ACTUAL_POSITIVES
    assert entry.recall is None
    assert entry.false_negative_rate is None
    assert entry.sufficient_for_recall is False
    # The negatives are plentiful, so those rates survive.
    assert entry.negatives == 45 >= MIN_ACTUAL_NEGATIVES
    assert entry.specificity is not None
    assert entry.false_positive_rate is not None


def test_a_small_group_withholds_every_rate() -> None:
    labels = np.array([1] * 10 + [0] * 10)
    predicted = np.zeros(20, dtype=int)

    entry = slice_metrics("axis", "group", labels, predicted)

    assert entry.n == 20 < MIN_GROUP_SIZE
    assert entry.sufficient_for_group_rates is False
    for rate in (
        entry.prevalence,
        entry.predicted_positive_rate,
        entry.precision,
        entry.recall,
        entry.specificity,
        entry.false_positive_rate,
        entry.false_negative_rate,
    ):
        assert rate is None


def test_precision_is_withheld_without_enough_predicted_positives() -> None:
    labels = np.array([1] * 30 + [0] * 70)
    predicted = np.array([1] * 5 + [0] * 95)

    entry = slice_metrics("axis", "group", labels, predicted)

    assert entry.predicted_positives == 5 < MIN_PREDICTED_POSITIVES
    assert entry.precision is None
    assert entry.recall is not None


def test_every_recorded_rate_has_its_support_flag() -> None:
    """No table anywhere reports a rate it declared unsupported."""
    results = _results()
    tables = [
        *results.slice_analyses.values(),
        *results.demographic_audit["axes"].values(),
        *results.interactions["cells"].values(),
    ]

    for rows in tables:
        for row in rows:
            if not row["sufficient_for_recall"]:
                assert row["recall"] is None
                assert row["false_negative_rate"] is None
            if not row["sufficient_for_specificity"]:
                assert row["specificity"] is None
                assert row["false_positive_rate"] is None
            if not row["sufficient_for_precision"]:
                assert row["precision"] is None
            if not row["sufficient_for_group_rates"]:
                assert row["prevalence"] is None


def test_slice_counts_sum_to_the_sample_and_to_their_own_n() -> None:
    results = _results()

    for axis, rows in results.slice_analyses.items():
        assert sum(int(row["n"]) for row in rows) == SAMPLE_ROWS, axis
        assert sum(int(row["positives"]) for row in rows) == 374, axis
        assert sum(int(row["negatives"]) for row in rows) == 1035, axis
        for row in rows:
            cells = sum(
                int(row[key])
                for key in (
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "true_negatives",
                )
            )
            assert cells == int(row["n"]), f"{axis}/{row['group']}"


def test_slice_error_counts_sum_to_the_recorded_confusion() -> None:
    results = _results()

    for axis, rows in results.slice_analyses.items():
        assert sum(int(row["false_positives"]) for row in rows) == RECORDED_CONFUSION["FP"], axis
        assert sum(int(row["false_negatives"]) for row in rows) == RECORDED_CONFUSION["FN"], axis


def test_slice_table_requires_a_real_axis(contract_frame: pd.DataFrame) -> None:
    labels = np.zeros(len(contract_frame), dtype=int)

    with pytest.raises(ErrorAnalysisError, match="not a column"):
        slice_table(contract_frame, "NoSuchColumn", labels, labels)


def test_slice_table_rejects_misaligned_lengths(contract_frame: pd.DataFrame) -> None:
    labels = np.zeros(len(contract_frame) - 1, dtype=int)

    with pytest.raises(ErrorAnalysisError, match="rows against"):
        slice_table(contract_frame, "Contract", labels, labels)


# --- tenure bands are the pre-specified ones ---------------------------------


def test_the_tenure_bands_are_the_eda_ones() -> None:
    results = _results()
    groups = [row["group"] for row in results.slice_analyses[TENURE_BAND]]

    assert TENURE_BAND_EDGES == (0, 6, 12, 24, 48, 72)
    assert groups == list(TENURE_BAND_LABELS)
    assert groups == ["0-6", "7-12", "13-24", "25-48", "49-72"]


def test_the_slice_axes_are_the_declared_ones() -> None:
    results = _results()

    assert list(results.slice_analyses) == list(SLICE_AXES)
    assert list(results.demographic_audit["axes"]) == list(AUDIT_AXES)
    assert results.methodology["features_swept"] == 0
    assert results.methodology["slice_axes_fixed_before_analysis"] is True
    assert results.interactions["pairs_examined"] == [list(pair) for pair in INTERACTION_AXES]
    assert results.interactions["automatic_combination_search"] is False


# --- margins -----------------------------------------------------------------


def test_margins_are_measured_from_the_frozen_threshold() -> None:
    probability = np.array([0.2, FROZEN_THRESHOLD, 0.9])

    values = margins(probability, FROZEN_THRESHOLD)

    assert values[1] == pytest.approx(0.0)
    assert values[0] < 0 and values[2] > 0
    assert np.allclose(values, probability - FROZEN_THRESHOLD)


def test_margin_bands_are_the_fixed_edges() -> None:
    absolute = np.array([0.0, 0.024, 0.025, 0.049, 0.05, 0.099, 0.10, 0.9])

    bands = margin_band_labels(absolute)

    assert bands.tolist() == [
        MARGIN_BAND_LABELS[0],
        MARGIN_BAND_LABELS[0],
        MARGIN_BAND_LABELS[1],
        MARGIN_BAND_LABELS[1],
        MARGIN_BAND_LABELS[2],
        MARGIN_BAND_LABELS[2],
        MARGIN_BAND_LABELS[3],
        MARGIN_BAND_LABELS[3],
    ]
    assert MARGIN_BAND_EDGES == (0.0, 0.025, 0.05, 0.10)


def test_the_margin_table_covers_every_row() -> None:
    labels, probability = _scores()
    outcomes = classify_outcomes(labels, (probability >= 0.5).astype(int))

    table = margin_table(outcomes, np.abs(margins(probability, 0.5)))

    assert sum(sum(row.values()) for row in table.values()) == labels.size


def test_the_recorded_margin_bands_cover_every_row() -> None:
    results = _results()
    counts = results.margin_analysis["counts_by_band"]

    assert list(counts) == list(MARGIN_BAND_LABELS)
    assert sum(sum(row.values()) for row in counts.values()) == SAMPLE_ROWS
    for name in OUTCOME_CLASSES:
        total = sum(int(counts[band][name]) for band in MARGIN_BAND_LABELS)
        assert total == RECORDED_CONFUSION[name]


def test_marginal_errors_are_counted_against_the_fixed_limit() -> None:
    results = _results()
    marginal = results.margin_analysis["marginal_errors"]

    assert marginal["margin_limit"] == MARGINAL_ERROR_MARGIN == 0.05
    assert marginal["total_false_positives"] == RECORDED_CONFUSION["FP"]
    assert marginal["total_false_negatives"] == RECORDED_CONFUSION["FN"]
    assert 0 <= marginal["marginal_false_positives"] <= marginal["total_false_positives"]
    assert 0 <= marginal["marginal_false_negatives"] <= marginal["total_false_negatives"]


def test_marginal_counts_agree_with_the_narrow_bands() -> None:
    """The 0.05 limit is exactly the first two bands, so the counts must match."""
    results = _results()
    counts = results.margin_analysis["counts_by_band"]
    marginal = results.margin_analysis["marginal_errors"]

    for name, key in (
        (FALSE_POSITIVE, "marginal_false_positives"),
        (FALSE_NEGATIVE, "marginal_false_negatives"),
    ):
        narrow = int(counts[MARGIN_BAND_LABELS[0]][name]) + int(counts[MARGIN_BAND_LABELS[1]][name])
        assert marginal[key] == narrow


def test_marginal_counts_are_computed_from_the_limit() -> None:
    labels = np.array([0, 0, 1, 1])
    probability = np.array([0.52, 0.90, 0.48, 0.10])
    outcomes = classify_outcomes(labels, (probability >= 0.5).astype(int))

    counts = marginal_error_counts(outcomes, np.abs(margins(probability, 0.5)), 0.05)

    assert counts["marginal_false_positives"] == 1
    assert counts["marginal_false_negatives"] == 1
    assert counts["total_false_positives"] == 2
    assert counts["total_false_negatives"] == 2


# --- high-confidence errors are diagnostic only ------------------------------


def test_high_confidence_errors_are_counted_on_each_side() -> None:
    labels = np.array([0, 0, 1, 1])
    probability = np.array([0.75, 0.40, 0.05, 0.30])
    outcomes = classify_outcomes(labels, (probability >= 0.3272694566222328).astype(int))

    confident = high_confidence_errors(probability, outcomes, 0.70, 0.10)

    assert confident.n_high_confidence_false_positives == 1
    assert confident.n_high_confidence_false_negatives == 1
    assert confident.diagnostic_only is True
    assert confident.decision_policy_candidate is False


def test_the_recorded_high_confidence_criteria_are_diagnostic_only() -> None:
    results = _results()
    confident = results.high_confidence_errors

    assert confident["diagnostic_only"] is True
    assert confident["decision_policy_candidate"] is False
    assert confident["false_positive_probability_minimum"] == HIGH_CONFIDENCE_FALSE_POSITIVE_MIN
    assert confident["false_negative_probability_maximum"] == HIGH_CONFIDENCE_FALSE_NEGATIVE_MAX
    assert confident["n_high_confidence_false_positives"] <= confident["n_false_positives"]
    assert confident["n_high_confidence_false_negatives"] <= confident["n_false_negatives"]
    # They are not the frozen threshold and are not offered as one.
    assert confident["false_positive_probability_minimum"] != FROZEN_THRESHOLD
    assert results.frozen_policy["final_threshold"] == FROZEN_THRESHOLD


def test_a_class_with_no_errors_reports_no_share() -> None:
    labels = np.array([0, 1])
    probability = np.array([0.1, 0.9])
    outcomes = classify_outcomes(labels, (probability >= 0.5).astype(int))

    confident = high_confidence_errors(probability, outcomes)

    assert confident.n_false_positives == 0
    assert confident.share_of_false_positives is None


# --- composition uses both denominators --------------------------------------


def test_composition_reports_the_error_share_and_the_eligible_share() -> None:
    frame = pd.DataFrame({"Contract": ["A", "A", "B", "B", "B"]})
    error = np.array([True, False, True, False, False])
    eligible = np.array([True, True, True, True, True])

    rows = {row["group"]: row for row in composition(frame, "Contract", error, eligible)}

    assert rows["A"]["errors"] == 1
    assert rows["A"]["share_of_errors"] == pytest.approx(0.5)
    assert rows["A"]["eligible"] == 2
    assert rows["A"]["share_of_eligible"] == pytest.approx(0.4)
    assert rows["B"]["share_of_errors"] == pytest.approx(0.5)
    assert rows["B"]["share_of_eligible"] == pytest.approx(0.6)


def test_the_recorded_composition_sums_to_the_error_counts() -> None:
    results = _results()

    for error_class, expected in (
        (FALSE_POSITIVE, RECORDED_CONFUSION["FP"]),
        (FALSE_NEGATIVE, RECORDED_CONFUSION["FN"]),
    ):
        for axis, rows in results.error_composition[error_class].items():
            assert sum(int(row["errors"]) for row in rows) == expected, f"{error_class}/{axis}"
            shares = [row["share_of_errors"] for row in rows if row["share_of_errors"]]
            assert sum(shares) == pytest.approx(1.0, abs=1e-4)


def test_the_composition_eligible_base_is_the_right_population() -> None:
    """False positives are conditioned on non-churners, false negatives on churners."""
    results = _results()

    for axis, rows in results.error_composition[FALSE_POSITIVE].items():
        assert sum(int(row["eligible"]) for row in rows) == 1035, axis
    for axis, rows in results.error_composition[FALSE_NEGATIVE].items():
        assert sum(int(row["eligible"]) for row in rows) == 374, axis


# --- probability summaries ---------------------------------------------------


def test_probability_summaries_describe_each_class() -> None:
    labels, probability = _scores()
    outcomes = classify_outcomes(labels, (probability >= 0.5).astype(int))

    summaries = summarise_probabilities(probability, outcomes)

    for name in OUTCOME_CLASSES:
        summary = summaries[name]
        assert summary.count == int((outcomes == name).sum())
        if summary.count:
            assert summary.minimum <= summary.q1 <= summary.median <= summary.q3 <= summary.maximum


def test_the_recorded_summaries_respect_the_decision_boundary() -> None:
    """Flagged classes sit at or above the threshold; unflagged ones below it."""
    results = _results()
    summaries = results.probability_summaries

    for flagged in (FALSE_POSITIVE, TRUE_POSITIVE):
        assert summaries[flagged]["min"] >= FROZEN_THRESHOLD - 1e-6
    for unflagged in (TRUE_NEGATIVE, FALSE_NEGATIVE):
        assert summaries[unflagged]["max"] < FROZEN_THRESHOLD


# --- insufficient cells are reported, not hidden -----------------------------


def test_insufficient_cells_are_listed_with_their_reason() -> None:
    results = _results()

    for entry in results.insufficient_cells:
        assert entry["reasons"]
        assert entry["axis"]
    # Every withheld rate corresponds to a listed cell.
    listed = {(entry["axis"], entry["group"]) for entry in results.insufficient_cells}
    for axis, rows in results.slice_analyses.items():
        for row in rows:
            if not all(
                (
                    row["sufficient_for_group_rates"],
                    row["sufficient_for_recall"],
                    row["sufficient_for_specificity"],
                    row["sufficient_for_precision"],
                )
            ):
                assert (axis, row["group"]) in listed


def test_the_insufficient_helper_explains_each_failure() -> None:
    table = [slice_metrics("axis", "tiny", np.array([1, 0]), np.array([1, 0]))]

    flagged = insufficient_cells([table])

    assert len(flagged) == 1
    assert any("n=2" in reason for reason in flagged[0]["reasons"])


def test_no_category_was_merged_to_clear_a_guardrail() -> None:
    results = _results()

    assert results.guardrails["categories_merged_to_clear_a_threshold"] is False
    assert results.guardrails["fixed_before_interpretation"] is True
    assert results.guardrails["applied_uniformly"] is True
    for axis, rows in results.slice_analyses.items():
        for row in rows:
            assert row["group"] != "Other", axis


# --- nothing was learned, nothing was selected -------------------------------


def test_the_analysis_calls_no_fit(monkeypatch) -> None:
    """Every fit entry point raises, and the real analysis still completes."""
    if not (PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json").is_file():
        pytest.skip("the Phase 9D evaluation has not been run yet")

    attempted: list[str] = []

    def forbid(name: str):
        def guard(*args, **kwargs):
            attempted.append(name)
            raise AssertionError(f"the post-hoc analysis called {name}")

        return guard

    monkeypatch.setattr(Pipeline, "fit", forbid("Pipeline.fit"))
    monkeypatch.setattr(Pipeline, "fit_transform", forbid("Pipeline.fit_transform"))
    monkeypatch.setattr(LogisticRegression, "fit", forbid("LogisticRegression.fit"))

    results, probability, outcomes, tables = run_error_analysis.analyse()

    assert attempted == [], f"the analysis attempted: {attempted}"
    assert probability.size == SAMPLE_ROWS
    assert outcomes.size == SAMPLE_ROWS
    assert results.methodology["fit_calls"] == 0
    assert set(tables) == set(SLICE_AXES) | set(AUDIT_AXES)


def test_the_frozen_artefacts_are_unchanged_by_the_analysis() -> None:
    results = _results()
    provenance = results.provenance

    assert (
        provenance["pipeline_sha256_before_analysis"]
        == provenance["pipeline_sha256_after_analysis"]
    )
    assert provenance["pipeline_unchanged_by_analysis"] is True

    policy = load_decision_policy()
    path = PROJECT_ROOT / policy.artifacts.pipeline_path
    if path.is_file():
        assert file_digest(path) == provenance["pipeline_sha256_after_analysis"]
        assert (
            model_fingerprint(load_pipeline(path))
            == (provenance["model_fingerprint_sha256_after_load"])
        )


def test_no_selection_no_calibration_no_alternative_model() -> None:
    results = _results()
    methodology = results.methodology

    assert results.selection_allowed is False
    assert results.model_change_allowed is False
    assert results.selection_after_analysis is False
    assert results.holdout_already_consumed is True
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
        assert methodology[counter] == 0, counter
    assert methodology["models_loaded"] == 1
    assert methodology["thresholds_scored"] == 1


def test_no_explainability_no_costs_no_causal_claim() -> None:
    """Asserted on flags and on field names, never on a raw substring search.

    The methodological notes legitimately *mention* SHAP, revenue and cost in
    order to disclaim them, so scanning the file's text would flag its own
    disclaimers. What must be absent is a **field carrying such a value**; what
    must be present is the disclaimer itself.
    """
    results = _results()
    payload = json.loads(
        (PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json").read_text(
            encoding="utf-8"
        )
    )

    assert results.methodology["explainability_computed"] is False
    assert results.methodology["financial_costs_assumed"] is False
    assert results.methodology["causal_claims"] is False

    def field_names(node, seen: set[str]) -> set[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                seen.add(str(key).lower())
                field_names(value, seen)
        elif isinstance(node, list):
            for item in node:
                field_names(item, seen)
        return seen

    names = field_names(payload, set())
    for forbidden in (
        "shap",
        "shap_values",
        "cost_fp",
        "cost_fn",
        "revenue",
        "profit",
        "roi",
        "lifetime_value",
        "feature_importance",
        "coefficients",
    ):
        assert forbidden not in names, f"a field named {forbidden!r} was persisted"

    notes = " ".join(results.methodological_notes).lower()
    assert "no explainability" in notes
    assert "no financial quantity" in notes
    assert "no causal claim" in notes


def test_the_demographic_audit_is_not_a_fairness_verdict() -> None:
    results = _results()
    audit = results.demographic_audit

    assert audit["status"] == "descriptive subgroup audit"
    assert audit["is_fairness_assessment"] is False
    assert audit["establishes_bias"] is False
    assert audit["inferential_tests_run"] == 0


# --- privacy -----------------------------------------------------------------


def test_no_identifier_is_persisted() -> None:
    """No customerID, and no per-customer row of any kind, in either artefact."""
    paths = [
        PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json",
        PROJECT_ROOT / "reports" / "error_analysis_report.md",
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("the post-hoc analysis has not been run yet")

    identifier = re.compile(r"\d{4}-[A-Z]{5}")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "customerID" not in text, path.name
        assert not identifier.search(text), path.name

    assert _results().methodology["identifiers_persisted"] is False
    assert _results().evaluation_sample["identifiers_persisted"] is False


def test_the_record_holds_only_aggregates() -> None:
    """No array in the artefact is as long as the sample: nothing is per-row."""
    payload = json.loads(
        (PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json").read_text(
            encoding="utf-8"
        )
    )

    def longest(node) -> int:
        if isinstance(node, list):
            return max([len(node), *[longest(item) for item in node]], default=0)
        if isinstance(node, dict):
            return max([longest(value) for value in node.values()], default=0)
        return 0

    assert longest(payload) < 100, "an array long enough to be per-customer was persisted"


# --- the record --------------------------------------------------------------


def test_the_record_declares_what_it_is() -> None:
    results = _results()

    assert results.schema_version == SCHEMA_VERSION
    assert results.phase == "9E"
    assert results.analysis_type == ANALYSIS_TYPE == "POST_HOC_ERROR_ANALYSIS"
    assert results.post_hoc is True
    assert results.descriptive_only is True


def test_the_record_passes_its_own_invariants() -> None:
    assert invariant_failures(_results()) == []


def test_the_verify_command_is_read_only_and_passes(monkeypatch) -> None:
    if not (PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json").is_file():
        pytest.skip("the post-hoc analysis has not been run yet")

    watched = [
        PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json",
        PROJECT_ROOT / "reports" / "error_analysis_report.md",
        PROJECT_ROOT / "reports" / "experiments" / "holdout_results.json",
    ]
    before = {path: (file_digest(path), path.stat().st_mtime_ns) for path in watched}

    def forbid(*args, **kwargs):
        raise AssertionError("--verify wrote a file")

    monkeypatch.setattr(pathlib.Path, "write_text", forbid)
    monkeypatch.setattr(pathlib.Path, "write_bytes", forbid)
    monkeypatch.setattr(Pipeline, "fit", forbid)

    assert run_error_analysis.main(["--verify"]) == 0

    for path in watched:
        assert (file_digest(path), path.stat().st_mtime_ns) == before[path]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"selection_after_analysis": True}, "selection"),
        ({"model_change_allowed": True}, "model change"),
        ({"holdout_already_consumed": False}, "already consumed"),
    ],
)
def test_the_invariant_checker_catches_a_broken_protocol_flag(
    mutation: dict[str, object], expected: str
) -> None:
    from churn.modeling.error_analysis_results import ErrorAnalysisResults

    payload = json.loads(_results().model_dump_json())
    payload.update(mutation)

    failures = invariant_failures(ErrorAnalysisResults.model_validate(payload))

    assert any(expected in failure for failure in failures)


def test_the_invariant_checker_catches_an_unsupported_rate() -> None:
    from churn.modeling.error_analysis_results import ErrorAnalysisResults

    payload = json.loads(_results().model_dump_json())
    row = payload["slice_analyses"]["Contract"][0]
    row["sufficient_for_recall"] = False
    row["recall"] = 0.9

    failures = invariant_failures(ErrorAnalysisResults.model_validate(payload))

    assert any("recall without support" in failure for failure in failures)


def test_the_invariant_checker_catches_a_broken_partition() -> None:
    from churn.modeling.error_analysis_results import ErrorAnalysisResults

    payload = json.loads(_results().model_dump_json())
    payload["confusion_reproduction"]["derived"]["TP"] += 1

    failures = invariant_failures(ErrorAnalysisResults.model_validate(payload))

    assert any("partition" in failure for failure in failures)


def test_the_record_carries_no_timestamp() -> None:
    path = PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json"
    if not path.is_file():
        pytest.skip("the post-hoc analysis has not been run yet")
    raw = path.read_text(encoding="utf-8")

    assert not [key for key in json.loads(raw) if re.search("time|stamp|generated", key, re.I)]
    assert not re.search(r"\d{4}-\d{2}-\d{2}", raw)


def test_the_upstream_artefacts_are_unchanged() -> None:
    results = _results()
    recorded = results.upstream_artefact_digests

    assert set(recorded) == set(UPSTREAM_ARTEFACTS)
    assert "reports/experiments/holdout_results.json" in recorded
    for artefact, digest in recorded.items():
        assert file_digest(PROJECT_ROOT / artefact) == digest, f"{artefact} changed"


def test_the_new_record_is_not_among_the_artefacts_it_must_not_change() -> None:
    results = _results()

    assert "reports/experiments/error_analysis_results.json" not in (
        results.upstream_artefact_digests
    )


# --- regression guards on methodological language -----------------------------

#: The two Phase 9E artefacts. These guards read their content, not their source,
#: because what must not come back is a *claim in the deliverable*.
_ARTEFACTS = (
    PROJECT_ROOT / "reports" / "error_analysis_report.md",
    PROJECT_ROOT / "reports" / "experiments" / "error_analysis_results.json",
)


def _artefact_text() -> dict[str, str]:
    """Return the two artefacts as flowing text, free of markdown decoration.

    Blockquote markers and emphasis are stripped and whitespace is collapsed
    before any phrase is looked for: a claim broken across two lines of a
    ``>``-quoted block is the same claim, and a guard that missed it because of
    the line wrapping would be checking the formatter rather than the wording.
    """
    if not all(path.is_file() for path in _ARTEFACTS):
        pytest.skip("the post-hoc analysis has not been run yet")

    cleaned: dict[str, str] = {}
    for path in _ARTEFACTS:
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^\s*>\s?", " ", text, flags=re.MULTILINE)
        cleaned[path.name] = re.sub(r"\s+", " ", text).replace("*", "").lower()
    return cleaned


def _is_negated(text: str, index: int, window: int = 90) -> bool:
    """Whether the phrase at ``index`` sits inside an explicit disclaimer."""
    preceding = text[max(0, index - window) : index]
    return any(
        marker in preceding
        for marker in (
            "not ",
            "never ",
            "cannot ",
            "no ",
            "without ",
            "outside the scope",
        )
    )


def test_phase9d_is_never_called_an_unbiased_estimate() -> None:
    """The analyst-exposure limitation forbids the unqualified claim.

    Phase 3 ran on all 7,043 rows, including the ones that later formed the
    evaluation sample. The sample was protected from Phase 4 onwards, but the
    earlier exposure came first, so "unbiased" overstates what the Phase 9D
    figure is. The word may appear only inside its own disclaimer.
    """
    for name, text in _artefact_text().items():
        assert "single unbiased estimate" not in text, name
        assert "the unbiased estimate" not in text, name
        assert "an unbiased estimate for" not in text, name
        for match in re.finditer(r"unbiased", text):
            window = text[max(0, match.start() - 40) : match.start()]
            assert "not described" in window or "not described as" in window, (
                f"{name}: unqualified 'unbiased' at offset {match.start()}"
            )


def test_the_analyst_exposure_limitation_is_stated() -> None:
    """Removing the claim is only half of it; the limitation must be present."""
    text = _artefact_text()

    for name, content in text.items():
        assert "analyst exposure" in content, name
        assert "7,043" in content or "7043" in content, name
        assert "cannot quantify" in content, name
    assert "designated final held-out evaluation" in text["error_analysis_report.md"]


def test_false_negatives_are_not_called_the_class_that_matters_most() -> None:
    """No cost ratio exists in this dataset, so no relative importance is claimed."""
    for name, text in _artefact_text().items():
        # Never present in any form.
        for forbidden in ("matters most", "most important error", "more costly"):
            assert forbidden not in text, f"{name}: {forbidden}"
        # May appear only inside an explicit disclaimer.
        for match in re.finditer(r"costs more than", text):
            assert _is_negated(text, match.start()), (
                f"{name}: unqualified cost comparison at offset {match.start()}"
            )

    report = _artefact_text()["error_analysis_report.md"]
    assert "operationally important" in report
    assert "cannot be established from this dataset" in report
    assert "no business-cost ratio is available" in report


def test_marginal_is_defined_as_boundary_proximate_not_as_a_coin_flip() -> None:
    """A score near 0.327 is near the boundary, not near 50%."""
    for name, text in _artefact_text().items():
        for pattern in ("coin-flip", "coin flip", "near a coin"):
            for match in re.finditer(re.escape(pattern), text):
                assert _is_negated(text, match.start()), (
                    f"{name}: unqualified coin-flip framing at offset {match.start()}"
                )

    report = _artefact_text()["error_analysis_report.md"]
    assert "boundary-proximate" in report
    assert "does not mean the estimated churn probability is close to 50%" in report


def test_the_margin_analysis_claims_no_dichotomy_between_error_kinds() -> None:
    """Both classes stay errors of the frozen decision; margin only measures distance."""
    for name, text in _artefact_text().items():
        for forbidden in (
            "cut fell here",
            "separates the cut",
            "rather than a confident statement",
        ):
            assert forbidden not in text, f"{name}: {forbidden}"

    report = _artefact_text()["error_analysis_report.md"]
    assert "both classes remain errors of the frozen decision" in report
    assert "does not divide the errors into real and incidental ones" in report


def test_no_quantitative_counterfactual_about_moving_the_threshold() -> None:
    """Describing what another threshold would yield is scoring it by implication."""
    for name, text in _artefact_text().items():
        for forbidden in (
            "recovered the marginal",
            "would convert a larger number",
            "would convert",
            "would recover",
            "if the threshold were",
        ):
            assert forbidden not in text, f"{name}: {forbidden}"

    report = _artefact_text()["error_analysis_report.md"]
    assert "boundary proximity is descriptive only" in report
    assert "would reopen decision selection" in report
    assert _results().frozen_policy["alternative_thresholds_scored"] == 0
    assert _results().methodology["alternative_thresholds_scored"] == 0


def test_high_confidence_is_qualified_as_an_extreme_score() -> None:
    """The shorthand stays in the field names; the semantics are made explicit."""
    text = _artefact_text()
    results = _results()

    for name, content in text.items():
        assert "not a validated per-case confidence guarantee" in content, name
        assert "extreme model probability" in content, name
    assert "extreme-score errors" in text["error_analysis_report.md"]

    # The field names are unchanged for compatibility, and the record carries the
    # terminology note that disambiguates them.
    confident = results.high_confidence_errors
    assert "n_high_confidence_false_positives" in confident
    assert "terminology" in confident
    assert "not a validated per-case confidence guarantee" in confident["terminology"].lower()


def test_the_diagnostic_cutoffs_are_not_presented_as_thresholds() -> None:
    results = _results()
    confident = results.high_confidence_errors
    report = _artefact_text()["error_analysis_report.md"]

    assert confident["diagnostic_only"] is True
    assert confident["decision_policy_candidate"] is False
    assert confident["false_positive_probability_minimum"] != FROZEN_THRESHOLD
    assert confident["false_negative_probability_maximum"] != FROZEN_THRESHOLD
    assert "the two cut-offs are not thresholds" in report
    # And the calibration decision is not reopened by them.
    assert results.frozen_policy["calibration_policy"] == "NONE"
    assert "reopen the calibration policy" in report or "nothing here reopens" in report


# --- audits ------------------------------------------------------------------

_PHASE9E_SOURCES = [
    PROJECT_ROOT / "src" / "churn" / "modeling" / "error_analysis.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "error_analysis_results.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "error_analysis_plots.py",
    PROJECT_ROOT / "scripts" / "run_error_analysis.py",
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
    "write_decision_policy",
    "chi2",
    "chi2_contingency",
    "ttest_ind",
    "mannwhitneyu",
    "f_oneway",
    "fisher_exact",
    "shap",
    "permutation_importance",
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


@pytest.mark.parametrize("source", _PHASE9E_SOURCES, ids=lambda path: path.name)
def test_phase9e_code_neither_learns_nor_tests(source) -> None:
    """No fit, no search, no calibrator, no significance test, no explainability."""
    used = _identifiers(source)

    found = sorted(name for name in FORBIDDEN_IDENTIFIERS if name in used)
    assert not found, (
        f"{source.name} references {found}: this phase describes a frozen result and must not "
        "fit, select, calibrate, run an inferential test or compute feature attributions"
    )


@pytest.mark.parametrize("source", _PHASE9E_SOURCES, ids=lambda path: path.name)
def test_phase9e_writes_only_its_own_artefacts(source) -> None:
    """It may not write the Phase 9C or 9D artefacts."""
    used = _identifiers(source)

    for forbidden in ("write_decision_policy", "dump_pipeline"):
        assert forbidden not in used, f"{source.name} writes a frozen artefact"
    # Only the Phase 9E writer is imported, never the earlier ones.
    if source.name == "run_error_analysis.py":
        assert "write_results" in used


def test_the_phase9e_script_writes_only_new_paths() -> None:
    """Its output paths are its own; no frozen artefact appears as a write target."""
    text = (PROJECT_ROOT / "scripts" / "run_error_analysis.py").read_text(encoding="utf-8")

    assert "error_analysis_report.md" in text
    for frozen in (
        "holdout_results.json",
        "holdout_evaluation_report.md",
        "decision_policy.json",
        "model_freeze_report.md",
        "churn_pipeline.joblib",
    ):
        assert f'REPORT_PATH = PROJECT_ROOT / "reports" / "{frozen}"' not in text


def test_the_metric_module_loads_no_data() -> None:
    """It receives frames and arrays; it cannot reach a partition on its own."""
    import churn.modeling.error_analysis as module

    for name in ("load_holdout", "load_training_pool", "load_split", "load_raw_typed"):
        assert not hasattr(module, name)


def test_the_report_generator_reads_no_clock() -> None:
    used = _identifiers(PROJECT_ROOT / "scripts" / "run_error_analysis.py")

    found = sorted(
        name
        for name in ("datetime", "date", "today", "now", "utcnow", "monotonic", "perf_counter")
        if name in used
    )
    assert not found, f"run_error_analysis.py reads wall-clock state via {found}"


def test_the_report_states_its_methodological_status() -> None:
    report = PROJECT_ROOT / "reports" / "error_analysis_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")
    text = report.read_text(encoding="utf-8")

    assert "POST_HOC_ERROR_ANALYSIS" in text
    assert "descriptive" in text.lower()
    assert "cannot support" in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    # Future work is phrased as a hypothesis, never as a change to make now.
    assert "could investigate" in text
    for forbidden in ("we should now modify", "we will change the model", "should be retrained"):
        assert forbidden not in text.lower(), forbidden
