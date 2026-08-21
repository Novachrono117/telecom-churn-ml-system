"""Phase 9B: the selection rule, the nested isolation, and the two decisions.

The leakage property this phase depends on is not "the outer validation fold is
predicted once" — it is that **no row ever contributes to the threshold that is
then applied to it**. That is asserted on row identities and on the probabilities
the search was handed, not on documentation.

The selection rule is checked against a brute-force reimplementation on synthetic
inputs: the fast sorted-pass arithmetic in
:func:`churn.modeling.threshold.sweep_thresholds` has to agree with scikit-learn
applied naively at each threshold, or it is not computing F1.

Both rules — POLICY_F1 and the eligibility gate — are tested on **synthetic**
inputs, every branch, so the branch the real numbers happen to take is not the
only one anybody exercised.
"""

from __future__ import annotations

import ast
import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from churn.config import PROJECT_ROOT
from churn.modeling.calibration_results import load_results as load_calibration_results
from churn.modeling.evaluation import build_splitter
from churn.modeling.metrics import DEFAULT_THRESHOLD
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.modeling.threshold import (
    ACCURACY,
    AVERAGE_PRECISION,
    CONFUSION_NAMES,
    DECISION_METRIC_NAMES,
    DEFAULT_EXPERIMENT,
    DEFAULT_POLICY,
    ELIGIBLE,
    F1,
    F1_POLICY,
    GUARDRAIL_METRIC_NAMES,
    MIN_IMPROVED_FOLDS,
    NESTED_EXPERIMENT,
    NOT_ELIGIBLE,
    PRECISION,
    PREDICTED_POSITIVE_RATE,
    RECALL,
    RECALL_TARGETS,
    ROC_AUC,
    SELECTION_METRIC,
    THRESHOLD_INNER_N_SPLITS,
    TIE_TOLERANCE,
    TOP_K_FRACTIONS,
    PairedThresholdDelta,
    RankingInvarianceError,
    build_frozen_pipeline,
    build_threshold_inner_splitter,
    candidate_thresholds,
    compare,
    decision_metrics,
    eligibility,
    evaluate_at_threshold,
    operating_curve,
    out_of_fold_probabilities,
    pair,
    recall_scenarios,
    run_threshold_nested_cv,
    select_f1_threshold,
    select_threshold_policy,
    sweep_thresholds,
    threshold_stability,
    top_k_analysis,
    verify_ranking_invariance,
)
from churn.modeling.threshold_results import (
    EXPECTED_CALIBRATION_POLICY,
    CalibrationPolicyMismatchError,
    verify_calibration_policy,
    verify_probability_source,
    verify_training_pool_probabilities,
)
from churn.modeling.tuning import build_modern_logistic
from churn.preprocessing.contracts import FEATURE_COLUMNS, build_feature_matrix
from churn.preprocessing.target import encode_target


@pytest.fixture(scope="module")
def synthetic_xy(comparison_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic feature matrix and target, for isolation tests."""
    return build_feature_matrix(comparison_frame), encode_target(comparison_frame["Churn"])


def _scores(n: int = 240, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Labels correlated with a continuous score, so F1 has a real maximum."""
    rng = np.random.default_rng(seed)
    probability = rng.random(n)
    labels = (rng.random(n) < probability * 0.8 + 0.05).astype(int)
    return labels, probability


# --- the frozen probability source --------------------------------------------


def test_the_phase_thresholds_the_frozen_phase8b_candidate() -> None:
    """The pipeline built here is the Phase 8B candidate, uncalibrated."""
    pipeline = build_frozen_pipeline()
    classifier = pipeline.named_steps[CLASSIFIER_STEP]

    assert list(pipeline.named_steps) == [PREPROCESSOR_STEP, CLASSIFIER_STEP]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.get_params() == build_modern_logistic().get_params()
    assert classifier.class_weight is None


def test_no_calibration_wrapper_sits_between_the_model_and_the_threshold() -> None:
    """The threshold acts on predict_proba directly, as Phase 9A decided."""
    pipeline = build_frozen_pipeline()

    assert not hasattr(pipeline.named_steps[CLASSIFIER_STEP], "calibrated_classifiers_")
    assert type(pipeline.named_steps[CLASSIFIER_STEP]).__name__ == "LogisticRegression"


def test_the_calibration_policy_gate_accepts_none_and_rejects_anything_else() -> None:
    class _Selection:
        def __init__(self, policy: str) -> None:
            self.selected_policy = policy

    class _Results:
        def __init__(self, policy: str) -> None:
            self.selection = _Selection(policy)

    assert verify_calibration_policy(_Results("NONE")) == EXPECTED_CALIBRATION_POLICY
    with pytest.raises(CalibrationPolicyMismatchError, match="C1"):
        verify_calibration_policy(_Results("C1"))


def test_the_real_phase9a_record_still_says_none() -> None:
    """The gate against the artefact actually in the repository."""
    try:
        calibration = load_calibration_results()
    except FileNotFoundError:
        pytest.skip("the Phase 9A artefact has not been generated yet")

    assert verify_calibration_policy(calibration) == EXPECTED_CALIBRATION_POLICY


# --- candidate thresholds -----------------------------------------------------


def test_candidates_are_every_distinct_probability_plus_the_default() -> None:
    probability = np.array([0.1, 0.4, 0.4, 0.9])

    candidates = candidate_thresholds(probability)

    assert candidates.tolist() == [0.1, 0.4, 0.5, 0.9]


def test_candidates_are_deterministic_and_sorted() -> None:
    probability = np.array([0.9, 0.1, 0.4, 0.4])

    first = candidate_thresholds(probability)
    second = candidate_thresholds(probability[::-1])

    assert np.array_equal(first, second)
    assert np.array_equal(first, np.sort(first))


def test_the_default_threshold_is_always_a_candidate() -> None:
    """Even when no observed probability is anywhere near it."""
    assert DEFAULT_THRESHOLD in candidate_thresholds(np.array([0.01, 0.02, 0.03])).tolist()
    assert DEFAULT_THRESHOLD in candidate_thresholds(np.array([0.97, 0.98])).tolist()
    assert DEFAULT_THRESHOLD in candidate_thresholds(np.array([0.5])).tolist()


def test_probabilities_are_not_rounded_before_the_search() -> None:
    """Two probabilities differing in the twelfth decimal stay two candidates."""
    probability = np.array([0.300000000001, 0.300000000002])

    candidates = candidate_thresholds(probability)

    assert candidates.size == 3
    assert candidates.tolist() == sorted([0.300000000001, 0.300000000002, DEFAULT_THRESHOLD])


# --- the decision rule --------------------------------------------------------


def test_the_rule_is_probability_greater_or_equal_threshold() -> None:
    """A probability exactly at the threshold is predicted positive."""
    labels = np.array([0, 1])
    probability = np.array([0.4, 0.5])

    point = evaluate_at_threshold(labels, probability, 0.5)

    assert point.true_positives == 1
    assert point.false_positives == 0
    assert point.predicted_positive_rate == pytest.approx(0.5)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_sweep_matches_scikit_learn_at_every_candidate(seed: int) -> None:
    """The fast sorted pass against a naive per-threshold reimplementation."""
    labels, probability = _scores(seed=seed)
    sweep = sweep_thresholds(labels, probability)

    for index in range(0, len(sweep), 11):
        threshold = float(sweep.thresholds[index])
        predicted = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()

        assert (
            int(sweep.true_negatives[index]),
            int(sweep.false_positives[index]),
            int(sweep.false_negatives[index]),
            int(sweep.true_positives[index]),
        ) == (tn, fp, fn, tp)
        assert sweep.f1[index] == pytest.approx(f1_score(labels, predicted, zero_division=0))
        assert sweep.precision[index] == pytest.approx(
            precision_score(labels, predicted, zero_division=0)
        )
        assert sweep.recall[index] == pytest.approx(
            recall_score(labels, predicted, zero_division=0)
        )
        assert sweep.accuracy[index] == pytest.approx(accuracy_score(labels, predicted))


def test_the_sweep_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="one probability per label"):
        sweep_thresholds(np.array([0, 1, 1]), np.array([0.2, 0.8]))
    with pytest.raises(ValueError, match="at least one row"):
        sweep_thresholds(np.array([]), np.array([]))


# --- POLICY_F1 ----------------------------------------------------------------


def test_f1_is_the_only_selection_metric() -> None:
    """A threshold beating the F1-max on accuracy is not selected."""
    labels, probability = _scores()
    selection = select_f1_threshold(labels, probability)
    sweep = sweep_thresholds(labels, probability)

    assert SELECTION_METRIC == F1
    assert selection.operating_point.f1 == pytest.approx(float(sweep.f1.max()))
    # Accuracy peaks elsewhere here, and the selection ignored it.
    assert float(sweep.accuracy.max()) > selection.operating_point.accuracy


def test_the_selected_threshold_maximises_f1_over_the_candidates() -> None:
    labels, probability = _scores()

    selection = select_f1_threshold(labels, probability)

    for threshold in candidate_thresholds(probability):
        point = evaluate_at_threshold(labels, probability, threshold)
        assert point.f1 <= selection.operating_point.f1 + TIE_TOLERANCE


def test_the_search_reports_the_tolerance_it_used() -> None:
    labels, probability = _scores()

    selection = select_f1_threshold(labels, probability)

    assert selection.tie_tolerance == TIE_TOLERANCE
    assert selection.n_tied_within_tolerance >= 1
    assert selection.default_in_candidates is True
    assert selection.policy == F1_POLICY


#: Eight rows, three churners, built so that exactly two cuts reach the maximal
#: F1 of 2/3: 0.15 (flags six customers, catching all three churners) and 0.45
#: (flags three, catching two). Every other cut is strictly worse, so only the
#: tie-break can decide, and the two candidates sit 0.35 and 0.05 from the
#: default.
_TIE_LABELS = np.array([0, 0, 1, 0, 0, 1, 1, 0])
_TIE_PROBABILITY = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.45, 0.60, 0.70])

#: Five rows, two churners, built so that exactly two cuts reach the maximal F1
#: of 2/3 — 0.3 and 0.7 — which are **equidistant** from the default, and the
#: default itself scores strictly lower. Only the second tie-break can decide.
_EQUIDISTANT_LABELS = np.array([0, 1, 0, 0, 1])
_EQUIDISTANT_PROBABILITY = np.array([0.10, 0.30, 0.40, 0.55, 0.70])


def _tied_thresholds(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    sweep = sweep_thresholds(labels, probability)
    return sweep.thresholds[sweep.f1 >= float(sweep.f1.max()) - TIE_TOLERANCE]


def test_a_tie_in_f1_prefers_the_threshold_closest_to_the_default() -> None:
    """Two candidates reach the same maximal F1; the nearer one to 0.5 wins."""
    tied = _tied_thresholds(_TIE_LABELS, _TIE_PROBABILITY)
    assert tied.tolist() == pytest.approx([0.15, 0.45])

    selection = select_f1_threshold(_TIE_LABELS, _TIE_PROBABILITY)

    assert selection.threshold == pytest.approx(0.45)
    assert selection.operating_point.f1 == pytest.approx(2 / 3)
    assert selection.n_tied_within_tolerance == 2


def test_an_exact_distance_tie_prefers_the_larger_threshold() -> None:
    """Two tied candidates equidistant from 0.5: the larger one is selected."""
    tied = _tied_thresholds(_EQUIDISTANT_LABELS, _EQUIDISTANT_PROBABILITY)
    assert tied.tolist() == pytest.approx([0.30, 0.70])
    assert abs(0.30 - DEFAULT_THRESHOLD) == pytest.approx(abs(0.70 - DEFAULT_THRESHOLD))
    assert evaluate_at_threshold(
        _EQUIDISTANT_LABELS, _EQUIDISTANT_PROBABILITY, DEFAULT_THRESHOLD
    ).f1 < float(sweep_thresholds(_EQUIDISTANT_LABELS, _EQUIDISTANT_PROBABILITY).f1.max())

    selection = select_f1_threshold(_EQUIDISTANT_LABELS, _EQUIDISTANT_PROBABILITY)

    assert selection.threshold == pytest.approx(0.70)
    assert selection.n_tied_within_tolerance == 2


def test_the_tie_tolerance_is_what_makes_two_candidates_tie() -> None:
    """With a zero tolerance the same input still resolves deterministically."""
    strict = select_f1_threshold(_TIE_LABELS, _TIE_PROBABILITY, tie_tolerance=0.0)

    assert strict.tie_tolerance == 0.0
    assert strict.threshold == pytest.approx(0.45)


def test_the_selection_is_deterministic_across_input_order() -> None:
    labels, probability = _scores()
    order = np.argsort(probability)

    first = select_f1_threshold(labels, probability)
    second = select_f1_threshold(labels[order], probability[order])

    assert first.threshold == second.threshold
    assert first.operating_point.f1 == pytest.approx(second.operating_point.f1)


def test_the_selection_never_reads_a_holdout_partition() -> None:
    """The search is a pure function of the two arrays it is handed.

    It loads nothing, so it has no way to reach a holdout row even by accident,
    and the module it lives in does not import the partition loader at all.
    """
    labels, probability = _scores()

    selection = select_f1_threshold(labels, probability)
    repeated = select_f1_threshold(labels.copy(), probability.copy())

    assert selection.threshold == repeated.threshold
    assert selection.n_candidates == candidate_thresholds(probability).size

    import churn.modeling.threshold as module

    assert not hasattr(module, "load_holdout")
    assert not hasattr(module, "load_training_pool")
    assert not hasattr(module, "load_split")


# --- nested isolation ---------------------------------------------------------


@pytest.fixture(scope="module")
def nested(synthetic_xy: tuple[pd.DataFrame, pd.Series]):
    """One nested run on the synthetic frame, shared by the isolation tests."""
    features, target = synthetic_xy
    return run_threshold_nested_cv(
        build_frozen_pipeline(),
        features,
        target,
        StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
    )


def test_every_row_receives_exactly_one_outer_prediction(
    nested, synthetic_xy: tuple[pd.DataFrame, pd.Series]
) -> None:
    _, target = synthetic_xy

    assert nested.oof_probability.shape == (len(target),)
    assert not np.isnan(nested.oof_probability).any()
    assert sum(fold.n_validation for fold in nested.default_run.folds) == len(target)


def test_both_arms_share_the_outer_folds(nested) -> None:
    default_shape = [(f.fold, f.n_train, f.n_validation) for f in nested.default_run.folds]
    selected_shape = [(f.fold, f.n_train, f.n_validation) for f in nested.selected_run.folds]

    assert default_shape == selected_shape


def test_the_default_arm_uses_zero_point_five_in_every_fold(nested) -> None:
    assert [fold.threshold for fold in nested.default_run.folds] == [DEFAULT_THRESHOLD] * len(
        nested.default_run.folds
    )
    assert nested.default_run.policy == DEFAULT_POLICY
    assert nested.selected_run.policy == F1_POLICY


def test_the_two_arms_report_identical_ranking_metrics(nested) -> None:
    """Not close — identical. A threshold cannot move AP or ROC-AUC."""
    for default_fold, selected_fold in zip(
        nested.default_run.folds, nested.selected_run.folds, strict=True
    ):
        for metric in GUARDRAIL_METRIC_NAMES:
            assert default_fold.metrics.value(metric) == selected_fold.metrics.value(metric)

    assert verify_ranking_invariance(nested) == 0.0


def test_the_invariance_gate_fails_when_the_arms_diverge(nested) -> None:
    """A drifted arm must abort the run instead of being reported."""
    import dataclasses

    broken_fold = dataclasses.replace(
        nested.selected_run.folds[0],
        metrics=dataclasses.replace(
            nested.selected_run.folds[0].metrics,
            average_precision=nested.selected_run.folds[0].metrics.average_precision + 1e-6,
        ),
    )
    broken_run = dataclasses.replace(
        nested.selected_run,
        folds=(broken_fold, *nested.selected_run.folds[1:]),
    )
    broken = dataclasses.replace(nested, selected_run=broken_run)

    with pytest.raises(RankingInvarianceError, match="same outer probabilities"):
        verify_ranking_invariance(broken)


def test_the_outer_validation_fold_never_enters_the_inner_out_of_fold(
    synthetic_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Asserted on row identities, not on documentation.

    The inner splitter is only ever given the outer training subframe, so the
    positions it can reach are a subset of the outer training positions. The
    property is proved by reconstructing the same subframes the run built.
    """
    features, target = synthetic_xy
    y = np.asarray(target)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for train_index, validation_index in outer.split(features, y):
        train_features = features.iloc[train_index]
        reachable = set(train_features.index)

        assert reachable.isdisjoint(set(features.iloc[validation_index].index))
        for inner_train, inner_validation in inner.split(train_features, y[train_index]):
            inner_rows = set(train_features.iloc[inner_train].index) | set(
                train_features.iloc[inner_validation].index
            )
            assert inner_rows <= reachable


def test_the_threshold_of_a_fold_is_chosen_only_from_inner_out_of_fold_scores(
    synthetic_xy: tuple[pd.DataFrame, pd.Series], nested
) -> None:
    """Reproduce each fold's threshold from the inner OOF alone."""
    features, target = synthetic_xy
    y = np.asarray(target)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for (train_index, _), selection in zip(
        outer.split(features, y), nested.selections, strict=True
    ):
        inner_oof, _ = out_of_fold_probabilities(
            build_frozen_pipeline(), features.iloc[train_index], y[train_index], inner
        )
        expected = select_f1_threshold(y[train_index], inner_oof, DEFAULT_THRESHOLD)

        assert selection.threshold == pytest.approx(expected.threshold)
        assert selection.n_outer_train == len(train_index)


def test_the_selected_threshold_is_applied_without_readjustment(
    synthetic_xy: tuple[pd.DataFrame, pd.Series], nested
) -> None:
    """The outer-fold metrics are the fold's threshold applied as-is."""
    features, target = synthetic_xy
    y = np.asarray(target)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for (_, validation_index), fold in zip(
        outer.split(features, y), nested.selected_run.folds, strict=True
    ):
        probability = nested.oof_probability[validation_index]
        expected = evaluate_at_threshold(y[validation_index], probability, fold.threshold)

        assert fold.metrics.operating_point == expected


def test_no_row_influences_the_threshold_that_judges_it(nested) -> None:
    """Each outer fold's threshold came from a search that could not see it."""
    thresholds = {selection.fold: selection.threshold for selection in nested.selections}

    for fold in nested.selected_run.folds:
        assert fold.threshold == thresholds[fold.fold]
    # The five searches were independent: at least two folds disagree, which
    # could not happen if one pooled search had produced them all.
    assert len({round(value, 12) for value in thresholds.values()}) >= 2


def test_the_inner_splitter_is_the_declared_one() -> None:
    inner = build_threshold_inner_splitter()

    assert inner.get_n_splits() == THRESHOLD_INNER_N_SPLITS
    assert inner.shuffle is True
    assert inner.random_state == 42


def test_the_outer_splitter_is_the_frozen_project_partition() -> None:
    outer = build_splitter()

    assert outer.get_n_splits() == 5
    assert outer.shuffle is True
    assert outer.random_state == 42


def test_paired_comparison_refuses_unpaired_runs(nested) -> None:
    import dataclasses

    shortened = dataclasses.replace(nested.default_run, folds=nested.default_run.folds[:-1])

    with pytest.raises(ValueError, match="not be paired"):
        compare(nested.selected_run, shortened)


def test_paired_deltas_cover_every_decision_metric(nested) -> None:
    pair(nested.selected_run, nested.default_run)
    deltas = nested.selected_run.deltas[DEFAULT_EXPERIMENT]

    assert set(deltas) == set(DECISION_METRIC_NAMES)
    for metric, delta in deltas.items():
        assert delta.reference == DEFAULT_EXPERIMENT
        assert len(delta.deltas) == len(nested.default_run.folds)
        assert delta.metric == metric


# --- eligibility --------------------------------------------------------------


def _delta(values: list[float]) -> PairedThresholdDelta:
    return PairedThresholdDelta(metric=F1, reference=DEFAULT_EXPERIMENT, deltas=tuple(values))


def test_five_strict_improvements_are_eligible() -> None:
    status, rationale = eligibility(_delta([0.01, 0.02, 0.03, 0.04, 0.05]), 5)

    assert status == ELIGIBLE
    assert "5 of 5" in rationale


def test_four_strict_improvements_are_eligible() -> None:
    status, _ = eligibility(_delta([0.05, 0.05, 0.05, 0.05, -0.01]), 5)

    assert status == ELIGIBLE
    assert MIN_IMPROVED_FOLDS == 4


def test_three_strict_improvements_are_not_eligible() -> None:
    status, rationale = eligibility(_delta([0.10, 0.10, 0.10, -0.01, -0.01]), 5)

    assert status == NOT_ELIGIBLE
    assert "3 of 5" in rationale


def test_three_improvements_and_one_tie_are_not_eligible() -> None:
    """A tie is not a win: it is exactly what a no-op procedure produces."""
    delta = _delta([0.10, 0.10, 0.10, 0.0, -0.01])

    status, _ = eligibility(delta, 5)

    assert delta.n_higher == 3
    assert delta.n_tied == 1
    assert status == NOT_ELIGIBLE


def test_a_non_positive_mean_is_not_eligible_even_with_four_improvements() -> None:
    status, _ = eligibility(_delta([0.001, 0.001, 0.001, 0.001, -0.010]), 5)

    assert status == NOT_ELIGIBLE


def test_a_zero_mean_is_not_eligible() -> None:
    status, _ = eligibility(_delta([0.01, 0.01, 0.01, 0.01, -0.04]), 5)

    assert status == NOT_ELIGIBLE


def test_no_gain_at_all_falls_back_to_the_default_policy() -> None:
    status, _ = eligibility(_delta([0.0, 0.0, 0.0, 0.0, 0.0]), 5)
    policy, rationale = select_threshold_policy(status)

    assert status == NOT_ELIGIBLE
    assert policy == DEFAULT_POLICY
    assert "0.5" in rationale


def test_an_eligible_procedure_selects_the_f1_policy() -> None:
    policy, rationale = select_threshold_policy(ELIGIBLE)

    assert policy == F1_POLICY
    assert "whole training pool" in rationale


# --- descriptive tables cannot decide -----------------------------------------


def test_recall_scenarios_are_descriptive_and_change_no_policy() -> None:
    labels, probability = _scores()
    selection = select_f1_threshold(labels, probability)

    scenarios = recall_scenarios(labels, probability)
    after = select_f1_threshold(labels, probability)

    assert len(scenarios) == len(RECALL_TARGETS)
    assert after.threshold == selection.threshold
    for scenario in scenarios:
        if scenario.attainable:
            assert scenario.operating_point.recall >= scenario.target_recall


def test_a_recall_scenario_returns_the_most_precise_threshold_reaching_it() -> None:
    labels, probability = _scores()

    scenario = recall_scenarios(labels, probability, [0.7])[0]
    sweep = sweep_thresholds(labels, probability)
    reaching = sweep.thresholds[sweep.recall >= 0.7]

    assert scenario.operating_point.threshold == pytest.approx(float(reaching.max()))


def test_an_unreachable_recall_target_is_reported_as_unattainable() -> None:
    labels, probability = _scores()

    scenario = recall_scenarios(labels, probability, [1.5])[0]

    assert scenario.attainable is False
    assert scenario.operating_point is None


def test_top_k_orders_by_descending_probability_only() -> None:
    labels = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])
    probability = np.array([0.05, 0.95, 0.10, 0.90, 0.85, 0.20, 0.15, 0.80, 0.30, 0.25])

    records = top_k_analysis(labels, probability, [0.4])

    assert records[0].n_contacted == 4
    assert records[0].churners_captured == 4
    assert records[0].precision == pytest.approx(1.0)


def test_top_k_does_not_change_the_selected_threshold() -> None:
    labels, probability = _scores()
    before = select_f1_threshold(labels, probability)

    top_k_analysis(labels, probability)
    after = select_f1_threshold(labels, probability)

    assert after.threshold == before.threshold
    assert len(top_k_analysis(labels, probability)) == len(TOP_K_FRACTIONS)


def test_lift_uses_only_the_training_pool_prevalence() -> None:
    labels, probability = _scores()
    prevalence = float(labels.mean())

    for record in top_k_analysis(labels, probability):
        assert record.lift == pytest.approx(record.precision / prevalence)


def test_an_explicit_prevalence_overrides_the_default_base_rate() -> None:
    labels, probability = _scores()

    records = top_k_analysis(labels, probability, [0.1], prevalence=0.5)

    assert records[0].lift == pytest.approx(records[0].precision / 0.5)


def test_a_budget_rounding_down_to_nobody_reports_no_cut() -> None:
    """An empty contact list has no probability at the cut, and says so."""
    labels, probability = _scores(n=10)

    record = top_k_analysis(labels, probability, [0.05])[0]

    assert record.n_contacted == 0
    assert record.probability_at_cut is None
    assert record.precision == 0.0


def test_top_k_rejects_a_fraction_outside_the_unit_interval() -> None:
    labels, probability = _scores()

    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        top_k_analysis(labels, probability, [1.5])


def test_no_financial_cost_is_invented_anywhere_in_the_module() -> None:
    """The module has no cost constant to accidentally use."""
    source = (PROJECT_ROOT / "src" / "churn" / "modeling" / "threshold.py").read_text(
        encoding="utf-8"
    )

    for forbidden in ("cost_fn", "cost_fp", "COST_FN", "COST_FP", "revenue", "churn_value"):
        assert forbidden not in source


def test_the_operating_curve_is_a_reporting_grid_not_a_search_space() -> None:
    labels, probability = _scores()

    curve = operating_curve(labels, probability, 11)
    selection = select_f1_threshold(labels, probability)

    assert len(curve) == 11
    assert curve.thresholds[0] == 0.0
    assert curve.thresholds[-1] == 1.0
    # The selected threshold is not constrained to the grid.
    assert selection.n_candidates > len(curve)


# --- stability ----------------------------------------------------------------


def test_stability_describes_the_selected_thresholds(nested) -> None:
    stability = threshold_stability(nested.selections)
    values = np.array([selection.threshold for selection in nested.selections])

    assert stability.thresholds == tuple(values)
    assert stability.mean == pytest.approx(float(values.mean()))
    assert stability.median == pytest.approx(float(np.median(values)))
    assert stability.std == pytest.approx(float(values.std(ddof=1)))
    assert stability.minimum == pytest.approx(float(values.min()))
    assert stability.maximum == pytest.approx(float(values.max()))
    assert stability.spread == pytest.approx(float(values.max() - values.min()))


# --- reproduction gates -------------------------------------------------------


def test_the_training_pool_identity_gate_accepts_identical_vectors() -> None:
    vector = np.linspace(0.0, 1.0, 50)

    check = verify_training_pool_probabilities(vector, vector.copy())

    assert check.reproduced is True
    assert check.max_absolute_difference == 0.0


def test_the_training_pool_identity_gate_rejects_a_drifted_vector() -> None:
    from churn.modeling.comparison_results import ReproductionError

    vector = np.linspace(0.0, 1.0, 50)
    drifted = vector.copy()
    drifted[3] += 1e-6

    with pytest.raises(ReproductionError, match="same computation"):
        verify_training_pool_probabilities(vector, drifted)


def test_the_probability_source_gate_rejects_a_drifted_reference(nested) -> None:
    from churn.modeling.comparison_results import ReproductionError

    class _Fold:
        def __init__(self, metrics: dict[str, float]) -> None:
            self.metrics = metrics

    class _Policy:
        experiment = "C0"

        def __init__(self, folds: list[_Fold]) -> None:
            self.folds = folds

    class _Results:
        schema_version = 1

        def __init__(self, policies: list[_Policy]) -> None:
            self.policies = policies

    matching = _Results(
        [
            _Policy(
                [
                    _Fold(
                        {
                            metric: round(fold.metrics.value(metric), 6)
                            for metric in GUARDRAIL_METRIC_NAMES
                        }
                    )
                    for fold in nested.default_run.folds
                ]
            )
        ]
    )
    assert verify_probability_source(nested.default_run, matching).reproduced is True

    matching.policies[0].folds[0].metrics[AVERAGE_PRECISION] += 1e-3
    with pytest.raises(ReproductionError, match="drifted"):
        verify_probability_source(nested.default_run, matching)


def test_decision_metrics_carry_both_groups() -> None:
    labels, probability = _scores()

    metrics = decision_metrics(labels, probability, 0.5)

    for name in DECISION_METRIC_NAMES + CONFUSION_NAMES + GUARDRAIL_METRIC_NAMES:
        assert name in metrics.as_dict()
    assert metrics.value(ROC_AUC) == pytest.approx(metrics.roc_auc)
    assert metrics.value(PRECISION) == pytest.approx(metrics.operating_point.precision)


# --- audits -------------------------------------------------------------------

_PHASE9B_SOURCES = [
    PROJECT_ROOT / "src" / "churn" / "modeling" / "threshold.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "threshold_results.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "threshold_plots.py",
    PROJECT_ROOT / "scripts" / "run_threshold.py",
]

FORBIDDEN_IDENTIFIERS = (
    "TunedThresholdClassifierCV",
    "CalibratedClassifierCV",
    "GridSearchCV",
    "RandomizedSearchCV",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "build_hist_gradient_boosting",
    "build_hgb_native",
    "build_native_pipeline",
    "build_calibrated_estimator",
    "CONTRACT_TENURE",
    "XGBClassifier",
    "LGBMClassifier",
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


@pytest.mark.parametrize("source", _PHASE9B_SOURCES, ids=lambda path: path.name)
def test_phase9b_code_never_reaches_the_holdout(source) -> None:
    used = _identifiers(source)

    assert "load_holdout" not in used, f"{source.name} imports or calls the holdout loader"
    assert "holdout" not in used, f"{source.name} reads a holdout partition"


@pytest.mark.parametrize("source", _PHASE9B_SOURCES, ids=lambda path: path.name)
def test_phase9b_code_stays_inside_its_scope(source) -> None:
    used = _identifiers(source)

    found = sorted(name for name in FORBIDDEN_IDENTIFIERS if name in used)
    assert not found, (
        f"{source.name} references {found}: this phase does not search hyperparameters, calibrate "
        "a probability, reopen a model family, use an engineered feature, or delegate threshold "
        "selection to a library tuner"
    )


@pytest.mark.parametrize("source", _PHASE9B_SOURCES, ids=lambda path: path.name)
def test_no_phase9b_call_sets_class_weight(source) -> None:
    """class_weight is a frozen input; passing it anywhere would reopen the fit."""
    tree = ast.parse(source.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "class_weight":
                continue
            assert isinstance(keyword.value, ast.Constant) and keyword.value.value is None, (
                f"{source.name} passes a non-None class_weight; it is frozen at None and "
                "cost-sensitive training is out of scope"
            )


@pytest.mark.parametrize("source", _PHASE9B_SOURCES, ids=lambda path: path.name)
def test_only_the_script_loads_data_and_only_the_training_pool(source) -> None:
    used = _identifiers(source)

    if source.name == "run_threshold.py":
        assert "load_training_pool" in used
        assert "load_split" not in used
        return
    assert "load_training_pool" not in used
    assert "load_raw_typed" not in used


def test_the_report_generator_reads_no_clock() -> None:
    used = _identifiers(PROJECT_ROOT / "scripts" / "run_threshold.py")

    found = sorted(
        name
        for name in ("datetime", "date", "today", "now", "utcnow", "monotonic", "perf_counter")
        if name in used
    )
    assert not found, f"run_threshold.py reads wall-clock state via {found}"


# --- generated artefacts ------------------------------------------------------


def _results_path():
    return PROJECT_ROOT / "reports" / "experiments" / "threshold_results.json"


def _payload():
    path = _results_path()
    if not path.is_file():
        pytest.skip("the artefact has not been generated yet")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_artefact_carries_no_timestamp_and_no_holdout() -> None:
    raw = _results_path().read_text(encoding="utf-8") if _results_path().is_file() else None
    if raw is None:
        pytest.skip("the artefact has not been generated yet")
    payload = json.loads(raw)

    assert not [key for key in payload if re.search("time|stamp|generated", key, re.I)]
    assert not re.search(r"\d{4}-\d{2}-\d{2}", raw)
    assert "load_holdout" not in raw
    assert payload["holdout_touched"] is False


def test_the_artefact_records_the_frozen_inputs_unchanged() -> None:
    payload = _payload()

    assert payload["calibration_policy"] == EXPECTED_CALIBRATION_POLICY
    assert payload["class_weight"] is None
    assert payload["engineered_features_used"] == []
    assert payload["frozen_estimator"]["reopened_here"] is False
    assert payload["frozen_estimator"]["hyperparameters"]["class_weight"] is None
    assert payload["feature_set"]["n_features"] == len(FEATURE_COLUMNS)


def test_the_artefact_records_the_pre_registered_rule_and_its_tie_break() -> None:
    payload = _payload()

    assert payload["threshold_policy"]["objective"] == SELECTION_METRIC
    assert payload["threshold_policy"]["pre_registered"] is True
    assert payload["threshold_policy"]["changed_after_seeing_results"] is False
    assert "research operating point" in payload["threshold_policy"]["scope_statement"]
    assert len(payload["tie_breaking_rule"]) == 3
    assert payload["eligibility"]["minimum_gain_cutoff"] is None
    assert payload["eligibility"]["rule_fixed_before_results"] is True


def test_the_artefact_invents_no_cost() -> None:
    payload = _payload()
    costs = payload["cost_framework"]

    assert costs["cost_false_negative"] is None
    assert costs["cost_false_positive"] is None
    assert costs["optimal_threshold_computed"] is False


def test_the_descriptive_tables_are_marked_non_decisional() -> None:
    payload = _payload()

    assert payload["hypothetical_recall_scenarios"]["decisional"] is False
    assert payload["top_k_capacity_analysis"]["decisional"] is False
    assert "hypothetical" in payload["hypothetical_recall_scenarios"]["status"]
    assert "hypothetical" in payload["top_k_capacity_analysis"]["status"]


def test_the_artefact_reports_identical_ranking_metrics_for_both_arms() -> None:
    payload = _payload()
    policies = {record["experiment"]: record for record in payload["policies"]}

    assert payload["ranking_invariance"]["holds"] is True
    assert payload["ranking_invariance"]["max_absolute_difference"] == 0.0
    for d0_fold, d1_fold in zip(
        policies[DEFAULT_EXPERIMENT]["folds"],
        policies[NESTED_EXPERIMENT]["folds"],
        strict=True,
    ):
        for metric in GUARDRAIL_METRIC_NAMES:
            assert d0_fold["metrics"][metric] == d1_fold["metrics"][metric]


def test_the_artefact_keeps_the_default_arm_at_zero_point_five() -> None:
    payload = _payload()
    policies = {record["experiment"]: record for record in payload["policies"]}

    assert payload["default_threshold"] == DEFAULT_THRESHOLD
    for fold in policies[DEFAULT_EXPERIMENT]["folds"]:
        assert fold["threshold"] == DEFAULT_THRESHOLD


def test_the_frozen_threshold_matches_the_selected_policy() -> None:
    payload = _payload()
    selection = payload["selection"]

    if selection["selected_threshold_policy"] == DEFAULT_POLICY:
        assert selection["final_threshold"] == DEFAULT_THRESHOLD
        assert selection["f1_max_adopted"] is False
        return
    assert selection["selected_threshold_policy"] == F1_POLICY
    assert payload["eligibility"]["status"] == ELIGIBLE
    assert (
        selection["final_threshold"]
        == (selection["f1_max_on_full_training_out_of_fold"]["threshold"])
    )


def test_the_operating_point_is_labelled_as_selection_not_performance() -> None:
    payload = _payload()
    label = payload["selection"]["operating_point_label"]

    assert "NOT final performance" in label
    assert "optimistically biased" in label


def test_the_artefact_records_every_metric_the_protocol_requires() -> None:
    payload = _payload()
    fold = payload["policies"][0]["folds"][0]

    for metric in (PRECISION, RECALL, F1, ACCURACY, PREDICTED_POSITIVE_RATE):
        assert metric in fold["metrics"]
    for metric in GUARDRAIL_METRIC_NAMES:
        assert metric in fold["metrics"]
    for cell in CONFUSION_NAMES:
        assert cell in fold["confusion"]


def test_the_artefact_records_five_outer_thresholds() -> None:
    payload = _payload()

    assert len(payload["outer_thresholds"]) == payload["outer_cross_validation"]["n_splits"]
    assert len(payload["threshold_stability"]["per_fold"]) == len(payload["outer_thresholds"])
    for record in payload["outer_thresholds"]:
        assert record["default_in_candidates"] is True
        assert record["n_inner_splits"] == THRESHOLD_INNER_N_SPLITS


def test_the_report_carries_no_generation_date() -> None:
    report = PROJECT_ROOT / "reports" / "threshold_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")

    text = report.read_text(encoding="utf-8")
    assert "Generated on" not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert "research operating point, not a business-optimal threshold" in text
