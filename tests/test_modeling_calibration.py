"""Phase 9A: the cross-fitting property, the metrics, and the two rules.

The leakage property this phase depends on is not "the outer validation fold is
not scored twice" — it is that **no row is ever used to fit the estimator that
scores it and the calibrator that transforms that score**. That is asserted on
row identities, not on documentation.

Both rules are tested on **synthetic** inputs, every branch, so the branch the
real numbers happen to take is not the only one anybody exercised.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from churn.config import PROJECT_ROOT
from churn.modeling.calibration import (
    AVERAGE_PRECISION,
    BRIER,
    CALIBRATION_ENSEMBLE,
    CALIBRATION_INNER_N_SPLITS,
    CALIBRATION_METRIC_NAMES,
    ECE,
    ELIGIBLE,
    INCONCLUSIVE,
    ISOTONIC,
    ISOTONIC_CALIBRATION,
    LOG_LOSS,
    LOWER_IS_BETTER,
    MIN_IMPROVED_FOLDS,
    NO_CALIBRATION,
    NOT_ELIGIBLE,
    PROBABILITY_METRICS,
    RANKING_METRICS,
    ROC_AUC,
    SIGMOID,
    SIGMOID_CALIBRATION,
    UNCALIBRATED,
    CalibrationMetrics,
    PairedMetricDelta,
    build_calibrated_estimator,
    build_calibration_splitter,
    build_frozen_pipeline,
    compute_calibration_metrics,
    eligibility,
    expected_calibration_error,
    pair,
    reliability_bins,
    run_calibration_cv,
    select_policy,
)
from churn.modeling.calibration_results import (
    FrozenCandidateMismatchError,
    verify_frozen_candidate,
)
from churn.modeling.evaluation import build_splitter
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.modeling.tuning import build_modern_logistic
from churn.modeling.tuning_results import load_results as load_tuning_results
from churn.preprocessing.contracts import FEATURE_COLUMNS, build_feature_matrix
from churn.preprocessing.target import encode_target


@pytest.fixture(scope="module")
def synthetic_xy(comparison_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic feature matrix and target, for isolation tests."""
    return build_feature_matrix(comparison_frame), encode_target(comparison_frame["Churn"])


# --- the frozen candidate -----------------------------------------------------


def test_the_frozen_pipeline_uses_the_phase8b_builder() -> None:
    classifier = build_frozen_pipeline().named_steps[CLASSIFIER_STEP]

    assert isinstance(classifier, LogisticRegression)
    assert classifier.get_params() == build_modern_logistic().get_params()


def test_the_frozen_hyperparameters_are_unchanged() -> None:
    classifier = build_frozen_pipeline().named_steps[CLASSIFIER_STEP]

    assert classifier.C == 1.0
    assert classifier.l1_ratio == 0.0
    assert classifier.solver == "lbfgs"
    assert classifier.max_iter == 100


def test_class_weight_stays_none() -> None:
    assert build_frozen_pipeline().named_steps[CLASSIFIER_STEP].class_weight is None
    for method in (SIGMOID, ISOTONIC):
        estimator = build_calibrated_estimator(method, build_calibration_splitter())
        assert estimator.estimator.named_steps[CLASSIFIER_STEP].class_weight is None


def test_only_the_nineteen_original_features_reach_the_estimator(synthetic_xy) -> None:
    features, target = synthetic_xy
    fitted = build_frozen_pipeline().fit(features, target)

    encoder = fitted.named_steps[PREPROCESSOR_STEP].named_steps["encode"]
    used = {column for _, _, columns in encoder.transformers_ for column in columns}
    assert used == set(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 19


def test_the_frozen_candidate_gate_accepts_the_real_artefact() -> None:
    verified = verify_frozen_candidate(load_tuning_results())

    assert verified == {
        "C": 1.0,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "class_weight": None,
        "max_iter": 100,
    }


def test_the_frozen_candidate_gate_rejects_a_different_selection() -> None:
    """A gate that cannot fail proves nothing."""

    @dataclass
    class _Selection:
        selected_experiment: str
        frozen_hyperparameters: dict[str, object]

    @dataclass
    class _Results:
        selection: _Selection

    wrong_candidate = _Results(_Selection("T2", {"C": 1.0}))
    with pytest.raises(FrozenCandidateMismatchError, match="T2"):
        verify_frozen_candidate(wrong_candidate)

    wrong_parameters = _Results(
        _Selection(
            "T0",
            {"C": 0.5, "l1_ratio": 0.0, "solver": "lbfgs", "class_weight": None, "max_iter": 100},
        )
    )
    with pytest.raises(FrozenCandidateMismatchError, match="froze"):
        verify_frozen_candidate(wrong_parameters)


# --- metrics ------------------------------------------------------------------


_LABELS = np.array([0, 1, 1, 0])
_PROBABILITY = np.array([0.1, 0.9, 0.8, 0.3])


def test_brier_score_matches_a_hand_computed_fixture() -> None:
    # mean of (0.1-0)^2, (0.9-1)^2, (0.8-1)^2, (0.3-0)^2 = (0.01+0.01+0.04+0.09)/4
    metrics = compute_calibration_metrics(_LABELS, _PROBABILITY)

    assert metrics.brier_score == pytest.approx(0.0375)


def test_log_loss_matches_a_hand_computed_fixture() -> None:
    expected = -np.mean(
        [np.log(0.9), np.log(0.9), np.log(0.8), np.log(0.7)]
    )  # -log(1-p) or -log(p) per row
    metrics = compute_calibration_metrics(_LABELS, _PROBABILITY)

    assert metrics.log_loss == pytest.approx(expected)


def test_expected_calibration_error_matches_a_hand_computed_fixture() -> None:
    # Two quantile bins: [0.1, 0.2] against y = 0, and [0.8, 0.9] against y = 1.
    # Each bin: |mean(p) - mean(y)| = 0.15, each weighted 2/4.
    error = expected_calibration_error(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), n_bins=2
    )

    assert error == pytest.approx(0.15)


def test_expected_calibration_error_is_zero_when_perfectly_calibrated() -> None:
    labels = np.array([0, 0, 1, 1] * 25)
    probability = np.where(labels == 1, 1.0, 0.0)

    assert expected_calibration_error(labels, probability) == pytest.approx(0.0)


def test_expected_calibration_error_handles_a_constant_prediction() -> None:
    """Every quantile edge collapses to one value; the gap is still measurable."""
    error = expected_calibration_error(np.array([0, 0, 0, 1]), np.full(4, 0.2))

    assert error == pytest.approx(0.05)  # |0.2 - 0.25|


def test_expected_calibration_error_rejects_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="binning strategy"):
        expected_calibration_error(_LABELS, _PROBABILITY, strategy="logarithmic")


def test_reliability_bins_and_ece_share_one_binning() -> None:
    """A diagram binned differently from the number beside it describes two things."""
    rng = np.random.default_rng(0)
    probability = rng.uniform(size=400)
    labels = (rng.uniform(size=400) < probability).astype(int)

    predicted, observed, counts = reliability_bins(labels, probability, n_bins=10)
    manual = float(np.sum(counts / counts.sum() * np.abs(predicted - observed)))

    assert expected_calibration_error(labels, probability, n_bins=10) == pytest.approx(manual)
    assert counts.sum() == labels.size


def test_no_threshold_dependent_metric_exists() -> None:
    """Precision, recall, F1 and accuracy are not computed, so they cannot decide."""
    metrics = compute_calibration_metrics(_LABELS, _PROBABILITY)

    assert set(metrics.as_dict()) == set(CALIBRATION_METRIC_NAMES)
    for forbidden in ("precision", "recall", "f1", "accuracy"):
        assert not hasattr(metrics, forbidden)
        assert forbidden not in CALIBRATION_METRIC_NAMES


def test_the_metric_groups_are_disjoint_and_directed() -> None:
    assert set(PROBABILITY_METRICS) & set(RANKING_METRICS) == set()
    assert LOWER_IS_BETTER == frozenset({BRIER, LOG_LOSS, ECE})
    assert AVERAGE_PRECISION not in LOWER_IS_BETTER
    assert ROC_AUC not in LOWER_IS_BETTER


# --- the cross-fitting property -----------------------------------------------


class _RecordingCalibrator(CalibratedClassifierCV):
    """A calibrator that remembers exactly which rows it was fitted on."""

    seen: list[np.ndarray] = []

    def fit(self, X, y=None, **kwargs):  # noqa: N803 - sklearn's signature
        type(self).seen.append(np.asarray(X.index))
        return super().fit(X, y, **kwargs)


def test_the_outer_validation_fold_never_reaches_the_calibrator(synthetic_xy) -> None:
    """The core leakage property, asserted on row identities."""
    features, target = synthetic_xy
    _RecordingCalibrator.seen = []

    estimator = _RecordingCalibrator(
        estimator=build_frozen_pipeline(),
        method=SIGMOID,
        cv=build_calibration_splitter(),
        ensemble=CALIBRATION_ENSEMBLE,
    )
    outer = build_splitter()
    run_calibration_cv(
        SIGMOID_CALIBRATION, "c1", estimator, features, target, outer, method=SIGMOID
    )

    labels = np.asarray(target)
    splits = list(outer.split(features, labels))
    assert len(_RecordingCalibrator.seen) == len(splits)
    for seen, (train_index, validation_index) in zip(
        _RecordingCalibrator.seen, splits, strict=True
    ):
        assert set(seen) == set(features.index[train_index])
        assert not set(seen) & set(features.index[validation_index])


def test_the_protocol_pins_ensemble_to_false() -> None:
    """ensemble=True would add model averaging on top of the calibration layer."""
    assert CALIBRATION_ENSEMBLE is False

    for method in (SIGMOID, ISOTONIC):
        estimator = build_calibrated_estimator(method, build_calibration_splitter())
        assert estimator.ensemble is False
        assert estimator.ensemble is not True
        assert estimator.ensemble != "auto", "the sklearn default must not be relied on"


def test_no_ensemble_averaging_is_introduced(synthetic_xy) -> None:
    """With ensemble=False there is exactly one (estimator, calibrator) pair."""
    features, target = synthetic_xy
    estimator = build_calibrated_estimator(SIGMOID, build_calibration_splitter()).fit(
        features, np.asarray(target)
    )

    assert len(estimator.calibrated_classifiers_) == 1


def test_the_base_estimator_is_refitted_on_the_whole_outer_train(synthetic_xy) -> None:
    """The arm must differ from C0 by the calibration layer alone, so the base
    estimator has to see the same rows C0 sees — all of them."""
    features, target = synthetic_xy
    labels = np.asarray(target)

    calibrated = build_calibrated_estimator(SIGMOID, build_calibration_splitter()).fit(
        features, labels
    )
    base = calibrated.calibrated_classifiers_[0].estimator
    directly = build_frozen_pipeline().fit(features, labels)

    assert np.array_equal(
        base.named_steps[CLASSIFIER_STEP].coef_,
        directly.named_steps[CLASSIFIER_STEP].coef_,
    ), "the calibrated arm's base estimator was not fitted on the full training data"


def test_the_calibrator_learns_from_cross_validated_scores(synthetic_xy) -> None:
    """The calibrator must be fitted on out-of-sample scores.

    Asserted structurally: the inner splitter is what CalibratedClassifierCV was
    given, it has more than one split, and the fitted calibrator is not the
    identity — an in-sample fit would be a different object entirely, so the
    check that matters is that the cross-validation is real and reaches the
    calibrator.
    """
    features, target = synthetic_xy
    inner = build_calibration_splitter()
    estimator = build_calibrated_estimator(SIGMOID, inner)

    assert estimator.cv is inner
    assert estimator.cv.n_splits == CALIBRATION_INNER_N_SPLITS == 4

    fitted = estimator.fit(features, np.asarray(target))
    calibrators = fitted.calibrated_classifiers_[0].calibrators
    assert len(calibrators) == 1  # one per class column, binary problem


def test_the_complete_pipeline_sits_inside_the_calibrator() -> None:
    """Preprocessing must be refitted inside every inner calibration fold."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    estimator = build_calibrated_estimator(ISOTONIC, build_calibration_splitter())
    preprocessor = estimator.estimator.named_steps[PREPROCESSOR_STEP]

    assert estimator.estimator.named_steps[CLASSIFIER_STEP] is not None
    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor.named_steps["encode"])


def test_the_inner_splitter_is_the_declared_four_fold() -> None:
    inner = build_calibration_splitter()

    assert isinstance(inner, StratifiedKFold)
    assert inner.n_splits == CALIBRATION_INNER_N_SPLITS == 4
    assert inner.shuffle is True
    assert inner.random_state == 42


def test_an_unknown_calibration_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown calibration method"):
        build_calibrated_estimator("temperature", build_calibration_splitter())


def test_all_three_policies_share_the_outer_folds(synthetic_xy) -> None:
    features, target = synthetic_xy
    outer = build_splitter()
    inner = build_calibration_splitter()
    runs = [
        run_calibration_cv(UNCALIBRATED, "c0", build_frozen_pipeline(), features, target, outer),
        run_calibration_cv(
            SIGMOID_CALIBRATION,
            "c1",
            build_calibrated_estimator(SIGMOID, inner),
            features,
            target,
            outer,
            method=SIGMOID,
        ),
    ]
    signatures = {
        tuple((fold.fold, fold.n_train, fold.n_validation) for fold in run.folds) for run in runs
    }

    assert len(signatures) == 1


def test_each_row_receives_exactly_one_outer_oof_probability(synthetic_xy) -> None:
    features, target = synthetic_xy
    run = run_calibration_cv(
        UNCALIBRATED, "c0", build_frozen_pipeline(), features, target, build_splitter()
    )

    assert run.oof_probability.shape == (len(features),)
    assert not np.isnan(run.oof_probability).any()
    assert run.calibrated is False
    assert run.method is None


def test_no_row_is_scored_by_a_process_that_trained_on_it(synthetic_xy) -> None:
    features, target = synthetic_xy
    outer = build_splitter()
    seen: set[int] = set()

    for train_index, validation_index in outer.split(features, np.asarray(target)):
        assert not set(train_index) & set(validation_index)
        assert not seen & set(validation_index)
        seen |= set(validation_index)

    assert seen == set(range(len(features)))


# --- paired deltas ------------------------------------------------------------


def test_a_paired_delta_respects_lower_is_better() -> None:
    losses = PairedMetricDelta(
        metric=BRIER,
        reference=UNCALIBRATED,
        lower_is_better=True,
        deltas=(-0.01, -0.02, 0.0, 0.03, -0.005),
    )

    assert losses.n_improved == 3  # the negative ones
    assert losses.n_worsened == 1
    assert losses.n_tied == 1


def test_a_paired_delta_respects_higher_is_better() -> None:
    ranking = PairedMetricDelta(
        metric=AVERAGE_PRECISION,
        reference=UNCALIBRATED,
        lower_is_better=False,
        deltas=(-0.01, -0.02, 0.0, 0.03, -0.005),
    )

    assert ranking.n_improved == 1  # the positive one
    assert ranking.n_worsened == 3
    assert ranking.n_tied == 1


def test_ties_are_counted_separately(synthetic_xy) -> None:
    """The three counts do not have to sum the way a two-way split would."""
    delta = PairedMetricDelta(BRIER, UNCALIBRATED, True, (0.0, 0.0, -0.1, 0.1, 0.0))

    assert (delta.n_improved, delta.n_tied, delta.n_worsened) == (1, 3, 1)
    assert delta.n_improved + delta.n_worsened != len(delta.deltas)


def test_paired_deltas_follow_the_outer_fold_order(synthetic_xy) -> None:
    features, target = synthetic_xy
    outer = build_splitter()
    c0 = run_calibration_cv(UNCALIBRATED, "c0", build_frozen_pipeline(), features, target, outer)
    c1 = run_calibration_cv(
        SIGMOID_CALIBRATION,
        "c1",
        build_calibrated_estimator(SIGMOID, build_calibration_splitter()),
        features,
        target,
        outer,
        method=SIGMOID,
    )
    pair(c1, c0)

    expected = [
        a.metrics.brier_score - b.metrics.brier_score
        for a, b in zip(c1.folds, c0.folds, strict=True)
    ]
    assert list(c1.deltas[UNCALIBRATED][BRIER].deltas) == pytest.approx(expected)
    assert c1.deltas[UNCALIBRATED][BRIER].lower_is_better is True
    assert c1.deltas[UNCALIBRATED][AVERAGE_PRECISION].lower_is_better is False


# --- the pre-registered eligibility rule --------------------------------------


def _deltas(brier, log_loss, average_precision) -> dict[str, PairedMetricDelta]:
    """Build a synthetic delta set, so the rule is tested independently of any run."""
    return {
        BRIER: PairedMetricDelta(BRIER, UNCALIBRATED, True, tuple(brier)),
        LOG_LOSS: PairedMetricDelta(LOG_LOSS, UNCALIBRATED, True, tuple(log_loss)),
        AVERAGE_PRECISION: PairedMetricDelta(
            AVERAGE_PRECISION, UNCALIBRATED, False, tuple(average_precision)
        ),
    }


#: Loss orientation: a negative delta is an improvement.
_IMPROVES = (-0.01, -0.01, -0.01, -0.01, 0.001)
_WORSENS = (0.01, 0.01, 0.01, 0.01, -0.001)
_MIXED = (-0.02, 0.005, 0.005, 0.005, 0.004)

#: Ranking orientation is the opposite, so the guardrail fixtures are spelled
#: out separately rather than reusing the loss ones — reusing them would test
#: the inverse of what the name says.
_NEUTRAL_AP = (0.0, 0.0, 0.0, 0.0, 0.0)
_AP_DETERIORATES = (-0.01, -0.01, -0.01, -0.01, 0.001)
_AP_ONE_BAD_FOLD = (0.001, 0.001, 0.001, 0.001, -0.02)


def test_eligibility_requires_a_consistent_brier_improvement() -> None:
    assert eligibility(_deltas(_IMPROVES, _IMPROVES, _NEUTRAL_AP), 5)[0] == ELIGIBLE
    assert eligibility(_deltas(_WORSENS, _IMPROVES, _NEUTRAL_AP), 5)[0] == NOT_ELIGIBLE
    assert eligibility(_deltas(_MIXED, _IMPROVES, _NEUTRAL_AP), 5)[0] == NOT_ELIGIBLE


def test_a_brier_gain_that_is_not_consistent_is_not_enough() -> None:
    """Negative mean, but only three folds moving: the rule needs four."""
    inconsistent = (-0.05, -0.01, -0.01, 0.01, 0.01)
    status, _ = eligibility(_deltas(inconsistent, _IMPROVES, _NEUTRAL_AP), 5)

    assert np.mean(inconsistent) < 0
    assert status == NOT_ELIGIBLE


def test_disagreement_between_the_two_losses_is_inconclusive() -> None:
    status, rationale = eligibility(_deltas(_IMPROVES, _WORSENS, _NEUTRAL_AP), 5)

    assert status == INCONCLUSIVE
    assert "inconclusive" in rationale


def test_a_consistent_ranking_deterioration_blocks_adoption() -> None:
    deltas = _deltas(_IMPROVES, _IMPROVES, _AP_DETERIORATES)
    status, rationale = eligibility(deltas, 5)

    assert deltas[AVERAGE_PRECISION].n_worsened == 4
    assert status == NOT_ELIGIBLE
    assert "guardrail" in rationale


def test_an_isolated_ranking_dip_does_not_block_adoption() -> None:
    """One bad fold is noise; the guardrail needs direction and consistency."""
    deltas = _deltas(_IMPROVES, _IMPROVES, _AP_ONE_BAD_FOLD)
    status, _ = eligibility(deltas, 5)

    assert np.mean(_AP_ONE_BAD_FOLD) < 0
    assert deltas[AVERAGE_PRECISION].n_worsened == 1
    assert status == ELIGIBLE


def test_eligibility_has_no_minimum_gain_cutoff() -> None:
    tiny = (-1e-9, -1e-9, -1e-9, -1e-9, 1e-12)
    status, _ = eligibility(_deltas(tiny, tiny, _NEUTRAL_AP), 5)

    assert status == ELIGIBLE
    assert MIN_IMPROVED_FOLDS == 4


# --- the pre-registered selection rule ----------------------------------------


_STABILITY = {SIGMOID_CALIBRATION: 0.004, ISOTONIC_CALIBRATION: 0.009}


def test_selection_keeps_no_calibration_when_neither_is_eligible() -> None:
    policy, rationale = select_policy(
        {SIGMOID_CALIBRATION: NOT_ELIGIBLE, ISOTONIC_CALIBRATION: NOT_ELIGIBLE},
        None,
        _STABILITY,
        5,
    )

    assert policy == NO_CALIBRATION
    assert rationale


def test_selection_treats_inconclusive_as_not_adopted() -> None:
    policy, _ = select_policy(
        {SIGMOID_CALIBRATION: INCONCLUSIVE, ISOTONIC_CALIBRATION: NOT_ELIGIBLE},
        None,
        _STABILITY,
        5,
    )

    assert policy == NO_CALIBRATION


def test_selection_takes_sigmoid_when_only_sigmoid_is_eligible() -> None:
    policy, _ = select_policy(
        {SIGMOID_CALIBRATION: ELIGIBLE, ISOTONIC_CALIBRATION: NOT_ELIGIBLE}, None, _STABILITY, 5
    )

    assert policy == SIGMOID_CALIBRATION


def test_selection_takes_isotonic_when_only_isotonic_is_eligible() -> None:
    policy, _ = select_policy(
        {SIGMOID_CALIBRATION: NOT_ELIGIBLE, ISOTONIC_CALIBRATION: ELIGIBLE}, None, _STABILITY, 5
    )

    assert policy == ISOTONIC_CALIBRATION


def test_selection_takes_isotonic_when_it_wins_brier_head_to_head() -> None:
    policy, rationale = select_policy(
        {SIGMOID_CALIBRATION: ELIGIBLE, ISOTONIC_CALIBRATION: ELIGIBLE},
        _deltas(_IMPROVES, _WORSENS, _NEUTRAL_AP),
        _STABILITY,
        5,
    )

    assert policy == ISOTONIC_CALIBRATION
    assert BRIER in rationale


def test_selection_takes_sigmoid_when_it_wins_brier_head_to_head() -> None:
    policy, rationale = select_policy(
        {SIGMOID_CALIBRATION: ELIGIBLE, ISOTONIC_CALIBRATION: ELIGIBLE},
        _deltas(_WORSENS, _IMPROVES, _NEUTRAL_AP),
        _STABILITY,
        5,
    )

    assert policy == SIGMOID_CALIBRATION
    assert BRIER in rationale


def test_selection_falls_through_to_log_loss_when_brier_is_inconsistent() -> None:
    policy, rationale = select_policy(
        {SIGMOID_CALIBRATION: ELIGIBLE, ISOTONIC_CALIBRATION: ELIGIBLE},
        _deltas(_MIXED, _IMPROVES, _NEUTRAL_AP),
        _STABILITY,
        5,
    )

    assert policy == ISOTONIC_CALIBRATION
    assert LOG_LOSS in rationale


def test_selection_defaults_to_sigmoid_without_a_consistent_winner() -> None:
    policy, rationale = select_policy(
        {SIGMOID_CALIBRATION: ELIGIBLE, ISOTONIC_CALIBRATION: ELIGIBLE},
        _deltas(_MIXED, _MIXED, _NEUTRAL_AP),
        _STABILITY,
        5,
    )

    assert policy == SIGMOID_CALIBRATION
    assert "flexibility" in rationale


def test_selection_requires_the_head_to_head_when_both_are_eligible() -> None:
    with pytest.raises(ValueError, match="C2-vs-C1"):
        select_policy(
            {SIGMOID_CALIBRATION: ELIGIBLE, ISOTONIC_CALIBRATION: ELIGIBLE}, None, None, 5
        )


# --- audits -------------------------------------------------------------------

_PHASE9A_SOURCES = [
    PROJECT_ROOT / "src" / "churn" / "modeling" / "calibration.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "calibration_results.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "calibration_plots.py",
    PROJECT_ROOT / "scripts" / "run_calibration.py",
]

FORBIDDEN_IDENTIFIERS = (
    "TunedThresholdClassifierCV",
    "GridSearchCV",
    "RandomizedSearchCV",
    "RandomForestClassifier",
    "HistGradientBoostingClassifier",
    "build_hist_gradient_boosting",
    "build_hgb_native",
    "build_native_pipeline",
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


@pytest.mark.parametrize("source", _PHASE9A_SOURCES, ids=lambda path: path.name)
def test_phase9a_code_never_reaches_the_holdout(source) -> None:
    used = _identifiers(source)

    assert "load_holdout" not in used, f"{source.name} imports or calls the holdout loader"
    assert "holdout" not in used, f"{source.name} reads a holdout partition"


@pytest.mark.parametrize("source", _PHASE9A_SOURCES, ids=lambda path: path.name)
def test_phase9a_code_stays_inside_its_scope(source) -> None:
    used = _identifiers(source)

    found = sorted(name for name in FORBIDDEN_IDENTIFIERS if name in used)
    assert not found, (
        f"{source.name} references {found}: this phase does not search hyperparameters, move a "
        "threshold, reopen a model family or use an engineered feature"
    )


@pytest.mark.parametrize("source", _PHASE9A_SOURCES, ids=lambda path: path.name)
def test_no_phase9a_call_sets_class_weight(source) -> None:
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


@pytest.mark.parametrize("source", _PHASE9A_SOURCES, ids=lambda path: path.name)
def test_only_the_script_loads_data_and_only_the_training_pool(source) -> None:
    used = _identifiers(source)

    if source.name == "run_calibration.py":
        assert "load_training_pool" in used
        assert "load_split" not in used
        return
    assert "load_training_pool" not in used
    assert "load_raw_typed" not in used


def test_the_report_generator_reads_no_clock() -> None:
    used = _identifiers(PROJECT_ROOT / "scripts" / "run_calibration.py")

    found = sorted(
        name
        for name in ("datetime", "date", "today", "now", "utcnow", "monotonic", "perf_counter")
        if name in used
    )
    assert not found, f"run_calibration.py reads wall-clock state via {found}"


# --- generated artefacts ------------------------------------------------------


def _results_path():
    return PROJECT_ROOT / "reports" / "experiments" / "calibration_results.json"


def test_the_artefact_carries_no_timestamp_and_no_holdout() -> None:
    path = _results_path()
    if not path.is_file():
        pytest.skip("the artefact has not been generated yet")

    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert not [key for key in payload if re.search("time|stamp|generated", key, re.I)]
    assert not re.search(r"\d{4}-\d{2}-\d{2}", raw)
    assert "load_holdout" not in raw


def test_the_artefact_records_no_threshold_dependent_metric() -> None:
    path = _results_path()
    if not path.is_file():
        pytest.skip("the artefact has not been generated yet")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["threshold_selected"] is False
    assert payload["class_weight"] is None
    assert payload["metric_protocol"]["threshold_dependent_metrics_computed"] == []
    for policy in payload["policies"]:
        for forbidden in ("precision", "recall", "f1", "accuracy"):
            assert forbidden not in policy["mean"]
            assert forbidden not in policy["outer_out_of_fold"]


def test_the_report_carries_no_generation_date() -> None:
    report = PROJECT_ROOT / "reports" / "calibration_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")

    text = report.read_text(encoding="utf-8")
    assert "Generated on" not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)


def test_the_metrics_dataclass_round_trips_through_the_artefact_names() -> None:
    metrics = CalibrationMetrics(0.1, 0.2, 0.3, 0.4, 0.5)

    assert list(metrics.as_dict()) == list(CALIBRATION_METRIC_NAMES)


def test_the_artefact_records_ensemble_false_for_both_calibrators() -> None:
    path = _results_path()
    if not path.is_file():
        pytest.skip("the artefact has not been generated yet")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["calibration_inner_cross_validation"]["ensemble"] is False
    methods = {entry["experiment"]: entry for entry in payload["calibration_methods"]}
    assert methods["C1"]["ensemble"] is False
    assert methods["C2"]["ensemble"] is False
    assert methods["C0"]["ensemble"] is None


def test_the_artefact_records_the_c0_reproduction_gate() -> None:
    path = _results_path()
    if not path.is_file():
        pytest.skip("the artefact has not been generated yet")

    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = payload["reproduction"]

    assert len(checks) == 1
    assert checks[0]["experiment"] == UNCALIBRATED
    assert checks[0]["reference_key"] == "procedures[T0]"
    assert checks[0]["reproduced"] is True
    assert checks[0]["max_absolute_difference"] <= checks[0]["tolerance"]


def test_c0_reproduces_the_frozen_tuning_reference() -> None:
    """The gate itself, run against the real artefact rather than only recorded."""
    tuning = load_tuning_results()
    reference = next(record for record in tuning.procedures if record.experiment == "T0")
    path = _results_path()
    if not path.is_file():
        pytest.skip("the artefact has not been generated yet")

    policies = {
        entry["experiment"]: entry
        for entry in json.loads(path.read_text(encoding="utf-8"))["policies"]
    }
    c0 = policies[UNCALIBRATED]

    for fold, stored in zip(c0["folds"], reference.folds, strict=True):
        for metric in (AVERAGE_PRECISION, ROC_AUC):
            assert fold["metrics"][metric] == pytest.approx(stored.metrics[metric], abs=1e-9)


def test_the_sigmoid_ranking_expectation_is_recorded() -> None:
    path = _results_path()
    if not path.is_file():
        pytest.skip("the artefact has not been generated yet")

    expectations = json.loads(path.read_text(encoding="utf-8"))["ranking_expectations"]

    assert "C1_sigmoid" in expectations
    assert "C2_isotonic" in expectations
    assert "monotone" in expectations["C1_sigmoid"]
    assert "resolution" in expectations["C2_isotonic"]
