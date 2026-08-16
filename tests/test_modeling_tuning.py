"""Phase 8B: the compatibility gate, the nested protocol and the selection rule.

The selection rule is tested on **synthetic** paired comparisons, every branch of
it, so that the branch the real numbers happen to take is not the only one
anybody ever exercised. A rule written before the results and checked only
against the results it produced is not pre-registered in any meaningful sense.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from churn.config import PROJECT_ROOT
from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC, PairedComparison
from churn.modeling.evaluation import build_splitter, evaluate_model
from churn.modeling.families import HIST_GRADIENT_BOOSTING_PARAMS, build_family_pipeline
from churn.modeling.metrics import DEFAULT_THRESHOLD, METRIC_NAMES
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP, build_logistic_baseline
from churn.modeling.results import load_results as load_baseline_results
from churn.modeling.tuning import (
    ELIGIBILITY_MIN_POSITIVE_FOLDS,
    ELIGIBLE,
    FROZEN_LOGISTIC,
    INNER_N_SPLITS,
    LOGISTIC_C_GRID,
    NOT_ELIGIBLE,
    SEARCH_SCORING,
    T1_SEARCH_SPACE,
    T2_FIXED_PARAMS,
    T2_SEARCH_SPACE,
    TUNED_HGB,
    TUNED_LOGISTIC,
    build_hgb_pipeline,
    build_inner_splitter,
    build_modern_logistic,
    build_modern_logistic_pipeline,
    build_search,
    eligibility,
    frozen_hgb_configuration,
    n_candidates,
    pair,
    run_nested_cv,
    select_candidate,
    selection_frequency,
)
from churn.modeling.tuning_results import verify_migration
from churn.preprocessing.contracts import FEATURE_COLUMNS, build_feature_matrix
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target


@pytest.fixture(scope="module")
def synthetic_xy(comparison_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic feature matrix and target, for isolation tests."""
    return build_feature_matrix(comparison_frame), encode_target(comparison_frame["Churn"])


@pytest.fixture(scope="module")
def real_training_xy() -> tuple[pd.DataFrame, pd.Series]:
    """The frozen training pool. Never the holdout."""
    training = load_training_pool()
    return build_feature_matrix(training), encode_target(training["Churn"])


# --- compatibility gate -------------------------------------------------------


def test_legacy_logistic_reproduces_the_historical_artefact(real_training_xy) -> None:
    features, target = real_training_xy
    evaluation = evaluate_model(
        "legacy",
        build_family_pipeline("logistic_regression", ()),
        features,
        target,
        build_splitter(),
    )

    reference = load_baseline_results()
    model = next(item for item in reference.models if item.name == "logistic_regression")
    for fold, stored in zip(evaluation.folds, model.folds, strict=True):
        for metric in METRIC_NAMES:
            assert round(getattr(fold.metrics, metric), 6) == pytest.approx(
                stored.metrics[metric], abs=1e-9
            )


def test_modern_logistic_reproduces_the_legacy_one(real_training_xy) -> None:
    """The whole migration rests on this: same metrics *and* same probabilities."""
    features, target = real_training_xy
    splitter = build_splitter()
    legacy = evaluate_model(
        "legacy", build_family_pipeline("logistic_regression", ()), features, target, splitter
    )
    modern = evaluate_model("modern", build_modern_logistic_pipeline(), features, target, splitter)

    check = verify_migration(legacy, modern, legacy_warned=True, modern_warned=False)
    assert check.metrics_equivalent
    assert check.probabilities_equivalent
    assert check.max_metric_difference <= 1e-9
    assert check.max_oof_probability_difference <= 1e-9
    assert check.classification == "COMPATIBILITY_MIGRATION"


def test_the_migration_gate_rejects_a_genuine_model_change(real_training_xy) -> None:
    """A different C must fail the gate, or the gate proves nothing."""
    from churn.modeling.comparison_results import ReproductionError

    features, target = real_training_xy
    splitter = build_splitter()
    legacy = evaluate_model(
        "legacy", build_family_pipeline("logistic_regression", ()), features, target, splitter
    )
    changed = build_modern_logistic_pipeline()
    changed.set_params(**{f"{CLASSIFIER_STEP}__C": 0.5})
    other = evaluate_model("changed", changed, features, target, splitter)

    with pytest.raises(ReproductionError):
        verify_migration(legacy, other, legacy_warned=True, modern_warned=False)


def test_the_modern_builder_never_passes_penalty_explicitly() -> None:
    modern = build_modern_logistic()

    assert modern.l1_ratio == 0.0
    assert modern.get_params()["l1_ratio"] == 0.0


def test_the_modern_builder_emits_no_penalty_warning(synthetic_xy) -> None:
    features, target = synthetic_xy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_modern_logistic_pipeline().fit(features, target)

    offenders = [entry for entry in caught if "penalty" in str(entry.message)]
    assert not offenders, f"the modern builder still raises: {[str(e.message) for e in offenders]}"


def test_the_legacy_builder_is_preserved_unchanged() -> None:
    """It is what reproduces the historical artefacts; it must not be migrated."""
    legacy = build_logistic_baseline().named_steps[CLASSIFIER_STEP]

    assert legacy.C == 1.0
    assert legacy.solver == "lbfgs"
    assert legacy.max_iter == 100
    assert legacy.class_weight is None


def test_the_modern_builder_keeps_class_weight_none() -> None:
    assert build_modern_logistic().class_weight is None


def test_the_logistic_grid_contains_the_baseline_value() -> None:
    assert 1.0 in LOGISTIC_C_GRID
    assert 1.0 in T1_SEARCH_SPACE[f"{CLASSIFIER_STEP}__C"]


# --- search spaces ------------------------------------------------------------


def test_t1_search_has_exactly_nine_candidates() -> None:
    assert n_candidates(T1_SEARCH_SPACE) == 9
    assert len(LOGISTIC_C_GRID) == 9


def test_t2_search_has_exactly_forty_eight_candidates() -> None:
    assert n_candidates(T2_SEARCH_SPACE) == 48
    assert 2 * 2 * 2 * 3 * 2 == 48


def test_t2_grid_contains_the_frozen_phase7_configuration() -> None:
    frozen = frozen_hgb_configuration()

    assert frozen == {
        f"{CLASSIFIER_STEP}__learning_rate": 0.1,
        f"{CLASSIFIER_STEP}__max_iter": 100,
        f"{CLASSIFIER_STEP}__max_leaf_nodes": 31,
        f"{CLASSIFIER_STEP}__min_samples_leaf": 20,
        f"{CLASSIFIER_STEP}__l2_regularization": 0.0,
    }
    for parameter, value in frozen.items():
        assert value in T2_SEARCH_SPACE[parameter]


def test_t2_freezes_the_parameters_it_does_not_search() -> None:
    classifier = build_hgb_pipeline().named_steps[CLASSIFIER_STEP]

    for name in T2_FIXED_PARAMS:
        if name == "random_state":
            assert classifier.random_state == 42
            continue
        assert getattr(classifier, name) == HIST_GRADIENT_BOOSTING_PARAMS[name]
    assert classifier.early_stopping is False
    assert classifier.class_weight is None


def test_no_search_space_touches_class_weight_or_threshold() -> None:
    for space in (T1_SEARCH_SPACE, T2_SEARCH_SPACE):
        joined = " ".join(space)
        assert "class_weight" not in joined
        assert "threshold" not in joined


def test_the_search_selects_on_average_precision_alone() -> None:
    search = build_search(build_modern_logistic_pipeline(), T1_SEARCH_SPACE, build_inner_splitter())

    assert search.scoring == SEARCH_SCORING == "average_precision"
    assert isinstance(search.scoring, str), "a multi-metric objective would change the selection"
    assert search.refit is True


def test_the_inner_splitter_is_the_declared_four_fold() -> None:
    inner = build_inner_splitter()

    assert isinstance(inner, StratifiedKFold)
    assert inner.n_splits == INNER_N_SPLITS == 4
    assert inner.shuffle is True
    assert inner.random_state == 42


# --- nested protocol ----------------------------------------------------------


class _RecordingSearch(GridSearchCV):
    """A GridSearchCV that remembers exactly which rows it was fitted on."""

    seen: list[np.ndarray] = []

    def fit(self, X, y=None, **kwargs):  # noqa: N803 - sklearn's signature
        type(self).seen.append(np.asarray(X.index))
        return super().fit(X, y, **kwargs)


def test_the_outer_validation_fold_never_reaches_the_search(synthetic_xy, monkeypatch) -> None:
    """The core leakage property of nested CV, asserted on row identities."""
    features, target = synthetic_xy
    _RecordingSearch.seen = []

    def recording_search(estimator, search_space, inner):
        return _RecordingSearch(
            estimator=estimator,
            param_grid=dict(search_space),
            scoring=SEARCH_SCORING,
            cv=inner,
            refit=True,
            n_jobs=1,
        )

    monkeypatch.setattr("churn.modeling.tuning.build_search", recording_search)
    outer = build_splitter()
    run_nested_cv(
        TUNED_LOGISTIC,
        "t1",
        build_modern_logistic_pipeline(),
        features,
        target,
        outer,
        {f"{CLASSIFIER_STEP}__C": [0.1, 1.0]},
        build_inner_splitter(),
    )

    labels = np.asarray(target)
    expected = [features.index[train] for train, _ in outer.split(features, labels)]
    assert len(_RecordingSearch.seen) == len(expected)
    for seen, train_index in zip(_RecordingSearch.seen, expected, strict=True):
        assert set(seen) == set(train_index)

    # And the complement — every outer validation row is absent from its search.
    for seen, (_, validation) in zip(
        _RecordingSearch.seen, outer.split(features, labels), strict=True
    ):
        assert not set(seen) & set(features.index[validation])


def test_the_estimator_given_to_the_search_is_an_unfitted_pipeline() -> None:
    """Preprocessing must be inside the search, so every inner fold refits it."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    search = build_search(build_modern_logistic_pipeline(), T1_SEARCH_SPACE, build_inner_splitter())
    preprocessor = search.estimator.named_steps[PREPROCESSOR_STEP]

    assert search.estimator.named_steps[CLASSIFIER_STEP] is not None
    with pytest.raises(NotFittedError):
        check_is_fitted(preprocessor.named_steps["encode"])


def test_preprocessing_is_refitted_inside_every_inner_fold(synthetic_xy) -> None:
    """Two inner folds must learn different scaler statistics, or something is pre-fitted."""
    features, target = synthetic_xy
    inner = build_inner_splitter()
    statistics = []

    for train_index, _ in inner.split(features, np.asarray(target)):
        pipeline = build_modern_logistic_pipeline().fit(
            features.iloc[train_index], np.asarray(target)[train_index]
        )
        scaler = (
            pipeline.named_steps[PREPROCESSOR_STEP]
            .named_steps["encode"]
            .named_transformers_["numeric"]
        )
        statistics.append(tuple(scaler.mean_))

    assert len(set(statistics)) > 1


def test_all_three_procedures_share_the_outer_folds(synthetic_xy) -> None:
    features, target = synthetic_xy
    outer = build_splitter()
    runs = [
        run_nested_cv(
            FROZEN_LOGISTIC, "t0", build_modern_logistic_pipeline(), features, target, outer
        ),
        run_nested_cv(
            TUNED_LOGISTIC,
            "t1",
            build_modern_logistic_pipeline(),
            features,
            target,
            outer,
            {f"{CLASSIFIER_STEP}__C": [0.1, 1.0]},
            build_inner_splitter(),
        ),
    ]
    signatures = {
        tuple((fold.fold, fold.n_train, fold.n_validation) for fold in run.folds) for run in runs
    }
    assert len(signatures) == 1


def test_each_row_receives_exactly_one_outer_oof_prediction(synthetic_xy) -> None:
    features, target = synthetic_xy
    run = run_nested_cv(
        FROZEN_LOGISTIC, "t0", build_modern_logistic_pipeline(), features, target, build_splitter()
    )

    assert run.oof_probability.shape == (len(features),)
    assert not np.isnan(run.oof_probability).any()


def test_no_row_is_predicted_by_a_procedure_that_trained_on_it(synthetic_xy) -> None:
    features, target = synthetic_xy
    outer = build_splitter()
    seen: set[int] = set()

    for train_index, validation_index in outer.split(features, np.asarray(target)):
        assert not set(train_index) & set(validation_index)
        assert not seen & set(validation_index)
        seen |= set(validation_index)

    assert seen == set(range(len(features)))


def test_a_search_space_without_an_inner_splitter_is_rejected(synthetic_xy) -> None:
    features, target = synthetic_xy

    with pytest.raises(ValueError, match="inner splitter"):
        run_nested_cv(
            TUNED_LOGISTIC,
            "t1",
            build_modern_logistic_pipeline(),
            features,
            target,
            build_splitter(),
            T1_SEARCH_SPACE,
        )


def test_the_untuned_run_records_no_hyperparameter_choice(synthetic_xy) -> None:
    features, target = synthetic_xy
    run = run_nested_cv(
        FROZEN_LOGISTIC, "t0", build_modern_logistic_pipeline(), features, target, build_splitter()
    )

    assert run.tuned is False
    assert all(fold.best_params is None for fold in run.folds)
    assert selection_frequency(run) == {}


def test_the_tuned_run_records_a_choice_per_outer_fold(synthetic_xy) -> None:
    features, target = synthetic_xy
    run = run_nested_cv(
        TUNED_LOGISTIC,
        "t1",
        build_modern_logistic_pipeline(),
        features,
        target,
        build_splitter(),
        {f"{CLASSIFIER_STEP}__C": [0.1, 1.0]},
        build_inner_splitter(),
    )

    assert run.tuned is True
    assert all(fold.best_params for fold in run.folds)
    assert all(fold.best_inner_score is not None for fold in run.folds)
    frequency = selection_frequency(run)
    assert set(frequency) == {"C"}
    assert sum(frequency["C"].values()) == len(run.folds)


def test_paired_deltas_follow_the_outer_fold_order(synthetic_xy) -> None:
    features, target = synthetic_xy
    outer = build_splitter()
    t0 = run_nested_cv(
        FROZEN_LOGISTIC, "t0", build_modern_logistic_pipeline(), features, target, outer
    )
    t1 = run_nested_cv(
        TUNED_LOGISTIC,
        "t1",
        build_modern_logistic_pipeline(),
        features,
        target,
        outer,
        {f"{CLASSIFIER_STEP}__C": [0.1, 1.0]},
        build_inner_splitter(),
    )
    pair(t1, t0, (PRIMARY_METRIC, SECONDARY_METRIC))

    expected = [
        getattr(a.metrics, PRIMARY_METRIC) - getattr(b.metrics, PRIMARY_METRIC)
        for a, b in zip(t1.folds, t0.folds, strict=True)
    ]
    assert list(t1.paired_deltas[FROZEN_LOGISTIC][PRIMARY_METRIC].deltas) == pytest.approx(expected)


def test_the_threshold_is_never_moved(synthetic_xy) -> None:
    features, target = synthetic_xy
    assert DEFAULT_THRESHOLD == 0.5

    run = run_nested_cv(
        FROZEN_LOGISTIC, "t0", build_modern_logistic_pipeline(), features, target, build_splitter()
    )
    labels = np.asarray(target)
    predicted = (run.oof_probability >= DEFAULT_THRESHOLD).astype(int)
    assert run.oof_metrics.accuracy == pytest.approx(float((predicted == labels).mean()))


def test_only_the_nineteen_original_features_reach_the_estimators(synthetic_xy) -> None:
    features, target = synthetic_xy

    for pipeline in (build_modern_logistic_pipeline(), build_hgb_pipeline()):
        fitted = pipeline.fit(features, target)
        encoder = fitted.named_steps[PREPROCESSOR_STEP].named_steps["encode"]
        used = {column for _, _, columns in encoder.transformers_ for column in columns}
        assert used == set(FEATURE_COLUMNS)


# --- the pre-registered selection rule ----------------------------------------


@dataclass
class _Comparison:
    """A synthetic paired comparison, so the rule is tested independently."""

    deltas: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.mean(self.deltas))

    @property
    def n_positive(self) -> int:
        return int(sum(delta > 0 for delta in self.deltas))

    @property
    def n_negative(self) -> int:
        return int(sum(delta < 0 for delta in self.deltas))

    @property
    def n_zero(self) -> int:
        return int(sum(delta == 0 for delta in self.deltas))


WINS = _Comparison((0.01, 0.01, 0.01, 0.01, -0.001))
LOSES = _Comparison((-0.01, -0.01, -0.01, -0.01, 0.001))
MIXED = _Comparison((0.02, -0.005, -0.005, -0.005, -0.004))


def test_eligibility_requires_direction_and_consistency() -> None:
    assert eligibility(WINS, 5)[0] == ELIGIBLE
    assert eligibility(LOSES, 5)[0] == NOT_ELIGIBLE
    assert eligibility(MIXED, 5)[0] == NOT_ELIGIBLE
    assert eligibility(_Comparison((0.0,) * 5), 5)[0] == NOT_ELIGIBLE


def test_a_tied_fold_counts_as_neither_improvement_nor_loss() -> None:
    """Ties are their own count: the three do not have to sum the two-way way."""
    tied = _Comparison((0.01, 0.0, 0.0, -0.01, 0.02))

    assert tied.n_positive == 2
    assert tied.n_zero == 2
    assert tied.n_negative == 1
    assert tied.n_positive + tied.n_negative != len(tied.deltas)


def test_a_tie_can_never_make_a_procedure_eligible() -> None:
    """Four strictly positive folds pass; three positives plus a tie do not."""
    strictly = _Comparison((0.01, 0.01, 0.01, 0.01, -0.001))
    with_a_tie = _Comparison((0.01, 0.01, 0.01, 0.0, -0.001))

    assert eligibility(strictly, 5)[0] == ELIGIBLE
    assert eligibility(with_a_tie, 5)[0] == NOT_ELIGIBLE
    assert "tied" in eligibility(with_a_tie, 5)[1]


def test_the_real_paired_comparison_reports_the_three_counts() -> None:
    """PairedComparison itself, not only the synthetic stand-in, counts ties."""
    comparison = PairedComparison(metric=PRIMARY_METRIC, deltas=(0.01, 0.0, -0.01))

    assert (comparison.n_positive, comparison.n_zero, comparison.n_negative) == (1, 1, 1)


def test_eligibility_has_no_minimum_gain_cutoff() -> None:
    """A vanishingly small but consistent gain is eligible; magnitude is a reader's call."""
    tiny = _Comparison((1e-9, 1e-9, 1e-9, 1e-9, -1e-12))

    assert eligibility(tiny, 5)[0] == ELIGIBLE
    assert ELIGIBILITY_MIN_POSITIVE_FOLDS == 4


def test_selection_keeps_the_baseline_when_neither_is_eligible() -> None:
    selected, rationale = select_candidate(
        {TUNED_LOGISTIC: NOT_ELIGIBLE, TUNED_HGB: NOT_ELIGIBLE}, None, 5
    )

    assert selected == FROZEN_LOGISTIC
    assert rationale


def test_selection_takes_t1_when_only_t1_is_eligible() -> None:
    selected, _ = select_candidate({TUNED_LOGISTIC: ELIGIBLE, TUNED_HGB: NOT_ELIGIBLE}, None, 5)

    assert selected == TUNED_LOGISTIC


def test_selection_takes_t2_when_only_t2_is_eligible() -> None:
    selected, _ = select_candidate({TUNED_LOGISTIC: NOT_ELIGIBLE, TUNED_HGB: ELIGIBLE}, None, 5)

    assert selected == TUNED_HGB


def test_selection_takes_t2_when_both_are_eligible_and_t2_wins_head_to_head() -> None:
    selected, rationale = select_candidate({TUNED_LOGISTIC: ELIGIBLE, TUNED_HGB: ELIGIBLE}, WINS, 5)

    assert selected == TUNED_HGB
    assert "direct comparison" in rationale


def test_selection_takes_t1_when_both_are_eligible_and_t1_wins_head_to_head() -> None:
    selected, rationale = select_candidate(
        {TUNED_LOGISTIC: ELIGIBLE, TUNED_HGB: ELIGIBLE}, LOSES, 5
    )

    assert selected == TUNED_LOGISTIC
    assert "direct comparison" in rationale


def test_selection_prefers_logistic_when_both_are_eligible_without_a_clear_winner() -> None:
    selected, rationale = select_candidate(
        {TUNED_LOGISTIC: ELIGIBLE, TUNED_HGB: ELIGIBLE}, MIXED, 5
    )

    assert selected == TUNED_LOGISTIC
    assert "interpretability" in rationale


def test_selection_requires_the_head_to_head_when_both_are_eligible() -> None:
    with pytest.raises(ValueError, match="T2-vs-T1"):
        select_candidate({TUNED_LOGISTIC: ELIGIBLE, TUNED_HGB: ELIGIBLE}, None, 5)


# --- audits -------------------------------------------------------------------

_PHASE8B_SOURCES = [
    PROJECT_ROOT / "src" / "churn" / "modeling" / "tuning.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "tuning_results.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "tuning_plots.py",
    PROJECT_ROOT / "scripts" / "run_tuning.py",
]

FORBIDDEN_IDENTIFIERS = (
    "CalibratedClassifierCV",
    "IsotonicRegression",
    "TunedThresholdClassifierCV",
    "brier_score_loss",
    "SMOTE",
    "RandomOverSampler",
    "RandomForestClassifier",
    "build_random_forest",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
    "optuna",
    "hyperopt",
    "build_native_pipeline",
    "build_hgb_native",
    "CONTRACT_TENURE",
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


@pytest.mark.parametrize("source", _PHASE8B_SOURCES, ids=lambda path: path.name)
def test_phase8b_code_never_reaches_the_holdout(source) -> None:
    used = _identifiers(source)

    assert "load_holdout" not in used, f"{source.name} imports or calls the holdout loader"
    assert "holdout" not in used, f"{source.name} reads a holdout partition"


@pytest.mark.parametrize("source", _PHASE8B_SOURCES, ids=lambda path: path.name)
def test_phase8b_code_stays_inside_its_scope(source) -> None:
    used = _identifiers(source)

    found = sorted(name for name in FORBIDDEN_IDENTIFIERS if name in used)
    assert not found, (
        f"{source.name} references {found}: this phase does not calibrate, move the threshold, "
        "reopen random forest, reopen the native representation or use an engineered feature"
    )


@pytest.mark.parametrize("source", _PHASE8B_SOURCES, ids=lambda path: path.name)
def test_only_the_script_loads_data_and_only_the_training_pool(source) -> None:
    used = _identifiers(source)

    if source.name == "run_tuning.py":
        assert "load_training_pool" in used
        assert "load_split" not in used
        return
    assert "load_training_pool" not in used
    assert "load_raw_typed" not in used


def test_no_grid_search_is_constructed_outside_the_declared_builders() -> None:
    """Every search must go through build_search or run_final_search, which fix the scoring."""
    for source in _PHASE8B_SOURCES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "GridSearchCV":
                assert source.name == "tuning.py", (
                    f"{source.name} builds a GridSearchCV directly; it must use build_search "
                    "or run_final_search so the scoring and the refit policy cannot drift"
                )


def test_the_modern_builder_is_a_logistic_regression() -> None:
    assert isinstance(build_modern_logistic(), LogisticRegression)


# --- determinism --------------------------------------------------------------

_CLOCK_CALLS = ("today", "now", "utcnow", "time", "monotonic", "perf_counter", "getpid")


def test_the_report_generator_reads_no_clock() -> None:
    """A generation date guarantees a spurious diff on every rerun; Git dates files."""
    source = PROJECT_ROOT / "scripts" / "run_tuning.py"
    used = _identifiers(source)

    found = sorted(name for name in ("datetime", "date", "time", *_CLOCK_CALLS) if name in used)
    assert not found, (
        f"run_tuning.py reads wall-clock state via {found}: the Phase 8B report and artefact "
        "must be pure functions of their inputs so a rerun reproduces them byte for byte"
    )


def test_the_generated_report_carries_no_date() -> None:
    report = PROJECT_ROOT / "reports" / "tuning_report.md"
    if not report.is_file():
        pytest.skip("the report has not been generated yet")

    text = report.read_text(encoding="utf-8")
    assert "Generated on" not in text
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text), "an ISO date leaked into the report"


def test_the_artefact_carries_no_timestamp() -> None:
    results = PROJECT_ROOT / "reports" / "experiments" / "tuning_results.json"
    if not results.is_file():
        pytest.skip("the artefact has not been generated yet")

    raw = results.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert not [key for key in payload if re.search("time|stamp|generated", key, re.I)]
    assert not re.search(r"\d{4}-\d{2}-\d{2}", raw)


def test_the_frozen_candidate_is_the_modern_implementation() -> None:
    """Phase 9 must instantiate the modern spelling, never the legacy builder."""
    results = PROJECT_ROOT / "reports" / "experiments" / "tuning_results.json"
    if not results.is_file():
        pytest.skip("the artefact has not been generated yet")

    selection = json.loads(results.read_text(encoding="utf-8"))["selection"]

    assert selection["frozen_hyperparameters"] == {
        "C": 1.0,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "class_weight": None,
        "max_iter": 100,
    }
    assert "build_modern_logistic" in selection["estimator_implementation"]
    assert "HISTORICAL REPRODUCTION ONLY" in selection["legacy_builder_status"]
    assert "DEFERRED" in selection["class_weight_status"]
    assert len(selection["phase9_sequence"]) == 5
    assert selection["phase9_sequence"][3].startswith("D.")


def test_calibration_is_decided_before_the_threshold() -> None:
    """Calibration rewrites the probabilities a threshold acts on, so it comes first."""
    results = PROJECT_ROOT / "reports" / "experiments" / "tuning_results.json"
    if not results.is_file():
        pytest.skip("the artefact has not been generated yet")

    sequence = json.loads(results.read_text(encoding="utf-8"))["selection"]["phase9_sequence"]

    assert [text[:2] for text in sequence] == ["A.", "B.", "C.", "D.", "E."]

    calibration = next(index for index, text in enumerate(sequence) if "CALIBRATION GATE" in text)
    threshold = next(index for index, text in enumerate(sequence) if "THRESHOLD POLICY" in text)
    freeze = next(index for index, text in enumerate(sequence) if "FREEZE" in text)
    holdout = next(
        index for index, text in enumerate(sequence) if "FINAL HOLDOUT EVALUATION" in text
    )
    post_hoc = next(
        index for index, text in enumerate(sequence) if "POST-HOC ERROR ANALYSIS" in text
    )

    assert calibration < threshold < freeze < holdout < post_hoc
    assert (calibration, holdout) == (0, 3)


def test_the_holdout_is_evaluated_only_after_everything_is_frozen() -> None:
    results = PROJECT_ROOT / "reports" / "experiments" / "tuning_results.json"
    if not results.is_file():
        pytest.skip("the artefact has not been generated yet")

    sequence = json.loads(results.read_text(encoding="utf-8"))["selection"]["phase9_sequence"]

    # No step before D may evaluate the holdout, and the post-hoc step must
    # forbid a second one rather than merely discourage it.
    for text in sequence[:3]:
        assert "FINAL HOLDOUT EVALUATION" not in text
    assert "must not trigger" in sequence[4]
    assert "class_weight" in sequence[4], "the feedback ban has to name class_weight too"
