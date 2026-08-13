"""The Phase 7 protocol: same folds, paired deltas, honest out-of-fold scores.

The properties pinned here are the ones that would silently invalidate a family
comparison if they broke: folds shared across experiments, deltas subtracted in
fold order, every row predicted exactly once by a model that never saw it, the
sensitivity analysis paired inside a family, and the two reproduction gates.

The isolation tests run on the synthetic contract fixture. The two reproduction
tests need the real artefacts and are skipped while the dataset has not been
acquired, so the suite stays green on a fresh clone.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from churn.config import get_config
from churn.data.loader import raw_csv_path
from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC
from churn.features.groups import CONTRACT_TENURE
from churn.features.results import RESULTS_PATH as ABLATION_RESULTS_PATH
from churn.features.results import load_results as load_ablation_results
from churn.modeling.comparison import (
    SENSITIVITY_GROUPS,
    collect_warnings,
    ranked_by,
    run_family,
    run_main_comparison,
    run_sensitivity,
)
from churn.modeling.comparison_results import (
    ReproductionError,
    verify_baseline,
    verify_contract_tenure,
    verify_reproduction,
)
from churn.modeling.evaluation import build_splitter, evaluate_model
from churn.modeling.families import (
    FAMILY_ORDER,
    LOGISTIC_REGRESSION,
    MAIN_EXPERIMENTS,
    SENSITIVITY_EXPERIMENTS,
    build_family_pipeline,
)
from churn.modeling.metrics import DEFAULT_THRESHOLD, METRIC_NAMES, compute_metrics
from churn.modeling.results import RESULTS_PATH as BASELINE_RESULTS_PATH
from churn.modeling.results import load_results as load_baseline_results
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target


@pytest.fixture(scope="module")
def training_xy(comparison_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic features and target, large enough for a stratified 5-fold split."""
    return (
        build_feature_matrix(comparison_frame),
        encode_target(comparison_frame[get_config().target.column]),
    )


@pytest.fixture(scope="module")
def main_runs(training_xy):
    features, target = training_xy
    return run_main_comparison(features, target, build_splitter())


@pytest.fixture(scope="module")
def sensitivity_runs(training_xy, main_runs):
    features, target = training_xy
    return run_sensitivity(features, target, build_splitter(), main_runs)


# --- structure of the comparison ----------------------------------------------


def test_main_comparison_runs_every_family_once(main_runs) -> None:
    assert list(main_runs) == [MAIN_EXPERIMENTS[family] for family in FAMILY_ORDER]
    assert [run.family for run in main_runs.values()] == list(FAMILY_ORDER)


def test_the_logistic_run_is_the_reference_and_has_no_deltas(main_runs) -> None:
    baseline = main_runs["M0"]

    assert baseline.family == LOGISTIC_REGRESSION
    assert baseline.is_reference
    assert baseline.deltas == {}


def test_every_candidate_is_paired_against_the_logistic_run(main_runs) -> None:
    for experiment, run in main_runs.items():
        if experiment == "M0":
            continue
        assert run.reference == "M0"
        assert set(run.deltas) == {PRIMARY_METRIC, SECONDARY_METRIC}


def test_main_comparison_uses_only_the_original_features(main_runs) -> None:
    for run in main_runs.values():
        assert run.feature_groups == ()


def test_every_experiment_uses_the_same_folds(main_runs, sensitivity_runs) -> None:
    shapes = {
        run.experiment: [
            (fold.fold, fold.n_train, fold.n_validation) for fold in run.evaluation.folds
        ]
        for run in (*main_runs.values(), *sensitivity_runs.values())
    }
    reference = shapes["M0"]

    for experiment, folds in shapes.items():
        assert folds == reference, f"{experiment} did not see the Phase 5 folds"


def test_the_splitter_is_deterministic(training_xy) -> None:
    features, target = training_xy
    first = [tuple(validation) for _, validation in build_splitter().split(features, target)]
    second = [tuple(validation) for _, validation in build_splitter().split(features, target)]

    assert first == second


def test_paired_deltas_follow_the_fold_order(main_runs) -> None:
    baseline = main_runs["M0"].evaluation

    for experiment, run in main_runs.items():
        if experiment == "M0":
            continue
        for metric, comparison in run.deltas.items():
            expected = tuple(
                getattr(candidate.metrics, metric) - getattr(reference.metrics, metric)
                for candidate, reference in zip(run.evaluation.folds, baseline.folds, strict=True)
            )
            assert comparison.deltas == pytest.approx(expected, abs=1e-12)


def test_delta_fold_counts_add_up(main_runs) -> None:
    for run in main_runs.values():
        for comparison in run.deltas.values():
            assert comparison.n_positive + comparison.n_negative <= len(comparison.deltas)


# --- out-of-fold honesty -------------------------------------------------------


def test_out_of_fold_covers_every_row_exactly_once(main_runs, training_xy) -> None:
    features, _ = training_xy

    for run in main_runs.values():
        oof = run.evaluation.oof_probability
        assert oof.shape == (len(features),)
        assert not np.isnan(oof).any()
        assert sum(fold.n_validation for fold in run.evaluation.folds) == len(features)


class _MemorisingClassifier(ClassifierMixin, BaseEstimator):
    """Scores 1.0 for a row it was fitted on and 0.0 for anything else."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _MemorisingClassifier:
        self.seen_ = set(X.index)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        seen = np.fromiter((index in self.seen_ for index in X.index), dtype=float, count=len(X))
        return np.column_stack([1.0 - seen, seen])


def test_no_row_is_predicted_by_the_model_that_trained_on_it(training_xy) -> None:
    """The single evaluation path every Phase 7 experiment goes through.

    The probe returns 1.0 for any row present in its own training fold, so a
    non-zero out-of-fold score would be direct evidence of a row being scored by
    the model that learned it.
    """
    features, target = training_xy

    evaluation = evaluate_model(
        "probe", _MemorisingClassifier(), features, target, build_splitter()
    )

    assert evaluation.oof_probability.max() == 0.0


def test_out_of_fold_metrics_use_the_default_threshold(main_runs, training_xy) -> None:
    _, target = training_xy
    labels = np.asarray(target)

    for run in main_runs.values():
        recomputed = compute_metrics(labels, run.evaluation.oof_probability, DEFAULT_THRESHOLD)
        for metric in METRIC_NAMES:
            assert getattr(run.evaluation.oof_metrics, metric) == pytest.approx(
                getattr(recomputed, metric)
            )


def test_the_decision_threshold_is_still_the_untouched_default() -> None:
    assert DEFAULT_THRESHOLD == 0.5


# --- sensitivity analysis ------------------------------------------------------


def test_sensitivity_runs_every_family_with_contract_tenure(sensitivity_runs) -> None:
    assert list(sensitivity_runs) == [SENSITIVITY_EXPERIMENTS[family] for family in FAMILY_ORDER]
    assert SENSITIVITY_GROUPS == (CONTRACT_TENURE,)

    for run in sensitivity_runs.values():
        assert run.feature_groups == (CONTRACT_TENURE,)


def test_sensitivity_pairs_each_family_against_itself(sensitivity_runs) -> None:
    for run in sensitivity_runs.values():
        assert run.reference == MAIN_EXPERIMENTS[run.family]


def test_sensitivity_changes_only_the_feature_set(main_runs, sensitivity_runs) -> None:
    for run in sensitivity_runs.values():
        counterpart = main_runs[MAIN_EXPERIMENTS[run.family]]

        assert run.family == counterpart.family
        assert run.n_transformed_features > counterpart.n_transformed_features
        assert (
            build_family_pipeline(run.family, run.feature_groups)
            .named_steps["classifier"]
            .get_params()
            == build_family_pipeline(run.family).named_steps["classifier"].get_params()
        )


def test_sensitivity_deltas_are_paired_on_the_same_folds(main_runs, sensitivity_runs) -> None:
    for run in sensitivity_runs.values():
        reference = main_runs[MAIN_EXPERIMENTS[run.family]].evaluation
        expected = tuple(
            getattr(candidate.metrics, PRIMARY_METRIC) - getattr(base.metrics, PRIMARY_METRIC)
            for candidate, base in zip(run.evaluation.folds, reference.folds, strict=True)
        )
        assert run.deltas[PRIMARY_METRIC].deltas == pytest.approx(expected, abs=1e-12)


def test_no_rejected_phase6_candidate_is_reopened(sensitivity_runs) -> None:
    reopened = {
        group
        for run in sensitivity_runs.values()
        for group in run.feature_groups
        if group != CONTRACT_TENURE
    }

    assert not reopened


def test_ranking_orders_by_the_primary_metric(main_runs) -> None:
    ordered = ranked_by(list(main_runs.values()))
    values = [run.evaluation.mean(PRIMARY_METRIC) for run in ordered]

    assert values == sorted(values, reverse=True)


# --- reproduction machinery ----------------------------------------------------


def _reference_folds(evaluation) -> list[dict[str, float]]:
    return [
        {metric: round(float(getattr(fold.metrics, metric)), 6) for metric in METRIC_NAMES}
        for fold in evaluation.folds
    ]


def test_verify_reproduction_accepts_identical_metrics(main_runs) -> None:
    evaluation = main_runs["M0"].evaluation

    check = verify_reproduction(
        "M0",
        evaluation,
        _reference_folds(evaluation),
        "reports/experiments/fake.json",
        1,
        "models[probe]",
    )

    assert check.reproduced
    assert check.max_absolute_difference == 0.0


def test_verify_reproduction_rejects_a_changed_metric(main_runs) -> None:
    evaluation = main_runs["M0"].evaluation
    folds = _reference_folds(evaluation)
    folds[2][PRIMARY_METRIC] += 0.01

    with pytest.raises(ReproductionError, match="does not reproduce"):
        verify_reproduction(
            "M0", evaluation, folds, "reports/experiments/fake.json", 1, "models[probe]"
        )


def test_verify_reproduction_rejects_a_different_fold_count(main_runs) -> None:
    evaluation = main_runs["M0"].evaluation

    with pytest.raises(ReproductionError, match="folds against"):
        verify_reproduction(
            "M0",
            evaluation,
            _reference_folds(evaluation)[:-1],
            "reports/experiments/fake.json",
            1,
            "models[probe]",
        )


def test_collect_warnings_tallies_by_identity() -> None:
    with collect_warnings() as captured:
        for _ in range(3):
            warnings.warn("repeated", FutureWarning, stacklevel=1)
        warnings.warn("distinct", UserWarning, stacklevel=1)

    assert [(item.category, item.message, item.count) for item in captured] == [
        ("FutureWarning", "repeated", 3),
        ("UserWarning", "distinct", 1),
    ]


# --- the two gates, against the real frozen artefacts --------------------------

_artefacts_present = (
    raw_csv_path().is_file() and BASELINE_RESULTS_PATH.is_file() and ABLATION_RESULTS_PATH.is_file()
)

reproduction = pytest.mark.skipif(
    not _artefacts_present,
    reason="raw dataset or a frozen experiment record is missing (see data/README.md)",
)


@pytest.fixture(scope="module")
def real_training_xy():
    training = load_training_pool()
    return (
        build_feature_matrix(training),
        encode_target(training[get_config().target.column]),
    )


@reproduction
def test_m0_reproduces_the_phase5_baseline(real_training_xy) -> None:
    features, target = real_training_xy

    run = run_family("M0", LOGISTIC_REGRESSION, (), features, target, build_splitter())
    check = verify_baseline(run, load_baseline_results())

    assert check.reproduced
    assert check.max_absolute_difference == 0.0


@reproduction
def test_s0_reproduces_e3_of_the_phase6_ablation(real_training_xy) -> None:
    features, target = real_training_xy

    run = run_family(
        "S0", LOGISTIC_REGRESSION, SENSITIVITY_GROUPS, features, target, build_splitter()
    )
    check = verify_contract_tenure(run, load_ablation_results())

    assert check.reproduced
    assert check.max_absolute_difference == 0.0
