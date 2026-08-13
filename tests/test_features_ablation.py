"""The ablation protocol: same folds, same model, one variable.

An ablation study is only as good as its controls. These tests pin the controls:
that a candidate pipeline differs from the baseline in the feature set and in
nothing else, that the folds are the ones Phase 5 used, and that the
classification rule is the one the protocol declares.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest

from churn.config import PROJECT_ROOT
from churn.features.ablation import (
    INCONCLUSIVE,
    NOT_SUPPORTED,
    PRIMARY_METRIC,
    PROMISING,
    SECONDARY_METRIC,
    PairedComparison,
    classify,
    compare,
    promising_groups,
    run_experiment,
)
from churn.features.groups import ABLATION_ORDER, FEATURE_GROUPS, added_numeric, resolve
from churn.features.pipeline import build_model_pipeline
from churn.modeling.evaluation import build_splitter, evaluate_model
from churn.modeling.models import CLASSIFIER_STEP, build_logistic_baseline
from churn.preprocessing.contracts import NUMERIC_FEATURES, build_feature_matrix
from churn.preprocessing.pipeline import ENCODER_STEP, NUMERIC_STEP
from churn.preprocessing.target import encode_target


@pytest.fixture
def training_xy(contract_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return build_feature_matrix(contract_frame), encode_target(contract_frame["Churn"])


def _deltas(values: list[float]) -> PairedComparison:
    return PairedComparison(metric=PRIMARY_METRIC, deltas=tuple(values))


# --- pipeline construction ----------------------------------------------------


def test_empty_feature_groups_reproduce_the_baseline_pipeline(
    training_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_xy
    baseline = build_model_pipeline()
    from churn.preprocessing.pipeline import build_preprocessor

    phase4 = build_preprocessor()

    baseline_steps = list(baseline.named_steps["preprocessor"].named_steps)

    assert baseline_steps == list(phase4.named_steps)
    assert np.allclose(
        baseline.fit(features, target).predict_proba(features),
        build_model_pipeline([]).fit(features, target).predict_proba(features),
    )


@pytest.mark.parametrize("group", ABLATION_ORDER)
def test_each_group_adds_only_its_own_columns(
    training_xy: tuple[pd.DataFrame, pd.Series], group: str
) -> None:
    features, target = training_xy
    baseline = build_model_pipeline().fit(features, target)
    candidate = build_model_pipeline([group]).fit(features, target)

    def numeric_columns(pipeline):
        encoder = pipeline.named_steps["preprocessor"].named_steps[ENCODER_STEP]
        return list(encoder.named_transformers_[NUMERIC_STEP].feature_names_in_)

    added = set(numeric_columns(candidate)) - set(numeric_columns(baseline))

    assert added == set(FEATURE_GROUPS[group].numeric)


@pytest.mark.parametrize("group", ABLATION_ORDER)
def test_no_group_changes_the_classifier(group: str) -> None:
    """One definition of the estimator; a feature group cannot retune it."""
    reference = build_logistic_baseline().named_steps[CLASSIFIER_STEP].get_params()

    candidate = build_model_pipeline([group]).named_steps[CLASSIFIER_STEP].get_params()

    assert candidate == reference


@pytest.mark.parametrize("group", ABLATION_ORDER)
def test_no_group_removes_an_original_feature(
    training_xy: tuple[pd.DataFrame, pd.Series], group: str
) -> None:
    features, target = training_xy
    candidate = build_model_pipeline([group]).fit(features, target)
    encoder = candidate.named_steps["preprocessor"].named_steps[ENCODER_STEP]
    numeric = list(encoder.named_transformers_[NUMERIC_STEP].feature_names_in_)

    assert set(NUMERIC_FEATURES) <= set(numeric)


def test_unknown_group_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown feature group"):
        build_model_pipeline(["not_a_group"])


def test_resolve_returns_registry_order() -> None:
    assert [group.name for group in resolve(list(reversed(ABLATION_ORDER)))] == list(ABLATION_ORDER)


def test_added_numeric_matches_the_registry() -> None:
    groups = resolve(ABLATION_ORDER)

    assert added_numeric(groups) == tuple(
        column for name in ABLATION_ORDER for column in FEATURE_GROUPS[name].numeric
    )


# --- paired comparison --------------------------------------------------------


def test_experiments_share_the_baseline_folds(
    training_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_xy
    splitter = build_splitter()
    baseline = evaluate_model("E0", build_model_pipeline(), features, target, splitter)
    candidate = evaluate_model(
        "E3", build_model_pipeline(["contract_tenure"]), features, target, splitter
    )

    assert [f.fold for f in baseline.folds] == [f.fold for f in candidate.folds]
    assert [f.n_validation for f in baseline.folds] == [f.n_validation for f in candidate.folds]


def test_deltas_are_computed_fold_by_fold_in_order(
    training_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_xy
    splitter = build_splitter()
    baseline = evaluate_model("E0", build_model_pipeline(), features, target, splitter)
    candidate = evaluate_model(
        "E1", build_model_pipeline(["protective_services"]), features, target, splitter
    )

    comparison = compare(candidate, baseline, PRIMARY_METRIC)
    expected = [
        getattr(c.metrics, PRIMARY_METRIC) - getattr(b.metrics, PRIMARY_METRIC)
        for c, b in zip(candidate.folds, baseline.folds, strict=True)
    ]

    assert list(comparison.deltas) == pytest.approx(expected)


def test_comparison_rejects_mismatched_folds(
    training_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_xy
    five = evaluate_model("a", build_model_pipeline(), features, target, build_splitter())
    from sklearn.model_selection import StratifiedKFold

    four = evaluate_model(
        "b",
        build_model_pipeline(),
        features,
        target,
        StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
    )

    with pytest.raises(ValueError, match="same folds"):
        compare(four, five, PRIMARY_METRIC)


def test_paired_comparison_aggregates_correctly() -> None:
    comparison = _deltas([0.01, -0.02, 0.03, 0.0, 0.04])

    assert comparison.mean == pytest.approx(np.mean([0.01, -0.02, 0.03, 0.0, 0.04]))
    assert comparison.std == pytest.approx(np.std([0.01, -0.02, 0.03, 0.0, 0.04], ddof=1))
    assert comparison.n_positive == 3
    assert comparison.n_negative == 1


# --- classification -----------------------------------------------------------


def _secondary(mean: float) -> PairedComparison:
    return PairedComparison(metric=SECONDARY_METRIC, deltas=(mean,) * 5)


def test_consistent_gain_is_promising() -> None:
    label, _ = classify(_deltas([0.01, 0.02, 0.01, -0.001, 0.03]), _secondary(0.01))

    assert label == PROMISING


def test_gain_in_all_folds_is_promising() -> None:
    label, _ = classify(_deltas([0.01] * 5), _secondary(0.005))

    assert label == PROMISING


def test_negative_mean_is_not_supported() -> None:
    label, _ = classify(_deltas([0.01, -0.05, 0.01, 0.01, 0.01]), _secondary(0.01))

    assert label == NOT_SUPPORTED


def test_majority_of_folds_worse_is_not_supported() -> None:
    label, _ = classify(_deltas([0.10, -0.01, -0.01, -0.01, 0.001]), _secondary(0.01))

    assert label == NOT_SUPPORTED


def test_three_of_five_folds_is_inconclusive() -> None:
    label, _ = classify(_deltas([0.02, 0.02, 0.02, -0.01, -0.01]), _secondary(0.01))

    assert label == INCONCLUSIVE


def test_secondary_metric_moving_against_is_inconclusive() -> None:
    """A trade-off is a judgement call, not a clean gain."""
    label, rationale = classify(_deltas([0.01] * 5), _secondary(-0.02))

    assert label == INCONCLUSIVE
    assert "trade-off" in rationale


def test_classification_has_no_minimum_gain_cutoff() -> None:
    """A tiny but perfectly consistent gain is still PROMISING by the protocol."""
    label, _ = classify(_deltas([1e-9] * 5), _secondary(1e-9))

    assert label == PROMISING


# --- combination --------------------------------------------------------------


def test_combination_uses_only_promising_groups(
    training_xy: tuple[pd.DataFrame, pd.Series],
) -> None:
    features, target = training_xy
    splitter = build_splitter()
    baseline = run_experiment("E0", "baseline", (), features, target, splitter)
    results = [
        baseline,
        run_experiment(
            "E1", "a", ("protective_services",), features, target, splitter, baseline.evaluation
        ),
        run_experiment(
            "E3", "b", ("contract_tenure",), features, target, splitter, baseline.evaluation
        ),
    ]

    winners = promising_groups(results)

    assert all(
        result.classification == PROMISING
        for result in results
        if set(result.feature_groups) & set(winners)
    )
    assert set(winners) <= set(ABLATION_ORDER)


def test_promising_groups_ignores_other_classifications() -> None:
    from churn.features.ablation import ExperimentResult

    def result(name: str, classification: str) -> ExperimentResult:
        return ExperimentResult(
            experiment=name,
            label=name,
            feature_groups=(name,),
            n_transformed_features=46,
            evaluation=None,  # type: ignore[arg-type]
            comparisons={},
            classification=classification,
            rationale="",
        )

    winners = promising_groups(
        [
            result("a", PROMISING),
            result("b", INCONCLUSIVE),
            result("c", NOT_SUPPORTED),
        ]
    )

    assert winners == ("a",)


# --- leakage audit ------------------------------------------------------------

_FEATURE_MODULES = sorted((PROJECT_ROOT / "src" / "churn" / "features").glob("*.py"))
_ORCHESTRATION_SCRIPT = PROJECT_ROOT / "scripts" / "run_feature_engineering.py"
_PHASE6_SOURCES = [*_FEATURE_MODULES, _ORCHESTRATION_SCRIPT]


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


@pytest.mark.parametrize("source", _PHASE6_SOURCES, ids=lambda path: path.name)
def test_phase6_code_never_reaches_the_holdout(source) -> None:
    used = _identifiers(source)

    assert "load_holdout" not in used, f"{source.name} imports or calls the holdout loader"
    assert "holdout" not in used, f"{source.name} reads a holdout partition"


@pytest.mark.parametrize("source", _FEATURE_MODULES, ids=lambda path: path.name)
def test_no_candidate_feature_reads_the_target(source) -> None:
    """No feature module may touch the target. Only the script may, to build y."""
    used = _identifiers(source)

    assert "Churn" not in used
    assert "encode_target" not in used


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def test_script_touches_the_target_only_to_build_y() -> None:
    """The one source allowed to read the target is audited, not exempted.

    The orchestration script must encode the target — that is how ``y`` comes to
    exist — so the module-level rule above cannot apply to it. It is checked
    here instead: the target is encoded exactly once, and the feature matrix is
    built without it.
    """
    tree = ast.parse(_ORCHESTRATION_SCRIPT.read_text(encoding="utf-8"))

    encodings = _calls(tree, "encode_target")
    assert len(encodings) == 1, "the target is encoded once, to build y, and nowhere else"

    matrices = _calls(tree, "build_feature_matrix")
    assert matrices, "the script builds the feature matrix"
    for call in matrices:
        referenced = {node.id for node in ast.walk(call) if isinstance(node, ast.Name)}
        assert not referenced & {"target", "labels"}, "the feature matrix must not see the target"


def test_ablation_module_cannot_load_data() -> None:
    used = _identifiers(PROJECT_ROOT / "src" / "churn" / "features" / "ablation.py")

    assert "load_training_pool" not in used
    assert "load_split" not in used
    assert "load_raw_typed" not in used
