"""The Phase 7 families: declared configurations and a shared representation.

A family comparison is only a family comparison if exactly one thing differs
between the experiments. These tests pin that: the same preprocessing block, the
same 19 features, the same folds downstream, and estimators whose configuration
is the declared one rather than whatever a default happened to be.

They also pin the two things this phase must *not* contain — a search object and
a calibrator — because both would silently change what the reported numbers mean.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn.config import PROJECT_ROOT, get_config
from churn.features.groups import CONTRACT_TENURE, FEATURE_GROUPS
from churn.features.pipeline import FEATURE_STEP_PREFIX, transformed_width
from churn.modeling.families import (
    FAMILY_ORDER,
    HIST_GRADIENT_BOOSTING,
    HIST_GRADIENT_BOOSTING_PARAMS,
    RANDOM_FOREST,
    RANDOM_FOREST_PARAMS,
    build_family_pipeline,
    build_hist_gradient_boosting,
    build_random_forest,
)
from churn.modeling.models import (
    CLASSIFIER_STEP,
    LOGISTIC_REGRESSION,
    PREPROCESSOR_STEP,
    build_logistic_baseline,
)
from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_matrix,
)
from churn.preprocessing.pipeline import CATEGORICAL_STEP, CLEANER_STEP, ENCODER_STEP, NUMERIC_STEP
from churn.preprocessing.target import encode_target

#: The configuration this phase declared before any result existed. Written out
#: literally, not imported from the module under test: a test that reads the
#: same constant it checks would pass after an accidental edit.
DECLARED_RANDOM_FOREST = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "oob_score": False,
    "class_weight": None,
    "random_state": 42,
    "n_jobs": 1,
}

DECLARED_HIST_GRADIENT_BOOSTING = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "early_stopping": False,
    "class_weight": None,
    "categorical_features": None,
    "random_state": 42,
}

DECLARED_LOGISTIC = {
    "C": 1.0,
    "penalty": "l2",
    "class_weight": None,
    "solver": "lbfgs",
    "max_iter": 100,
}


@pytest.fixture
def training_xy(contract_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic feature matrix and encoded target with the real schema."""
    return build_feature_matrix(contract_frame), encode_target(
        contract_frame[get_config().target.column]
    )


# --- declared configurations ---------------------------------------------------


def test_random_forest_matches_the_declared_configuration() -> None:
    parameters = build_random_forest().get_params()

    for key, value in DECLARED_RANDOM_FOREST.items():
        assert parameters[key] == value, f"random forest {key} drifted from the protocol"


def test_hist_gradient_boosting_matches_the_declared_configuration() -> None:
    parameters = build_hist_gradient_boosting().get_params()

    for key, value in DECLARED_HIST_GRADIENT_BOOSTING.items():
        assert parameters[key] == value, f"boosting {key} drifted from the protocol"


def test_hist_gradient_boosting_disables_early_stopping() -> None:
    """The default would select a stopping point inside each training fold."""
    estimator = build_hist_gradient_boosting()

    assert estimator.get_params()["early_stopping"] is False
    assert estimator.get_params()["categorical_features"] is None


def test_logistic_family_is_the_frozen_phase5_classifier() -> None:
    """M0 must be the Phase 5 estimator itself, not a re-declaration of it."""
    phase7 = build_family_pipeline(LOGISTIC_REGRESSION).named_steps[CLASSIFIER_STEP]
    phase5 = build_logistic_baseline().named_steps[CLASSIFIER_STEP]

    assert isinstance(phase7, LogisticRegression)
    assert phase7.get_params() == phase5.get_params()
    for key, value in DECLARED_LOGISTIC.items():
        assert phase7.get_params()[key] == value


def test_declared_parameters_are_the_ones_the_module_ships() -> None:
    """The module constants may not drift from the declared protocol either."""
    assert RANDOM_FOREST_PARAMS == {
        key: value for key, value in DECLARED_RANDOM_FOREST.items() if key != "random_state"
    }
    assert HIST_GRADIENT_BOOSTING_PARAMS == {
        key: value
        for key, value in DECLARED_HIST_GRADIENT_BOOSTING.items()
        if key != "random_state"
    }


def test_no_family_uses_class_weight() -> None:
    for family in FAMILY_ORDER:
        classifier = build_family_pipeline(family).named_steps[CLASSIFIER_STEP]
        assert classifier.get_params()["class_weight"] is None, f"{family} weights its classes"


def test_estimator_types_are_the_three_declared_families() -> None:
    types = {
        family: type(build_family_pipeline(family).named_steps[CLASSIFIER_STEP])
        for family in FAMILY_ORDER
    }

    assert types == {
        LOGISTIC_REGRESSION: LogisticRegression,
        RANDOM_FOREST: RandomForestClassifier,
        HIST_GRADIENT_BOOSTING: HistGradientBoostingClassifier,
    }


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown model family"):
        build_family_pipeline("xgboost")


# --- one representation for every family --------------------------------------


def _encoder(pipeline):
    return pipeline.named_steps[PREPROCESSOR_STEP].named_steps[ENCODER_STEP]


def test_every_family_shares_the_same_preprocessing() -> None:
    references = None
    for family in FAMILY_ORDER:
        pipeline = build_family_pipeline(family)
        preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]

        assert list(preprocessor.named_steps) == [CLEANER_STEP, ENCODER_STEP]
        columns = {
            name: tuple(cols) for name, _, cols in _encoder(pipeline).transformers if name != "drop"
        }
        assert columns[NUMERIC_STEP] == NUMERIC_FEATURES
        assert columns[CATEGORICAL_STEP] == CATEGORICAL_FEATURES

        if references is None:
            references = columns
        assert columns == references, f"{family} sees a different representation"


def test_main_comparison_pipelines_differ_only_in_the_classifier() -> None:
    shapes = {
        family: [
            (name, type(step).__name__)
            for name, step in build_family_pipeline(family)
            .named_steps[PREPROCESSOR_STEP]
            .named_steps.items()
        ]
        for family in FAMILY_ORDER
    }

    assert len(set(map(str, shapes.values()))) == 1


def test_every_family_receives_the_same_transformed_width(training_xy) -> None:
    features, target = training_xy
    widths = {
        family: transformed_width(build_family_pipeline(family).fit(features, target))
        for family in FAMILY_ORDER
    }

    assert len(set(widths.values())) == 1, f"representation width differs per family: {widths}"


# --- the sensitivity analysis changes one thing --------------------------------


def test_sensitivity_pipeline_adds_only_contract_tenure() -> None:
    for family in FAMILY_ORDER:
        main = build_family_pipeline(family).named_steps[PREPROCESSOR_STEP]
        sensitivity = build_family_pipeline(family, (CONTRACT_TENURE,)).named_steps[
            PREPROCESSOR_STEP
        ]

        added = [name for name in sensitivity.named_steps if name not in main.named_steps]
        assert added == [f"{FEATURE_STEP_PREFIX}{CONTRACT_TENURE}"]


def test_sensitivity_keeps_the_identical_classifier() -> None:
    for family in FAMILY_ORDER:
        main = build_family_pipeline(family).named_steps[CLASSIFIER_STEP]
        sensitivity = build_family_pipeline(family, (CONTRACT_TENURE,)).named_steps[CLASSIFIER_STEP]

        assert type(main) is type(sensitivity)
        assert main.get_params() == sensitivity.get_params()


def test_sensitivity_widens_the_matrix_by_the_feature_columns(training_xy) -> None:
    features, target = training_xy
    expected = len(FEATURE_GROUPS[CONTRACT_TENURE].numeric)

    for family in FAMILY_ORDER:
        main = transformed_width(build_family_pipeline(family).fit(features, target))
        widened = transformed_width(
            build_family_pipeline(family, (CONTRACT_TENURE,)).fit(features, target)
        )
        assert widened - main == expected


# --- audits --------------------------------------------------------------------

_PHASE7_SOURCES = [
    PROJECT_ROOT / "src" / "churn" / "modeling" / "families.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "comparison.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "comparison_results.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "comparison_plots.py",
    PROJECT_ROOT / "scripts" / "run_model_comparison.py",
]

#: Anything that would turn this phase into a search, or that would change what
#: a predicted probability means before it is measured.
FORBIDDEN_IDENTIFIERS = (
    "GridSearchCV",
    "RandomizedSearchCV",
    "HalvingGridSearchCV",
    "HalvingRandomSearchCV",
    "BayesSearchCV",
    "optuna",
    "hyperopt",
    "CalibratedClassifierCV",
    "IsotonicRegression",
    "brier_score_loss",
    "SMOTE",
    "RandomOverSampler",
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


@pytest.mark.parametrize("source", _PHASE7_SOURCES, ids=lambda path: path.name)
def test_phase7_code_never_reaches_the_holdout(source) -> None:
    used = _identifiers(source)

    assert "load_holdout" not in used, f"{source.name} imports or calls the holdout loader"
    assert "holdout" not in used, f"{source.name} reads a holdout partition"


@pytest.mark.parametrize("source", _PHASE7_SOURCES, ids=lambda path: path.name)
def test_phase7_code_creates_no_search_or_calibration_object(source) -> None:
    used = _identifiers(source)

    found = sorted(name for name in FORBIDDEN_IDENTIFIERS if name in used)
    assert not found, f"{source.name} references {found}: this phase neither tunes nor calibrates"


@pytest.mark.parametrize("source", _PHASE7_SOURCES, ids=lambda path: path.name)
def test_phase7_code_never_fits_on_the_whole_dataset_loader(source) -> None:
    """Only the script may load data, and only the training pool."""
    used = _identifiers(source)

    if source.name == "run_model_comparison.py":
        assert "load_training_pool" in used
        assert "load_split" not in used
        return
    assert "load_training_pool" not in used
    assert "load_raw_typed" not in used
