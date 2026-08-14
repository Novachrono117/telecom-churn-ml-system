"""The Phase 8A representation gate: layout, leakage, isolation and audits.

The expensive tests — the two reproduction gates against the frozen Phase 7
artefact — run on the real training pool and are marked ``slow`` nowhere,
because a gate that is not run is not a gate. Everything about the mechanics of
the representation is proved on synthetic frames instead, so that a failure
points at the code rather than at the dataset.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from churn.config import PROJECT_ROOT
from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC
from churn.modeling.comparison_results import load_results as load_comparison_results
from churn.modeling.evaluation import build_splitter, evaluate_model
from churn.modeling.families import HIST_GRADIENT_BOOSTING_PARAMS, build_family_pipeline
from churn.modeling.metrics import DEFAULT_THRESHOLD, METRIC_NAMES
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.modeling.representation import (
    CATEGORICAL_MASK,
    GUARD_STEP,
    MAX_CATEGORICAL_CARDINALITY,
    NATIVE_HGB,
    ORDINAL_ENCODER_PARAMS,
    REFERENCE_HGB_COMMON,
    REFERENCE_LOGISTIC,
    REPRESENTATION_HELPFUL,
    REPRESENTATION_INCONCLUSIVE,
    REPRESENTATION_NOT_SUPPORTED,
    CategoricalCardinalityError,
    CategoricalCardinalityGuard,
    build_hgb_native,
    build_native_column_encoder,
    build_native_pipeline,
    build_native_preprocessor,
    categorical_indices,
    classify_representation,
    observed_cardinality,
    run_representation_gate,
    standing_against_logistic,
)
from churn.modeling.representation_results import (
    REFERENCE_EXPERIMENTS,
    verify_against_comparison,
)
from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_feature_matrix,
)
from churn.preprocessing.pipeline import CATEGORICAL_STEP, ENCODER_STEP, NUMERIC_STEP
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target


@pytest.fixture(scope="module")
def synthetic_xy(comparison_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """A synthetic feature matrix and target, for isolation tests."""
    features = build_feature_matrix(comparison_frame)
    target = encode_target(comparison_frame["Churn"])
    return features, target


@pytest.fixture(scope="module")
def real_training_xy() -> tuple[pd.DataFrame, pd.Series]:
    """The frozen training pool. Never the holdout."""
    training = load_training_pool()
    return build_feature_matrix(training), encode_target(training["Churn"])


# --- representation layout ----------------------------------------------------


def test_the_mask_is_three_numeric_then_sixteen_categorical() -> None:
    assert len(CATEGORICAL_MASK) == len(FEATURE_COLUMNS) == 19
    assert CATEGORICAL_MASK[: len(NUMERIC_FEATURES)] == (False, False, False)
    assert all(CATEGORICAL_MASK[len(NUMERIC_FEATURES) :])
    assert sum(CATEGORICAL_MASK) == 16
    assert sum(not flag for flag in CATEGORICAL_MASK) == 3


def test_categorical_indices_match_the_mask() -> None:
    assert categorical_indices() == tuple(range(3, 19))


def test_r2_uses_exactly_the_nineteen_raw_features(synthetic_xy) -> None:
    features, target = synthetic_xy
    pipeline = build_native_pipeline().fit(features, target)

    encoder = pipeline.named_steps[PREPROCESSOR_STEP].named_steps[ENCODER_STEP]
    used = [column for _, _, columns in encoder.transformers_ for column in columns]
    assert used == list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    assert len(used) == 19


def test_r2_produces_nineteen_transformed_features(synthetic_xy) -> None:
    features, target = synthetic_xy
    pipeline = build_native_pipeline().fit(features, target)

    transformed = pipeline.named_steps[PREPROCESSOR_STEP].transform(features)
    assert transformed.shape[1] == 19
    assert transformed.shape[1] == len(CATEGORICAL_MASK)


def test_the_transformed_column_order_is_numeric_then_categorical(synthetic_xy) -> None:
    """The mask is positional, so this ordering is load bearing, not cosmetic."""
    features, target = synthetic_xy
    pipeline = build_native_pipeline().fit(features, target)
    encoder = pipeline.named_steps[PREPROCESSOR_STEP].named_steps[ENCODER_STEP]

    names = list(encoder.get_feature_names_out())
    prefixes = [name.split("__", 1)[0] for name in names]
    assert prefixes == [NUMERIC_STEP] * 3 + [CATEGORICAL_STEP] * 16
    assert [name.split("__", 1)[1] for name in names] == list(FEATURE_COLUMNS)


def test_numeric_branch_is_still_exactly_three_scaled_columns() -> None:
    encoder = build_native_column_encoder()
    name, transformer, columns = encoder.transformers[0]

    assert name == NUMERIC_STEP
    assert isinstance(transformer, StandardScaler)
    assert columns == list(NUMERIC_FEATURES)
    assert len(columns) == 3


def test_categorical_branch_is_the_sixteen_contracted_columns() -> None:
    encoder = build_native_column_encoder()
    name, transformer, columns = encoder.transformers[1]

    assert name == CATEGORICAL_STEP
    assert isinstance(transformer, OrdinalEncoder)
    assert columns == list(CATEGORICAL_FEATURES)
    assert len(columns) == 16


# --- the estimator ------------------------------------------------------------


def test_categorical_features_is_declared_explicitly_as_the_mask() -> None:
    classifier = build_hgb_native()

    assert classifier.categorical_features == list(CATEGORICAL_MASK)
    assert classifier.categorical_features != "from_dtype"
    assert classifier.categorical_features is not None


def test_native_hgb_disables_early_stopping() -> None:
    assert build_hgb_native().early_stopping is False


def test_r1_and_r2_differ_only_in_categorical_features() -> None:
    """The one-variable guarantee of this phase, asserted parameter by parameter."""
    common = build_family_pipeline("hist_gradient_boosting", ()).named_steps[CLASSIFIER_STEP]
    native = build_hgb_native()

    common_params = common.get_params()
    native_params = native.get_params()
    differing = {
        key
        for key in common_params.keys() | native_params.keys()
        if common_params.get(key) != native_params.get(key)
    }
    assert differing == {"categorical_features"}


def test_native_hgb_is_built_from_the_frozen_phase7_dictionary() -> None:
    params = build_hgb_native().get_params()

    for key, value in HIST_GRADIENT_BOOSTING_PARAMS.items():
        if key == "categorical_features":
            continue
        assert params[key] == value, f"{key} drifted away from the frozen Phase 7 configuration"


def test_no_experiment_uses_class_weight() -> None:
    assert build_hgb_native().class_weight is None


# --- the ordinal encoding is not an ordinal assumption ------------------------


def test_ordinal_encoder_uses_the_declared_unknown_policy() -> None:
    _, transformer, _ = build_native_column_encoder().transformers[1]
    params = transformer.get_params()

    assert params["handle_unknown"] == "use_encoded_value"
    assert ORDINAL_ENCODER_PARAMS["handle_unknown"] == "use_encoded_value"
    assert np.isnan(params["unknown_value"])
    assert params["dtype"] is np.float64


def test_codes_carry_no_continuous_interpretation(synthetic_xy) -> None:
    """Every categorical column arrives as a contiguous code set declared categorical.

    The codes being ``0..k-1`` is precisely why they must not be read as a
    quantity, and the mask is what stops that from happening.
    """
    features, target = synthetic_xy
    preprocessor = build_native_preprocessor().fit(features, target)
    transformed = preprocessor.transform(features)
    encoder = preprocessor.named_steps[ENCODER_STEP]
    ordinal = encoder.named_transformers_[CATEGORICAL_STEP]

    for offset, categories in enumerate(ordinal.categories_):
        column = transformed[:, len(NUMERIC_FEATURES) + offset]
        observed = np.unique(column[~np.isnan(column)])
        assert set(observed).issubset(set(range(len(categories))))
        assert CATEGORICAL_MASK[len(NUMERIC_FEATURES) + offset] is True


def test_the_estimator_is_told_which_columns_are_categorical(synthetic_xy) -> None:
    features, target = synthetic_xy
    pipeline = build_native_pipeline().fit(features, target)

    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    assert list(classifier.is_categorical_) == list(CATEGORICAL_MASK)


def test_hgb_accepts_the_native_representation(synthetic_xy) -> None:
    features, target = synthetic_xy
    pipeline = build_native_pipeline().fit(features, target)

    probability = pipeline.predict_proba(features)[:, 1]
    assert probability.shape == (len(features),)
    assert np.all((probability >= 0.0) & (probability <= 1.0))


# --- leakage-safe vocabulary --------------------------------------------------


def test_encoder_is_fitted_only_on_the_rows_it_is_given(contract_frame_factory) -> None:
    """A category present only in the held-out rows must not enter categories_."""
    frame = contract_frame_factory(40)
    features = build_feature_matrix(frame)
    train = features.iloc[:20].copy()
    validation = features.iloc[20:].copy()
    validation.loc[validation.index[0], "Contract"] = "Quarterly promo"

    preprocessor = build_native_preprocessor().fit(train)
    ordinal = preprocessor.named_steps[ENCODER_STEP].named_transformers_[CATEGORICAL_STEP]
    position = list(CATEGORICAL_FEATURES).index("Contract")

    assert "Quarterly promo" not in set(ordinal.categories_[position])


def test_transforming_a_validation_fold_does_not_change_the_vocabulary(
    contract_frame_factory,
) -> None:
    frame = contract_frame_factory(40)
    features = build_feature_matrix(frame)
    train, validation = features.iloc[:20].copy(), features.iloc[20:].copy()
    validation.loc[validation.index[0], "Contract"] = "Quarterly promo"

    preprocessor = build_native_preprocessor().fit(train)
    ordinal = preprocessor.named_steps[ENCODER_STEP].named_transformers_[CATEGORICAL_STEP]
    before = [list(categories) for categories in ordinal.categories_]

    preprocessor.transform(validation)
    after = [list(categories) for categories in ordinal.categories_]
    assert before == after


def test_an_unseen_category_becomes_the_unknown_representation(contract_frame_factory) -> None:
    frame = contract_frame_factory(40)
    features = build_feature_matrix(frame)
    train, validation = features.iloc[:20].copy(), features.iloc[20:].copy()
    validation.loc[validation.index[0], "Contract"] = "Quarterly promo"

    preprocessor = build_native_preprocessor().fit(train)
    transformed = preprocessor.transform(validation)
    column = len(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES).index("Contract")

    assert np.isnan(transformed[0, column])
    assert not np.isnan(transformed[1:, column]).any()


def test_the_estimator_predicts_through_an_unseen_category(contract_frame_factory) -> None:
    """A new but valid category must degrade to the missing branch, not raise."""
    frame = contract_frame_factory(60)
    features = build_feature_matrix(frame)
    target = encode_target(frame["Churn"])
    train = slice(0, 40)
    pipeline = build_native_pipeline().fit(features.iloc[train], target.iloc[train])

    unseen = features.iloc[40:].copy()
    unseen.loc[unseen.index[0], "Contract"] = "Quarterly promo"
    probability = pipeline.predict_proba(unseen)[:, 1]

    assert np.all(np.isfinite(probability))


# --- cardinality guard --------------------------------------------------------


def test_the_guard_counts_categories_on_the_frame_it_is_fitted_on(synthetic_xy) -> None:
    features, _ = synthetic_xy
    guard = CategoricalCardinalityGuard().fit(features)

    assert set(guard.cardinality_) == set(CATEGORICAL_FEATURES)
    for column, count in guard.cardinality_.items():
        assert count == features[column].nunique(dropna=False)


def test_the_guard_returns_its_input_unchanged(synthetic_xy) -> None:
    features, _ = synthetic_xy
    transformed = CategoricalCardinalityGuard().fit(features).transform(features)

    pd.testing.assert_frame_equal(transformed, features)


def test_the_guard_raises_instead_of_grouping_categories(contract_frame_factory) -> None:
    frame = contract_frame_factory(40)
    features = build_feature_matrix(frame)
    features["PaymentMethod"] = [f"method-{index}" for index in range(len(features))]

    with pytest.raises(CategoricalCardinalityError, match="PaymentMethod"):
        CategoricalCardinalityGuard(max_cardinality=5).fit(features)


def test_the_limit_comes_from_the_estimator_not_from_a_literal() -> None:
    assert MAX_CATEGORICAL_CARDINALITY == HistGradientBoostingClassifier().max_bins


def test_observed_cardinality_reports_every_categorical_column(synthetic_xy) -> None:
    features, _ = synthetic_xy
    cardinality = observed_cardinality(features)

    assert set(cardinality) == set(CATEGORICAL_FEATURES)
    assert max(cardinality.values()) <= MAX_CATEGORICAL_CARDINALITY


def test_the_guard_is_part_of_the_fitted_pipeline(synthetic_xy) -> None:
    features, target = synthetic_xy
    pipeline = build_native_pipeline().fit(features, target)

    guard = pipeline.named_steps[PREPROCESSOR_STEP].named_steps[GUARD_STEP]
    assert isinstance(guard, CategoricalCardinalityGuard)
    assert guard.cardinality_


def test_the_guard_survives_cloning() -> None:
    guard = clone(CategoricalCardinalityGuard(max_cardinality=7))

    assert guard.max_cardinality == 7
    assert tuple(guard.columns) == CATEGORICAL_FEATURES


# --- the gate protocol --------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_runs(synthetic_xy):
    features, target = synthetic_xy
    return run_representation_gate(
        features, target, build_splitter(), (PRIMARY_METRIC, SECONDARY_METRIC)
    )


def test_the_gate_runs_three_experiments(synthetic_runs) -> None:
    assert list(synthetic_runs) == [REFERENCE_LOGISTIC, REFERENCE_HGB_COMMON, NATIVE_HGB]


def test_only_the_native_run_carries_deltas(synthetic_runs) -> None:
    assert synthetic_runs[REFERENCE_LOGISTIC].paired_deltas == {}
    assert synthetic_runs[REFERENCE_HGB_COMMON].paired_deltas == {}
    assert set(synthetic_runs[NATIVE_HGB].paired_deltas) == {
        REFERENCE_HGB_COMMON,
        REFERENCE_LOGISTIC,
    }


def test_every_experiment_uses_the_same_folds(synthetic_runs) -> None:
    signatures = {
        run.experiment: tuple(
            (fold.fold, fold.n_train, fold.n_validation) for fold in run.evaluation.folds
        )
        for run in synthetic_runs.values()
    }
    assert len(set(signatures.values())) == 1


def test_the_references_keep_the_common_representation(synthetic_runs) -> None:
    common = synthetic_runs[REFERENCE_LOGISTIC].n_transformed_features

    # Both references share one representation, whatever width the synthetic
    # frame produces; only the native run is narrowed to one column per feature.
    assert synthetic_runs[REFERENCE_HGB_COMMON].n_transformed_features == common
    assert synthetic_runs[NATIVE_HGB].n_transformed_features == 19
    assert common > synthetic_runs[NATIVE_HGB].n_transformed_features


def test_paired_deltas_follow_the_fold_order(synthetic_runs) -> None:
    native = synthetic_runs[NATIVE_HGB]
    reference = synthetic_runs[REFERENCE_HGB_COMMON]
    comparison = native.paired_deltas[REFERENCE_HGB_COMMON][PRIMARY_METRIC]

    expected = [
        getattr(candidate.metrics, PRIMARY_METRIC) - getattr(baseline.metrics, PRIMARY_METRIC)
        for candidate, baseline in zip(
            native.evaluation.folds, reference.evaluation.folds, strict=True
        )
    ]
    assert list(comparison.deltas) == pytest.approx(expected)


def test_the_two_comparisons_use_different_references(synthetic_runs) -> None:
    native = synthetic_runs[NATIVE_HGB]
    against_common = native.paired_deltas[REFERENCE_HGB_COMMON][PRIMARY_METRIC].deltas
    against_logistic = native.paired_deltas[REFERENCE_LOGISTIC][PRIMARY_METRIC].deltas

    assert against_common != against_logistic


def test_out_of_fold_covers_every_row_exactly_once(synthetic_runs, synthetic_xy) -> None:
    features, _ = synthetic_xy
    for run in synthetic_runs.values():
        oof = run.evaluation.oof_probability
        assert oof.shape == (len(features),)
        assert not np.isnan(oof).any()


def test_no_row_is_predicted_by_the_estimator_that_trained_on_it(synthetic_xy) -> None:
    """Proved structurally: the validation index of each fold is disjoint from its train index."""
    features, target = synthetic_xy
    splitter = build_splitter()
    seen: set[int] = set()

    for train_index, validation_index in splitter.split(features, np.asarray(target)):
        assert not set(train_index) & set(validation_index)
        assert not seen & set(validation_index)
        seen |= set(validation_index)

    assert seen == set(range(len(features)))


def test_the_threshold_is_still_the_untouched_default(synthetic_runs, synthetic_xy) -> None:
    features, target = synthetic_xy
    assert DEFAULT_THRESHOLD == 0.5

    labels = np.asarray(target)
    for run in synthetic_runs.values():
        predicted = (run.evaluation.oof_probability >= DEFAULT_THRESHOLD).astype(int)
        expected = float((predicted == labels).mean())
        assert run.evaluation.oof_metrics.accuracy == pytest.approx(expected)


def test_no_engineered_feature_reaches_the_native_pipeline(synthetic_xy) -> None:
    """contract_tenure and the rejected Phase 6 candidates stay out of this phase."""
    features, target = synthetic_xy
    pipeline = build_native_pipeline().fit(features, target)
    encoder = pipeline.named_steps[PREPROCESSOR_STEP].named_steps[ENCODER_STEP]

    used = {column for _, _, columns in encoder.transformers_ for column in columns}
    assert used == set(FEATURE_COLUMNS)
    for engineered in (
        "tenure_x_month_to_month",
        "tenure_x_one_year",
        "protective_service_count",
        "automatic_payment",
        "historical_average_charge",
        "charge_intensity",
    ):
        assert engineered not in used


# --- classification heuristics ------------------------------------------------


class _Comparison:
    def __init__(self, deltas: tuple[float, ...]) -> None:
        self.deltas = deltas

    @property
    def mean(self) -> float:
        return float(np.mean(self.deltas))

    @property
    def n_positive(self) -> int:
        return int(sum(delta > 0 for delta in self.deltas))

    @property
    def n_negative(self) -> int:
        return int(sum(delta < 0 for delta in self.deltas))


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ((0.01, 0.01, 0.01, 0.01, -0.001), REPRESENTATION_HELPFUL),
        ((0.01, 0.01, 0.01, -0.001, -0.001), REPRESENTATION_INCONCLUSIVE),
        ((-0.01, -0.01, 0.001, 0.001, 0.001), REPRESENTATION_NOT_SUPPORTED),
        ((-0.01, -0.01, -0.01, -0.01, -0.01), REPRESENTATION_NOT_SUPPORTED),
        ((0.0, 0.0, 0.0, 0.0, 0.0), REPRESENTATION_NOT_SUPPORTED),
    ],
)
def test_representation_classification_rules(deltas, expected) -> None:
    classification, rationale = classify_representation(_Comparison(deltas), 5)

    assert classification == expected
    assert rationale


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ((-0.01, -0.01, -0.01, -0.01, -0.01), "BELOW_LOGISTIC"),
        ((0.01, 0.01, 0.01, 0.01, 0.01), "ABOVE_LOGISTIC"),
        # A positive mean carried by a single fold is not a standing above the
        # reference: four of five folds went the other way.
        ((0.02, -0.005, -0.005, -0.005, -0.004), "COMPARABLE_TO_LOGISTIC"),
    ],
)
def test_standing_rules(deltas, expected) -> None:
    standing, rationale = standing_against_logistic(_Comparison(deltas), 5)

    assert standing == expected
    assert rationale


def test_no_classification_uses_a_magnitude_cutoff() -> None:
    """A vanishingly small but consistent gain is classified on direction alone."""
    tiny = _Comparison((1e-9, 1e-9, 1e-9, 1e-9, -1e-12))

    assert classify_representation(tiny, 5)[0] == REPRESENTATION_HELPFUL


# --- reproduction gates on the real training pool -----------------------------


def test_r0_reproduces_m0_of_the_phase7_comparison(real_training_xy) -> None:
    features, target = real_training_xy
    pipeline = build_family_pipeline("logistic_regression", ())
    evaluation = evaluate_model(REFERENCE_LOGISTIC, pipeline, features, target, build_splitter())

    reference = load_comparison_results()
    record = next(
        item
        for item in reference.main_comparison
        if item.experiment == REFERENCE_EXPERIMENTS[REFERENCE_LOGISTIC]
    )
    for fold, stored in zip(evaluation.folds, record.folds, strict=True):
        for metric in METRIC_NAMES:
            assert round(getattr(fold.metrics, metric), 6) == pytest.approx(
                stored.metrics[metric], abs=1e-9
            )


def test_r1_reproduces_m2_of_the_phase7_comparison(real_training_xy) -> None:
    features, target = real_training_xy
    pipeline = build_family_pipeline("hist_gradient_boosting", ())
    evaluation = evaluate_model(REFERENCE_HGB_COMMON, pipeline, features, target, build_splitter())

    reference = load_comparison_results()
    record = next(
        item
        for item in reference.main_comparison
        if item.experiment == REFERENCE_EXPERIMENTS[REFERENCE_HGB_COMMON]
    )
    for fold, stored in zip(evaluation.folds, record.folds, strict=True):
        for metric in METRIC_NAMES:
            assert round(getattr(fold.metrics, metric), 6) == pytest.approx(
                stored.metrics[metric], abs=1e-9
            )


def test_the_reproduction_gate_rejects_a_changed_metric(synthetic_runs) -> None:
    """The gate must fail loudly, not warn: everything downstream depends on it."""
    from churn.modeling.comparison_results import ReproductionError

    reference = load_comparison_results()
    tampered = reference.model_copy(deep=True)
    record = tampered.main_comparison[0]
    record.folds[0].metrics[PRIMARY_METRIC] += 0.01

    with pytest.raises(ReproductionError):
        verify_against_comparison(synthetic_runs[REFERENCE_LOGISTIC], tampered)


# --- audits -------------------------------------------------------------------

_PHASE8A_SOURCES = [
    PROJECT_ROOT / "src" / "churn" / "modeling" / "representation.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "representation_results.py",
    PROJECT_ROOT / "scripts" / "run_hgb_representation.py",
]

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
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier",
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


@pytest.mark.parametrize("source", _PHASE8A_SOURCES, ids=lambda path: path.name)
def test_phase8a_code_never_reaches_the_holdout(source) -> None:
    used = _identifiers(source)

    assert "load_holdout" not in used, f"{source.name} imports or calls the holdout loader"
    assert "holdout" not in used, f"{source.name} reads a holdout partition"


@pytest.mark.parametrize("source", _PHASE8A_SOURCES, ids=lambda path: path.name)
def test_phase8a_code_creates_no_search_or_calibration_object(source) -> None:
    used = _identifiers(source)

    found = sorted(name for name in FORBIDDEN_IDENTIFIERS if name in used)
    assert not found, f"{source.name} references {found}: this phase neither tunes nor calibrates"


@pytest.mark.parametrize("source", _PHASE8A_SOURCES, ids=lambda path: path.name)
def test_phase8a_code_never_fits_on_the_whole_dataset_loader(source) -> None:
    """Only the script may load data, and only the training pool."""
    used = _identifiers(source)

    if source.name == "run_hgb_representation.py":
        assert "load_training_pool" in used
        assert "load_split" not in used
        return
    assert "load_training_pool" not in used
    assert "load_raw_typed" not in used


def _categorical_features_values(source) -> list[ast.expr]:
    """Every value assigned to ``categorical_features``, as keyword or dict key.

    Deliberately AST-based rather than a text scan. These modules *discuss*
    ``from_dtype`` in prose to explain why it is not used, and an audit that
    reads prose would fail on its own documentation. This one reads code.
    """
    values: list[ast.expr] = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.keyword) and node.arg == "categorical_features":
            values.append(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "categorical_features":
                    values.append(value)
    return values


@pytest.mark.parametrize("source", _PHASE8A_SOURCES, ids=lambda path: path.name)
def test_phase8a_code_never_declares_categorical_features_from_dtype(source) -> None:
    """The mask must be positional, never inferred from an upstream dtype."""
    for value in _categorical_features_values(source):
        assert not (isinstance(value, ast.Constant) and value.value == "from_dtype"), (
            f"{source.name} declares categorical_features='from_dtype'"
        )


def test_the_native_estimator_declares_the_mask_and_not_a_dtype_rule() -> None:
    """The positive half of the audit above: what the value actually is."""
    declared = build_hgb_native().categorical_features

    assert isinstance(declared, list)
    assert all(isinstance(flag, bool) for flag in declared)
    assert declared == list(CATEGORICAL_MASK)


def test_no_engineered_feature_group_is_imported_by_phase8a() -> None:
    for source in _PHASE8A_SOURCES:
        used = _identifiers(source)
        assert "CONTRACT_TENURE" not in used, f"{source.name} reaches for contract_tenure"
        assert "FEATURE_GROUPS" not in used, f"{source.name} reaches for the feature groups"
