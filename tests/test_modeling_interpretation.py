"""Phase 10: the decomposition is exact, and the phase only described.

Three properties carry this phase and none can be read off the source, so all
three are asserted behaviourally.

**The decomposition reproduces the model.** Not "closely" — the manual logit is
compared against ``decision_function``, its sigmoid against ``predict_proba``, and
the regrouped 19 contributions against ``decision_function`` again, all to a
strict tolerance. A decomposition that misses the intercept, transposes a
coefficient or drops a one-hot column fails these by a wide margin.

**Nothing was learned.** Every fit entry point — the pipeline's, the estimator's,
the scaler's, the encoder's and the column transformer's — is replaced by a guard
that raises, and the real interpretation still runs to completion.

**The holdout was never opened.** ``load_holdout`` is replaced by a guard that
raises, and the real interpretation still runs to completion; an AST sweep over
the four Phase 10 files backs that up statically.

Synthetic fixtures are used wherever the question does not need the real frozen
model. Where it does, the committed artefacts are read and the tests skip if the
phase has not been run.
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import PROJECT_ROOT
from churn.modeling.freeze import file_digest, load_pipeline, model_fingerprint
from churn.modeling.freeze_results import load_decision_policy
from churn.modeling.interpretation import (
    ANALYSIS_TYPE,
    CATEGORICAL_KIND,
    NUMERIC_KIND,
    RANKING_METRIC,
    RECONSTRUCTION_TOLERANCE,
    STRUCTURAL_RULES,
    SUPPORTING_METRICS,
    FeatureGroup,
    InterpretationError,
    ReconstructionError,
    StructuralCheck,
    StructuralRule,
    categorical_terms,
    check_structural_dependencies,
    check_structural_dependency,
    contribution_dispersion,
    coupled_column_blocks,
    coupled_levels_from,
    extract_terms,
    feature_groups,
    group_contributions,
    level_contrasts,
    logit_of,
    manual_logit,
    numeric_terms,
    positive_class_column,
    rank_by_dispersion,
    sigmoid,
    transform_features,
    verify_group_partition,
    verify_reconstruction,
)
from churn.modeling.interpretation_results import (
    ERROR_ANALYSIS_COMMIT,
    EVALUATION_COMMIT,
    FREEZE_COMMIT,
    SCHEMA_VERSION,
    UPSTREAM_ARTEFACTS,
    invariant_failures,
    load_results,
)
from churn.modeling.models import CLASSIFIER_STEP, PREPROCESSOR_STEP
from churn.modeling.tuning import build_modern_logistic_pipeline
from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_matrix,
)
from churn.preprocessing.pipeline import ENCODER_STEP
from churn.preprocessing.target import encode_target

#: The frozen values, pinned as the human-approved ones so a drift in an artefact
#: fails a test rather than silently changing what is being interpreted.
FROZEN_THRESHOLD = 0.3272694566222328
FROZEN_INTERCEPT = -0.5300483006137556
MODEL_FINGERPRINT = "a57568e18ec0333e1c85b590f98e75da93ebbfafff063860ec5daf4e460d939f"
PIPELINE_SHA256 = "574fde36c6e2e991de7c3dfdab21981dc41eae2810d003ecbfb179dd504dc3d8"
TRAINING_POOL_ROWS = 5634

N_TRANSFORMED = 46
N_RAW = 19
N_NUMERIC = 3
N_CATEGORICAL = 16
N_ONE_HOT = N_TRANSFORMED - N_NUMERIC

RESULTS_FILE = PROJECT_ROOT / "reports" / "experiments" / "model_interpretation_results.json"
REPORT_FILE = PROJECT_ROOT / "reports" / "model_interpretation_report.md"

#: The four files that make up Phase 10. The AST guards sweep exactly these.
PHASE_10_SOURCES: tuple[pathlib.Path, ...] = (
    PROJECT_ROOT / "src" / "churn" / "modeling" / "interpretation.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "interpretation_results.py",
    PROJECT_ROOT / "src" / "churn" / "modeling" / "interpretation_plots.py",
    PROJECT_ROOT / "scripts" / "run_model_interpretation.py",
)


def _load_script(name: str):
    """Import a file in ``scripts/`` as a module. It is not an installed package."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_model_interpretation = _load_script("run_model_interpretation")


def _results():
    if not RESULTS_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    return load_results()


def _frozen_pipeline():
    policy_path = PROJECT_ROOT / "reports" / "decision_policy.json"
    pipeline_path = PROJECT_ROOT / "artifacts" / "model" / "churn_pipeline.joblib"
    if not (policy_path.is_file() and pipeline_path.is_file()):
        pytest.skip("the frozen model has not been produced yet")
    return load_pipeline(pipeline_path)


@pytest.fixture(scope="module")
def toy(contract_frame_factory):
    """A synthetic fitted pipeline with the real Phase 4 structure.

    Fitted **here**, in a fixture, on synthetic rows. That is not a violation of
    the phase's no-fit rule: the rule constrains the Phase 10 modules, and the
    unit tests need a model whose parameters they can perturb without touching
    the frozen artefact.
    """
    frame = contract_frame_factory(120)
    features = build_feature_matrix(frame)
    target = np.asarray(encode_target(frame["Churn"]))
    pipeline = build_modern_logistic_pipeline().fit(features, target)
    return pipeline, features


# --- the frozen model is loaded, never rebuilt --------------------------------


def test_the_frozen_policy_is_unchanged() -> None:
    policy = load_decision_policy()

    assert policy.threshold.final_threshold == FROZEN_THRESHOLD
    assert policy.decision_rule.comparison == ">="
    assert policy.calibration.calibration_policy == "NONE"
    assert policy.model.estimator == "LogisticRegression"
    assert policy.feature_contract.n_features == N_RAW
    assert policy.feature_contract.n_transformed_features == N_TRANSFORMED
    assert policy.feature_contract.engineered == []


def test_a_valid_freeze_is_required_before_interpreting(monkeypatch) -> None:
    """The freeze gate runs first; if it fails, nothing is interpreted."""
    from churn.modeling.freeze import IntegrityError

    def refuse(*args, **kwargs):
        raise IntegrityError("simulated freeze failure")

    monkeypatch.setattr(run_model_interpretation, "verify_or_raise", refuse)
    with pytest.raises(IntegrityError):
        run_model_interpretation.interpret()

    assert run_model_interpretation.main([]) == 4


def test_the_loaded_pipeline_is_the_frozen_one() -> None:
    pipeline = _frozen_pipeline()

    assert model_fingerprint(pipeline) == MODEL_FINGERPRINT
    assert (
        file_digest(PROJECT_ROOT / "artifacts" / "model" / "churn_pipeline.joblib")
        == PIPELINE_SHA256
    )


def test_the_record_confirms_both_frozen_fingerprints() -> None:
    results = _results()
    provenance = results.provenance

    assert provenance["model_fingerprint_sha256_after_load"] == MODEL_FINGERPRINT
    assert provenance["model_fingerprint_verified"] is True
    assert provenance["pipeline_sha256_after_analysis"] == PIPELINE_SHA256
    assert provenance["pipeline_sha256_recorded"] == PIPELINE_SHA256
    assert provenance["pipeline_unchanged_by_analysis"] is True
    assert provenance["refitted_here"] is False
    assert provenance["reopened_here"] is False


def test_the_three_reference_commits_are_recorded() -> None:
    results = _results()

    assert results.provenance["freeze_commit"] == FREEZE_COMMIT
    assert results.provenance["evaluation_commit"] == EVALUATION_COMMIT
    assert results.provenance["error_analysis_commit"] == ERROR_ANALYSIS_COMMIT


# --- the model is read structurally -------------------------------------------


def test_the_positive_class_is_resolved_from_classes_not_assumed() -> None:
    assert positive_class_column([0, 1], 1) == 1
    assert positive_class_column([1, 0], 1) == 0

    with pytest.raises(InterpretationError, match="not identifiable"):
        positive_class_column([0, 2], 1)
    with pytest.raises(InterpretationError, match="not identifiable"):
        positive_class_column([1, 1], 1)


def test_a_positive_class_outside_the_decision_function_column_is_refused(toy) -> None:
    """A model whose positive class is ``classes_[0]`` would need a sign flip."""
    pipeline, _ = toy
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    original = classifier.classes_
    classifier.classes_ = np.array([1, 0])
    try:
        with pytest.raises(InterpretationError, match="wrong sign"):
            extract_terms(pipeline, positive_label=1)
    finally:
        classifier.classes_ = original


def test_a_pipeline_without_the_frozen_structure_is_refused() -> None:
    with pytest.raises(InterpretationError, match="frozen structure"):
        extract_terms(Pipeline(steps=[("only", LogisticRegression())]))


def test_the_frozen_model_has_46_transformed_features_and_46_coefficients() -> None:
    terms = extract_terms(_frozen_pipeline())

    assert terms.n_transformed_features == N_TRANSFORMED
    assert terms.coefficients.shape == (N_TRANSFORMED,)
    assert len(terms.names) == N_TRANSFORMED
    assert terms.classes == (0, 1)
    assert terms.positive_class_column == 1
    assert terms.intercept == pytest.approx(FROZEN_INTERCEPT, abs=1e-15)


def test_the_frozen_model_has_19_raw_feature_groups_3_numeric_and_16_categorical() -> None:
    terms = extract_terms(_frozen_pipeline())
    groups = feature_groups(terms)

    assert len(groups) == N_RAW
    assert sum(group.kind == NUMERIC_KIND for group in groups) == N_NUMERIC
    assert sum(group.kind == CATEGORICAL_KIND for group in groups) == N_CATEGORICAL
    assert [group.feature for group in groups] == list(NUMERIC_FEATURES) + list(
        CATEGORICAL_FEATURES
    )


# --- the one-hot mapping is complete, disjoint and total -----------------------


def test_every_transformed_column_has_exactly_one_raw_parent() -> None:
    terms = extract_terms(_frozen_pipeline())
    groups = feature_groups(terms)

    assigned = [column for group in groups for column in group.columns]
    assert sorted(assigned) == list(range(N_TRANSFORMED))
    assert len(assigned) == len(set(assigned)), "a column was assigned twice"

    one_hot = sum(group.n_columns for group in groups if group.kind == CATEGORICAL_KIND)
    numeric = sum(group.n_columns for group in groups if group.kind == NUMERIC_KIND)
    assert one_hot == N_ONE_HOT
    assert numeric == N_NUMERIC
    assert numeric + one_hot == N_TRANSFORMED


def test_each_categorical_group_owns_one_column_per_level() -> None:
    terms = extract_terms(_frozen_pipeline())

    for group in feature_groups(terms):
        if group.kind == CATEGORICAL_KIND:
            assert len(group.levels) == group.n_columns
            assert len(set(group.levels)) == group.n_columns
        else:
            assert group.levels == ()
            assert group.n_columns == 1


def test_the_mapping_is_not_derived_by_textual_parsing() -> None:
    """The level values themselves contain the delimiter a naive parse would use."""
    terms = extract_terms(_frozen_pipeline())
    groups = {group.feature: group for group in feature_groups(terms)}

    payment = groups["PaymentMethod"]
    assert any("_" in level or " " in level for level in payment.levels)
    for column, level in zip(payment.columns, payment.levels, strict=True):
        assert terms.names[column].endswith(level)

    assert _results().feature_mapping["textual_parsing_used"] is False


def test_an_orphaned_column_is_refused() -> None:
    terms = extract_terms(_frozen_pipeline())
    truncated = feature_groups(terms)[:-1]

    with pytest.raises(InterpretationError, match="belong to no raw feature"):
        verify_group_partition(truncated, terms)


def test_a_column_claimed_by_two_features_is_refused() -> None:
    terms = extract_terms(_frozen_pipeline())
    groups = list(feature_groups(terms))
    groups.append(FeatureGroup(feature="duplicate", kind=NUMERIC_KIND, columns=(0,), levels=()))

    with pytest.raises(InterpretationError, match="more than one raw feature"):
        verify_group_partition(groups, terms)


def test_a_mapping_that_disagrees_with_the_encoder_names_is_refused(toy) -> None:
    """The generated names are a cross-check, and the cross-check has teeth."""
    pipeline, _ = toy
    terms = extract_terms(pipeline)
    swapped = terms.__class__(
        **{
            **terms.__dict__,
            "categorical_features": ("NotAFeature",) + terms.categorical_features[1:],
        }
    )
    with pytest.raises(InterpretationError, match="does not mention that feature"):
        feature_groups(swapped)


# --- the reconstruction gates -------------------------------------------------


def test_the_manual_logit_reproduces_decision_function() -> None:
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    features = _training_features()

    derived = manual_logit(terms, transform_features(pipeline, features))
    reference = np.asarray(pipeline.decision_function(features), dtype=float)

    assert np.max(np.abs(derived - reference)) <= RECONSTRUCTION_TOLERANCE


def test_the_sigmoid_of_the_manual_logit_reproduces_predict_proba() -> None:
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    features = _training_features()

    derived = sigmoid(manual_logit(terms, transform_features(pipeline, features)))
    reference = pipeline.predict_proba(features)[:, terms.positive_class_column]

    assert np.max(np.abs(derived - reference)) <= RECONSTRUCTION_TOLERANCE


def test_the_grouped_raw_contributions_reproduce_the_logit() -> None:
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    features = _training_features()

    contributions = group_contributions(terms, groups, transform_features(pipeline, features))
    assert contributions.shape == (len(features), N_RAW)

    grouped = terms.intercept + contributions.sum(axis=1)
    reference = np.asarray(pipeline.decision_function(features), dtype=float)
    assert np.max(np.abs(grouped - reference)) <= RECONSTRUCTION_TOLERANCE


def test_a_corrupted_decomposition_fails_the_gate(toy) -> None:
    """A dropped intercept must not pass as an interpretation."""
    pipeline, features = toy
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)

    broken = terms.__class__(**{**terms.__dict__, "intercept": terms.intercept + 0.5})
    with pytest.raises(ReconstructionError, match="does not reproduce"):
        verify_reconstruction(pipeline, broken, groups, features)


def test_the_sigmoid_is_stable_on_both_tails() -> None:
    extreme = np.array([-800.0, -40.0, 0.0, 40.0, 800.0])
    values = sigmoid(extreme)

    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(0.0, abs=1e-300)
    assert values[2] == pytest.approx(0.5)
    assert values[4] == pytest.approx(1.0)


def test_the_recorded_reconstruction_errors_clear_the_recorded_tolerance() -> None:
    reconstruction = _results().reconstruction

    assert reconstruction["n_rows"] == TRAINING_POOL_ROWS
    assert reconstruction["tolerance"] == RECONSTRUCTION_TOLERANCE
    assert reconstruction["within_tolerance"] is True
    for key in (
        "max_abs_logit_error",
        "max_abs_probability_error",
        "max_grouped_reconstruction_error",
    ):
        assert reconstruction[key] <= RECONSTRUCTION_TOLERANCE, key


# --- the frozen decision boundary, re-expressed --------------------------------


def test_the_threshold_logit_is_derived_correctly() -> None:
    value = logit_of(FROZEN_THRESHOLD)

    assert value == pytest.approx(
        float(np.log(FROZEN_THRESHOLD / (1 - FROZEN_THRESHOLD))), abs=1e-15
    )
    assert sigmoid(np.array([value]))[0] == pytest.approx(FROZEN_THRESHOLD, abs=1e-15)

    with pytest.raises(InterpretationError):
        logit_of(0.0)
    with pytest.raises(InterpretationError):
        logit_of(1.0)


def test_the_log_odds_rule_and_the_probability_rule_agree_on_every_row() -> None:
    """The re-expression is the same frozen rule, not a new threshold."""
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    features = _training_features()

    probability = pipeline.predict_proba(features)[:, terms.positive_class_column]
    logits = np.asarray(pipeline.decision_function(features), dtype=float)

    by_probability = probability >= FROZEN_THRESHOLD
    by_logit = logits >= logit_of(FROZEN_THRESHOLD)
    assert np.array_equal(by_probability, by_logit)


def test_the_record_keeps_the_frozen_threshold_untouched() -> None:
    decision = _results().frozen_decision

    assert decision["threshold_probability"] == FROZEN_THRESHOLD
    assert decision["comparison"] == ">="
    assert decision["calibration_policy"] == "NONE"
    assert decision["threshold_changed_here"] is False
    assert decision["calibration_changed_here"] is False
    assert decision["alternative_thresholds_scored"] == 0
    assert decision["threshold_logit"] == pytest.approx(logit_of(FROZEN_THRESHOLD), abs=1e-9)


# --- numeric terms ------------------------------------------------------------


def test_the_raw_logit_coefficient_is_the_exact_rescaling() -> None:
    """beta*((x+d-m)/s) - beta*((x-m)/s) == (beta/s)*d, for any x and d."""
    terms = extract_terms(_frozen_pipeline())
    groups = feature_groups(terms)

    for term in numeric_terms(terms, groups):
        beta = term.standardized_coefficient
        mean = term.scaler_mean
        scale = term.scaler_scale
        for x, delta in ((0.0, 1.0), (12.0, 3.5), (2400.0, -7.25), (-5.0, 0.125)):
            moved = beta * ((x + delta - mean) / scale)
            base = beta * ((x - mean) / scale)
            assert moved - base == pytest.approx(term.raw_logit_coefficient * delta, abs=1e-12)


def test_the_odds_ratio_per_standard_deviation_is_exp_of_the_coefficient() -> None:
    terms = extract_terms(_frozen_pipeline())

    for term in numeric_terms(terms, feature_groups(terms)):
        assert term.odds_ratio_per_1_sd == pytest.approx(
            float(np.exp(term.standardized_coefficient)), rel=1e-12
        )


def test_the_numeric_contribution_is_beta_times_the_standardised_value() -> None:
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    features = _training_features().head(50)

    transformed = transform_features(pipeline, features)
    contributions = group_contributions(terms, groups, transformed)

    for position, term in enumerate(numeric_terms(terms, groups)):
        expected = term.standardized_coefficient * transformed[:, term.column]
        assert np.allclose(contributions[:, position], expected, atol=1e-15)


def test_the_record_reports_three_numeric_terms_consistently() -> None:
    results = _results()

    assert len(results.numeric_terms) == N_NUMERIC
    assert [term["feature"] for term in results.numeric_terms] == list(NUMERIC_FEATURES)
    for term in results.numeric_terms:
        assert term["scaler_scale"] > 0
        assert term["raw_logit_coefficient"] == pytest.approx(
            term["standardized_coefficient"] / term["scaler_scale"], rel=1e-8
        )
        assert term["odds_ratio_per_1_sd"] == pytest.approx(
            float(np.exp(term["standardized_coefficient"])), rel=1e-8
        )


# --- categorical terms and contrasts ------------------------------------------


def test_swapping_the_active_level_moves_the_logit_by_exactly_the_contrast() -> None:
    """Tested in the transformed space, so no impossible payload is constructed."""
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    row = transform_features(pipeline, _training_features().head(1))[0]

    for group in groups:
        if group.kind != CATEGORICAL_KIND:
            continue
        columns = list(group.columns)
        for first, second in zip(columns, columns[1:], strict=False):
            with_first = row.copy()
            with_first[columns] = 0.0
            with_first[first] = 1.0

            with_second = with_first.copy()
            with_second[first] = 0.0
            with_second[second] = 1.0

            moved = manual_logit(terms, with_second[None, :])[0]
            base = manual_logit(terms, with_first[None, :])[0]
            expected = terms.coefficients[second] - terms.coefficients[first]
            assert moved - base == pytest.approx(expected, abs=1e-12)


def test_the_recorded_contrast_is_beta_a_minus_beta_b() -> None:
    terms = extract_terms(_frozen_pipeline())
    groups = feature_groups(terms)

    for group in groups:
        if group.kind != CATEGORICAL_KIND:
            continue
        betas = dict(
            zip(
                group.levels,
                [terms.coefficients[column] for column in group.columns],
                strict=True,
            )
        )
        contrasts = level_contrasts(terms, group)
        assert len(contrasts) == len(group.levels) * (len(group.levels) - 1)
        for contrast in contrasts:
            expected = betas[contrast.level_a] - betas[contrast.level_b]
            assert contrast.delta_log_odds == pytest.approx(expected, abs=1e-15)
            assert contrast.modeled_odds_ratio == pytest.approx(float(np.exp(expected)), rel=1e-12)


def test_a_contrast_is_refused_for_a_numeric_feature() -> None:
    terms = extract_terms(_frozen_pipeline())
    numeric = next(group for group in feature_groups(terms) if group.kind == NUMERIC_KIND)

    with pytest.raises(InterpretationError, match="not a categorical feature"):
        level_contrasts(terms, numeric)


def test_the_spread_is_the_widest_contrast_and_its_odds_ratio_is_exp_of_it() -> None:
    terms = extract_terms(_frozen_pipeline())
    groups = feature_groups(terms)

    for term in categorical_terms(terms, groups):
        widest = max(
            contrast.delta_log_odds
            for contrast in level_contrasts(
                terms, next(group for group in groups if group.feature == term.feature)
            )
        )
        assert term.coefficient_spread == pytest.approx(widest, abs=1e-15)
        assert term.max_pairwise_modeled_odds_ratio == pytest.approx(
            float(np.exp(widest)), rel=1e-12
        )
        assert term.coefficient_per_level[term.highest_coefficient_level] == term.maximum
        assert term.coefficient_per_level[term.lowest_coefficient_level] == term.minimum


def test_no_fictitious_reference_category_is_declared() -> None:
    """No baseline was dropped, so no baseline may be claimed."""
    results = _results()

    assert results.methodology["reference_category_declared"] is False
    assert results.intercept_interpretation["reference_level_declared"] is False
    assert results.intercept_interpretation["redundant_parameterisation"] is True

    assert len(results.categorical_terms) == N_CATEGORICAL
    for term in results.categorical_terms:
        assert term["reference_level_removed"] is False
        assert term["reference_level"] is None
        assert len(term["coefficient_per_level"]) == term["n_levels"]


def test_the_encoder_really_kept_every_level(toy) -> None:
    """The premise of the whole section: ``drop`` is not configured."""
    pipeline, _ = toy
    encoder = pipeline.named_steps[PREPROCESSOR_STEP].named_steps[ENCODER_STEP]
    one_hot = encoder.named_transformers_["categorical"]

    assert one_hot.drop is None
    assert one_hot.handle_unknown == "ignore"

    frozen = _frozen_pipeline()
    frozen_one_hot = (
        frozen.named_steps[PREPROCESSOR_STEP]
        .named_steps[ENCODER_STEP]
        .named_transformers_["categorical"]
    )
    assert frozen_one_hot.drop is None
    assert sum(len(levels) for levels in frozen_one_hot.categories_) == N_ONE_HOT


def test_the_intercept_is_not_presented_as_an_average_customer() -> None:
    results = _results()

    assert results.intercept_interpretation["is_the_risk_of_an_average_customer"] is False
    assert results.intercept_interpretation["value"] == pytest.approx(FROZEN_INTERCEPT, abs=1e-9)
    assert "is not a customer" in results.intercept_interpretation["why_not"]


# --- contribution dispersion and the ranking ----------------------------------


def test_the_contribution_dispersion_metrics_are_computed_as_documented() -> None:
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    features = _training_features()

    contributions = group_contributions(terms, groups, transform_features(pipeline, features))
    records = contribution_dispersion(groups, contributions)

    assert len(records) == N_RAW
    for position, record in enumerate(records):
        column = contributions[:, position]
        assert record.mean == pytest.approx(float(column.mean()), abs=1e-15)
        assert record.std == pytest.approx(float(column.std(ddof=0)), abs=1e-15)
        assert record.median == pytest.approx(float(np.median(column)), abs=1e-15)
        assert record.iqr == pytest.approx(record.q3 - record.q1, abs=1e-15)
        assert record.mean_absolute_centered_contribution == pytest.approx(
            float(np.mean(np.abs(column - column.mean()))), abs=1e-15
        )


def test_a_dispersion_matrix_that_does_not_match_the_groups_is_refused() -> None:
    terms = extract_terms(_frozen_pipeline())
    groups = feature_groups(terms)

    with pytest.raises(InterpretationError, match="against 19 feature groups"):
        contribution_dispersion(groups, np.zeros((10, 5)))


def test_the_ranking_uses_the_predeclared_metric_and_nothing_else() -> None:
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    features = _training_features()

    records = contribution_dispersion(
        groups, group_contributions(terms, groups, transform_features(pipeline, features))
    )
    ranking = rank_by_dispersion(records, RANKING_METRIC)

    assert RANKING_METRIC == "std"
    assert [row["rank"] for row in ranking] == list(range(1, N_RAW + 1))
    values = [row["ranking_value"] for row in ranking]
    assert values == sorted(values, reverse=True)
    assert all(row["ranking_metric"] == RANKING_METRIC for row in ranking)

    with pytest.raises(InterpretationError, match="not a declared dispersion metric"):
        rank_by_dispersion(records, "mean")


def test_the_ranking_is_a_permutation_of_the_nineteen_features() -> None:
    results = _results()
    dispersion = results.contribution_dispersion

    assert dispersion["ranking_metric"] == RANKING_METRIC
    assert dispersion["ranking_metric_fixed_before_computing_values"] is True
    assert dispersion["population"] == "training pool"
    assert dispersion["n_rows"] == TRAINING_POOL_ROWS
    assert list(dispersion["supporting_metrics"]) == list(SUPPORTING_METRICS)

    ranked = [row["feature"] for row in dispersion["ranking"]]
    assert sorted(ranked) == sorted(list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES))
    assert len(ranked) == len(set(ranked)) == N_RAW


def test_the_ranking_is_never_called_importance() -> None:
    results = _results()

    assert "importance" not in results.contribution_dispersion["ranking_name"]
    for forbidden in results.contribution_dispersion["ranking_is_not"]:
        assert "importance" in forbidden
    assert results.methodology["naive_abs_coefficient_ranking_produced"] is False


# --- nothing was fitted, nothing was selected, no holdout ---------------------


def test_the_interpretation_calls_no_fit(monkeypatch) -> None:
    """Every fit entry point raises, and the real interpretation still completes."""
    if not (PROJECT_ROOT / "reports" / "decision_policy.json").is_file():
        pytest.skip("the frozen model has not been produced yet")

    attempted: list[str] = []

    def forbid(name: str):
        def guard(*args, **kwargs):
            attempted.append(name)
            raise AssertionError(f"the interpretation called {name}")

        return guard

    for owner, method in (
        (Pipeline, "fit"),
        (Pipeline, "fit_transform"),
        (LogisticRegression, "fit"),
        (StandardScaler, "fit"),
        (StandardScaler, "fit_transform"),
        (OneHotEncoder, "fit"),
        (OneHotEncoder, "fit_transform"),
        (ColumnTransformer, "fit"),
        (ColumnTransformer, "fit_transform"),
    ):
        monkeypatch.setattr(owner, method, forbid(f"{owner.__name__}.{method}"))

    results, numeric, categorical, dispersions = run_model_interpretation.interpret()

    assert attempted == [], f"the interpretation attempted: {attempted}"
    assert results.fit_calls == 0
    assert results.methodology["fit_calls"] == 0
    assert len(numeric) == N_NUMERIC
    assert len(categorical) == N_CATEGORICAL
    assert len(dispersions) == N_RAW


def test_the_interpretation_never_loads_the_holdout(monkeypatch) -> None:
    """``load_holdout`` raises, and the real interpretation still completes."""
    if not (PROJECT_ROOT / "reports" / "decision_policy.json").is_file():
        pytest.skip("the frozen model has not been produced yet")

    import churn.preprocessing.splitting as splitting

    def forbid(*args, **kwargs):
        raise AssertionError("the interpretation loaded the holdout")

    monkeypatch.setattr(splitting, "load_holdout", forbid)

    results, *_ = run_model_interpretation.interpret()

    assert results.holdout_used_for_global_interpretation is False
    assert results.methodology["holdout_loaded"] is False
    assert results.interpretation_population["partition"] == "training pool"
    assert results.interpretation_population["n_rows"] == TRAINING_POOL_ROWS


def test_no_phase_10_source_mentions_the_holdout_loader() -> None:
    """An AST sweep, as a static backstop to the behavioural test above."""
    for path in PHASE_10_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id != "load_holdout", path.name
            if isinstance(node, ast.Attribute):
                assert node.attr != "load_holdout", path.name
            if isinstance(node, ast.ImportFrom):
                assert all(alias.name != "load_holdout" for alias in node.names), path.name


def test_no_phase_10_source_calls_a_fit_or_a_search() -> None:
    forbidden = {
        "fit",
        "fit_transform",
        "fit_predict",
        "GridSearchCV",
        "RandomizedSearchCV",
        "CalibratedClassifierCV",
        "HalvingGridSearchCV",
        "cross_val_score",
        "cross_val_predict",
    }
    for path in PHASE_10_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, f"{path.name} touches .{node.attr}"
            if isinstance(node, ast.Name):
                assert node.id not in forbidden, f"{path.name} names {node.id}"


def test_no_phase_10_source_uses_an_approximate_explainer() -> None:
    forbidden = {
        "shap",
        "lime",
        "eli5",
        "TreeExplainer",
        "KernelExplainer",
        "LinearExplainer",
        "Explainer",
        "permutation_importance",
        "partial_dependence",
        "PartialDependenceDisplay",
        "plot_partial_dependence",
    }
    for path in PHASE_10_SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in forbidden, f"{path.name} names {node.id}"
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, f"{path.name} touches .{node.attr}"
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in forbidden for alias in node.names), (
                    path.name
                )
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in forbidden, path.name
                assert all(alias.name not in forbidden for alias in node.names), path.name


def test_no_explainer_dependency_was_added_to_the_project() -> None:
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()

    for package in ('"shap', '"lime', '"eli5', '"interpret', '"dalex'):
        assert package not in text, f"{package} was added as a dependency"


def test_the_record_flags_every_forbidden_technique_as_unused() -> None:
    results = _results()
    methodology = results.methodology

    for flag in (
        "shap_used",
        "lime_used",
        "eli5_used",
        "permutation_importance_used",
        "pdp_used",
        "ice_used",
        "post_hoc_holdout_selection",
        "causal_claims",
        "selection_after_analysis",
        "model_changed",
        "threshold_changed",
        "calibration_changed",
    ):
        assert methodology[flag] is False, flag

    assert results.exact_functional_form["approximation_used"] is False
    assert results.exact_functional_form["explainer_library_used"] is None


def test_no_selection_and_no_model_change_are_permitted() -> None:
    results = _results()

    assert results.selection_allowed is False
    assert results.model_change_allowed is False
    assert results.fit_calls == 0
    assert results.analysis_type == ANALYSIS_TYPE
    assert results.schema_version == SCHEMA_VERSION
    for counter in (
        "fit_calls",
        "alternative_models_evaluated",
        "thresholds_scored",
        "alternative_thresholds_scored",
        "calibrations_performed",
        "hyperparameter_searches",
        "feature_selections",
        "features_added",
        "features_removed",
        "new_performance_estimates_produced",
        "uncertainty_intervals_computed",
        "new_dependencies_added",
    ):
        assert results.methodology[counter] == 0, counter
    assert results.methodology["models_loaded"] == 1


def test_no_identifier_and_no_individual_explanation_is_persisted() -> None:
    results = _results()
    raw = RESULTS_FILE.read_text(encoding="utf-8")

    assert results.methodology["identifiers_persisted"] is False
    assert results.methodology["individual_explanations_persisted"] is False
    assert "customerID" not in raw
    assert not re.search(r"\b\d{4}-[A-Z]{5}\b", raw)

    report = REPORT_FILE.read_text(encoding="utf-8")
    assert "customerID" not in report
    assert not re.search(r"\b\d{4}-[A-Z]{5}\b", report)


def test_no_per_row_field_reached_the_record() -> None:
    """The artefact holds aggregates; a per-customer array would be a leak."""
    if not RESULTS_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    payload = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))

    def longest_list(node) -> int:
        if isinstance(node, dict):
            return max((longest_list(value) for value in node.values()), default=0)
        if isinstance(node, list):
            return max([len(node), *(longest_list(item) for item in node)], default=0)
        return 0

    assert longest_list(payload) < 100, "a list long enough to be per-row was persisted"


# --- the upstream artefacts are untouched -------------------------------------


def test_every_upstream_artefact_digest_still_matches() -> None:
    if not all((PROJECT_ROOT / artefact).is_file() for artefact in UPSTREAM_ARTEFACTS):
        pytest.skip("not every upstream artefact is present")

    results = _results()
    for artefact, recorded in results.upstream_artefact_digests.items():
        assert file_digest(PROJECT_ROOT / artefact) == recorded, artefact


def test_the_upstream_list_covers_every_earlier_phase_record() -> None:
    for artefact in (
        "reports/split_manifest.json",
        "reports/decision_policy.json",
        "reports/model_freeze_report.md",
        "reports/holdout_evaluation_report.md",
        "reports/error_analysis_report.md",
        "reports/experiments/holdout_results.json",
        "reports/experiments/error_analysis_results.json",
    ):
        assert artefact in UPSTREAM_ARTEFACTS


# --- the record, the report and the command -----------------------------------


def test_the_record_satisfies_every_invariant() -> None:
    assert invariant_failures(_results()) == []


def test_a_tampered_ranking_fails_the_invariants() -> None:
    results = _results()
    payload = results.model_dump()
    payload["contribution_dispersion"]["ranking"][0]["ranking_value"] = -1.0

    failures = invariant_failures(results.__class__(**payload))
    assert any("sorted by the declared metric" in failure for failure in failures)


def test_a_tampered_partition_fails_the_invariants() -> None:
    results = _results()
    payload = results.model_dump()
    payload["feature_mapping"]["groups"][0]["transformed_columns"] = [1]

    failures = invariant_failures(results.__class__(**payload))
    assert any("not a partition" in failure for failure in failures)


def test_a_tampered_reconstruction_fails_the_invariants() -> None:
    results = _results()
    payload = results.model_dump()
    payload["reconstruction"]["max_abs_logit_error"] = 1.0

    failures = invariant_failures(results.__class__(**payload))
    assert any("max_abs_logit_error" in failure for failure in failures)


def test_the_verify_command_is_read_only_and_passes(monkeypatch) -> None:
    if not RESULTS_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")

    def forbid(*args, **kwargs):
        raise AssertionError("--verify wrote a file")

    monkeypatch.setattr(pathlib.Path, "write_text", forbid)
    monkeypatch.setattr(pathlib.Path, "write_bytes", forbid)
    monkeypatch.setattr(Pipeline, "fit", forbid)

    assert run_model_interpretation.main(["--verify"]) == 0


def test_the_generation_is_deterministic() -> None:
    """Two builds of the record from the same inputs agree field for field."""
    if not RESULTS_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")

    first = load_results()
    second = load_results()
    assert first.model_dump() == second.model_dump()

    text = RESULTS_FILE.read_text(encoding="utf-8")
    for stamp in ("generated_at", "timestamp", "run_date", "created_at", "duration_seconds"):
        assert stamp not in text, f"{stamp} would make the artefact non-reproducible"

    report = REPORT_FILE.read_text(encoding="utf-8")
    assert "no generation date" in report


def test_the_report_states_that_odds_are_not_probability() -> None:
    """A content regression guard against the commonest misreading."""
    if not REPORT_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    report = REPORT_FILE.read_text(encoding="utf-8")

    assert "Odds are not probability" in report
    assert "does **not** mean the probability doubles" in report
    assert "log-odds" in report


#: Phrases the report must never assert. Each one is still allowed to *appear*
#: inside a disclaimer — the report says what it is not — so the guard checks the
#: window immediately before every occurrence for a negation rather than banning
#: the substring outright, which would flag the report's own caveats.
FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "causes churn",
    "cause churn",
    "makes customers leave",
    "make customers leave",
    "will reduce churn",
    "reduces churn",
    "protects the customer",
    "drives churn",
    "the most important factor",
    "most important feature",
)

_NEGATIONS = ("not", "never", "no ", "cannot", "does not", "must not")


def test_the_report_never_claims_causation_or_universal_importance() -> None:
    if not REPORT_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    lowered = REPORT_FILE.read_text(encoding="utf-8").lower()

    for phrase in FORBIDDEN_CLAIMS:
        for match in re.finditer(re.escape(phrase), lowered):
            window = lowered[max(0, match.start() - 60) : match.start()]
            assert any(negation in window for negation in _NEGATIONS), (
                f"{phrase!r} appears without a negation in front of it"
            )


def test_every_mention_of_top_features_carries_its_qualifier() -> None:
    if not REPORT_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    lowered = REPORT_FILE.read_text(encoding="utf-8").lower()

    for match in re.finditer(r"top features", lowered):
        window = lowered[match.start() : match.start() + 140]
        assert "empirical contribution dispersion" in window


def test_the_report_qualifies_the_ranking_wherever_it_names_it() -> None:
    if not REPORT_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    report = REPORT_FILE.read_text(encoding="utf-8")

    assert "empirical contribution dispersion in the training pool" in report
    assert "*true feature importance*" in report
    assert "**not** called importance" in report
    assert "`importance = abs(coefficient)` ranking is produced here" in report
    assert "no model change" in report
    assert "no selection" in report
    assert "no holdout-based feature ranking" in report


def _training_features():
    """The frozen training pool as a validated feature matrix.

    Defined at the bottom because it is a helper, and it loads the **training
    pool** — never the holdout, which is what the tests above are about.
    """
    from churn.preprocessing.splitting import load_training_pool

    return build_feature_matrix(load_training_pool())


# --- structural dependencies between raw features ------------------------------
#
# The correction these cover: an exact algebraic contrast is not automatically a
# change a customer could undergo. What is coupled must be *measured*, never
# assumed, and a rule that fails must be incapable of claiming it held.


NO_INTERNET_SERVICES = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

#: What the training pool actually shows, pinned so a drift fails a test.
NO_INTERNET_ROWS = 1214
NO_PHONE_ROWS = 559
N_STRUCTURAL_RULES = 7


def _pool():
    from churn.preprocessing.splitting import load_training_pool

    return load_training_pool()


def test_the_declared_rules_are_the_seven_product_semantics_ones() -> None:
    """Declared from the data dictionary, before any of them was checked."""
    relationships = {rule.relationship for rule in STRUCTURAL_RULES}

    assert len(STRUCTURAL_RULES) == N_STRUCTURAL_RULES
    for service in NO_INTERNET_SERVICES:
        assert f"InternetService=No <-> {service}=No internet service" in relationships
    assert "PhoneService=No <-> MultipleLines=No phone service" in relationships


def test_the_structural_check_reads_only_the_frame_it_is_given(monkeypatch) -> None:
    """It loads nothing, so it cannot reach the holdout even by accident."""
    import churn.preprocessing.splitting as splitting

    def forbid(*args, **kwargs):
        raise AssertionError("the structural check loaded a partition")

    monkeypatch.setattr(splitting, "load_holdout", forbid)
    monkeypatch.setattr(splitting, "load_training_pool", forbid)
    monkeypatch.setattr(splitting, "load_split", forbid)

    frame = pd.DataFrame(
        {
            "InternetService": ["No", "DSL"],
            "OnlineSecurity": ["No internet service", "Yes"],
        }
    )
    check = check_structural_dependency(
        frame, StructuralRule("InternetService", "No", "OnlineSecurity", "No internet service")
    )

    assert check.n_rows_checked == 2
    assert check.violations == 0
    assert check.deterministic is True


def test_every_declared_rule_holds_on_the_training_pool() -> None:
    checks = check_structural_dependencies(_pool())

    assert len(checks) == N_STRUCTURAL_RULES
    for check in checks:
        assert check.n_rows_checked == TRAINING_POOL_ROWS, check.rule.relationship
        assert check.violations == 0, check.rule.relationship
        assert check.left_only == 0 and check.right_only == 0, check.rule.relationship
        assert check.deterministic is True, check.rule.relationship
        assert check.n_left == check.n_right == check.n_both, check.rule.relationship

    by_left = {check.rule.right_feature: check for check in checks}
    for service in NO_INTERNET_SERVICES:
        assert by_left[service].n_both == NO_INTERNET_ROWS
    assert by_left["MultipleLines"].n_both == NO_PHONE_ROWS


def test_a_rule_with_violations_can_never_report_itself_deterministic() -> None:
    """``deterministic`` is derived from the count; it cannot be asserted."""
    frame = pd.DataFrame(
        {
            # One row breaks the equivalence in each direction.
            "InternetService": ["No", "No", "DSL", "DSL"],
            "OnlineSecurity": ["No internet service", "Yes", "No internet service", "No"],
        }
    )
    check = check_structural_dependency(
        frame, StructuralRule("InternetService", "No", "OnlineSecurity", "No internet service")
    )

    assert check.left_only == 1
    assert check.right_only == 1
    assert check.violations == 2
    assert check.deterministic is False

    record = check.as_dict("synthetic")
    assert record["deterministic_in_training_pool"] is False
    assert record["violations"] == 2
    assert "does NOT hold" in record["interpretation_consequence"]


def test_only_deterministic_rules_contribute_coupled_levels() -> None:
    """A tendency is not a reason to qualify an exact contrast."""
    holds = StructuralCheck(
        rule=StructuralRule("A", "x", "B", "y"),
        n_rows_checked=10,
        n_left=3,
        n_right=3,
        n_both=3,
        left_only=0,
        right_only=0,
    )
    fails = StructuralCheck(
        rule=StructuralRule("C", "p", "D", "q"),
        n_rows_checked=10,
        n_left=4,
        n_right=3,
        n_both=3,
        left_only=1,
        right_only=0,
    )

    coupled = coupled_levels_from([holds, fails])

    assert coupled == {"A": ("x",), "B": ("y",)}
    assert "C" not in coupled and "D" not in coupled


def test_the_coupled_columns_are_literally_the_same_vector() -> None:
    """The premise of the redundancy argument, checked rather than asserted."""
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    transformed = transform_features(pipeline, _training_features())

    blocks = coupled_column_blocks(terms, groups, transformed)
    assert len(blocks) == 2

    sizes = sorted(len(block.columns) for block in blocks)
    assert sizes == [2, 7]

    for block in blocks:
        assert block.spans_multiple_raw_features
        reference = transformed[:, block.columns[0]]
        for column in block.columns:
            assert np.array_equal(reference, transformed[:, column])
        # Identical columns are perfectly collinear, so L2 splits the shared
        # weight evenly. That is the whole reason the repeats are not evidence.
        assert block.max_pairwise_coefficient_difference == 0.0
        assert block.aggregate_coefficient == pytest.approx(
            block.equal_split_share * len(block.columns), abs=1e-15
        )


def test_the_no_internet_block_covers_internet_service_and_its_six_services() -> None:
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    transformed = transform_features(pipeline, _training_features())

    block = max(coupled_column_blocks(terms, groups, transformed), key=lambda b: len(b.columns))
    members = dict(block.members)

    assert block.n_active_rows == NO_INTERNET_ROWS
    assert members["InternetService"] == "No"
    for service in NO_INTERNET_SERVICES:
        assert members[service] == "No internet service"
    assert block.aggregate_coefficient == pytest.approx(sum(block.coefficients), abs=1e-15)


def test_a_within_feature_block_is_never_reported() -> None:
    """Two levels of one feature cannot be the same vector under one-hot."""
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    transformed = transform_features(pipeline, _training_features())

    for block in coupled_column_blocks(terms, groups, transformed):
        assert len({feature for feature, _ in block.members}) == len(block.members)


def test_contrasts_are_still_beta_b_minus_beta_a_when_flagged() -> None:
    """The flag qualifies the reading; it must not touch the arithmetic."""
    terms = extract_terms(_frozen_pipeline())
    groups = feature_groups(terms)
    group = next(g for g in groups if g.feature == "OnlineSecurity")

    plain = level_contrasts(terms, group)
    flagged = level_contrasts(terms, group, coupled_levels=("No internet service",))

    assert len(plain) == len(flagged)
    for first, second in zip(plain, flagged, strict=True):
        assert first.level_a == second.level_a
        assert first.level_b == second.level_b
        assert first.delta_log_odds == second.delta_log_odds
        assert first.modeled_odds_ratio == second.modeled_odds_ratio

    touching = [c for c in flagged if "No internet service" in (c.level_a, c.level_b)]
    assert touching, "the sentinel takes part in some contrast"
    for contrast in touching:
        assert contrast.structurally_coupled is True
        assert contrast.raw_single_feature_counterfactual_supported is False
        assert "algebraic contrast" in contrast.reading
    for contrast in flagged:
        if "No internet service" not in (contrast.level_a, contrast.level_b):
            assert contrast.structurally_coupled is False
            assert contrast.raw_single_feature_counterfactual_supported is True


def test_the_record_flags_exactly_the_coupled_levels() -> None:
    results = _results()
    coupled = results.structural_dependencies["coupled_levels"]

    assert set(coupled) == {
        "InternetService",
        "PhoneService",
        "MultipleLines",
        *NO_INTERNET_SERVICES,
    }
    assert coupled["InternetService"] == ["No"]
    assert coupled["PhoneService"] == ["No"]
    assert coupled["MultipleLines"] == ["No phone service"]
    for service in NO_INTERNET_SERVICES:
        assert coupled[service] == ["No internet service"]

    for term in results.categorical_terms:
        expected = coupled.get(term["feature"], [])
        assert term["structurally_coupled_levels"] == expected, term["feature"]
        assert term["has_structurally_coupled_level"] is bool(expected)
        for level, flag in term["structurally_coupled_per_level"].items():
            assert flag is (level in expected)


def test_the_record_reports_seven_deterministic_rules_and_no_violations() -> None:
    structural = _results().structural_dependencies

    assert structural["population"] == "training pool"
    assert structural["n_rows_checked"] == TRAINING_POOL_ROWS
    assert structural["holdout_used"] is False
    assert structural["rules_declared_before_checking"] is True
    assert structural["n_rules_checked"] == N_STRUCTURAL_RULES
    assert structural["n_deterministic"] == N_STRUCTURAL_RULES
    assert structural["n_with_violations"] == 0
    assert structural["contrasts_removed_or_altered"] is False
    assert structural["coefficients_removed_or_altered"] is False

    for rule in structural["rules"]:
        assert rule["violations"] == 0
        assert rule["deterministic_in_training_pool"] is True
        assert rule["n_rows_checked"] == TRAINING_POOL_ROWS


def test_a_tampered_determinism_claim_fails_the_invariants() -> None:
    """A rule with violations claiming determinism must be caught."""
    results = _results()
    payload = results.model_dump()
    payload["structural_dependencies"]["rules"][0]["violations"] = 3
    payload["structural_dependencies"]["rules"][0]["left_only"] = 3

    failures = invariant_failures(results.__class__(**payload))
    assert any("determinism inconsistent" in failure for failure in failures)


def test_a_tampered_coupling_flag_fails_the_invariants() -> None:
    results = _results()
    payload = results.model_dump()
    for term in payload["categorical_terms"]:
        if term["feature"] == "OnlineSecurity":
            term["structurally_coupled_levels"] = []

    failures = invariant_failures(results.__class__(**payload))
    assert any("flags the wrong coupled levels" in failure for failure in failures)


def test_the_methodology_records_the_structural_correction() -> None:
    methodology = _results().methodology

    assert methodology["structural_dependencies_verified_empirically"] is True
    assert methodology["structural_dependencies_assumed_without_checking"] is False
    assert methodology["coupled_contrasts_flagged"] is True
    assert methodology["coupled_contrasts_removed"] is False
    assert methodology["repeated_coefficients_read_as_independent_evidence"] is False
    assert methodology["contribution_dispersion_representation_dependent"] is True


def test_the_ranking_records_the_redundancy_limitation() -> None:
    dispersion = _results().contribution_dispersion

    assert dispersion["representation_dependent"] is True
    assert dispersion["ranks_may_be_summed"] is False
    assert dispersion["coupled_features_are_independent_signals"] is False
    assert set(dispersion["features_with_a_structurally_coupled_level"]) == {
        "InternetService",
        "PhoneService",
        "MultipleLines",
        *NO_INTERNET_SERVICES,
    }
    assert "representation" in dispersion["redundancy_caveat"].lower()


def test_the_report_explains_the_repeated_sentinel_coefficients() -> None:
    if not REPORT_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    report = REPORT_FILE.read_text(encoding="utf-8")

    assert "Structural coupling between raw features" in report
    assert "must not be read as" in report
    assert "independent evidence from several services" in report
    assert "structurally coupled in this dataset" in report
    assert "distributes one" in report and "underlying no-internet state" in report
    assert "perfectly collinear" in report
    assert "identical predictions" in report
    # The verified counts must be visible, not merely asserted in prose.
    assert "Violations" in report
    assert "`InternetService=No <-> OnlineSecurity=No internet service`" in report


def test_the_report_marks_the_ranking_as_representation_dependent() -> None:
    if not REPORT_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    report = REPORT_FILE.read_text(encoding="utf-8")

    assert "representation-dependent" in report
    assert "Do not sum ranks" in report
    assert "Do not read the coupled service features as that many independent signals" in report
    assert "could redistribute the same variability" in report


def test_the_report_qualifies_coupled_contrasts_as_algebraic() -> None:
    if not REPORT_FILE.is_file():
        pytest.skip("the interpretation has not been run yet")
    report = REPORT_FILE.read_text(encoding="utf-8")

    assert "Single-feature counterfactual supported" in report
    assert "algebraic" in report.lower()
    assert "structurally coupled" in report.lower()
    assert "describes a row that does not occur" in report


def test_the_correction_left_every_central_number_untouched() -> None:
    """The whole point: qualification changed the prose, not the arithmetic."""
    results = _results()

    assert results.model["intercept"] == FROZEN_INTERCEPT
    assert results.model["n_transformed_features"] == N_TRANSFORMED
    assert results.model["n_raw_features"] == N_RAW
    assert results.reconstruction["max_abs_logit_error"] == 0.0
    assert results.reconstruction["max_abs_probability_error"] == 1.1102230246251565e-16
    assert results.reconstruction["max_grouped_reconstruction_error"] == 1.7763568394002505e-15
    assert results.frozen_decision["threshold_probability"] == FROZEN_THRESHOLD
    assert results.frozen_decision["threshold_logit"] == -0.7205610102182292
    assert results.provenance["model_fingerprint_sha256_after_load"] == MODEL_FINGERPRINT
    assert results.provenance["pipeline_sha256_after_analysis"] == PIPELINE_SHA256

    ranked = [(row["rank"], row["feature"]) for row in results.contribution_dispersion["ranking"]]
    assert ranked[:5] == [
        (1, "tenure"),
        (2, "MonthlyCharges"),
        (3, "InternetService"),
        (4, "Contract"),
        (5, "TotalCharges"),
    ]
    assert ranked[-1] == (19, "PhoneService")


def test_the_regrouped_reconstruction_still_holds_after_the_correction() -> None:
    pipeline = _frozen_pipeline()
    terms = extract_terms(pipeline)
    groups = feature_groups(terms)
    features = _training_features()

    reconstruction = verify_reconstruction(pipeline, terms, groups, features)

    assert reconstruction.max_grouped_reconstruction_error <= RECONSTRUCTION_TOLERANCE
    assert reconstruction.within_tolerance is True
