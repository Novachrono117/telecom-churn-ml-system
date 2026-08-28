"""The local explanation must be the model's own arithmetic, not a story about it.

Every test here is a form of one question: **does this decomposition reproduce the
frozen pipeline?** An attribution method that is merely plausible is worse than none,
because it is believed. A logistic regression's response surface is known exactly, so
"plausible" is not the standard available — "identical" is, and that is what is
checked.

The identity under test::

    intercept + sum over the 19 raw features of contribution_f  ==  model logit
    sigmoid(model logit)                                        ==  served probability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.modeling.freeze import load_pipeline, positive_probability
from churn.modeling.interpretation import (
    RECONSTRUCTION_TOLERANCE,
    extract_terms,
    feature_groups,
    logit_of,
    sigmoid,
)
from churn.portfolio.explanation import (
    EXPLANATION_METHOD,
    ExplanationError,
    explain_prepared_row,
)
from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    prepare_features,
)

BASE_RECORD: dict[str, object] = {
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "TotalCharges": "845.50",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

#: Records chosen to exercise different corners of the contract, not to be typical.
CASES: dict[str, dict[str, object]] = {
    "fiber_month_to_month": BASE_RECORD,
    "no_internet": {
        **BASE_RECORD,
        "InternetService": "No",
        "OnlineSecurity": "No internet service",
        "OnlineBackup": "No internet service",
        "DeviceProtection": "No internet service",
        "TechSupport": "No internet service",
        "StreamingTV": "No internet service",
        "StreamingMovies": "No internet service",
        "Contract": "Two year",
    },
    "no_phone": {**BASE_RECORD, "PhoneService": "No", "MultipleLines": "No phone service"},
    "structural_zero": {**BASE_RECORD, "tenure": 0, "TotalCharges": ""},
    "long_tenure": {**BASE_RECORD, "tenure": 72, "TotalCharges": "8200.00", "Contract": "Two year"},
    "unseen_category": {**BASE_RECORD, "PaymentMethod": "Instant transfer (PIX)"},
}


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return load_pipeline()


@pytest.fixture(scope="module")
def terms(pipeline: Pipeline):
    return extract_terms(pipeline)


@pytest.fixture(scope="module")
def groups(terms):
    return feature_groups(terms)


def _prepared(record: dict[str, object]) -> pd.DataFrame:
    return prepare_features(pd.DataFrame([record]).loc[:, list(FEATURE_COLUMNS)])


def _explain(pipeline, terms, groups, record: dict[str, object]):
    """Score through the frozen pipeline, then decompose that row."""
    features = _prepared(record)
    probability = float(np.asarray(positive_probability(pipeline, features), dtype=float)[0])
    prediction = int(probability >= 0.3272694566222328)
    return explain_prepared_row(
        pipeline=pipeline,
        terms=terms,
        groups=groups,
        features=features,
        served_probability=probability,
        prediction=prediction,
        decision="churn" if prediction else "retained",
        threshold=0.3272694566222328,
        comparison=">=",
        calibration_policy="NONE",
    )


# --------------------------------------------------------------------------- #
# The identity.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_contributions_reconstruct_the_logit_exactly(pipeline, terms, groups, name) -> None:
    """``intercept + sum(contributions) == model_logit``, to floating-point noise.

    This is the claim that makes the display honest. If the terms did not sum back to
    the logit, the page would be showing a set of numbers that resembles the model's
    reasoning without being it.
    """
    explanation = _explain(pipeline, terms, groups, CASES[name])

    total = sum(item.contribution_log_odds for item in explanation.contributions)

    assert explanation.intercept + total == pytest.approx(explanation.model_logit, abs=1e-12)
    assert explanation.reconstruction.within_tolerance
    assert explanation.reconstruction.tolerance == RECONSTRUCTION_TOLERANCE


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_sigmoid_of_the_logit_reconstructs_the_probability(
    pipeline, terms, groups, name
) -> None:
    """The second half of the identity, against the probability that was served."""
    explanation = _explain(pipeline, terms, groups, CASES[name])

    derived = float(sigmoid(np.asarray([explanation.model_logit], dtype=float))[0])

    assert derived == pytest.approx(explanation.churn_probability, abs=RECONSTRUCTION_TOLERANCE)
    assert explanation.reconstruction.probability_error <= RECONSTRUCTION_TOLERANCE


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_logit_matches_the_pipelines_own_decision_function(
    pipeline, terms, groups, name
) -> None:
    """Checked against scikit-learn's own linear predictor, not only against itself."""
    features = _prepared(CASES[name])
    reference = float(np.asarray(pipeline.decision_function(features), dtype=float).ravel()[0])

    explanation = _explain(pipeline, terms, groups, CASES[name])

    assert explanation.model_logit == pytest.approx(reference, abs=RECONSTRUCTION_TOLERANCE)


def test_the_threshold_logit_is_the_threshold_in_log_odds(pipeline, terms, groups) -> None:
    """So the margin the page shows is the distance to the real boundary."""
    explanation = _explain(pipeline, terms, groups, BASE_RECORD)

    assert explanation.threshold_logit == pytest.approx(logit_of(explanation.threshold))
    assert explanation.margin_log_odds == pytest.approx(
        explanation.model_logit - explanation.threshold_logit
    )
    # The sign of the margin and the served decision must agree, or the picture on
    # screen would contradict the answer beside it.
    assert (explanation.margin_log_odds >= 0.0) == (explanation.prediction == 1)


# --------------------------------------------------------------------------- #
# Coverage of the contract.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_raw_feature_appears_exactly_once(pipeline, terms, groups, name) -> None:
    """Nineteen features in, nineteen terms out. No omission, no duplication."""
    explanation = _explain(pipeline, terms, groups, CASES[name])

    features = [item.feature for item in explanation.contributions]

    assert len(features) == 19
    assert len(set(features)) == 19
    assert set(features) == set(FEATURE_COLUMNS)
    assert {item.feature for item in explanation.contributions if item.kind == "numeric"} == set(
        NUMERIC_FEATURES
    )
    assert {
        item.feature for item in explanation.contributions if item.kind == "categorical"
    } == set(CATEGORICAL_FEATURES)


def test_the_grouped_and_ungrouped_views_partition_the_features(pipeline, terms, groups) -> None:
    """Grouping is a layout change, so the two views must cover the same 19 terms."""
    explanation = _explain(pipeline, terms, groups, CASES["no_internet"])

    grouped_members = [name for item in explanation.grouped for name in item.members]
    ungrouped = [item.feature for item in explanation.ungrouped]

    assert len(grouped_members) + len(ungrouped) == 19
    assert set(grouped_members).isdisjoint(ungrouped)
    assert set(grouped_members) | set(ungrouped) == set(FEATURE_COLUMNS)

    total = sum(item.contribution_log_odds for item in explanation.grouped) + sum(
        item.contribution_log_odds for item in explanation.ungrouped
    )
    assert explanation.intercept + total == pytest.approx(explanation.model_logit, abs=1e-12)


def test_an_unseen_category_contributes_exactly_nothing(pipeline, terms, groups) -> None:
    """``handle_unknown="ignore"`` encodes it as an all-zero block, and the page says so.

    Reporting a bare 0.0 would be true and useless: a reader cannot tell "this level
    happens to have no effect" from "the model never saw this level".
    """
    explanation = _explain(pipeline, terms, groups, CASES["unseen_category"])

    entry = next(item for item in explanation.contributions if item.feature == "PaymentMethod")

    assert entry.is_unseen_level is True
    assert entry.active_level is None
    assert entry.contribution_log_odds == 0.0
    assert entry.direction == "neutral"
    # Every other feature is still recognised.
    assert all(
        item.is_unseen_level is False
        for item in explanation.contributions
        if item.feature != "PaymentMethod"
    )


def test_a_structural_zero_is_displayed_as_blank_not_lost(pipeline, terms, groups) -> None:
    """A blank TotalCharges at zero tenure is legitimate, and must read as one."""
    explanation = _explain(pipeline, terms, groups, CASES["structural_zero"])

    entry = next(item for item in explanation.contributions if item.feature == "TotalCharges")

    assert entry.value_display == "(blank)"
    assert explanation.reconstruction.within_tolerance


def test_directions_follow_the_sign_of_the_term(pipeline, terms, groups) -> None:
    explanation = _explain(pipeline, terms, groups, BASE_RECORD)

    for item in explanation.contributions:
        if item.contribution_log_odds > 0:
            assert item.direction == "increases"
        elif item.contribution_log_odds < 0:
            assert item.direction == "decreases"
        else:
            assert item.direction == "neutral"


def test_the_local_ranking_is_by_absolute_contribution(pipeline, terms, groups) -> None:
    """Ordered for this request. Deliberately not called feature importance."""
    explanation = _explain(pipeline, terms, groups, BASE_RECORD)

    magnitudes = [item.magnitude for item in explanation.ranked()]

    assert magnitudes == sorted(magnitudes, reverse=True)
    assert len(explanation.ranked()) == len(explanation.grouped) + len(explanation.ungrouped)


# --------------------------------------------------------------------------- #
# The gate.
# --------------------------------------------------------------------------- #


def test_a_decomposition_that_does_not_close_is_refused(pipeline, terms, groups) -> None:
    """A wrong explanation must fail loudly rather than be shown with a caveat.

    The served probability is perturbed, which is what a genuine divergence between
    the scoring path and the decomposition would look like from in here.
    """
    features = _prepared(BASE_RECORD)
    probability = float(np.asarray(positive_probability(pipeline, features), dtype=float)[0])

    with pytest.raises(ExplanationError) as error:
        explain_prepared_row(
            pipeline=pipeline,
            terms=terms,
            groups=groups,
            features=features,
            served_probability=probability + 0.05,
            prediction=1,
            decision="churn",
            threshold=0.3272694566222328,
            comparison=">=",
            calibration_policy="NONE",
        )

    assert "does not reproduce" in str(error.value)
    assert "prediction itself is unaffected" in str(error.value)


def test_an_explanation_describes_one_record(pipeline, terms, groups) -> None:
    features = prepare_features(
        pd.DataFrame([BASE_RECORD, BASE_RECORD]).loc[:, list(FEATURE_COLUMNS)]
    )

    with pytest.raises(ValueError, match="one record"):
        explain_prepared_row(
            pipeline=pipeline,
            terms=terms,
            groups=groups,
            features=features,
            served_probability=0.5,
            prediction=1,
            decision="churn",
            threshold=0.3272694566222328,
            comparison=">=",
            calibration_policy="NONE",
        )


# --------------------------------------------------------------------------- #
# What the core must not do.
# --------------------------------------------------------------------------- #


def test_the_explanation_core_never_calls_predict_proba(
    pipeline, terms, groups, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It receives a probability; it does not produce one.

    If the core could score, there would be two inference paths in the process and
    the explanation could describe a number the API never returned.
    """

    def explode(*_: object, **__: object) -> None:
        raise AssertionError("the explanation core scored a record itself.")

    features = _prepared(BASE_RECORD)
    probability = float(np.asarray(positive_probability(pipeline, features), dtype=float)[0])

    monkeypatch.setattr(Pipeline, "predict_proba", explode)
    monkeypatch.setattr(Pipeline, "predict", explode)
    monkeypatch.setattr(Pipeline, "decision_function", explode)

    explanation = explain_prepared_row(
        pipeline=pipeline,
        terms=terms,
        groups=groups,
        features=features,
        served_probability=probability,
        prediction=1,
        decision="churn",
        threshold=0.3272694566222328,
        comparison=">=",
        calibration_policy="NONE",
    )

    assert explanation.churn_probability == probability


def test_the_explanation_is_deterministic(pipeline, terms, groups) -> None:
    """No sampling, no seed, no ordering that depends on dictionary iteration."""
    first = _explain(pipeline, terms, groups, BASE_RECORD).as_record()
    second = _explain(pipeline, terms, groups, BASE_RECORD).as_record()

    assert first == second


def test_the_record_names_its_method_and_carries_its_caveats(pipeline, terms, groups) -> None:
    explanation = _explain(pipeline, terms, groups, BASE_RECORD).as_record()

    assert explanation["explanation_method"] == EXPLANATION_METHOD
    assert "not causal effects" in explanation["causal_note"]
    assert "no post-hoc calibration" in explanation["calibration_note"]
    assert explanation["reconstruction"]["within_tolerance"] is True


def test_no_identifier_or_target_can_reach_the_explanation(pipeline, terms, groups) -> None:
    """The contract drops them upstream; this asserts none survives into the output."""
    explanation = _explain(pipeline, terms, groups, BASE_RECORD).as_record()
    rendered = str(explanation)

    assert "customerID" not in rendered
    assert "Churn" not in {item["feature"] for item in explanation["contributions"]}
    assert set(FEATURE_COLUMNS) == {item["feature"] for item in explanation["contributions"]}
