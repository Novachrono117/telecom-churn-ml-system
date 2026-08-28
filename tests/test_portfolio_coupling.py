"""Grouping coupled features must change the layout and never the arithmetic.

The failure this guards against is a reading error with a numeric consequence. A
customer without internet produces seven correct terms — the trigger plus six add-ons
pinned to ``"No internet service"`` — and shown as seven rows they read as seven
independent reasons the score moved. They are one fact about the product's encoding.

So they are summed into one row. The two things that must then stay true are that the
sum is **exact** (nothing dropped, nothing reweighted) and that the grand total still
reconstructs the logit. A grouping that quietly changed a number would be worse than
no grouping at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.modeling.freeze import load_pipeline, positive_probability
from churn.modeling.interpretation import extract_terms, feature_groups
from churn.monitoring.structural import STRUCTURAL_RULES
from churn.portfolio.coupling import (
    COUPLING_BLOCKS,
    active_blocks,
    block_records,
    coupled_features,
    is_block_consistent,
    partition,
)
from churn.portfolio.explanation import explain_prepared_row
from churn.preprocessing.contracts import FEATURE_COLUMNS, prepare_features

#: One ordinary customer. Its values do not matter here — only that it activates no
#: block, so the coupled cases below are the ones being measured.
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

NO_INTERNET = {
    **BASE_RECORD,
    "InternetService": "No",
    "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service",
    "DeviceProtection": "No internet service",
    "TechSupport": "No internet service",
    "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service",
}
NO_PHONE = {**BASE_RECORD, "PhoneService": "No", "MultipleLines": "No phone service"}

#: Accepted by the API, counted by Phase 12 as a structural violation, and therefore
#: NOT grouped here: the columns are not saying one thing.
BROKEN_INTERNET = {**BASE_RECORD, "InternetService": "No", "OnlineSecurity": "No"}


@pytest.fixture(scope="module")
def machinery():
    pipeline = load_pipeline()
    terms = extract_terms(pipeline)
    return pipeline, terms, feature_groups(terms)


def _explain(machinery, record: dict[str, object]):
    pipeline, terms, groups = machinery
    features = prepare_features(pd.DataFrame([record]).loc[:, list(FEATURE_COLUMNS)])
    probability = float(np.asarray(positive_probability(pipeline, features), dtype=float)[0])
    return explain_prepared_row(
        pipeline=pipeline,
        terms=terms,
        groups=groups,
        features=features,
        served_probability=probability,
        prediction=int(probability >= 0.3272694566222328),
        decision="churn" if probability >= 0.3272694566222328 else "retained",
        threshold=0.3272694566222328,
        comparison=">=",
        calibration_policy="NONE",
    )


# --------------------------------------------------------------------------- #
# Where the blocks come from.
# --------------------------------------------------------------------------- #


def test_the_blocks_are_derived_from_the_frozen_equivalences() -> None:
    """Not restated. A rule added in Phase 12 must appear here without a second edit."""
    from_rules: dict[tuple[str, str], set[str]] = {}
    for rule in STRUCTURAL_RULES:
        from_rules.setdefault((rule.left_feature, rule.left_level), set()).add(rule.right_feature)

    derived = {
        (block.trigger_feature, block.trigger_level): set(block.dependents)
        for block in COUPLING_BLOCKS
    }

    assert derived == from_rules
    assert len(COUPLING_BLOCKS) == 2
    assert coupled_features() == {
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PhoneService",
        "MultipleLines",
    }


def test_a_block_carries_its_trigger_as_a_member() -> None:
    """Leaving the trigger out would break the partition and the grand total."""
    for block in COUPLING_BLOCKS:
        assert block.members[0] == block.trigger_feature
        assert len(block.members) == len(block.dependents) + 1

    internet = next(b for b in COUPLING_BLOCKS if b.trigger_feature == "InternetService")
    phone = next(b for b in COUPLING_BLOCKS if b.trigger_feature == "PhoneService")

    assert len(internet.members) == 7
    assert len(phone.members) == 2
    assert internet.dependent_level == "No internet service"
    assert phone.dependent_level == "No phone service"


def test_the_block_records_are_plain_serialisable_data() -> None:
    records = block_records()

    assert [entry["name"] for entry in records] == ["no_internet_service", "no_phone_service"]
    assert all(entry["n_members"] == len(entry["members"]) for entry in records)


# --------------------------------------------------------------------------- #
# When a block is active.
# --------------------------------------------------------------------------- #


def test_a_consistent_no_internet_record_activates_one_block() -> None:
    blocks = active_blocks(NO_INTERNET)

    assert [block.name for block in blocks] == ["no_internet_service"]


def test_a_consistent_no_phone_record_activates_the_phone_block() -> None:
    blocks = active_blocks(NO_PHONE)

    assert [block.name for block in blocks] == ["no_phone_service"]


def test_an_ordinary_record_activates_nothing() -> None:
    assert active_blocks(BASE_RECORD) == ()


def test_a_broken_equivalence_is_not_presented_as_one_block() -> None:
    """The record contradicted the relationship, so the presentation must not assert it.

    Phase 12 counts this as a structural violation and still scores the record. Here
    it means the seven columns are not one fact, and summing them would claim they
    were.
    """
    blocks = active_blocks(BROKEN_INTERNET)

    assert blocks == ()
    internet = next(b for b in COUPLING_BLOCKS if b.trigger_feature == "InternetService")
    assert is_block_consistent(internet, BROKEN_INTERNET) is False
    assert is_block_consistent(internet, NO_INTERNET) is True
    assert is_block_consistent(internet, BASE_RECORD) is True


def test_a_dependent_pinned_without_its_trigger_is_also_inconsistent() -> None:
    """The equivalence breaks in both directions, exactly as Phase 12 defines it."""
    orphan = {**BASE_RECORD, "OnlineSecurity": "No internet service"}
    internet = next(b for b in COUPLING_BLOCKS if b.trigger_feature == "InternetService")

    assert is_block_consistent(internet, orphan) is False
    assert active_blocks(orphan) == ()


def test_partition_returns_what_no_block_accounts_for() -> None:
    ungrouped = partition(list(FEATURE_COLUMNS), active_blocks(NO_INTERNET))

    assert "InternetService" not in ungrouped
    assert "OnlineSecurity" not in ungrouped
    assert "tenure" in ungrouped
    assert len(ungrouped) == 19 - 7


# --------------------------------------------------------------------------- #
# The arithmetic.
# --------------------------------------------------------------------------- #


def test_a_grouped_block_equals_the_sum_of_its_members(machinery) -> None:
    """Exact equality, not approximate: the group is a sum, not a summary."""
    explanation = _explain(machinery, NO_INTERNET)
    by_feature = {item.feature: item for item in explanation.contributions}

    assert len(explanation.grouped) == 1
    block = explanation.grouped[0]
    expected = sum(by_feature[name].contribution_log_odds for name in block.members)

    assert block.contribution_log_odds == expected
    assert len(block.members) == 7
    assert len(block.member_contributions) == 7


def test_the_grouped_total_still_reconstructs_the_logit(machinery) -> None:
    """The only property that makes grouping safe."""
    for record in (NO_INTERNET, NO_PHONE, BASE_RECORD, BROKEN_INTERNET):
        explanation = _explain(machinery, record)

        total = sum(item.contribution_log_odds for item in explanation.grouped) + sum(
            item.contribution_log_odds for item in explanation.ungrouped
        )

        assert explanation.intercept + total == pytest.approx(explanation.model_logit, abs=1e-12), (
            record["InternetService"]
        )


def test_grouping_changes_no_individual_contribution(machinery) -> None:
    """The flat list is unchanged by the grouped view existing beside it."""
    explanation = _explain(machinery, NO_INTERNET)

    flat = sum(item.contribution_log_odds for item in explanation.contributions)
    grouped = sum(item.contribution_log_odds for item in explanation.grouped) + sum(
        item.contribution_log_odds for item in explanation.ungrouped
    )

    assert flat == pytest.approx(grouped, abs=1e-12)
    assert len(explanation.contributions) == 19


def test_a_broken_record_is_explained_ungrouped_and_still_closes(machinery) -> None:
    """It is still scored, still explained, and its arithmetic still reconstructs."""
    explanation = _explain(machinery, BROKEN_INTERNET)

    assert explanation.grouped == ()
    assert len(explanation.ungrouped) == 19
    assert explanation.reconstruction.within_tolerance


def test_the_block_describes_why_it_is_one_row(machinery) -> None:
    explanation = _explain(machinery, NO_INTERNET)

    description = explanation.grouped[0].description

    assert "one fact about the customer" in description
    assert "summed rather than listed as independent factors" in description
