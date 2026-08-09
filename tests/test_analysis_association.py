"""Tests for the association measures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn.analysis.association import (
    association_ranking,
    categorical_association,
    cramers_v,
    interpret_cramers_v,
    numeric_comparison,
)
from churn.analysis.frames import CHURN_FLAG


def test_cramers_v_is_one_for_a_perfect_association() -> None:
    table = pd.DataFrame([[50, 0], [0, 50]], index=["a", "b"], columns=["No", "Yes"])

    assert cramers_v(table) == pytest.approx(1.0)


def test_cramers_v_is_zero_when_groups_are_identical() -> None:
    table = pd.DataFrame([[25, 25], [25, 25]], index=["a", "b"], columns=["No", "Yes"])

    assert cramers_v(table) == pytest.approx(0.0, abs=1e-12)


def test_cramers_v_is_bounded() -> None:
    rng = np.random.default_rng(42)
    table = pd.DataFrame(rng.integers(5, 100, size=(4, 2)))

    assert 0.0 <= cramers_v(table) <= 1.0


def test_categorical_association_reports_shape_and_effect() -> None:
    frame = pd.DataFrame({"x": ["a"] * 50 + ["b"] * 50, "Churn": ["Yes"] * 50 + ["No"] * 50})

    result = categorical_association(frame, "x", "Churn")

    assert result.column == "x"
    assert result.n == 100
    assert result.n_categories == 2
    assert result.cramers_v == pytest.approx(1.0)
    assert result.p_value < 0.01


def test_association_ranking_orders_by_effect_size_not_p_value() -> None:
    rng = np.random.default_rng(7)
    n = 600
    churn = np.where(np.arange(n) % 2 == 0, "Yes", "No")
    frame = pd.DataFrame(
        {
            "strong": churn,
            "noise": rng.choice(["a", "b"], size=n),
            "Churn": churn,
        }
    )

    ranking = association_ranking(frame, ["noise", "strong"], "Churn")

    assert ranking.loc[0, "column"] == "strong"
    assert ranking.loc[0, "cramers_v"] > ranking.loc[1, "cramers_v"]


def test_numeric_comparison_sign_means_churners_rank_higher() -> None:
    frame = pd.DataFrame({"value": [1, 2, 3, 10, 11, 12], CHURN_FLAG: [0, 0, 0, 1, 1, 1]})

    result = numeric_comparison(frame, "value")

    assert result.rank_biserial == pytest.approx(1.0)
    assert result.median_churned > result.median_retained


def test_numeric_comparison_sign_flips_when_churners_rank_lower() -> None:
    frame = pd.DataFrame({"value": [10, 11, 12, 1, 2, 3], CHURN_FLAG: [0, 0, 0, 1, 1, 1]})

    result = numeric_comparison(frame, "value")

    assert result.rank_biserial == pytest.approx(-1.0)


def test_numeric_comparison_excludes_missing_values() -> None:
    frame = pd.DataFrame(
        {"value": [1.0, 2.0, None, 10.0, 11.0, None], CHURN_FLAG: [0, 0, 0, 1, 1, 1]}
    )

    result = numeric_comparison(frame, "value")

    assert result.n_retained == 2
    assert result.n_churned == 2


@pytest.mark.parametrize(
    ("value", "label"),
    [(0.05, "negligible"), (0.15, "weak"), (0.30, "moderate"), (0.45, "strong")],
)
def test_interpret_cramers_v_labels(value: float, label: str) -> None:
    assert interpret_cramers_v(value) == label
