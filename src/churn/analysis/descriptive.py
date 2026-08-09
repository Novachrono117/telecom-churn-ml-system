"""Descriptive churn statistics.

Every function returns a table; interpretation stays with the caller. Rates are
always reported next to the group size, because a high rate over 29 customers
and a high rate over 3 875 customers are not the same finding.
"""

from __future__ import annotations

import pandas as pd

from churn.analysis.frames import CHURN_FLAG


def overall_churn_rate(frame: pd.DataFrame) -> float:
    """Return the population churn rate (share of the positive class)."""
    return float(frame[CHURN_FLAG].mean())


def churn_rate_by(frame: pd.DataFrame, column: str, sort: bool = True) -> pd.DataFrame:
    """Churn rate per category of ``column``.

    Args:
        frame: EDA frame containing ``churn_flag``.
        column: Categorical column to group by.
        sort: Sort by descending churn rate instead of category order.

    Returns:
        A table indexed by category with ``n``, ``churned``, ``churn_rate`` and
        ``share`` (the group's share of the population).
    """
    grouped = frame.groupby(column, observed=True)[CHURN_FLAG].agg(["size", "sum", "mean"])
    grouped.columns = ["n", "churned", "churn_rate"]
    grouped["share"] = grouped["n"] / len(frame)
    return grouped.sort_values("churn_rate", ascending=False) if sort else grouped


def churn_rate_matrix(
    frame: pd.DataFrame,
    index: str,
    columns: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-way churn rate table and the matching group sizes.

    Returns:
        ``(rates, counts)``. Cells backed by few customers must be read with the
        counts table in hand.
    """
    rates = frame.pivot_table(
        index=index, columns=columns, values=CHURN_FLAG, aggfunc="mean", observed=True
    )
    counts = frame.pivot_table(
        index=index, columns=columns, values=CHURN_FLAG, aggfunc="size", observed=True
    )
    return rates, counts


def numeric_summary_by_target(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Distribution summary of a numeric column split by churn.

    Missing values are excluded from the statistics and counted separately.
    """
    grouped = frame.groupby(CHURN_FLAG)[column]
    summary = grouped.agg(
        n="size",
        missing=lambda s: int(s.isna().sum()),
        mean="mean",
        std="std",
        minimum="min",
        q1=lambda s: s.quantile(0.25),
        median="median",
        q3=lambda s: s.quantile(0.75),
        maximum="max",
    )
    summary.index = summary.index.map({0: "retained", 1: "churned"})
    return summary


def count_yes(frame: pd.DataFrame, columns: list[str], value: str = "Yes") -> pd.Series:
    """Count how many of ``columns`` equal ``value`` on each row.

    A descriptive counter used to probe whether *how many* services a customer
    holds carries signal. It is not a pipeline feature: Phase 6 decides that.
    """
    return sum((frame[column] == value).astype(int) for column in columns)


def value_share(frame: pd.DataFrame, column: str, among_churners: bool = False) -> pd.Series:
    """Share of each category in the population, or among churners only."""
    subset = frame[frame[CHURN_FLAG] == 1] if among_churners else frame
    return subset[column].value_counts(normalize=True)
