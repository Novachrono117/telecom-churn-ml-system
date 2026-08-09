"""Association measures between features and the target.

With 7 043 observations a p-value says almost nothing on its own: trivial
differences reach significance. Every test here is therefore returned **next to
an effect size**, and callers are expected to rank by effect, not by p-value.

None of these measures implies causation. They quantify association in this
particular sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from churn.analysis.frames import CHURN_FLAG


@dataclass(frozen=True)
class CategoricalAssociation:
    """Chi-square test plus Cramér's V for one categorical column."""

    column: str
    n_categories: int
    n: int
    chi2: float
    p_value: float
    cramers_v: float


@dataclass(frozen=True)
class NumericComparison:
    """Mann-Whitney U comparison of a numeric column between the two classes."""

    column: str
    n_churned: int
    n_retained: int
    median_churned: float
    median_retained: float
    u_statistic: float
    p_value: float
    rank_biserial: float


#: Yates' continuity correction is disabled throughout. SciPy applies it to 2x2
#: tables by default, which would shrink chi-square — and therefore Cramér's V —
#: only for the binary features, making them incomparable with the three- and
#: four-level ones. With expected counts in the hundreds here the correction is
#: also unnecessarily conservative.
YATES_CORRECTION = False


def cramers_v(contingency: pd.DataFrame) -> float:
    """Cramér's V for a contingency table.

    ``V = sqrt(chi2 / (n * (min(rows, cols) - 1)))``, bounded in ``[0, 1]``.
    Reported without bias correction; with these group sizes the correction is
    negligible and the uncorrected value is the one usually quoted.
    """
    table = contingency.to_numpy()
    chi2 = stats.chi2_contingency(table, correction=YATES_CORRECTION)[0]
    n = table.sum()
    smaller_dimension = min(table.shape) - 1
    if n == 0 or smaller_dimension == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * smaller_dimension)))


def categorical_association(
    frame: pd.DataFrame, column: str, target: str
) -> CategoricalAssociation:
    """Chi-square test of independence and Cramér's V between ``column`` and the target."""
    contingency = pd.crosstab(frame[column], frame[target])
    chi2, p_value, _, _ = stats.chi2_contingency(
        contingency.to_numpy(), correction=YATES_CORRECTION
    )
    return CategoricalAssociation(
        column=column,
        n_categories=int(contingency.shape[0]),
        n=int(contingency.to_numpy().sum()),
        chi2=float(chi2),
        p_value=float(p_value),
        cramers_v=cramers_v(contingency),
    )


def association_ranking(frame: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    """Rank categorical columns by Cramér's V (effect size), strongest first."""
    results = [categorical_association(frame, column, target) for column in columns]
    table = pd.DataFrame(
        {
            "column": [r.column for r in results],
            "n_categories": [r.n_categories for r in results],
            "chi2": [r.chi2 for r in results],
            "p_value": [r.p_value for r in results],
            "cramers_v": [r.cramers_v for r in results],
        }
    )
    return table.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numeric_comparison(frame: pd.DataFrame, column: str) -> NumericComparison:
    """Compare a numeric column between churned and retained customers.

    Uses Mann-Whitney U (no normality assumption; the charge and tenure
    distributions are visibly skewed and multimodal) and reports the
    rank-biserial correlation as effect size: **positive means churned
    customers rank higher**, ``0`` means no stochastic difference.
    """
    churned = frame.loc[frame[CHURN_FLAG] == 1, column].dropna()
    retained = frame.loc[frame[CHURN_FLAG] == 0, column].dropna()
    u_statistic, p_value = stats.mannwhitneyu(churned, retained, alternative="two-sided")
    pairs = len(churned) * len(retained)
    return NumericComparison(
        column=column,
        n_churned=len(churned),
        n_retained=len(retained),
        median_churned=float(churned.median()),
        median_retained=float(retained.median()),
        u_statistic=float(u_statistic),
        p_value=float(p_value),
        rank_biserial=float(2 * u_statistic / pairs - 1) if pairs else 0.0,
    )


def interpret_cramers_v(value: float) -> str:
    """Coarse verbal label for an effect size, to keep tables readable.

    Thresholds are the conventional 0.10 / 0.20 / 0.40 reading for tables of this
    size. They are a communication aid, not a decision rule.
    """
    if value < 0.10:
        return "negligible"
    if value < 0.20:
        return "weak"
    if value < 0.40:
        return "moderate"
    return "strong"
