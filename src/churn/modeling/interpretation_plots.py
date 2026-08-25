"""Figures for the interpretation of the frozen model.

Three rules govern all of them, and each exists to stop a chart from asserting
more than the arithmetic behind it.

**Never one bar per dummy.** Putting all 43 one-hot coefficients on a single axis
would invite exactly the reading the phase refuses: that a dummy's magnitude is
comparable across features and interpretable on its own. The categorical figure
is therefore faceted by raw feature, so every comparison the eye makes is a
**within-feature** one, which is the identified quantity.

**Axes are labelled in the unit they are in.** Coefficients and contributions are
log-odds, not probabilities and not percentages, and every axis says so.

**Nothing claims causality.** Titles say *frozen model*, captions say *modelled*,
and no figure is titled "importance".
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from churn.analysis.plots import (
    BASELINE,
    COLOR_CHURNED,
    COLOR_RETAINED,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
)
from churn.modeling.interpretation import (
    CategoricalTerm,
    ContributionDispersion,
    NumericTerm,
)

#: One colour per direction of the effect on the model's log-odds. The same two
#: slots the whole project uses for churned/retained, so a bar pushing the score
#: up is the churn colour everywhere.
COLOR_UP = COLOR_CHURNED
COLOR_DOWN = COLOR_RETAINED

FROZEN = "frozen model"


def _finish(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _direction_colour(value: float) -> str:
    return COLOR_UP if value >= 0 else COLOR_DOWN


def plot_numeric_coefficients(terms: Sequence[NumericTerm]) -> Figure:
    """The three standardised numeric coefficients, with their 1-SD odds ratios.

    Standardised coefficients are on a shared scale — one standard deviation of
    each feature *as measured on the training pool* — which is what makes these
    three bars comparable with one another. They are **not** comparable with a
    one-hot coefficient, which is why no dummy appears on this axis.
    """
    features = [term.feature for term in terms]
    values = [term.standardized_coefficient for term in terms]
    positions = np.arange(len(features), dtype=float)

    figure, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.barh(
        positions,
        values,
        height=0.55,
        color=[_direction_colour(value) for value in values],
    )
    # The annotation sits on the empty side of zero, opposite its own bar. Placing
    # it beyond the bar's far end would push a long label off the axis and over
    # the tick labels once a coefficient is large.
    for position, term in zip(positions, terms, strict=True):
        value = term.standardized_coefficient
        ax.text(
            0.06 if value < 0 else -0.06,
            position,
            f"β = {value:+.4f}   OR per 1 SD = {term.odds_ratio_per_1_sd:.3f}",
            va="center",
            ha="left" if value < 0 else "right",
            fontsize=8.6,
            color=INK_SECONDARY,
        )

    ax.axvline(0.0, color=INK_PRIMARY, linewidth=1.2, zorder=3)
    ax.set_yticks(positions, features)
    ax.invert_yaxis()
    span = max(abs(min(values)), abs(max(values)))
    ax.set_xlim(-span * 1.9, span * 1.9)
    _finish(
        ax,
        f"Standardised numeric coefficients — {FROZEN}",
        "Change in model log-odds per 1 standard deviation of the standardised feature",
    )
    ax.text(
        0.0,
        -0.42,
        "Positive pushes the model's log-odds toward a higher churn score; negative toward a "
        "lower one.\nHolding the other transformed features fixed. Modelled association, not a "
        "causal effect.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=INK_SECONDARY,
        verticalalignment="top",
    )
    return figure


def plot_categorical_contrasts(terms: Sequence[CategoricalTerm], n_columns: int = 4) -> Figure:
    """One small panel per categorical feature: its levels and their coefficients.

    Faceted rather than pooled, on purpose. Within a panel the horizontal
    distance between two dots **is** the identified quantity ``beta_a - beta_b``,
    and its exponential is the modelled odds ratio between those two levels.
    Across panels the absolute positions are not comparable, because the
    redundant one-hot parameterisation lets a constant move between a feature's
    levels and the shared intercept.

    Every panel is drawn on the same symmetric x range so the *widths* stay
    visually comparable even though the positions are not.
    """
    n_rows = int(np.ceil(len(terms) / n_columns))
    span = max(max(abs(value) for value in term.coefficients) for term in terms) * 1.25

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(3.6 * n_columns, 1.55 * n_rows + 1.4),
        sharex=True,
    )
    flat = np.asarray(axes).ravel()

    for ax, term in zip(flat, terms, strict=False):
        positions = np.arange(len(term.levels), dtype=float)
        ax.axvline(0.0, color=BASELINE, linewidth=1.0, zorder=1)
        ax.hlines(
            positions,
            0.0,
            term.coefficients,
            color=INK_MUTED,
            linewidth=1.0,
            zorder=2,
        )
        ax.scatter(
            term.coefficients,
            positions,
            s=46,
            zorder=3,
            color=[_direction_colour(value) for value in term.coefficients],
        )
        ax.set_yticks(positions, [level[:26] for level in term.levels], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-span, span)
        ax.set_title(
            f"{term.feature}  ·  spread {term.coefficient_spread:.3f}  "
            f"(OR {term.max_pairwise_modeled_odds_ratio:.2f})",
            loc="left",
            fontsize=9.2,
            pad=6,
        )
        ax.grid(axis="x", alpha=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    for ax in flat[len(terms) :]:
        ax.set_visible(False)

    figure.suptitle(
        f"Categorical coefficients by raw feature — {FROZEN}",
        x=0.005,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    figure.supxlabel(
        "Coefficient (log-odds). WITHIN a panel the distance between two dots is the identified "
        "contrast β_a − β_b, and exp of it is the modelled odds ratio between those two levels.\n"
        "ACROSS panels the absolute positions are not comparable: no baseline category was "
        "dropped, so a constant can move between a feature's levels and the shared intercept.",
        fontsize=8.4,
        color=INK_SECONDARY,
    )
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    return figure


def plot_contribution_dispersion(
    ranking: Sequence[dict],
    metric: str,
    metric_label: str,
) -> Figure:
    """The 19 raw features ordered by the pre-declared dispersion metric.

    Not titled "importance". The bar length is how much that feature's additive
    term moved around in the logit **across the training pool**, which is a joint
    property of the coefficients, the preprocessing and this population's
    composition.
    """
    features = [row["feature"] for row in ranking]
    values = [float(row["ranking_value"]) for row in ranking]
    kinds = [row["kind"] for row in ranking]
    positions = np.arange(len(features), dtype=float)

    figure, ax = plt.subplots(figsize=(9.0, 0.42 * len(features) + 2.4))
    ax.barh(
        positions,
        values,
        height=0.62,
        color=[COLOR_UP if kind == "numeric" else COLOR_DOWN for kind in kinds],
    )
    for position, value in zip(positions, values, strict=True):
        ax.text(
            value + max(values) * 0.012,
            position,
            f"{value:.4f}",
            va="center",
            fontsize=8.4,
            color=INK_SECONDARY,
        )

    ax.set_yticks(positions, features)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.bar(np.nan, 0, color=COLOR_UP, label="numeric (standardised)")
    ax.bar(np.nan, 0, color=COLOR_DOWN, label="categorical (one-hot)")
    ax.legend(loc="lower right")
    _finish(
        ax,
        f"Ranking by empirical contribution dispersion in the training pool — {FROZEN}",
        f"{metric_label} of that feature's contribution to the logit ({metric}, log-odds)",
    )
    ax.text(
        0.0,
        -0.10,
        "NOT an importance ranking in any universal sense. It depends on the fitted coefficients, "
        "the preprocessing, the composition of\nthe training pool and the correlations among the "
        "features. Not causal, not business importance, not a property of telecom churn.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=INK_SECONDARY,
        verticalalignment="top",
    )
    return figure


def plot_contribution_direction(
    dispersions: Sequence[ContributionDispersion],
    order: Sequence[str],
) -> Figure:
    """Where each raw feature's contribution sits relative to zero, and how wide.

    A direction summary, not a second ranking. The box is the interquartile
    range, the marker the median and the thin line the observed min-to-max span
    of that feature's additive term over the training pool. A feature whose whole
    box sits to the right of zero contributed a positive term to almost every
    customer's log-odds **in this population**; that is a statement about the
    distribution of the encoded inputs, not about any customer.
    """
    lookup = {record.feature: record for record in dispersions}
    records = [lookup[name] for name in order]
    positions = np.arange(len(records), dtype=float)

    figure, ax = plt.subplots(figsize=(9.0, 0.42 * len(records) + 2.4))
    ax.axvline(0.0, color=INK_PRIMARY, linewidth=1.2, zorder=4)

    for position, record in zip(positions, records, strict=True):
        ax.hlines(
            position,
            record.minimum,
            record.maximum,
            color=INK_MUTED,
            linewidth=1.1,
            zorder=2,
        )
        ax.barh(
            position,
            record.q3 - record.q1,
            left=record.q1,
            height=0.5,
            color=_direction_colour(record.median),
            alpha=0.85,
            zorder=3,
        )
        ax.scatter(
            record.median,
            position,
            marker="|",
            s=190,
            linewidths=1.9,
            color=INK_PRIMARY,
            zorder=5,
        )

    ax.set_yticks(positions, [record.feature for record in records])
    ax.invert_yaxis()
    ax.barh(np.nan, 0, color=COLOR_UP, label="median contribution ≥ 0")
    ax.barh(np.nan, 0, color=COLOR_DOWN, label="median contribution < 0")
    ax.legend(loc="lower right")
    _finish(
        ax,
        f"Direction and width of each feature's contribution — {FROZEN}",
        "Contribution to the model log-odds (bar = Q1–Q3, tick = median, line = min–max)",
    )
    ax.text(
        0.0,
        -0.10,
        "Right of zero pushes the model's log-odds toward a higher churn score, left toward a "
        "lower one, over the training pool.\nThe intercept is not shown: it is the shared constant "
        "and belongs to no feature. Modelled association, not a causal effect.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=INK_SECONDARY,
        verticalalignment="top",
    )
    return figure
