"""Figures for the post-hoc error analysis.

Two rules govern all of them, and both exist to stop a descriptive chart from
reading as a finding.

**Rates only where the denominator supports them.** A bar for a group whose
guardrail failed is not drawn short — it is not drawn at all, and the group is
labelled as unavailable. A missing bar is honest; a bar over eleven churners is
not.

**Nothing is sorted by the result.** Categories keep their own order, so the
reader's eye lands where the data puts it rather than where the largest gap is.

Every title says *post-hoc diagnostic* and names the frozen threshold, because
these are descriptions of a closed result, not evidence for a decision.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
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
from churn.modeling.error_analysis import (
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    MARGIN_BAND_LABELS,
    OUTCOME_CLASSES,
    TRUE_NEGATIVE,
    TRUE_POSITIVE,
    SliceMetrics,
)

#: One fixed colour per outcome class. Correct decisions in the muted pair,
#: errors in the saturated one, so an error is visible without reading a legend.
OUTCOME_COLORS: dict[str, str] = {
    TRUE_NEGATIVE: "#9ec5f4",
    FALSE_POSITIVE: COLOR_RETAINED,
    FALSE_NEGATIVE: COLOR_CHURNED,
    TRUE_POSITIVE: "#f0a883",
}

OUTCOME_LABELS: dict[str, str] = {
    TRUE_NEGATIVE: "TN — retained, not flagged",
    FALSE_POSITIVE: "FP — retained, flagged",
    FALSE_NEGATIVE: "FN — churned, not flagged",
    TRUE_POSITIVE: "TP — churned, flagged",
}

DIAGNOSTIC = "post-hoc diagnostic"


def _finish(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_error_probability_distribution(
    probability: np.ndarray,
    outcomes: np.ndarray,
    threshold: float,
    n_bins: int = 40,
) -> Figure:
    """Predicted probability by outcome class, with the frozen cut marked.

    The chart that makes the confusion matrix legible as a picture: the two error
    classes sit on opposite sides of the line, and how far they sit from it is
    the difference between a boundary effect and a confident mistake. Drawn as
    outlines rather than filled bars because the four distributions overlap
    heavily and filled areas would hide the smaller two.
    """
    values = np.asarray(outcomes)
    scores = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    figure, ax = plt.subplots(figsize=(8.4, 4.8))
    for name in OUTCOME_CLASSES:
        selected = scores[values == name]
        if selected.size == 0:
            continue
        ax.hist(
            selected,
            bins=edges,
            histtype="step",
            linewidth=1.9,
            color=OUTCOME_COLORS[name],
            label=f"{OUTCOME_LABELS[name]} (n = {selected.size:,})",
        )

    ax.axvline(
        threshold,
        color=INK_PRIMARY,
        linestyle="-.",
        linewidth=1.6,
        zorder=4,
        label=f"Frozen threshold ({threshold:.4f})",
    )
    ax.legend(loc="upper right", fontsize=8.5)
    _finish(
        ax,
        f"Predicted probability by outcome class — {DIAGNOSTIC}",
        "Predicted probability of churn",
        "Customers",
    )
    return figure


def plot_margin_distribution(counts_by_band: dict[str, dict[str, int]]) -> Figure:
    """How many of each outcome sit at each distance from the frozen threshold.

    Read the two error series: if most false negatives fall in the narrowest
    band, the misses are a boundary effect that any nearby threshold would have
    shuffled. If they fall in the widest, they are cases the model scored
    confidently and wrongly, which no threshold would have fixed.
    """
    bands = list(MARGIN_BAND_LABELS)
    positions = np.arange(len(bands), dtype=float)
    width = 0.8 / len(OUTCOME_CLASSES)

    figure, ax = plt.subplots(figsize=(8.6, 4.6))
    for offset, name in enumerate(OUTCOME_CLASSES):
        values = [counts_by_band[band][name] for band in bands]
        bars = ax.bar(
            positions + (offset - (len(OUTCOME_CLASSES) - 1) / 2) * width,
            values,
            width=width * 0.9,
            color=OUTCOME_COLORS[name],
            label=OUTCOME_LABELS[name],
        )
        for bar, value in zip(bars, values, strict=True):
            if value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK_SECONDARY,
                )

    ax.set_xticks(positions, bands)
    ax.legend(loc="upper left", fontsize=8.5)
    _finish(
        ax,
        f"Distance from the frozen threshold by outcome — {DIAGNOSTIC}",
        "|probability − threshold|",
        "Customers",
    )
    return figure


def plot_error_rates_by_slice(
    table: Sequence[SliceMetrics],
    axis_label: str,
    threshold: float,
) -> Figure | None:
    """False-positive and false-negative rate per group, with their denominators.

    Returns ``None`` when no group in the table has both rates available: a chart
    made of gaps would mislead more than a table of nulls, so the report falls
    back to the table in that case.

    Each bar is annotated with the denominator the rate rests on, because a
    22% false-negative rate over 24 churners and the same rate over 240 are not
    the same statement.
    """
    usable = [
        entry
        for entry in table
        if entry.false_positive_rate is not None or entry.false_negative_rate is not None
    ]
    if not usable:
        return None

    groups = [entry.group for entry in table]
    positions = np.arange(len(groups), dtype=float)
    width = 0.38

    figure, ax = plt.subplots(figsize=(1.7 * len(groups) + 3.4, 4.6))
    for offset, (attribute, colour, label, denominator) in enumerate(
        (
            ("false_positive_rate", COLOR_RETAINED, "FPR = FP / actual negatives", "negatives"),
            ("false_negative_rate", COLOR_CHURNED, "FNR = FN / actual positives", "positives"),
        )
    ):
        centres = positions + (offset - 0.5) * width
        for centre, entry in zip(centres, table, strict=True):
            value = getattr(entry, attribute)
            if value is None:
                ax.text(
                    centre,
                    0.01,
                    "n/a",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK_MUTED,
                    rotation=90,
                )
                continue
            ax.bar(centre, value, width=width * 0.92, color=colour)
            ax.text(
                centre,
                value,
                f"{value:.1%}\nn={getattr(entry, denominator):,}",
                ha="center",
                va="bottom",
                fontsize=7.8,
                color=INK_SECONDARY,
            )
        ax.bar(np.nan, 0, color=colour, label=label)

    ax.set_xticks(positions, groups)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    _finish(ax, f"Error rates by {axis_label} — {DIAGNOSTIC}", "", "Rate")
    ax.text(
        0.0,
        -0.30,
        f"Frozen threshold {threshold:.4f}. Rates whose denominator falls below the guardrail are "
        f"marked n/a rather than drawn.\nGroup order is the category order, never sorted by the "
        f"result. Descriptive only: nothing here selects anything.",
        transform=ax.transAxes,
        fontsize=8.2,
        color=INK_SECONDARY,
        verticalalignment="top",
    )
    for side in ("bottom",):
        ax.spines[side].set_color(BASELINE)
    return figure
