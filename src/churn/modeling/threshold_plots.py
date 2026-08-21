"""The figures Phase 9B needs that no earlier phase already draws.

Three of them plot a quantity against the threshold, which is the one axis every
earlier phase deliberately did not have. The fourth reads the same probabilities
as a ranking instead of as a cut.

Two rules govern all of them. Every threshold-dependent curve is drawn from
**out-of-fold probabilities on the training pool** and says so, because a
precision-recall curve with no provenance is routinely read as a test result and
this repository has none. And the two marked thresholds — the unoptimised default
and the frozen decision — are drawn as reference lines rather than annotated
after the fact, so a reader sees where the decision sits relative to the whole
curve instead of being handed one point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
from churn.modeling.threshold import ThresholdSweep, TopKRecord

#: Colour of the frozen decision line, distinct from both metric colours.
DECISION_COLOR = "#0f7a6e"

#: Line style per marked threshold, so the two are separable in grayscale too.
_MARKER_STYLE: dict[str, tuple[str, str]] = {
    "default": (INK_MUTED, "--"),
    "final": (DECISION_COLOR, "-."),
}


def _finish(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _mark(ax: plt.Axes, markers: Mapping[str, tuple[float, str]]) -> None:
    """Draw one vertical reference line per marked threshold."""
    for key, (threshold, label) in markers.items():
        color, style = _MARKER_STYLE.get(key, (INK_SECONDARY, ":"))
        ax.axvline(
            float(threshold),
            color=color,
            linestyle=style,
            linewidth=1.5,
            zorder=2,
            label=label,
        )


def plot_precision_recall_vs_threshold(
    sweep: ThresholdSweep,
    markers: Mapping[str, tuple[float, str]],
    prevalence: float,
    title: str,
) -> Figure:
    """Precision and recall as the decision threshold moves.

    The two curves cross where the model is equally good at not missing churners
    and at not bothering loyal customers; F1 is maximised near, but not exactly
    at, that crossing. The predicted positive rate is drawn on the same axis
    because a precision gain that comes from contacting a tenth as many customers
    is a different result from one that does not.

    Args:
        sweep: A sweep over the candidate thresholds of the training-pool OOF
            probabilities.
        markers: ``{key: (threshold, legend label)}`` reference lines.
        prevalence: Training-pool positive prevalence, the no-skill precision.
        title: Figure title.
    """
    figure, ax = plt.subplots(figsize=(8.2, 5.0))

    ax.axhline(
        prevalence,
        color=BASELINE,
        linestyle="--",
        linewidth=1.3,
        zorder=1,
        label=f"No-skill precision = prevalence ({prevalence:.3f})",
    )
    ax.plot(
        sweep.thresholds,
        sweep.precision,
        color=COLOR_RETAINED,
        linewidth=1.9,
        zorder=3,
        label="Precision",
    )
    ax.plot(
        sweep.thresholds,
        sweep.recall,
        color=COLOR_CHURNED,
        linewidth=1.9,
        zorder=3,
        label="Recall",
    )
    ax.plot(
        sweep.thresholds,
        sweep.predicted_positive_rate,
        color=INK_MUTED,
        linewidth=1.4,
        linestyle=":",
        zorder=3,
        label="Predicted positive rate",
    )
    _mark(ax, markers)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3)
    _finish(ax, title, "Decision threshold on the predicted probability", "Rate")
    return figure


def plot_f1_vs_threshold(
    sweep: ThresholdSweep,
    markers: Mapping[str, tuple[float, str]],
    title: str,
) -> Figure:
    """The selection objective across the whole threshold range.

    This is the curve ``POLICY_F1`` maximises, so it is the one that shows how
    much of a decision the maximum really is: a broad flat top means many
    thresholds are nearly equivalent and the exact value carries little
    information, while a sharp peak means the opposite.

    Args:
        sweep: A sweep over the candidate thresholds of the training-pool OOF
            probabilities.
        markers: ``{key: (threshold, legend label)}`` reference lines.
        title: Figure title.
    """
    figure, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.plot(sweep.thresholds, sweep.f1, color=INK_PRIMARY, linewidth=1.9, zorder=3, label="F1")
    best = int(np.argmax(sweep.f1))
    ax.plot(
        [sweep.thresholds[best]],
        [sweep.f1[best]],
        marker="o",
        markersize=6.5,
        color=DECISION_COLOR,
        zorder=4,
        linestyle="none",
        label=f"Maximum F1 = {sweep.f1[best]:.4f}",
    )
    _mark(ax, markers)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(0.05, float(sweep.f1.max()) * 1.15))
    ax.legend(loc="upper right")
    _finish(ax, title, "Decision threshold on the predicted probability", "F1 (positive class)")
    ax.text(
        0.0,
        -0.20,
        "F1-max is a research operating point, not a business-optimal threshold: no cost of a "
        "lost customer\nand no cost of a retention contact exists in this dataset.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_SECONDARY,
        verticalalignment="top",
    )
    return figure


def plot_threshold_stability(
    thresholds: Sequence[float],
    default: float,
    final_threshold: float,
    title: str,
) -> Figure:
    """The threshold each outer fold selected, against the two reference values.

    The question is whether the selection procedure lands in the same region
    regardless of the partition. A tight cluster means the threshold is a property
    of the data; a wide scatter means it is partly a property of the split, which
    is a limitation to state rather than a number to average away.

    Args:
        thresholds: One selected threshold per outer fold, in fold order.
        default: The unoptimised default threshold.
        final_threshold: The frozen decision threshold.
        title: Figure title.
    """
    values = np.asarray(thresholds, dtype=float)
    positions = np.arange(1, values.size + 1)

    figure, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.axhline(
        default,
        color=INK_MUTED,
        linestyle="--",
        linewidth=1.4,
        zorder=1,
        label=f"Default ({default:.3f})",
    )
    ax.axhline(
        final_threshold,
        color=DECISION_COLOR,
        linestyle="-.",
        linewidth=1.5,
        zorder=1,
        label=f"Frozen decision ({final_threshold:.4f})",
    )
    ax.plot(
        positions,
        values,
        marker="o",
        markersize=7.0,
        linestyle="none",
        color=COLOR_CHURNED,
        zorder=3,
        label="Selected inside the outer training fold",
    )
    for position, value in zip(positions, values, strict=True):
        ax.text(
            position,
            value,
            f"  {value:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK_SECONDARY,
        )

    span = float(values.max() - values.min())
    padding = max(0.02, span * 0.6)
    lower = min(float(values.min()), default, final_threshold) - padding
    upper = max(float(values.max()), default, final_threshold) + padding
    ax.set_xlim(0.5, values.size + 0.7)
    ax.set_ylim(max(0.0, lower), min(1.0, upper))
    ax.set_xticks(positions, [f"Fold {position}" for position in positions])
    # Below the axis: inside, the legend lands on the point labels.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncols=3)
    _finish(ax, title, "", "Selected threshold")
    ax.text(
        0.0,
        -0.30,
        "Dispersion is described, not used as a criterion: the eligibility rule was fixed before "
        "these thresholds existed.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    return figure


def plot_top_k_capture(
    records: Sequence[TopKRecord],
    prevalence: float,
    title: str,
) -> Figure:
    """What a fixed contact budget would capture, by rank.

    Recall and precision answer different halves of the capacity question — how
    much of the churn a budget reaches, and how much of the budget is spent on
    customers who would have stayed — so both are drawn, with the lift over the
    base rate annotated on the precision points.

    Args:
        records: One record per capacity level, in increasing order.
        prevalence: Training-pool positive prevalence, the no-targeting baseline.
        title: Figure title.
    """
    fractions = np.array([record.fraction for record in records], dtype=float)
    recall = np.array([record.recall for record in records], dtype=float)
    precision = np.array([record.precision for record in records], dtype=float)

    figure, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.axhline(
        prevalence,
        color=BASELINE,
        linestyle="--",
        linewidth=1.3,
        zorder=1,
        label=f"Precision of an untargeted draw ({prevalence:.3f})",
    )
    ax.plot(
        fractions,
        recall,
        marker="o",
        markersize=6.0,
        linewidth=1.9,
        color=COLOR_CHURNED,
        zorder=3,
        label="Recall — share of churners reached",
    )
    ax.plot(
        fractions,
        precision,
        marker="s",
        markersize=6.0,
        linewidth=1.9,
        color=COLOR_RETAINED,
        zorder=3,
        label="Precision — share of contacts that churn",
    )
    for record in records:
        ax.text(
            record.fraction,
            record.precision,
            f"  ×{record.lift:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK_SECONDARY,
        )

    ax.set_xlim(0.0, float(fractions.max()) + 0.045)
    ax.set_ylim(0.0, 1.0)
    # Explicit tick labels, so no percent formatter on this axis: it would
    # override them and render "5.0%" where the capacity level is "5%".
    ax.set_xticks(fractions, [f"{fraction:.0%}" for fraction in fractions])
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper left")
    _finish(ax, title, "Share of the training pool contacted, highest risk first", "Rate")
    ax.text(
        0.0,
        -0.18,
        "Hypothetical capacity: no evidence exists that a real retention team has any of these "
        "budgets.\nAnnotations are lift over the training-pool prevalence. This analysis does not "
        "select the frozen threshold.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_SECONDARY,
        verticalalignment="top",
    )
    return figure
