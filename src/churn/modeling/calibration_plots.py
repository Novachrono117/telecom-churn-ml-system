"""The two figures Phase 9A needs that no earlier phase already draws.

The per-fold Brier and log-loss columns are drawn by the Phase 7 module, so a
chart keeps the same visual grammar across the repository; only the axis label
carries the direction, because in this phase lower is better.

A calibration figure has one specific way of lying: with uniform bins on a skewed
score distribution, the right-hand bins hold a handful of customers and their
observed frequency swings between 0 and 1 for reasons that have nothing to do
with calibration. Quantile bins keep every point backed by a comparable sample,
and the counts are annotated so a reader can see what each point rests on.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from churn.analysis.plots import BASELINE, INK_MUTED, INK_SECONDARY
from churn.modeling.calibration import (
    RELIABILITY_N_BINS,
    RELIABILITY_STRATEGY,
    reliability_bins,
)
from churn.modeling.plots import MODEL_COLORS, MODEL_LABELS


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, INK_SECONDARY)


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def plot_reliability_diagram(
    y_true: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    title: str,
    n_bins: int = RELIABILITY_N_BINS,
    strategy: str = RELIABILITY_STRATEGY,
) -> Figure:
    """Observed churn frequency against predicted probability, out of fold.

    The diagonal is perfect calibration. A curve **below** it means the model
    promises more churn than it delivers in that band — overconfident on the
    positive class; **above** it means the opposite. The vertical distance at
    each point is the per-bin gap that the Expected Calibration Error averages,
    which is why both use the same binning.

    Args:
        y_true: Binary ground truth, aligned with every probability vector.
        probabilities: ``{model key: out-of-fold probability}``.
        title: Figure title.
        n_bins: Requested bins per curve.
        strategy: Binning strategy, applied identically to every curve.
    """
    figure, ax = plt.subplots(figsize=(7.2, 5.8))

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=BASELINE,
        linestyle="--",
        linewidth=1.3,
        zorder=1,
        label="Perfect calibration",
    )

    limit = 0.0
    for name, probability in probabilities.items():
        predicted, observed, counts = reliability_bins(y_true, probability, n_bins, strategy)
        limit = max(limit, float(predicted.max()), float(observed.max()))
        ax.plot(
            predicted,
            observed,
            marker="o",
            markersize=5.5,
            linewidth=1.8,
            color=_color(name),
            alpha=0.9,
            zorder=3,
            label=f"{_label(name)} ({len(counts)} bins)",
        )

    ax.text(
        0.0,
        -0.20,
        f"Bins: {n_bins} requested, {strategy} edges, computed per curve so every point rests on "
        f"a comparable number of customers.\nPoints below the diagonal: the model promised more "
        f"churn than occurred in that band.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_SECONDARY,
        verticalalignment="top",
    )

    upper = min(1.0, limit * 1.08)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("Mean predicted probability in bin")
    ax.set_ylabel("Observed churn frequency in bin")
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left")
    return figure


def plot_probability_distribution(
    probabilities: Mapping[str, np.ndarray],
    distinct: Mapping[str, int],
    title: str,
    n_bins: int = 40,
) -> Figure:
    """Where each policy puts its probability mass, and how granular it is.

    This is not decoration. A calibrator can leave the Brier score barely moved
    while reshaping the distribution, and a step function that collapses a
    continuous score into a few plateaus is a real constraint on any threshold
    chosen later: a threshold can only land between plateaus. The count of
    distinct values makes that visible before Phase 9B runs into it.

    Args:
        probabilities: ``{model key: out-of-fold probability}``.
        distinct: ``{model key: number of distinct probabilities}``.
        title: Figure title.
        n_bins: Histogram bins, shared by every policy.
    """
    figure, ax = plt.subplots(figsize=(8.2, 4.6))
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    for name, probability in probabilities.items():
        ax.hist(
            probability,
            bins=edges,
            histtype="step",
            linewidth=1.8,
            color=_color(name),
            alpha=0.9,
            label=f"{_label(name)} — {distinct.get(name, 0):,} distinct values",
        )

    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("Out-of-fold predicted probability of churn")
    ax.set_ylabel("Customers")
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right")
    ax.text(
        0.0,
        -0.18,
        "A policy with few distinct values has collapsed the score into plateaus: a threshold "
        "chosen later can only land between them.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_MUTED,
    )
    return figure
