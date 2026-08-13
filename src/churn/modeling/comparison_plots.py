"""Figures for the Phase 7 model-family comparison.

Five, and no more. A confusion matrix per family is deliberately absent: the 0.5
rule selects nothing in this phase, so three of them would put a decision
boundary at the centre of attention that the protocol explicitly does not use.

Two of the five — the out-of-fold ROC and precision-recall curves — are drawn by
the Phase 5 plotting module rather than redrawn here, so a curve keeps the same
visual grammar across the repository.

The paired figures plot **every fold**, not only the mean. The question these
charts exist to answer is whether an advantage is *consistent*, and a mean alone
cannot show that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from churn.analysis.plots import INK_MUTED, INK_PRIMARY, INK_SECONDARY
from churn.modeling.plots import MODEL_COLORS, MODEL_LABELS


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, INK_SECONDARY)


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def _finish(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_metric_by_family(
    per_fold: Mapping[str, Sequence[float]],
    reference: str,
    metric_label: str,
    title: str,
) -> Figure:
    """Per-fold values and mean of one metric, one column per family.

    The reference model's mean is drawn as a horizontal line so the eye compares
    against it, and the five folds are drawn as points so the spread is visible
    next to the mean instead of being summarised away.
    """
    names = list(per_fold)
    figure, ax = plt.subplots(figsize=(1.9 * len(names) + 3.4, 4.6))
    reference_mean = float(np.mean(per_fold[reference]))

    ax.axhline(
        reference_mean,
        color=INK_MUTED,
        linestyle="--",
        linewidth=1.2,
        zorder=1,
        label=f"{_label(reference)} mean ({reference_mean:.4f})",
    )

    for position, name in enumerate(names):
        values = np.asarray(per_fold[name], dtype=float)
        color = _color(name)
        ax.scatter(
            np.full(values.shape, position),
            values,
            s=44,
            color=color,
            alpha=0.7,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            [position],
            [values.mean()],
            marker="_",
            s=900,
            color=color,
            linewidths=2.0,
            zorder=4,
        )
        ax.text(
            position,
            values.mean(),
            f"  {values.mean():.4f} ± {values.std(ddof=1):.4f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK_PRIMARY,
        )

    ax.set_xticks(range(len(names)), [_label(name) for name in names])
    ax.set_xlim(-0.5, len(names) - 0.2)
    _finish(ax, title, "", f"{metric_label} per fold (bar = mean over 5 folds)")
    return figure


def plot_paired_deltas(
    deltas: Mapping[str, Sequence[float]],
    reference_label: str,
    metric_label: str,
    title: str,
    caption: str,
    labels: Mapping[str, str] | None = None,
) -> Figure:
    """One row per experiment: its five paired per-fold deltas and their mean.

    Keys are family names so a family keeps its colour across every figure;
    ``labels`` supplies the row text, which carries the experiment id.
    """
    names = list(deltas)
    row_labels = dict(labels or {})
    figure, ax = plt.subplots(figsize=(7.8, 0.95 * len(names) + 2.4))
    ax.axvline(0.0, color=INK_PRIMARY, linewidth=1.3, zorder=2)

    for row, name in enumerate(names):
        values = np.asarray(deltas[name], dtype=float)
        color = _color(name)
        ax.scatter(
            values,
            np.full(values.shape, row),
            s=44,
            color=color,
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            [values.mean()],
            [row],
            marker="|",
            s=430,
            color=INK_PRIMARY,
            linewidths=1.8,
            zorder=4,
        )
        positive = int((values > 0).sum())
        ax.text(
            0.995,
            row,
            f" {values.mean():+.4f} · {positive}/{values.size} folds",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=color,
        )

    ax.set_yticks(range(len(names)), [row_labels.get(name, _label(name)) for name in names])
    ax.invert_yaxis()
    ax.margins(x=0.26)
    _finish(ax, title, f"Paired delta {metric_label} (− {reference_label}, same fold)", "")
    ax.text(0.0, -0.18, caption, transform=ax.transAxes, fontsize=8.5, color=INK_SECONDARY)
    return figure
