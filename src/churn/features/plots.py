"""Figures for the Phase 6 ablation.

Three, and no more. A confusion matrix per candidate would be noise: the 0.5
rule is not what this phase selects on, and repeating a chart five times says
less than one chart that puts the five candidates side by side.

Every figure shows cross-validated results on the training pool. The deltas are
**paired** — candidate minus baseline within the same fold — which is why the
zero line, not the baseline's spread, is the reference the eye should use.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

from churn.analysis.plots import (
    BASELINE,
    COLOR_CHURNED,
    COLOR_RETAINED,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
)

CLASSIFICATION_COLORS: dict[str, str] = {
    "PROMISING": COLOR_CHURNED,
    "INCONCLUSIVE": "#8a8880",
    "NOT_SUPPORTED": COLOR_RETAINED,
}

OOF_CAPTION = "Cross-validated out-of-fold performance on the training pool"


def _finish(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_paired_deltas(
    deltas: Mapping[str, Sequence[float]],
    classifications: Mapping[str, str],
    metric_label: str,
    title: str,
) -> Figure:
    """One row per candidate: its five paired per-fold deltas and their mean.

    The individual folds are plotted, not only the average, because the question
    the classification asks is about **consistency of direction** — whether the
    dots sit on one side of zero — and an average alone hides that.
    """
    names = list(deltas)
    figure, ax = plt.subplots(figsize=(7.6, 0.78 * len(names) + 2.2))
    ax.axvline(0.0, color=INK_PRIMARY, linewidth=1.3, zorder=2)

    for row, name in enumerate(names):
        values = np.asarray(deltas[name], dtype=float)
        color = CLASSIFICATION_COLORS.get(classifications.get(name, ""), INK_SECONDARY)
        ax.scatter(
            values,
            np.full(values.shape, row),
            s=42,
            color=color,
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            [values.mean()],
            [row],
            marker="|",
            s=420,
            color=INK_PRIMARY,
            linewidths=1.8,
            zorder=4,
        )
        ax.text(
            0.995,
            row,
            f" {classifications.get(name, '')}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8.5,
            color=color,
        )

    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.margins(x=0.18)
    _finish(ax, title, f"Paired delta {metric_label} (candidate − baseline, same fold)", "")
    ax.text(
        0.0,
        -0.16,
        "Dots: one per fold.  Vertical bar: mean of the five deltas.  "
        "Right of the line: the candidate improved on that fold.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    return figure


def plot_absolute_metrics(
    means: Mapping[str, Mapping[str, float]],
    stds: Mapping[str, Mapping[str, float]],
    metrics: Sequence[str],
    metric_labels: Mapping[str, str],
    title: str,
) -> Figure:
    """Absolute mean per experiment for the two ranking metrics, with fold spread."""
    names = list(means)
    figure, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.2), sharey=False)
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics, strict=True):
        values = [means[name][metric] for name in names]
        errors = [stds[name][metric] for name in names]
        baseline_value = means[names[0]][metric]
        ax.axhline(
            baseline_value,
            color=INK_MUTED,
            linestyle="--",
            linewidth=1.2,
            zorder=1,
            label=f"E0 baseline ({baseline_value:.4f})",
        )
        ax.bar(
            names,
            values,
            color=[COLOR_RETAINED if name == names[0] else COLOR_CHURNED for name in names],
            width=0.6,
            yerr=errors,
            capsize=3,
            error_kw={"ecolor": INK_PRIMARY, "elinewidth": 1.0},
            zorder=2,
        )
        low = min(v - e for v, e in zip(values, errors, strict=True))
        high = max(v + e for v, e in zip(values, errors, strict=True))
        padding = max((high - low) * 0.35, 0.005)
        ax.set_ylim(low - padding, high + padding)
        ax.legend(loc="lower right")
        _finish(ax, metric_labels.get(metric, metric), "", "Mean over 5 folds (bars = 1 SD)")

    figure.suptitle(title, x=0.005, ha="left", fontsize=12, fontweight="semibold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def plot_oof_precision_recall(
    target: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    average_precisions: Mapping[str, float],
    prevalence: float,
    title: str,
) -> Figure:
    """Out-of-fold precision-recall, baseline against the retained candidate."""
    figure, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.axhline(
        prevalence,
        color=INK_MUTED,
        linestyle="--",
        linewidth=1.4,
        label=f"No skill = prevalence ({prevalence:.3f})",
    )

    palette = (COLOR_RETAINED, COLOR_CHURNED)
    for (name, probability), color in zip(probabilities.items(), palette, strict=False):
        precision, recall, _ = precision_recall_curve(target, probability)
        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=1.9,
            label=f"{name} (AP {average_precisions[name]:.4f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_color(BASELINE)
    _finish(ax, title, "Recall", "Precision")
    return figure
