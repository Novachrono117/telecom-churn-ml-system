"""Figures for the baseline comparison.

Every chart here shows **cross-validated out-of-fold performance on the training
pool**. That phrase is written into each title, because a ROC curve with no
provenance is routinely read as a test-set result, and in this project there is
no test-set result until Phase 9.

The visual language (surface, ink, categorical slots, sequential ramp, recessive
grid) is imported from the EDA plotting module rather than redefined, so the
whole repository keeps one look.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve, roc_curve

from churn.analysis.plots import (
    BASELINE,
    COLOR_CHURNED,
    COLOR_RETAINED,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SEQUENTIAL_CMAP,
    SURFACE,
)

#: One fixed colour slot per baseline, so a model keeps its colour across every
#: figure. Validated all-pairs for colour-vision deficiency on the light surface.
MODEL_COLORS: dict[str, str] = {
    "majority": "#8a8880",
    "stratified_random": COLOR_RETAINED,
    "logistic_regression": COLOR_CHURNED,
    # Phase 7 families. Orange / deep blue / near-black separate under every
    # common colour-vision deficiency and also in grayscale, which a
    # blue-vs-violet or green-vs-orange pair would not.
    "random_forest": "#3c3b38",
    "hist_gradient_boosting": "#1b5fa8",
    # Phase 8A. The same family under a different representation, so a teal that
    # reads as related to the boosted blue without being confusable with it.
    "hgb_native": "#0f7a6e",
    # Phase 8B. The frozen baseline keeps the logistic hue; its tuned sibling
    # takes a darker shade of the same family, and tuned boosting keeps the
    # Phase 7 blue so a family is recognisable across reports.
    "logistic_frozen": COLOR_CHURNED,
    "logistic_tuned": "#a33d12",
    "hgb_tuned": "#1b5fa8",
    # Phase 9A. The uncalibrated policy IS the frozen logistic, so it keeps that
    # hue; the two calibrators take the blue and teal slots, a triple that stays
    # separable under every common colour-vision deficiency and in grayscale.
    "logistic_uncalibrated": COLOR_CHURNED,
    "logistic_sigmoid": "#2a78d6",
    "logistic_isotonic": "#0f7a6e",
    # Phase 9B. The same probabilities under two decision rules, so the two keep
    # the logistic hue family: muted grey for the unoptimised default, teal for
    # the selected policy, matching the decision line drawn in the figures.
    "threshold_default": "#8a8880",
    "threshold_nested_f1": "#0f7a6e",
}

#: Human-readable names, used in legends and axis labels.
MODEL_LABELS: dict[str, str] = {
    "majority": "Majority (always 'no churn')",
    "stratified_random": "Stratified random",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
    "hgb_native": "Hist gradient boosting (native categorical)",
    "logistic_frozen": "Logistic (frozen baseline)",
    "logistic_tuned": "Logistic (tuned)",
    "hgb_tuned": "Hist gradient boosting (tuned)",
    "logistic_uncalibrated": "C0 — uncalibrated",
    "logistic_sigmoid": "C1 — sigmoid calibration",
    "logistic_isotonic": "C2 — isotonic calibration",
    "threshold_default": "D0 — default threshold 0.5",
    "threshold_nested_f1": "D1 — nested F1-max threshold",
}

OOF_CAPTION = "Cross-validated out-of-fold performance on the training pool"


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


def plot_metric_comparison(
    summaries: Mapping[str, Mapping[str, Mapping[str, float]]],
    metrics: tuple[str, ...],
    metric_labels: Mapping[str, str],
    title: str,
) -> Figure:
    """Grouped bars of the cross-validated mean per model, with the fold spread.

    The error bar is one sample standard deviation across the five folds. It is
    drawn because a mean without its spread invites reading a 0.005 gap as a
    result.
    """
    names = list(summaries)
    positions = np.arange(len(metrics), dtype=float)
    width = 0.8 / len(names)

    figure, ax = plt.subplots(figsize=(1.9 * len(metrics) + 3.2, 4.2))
    for offset, name in enumerate(names):
        means = [summaries[name][metric]["mean"] for metric in metrics]
        errors = [summaries[name][metric]["std"] for metric in metrics]
        ax.bar(
            positions + (offset - (len(names) - 1) / 2) * width,
            means,
            width=width * 0.92,
            color=_color(name),
            label=_label(name),
            yerr=errors,
            capsize=3,
            error_kw={"ecolor": INK_PRIMARY, "elinewidth": 1.0},
        )

    ax.set_xticks(positions, [metric_labels.get(metric, metric) for metric in metrics])
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    # Below the axis: inside the plot the legend would sit on top of the tallest bar.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=len(names))
    _finish(ax, title, "", "Score (mean over 5 folds, bars = 1 SD)")
    return figure


def plot_roc_curves(
    target: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    roc_aucs: Mapping[str, float],
    title: str,
) -> Figure:
    """Out-of-fold ROC curves with the no-skill diagonal."""
    figure, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(
        [0, 1],
        [0, 1],
        color=INK_MUTED,
        linestyle="--",
        linewidth=1.2,
        label="No skill (0.500)",
    )

    for name, probability in probabilities.items():
        false_positive, true_positive, _ = roc_curve(target, probability)
        ax.plot(
            false_positive,
            true_positive,
            color=_color(name),
            linewidth=1.9,
            label=f"{_label(name)} ({roc_aucs[name]:.3f})",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    _finish(ax, title, "False positive rate", "True positive rate (recall)")
    return figure


#: A model whose scores take at most this many distinct values has no ranking to
#: speak of, so it has no precision-recall *curve* — only a single achievable
#: operating point.
_DEGENERATE_SCORE_LEVELS = 2


def plot_precision_recall_curves(
    target: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    average_precisions: Mapping[str, float],
    prevalence: float,
    title: str,
) -> Figure:
    """Out-of-fold precision-recall curves against the prevalence baseline.

    Constant-score baselines are deliberately **not** drawn as curves.
    ``precision_recall_curve`` returns two points for them, and joining those
    points produces a long straight segment that reads as a series of achievable
    precision/recall trade-offs. No such trade-offs exist: a model that assigns
    every customer the same score has one operating point. Drawing the line would
    make the majority baseline look better than random over most of the range,
    which is exactly backwards. The dashed prevalence line already carries what
    they contribute — the no-skill reference.
    """
    figure, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.axhline(
        prevalence,
        color=INK_MUTED,
        linestyle="--",
        linewidth=1.4,
        label=f"No skill = prevalence ({prevalence:.3f})",
    )

    skipped: list[str] = []
    for name, probability in probabilities.items():
        if np.unique(probability).size <= _DEGENERATE_SCORE_LEVELS:
            skipped.append(_label(name))
            continue
        precision, recall, _ = precision_recall_curve(target, probability)
        ax.plot(
            recall,
            precision,
            color=_color(name),
            linewidth=1.9,
            label=f"{_label(name)} (AP {average_precisions[name]:.3f})",
        )

    if skipped:
        ax.text(
            0.02,
            0.06,
            "Not drawn (constant-score, one operating point each):\n"
            + "\n".join(f"· {name} — AP ≈ prevalence" for name in skipped),
            transform=ax.transAxes,
            fontsize=8.5,
            color=INK_SECONDARY,
            va="bottom",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    _finish(ax, title, "Recall", "Precision")
    return figure


def plot_confusion_matrix(matrix: np.ndarray, title: str, subtitle: str) -> Figure:
    """A 2x2 confusion matrix annotated with counts and row shares."""
    figure, ax = plt.subplots(figsize=(5.4, 4.4))
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    ax.imshow(shares, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")

    corners = (("True negative", "False positive"), ("False negative", "True positive"))
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{corners[row][column]}\n{matrix[row, column]:,}\n"
                f"{shares[row, column]:.1%} of row",
                ha="center",
                va="center",
                fontsize=9.5,
                color="#ffffff" if shares[row, column] > 0.45 else INK_PRIMARY,
            )

    ax.set_xticks([0, 1], ["Predicted: retain", "Predicted: churn"])
    ax.set_yticks([0, 1], ["Actual: retained", "Actual: churned"])
    # The title is lifted far enough to leave room for the subtitle beneath it.
    ax.set_title(title, loc="left", pad=32)
    ax.text(
        0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
    )
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color(BASELINE)
    figure.patch.set_facecolor(SURFACE)
    return figure
