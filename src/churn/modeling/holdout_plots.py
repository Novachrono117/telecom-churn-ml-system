"""The four figures of the final holdout evaluation.

Every one of them carries **FINAL TEST** in its title. That is not decoration:
every other curve in this repository is cross-validated training-pool
performance, and these are the only figures in the project that show a test
result. A reader who confuses the two draws the wrong conclusion from both.

Each figure also marks the **frozen operating point**, because a PR or ROC curve
describes every threshold the model could have used, while the system that was
frozen uses exactly one. Showing the curve without the point invites reading the
best part of it as the result.
"""

from __future__ import annotations

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
)

#: Colour of the frozen decision, shared with the Phase 9B and 9C figures.
DECISION_COLOR = "#0f7a6e"

#: Suffix appended to every title in this module.
FINAL_TEST = "final test set"


def _finish(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    average_precision: float,
    prevalence: float,
    operating_point: tuple[float, float],
    threshold: float,
) -> Figure:
    """The holdout precision-recall curve, with the frozen operating point marked.

    Precision-recall rather than ROC as the primary curve: at a ~26.5% positive
    rate the negative class dominates, and ROC's specificity axis is generous
    about false positives in a way precision is not. The dashed line is the
    no-skill precision, which for a PR curve is the prevalence — a curve is only
    meaningful relative to it.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Probability of the positive class.
        average_precision: AP on the holdout, for the legend.
        prevalence: Holdout positive rate.
        operating_point: ``(recall, precision)`` of the frozen threshold.
        threshold: The frozen threshold, for the annotation.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_probability)

    figure, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.axhline(
        prevalence,
        color=BASELINE,
        linestyle="--",
        linewidth=1.4,
        label=f"No skill = prevalence ({prevalence:.3f})",
    )
    ax.plot(
        recall,
        precision,
        color=COLOR_CHURNED,
        linewidth=1.9,
        label=f"Frozen model (AP {average_precision:.3f})",
    )
    ax.plot(
        [operating_point[0]],
        [operating_point[1]],
        marker="o",
        markersize=8.5,
        markerfacecolor=DECISION_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.4,
        linestyle="none",
        zorder=4,
        label=f"Frozen operating point (t = {threshold:.4f})",
    )
    ax.annotate(
        f"recall {operating_point[0]:.3f}\nprecision {operating_point[1]:.3f}",
        xy=operating_point,
        xytext=(operating_point[0] - 0.30, operating_point[1] + 0.16),
        fontsize=8.5,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.9},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    ax.legend(loc="upper right")
    _finish(ax, f"Precision-recall — {FINAL_TEST}", "Recall", "Precision")
    return figure


def plot_roc_curve(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    roc_auc: float,
    operating_point: tuple[float, float],
    threshold: float,
) -> Figure:
    """The holdout ROC curve, with the frozen operating point marked.

    Reported alongside the PR curve rather than instead of it. ROC-AUC has one
    property that matters here: it is invariant to the class balance, which makes
    it comparable across samples with different prevalence — while also being the
    reason it looks flattering on an imbalanced problem.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Probability of the positive class.
        roc_auc: ROC-AUC on the holdout, for the legend.
        operating_point: ``(false positive rate, recall)`` of the frozen threshold.
        threshold: The frozen threshold, for the annotation.
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_probability)

    figure, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.plot(
        [0, 1],
        [0, 1],
        color=INK_MUTED,
        linestyle="--",
        linewidth=1.2,
        label="No skill (0.500)",
    )
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        color=COLOR_RETAINED,
        linewidth=1.9,
        label=f"Frozen model (ROC-AUC {roc_auc:.3f})",
    )
    ax.plot(
        [operating_point[0]],
        [operating_point[1]],
        marker="o",
        markersize=8.5,
        markerfacecolor=DECISION_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.4,
        linestyle="none",
        zorder=4,
        label=f"Frozen operating point (t = {threshold:.4f})",
    )
    ax.annotate(
        f"FPR {operating_point[0]:.3f}\nrecall {operating_point[1]:.3f}",
        xy=operating_point,
        xytext=(operating_point[0] + 0.10, operating_point[1] - 0.22),
        fontsize=8.5,
        color=INK_SECONDARY,
        arrowprops={"arrowstyle": "-", "color": INK_MUTED, "linewidth": 0.9},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    _finish(
        ax,
        f"ROC — {FINAL_TEST}",
        "False positive rate",
        "True positive rate (recall)",
    )
    return figure


def plot_probability_distribution(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    n_bins: int = 40,
) -> Figure:
    """Where the frozen model puts its probability mass, by true class.

    The diagnostic that makes the confusion matrix legible: the two
    distributions overlap, and the threshold is a vertical line through that
    overlap. Everything to the right of it is flagged, so the churners left of
    the line are the false negatives and the retained customers right of it are
    the false positives — visible as areas rather than as four numbers.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_probability: Probability of the positive class.
        threshold: The frozen decision threshold.
        n_bins: Histogram bins, shared by both classes.
    """
    labels = np.asarray(y_true).astype(int)
    probability = np.asarray(y_probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    figure, ax = plt.subplots(figsize=(8.0, 4.6))
    for value, colour, name in ((0, COLOR_RETAINED, "Retained"), (1, COLOR_CHURNED, "Churned")):
        selected = probability[labels == value]
        ax.hist(
            selected,
            bins=edges,
            color=colour,
            alpha=0.55,
            label=f"{name} (n = {selected.size:,})",
        )

    ax.axvline(
        threshold,
        color=DECISION_COLOR,
        linestyle="-.",
        linewidth=1.7,
        zorder=4,
        label=f"Frozen threshold ({threshold:.4f})",
    )

    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    _finish(
        ax,
        f"Predicted probability by true class — {FINAL_TEST}",
        "Predicted probability of churn",
        "Customers",
    )
    ax.text(
        0.0,
        -0.20,
        "Right of the line is flagged. Churners to the left are the false negatives; retained "
        "customers to the right are the false positives.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_SECONDARY,
        verticalalignment="top",
    )
    return figure


def plot_confusion_matrix(
    true_negatives: int,
    false_positives: int,
    false_negatives: int,
    true_positives: int,
    threshold: float,
) -> Figure:
    """The 2x2 confusion matrix of the frozen rule, annotated with row shares.

    Row-normalised shading rather than raw counts: with 74% of the holdout in the
    negative class, colouring by count would make the two negative cells dominate
    and say nothing about the errors. Shading by row means each row reads as
    "of the customers who actually did this, what fraction did the model call".

    Args:
        true_negatives: Retained, predicted retained.
        false_positives: Retained, predicted churn.
        false_negatives: Churned, predicted retained.
        true_positives: Churned, predicted churn.
        threshold: The frozen threshold, for the subtitle.
    """
    matrix = np.array(
        [[true_negatives, false_positives], [false_negatives, true_positives]], dtype=int
    )
    totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)

    figure, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")

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
                fontsize=10,
                color="#ffffff" if shares[row, column] > 0.45 else INK_PRIMARY,
            )

    ax.set_xticks([0, 1], ["Predicted: retain", "Predicted: churn"])
    ax.set_yticks([0, 1], ["Actual: retained", "Actual: churned"])
    ax.set_title(f"Confusion matrix — {FINAL_TEST}", loc="left", pad=32)
    ax.text(
        0,
        1.015,
        f"n = {matrix.sum():,} · frozen threshold {threshold:.4f} · rows are the true class",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
    )
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_color(BASELINE)
    return figure
