"""Small plotting helpers for the EDA.

Design rules applied consistently to every figure:

* churn is a **rate** question, so bars encode rates and the group size is
  annotated — a 100% rate over 29 customers must not look like a finding;
* two fixed categorical colors (retained / churned), validated for colour-vision
  deficiency, never cycled;
* two-way rate tables use a single-hue sequential ramp (magnitude), never a
  rainbow;
* the population churn rate is drawn as a reference line so every bar is read
  against the baseline;
* recessive grid and axes, no 3D, no pie charts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Fixed categorical slots. Validated all-pairs for CVD on the light surface.
COLOR_RETAINED = "#2a78d6"
COLOR_CHURNED = "#eb6834"

#: Single-hue sequential ramp (blue 100 -> 700) for magnitude encodings.
SEQUENTIAL_STEPS = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("churn_blue", SEQUENTIAL_STEPS)


def use_project_style() -> None:
    """Apply the shared figure style. Idempotent."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": BASELINE,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 130,
        }
    )


def _finish(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _baseline(ax: plt.Axes, rate: float, horizontal: bool = True) -> None:
    draw = ax.axhline if horizontal else ax.axvline
    draw(rate, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
    if horizontal:
        ax.text(
            1.0,
            rate,
            f" baseline {rate:.1%}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )


def save_figure(figure: Figure, path: Path) -> Path:
    """Write a figure to ``path`` (creating parent directories) and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_target_distribution(counts: pd.Series, title: str) -> Figure:
    """Absolute size of each class, annotated with its share."""
    total = counts.sum()
    figure, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = [COLOR_RETAINED if label == "No" else COLOR_CHURNED for label in counts.index]
    bars = ax.bar(counts.index.astype(str), counts.to_numpy(), color=colors, width=0.55)
    for bar, value in zip(bars, counts.to_numpy(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}\n{value / total:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_SECONDARY,
        )
    ax.set_ylim(0, counts.max() * 1.22)
    _finish(ax, title, "Churn", "Customers")
    return figure


def plot_churn_rate_bars(
    rates: pd.DataFrame,
    title: str,
    xlabel: str,
    baseline: float,
    horizontal: bool = False,
) -> Figure:
    """Churn rate per category, annotated with the group size."""
    labels = [str(index) for index in rates.index]
    values = rates["churn_rate"].to_numpy()
    sizes = rates["n"].to_numpy()

    if horizontal:
        figure, ax = plt.subplots(figsize=(7.2, 0.42 * len(labels) + 1.6))
        bars = ax.barh(labels, values, color=COLOR_CHURNED, height=0.62)
        ax.invert_yaxis()
        ax.axvline(baseline, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
        for bar, value, size in zip(bars, values, sizes, strict=True):
            ax.text(
                value + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1%}  (n={size:,})",
                va="center",
                fontsize=8.5,
                color=INK_SECONDARY,
            )
        ax.set_xlim(0, min(1.0, values.max() * 1.35))
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        ax.grid(axis="x", alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel("Churn rate")
        ax.set_ylabel(xlabel)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return figure

    figure, ax = plt.subplots(figsize=(max(5.0, 1.35 * len(labels) + 1.6), 3.8))
    bars = ax.bar(labels, values, color=COLOR_CHURNED, width=0.55)
    for bar, value, size in zip(bars, values, sizes, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1%}\nn={size:,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_ylim(0, min(1.0, values.max() * 1.35))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    _baseline(ax, baseline)
    _finish(ax, title, xlabel, "Churn rate")
    return figure


def plot_effect_sizes(table: pd.DataFrame, title: str) -> Figure:
    """Rank categorical features by Cramér's V.

    Effect size, not p-value: with 7 043 rows almost everything is "significant",
    so the ordering that matters is by magnitude.
    """
    figure, ax = plt.subplots(figsize=(7.0, 0.36 * len(table) + 1.6))
    labels = table["column"].tolist()
    values = table["cramers_v"].to_numpy()
    bars = ax.barh(labels, values, color=COLOR_RETAINED, height=0.62)
    ax.invert_yaxis()
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_xlim(0, values.max() * 1.18)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("Cramér's V (0 = no association, 1 = perfect)")
    ax.grid(axis="x", alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return figure


def plot_distribution_by_churn(
    frame: pd.DataFrame,
    column: str,
    flag: str,
    title: str,
    xlabel: str,
    bins: int = 36,
) -> Figure:
    """Overlaid distributions of a numeric column for each class (density scale)."""
    figure, ax = plt.subplots(figsize=(7.0, 3.8))
    retained = frame.loc[frame[flag] == 0, column].dropna()
    churned = frame.loc[frame[flag] == 1, column].dropna()
    ax.hist(
        [retained, churned],
        bins=bins,
        density=True,
        color=[COLOR_RETAINED, COLOR_CHURNED],
        label=["Retained", "Churned"],
    )
    ax.legend(loc="upper right")
    _finish(ax, title, xlabel, "Density")
    return figure


def plot_boxplot_by_group(
    frame: pd.DataFrame,
    value: str,
    group: str,
    flag: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Figure:
    """Distribution of a numeric value per group, split by churn."""
    figure, ax = plt.subplots(figsize=(7.4, 4.0))
    groups = list(pd.unique(frame[group].dropna()))
    width = 0.34
    series = (("Retained", COLOR_RETAINED, 0), ("Churned", COLOR_CHURNED, 1))
    for offset, (label, color, is_churn) in enumerate(series):
        data = [
            frame.loc[(frame[group] == name) & (frame[flag] == is_churn), value].dropna()
            for name in groups
        ]
        positions = [i + (offset - 0.5) * width for i in range(len(groups))]
        box = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": INK_PRIMARY, "linewidth": 1.4},
            whiskerprops={"color": BASELINE},
            capprops={"color": BASELINE},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(1.2)
        box["boxes"][0].set_label(label)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(name) for name in groups])
    ax.legend(loc="upper left")
    _finish(ax, title, xlabel, ylabel)
    return figure


def plot_rate_heatmap(
    rates: pd.DataFrame,
    counts: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Figure:
    """Two-way churn rate matrix, each cell annotated with rate and group size."""
    figure, ax = plt.subplots(figsize=(1.35 * rates.shape[1] + 3.4, 0.72 * rates.shape[0] + 2.4))
    image = ax.imshow(rates.to_numpy(), cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")

    for row in range(rates.shape[0]):
        for column in range(rates.shape[1]):
            rate = rates.to_numpy()[row, column]
            size = counts.to_numpy()[row, column]
            if pd.isna(rate):
                continue
            ax.text(
                column,
                row,
                f"{rate:.1%}\nn={int(size):,}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="#ffffff" if rate > 0.45 else INK_PRIMARY,
            )
    ax.set_xticks(range(rates.shape[1]), [str(c) for c in rates.columns])
    ax.set_yticks(range(rates.shape[0]), [str(i) for i in rates.index])
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    colorbar = figure.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Churn rate", color=INK_SECONDARY, fontsize=9)
    colorbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
    colorbar.outline.set_visible(False)
    return figure


def plot_scatter_by_churn(
    frame: pd.DataFrame,
    x: str,
    y: str,
    flag: str,
    title: str,
    xlabel: str,
    ylabel: str,
    highlight_x: pd.Series | None = None,
    highlight_y: pd.Series | None = None,
    highlight_label: str = "",
) -> Figure:
    """Scatter of two numeric columns coloured by churn, with an optional highlight.

    ``highlight_x``/``highlight_y`` draw hollow rings over specific rows; they are
    passed explicitly because the highlighted points may not have a value in ``y``
    (that is exactly the case for the blank ``TotalCharges`` records).
    """
    figure, ax = plt.subplots(figsize=(7.0, 4.4))
    for value, color, label in ((0, COLOR_RETAINED, "Retained"), (1, COLOR_CHURNED, "Churned")):
        subset = frame[frame[flag] == value]
        ax.scatter(subset[x], subset[y], s=6, alpha=0.35, color=color, label=label, linewidths=0)
    if highlight_x is not None and highlight_y is not None and len(highlight_x):
        ax.scatter(
            highlight_x,
            highlight_y,
            s=58,
            facecolors="none",
            edgecolors=INK_PRIMARY,
            linewidths=1.4,
            label=highlight_label,
        )
    ax.legend(loc="upper left", markerscale=1.6)
    _finish(ax, title, xlabel, ylabel)
    return figure
