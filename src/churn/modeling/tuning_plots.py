"""The one figure Phase 8B needs that no earlier phase already draws.

Everything else — per-fold metric columns, paired delta rows, out-of-fold
curves — is drawn by the Phase 5 and Phase 7 modules, so a chart keeps the same
visual grammar across the repository.

A 48-cell heatmap of the boosting grid is deliberately **not** drawn. It would
show mean inner scores that differ in the fourth decimal, invite reading a
ranking into noise, and put the search's own selection scores at the centre of
attention when the honest estimate is the outer one.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from churn.analysis.plots import INK_MUTED, INK_PRIMARY, INK_SECONDARY


def plot_selection_stability(
    frequency: Mapping[str, Mapping[str, Mapping[str, int]]],
    n_folds: int,
    title: str,
) -> Figure:
    """How often each hyperparameter value won across the outer folds.

    One row per parameter, one bar segment per value. The question is whether
    the search lands in the same region regardless of the partition; a parameter
    whose bar is a single full-width block was chosen unanimously, and one split
    into several segments moved with the fold.

    Args:
        frequency: ``{experiment: {parameter: {value: count}}}``.
        n_folds: Number of outer folds, i.e. the width of a full bar.
        title: Figure title.
    """
    rows: list[tuple[str, str, Mapping[str, int]]] = [
        (experiment, parameter, values)
        for experiment, parameters in frequency.items()
        for parameter, values in parameters.items()
    ]
    figure, ax = plt.subplots(figsize=(8.2, 0.62 * len(rows) + 2.2))
    shades = ("#1b5fa8", "#4d8fd0", "#8fb9e6", "#c9dcf3")

    for row, (_, _, values) in enumerate(rows):
        left = 0.0
        ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
        for index, (value, count) in enumerate(ordered):
            ax.barh(
                row,
                count,
                left=left,
                height=0.62,
                color=shades[index % len(shades)],
                edgecolor="white",
                linewidth=1.2,
            )
            ax.text(
                left + count / 2,
                row,
                f"{value} ×{count}",
                va="center",
                ha="center",
                fontsize=8.5,
                color="white" if index < 2 else INK_PRIMARY,
            )
            left += count
        ax.text(
            n_folds + 0.08,
            row,
            "unanimous" if len(ordered) == 1 else f"{len(ordered)} distinct values",
            va="center",
            fontsize=8.5,
            color=INK_MUTED,
        )

    ax.set_yticks(
        range(len(rows)),
        [f"{experiment} · {parameter}" for experiment, parameter, _ in rows],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, n_folds * 1.42)
    ax.set_xticks(np.arange(0, n_folds + 1))
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel(f"Outer folds selecting the value (of {n_folds})")
    ax.grid(alpha=0.9, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.text(
        0.0,
        -0.14,
        "Frequency describes stability of the search, not evidence that a value is correct.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    return figure
