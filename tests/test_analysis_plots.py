"""Smoke tests for the plotting helpers.

These assert that a figure is produced with the labels a reader needs — never
pixels, colours or exact layout, which would be brittle and meaningless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from churn.analysis.descriptive import churn_rate_by  # noqa: E402
from churn.analysis.frames import CHURN_FLAG  # noqa: E402
from churn.analysis.plots import (  # noqa: E402
    plot_churn_rate_bars,
    plot_rate_heatmap,
    plot_target_distribution,
    save_figure,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Contract": ["M", "M", "M", "Y", "Y", "Y"],
            "Churn": ["Yes", "Yes", "No", "No", "No", "Yes"],
            CHURN_FLAG: [1, 1, 0, 0, 0, 1],
        }
    )


def test_target_distribution_figure_has_a_title(frame: pd.DataFrame) -> None:
    figure = plot_target_distribution(frame["Churn"].value_counts(), "Target")

    assert figure.axes[0].get_title(loc="left") == "Target"


def test_churn_rate_bars_label_the_axes(frame: pd.DataFrame) -> None:
    figure = plot_churn_rate_bars(churn_rate_by(frame, "Contract"), "T", "Contract", 0.5)
    ax = figure.axes[0]

    assert ax.get_xlabel() == "Contract"
    assert ax.get_ylabel() == "Churn rate"


def test_horizontal_churn_rate_bars_swap_the_axis_meaning(frame: pd.DataFrame) -> None:
    figure = plot_churn_rate_bars(
        churn_rate_by(frame, "Contract"), "T", "Contract", 0.5, horizontal=True
    )
    ax = figure.axes[0]

    assert ax.get_xlabel() == "Churn rate"
    assert ax.get_ylabel() == "Contract"


def test_rate_heatmap_annotates_every_populated_cell(frame: pd.DataFrame) -> None:
    rates = pd.DataFrame([[0.5, 0.25]], index=["M"], columns=["a", "b"])
    counts = pd.DataFrame([[10, 20]], index=["M"], columns=["a", "b"])

    figure = plot_rate_heatmap(rates, counts, "T", "x", "y")
    annotations = [text.get_text() for text in figure.axes[0].texts]

    assert any("50.0%" in text and "n=10" in text for text in annotations)
    assert any("25.0%" in text and "n=20" in text for text in annotations)


def test_save_figure_writes_the_file_and_creates_parents(
    frame: pd.DataFrame, tmp_path: Path
) -> None:
    figure = plot_target_distribution(frame["Churn"].value_counts(), "Target")
    target = tmp_path / "nested" / "figure.png"

    result = save_figure(figure, target)

    assert result == target
    assert target.is_file()
    assert target.stat().st_size > 0
