"""Binning and summarising — the arithmetic every drift number is built on.

Three decisions here determine whether a drift number means anything at all.

**Bins are computed once, from the reference, and never again.** A histogram
recomputed per production window would compare two different partitions of the real
line, and the resulting "drift" would partly measure the binning. The edges live in
the reference profile and are read from it.

**Bins are total: they extend to ±∞.** The interior edges come from reference
quantiles, and the outermost bins are ``(-inf, e₁]`` and ``(e_{k-1}, +inf)``. Every
finite value therefore falls in exactly one bin, so no mass is ever lost or
silently clipped into a neighbour. Values outside the reference range are still
worth knowing about — they are reported separately as
``out_of_reference_range_rate``, computed against the reference min/max, so that
"unseen territory" is a named signal instead of a distortion hidden inside the
first and last bins.

**Edges are left-open, right-closed.** A value equal to an edge belongs to the bin
below it, resolved by :func:`numpy.searchsorted` with ``side="left"``. Stating this
matters because ``tenure`` and ``SeniorCitizen``-like features are lumpy: with
duplicate quantiles, which side of an edge a value lands on decides whole
percentage points of a histogram.

Duplicate quantile edges are deduplicated rather than kept. A feature whose
distribution is concentrated — ``tenure`` has a large spike at its minimum — yields
repeated quantiles, and a zero-width bin can never receive a value while still
contributing a term to a PSI sum. Dropping it leaves fewer, honest bins.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Quantiles recorded for every numeric feature and for the model score.
REFERENCE_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

#: Number of histogram bins requested from the reference. Deciles: ten bins from
#: nine interior edges, before any duplicate edges are removed.
DEFAULT_N_BINS = 10

#: The bucket every category absent from the reference is counted into. A literal
#: string rather than ``None`` so it survives JSON, and one that cannot collide
#: with a real Telco level.
UNSEEN_LEVEL = "__UNSEEN__"


@dataclass(frozen=True)
class NumericSummary:
    """Order statistics of one numeric feature over one population."""

    count: int
    mean: float
    std: float
    minimum: float
    q01: float
    q05: float
    q25: float
    median: float
    q75: float
    q95: float
    q99: float
    maximum: float

    def as_record(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "q01": self.q01,
            "q05": self.q05,
            "q25": self.q25,
            "median": self.median,
            "q75": self.q75,
            "q95": self.q95,
            "q99": self.q99,
            "max": self.maximum,
        }


def _as_float_array(values: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(pd.Series(values).to_numpy(), dtype=float)
    if array.size == 0:
        raise ValueError("A numeric summary needs at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Non-finite value in a numeric feature. Monitoring observes records that "
            "already passed the frozen feature contract, so this means the caller "
            "handed it something the pipeline never saw."
        )
    return array


def summarise_numeric(values: Sequence[float] | pd.Series | np.ndarray) -> NumericSummary:
    """Return the recorded order statistics of a numeric feature.

    ``std`` is the population standard deviation (``ddof=0``): the reference is the
    whole frozen training pool, not a sample drawn from something larger, and the
    number is used as a scale for expressing shifts, never in an inference.
    """
    array = _as_float_array(values)
    q01, q05, q25, median, q75, q95, q99 = np.quantile(array, REFERENCE_QUANTILES)
    return NumericSummary(
        count=int(array.size),
        mean=float(array.mean()),
        std=float(array.std(ddof=0)),
        minimum=float(array.min()),
        q01=float(q01),
        q05=float(q05),
        q25=float(q25),
        median=float(median),
        q75=float(q75),
        q95=float(q95),
        q99=float(q99),
        maximum=float(array.max()),
    )


def quantile_bin_edges(
    values: Sequence[float] | pd.Series | np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[float, ...]:
    """Return the interior bin edges for a reference population.

    Args:
        values: The reference values. Computed once, at profile-build time.
        n_bins: Requested bin count. ``n_bins - 1`` interior edges are taken from
            evenly spaced quantiles.

    Returns:
        Strictly increasing interior edges, possibly fewer than ``n_bins - 1``
        after duplicates are removed. May be empty for a constant feature, which
        yields a single bin covering the whole line — correct, if uninformative.
    """
    if n_bins < 2:
        raise ValueError("At least two bins are required.")
    array = _as_float_array(values)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(array, quantiles)
    unique = np.unique(edges)
    if unique.size < edges.size:
        logger.debug(
            "Dropped %d duplicate quantile edge(s): the feature is concentrated.",
            edges.size - unique.size,
        )
    return tuple(float(edge) for edge in unique)


def bin_counts(
    values: Sequence[float] | pd.Series | np.ndarray,
    edges: Sequence[float],
) -> tuple[int, ...]:
    """Return counts per bin for fixed interior ``edges``.

    The partition is ``(-inf, e₁], (e₁, e₂], …, (e_{k-1}, +inf)`` — ``len(edges) + 1``
    bins, jointly exhaustive, so the counts always sum to the number of values.
    """
    array = _as_float_array(values)
    boundaries = np.asarray(edges, dtype=float)
    indices = np.searchsorted(boundaries, array, side="left")
    counts = np.bincount(indices, minlength=len(boundaries) + 1)
    return tuple(int(count) for count in counts)


def out_of_range_rate(
    values: Sequence[float] | pd.Series | np.ndarray,
    minimum: float,
    maximum: float,
) -> float:
    """Return the share of values outside the reference ``[minimum, maximum]``.

    Deliberately separate from the histogram. The outermost bins are unbounded, so
    an extreme value does not distort them; whether the population is straying
    beyond anything the model was fitted on is a different question, and it gets
    its own number.
    """
    array = _as_float_array(values)
    outside = (array < float(minimum)) | (array > float(maximum))
    return float(outside.mean())


def mean_shift_in_reference_sd(actual_mean: float, reference: NumericSummary) -> float:
    """Return ``(actual_mean - reference_mean) / reference_std``.

    Signed, and expressed in reference standard deviations so that a shift in
    ``tenure`` and a shift in ``MonthlyCharges`` are on a comparable scale. Returns
    ``0.0`` when the reference is constant: there is no scale to divide by, and
    inventing one would manufacture a large number out of a degenerate feature.
    """
    if reference.std <= 0.0:
        return 0.0
    return float((float(actual_mean) - reference.mean) / reference.std)


def level_counts(values: Sequence[object] | pd.Series, known: Sequence[str]) -> dict[str, int]:
    """Return counts per known level plus :data:`UNSEEN_LEVEL`.

    Every value that is not a known level is counted into ``__UNSEEN__`` — the
    frozen encoder does the same thing with ``handle_unknown="ignore"``, and
    monitoring exists to make that visible, not to reject it.

    The individual unseen *values* are never returned. Their count is what a drift
    signal needs; the strings themselves are customer-supplied text with unbounded
    cardinality, and this package does not persist them.
    """
    series = pd.Series(list(values), dtype=object)
    known_levels = list(known)
    counts = {level: 0 for level in known_levels}
    counts[UNSEEN_LEVEL] = 0

    observed = series.astype("string")
    for level in known_levels:
        counts[level] = int((observed == level).sum())
    counts[UNSEEN_LEVEL] = int(len(series) - sum(counts[level] for level in known_levels))
    return counts


def n_distinct_unseen(values: Sequence[object] | pd.Series, known: Sequence[str]) -> int:
    """Return how many *distinct* unrecognised values appeared.

    A cardinality signal without the payload: "eleven new contract terms" is
    actionable, and the eleven strings are not needed to act on it.
    """
    series = pd.Series(list(values), dtype=object).astype("string")
    unseen = series[~series.isin(list(known))]
    return int(unseen.nunique(dropna=False))


def frequencies(counts: dict[str, int]) -> dict[str, float]:
    """Return the normalised distribution of a count table.

    An empty table returns zeros rather than raising: a window with no records is a
    real state, and it should report zeros instead of failing the whole report.
    """
    total = sum(counts.values())
    if total == 0:
        return {level: 0.0 for level in counts}
    return {level: count / total for level, count in counts.items()}


__all__ = [
    "DEFAULT_N_BINS",
    "REFERENCE_QUANTILES",
    "UNSEEN_LEVEL",
    "NumericSummary",
    "bin_counts",
    "frequencies",
    "level_counts",
    "mean_shift_in_reference_sd",
    "n_distinct_unseen",
    "out_of_range_rate",
    "quantile_bin_edges",
    "summarise_numeric",
]
