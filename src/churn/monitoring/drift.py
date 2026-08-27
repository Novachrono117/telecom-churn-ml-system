"""Drift distances: PSI and TVD, written out so they can be audited.

Both are **descriptive distances between two histograms**. Neither is a hypothesis
test. There is no null distribution here, no p-value, no significance level, and no
sampling assumption. This module can say that a window stopped resembling the
reference; it cannot say that the difference is "significant", and it certainly
cannot say the model degraded — that is a claim about labels, and this phase has
none.

Population Stability Index
--------------------------
::

    PSI = Σ_i (actual_i - expected_i) * ln(actual_i / expected_i)

over the fixed reference bins, with ``actual_i`` and ``expected_i`` the *shares* of
each population in bin ``i``.

**Zeros.** The formula is undefined when a bin is empty on either side, and empty
bins are normal — a decile bin can easily receive nothing in a small window. Every
share is therefore floored at :data:`EPSILON` before the logarithm, and the choice
is stated rather than hidden: with ``ε = 1e-6``, one empty bin against a decile of
reference mass contributes ``(1e-6 - 0.1) * ln(1e-6 / 0.1) ≈ 1.15``, which is
already far past any cutoff in the operational policy. That is intended — a bin
that emptied out *should* dominate — but it means a PSI computed over a window with
several empty bins is driven by the epsilon, not by the data. This is exactly why
:mod:`churn.monitoring.settings` refuses to report drift below a minimum window
size.

PSI is symmetric in the sense that swapping the two populations leaves it
unchanged, and it is zero only when the two histograms are identical.

Total Variation Distance
------------------------
::

    TVD = 0.5 * Σ_c |p_actual(c) - p_reference(c)|

over the known levels plus the ``__UNSEEN__`` bucket. Bounded in ``[0, 1]``: zero
when the two distributions agree exactly, one when their supports are disjoint. No
epsilon is needed — the formula is defined at zero — and no logarithm means no
sensitivity to empty cells, which is why it is preferred here over a
Jensen-Shannon or chi-square style quantity for categorical features.

TVD is a distance, not a probability. "TVD = 0.3" does not mean a 30 % chance of
anything.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Floor applied to a histogram share before it reaches a logarithm.
#:
#: Small enough that a populated bin is unaffected to well past the precision any
#: cutoff cares about; large enough that an empty bin yields a finite, large
#: contribution instead of an infinity. Recorded in the machine-readable record so
#: a reader can reproduce every PSI in this project exactly.
EPSILON = 1e-6


def _shares(counts: Sequence[int]) -> np.ndarray:
    array = np.asarray(counts, dtype=float)
    if np.any(array < 0):
        raise ValueError("Histogram counts cannot be negative.")
    total = array.sum()
    if total <= 0:
        raise ValueError("A distribution needs at least one observation.")
    return array / total


def population_stability_index(
    actual_counts: Sequence[int],
    reference_counts: Sequence[int],
    epsilon: float = EPSILON,
) -> float:
    """Return the PSI between two histograms over identical bins.

    Args:
        actual_counts: Counts per bin in the window being judged.
        reference_counts: Counts per bin in the reference population.
        epsilon: Floor applied to both share vectors before the logarithm.

    Returns:
        A non-negative float. ``0.0`` exactly when the two share vectors agree.

    Raises:
        ValueError: If the bin counts differ in length — comparing histograms over
            different partitions would produce a number that measures the binning —
            or if either population is empty.
    """
    if len(actual_counts) != len(reference_counts):
        raise ValueError(
            f"PSI needs the same bins on both sides: got {len(actual_counts)} and "
            f"{len(reference_counts)}. The reference bins are fixed at profile-build "
            "time and must not be recomputed per window."
        )
    actual = np.maximum(_shares(actual_counts), epsilon)
    reference = np.maximum(_shares(reference_counts), epsilon)
    return float(np.sum((actual - reference) * np.log(actual / reference)))


def total_variation_distance(
    actual: Mapping[str, float],
    reference: Mapping[str, float],
) -> float:
    """Return the TVD between two categorical distributions.

    Levels present in only one mapping are treated as having share zero in the
    other, so the ``__UNSEEN__`` bucket participates like any other level and a
    window made entirely of new categories scores ``1.0``.

    Returns:
        A float in ``[0, 1]``.
    """
    levels = set(actual) | set(reference)
    distance = 0.5 * sum(
        abs(float(actual.get(level, 0.0)) - float(reference.get(level, 0.0))) for level in levels
    )
    # Guard the bound against floating-point accumulation rather than trusting it.
    return float(min(1.0, max(0.0, distance)))


def delta(actual: float, reference: float) -> float:
    """Return the signed change ``actual - reference``.

    Signed on purpose. A predicted-positive rate that halved and one that doubled
    are different operational situations, and an absolute value would erase the
    distinction before anyone saw it.
    """
    return float(actual) - float(reference)


__all__ = [
    "EPSILON",
    "delta",
    "population_stability_index",
    "total_variation_distance",
]
