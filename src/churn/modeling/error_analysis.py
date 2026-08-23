"""Phase 9E: where the frozen system errs, described and nothing more.

The final estimate already exists and is closed. This phase asks a different
question — **not** "how do we do better", which would require a decision, but
"under what conditions does this system get it wrong", which requires only
description. The distinction is the whole reason the phase is allowed to look at
the evaluation sample at all: it has been consumed, so it can no longer support a
choice, but it can still support an account of the errors already counted.

**Nothing here can select anything.** There is no fit, no search, no calibrator,
no threshold candidate and no alternative model in this module. The threshold is
read from the frozen policy and every rate is computed at that one value.

**Denominators are the point.** A large group produces more errors than a small
one for reasons that have nothing to do with the model, so a raw count of false
positives per category is close to meaningless on its own. Every rate here is
conditioned on the population that could have produced it — false negatives on
the actual churners, false positives on the actual non-churners — and every rate
whose denominator is too small to support it is returned as ``None`` rather than
as a number that looks precise.

**No inference.** No test, no p-value, no confidence interval on a slice. With
this many descriptive cells, examined after the outcome was known, a search for
"significant" subgroups would find some whether or not anything is there. The
slice axes were fixed in advance, and what is reported is what was observed.

**No individual is persisted.** Row-level outcomes exist only inside a call.
Everything this module returns is an aggregate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: What this phase is, recorded so no reader has to infer it.
ANALYSIS_TYPE = "POST_HOC_ERROR_ANALYSIS"

#: The four outcome classes. Every row falls in exactly one.
TRUE_NEGATIVE = "TN"
FALSE_POSITIVE = "FP"
FALSE_NEGATIVE = "FN"
TRUE_POSITIVE = "TP"
OUTCOME_CLASSES: tuple[str, ...] = (TRUE_NEGATIVE, FALSE_POSITIVE, FALSE_NEGATIVE, TRUE_POSITIVE)

#: The slice axes, **fixed in this module before any Phase 9E number was read**.
#: Each one carries a justification that predates the opening of the evaluation
#: sample: Contract and tenure showed the strongest associations in the Phase 3
#: EDA, InternetService and PaymentMethod showed relevant ones, and SeniorCitizen
#: is both a described difference and a demographic axis worth auditing.
#:
#: The list is deliberately short. Sweeping all 19 features, or their
#: combinations, would turn a description into a search.
SLICE_AXES: tuple[str, ...] = (
    "Contract",
    "tenure_band",
    "InternetService",
    "PaymentMethod",
    "SeniorCitizen",
)

#: A secondary demographic audit. Included because the EDA recorded an almost
#: null association with churn, which makes it a useful control: a large
#: difference in error rates here would be surprising in a way that a difference
#: on Contract would not.
AUDIT_AXES: tuple[str, ...] = ("gender",)

#: The one interaction examined, and only as a secondary table. It is the pair
#: the EDA already crossed (`reports/figures/eda/05_contract_tenure_churn_heatmap.png`),
#: so it is not a cell chosen after seeing the errors. No three-way interaction is
#: computed: the cell count would explode and the exercise would become mining.
INTERACTION_AXES: tuple[tuple[str, str], ...] = (("Contract", "tenure_band"),)

#: Guardrails, fixed before interpretation and applied uniformly.
#:
#: A rate is reported only when its own denominator supports it, which is why
#: there are three separate minimums rather than one: a group can be large enough
#: to describe overall while holding too few churners to say anything about
#: recall. Categories are **never** merged to clear a threshold — an
#: insufficient cell is reported as insufficient.
MIN_GROUP_SIZE = 30
MIN_ACTUAL_POSITIVES = 20
MIN_ACTUAL_NEGATIVES = 20
MIN_PREDICTED_POSITIVES = 20

#: Distance-to-threshold bands, fixed before the errors were placed in them.
#: The first is narrow enough that a customer inside it would change side under a
#: trivial perturbation of the score; the last is wide enough that a customer
#: inside it would not.
MARGIN_BAND_EDGES: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10)
MARGIN_BAND_LABELS: tuple[str, ...] = (
    "|margin| < 0.025",
    "0.025 <= |margin| < 0.05",
    "0.05 <= |margin| < 0.10",
    "|margin| >= 0.10",
)

#: Below this absolute margin an error is called **marginal**: its score lies
#: within this distance of the frozen decision boundary, so the binary decision is
#: boundary-proximate under the frozen operating point.
#:
#: That is a statement about proximity to *the frozen cut*, not about proximity to
#: a 50% probability. The boundary is well below 0.5 in this project, so a
#: boundary-proximate score is nowhere near a coin flip, and calling it one would
#: misdescribe the quantity.
MARGINAL_ERROR_MARGIN = 0.05

#: Score extremes on each side of the boundary, used to ask whether the model ever
#: assigns an extreme probability and is still wrong.
#:
#: **"High confidence" is descriptive shorthand for an extreme model probability
#: under the frozen score. It is not a validated per-case confidence guarantee.**
#: The probabilities are uncalibrated by the Phase 9A decision, so an extreme
#: score is an extreme *output*, not a demonstrated likelihood — the field names
#: keep the original wording for compatibility while the report says
#: "extreme-score errors".
#:
#: Diagnostic values, chosen to be round and legible, and **not** threshold
#: candidates. Nothing in this project may use them to move the decision boundary
#: or to reopen the calibration policy.
HIGH_CONFIDENCE_FALSE_POSITIVE_MIN = 0.70
HIGH_CONFIDENCE_FALSE_NEGATIVE_MAX = 0.10


class ErrorAnalysisError(RuntimeError):
    """The post-hoc analysis cannot proceed as specified."""


def classify_outcomes(y_true: np.ndarray, y_predicted: np.ndarray) -> np.ndarray:
    """Return the outcome class of every row, as one of :data:`OUTCOME_CLASSES`.

    A partition, not a labelling exercise: the four classes are mutually
    exclusive and exhaustive by construction, which is what lets every later
    table be checked against the total.

    Args:
        y_true: Binary ground truth, 1 = churn.
        y_predicted: Binary decisions from the frozen rule.

    Returns:
        An array of outcome labels, aligned with the input.

    Raises:
        ErrorAnalysisError: If the inputs are misaligned or not binary.
    """
    labels = np.asarray(y_true).astype(int).ravel()
    predicted = np.asarray(y_predicted).astype(int).ravel()
    if labels.size != predicted.size:
        raise ErrorAnalysisError(
            f"{labels.size} labels against {predicted.size} predictions; an outcome class needs "
            "both for every row."
        )
    if not np.isin(labels, (0, 1)).all() or not np.isin(predicted, (0, 1)).all():
        raise ErrorAnalysisError("Labels and predictions must both be binary 0/1.")

    outcomes = np.empty(labels.size, dtype="<U2")
    outcomes[(labels == 0) & (predicted == 0)] = TRUE_NEGATIVE
    outcomes[(labels == 0) & (predicted == 1)] = FALSE_POSITIVE
    outcomes[(labels == 1) & (predicted == 0)] = FALSE_NEGATIVE
    outcomes[(labels == 1) & (predicted == 1)] = TRUE_POSITIVE
    return outcomes


def outcome_counts(outcomes: np.ndarray) -> dict[str, int]:
    """Return the count of each outcome class, in :data:`OUTCOME_CLASSES` order."""
    values = np.asarray(outcomes)
    return {name: int((values == name).sum()) for name in OUTCOME_CLASSES}


class ConfusionReproductionError(RuntimeError):
    """The rederived confusion matrix is not the one Phase 9D recorded."""


def verify_confusion_reproduction(
    derived: dict[str, int],
    recorded_true_negatives: int,
    recorded_false_positives: int,
    recorded_false_negatives: int,
    recorded_true_positives: int,
) -> dict[str, object]:
    """Check the rederived outcomes against the frozen Phase 9D record.

    The integrity gate of this phase. Everything below is a description of *the*
    confusion matrix that was already reported; if the rederivation disagrees,
    the analysis would be describing a different classification than the one the
    final estimate came from, and it must not proceed.

    Raises:
        ConfusionReproductionError: On any disagreement.
    """
    expected = {
        TRUE_NEGATIVE: int(recorded_true_negatives),
        FALSE_POSITIVE: int(recorded_false_positives),
        FALSE_NEGATIVE: int(recorded_false_negatives),
        TRUE_POSITIVE: int(recorded_true_positives),
    }
    problems = [
        f"{name}: derived {derived[name]} against recorded {value}"
        for name, value in expected.items()
        if derived[name] != value
    ]
    if problems:
        raise ConfusionReproductionError(
            "The rederived confusion matrix does not reproduce the Phase 9D record, so this "
            "analysis would describe a different classification than the final estimate:\n  "
            + "\n  ".join(problems)
        )

    logger.info("Confusion matrix reproduces the Phase 9D record exactly: %s", derived)
    return {
        "derived": dict(derived),
        "recorded": expected,
        "reproduced": True,
        "total": sum(derived.values()),
    }


@dataclass(frozen=True)
class ProbabilitySummary:
    """Distribution of the predicted probability within one outcome class."""

    outcome: str
    count: int
    mean: float
    median: float
    q1: float
    q3: float
    minimum: float
    maximum: float

    def as_dict(self) -> dict[str, float]:
        """Return the summary as a plain mapping."""
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "median": float(self.median),
            "q1": float(self.q1),
            "q3": float(self.q3),
            "min": float(self.minimum),
            "max": float(self.maximum),
        }


def summarise_probabilities(
    y_probability: np.ndarray,
    outcomes: np.ndarray,
) -> dict[str, ProbabilitySummary]:
    """Describe the probability distribution inside each outcome class.

    This is what separates an error that sits on the boundary from one the model
    was confident about. The quartiles matter more than the mean here: a false
    negative class with a high Q3 and a low Q1 contains both kinds.
    """
    probability = np.asarray(y_probability, dtype=float).ravel()
    values = np.asarray(outcomes)

    summaries: dict[str, ProbabilitySummary] = {}
    for name in OUTCOME_CLASSES:
        selected = probability[values == name]
        if selected.size == 0:
            summaries[name] = ProbabilitySummary(name, 0, *([float("nan")] * 6))
            continue
        summaries[name] = ProbabilitySummary(
            outcome=name,
            count=int(selected.size),
            mean=float(selected.mean()),
            median=float(np.median(selected)),
            q1=float(np.percentile(selected, 25)),
            q3=float(np.percentile(selected, 75)),
            minimum=float(selected.min()),
            maximum=float(selected.max()),
        )
    return summaries


def margins(y_probability: np.ndarray, threshold: float) -> np.ndarray:
    """Return ``probability - threshold`` for every row.

    Positive on the flagged side of the frozen cut, negative on the other. The
    threshold is the frozen one and is never re-derived here.
    """
    return np.asarray(y_probability, dtype=float).ravel() - float(threshold)


def margin_band_labels(absolute_margin: np.ndarray) -> np.ndarray:
    """Assign each row to one of :data:`MARGIN_BAND_LABELS`."""
    values = np.asarray(absolute_margin, dtype=float).ravel()
    index = np.searchsorted(np.asarray(MARGIN_BAND_EDGES[1:]), values, side="right")
    return np.asarray(MARGIN_BAND_LABELS, dtype=object)[index]


def margin_table(outcomes: np.ndarray, absolute_margin: np.ndarray) -> dict[str, dict[str, int]]:
    """Return ``{band: {outcome: count}}`` over the fixed bands.

    Read across a row to see how many of each outcome sit that far from the cut;
    read down the FP and FN columns to see how much of the error is a boundary
    effect and how much is not.
    """
    bands = margin_band_labels(absolute_margin)
    values = np.asarray(outcomes)
    return {
        band: {name: int(((bands == band) & (values == name)).sum()) for name in OUTCOME_CLASSES}
        for band in MARGIN_BAND_LABELS
    }


@dataclass(frozen=True)
class HighConfidenceErrors:
    """Errors the model made while being far from the decision boundary.

    Diagnostic only. The two cut-offs are round numbers chosen to be legible, not
    threshold candidates, and nothing in this project may use them to move the
    decision boundary or to justify recalibration.
    """

    false_positive_minimum: float
    false_negative_maximum: float
    n_false_positives: int
    n_high_confidence_false_positives: int
    n_false_negatives: int
    n_high_confidence_false_negatives: int
    diagnostic_only: bool = True
    decision_policy_candidate: bool = False

    @property
    def share_of_false_positives(self) -> float | None:
        """Share of false positives that were confident, or ``None`` if there are none."""
        if self.n_false_positives == 0:
            return None
        return self.n_high_confidence_false_positives / self.n_false_positives

    @property
    def share_of_false_negatives(self) -> float | None:
        """Share of false negatives that were confident, or ``None`` if there are none."""
        if self.n_false_negatives == 0:
            return None
        return self.n_high_confidence_false_negatives / self.n_false_negatives


def high_confidence_errors(
    y_probability: np.ndarray,
    outcomes: np.ndarray,
    false_positive_minimum: float = HIGH_CONFIDENCE_FALSE_POSITIVE_MIN,
    false_negative_maximum: float = HIGH_CONFIDENCE_FALSE_NEGATIVE_MAX,
) -> HighConfidenceErrors:
    """Count the errors made far from the boundary, on each side."""
    probability = np.asarray(y_probability, dtype=float).ravel()
    values = np.asarray(outcomes)
    false_positive = values == FALSE_POSITIVE
    false_negative = values == FALSE_NEGATIVE

    return HighConfidenceErrors(
        false_positive_minimum=float(false_positive_minimum),
        false_negative_maximum=float(false_negative_maximum),
        n_false_positives=int(false_positive.sum()),
        n_high_confidence_false_positives=int(
            (false_positive & (probability >= false_positive_minimum)).sum()
        ),
        n_false_negatives=int(false_negative.sum()),
        n_high_confidence_false_negatives=int(
            (false_negative & (probability <= false_negative_maximum)).sum()
        ),
    )


def marginal_error_counts(
    outcomes: np.ndarray,
    absolute_margin: np.ndarray,
    limit: float = MARGINAL_ERROR_MARGIN,
) -> dict[str, int]:
    """Count the errors sitting within ``limit`` of the frozen threshold."""
    values = np.asarray(outcomes)
    close = np.asarray(absolute_margin, dtype=float).ravel() < float(limit)
    return {
        "margin_limit": float(limit),
        "marginal_false_positives": int(((values == FALSE_POSITIVE) & close).sum()),
        "marginal_false_negatives": int(((values == FALSE_NEGATIVE) & close).sum()),
        "total_false_positives": int((values == FALSE_POSITIVE).sum()),
        "total_false_negatives": int((values == FALSE_NEGATIVE).sum()),
    }


@dataclass(frozen=True)
class SliceMetrics:
    """Counts and conditional rates for one group of one slice axis.

    Every rate is ``None`` when its own denominator is below the guardrail. That
    is deliberate and is the difference between a description and a number that
    merely looks like one: recall over four churners is not an estimate of
    anything, and printing ``0.75`` would invite a reader to treat it as one.
    """

    axis: str
    group: str
    n: int
    positives: int
    negatives: int
    predicted_positives: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    prevalence: float | None
    predicted_positive_rate: float | None
    precision: float | None
    recall: float | None
    specificity: float | None
    false_positive_rate: float | None
    false_negative_rate: float | None
    sufficient_for_group_rates: bool
    sufficient_for_recall: bool
    sufficient_for_specificity: bool
    sufficient_for_precision: bool

    def as_dict(self) -> dict[str, object]:
        """Return the record as a JSON-encodable mapping."""
        return {
            "group": self.group,
            "n": self.n,
            "positives": self.positives,
            "negatives": self.negatives,
            "predicted_positives": self.predicted_positives,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "prevalence": self.prevalence,
            "predicted_positive_rate": self.predicted_positive_rate,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "sufficient_for_group_rates": self.sufficient_for_group_rates,
            "sufficient_for_recall": self.sufficient_for_recall,
            "sufficient_for_specificity": self.sufficient_for_specificity,
            "sufficient_for_precision": self.sufficient_for_precision,
        }


def _rate(numerator: int, denominator: int, sufficient: bool) -> float | None:
    """Return the rate, or ``None`` when the denominator does not support it."""
    if not sufficient or denominator == 0:
        return None
    return numerator / denominator


def slice_metrics(
    axis: str,
    group: str,
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> SliceMetrics:
    """Compute counts and guarded conditional rates for one group."""
    labels = np.asarray(y_true).astype(int).ravel()
    predicted = np.asarray(y_predicted).astype(int).ravel()

    n = int(labels.size)
    positives = int(labels.sum())
    negatives = n - positives
    predicted_positives = int(predicted.sum())
    true_positives = int(((labels == 1) & (predicted == 1)).sum())
    false_positives = int(((labels == 0) & (predicted == 1)).sum())
    false_negatives = int(((labels == 1) & (predicted == 0)).sum())
    true_negatives = int(((labels == 0) & (predicted == 0)).sum())

    group_ok = n >= MIN_GROUP_SIZE
    recall_ok = group_ok and positives >= MIN_ACTUAL_POSITIVES
    specificity_ok = group_ok and negatives >= MIN_ACTUAL_NEGATIVES
    precision_ok = group_ok and predicted_positives >= MIN_PREDICTED_POSITIVES

    return SliceMetrics(
        axis=axis,
        group=group,
        n=n,
        positives=positives,
        negatives=negatives,
        predicted_positives=predicted_positives,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
        prevalence=_rate(positives, n, group_ok),
        predicted_positive_rate=_rate(predicted_positives, n, group_ok),
        precision=_rate(true_positives, predicted_positives, precision_ok),
        recall=_rate(true_positives, positives, recall_ok),
        specificity=_rate(true_negatives, negatives, specificity_ok),
        false_positive_rate=_rate(false_positives, negatives, specificity_ok),
        false_negative_rate=_rate(false_negatives, positives, recall_ok),
        sufficient_for_group_rates=group_ok,
        sufficient_for_recall=recall_ok,
        sufficient_for_specificity=specificity_ok,
        sufficient_for_precision=precision_ok,
    )


def slice_table(
    frame: pd.DataFrame,
    axis: str,
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> list[SliceMetrics]:
    """Return one :class:`SliceMetrics` per group of ``axis``, in category order.

    Group order follows the column's own categories when it is categorical and
    the sorted unique values otherwise, so the table order is a property of the
    data rather than of the result — a table sorted by error rate would put the
    reader's eye where the analyst chose.

    Raises:
        ErrorAnalysisError: If the axis is absent or the lengths disagree.
    """
    if axis not in frame.columns:
        raise ErrorAnalysisError(f"Slice axis {axis!r} is not a column of the frame.")
    labels = np.asarray(y_true).astype(int).ravel()
    predicted = np.asarray(y_predicted).astype(int).ravel()
    if not (len(frame) == labels.size == predicted.size):
        raise ErrorAnalysisError(
            f"Frame has {len(frame)} rows against {labels.size} labels and {predicted.size} "
            "predictions."
        )

    column = frame[axis]
    if isinstance(column.dtype, pd.CategoricalDtype):
        groups = [value for value in column.cat.categories]
    else:
        groups = sorted(column.dropna().unique(), key=str)

    table: list[SliceMetrics] = []
    for group in groups:
        mask = (column == group).to_numpy()
        table.append(slice_metrics(axis, str(group), labels[mask], predicted[mask]))
    return table


def interaction_table(
    frame: pd.DataFrame,
    first: str,
    second: str,
    y_true: np.ndarray,
    y_predicted: np.ndarray,
) -> list[SliceMetrics]:
    """Return metrics per cell of a two-way interaction. Secondary and descriptive.

    Only the pair the EDA already crossed is passed here. The cells are small by
    construction, so most rates will fail their guardrails and be reported as
    unavailable — which is the honest outcome, not a defect of the table.
    """
    labels = np.asarray(y_true).astype(int).ravel()
    predicted = np.asarray(y_predicted).astype(int).ravel()
    outer = slice_table(frame, first, labels, predicted)

    cells: list[SliceMetrics] = []
    for entry in outer:
        mask = (frame[first] == entry.group).to_numpy()
        for inner in slice_table(frame.loc[mask], second, labels[mask], predicted[mask]):
            cells.append(
                SliceMetrics(
                    **{
                        **inner.as_dict(),
                        "axis": f"{first} x {second}",
                        "group": f"{entry.group} | {inner.group}",
                    }
                )
            )
    return cells


def composition(
    frame: pd.DataFrame,
    axis: str,
    error_mask: np.ndarray,
    eligible_mask: np.ndarray,
) -> list[dict[str, object]]:
    """Describe which categories the errors of one class fall into.

    Two shares, and both are needed. ``share_of_errors`` says how the error set
    is made up; ``share_of_eligible`` says how the population that *could* have
    produced that error is made up. A category is over-represented among the
    errors only relative to the second, and reporting the first alone is the
    mistake this function exists to prevent — the largest category usually
    supplies the most errors simply by being largest.

    Args:
        frame: The feature frame.
        axis: Column to break down.
        error_mask: Rows in the error class (false positives, or false negatives).
        eligible_mask: Rows that could have been in it — the actual negatives for
            false positives, the actual positives for false negatives.
    """
    column = frame[axis]
    errors = int(np.asarray(error_mask).sum())
    eligible = int(np.asarray(eligible_mask).sum())

    if isinstance(column.dtype, pd.CategoricalDtype):
        groups = list(column.cat.categories)
    else:
        groups = sorted(column.dropna().unique(), key=str)

    rows: list[dict[str, object]] = []
    for group in groups:
        mask = (column == group).to_numpy()
        in_errors = int((mask & np.asarray(error_mask)).sum())
        in_eligible = int((mask & np.asarray(eligible_mask)).sum())
        rows.append(
            {
                "group": str(group),
                "errors": in_errors,
                "share_of_errors": in_errors / errors if errors else None,
                "eligible": in_eligible,
                "share_of_eligible": in_eligible / eligible if eligible else None,
            }
        )
    return rows


def outcome_margin_summary(
    outcomes: np.ndarray,
    absolute_margin: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Describe the distance from the threshold within each outcome class."""
    values = np.asarray(outcomes)
    distance = np.asarray(absolute_margin, dtype=float).ravel()

    summary: dict[str, dict[str, float]] = {}
    for name in OUTCOME_CLASSES:
        selected = distance[values == name]
        if selected.size == 0:
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": int(selected.size),
            "mean": float(selected.mean()),
            "median": float(np.median(selected)),
            "q1": float(np.percentile(selected, 25)),
            "q3": float(np.percentile(selected, 75)),
            "max": float(selected.max()),
        }
    return summary


def insufficient_cells(tables: Sequence[Sequence[SliceMetrics]]) -> list[dict[str, object]]:
    """List every group whose rates could not all be estimated, and why.

    Reported rather than hidden. A table with silent gaps invites the reader to
    assume the missing cells were unremarkable.
    """
    flagged: list[dict[str, object]] = []
    for table in tables:
        for entry in table:
            reasons = []
            if not entry.sufficient_for_group_rates:
                reasons.append(f"n={entry.n} < {MIN_GROUP_SIZE}")
            if not entry.sufficient_for_recall:
                reasons.append(f"positives={entry.positives} < {MIN_ACTUAL_POSITIVES}")
            if not entry.sufficient_for_specificity:
                reasons.append(f"negatives={entry.negatives} < {MIN_ACTUAL_NEGATIVES}")
            if not entry.sufficient_for_precision:
                reasons.append(
                    f"predicted_positives={entry.predicted_positives} < {MIN_PREDICTED_POSITIVES}"
                )
            if reasons:
                flagged.append({"axis": entry.axis, "group": entry.group, "reasons": reasons})
    return flagged
