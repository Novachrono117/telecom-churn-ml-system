"""Structural and data-quality inspection of the raw dataset.

Every function here only *observes* the data. Nothing is cleaned, imputed,
converted or dropped: Phase 2 identifies problems, later phases decide what to
do about them.

Two aligned views of the same file are used throughout (see
:mod:`churn.data.loader`): the text view exposes blanks verbatim, the typed view
exposes what pandas inferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns with at most this many distinct values get their domain enumerated.
MAX_ENUMERATED_CATEGORIES = 12

#: Above this share of distinct values a column is reported as high-cardinality.
HIGH_CARDINALITY_RATIO = 0.5

#: Substrings that historically indicate post-outcome or target-derived fields.
#: This is a *name-based heuristic*, not proof; every hit needs human review.
LEAKAGE_NAME_HINTS = (
    "churn",
    "score",
    "reason",
    "label",
    "cltv",
    "lifetime",
    "status",
    "exit",
    "cancel",
    "termination",
)


@dataclass(frozen=True)
class ColumnProfile:
    """Observed facts about a single column."""

    name: str
    dtype: str
    missing_count: int
    blank_count: int
    whitespace_count: int
    unique_count: int
    categories: tuple[str, ...] | None
    is_numeric_coercible: bool
    non_numeric_examples: tuple[str, ...]
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class DatasetInspection:
    """Aggregated result of the structural inspection."""

    n_rows: int
    n_columns: int
    columns: tuple[str, ...]
    profiles: tuple[ColumnProfile, ...]
    duplicate_rows: int
    identifier_candidates: tuple[str, ...]
    constant_columns: tuple[str, ...]
    high_cardinality_columns: tuple[str, ...]
    target_counts: dict[str, int] = field(default_factory=dict)


def _blank_counts(text_series: pd.Series) -> tuple[int, int]:
    """Return ``(empty_string_count, whitespace_only_count)`` for a text column."""
    stripped = text_series.str.strip()
    empty = int((text_series == "").sum())
    whitespace_only = int(((stripped == "") & (text_series != "")).sum())
    return empty, whitespace_only


def _numeric_view(text_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a text column into its numeric interpretation and its offenders.

    Blank cells are excluded from both sides: they are a missingness problem,
    not a "not a number" problem.
    """
    non_blank = text_series[text_series.str.strip() != ""]
    numeric = pd.to_numeric(non_blank.str.strip(), errors="coerce")
    offenders = non_blank[numeric.isna()]
    return numeric, offenders


def profile_column(name: str, text_series: pd.Series, typed_series: pd.Series) -> ColumnProfile:
    """Build the observed profile of one column.

    Args:
        name: Column name.
        text_series: The column as read verbatim (text view).
        typed_series: The same column as read with dtype inference.

    Returns:
        The corresponding :class:`ColumnProfile`.
    """
    blank_count, whitespace_count = _blank_counts(text_series)
    distinct = text_series.unique()
    numeric, offenders = _numeric_view(text_series)
    is_numeric = len(offenders) == 0 and len(numeric) > 0

    return ColumnProfile(
        name=name,
        dtype=str(typed_series.dtype),
        missing_count=int(typed_series.isna().sum()),
        blank_count=blank_count,
        whitespace_count=whitespace_count,
        unique_count=int(len(distinct)),
        categories=tuple(sorted(map(str, distinct)))
        if len(distinct) <= MAX_ENUMERATED_CATEGORIES
        else None,
        is_numeric_coercible=is_numeric,
        non_numeric_examples=tuple(sorted(set(offenders.tolist()))[:5]),
        minimum=float(numeric.min()) if is_numeric else None,
        maximum=float(numeric.max()) if is_numeric else None,
    )


def target_distribution(text_frame: pd.DataFrame, target_column: str) -> dict[str, int]:
    """Return the absolute frequency of each observed target value.

    Args:
        text_frame: Dataset in its text view.
        target_column: Name of the target column.

    Returns:
        Mapping ``value -> count``, ordered from most to least frequent. Empty
        if the column is absent.
    """
    if target_column not in text_frame.columns:
        logger.warning("Target column %r not found in the dataset", target_column)
        return {}
    counts = text_frame[target_column].value_counts()
    return {str(value): int(count) for value, count in counts.items()}


@dataclass(frozen=True)
class AssumptionCheck:
    """Outcome of confronting a documented assumption with the real file."""

    description: str
    expected: str
    observed: str
    confirmed: bool


def check_target_assumptions(
    inspection: DatasetInspection,
    target_column: str,
    positive_label: str,
    negative_label: str,
) -> tuple[AssumptionCheck, ...]:
    """Confront the configured target assumptions with the observed data.

    The dataset is the source of truth: this function reports mismatches, it
    never rewrites the configuration.
    """
    observed_labels = tuple(sorted(inspection.target_counts))
    column_present = target_column in inspection.columns

    return (
        AssumptionCheck(
            description="Target column name",
            expected=target_column,
            observed=target_column if column_present else "absent",
            confirmed=column_present,
        ),
        AssumptionCheck(
            description="Target has exactly two classes",
            expected=f"{{{negative_label}, {positive_label}}}",
            observed="{" + ", ".join(observed_labels) + "}" if observed_labels else "none",
            confirmed=set(observed_labels) == {positive_label, negative_label},
        ),
        AssumptionCheck(
            description="Positive class label",
            expected=positive_label,
            observed=positive_label if positive_label in observed_labels else "not found",
            confirmed=positive_label in observed_labels,
        ),
        AssumptionCheck(
            description="Negative class label",
            expected=negative_label,
            observed=negative_label if negative_label in observed_labels else "not found",
            confirmed=negative_label in observed_labels,
        ),
    )


def identifier_candidates(profiles: tuple[ColumnProfile, ...], n_rows: int) -> tuple[str, ...]:
    """Return columns whose values are unique across every row."""
    return tuple(p.name for p in profiles if p.unique_count == n_rows and n_rows > 0)


def constant_columns(profiles: tuple[ColumnProfile, ...]) -> tuple[str, ...]:
    """Return columns carrying a single distinct value (no information)."""
    return tuple(p.name for p in profiles if p.unique_count <= 1)


def high_cardinality_columns(
    profiles: tuple[ColumnProfile, ...],
    n_rows: int,
    ratio: float = HIGH_CARDINALITY_RATIO,
) -> tuple[str, ...]:
    """Return non-numeric columns whose distinct-value share exceeds ``ratio``."""
    if n_rows == 0:
        return ()
    return tuple(
        p.name for p in profiles if not p.is_numeric_coercible and p.unique_count / n_rows > ratio
    )


def leakage_suspects(columns: tuple[str, ...], target_column: str) -> tuple[str, ...]:
    """Flag column names that *may* encode the outcome.

    Purely lexical: it catches renamed or derived target fields, and it cannot
    detect leakage that hides behind an innocent name. The target itself is
    excluded.
    """
    return tuple(
        column
        for column in columns
        if column != target_column and any(hint in column.lower() for hint in LEAKAGE_NAME_HINTS)
    )


def inspect_dataset(
    text_frame: pd.DataFrame,
    typed_frame: pd.DataFrame,
    target_column: str,
) -> DatasetInspection:
    """Run the full structural inspection over both views of the dataset.

    Args:
        text_frame: Dataset in its text view.
        typed_frame: Dataset with pandas dtype inference.
        target_column: Configured target column name.

    Returns:
        The aggregated :class:`DatasetInspection`.

    Raises:
        ValueError: If the two views do not describe the same table.
    """
    if list(text_frame.columns) != list(typed_frame.columns) or len(text_frame) != len(typed_frame):
        raise ValueError("Text and typed views must come from the same file")

    n_rows = len(text_frame)
    profiles = tuple(
        profile_column(name, text_frame[name], typed_frame[name]) for name in text_frame.columns
    )

    return DatasetInspection(
        n_rows=n_rows,
        n_columns=len(text_frame.columns),
        columns=tuple(text_frame.columns),
        profiles=profiles,
        duplicate_rows=int(text_frame.duplicated().sum()),
        identifier_candidates=identifier_candidates(profiles, n_rows),
        constant_columns=constant_columns(profiles),
        high_cardinality_columns=high_cardinality_columns(profiles, n_rows),
        target_counts=target_distribution(text_frame, target_column),
    )
