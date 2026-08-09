"""Markdown rendering of the Data Understanding artifacts.

Only observed facts are rendered. Where a statement comes from the dataset
documentation rather than from the file itself, the artifact says so.
"""

from __future__ import annotations

from churn.data.descriptions import describe
from churn.data.inspection import (
    AssumptionCheck,
    ColumnProfile,
    DatasetInspection,
    leakage_suspects,
)
from churn.data.provenance import FileProvenance

PROVENANCE_START = "<!-- provenance:start -->"
PROVENANCE_END = "<!-- provenance:end -->"


def _escape(value: str) -> str:
    """Make a value safe to place inside a Markdown table cell."""
    return value.replace("|", "\\|").replace("\n", " ")


def _quote_blanks(value: str) -> str:
    """Render whitespace-only or empty values visibly."""
    if value == "":
        return "`` (empty) ``"
    if value.strip() == "":
        return f"`{'␣' * len(value)}` (whitespace)"
    return f"`{_escape(value)}`"


def infer_role(profile: ColumnProfile, target_column: str, identifiers: tuple[str, ...]) -> str:
    """Return the likely modeling role of a column, based on observed structure."""
    if profile.name == target_column:
        return "target"
    if profile.name in identifiers:
        return "identifier"
    if profile.is_numeric_coercible:
        return "numeric feature"
    return "categorical feature"


def _domain(profile: ColumnProfile) -> str:
    if profile.is_numeric_coercible and profile.minimum is not None:
        return f"[{profile.minimum:g}, {profile.maximum:g}]"
    if profile.categories is not None:
        return ", ".join(_quote_blanks(value) for value in profile.categories)
    return f"{profile.unique_count} distinct values"


def _notes(profile: ColumnProfile) -> str:
    notes: list[str] = []
    if profile.whitespace_count:
        notes.append(f"{profile.whitespace_count} whitespace-only cell(s)")
    if profile.blank_count:
        notes.append(f"{profile.blank_count} empty string(s)")
    if not profile.is_numeric_coercible and profile.categories is None:
        notes.append("non-numeric with too many distinct values to enumerate")
    if profile.unique_count <= 1:
        notes.append("constant column")
    return "; ".join(notes) or "—"


def render_data_dictionary(
    inspection: DatasetInspection,
    target_column: str,
    provenance: FileProvenance,
) -> str:
    """Render the Data Dictionary from the columns actually found in the file."""
    lines = [
        "# Data Dictionary — Telco Customer Churn",
        "",
        f"Generated from `{provenance.filename}` (SHA-256 `{provenance.sha256}`).",
        f"{inspection.n_rows} rows x {inspection.n_columns} columns.",
        "",
        "`dtype` and `domain / range` are **observed in the file**. `description` comes",
        "from the dataset source documentation and is not inferred from the data.",
        "",
        "| # | Column | dtype | Likely role | Description (source doc) | Domain / range |"
        " Missing | Blank | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, profile in enumerate(inspection.profiles, start=1):
        blanks = profile.blank_count + profile.whitespace_count
        lines.append(
            f"| {index} | `{profile.name}` | `{profile.dtype}` | "
            f"{infer_role(profile, target_column, inspection.identifier_candidates)} | "
            f"{_escape(describe(profile.name))} | {_domain(profile)} | "
            f"{profile.missing_count} | {blanks} | {_notes(profile)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _findings(inspection: DatasetInspection, target_column: str) -> list[str]:
    total = sum(inspection.target_counts.values())
    lines = [
        f"- The file contains **{inspection.n_rows} rows** and **{inspection.n_columns} columns**.",
        f"- Column order: {', '.join(f'`{c}`' for c in inspection.columns)}.",
        f"- Complete duplicate rows: **{inspection.duplicate_rows}**.",
        "- Identifier candidates (unique in every row): "
        + (
            ", ".join(f"`{c}`" for c in inspection.identifier_candidates)
            if inspection.identifier_candidates
            else "none"
        )
        + ".",
        "- Constant columns: "
        + (
            ", ".join(f"`{c}`" for c in inspection.constant_columns)
            if inspection.constant_columns
            else "none"
        )
        + ".",
        "- High-cardinality non-numeric columns: "
        + (
            ", ".join(f"`{c}`" for c in inspection.high_cardinality_columns)
            if inspection.high_cardinality_columns
            else "none"
        )
        + ".",
    ]
    if inspection.target_counts:
        lines.append(f"- Target `{target_column}` distribution:")
        for label, count in inspection.target_counts.items():
            share = 100 * count / total if total else 0.0
            lines.append(f"  - `{label}`: {count} ({share:.2f}%)")
    else:
        lines.append(f"- Target `{target_column}` was **not found** in the file.")
    return lines


def _quality_issues(inspection: DatasetInspection) -> list[str]:
    issues: list[str] = []
    for profile in inspection.profiles:
        if profile.missing_count:
            issues.append(
                f"- `{profile.name}`: {profile.missing_count} value(s) read as missing by pandas."
            )
        if profile.whitespace_count:
            issues.append(
                f"- `{profile.name}`: {profile.whitespace_count} whitespace-only cell(s) — "
                "silently valid as text, invalid as a number."
            )
        if profile.blank_count:
            issues.append(f"- `{profile.name}`: {profile.blank_count} empty string(s).")
        # A free-text identifier being non-numeric is expected, not a defect, so
        # only unlabeled high-cardinality columns are reported here.
        if (
            not profile.is_numeric_coercible
            and profile.non_numeric_examples
            and profile.categories is None
            and profile.name not in inspection.identifier_candidates
        ):
            examples = ", ".join(_quote_blanks(v) for v in profile.non_numeric_examples)
            issues.append(f"- `{profile.name}`: not numerically coercible; examples: {examples}.")
    if inspection.duplicate_rows:
        issues.append(f"- {inspection.duplicate_rows} fully duplicated row(s).")
    if inspection.constant_columns:
        issues.append(
            "- Constant column(s) carrying no signal: "
            + ", ".join(f"`{c}`" for c in inspection.constant_columns)
            + "."
        )
    return issues or ["- No structural quality problem was detected by these checks."]


def _leakage_review(inspection: DatasetInspection, target_column: str) -> list[str]:
    suspects = leakage_suspects(inspection.columns, target_column)
    lines = [
        "Name-based screening only. It cannot prove absence of leakage: a field",
        "recorded *after* the churn decision can carry a perfectly innocent name.",
        "",
    ]
    if suspects:
        lines.append("Columns whose name suggests outcome information:")
        lines.extend(f"- `{c}` — requires manual review before modeling." for c in suspects)
    else:
        lines.append(
            "- No column name matches the outcome-related patterns screened for "
            "(besides the target itself)."
        )
    lines.extend(
        [
            "",
            "- Identifier column(s) must be excluded from the feature matrix: they are",
            "  unique per row and can only memorize.",
            "- Charge columns are cumulative by nature and must be re-examined in Phase 3",
            "  for consistency with `tenure` before being trusted as features.",
        ]
    )
    return lines


DEFERRED_DECISIONS = [
    "- How to handle the non-numeric `TotalCharges` cells (drop, impute, or derive):"
    " belongs to Phase 4 and must be decided **after** the train/test split.",
    "- Whether `TotalCharges` becomes numeric at all, and with which strategy.",
    "- Encoding of categorical variables, scaling of numeric variables.",
    "- Treatment of the identifier column beyond exclusion from features.",
    "- Any imputation rule: it must be fitted on training data only.",
    "- Class-imbalance handling (weights, resampling, threshold): Phases 7 and 9.",
    "- Feature engineering hypotheses: Phase 6.",
]


def render_report(
    inspection: DatasetInspection,
    assumptions: tuple[AssumptionCheck, ...],
    provenance: FileProvenance,
    target_column: str,
    inspected_on: str,
) -> str:
    """Render the Data Understanding report."""
    lines = [
        "# Data Understanding Report — Telco Customer Churn",
        "",
        f"- Inspected on: {inspected_on}",
        f"- File: `{provenance.filename}` ({provenance.size_bytes} bytes)",
        f"- SHA-256: `{provenance.sha256}`",
        "",
        "This report is generated by `scripts/inspect_raw_data.py`. No cleaning,",
        "conversion, split or modeling was performed to produce it.",
        "",
        "## Assumption verification",
        "",
        "| Assumption | Expected | Observed | Status |",
        "| --- | --- | --- | --- |",
    ]
    for check in assumptions:
        status = "confirmed" if check.confirmed else "**CONTRADICTED**"
        lines.append(
            f"| {check.description} | `{check.expected}` | `{check.observed}` | {status} |"
        )

    lines += ["", "## Findings", ""]
    lines += _findings(inspection, target_column)
    lines += ["", "## Data quality issues", ""]
    lines += _quality_issues(inspection)
    lines += ["", "## Leakage review", ""]
    lines += _leakage_review(inspection, target_column)
    lines += ["", "## Decisions deferred", ""]
    lines += DEFERRED_DECISIONS
    lines.append("")
    return "\n".join(lines)


def render_provenance_block(
    provenance: FileProvenance,
    source_url: str,
    dataset_id: str,
    inspected_on: str,
) -> str:
    """Render the provenance table injected into `data/README.md`."""
    return "\n".join(
        [
            PROVENANCE_START,
            "| Field | Value |",
            "| --- | --- |",
            f"| File name | `{provenance.filename}` |",
            f"| Size (bytes) | {provenance.size_bytes} |",
            f"| SHA-256 | `{provenance.sha256}` |",
            f"| Source | {source_url} |",
            f"| Dataset identifier | `{dataset_id}` |",
            f"| Recorded on | {inspected_on} |",
            PROVENANCE_END,
        ]
    )


def replace_provenance_section(document: str, block: str) -> str:
    """Swap the marked provenance section of a document for ``block``.

    Args:
        document: Current file content.
        block: Replacement block, markers included.

    Returns:
        The updated document.

    Raises:
        ValueError: If the provenance markers are missing or out of order.
    """
    start = document.find(PROVENANCE_START)
    end = document.find(PROVENANCE_END)
    if start == -1 or end == -1 or end < start:
        raise ValueError("Provenance markers not found in the document")
    return document[:start] + block + document[end + len(PROVENANCE_END) :]
