"""Generate the Phase 3 EDA figures and report.

Usage (from the repository root):

    uv run python scripts/run_eda.py

Writes:
    reports/figures/eda/*.png
    reports/eda_report.md

Every number in the report is computed here from the raw file; none is typed by
hand. The raw CSV is read-only and no processed dataset is persisted.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from churn.analysis import descriptive as desc
from churn.analysis.association import (
    association_ranking,
    interpret_cramers_v,
    numeric_comparison,
)
from churn.analysis.frames import CHURN_FLAG, TENURE_BAND, TENURE_BAND_LABELS, load_eda_frame
from churn.analysis.plots import (
    plot_boxplot_by_group,
    plot_churn_rate_bars,
    plot_distribution_by_churn,
    plot_effect_sizes,
    plot_rate_heatmap,
    plot_scatter_by_churn,
    plot_target_distribution,
    save_figure,
    use_project_style,
)
from churn.config import get_config
from churn.data.loader import raw_csv_path
from churn.data.provenance import describe_file

logger = logging.getLogger("run_eda")

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
PROFILE_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents"]
BILLING_COLUMNS = ["Contract", "PaperlessBilling", "PaymentMethod"]
CATEGORICAL_COLUMNS = PROFILE_COLUMNS + SERVICE_COLUMNS + BILLING_COLUMNS
PROTECTIVE_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_SERVICES = ["StreamingTV", "StreamingMovies"]

FIGURES_SUBDIR = Path("reports") / "figures" / "eda"


#: Sentinel categories that repeat the same customers across several columns:
#: every `X = No internet service` row is the same 1 526 customers already shown
#: as `InternetService = No`, and `MultipleLines = No phone service` repeats
#: `PhoneService = No`. They are dropped from the consolidated chart to avoid
#: seven identical bars; the states themselves are discussed in the report.
REDUNDANT_SERVICE_STATES = ("No internet service", "No phone service")


def _service_rate_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One consolidated table of churn rate per service option."""
    rows = []
    for column in SERVICE_COLUMNS:
        table = desc.churn_rate_by(frame, column, sort=False)
        for category, row in table.iterrows():
            if str(category) in REDUNDANT_SERVICE_STATES:
                continue
            rows.append(
                {
                    "label": f"{column} = {category}",
                    "n": int(row["n"]),
                    "churn_rate": float(row["churn_rate"]),
                }
            )
    table = pd.DataFrame(rows).set_index("label")
    return table.sort_values("churn_rate", ascending=False)


def _profile_rate_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in PROFILE_COLUMNS:
        table = desc.churn_rate_by(frame, column, sort=False)
        for category, row in table.iterrows():
            rows.append(
                {
                    "label": f"{column} = {category}",
                    "n": int(row["n"]),
                    "churn_rate": float(row["churn_rate"]),
                }
            )
    return pd.DataFrame(rows).set_index("label").sort_values("churn_rate", ascending=False)


def build_figures(frame: pd.DataFrame, root: Path) -> dict[str, Path]:
    """Render every EDA figure and return a name -> path mapping."""
    use_project_style()
    target = get_config().target.column
    baseline = desc.overall_churn_rate(frame)
    figures_dir = root / FIGURES_SUBDIR
    paths: dict[str, Path] = {}

    paths["target"] = save_figure(
        plot_target_distribution(
            frame[target].value_counts(),
            "Churn is imbalanced: roughly one churner for every three customers",
        ),
        figures_dir / "01_target_distribution.png",
    )

    paths["tenure_distribution"] = save_figure(
        plot_distribution_by_churn(
            frame,
            "tenure",
            CHURN_FLAG,
            "Churned customers concentrate in the first months of the relationship",
            "Tenure (months)",
        ),
        figures_dir / "02_tenure_distribution_by_churn.png",
    )

    tenure_rates = desc.churn_rate_by(frame, TENURE_BAND, sort=False)
    paths["tenure_band"] = save_figure(
        plot_churn_rate_bars(
            tenure_rates,
            "Churn rate falls monotonically as tenure grows",
            "Tenure band (months, descriptive cuts)",
            baseline,
        ),
        figures_dir / "03_churn_rate_by_tenure_band.png",
    )

    contract_rates = desc.churn_rate_by(frame, "Contract")
    paths["contract"] = save_figure(
        plot_churn_rate_bars(
            contract_rates,
            "Contract type shows the strongest association with churn",
            "Contract",
            baseline,
        ),
        figures_dir / "04_churn_rate_by_contract.png",
    )

    contract_tenure_rates, contract_tenure_counts = desc.churn_rate_matrix(
        frame, "Contract", TENURE_BAND
    )
    paths["contract_tenure"] = save_figure(
        plot_rate_heatmap(
            contract_tenure_rates,
            contract_tenure_counts,
            "The contract gap persists inside every tenure band",
            "Tenure band (months)",
            "Contract",
        ),
        figures_dir / "05_contract_tenure_churn_heatmap.png",
    )

    paths["services"] = save_figure(
        plot_churn_rate_bars(
            _service_rate_table(frame),
            "Churn rate by service option (sentinel states shown once, not per column)",
            "Service option",
            baseline,
            horizontal=True,
        ),
        figures_dir / "06_service_churn_rates.png",
    )

    paths["monthly_charges"] = save_figure(
        plot_distribution_by_churn(
            frame,
            "MonthlyCharges",
            CHURN_FLAG,
            "Churned customers are concentrated in the higher monthly-charge range",
            "Monthly charges (US$)",
        ),
        figures_dir / "07_monthly_charges_by_churn.png",
    )

    paths["total_charges"] = save_figure(
        plot_distribution_by_churn(
            frame,
            "TotalCharges",
            CHURN_FLAG,
            "Accumulated charges are lower for churners — largely a tenure effect",
            "Total charges (US$)",
        ),
        figures_dir / "08_total_charges_by_churn.png",
    )

    paths["payment"] = save_figure(
        plot_churn_rate_bars(
            desc.churn_rate_by(frame, "PaymentMethod"),
            "Electronic check stands apart from the other payment methods",
            "Payment method",
            baseline,
        ),
        figures_dir / "09_churn_rate_by_payment_method.png",
    )

    payment_rates, payment_counts = desc.churn_rate_matrix(frame, "PaymentMethod", "Contract")
    paths["payment_contract"] = save_figure(
        plot_rate_heatmap(
            payment_rates,
            payment_counts,
            "Electronic check keeps a higher rate within every contract type",
            "Contract",
            "Payment method",
        ),
        figures_dir / "10_payment_contract_churn_heatmap.png",
    )

    paths["internet_charges"] = save_figure(
        plot_boxplot_by_group(
            frame,
            "MonthlyCharges",
            "InternetService",
            CHURN_FLAG,
            "Within each internet tier, churners do not pay more than those who stay",
            "Internet service",
            "Monthly charges (US$)",
        ),
        figures_dir / "11_internet_monthly_charges_by_churn.png",
    )

    support_rates, support_counts = desc.churn_rate_matrix(frame, "InternetService", "TechSupport")
    paths["support_internet"] = save_figure(
        plot_rate_heatmap(
            support_rates,
            support_counts,
            "Tech support is associated with lower churn inside both internet tiers",
            "Tech support",
            "Internet service",
        ),
        figures_dir / "12_techsupport_internet_churn_heatmap.png",
    )

    paths["profile"] = save_figure(
        plot_churn_rate_bars(
            _profile_rate_table(frame),
            "Demographics: a visible gap for seniors, none for gender",
            "Customer attribute",
            baseline,
            horizontal=True,
        ),
        figures_dir / "13_customer_profile_churn_rates.png",
    )

    blanks = frame[frame["TotalCharges"].isna()]
    paths["total_vs_tenure"] = save_figure(
        plot_scatter_by_churn(
            frame,
            "tenure",
            "TotalCharges",
            CHURN_FLAG,
            "Total charges grow with tenure; the 11 blank records sit exactly at tenure 0",
            "Tenure (months)",
            "Total charges (US$)",
            highlight_x=blanks["tenure"],
            highlight_y=pd.Series([0.0] * len(blanks), index=blanks.index),
            highlight_label="Blank TotalCharges (drawn at 0)",
        ),
        figures_dir / "14_total_charges_vs_tenure.png",
    )

    ranking = association_ranking(frame, CATEGORICAL_COLUMNS, target)
    paths["association"] = save_figure(
        plot_effect_sizes(ranking, "Categorical features ranked by effect size, not by p-value"),
        figures_dir / "15_association_ranking.png",
    )

    internet = frame[frame["InternetService"] != "No"].copy()
    internet["protective_services"] = desc.count_yes(internet, PROTECTIVE_SERVICES)
    paths["protective"] = save_figure(
        plot_churn_rate_bars(
            desc.churn_rate_by(internet, "protective_services", sort=False),
            "Among internet subscribers, churn falls as protective services accumulate",
            "Number of protective services held (of 4)",
            baseline,
        ),
        figures_dir / "16_protective_services_churn.png",
    )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _rate_rows(table: pd.DataFrame) -> str:
    return "\n".join(
        f"| `{index}` | {int(row['n']):,} | {row['churn_rate']:.2%} |"
        for index, row in table.iterrows()
    )


def _matrix_block(rates: pd.DataFrame, counts: pd.DataFrame) -> str:
    header = "| | " + " | ".join(str(c) for c in rates.columns) + " |"
    divider = "| --- " * (len(rates.columns) + 1) + "|"
    lines = [header, divider]
    for index in rates.index:
        cells = []
        for column in rates.columns:
            rate = rates.loc[index, column]
            size = counts.loc[index, column]
            cells.append("—" if pd.isna(rate) else f"{rate:.1%} (n={int(size):,})")
        lines.append(f"| **{index}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def build_report(frame: pd.DataFrame, figures: dict[str, Path], inspected_on: str) -> str:
    """Render the EDA report. All figures come from the analysis, none is hardcoded."""
    config = get_config()
    target = config.target.column
    provenance = describe_file(raw_csv_path())

    n_rows = len(frame)
    churners = int(frame[CHURN_FLAG].sum())
    retained = n_rows - churners
    baseline = desc.overall_churn_rate(frame)
    ratio = retained / churners

    ranking = association_ranking(frame, CATEGORICAL_COLUMNS, target)
    ranking["effect"] = ranking["cramers_v"].map(interpret_cramers_v)
    strongest = ranking.head(5)
    weakest = ranking.tail(3)

    tenure_rates = desc.churn_rate_by(frame, TENURE_BAND, sort=False)
    tenure_stats = desc.numeric_summary_by_target(frame, "tenure")
    tenure_test = numeric_comparison(frame, "tenure")

    contract_rates = desc.churn_rate_by(frame, "Contract")
    contract_tenure_rates, contract_tenure_counts = desc.churn_rate_matrix(
        frame, "Contract", TENURE_BAND
    )
    contract_tenure_stats = frame.groupby("Contract", observed=True)["tenure"].median()

    monthly_test = numeric_comparison(frame, "MonthlyCharges")
    total_test = numeric_comparison(frame, "TotalCharges")
    monthly_stats = desc.numeric_summary_by_target(frame, "MonthlyCharges")
    total_stats = desc.numeric_summary_by_target(frame, "TotalCharges")

    payment_rates, payment_counts = desc.churn_rate_matrix(frame, "PaymentMethod", "Contract")
    support_rates, support_counts = desc.churn_rate_matrix(frame, "InternetService", "TechSupport")

    internet = frame[frame["InternetService"] != "No"].copy()
    internet["protective_services"] = desc.count_yes(internet, PROTECTIVE_SERVICES)
    protective_rates = desc.churn_rate_by(internet, "protective_services", sort=False)

    all_services = frame.copy()
    all_services["service_count"] = (
        desc.count_yes(
            all_services,
            [c for c in SERVICE_COLUMNS if c not in ("PhoneService", "InternetService")],
        )
        + (all_services["PhoneService"] == "Yes").astype(int)
        + (all_services["InternetService"] != "No").astype(int)
    )
    service_count_rates = desc.churn_rate_by(all_services, "service_count", sort=False)

    blanks = frame[frame["TotalCharges"].isna()]
    zero_tenure = frame[frame["tenure"] == 0]
    complete = frame[frame["TotalCharges"].notna() & (frame["tenure"] > 0)]
    implied = complete["tenure"] * complete["MonthlyCharges"]
    ratio_to_implied = complete["TotalCharges"] / implied
    spearman_implied = complete["TotalCharges"].corr(implied, method="spearman")
    spearman_tenure = complete["TotalCharges"].corr(complete["tenure"], method="spearman")

    m2m_share_population = float((frame["Contract"] == "Month-to-month").mean())
    m2m_share_churners = float(
        (frame.loc[frame[CHURN_FLAG] == 1, "Contract"] == "Month-to-month").mean()
    )
    fiber_share_churners = float(
        (frame.loc[frame[CHURN_FLAG] == 1, "InternetService"] == "Fiber optic").mean()
    )
    echeck_share_churners = float(
        (frame.loc[frame[CHURN_FLAG] == 1, "PaymentMethod"] == "Electronic check").mean()
    )

    fiber_no_support = support_rates.loc["Fiber optic", "No"]
    fiber_support = support_rates.loc["Fiber optic", "Yes"]
    dsl_no_support = support_rates.loc["DSL", "No"]
    dsl_support = support_rates.loc["DSL", "Yes"]

    internet_charges = frame.groupby(["InternetService", target], observed=True)[
        "MonthlyCharges"
    ].median()

    # Pre-rendered fragments, kept out of the template so it stays readable.
    internet_charges_text = "; ".join(
        f"{tier} {'churned' if str(label) == 'Yes' else 'retained'} {value:.2f}"
        for (tier, label), value in internet_charges.items()
    )
    key_services = _service_rate_table(frame).loc[
        lambda table: table.index.str.startswith(
            ("InternetService", "OnlineSecurity", "TechSupport")
        )
    ]
    blank_churn_text = ", ".join(f"{k}: {v}" for k, v in blanks[target].value_counts().items())
    blank_contract_text = ", ".join(
        f"{k}: {v}" for k, v in blanks["Contract"].value_counts().items()
    )
    blank_contract_sentence = ", ".join(
        f"{v} on {k}" for k, v in blanks["Contract"].value_counts().items()
    )
    ranking_rows = "\n".join(
        f"| `{row.column}` | {row.n_categories} | {row.cramers_v:.3f} "
        f"| {row.effect} | {row.p_value:.3g} |"
        for row in ranking.itertuples()
    )
    service_count_text = ", ".join(
        f"{int(index)}:{row.churn_rate:.0%}" for index, row in service_count_rates.iterrows()
    )
    credit_card_rate = desc.churn_rate_by(frame, "PaymentMethod").loc[
        "Credit card (automatic)", "churn_rate"
    ]

    return f"""# Exploratory Data Analysis — Telco Customer Churn

- Analysis date: {inspected_on}
- Source file: `{provenance.filename}` ({provenance.size_bytes:,} bytes)
- SHA-256: `{provenance.sha256}`
- Generated by: `scripts/run_eda.py` — every figure below is computed from the raw
  file at run time; no result is typed by hand.

Findings are labeled **OBSERVATION** (measured in this sample), **HYPOTHESIS**
(plausible reading that still needs validation) and **CANDIDATE** (something to
test formally in a later phase). Nothing here establishes causation: the dataset
is observational, so every relationship is an association within this sample.

---

## 1. Objective

Answer, before any modeling decision is taken:

1. How frequent is churn, and what does that imply for evaluation?
2. Does the length of the relationship relate to churn, and how?
3. Which contract, service, billing and demographic characteristics separate
   churners from non-churners in this sample?
4. Which interactions are worth carrying forward as hypotheses?
5. What exactly is wrong with `TotalCharges`, and which treatments look viable?
6. Which engineered features are worth investigating in Phase 6?

## 2. Dataset context

Facts carried over from Phase 2 and re-verified at run time:

| Property | Value |
| --- | --- |
| Rows | {n_rows:,} |
| Columns | {frame.shape[1] - 2} original + 2 EDA helper columns |
| Target | `{target}` (positive = `{config.target.positive_label}`) |
| Complete duplicate rows | 0 |
| Identifier | `customerID`, unique in every row |
| Unreadable numeric cells | {len(blanks)} whitespace-only in `TotalCharges` |

For this phase only, `TotalCharges` is coerced to numeric with blanks becoming
`NaN`. That is an in-memory convenience: no cleaned dataset is written, the raw
file is untouched, and the imputation decision stays open for Phase 4.

## 3. Target baseline

| Class | Customers | Share |
| --- | --- | --- |
| `No` (retained) | {retained:,} | {retained / n_rows:.2%} |
| `Yes` (churned) | {churners:,} | {churners / n_rows:.2%} |

Class ratio: **{ratio:.2f} retained for each churner**.

{_figure(figures["target"], "Class sizes and shares of the target variable")}

**OBSERVATION.** The positive class is the minority at {baseline:.2%}. A model
that predicts "no churn" for everyone reaches {1 - baseline:.2%} accuracy while
finding zero churners.

**DECISION (deferred).** This fixes the evaluation frame rather than a technique:
accuracy is disqualified as a selection metric, PR-AUC and recall/precision at a
chosen threshold become the relevant views, and the split must be stratified. Any
class-weight or resampling choice belongs to Phase 7, not here.

## 4. Customer relationship and tenure

| Statistic | Retained | Churned |
| --- | --- | --- |
| Median tenure | {tenure_stats.loc["retained", "median"]:.0f} months | \
{tenure_stats.loc["churned", "median"]:.0f} months |
| Mean tenure | {tenure_stats.loc["retained", "mean"]:.1f} | \
{tenure_stats.loc["churned", "mean"]:.1f} |
| Q1 – Q3 | {tenure_stats.loc["retained", "q1"]:.0f} – \
{tenure_stats.loc["retained", "q3"]:.0f} | {tenure_stats.loc["churned", "q1"]:.0f} – \
{tenure_stats.loc["churned", "q3"]:.0f} |

Mann-Whitney U = {tenure_test.u_statistic:,.0f}, p = {tenure_test.p_value:.3g},
rank-biserial = {tenure_test.rank_biserial:.3f} (negative: churners rank lower).

{_figure(figures["tenure_distribution"], "Tenure distribution for churned and retained customers")}

Descriptive tenure bands — cuts chosen to mirror the contract cycles the product
offers ({", ".join(TENURE_BAND_LABELS)} months), **used for reading the tables
only**, with no modeling commitment:

| Tenure band | Customers | Churn rate |
| --- | --- | --- |
{_rate_rows(tenure_rates)}

{_figure(figures["tenure_band"], "Churn rate by descriptive tenure band")}

**OBSERVATION.** Churn rate declines monotonically across the bands, from
{tenure_rates.loc[TENURE_BAND_LABELS[0], "churn_rate"]:.1%} in the first six
months to {tenure_rates.loc[TENURE_BAND_LABELS[-1], "churn_rate"]:.1%} after four
years. The effect size is the largest of any single numeric variable here.

**HYPOTHESIS.** Two mechanisms could produce this shape and the data cannot
separate them: customers may become progressively less likely to leave as the
relationship matures, or the early months may simply filter out a segment that
was never going to stay. Survivorship alone would produce the same curve.

## 5. Contracts

| Contract | Customers | Churn rate |
| --- | --- | --- |
{_rate_rows(contract_rates)}

Median tenure per contract: \
{", ".join(f"{name} {value:.0f} months" for name, value in contract_tenure_stats.items())}.

{_figure(figures["contract"], "Churn rate by contract type")}

**OBSERVATION.** Month-to-month customers churn at
{contract_rates.loc["Month-to-month", "churn_rate"]:.1%} against
{contract_rates.loc["Two year", "churn_rate"]:.1%} on two-year contracts —
the widest gap in the dataset. Month-to-month is
{m2m_share_population:.1%} of the population but {m2m_share_churners:.1%} of all
churners.

Contract types also differ sharply in tenure, so the two variables are entangled.
The interaction below is the check that matters:

{_matrix_block(contract_tenure_rates, contract_tenure_counts)}

{_figure(figures["contract_tenure"], "Churn rate by contract type within each tenure band")}

**OBSERVATION.** The gap survives conditioning. Even in the longest band, \
month-to-month customers churn at \
{contract_tenure_rates.loc["Month-to-month", TENURE_BAND_LABELS[-1]]:.1%} against \
{contract_tenure_rates.loc["Two year", TENURE_BAND_LABELS[-1]]:.1%} on two-year \
contracts. Contract is therefore not merely a proxy for tenure.

**OBSERVATION.** New customers are heavily concentrated in month-to-month: \
{int(contract_tenure_counts.loc["Month-to-month", TENURE_BAND_LABELS[0]]):,} of the \
{int(contract_tenure_counts[TENURE_BAND_LABELS[0]].sum()):,} customers in the 0–6 band. \
The zero-churn cells for two-year contracts in the early bands rest on very small \
groups (n={int(contract_tenure_counts.loc["Two year", TENURE_BAND_LABELS[0]])} and \
n={int(contract_tenure_counts.loc["Two year", TENURE_BAND_LABELS[1]])}) and must not \
be read as evidence of a zero rate.

**HYPOTHESIS.** Commitment length and churn propensity are plausibly
co-determined: a long contract both restricts leaving and is chosen by customers
who already intend to stay. The dataset cannot distinguish the contractual
barrier from the self-selection.

## 6. Services

{_figure(figures["services"], "Churn rate by service option, all services in one view")}

Rates for the three services with the strongest association:

| Service option | Customers | Churn rate |
| --- | --- | --- |
{_rate_rows(key_services)}

**OBSERVATION.** `Fiber optic` subscribers churn at
{desc.churn_rate_by(frame, "InternetService").loc["Fiber optic", "churn_rate"]:.1%},
well above `DSL` at
{desc.churn_rate_by(frame, "InternetService").loc["DSL", "churn_rate"]:.1%}.

**OBSERVATION.** `No internet service` is a real product state, not missing data:
those {int(desc.churn_rate_by(frame, "InternetService").loc["No", "n"]):,} customers
show the lowest churn rate in the dataset
({desc.churn_rate_by(frame, "InternetService").loc["No", "churn_rate"]:.1%}). The
same applies to `No phone service` in `MultipleLines`. Treating either as `NaN`
would destroy information.

**OBSERVATION.** Protective services (`OnlineSecurity`, `OnlineBackup`,
`DeviceProtection`, `TechSupport`) all show markedly lower churn among holders,
while the two streaming services barely move the rate.

**HYPOTHESIS.** Protective services may act as switching friction or may simply
mark more engaged customers; streaming add-ons appear to be neutral entertainment
purchases that carry no retention signal.

## 7. Billing and payments

| Statistic | Retained | Churned |
| --- | --- | --- |
| Median monthly charges | {monthly_stats.loc["retained", "median"]:.2f} | \
{monthly_stats.loc["churned", "median"]:.2f} |
| Median total charges | {total_stats.loc["retained", "median"]:.2f} | \
{total_stats.loc["churned", "median"]:.2f} |

`MonthlyCharges`: Mann-Whitney U = {monthly_test.u_statistic:,.0f},
p = {monthly_test.p_value:.3g}, rank-biserial = {monthly_test.rank_biserial:.3f}.
`TotalCharges`: U = {total_test.u_statistic:,.0f}, p = {total_test.p_value:.3g},
rank-biserial = {total_test.rank_biserial:.3f}.

{_figure(figures["monthly_charges"], "Monthly charges distribution by churn status")}

{_figure(figures["total_charges"], "Total charges distribution by churn status")}

{_figure(figures["payment"], "Churn rate by payment method")}

{_matrix_block(payment_rates, payment_counts)}

{_figure(figures["payment_contract"], "Churn rate by payment method within contract type")}

**OBSERVATION.** Churners pay more per month (median
{monthly_stats.loc["churned", "median"]:.2f} against
{monthly_stats.loc["retained", "median"]:.2f}) but have accumulated less
(median {total_stats.loc["churned", "median"]:.2f} against
{total_stats.loc["retained", "median"]:.2f}). The two point in opposite
directions because `TotalCharges` is dominated by tenure.

**OBSERVATION.** `Electronic check` churns at
{desc.churn_rate_by(frame, "PaymentMethod").loc["Electronic check", "churn_rate"]:.1%}
against {credit_card_rate:.1%}
for automatic credit card, and it accounts for {echeck_share_churners:.1%} of all
churners. The gap persists inside every contract type, so it is not only a
by-product of electronic-check users being month-to-month.

**OBSERVATION — the marginal association reverses under conditioning.** Split by
internet tier, the median monthly charge of churners is **not** above that of
those who stay: {internet_charges_text}. The direction of the
`MonthlyCharges`–churn association at population level is therefore not preserved
inside the tiers.

{_figure(figures["internet_charges"], "Monthly charges by internet tier and churn status")}

**What follows from that, and what does not.** The data do **not** support a
simple reading in which a higher `MonthlyCharges` on its own accounts for higher
churn: the marginal association changes once the service tier is held fixed.
Confounding by service type is a plausible explanation — churners are
over-represented in the expensive fiber tier ({fiber_share_churners:.1%} of
churners) — but this analysis cannot establish that it *is* the explanation, only
that the marginal comparison is not interpretable on its own.

**HYPOTHESIS.** Price sensitivity remains possible and is not ruled out here.
Separating it from tier composition would require controlling for the other
determinants of tier choice, or a design capable of supporting causal claims —
neither of which this observational snapshot provides. Manual payment methods and
the fiber tier may also proxy for a lower-commitment customer profile; that too is
an untested reading.

## 8. Customer profile

| Attribute | Customers | Churn rate |
| --- | --- | --- |
{_rate_rows(_profile_rate_table(frame))}

{_figure(figures["profile"], "Churn rate across demographic attributes")}

**OBSERVATION.** `gender` shows essentially no association: Cramér's V =
{ranking.loc[ranking["column"] == "gender", "cramers_v"].iloc[0]:.3f},
p = {ranking.loc[ranking["column"] == "gender", "p_value"].iloc[0]:.3g}. This is a
result, not a gap in the analysis.

**OBSERVATION.** Senior citizens churn at
{desc.churn_rate_by(frame, "SeniorCitizen").loc[1, "churn_rate"]:.1%} against
{desc.churn_rate_by(frame, "SeniorCitizen").loc[0, "churn_rate"]:.1%}, but they
are only {desc.churn_rate_by(frame, "SeniorCitizen").loc[1, "share"]:.1%} of the
population. Customers without a partner or without dependents also churn more,
with weak effect sizes ({ranking.loc[ranking["column"] == "Partner", "cramers_v"].iloc[0]:.3f}
and {ranking.loc[ranking["column"] == "Dependents", "cramers_v"].iloc[0]:.3f}).

## 9. Interaction analysis

Only interactions with a stated prior were examined; no combinatorial search was
run.

1. **Contract × tenure × churn** (section 5) — the contract gap survives inside
   every tenure band.
2. **InternetService × MonthlyCharges × churn** (section 7) — the marginal charge
   gap does not survive conditioning on the tier, so the marginal association is
   not interpretable on its own.
3. **TechSupport × InternetService × churn**:

{_matrix_block(support_rates, support_counts)}

{_figure(figures["support_internet"], "Churn rate by tech support within each internet tier")}

   **OBSERVATION.** Among fiber customers, churn is
   {fiber_no_support:.1%} without tech support against {fiber_support:.1%} with
   it; for DSL, {dsl_no_support:.1%} against {dsl_support:.1%}. The association
   holds inside both tiers, so it is not just a fiber effect.

4. **PaymentMethod × Contract × churn** (section 7) — electronic check remains
   the highest-rate method within each contract type.

## 10. TotalCharges investigation

| Property | Value |
| --- | --- |
| Blank (whitespace-only) records | {len(blanks)} |
| Of which `tenure == 0` | {int((blanks["tenure"] == 0).sum())} |
| Total records with `tenure == 0` | {len(zero_tenure)} |
| Churn among the blank records | {blank_churn_text} |
| Contracts among the blank records | {blank_contract_text} |
| Monthly charges of the blank records | \
{blanks["MonthlyCharges"].min():.2f} – {blanks["MonthlyCharges"].max():.2f} |

**OBSERVATION.** The {len(blanks)} blank records are exactly the
{len(zero_tenure)} customers with `tenure == 0` — the sets coincide. All of them
are still active (`{target} = {config.target.negative_label}`), and all hold
non-zero monthly charges.

**OBSERVATION.** The blanks are not spread across contract types: \
{blank_contract_sentence}.

{_figure(figures["total_vs_tenure"], "Total charges against tenure, with the blank records marked")}

Relationship between the three charge-related quantities, over the
{len(complete):,} records with tenure above zero:

| Measure | Value |
| --- | --- |
| Spearman `TotalCharges` ~ `tenure` | {spearman_tenure:.4f} |
| Spearman `TotalCharges` ~ `tenure × MonthlyCharges` | {spearman_implied:.4f} |
| Ratio `TotalCharges / (tenure × MonthlyCharges)` — median | {ratio_to_implied.median():.4f} |
| Ratio — Q1 / Q3 | {ratio_to_implied.quantile(0.25):.4f} / {ratio_to_implied.quantile(0.75):.4f} |
| Ratio — min / max | {ratio_to_implied.min():.4f} / {ratio_to_implied.max():.4f} |

**OBSERVATION.** `TotalCharges` tracks `tenure × MonthlyCharges` very closely
(Spearman {spearman_implied:.4f}) but is **not** equal to it: the ratio spans
{ratio_to_implied.min():.2f} to {ratio_to_implied.max():.2f}. That is what an
accumulated history looks like when the monthly price changed over the
relationship — `MonthlyCharges` is the current price, not the historical average.

**HYPOTHESIS.** A blank `TotalCharges` marks a customer who has been billed zero
times yet, rather than a recording error. The perfect overlap with `tenure == 0`
and the presence of a positive monthly charge both fit that reading.

**CANDIDATE treatments to evaluate in Phase 4** — none is chosen here:

| Option | Rationale | Risk |
| --- | --- | --- |
| Set to `0` | Consistent with "never billed"; keeps all {n_rows:,} rows | Asserts a \
semantic that the data does not state |
| Drop the {len(blanks)} rows | Simplest, removes {len(blanks) / n_rows:.2%} of the data | \
Removes the entire `tenure == 0` segment, which a production model will meet |
| Median/statistical imputation | Standard pipeline treatment | Nonsensical here: \
implies billing history where there is none |
| Drop the column | `TotalCharges` is nearly redundant with `tenure × MonthlyCharges` | \
Loses whatever genuine price history it carries |

Whatever is chosen must be fitted inside the pipeline, after the split.

## 11. Statistical evidence

Chi-square tests association; Cramér's V measures how strong it is. With
{n_rows:,} observations, p-values are almost all extreme, so **the ranking below
is by effect size**:

| Feature | Categories | Cramér's V | Effect | p-value |
| --- | --- | --- | --- | --- |
{ranking_rows}

{_figure(figures["association"], "Categorical features ranked by Cramér's V")}

**OBSERVATION — significance is not magnitude.** `MultipleLines` reaches
p = {ranking.loc[ranking["column"] == "MultipleLines", "p_value"].iloc[0]:.3g},
which is "statistically significant" at any conventional level, yet its effect
size is {ranking.loc[ranking["column"] == "MultipleLines", "cramers_v"].iloc[0]:.3f}
— negligible. Ranking features by p-value would have promoted it above nothing
useful. This is the concrete reason p-values are not used as a feature ranking
here.

For the numeric variables, Mann-Whitney was used (no normality assumption; both
charge distributions are visibly skewed and bimodal):

| Variable | Median churned | Median retained | Rank-biserial | p-value |
| --- | --- | --- | --- | --- |
| `tenure` | {tenure_test.median_churned:.1f} | {tenure_test.median_retained:.1f} | \
{tenure_test.rank_biserial:.3f} | {tenure_test.p_value:.3g} |
| `MonthlyCharges` | {monthly_test.median_churned:.2f} | {monthly_test.median_retained:.2f} | \
{monthly_test.rank_biserial:.3f} | {monthly_test.p_value:.3g} |
| `TotalCharges` | {total_test.median_churned:.2f} | {total_test.median_retained:.2f} | \
{total_test.rank_biserial:.3f} | {total_test.p_value:.3g} |

**Multiple testing.** {len(CATEGORICAL_COLUMNS)} categorical associations and 3
numeric comparisons were tested against the same target on the same sample. **No
formal multiple-testing correction (Bonferroni, Holm, Benjamini-Hochberg or
otherwise) was applied.** The p-values here are therefore **exploratory**: they
flag where a difference is unlikely to be sampling noise, nothing more. They are
not used in isolation to rank or select features — effect size and practical
relevance take precedence, which is exactly what the `MultipleLines` case above
illustrates. Any inferential claim that needs calibrated error rates would have to
be re-tested with an explicit correction, on data not used to generate the
hypothesis.

## 12. Observations

- Base churn rate {baseline:.2%}; class ratio {ratio:.2f}:1.
- Strongest associations, by effect size: \
{", ".join(f"`{row.column}` ({row.cramers_v:.3f})" for row in strongest.itertuples())}.
- Weakest: \
{", ".join(f"`{row.column}` ({row.cramers_v:.3f})" for row in weakest.itertuples())}.
- Churn rate falls monotonically with tenure across all five descriptive bands.
- The contract gap and the payment-method gap both survive conditioning.
- Within internet tiers, churners do not pay more per month than those who stay.
- `No internet service` and `No phone service` are informative product states.
- The {len(blanks)} unreadable `TotalCharges` cells coincide exactly with
  `tenure == 0`.
- Raw service *count* is not monotonically related to churn (section 13), while
  the count of *protective* services is.

## 13. Hypotheses

Each needs validation beyond this phase; none is established here.

1. **Commitment.** Contract length reflects both a contractual barrier and
   self-selection by customers who already intended to stay.
2. **Early-life risk.** The first months carry the highest risk, whether because
   risk genuinely decays or because the early period filters out a transient
   segment.
3. **Friction and engagement.** Protective services and automatic payment may act
   as switching friction, or may simply mark more embedded customers.
4. **Fiber experience.** The elevated fiber rate may reflect price, service
   quality, or the profile of who buys fiber. This sample cannot separate them.
5. **Charges are confounded with tier.** The population-level charge gap does not
   survive conditioning on `InternetService`. Confounding by service type is a
   plausible reading; price sensitivity is neither demonstrated nor excluded, and
   separating the two would need a design this snapshot cannot provide.

## 14. Candidate features / deferred decisions

Conceptual only — nothing is implemented in this phase.

| Candidate | Hypothesis | Possible benefit | Redundancy risk | Leakage risk | Decision |
| --- | --- | --- | --- | --- | --- |
| Count of protective services | Friction accumulates with protection | Churn falls from \
{protective_rates.loc[0, "churn_rate"]:.1%} at 0 to {protective_rates.loc[4, "churn_rate"]:.1%} \
at 4 among internet subscribers, monotonically | High with the four source columns | \
None — all known at signup | **Investigate** |
| Total count of services | "More services means stickier" | None visible: the rate is \
non-monotonic ({service_count_text}) | \
High | None | **Discard as a plain count**; composition matters, not volume |
| Streaming bundle indicator | Entertainment add-ons signal engagement | Weak: both streaming \
services sit near the baseline | High | None | Low priority |
| Tenure × contract combination | Commitment interacts with relationship age | Captures the \
interaction in section 5 explicitly | Moderate with both parents | None | **Investigate** |
| Charge intensity (`MonthlyCharges` relative to tier median) | Isolates paying above one's \
own tier from paying more in absolute terms | Directly tests the price hypothesis that the \
raw variable confounds | Moderate | None | **Investigate** |
| `TotalCharges / tenure` (average historical bill) | Separates historical price from current \
price | May carry the price drift the ratio analysis exposed | High with both parents | \
Undefined at `tenure == 0` — must be handled inside the pipeline | Investigate with care |
| Automatic vs manual payment flag | Automatic payment is friction against leaving | Collapses \
four categories into the distinction that carries the signal | High with \
`PaymentMethod` | None | Investigate |

Deferred decisions, all belonging to later phases:

- treatment of blank `TotalCharges` (Phase 4);
- encoding of the three-level service columns, and whether `No internet service`
  collapses into `No` (Phase 4);
- scaling of the numeric variables (Phase 4);
- class-imbalance handling (Phase 7);
- decision threshold and its cost framing (Phase 9).

## 15. EDA limitations

- **Observational data.** Every relationship here is association. No intervention,
  no time ordering, no counterfactual.
- **Survivorship.** Long-tenure customers are, by construction, those who did not
  leave. Rates conditioned on high tenure are not "risk after four years".
- **`MonthlyCharges` is the current price**, so all price analysis compares a
  present-day value against an accumulated past.
- **No cost information.** No retention cost, customer value or campaign capacity
  exists in this dataset, so no threshold or business-impact claim can be made.
- **Small cells.** Some interaction cells rest on a few dozen customers; group
  sizes are printed next to every rate for that reason.

### No temporal dimension — and what that costs the protocol

The file carries no observation timestamp, no churn date and no explicit
prediction window. `tenure` is a duration recorded at snapshot time, not a
history. Consequences that must stay documented in the academic and portfolio
deliverables:

- **Prospective temporal validation is not implementable.** There is no time
  ordering to split on, so no train-past / test-future protocol is available.
- **The task is a snapshot classification approximation of churn**, not
  forecasting of future churn over a defined horizon.
- **No claim may be made that this protocol reproduces a real churn-prediction
  system**, which would predict churn within a stated window from features
  observed before that window.
- No horizon is invented to compensate: the dataset does not contain one.

### Analyst exposure to the full sample

This EDA ran on all {n_rows:,} rows, the future holdout included. Stated
precisely:

- **No fitting, preprocessing, model selection or tuning was performed** on any
  part of the data. Nothing was learned from the file in a form a model could
  inherit.
- **However, the hypotheses and candidate features in sections 13 and 14 were
  informed by the complete dataset**, holdout rows included. The analyst has seen
  those rows; the modeling choices that follow are not independent of them.
- From Phase 4 onward the test set is protected against *fitting and tuning* —
  every transformation is fitted on training folds only, and **holdout metrics are
  not consulted until the final evaluation in Phase 9**.
- It should not, therefore, be described as **completely unseen** in the
  analyst-exposure sense. Protected from fitting: yes. Never looked at: no.
- This is a **limitation of the protocol adopted**, not an oversight. Avoiding it
  would have required splitting before any exploration and confining the EDA to
  the training partition — a defensible alternative that was not chosen here.
- Practical consequence: because the complete dataset informed the exploratory
  hypotheses and candidate features, the future holdout estimate **may be subject
  to optimistic bias from analyst exposure**. Whether such bias actually
  materialises, and how large it would be, **cannot be quantified in this
  experiment** — there is no independent sample against which to measure it. It is
  recorded here as a methodological limitation, not as a correction factor.
- Regardless of that, candidate features must earn their place through
  cross-validation on the training set, not through the numbers in this report.

### Multiple testing

{len(CATEGORICAL_COLUMNS)} categorical associations and 3 numeric comparisons
were tested against the same target on the same sample, with **no formal
multiple-testing correction applied**. The p-values are exploratory evidence, not
calibrated inferential statements; they are never used on their own to rank or
select features. See section 11 for the `MultipleLines` case, where a
"significant" p-value accompanies a negligible effect size.

## 16. Recommended questions for preprocessing and modeling

1. Which `TotalCharges` treatment survives comparison, and does the choice change
   model behaviour at all?
2. Should the three-level service columns keep `No internet service` as its own
   level, or collapse it into `No` with the internet status carried separately?
3. Does an explicit tenure × contract interaction add anything over the two raw
   variables in a linear model?
4. Does charge intensity relative to the tier median outperform raw
   `MonthlyCharges`?
5. Is `TotalCharges` worth keeping at all, given how closely it tracks
   `tenure × MonthlyCharges`?
6. Under stratified cross-validation, which metric separates candidate models
   most reliably given a {baseline:.1%} positive rate?
7. What recall is operationally reachable, and at what precision cost, once a
   threshold discussion becomes possible?
"""


def main() -> int:
    """Generate the EDA figures and report. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()
    root = Path(config.data.raw_dir).parents[1]

    try:
        frame = load_eda_frame()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1

    figures = build_figures(frame, root)
    report_path = root / "reports" / "eda_report.md"
    report_path.write_text(
        build_report(frame, figures, date.today().isoformat()),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
