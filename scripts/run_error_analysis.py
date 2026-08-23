"""Describe where the frozen churn system errs (Phase 9E).

Usage (from the repository root):

    uv run python scripts/build_split.py --verify
    uv run python scripts/freeze_model.py --verify
    uv run python scripts/evaluate_holdout.py --verify
    uv run python scripts/run_error_analysis.py            # the analysis
    uv run python scripts/run_error_analysis.py --verify   # check the record only

Writes:
    reports/experiments/error_analysis_results.json
    reports/error_analysis_report.md
    reports/figures/error_analysis/*.png

This is a **post-hoc, descriptive** analysis of a result that already exists. The
final evaluation estimate was produced after the model and policy freeze and
committed in `8c8e266`; nothing here revises it, and nothing discovered here may
change the model, the preprocessing, the features, the calibration policy or the
threshold — the sample described here can no longer provide a fresh independent
estimate for a decision taken after it was examined.

Nothing is fitted. No threshold is scored other than the frozen one. No
significance test is run. No identifier is written to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from churn.analysis.frames import TENURE_BAND, TENURE_BAND_LABELS, add_tenure_band
from churn.analysis.plots import save_figure, use_project_style
from churn.config import PROJECT_ROOT, get_config
from churn.data.loader import RawDatasetMismatchError
from churn.modeling.calibration_results import load_results as load_calibration_results
from churn.modeling.error_analysis import (
    AUDIT_AXES,
    FALSE_NEGATIVE,
    FALSE_POSITIVE,
    INTERACTION_AXES,
    MARGIN_BAND_LABELS,
    OUTCOME_CLASSES,
    SLICE_AXES,
    ConfusionReproductionError,
    ErrorAnalysisError,
    classify_outcomes,
    composition,
    high_confidence_errors,
    insufficient_cells,
    interaction_table,
    margin_table,
    marginal_error_counts,
    margins,
    outcome_counts,
    outcome_margin_summary,
    slice_table,
    summarise_probabilities,
    verify_confusion_reproduction,
)
from churn.modeling.error_analysis_plots import (
    plot_error_probability_distribution,
    plot_error_rates_by_slice,
    plot_margin_distribution,
)
from churn.modeling.error_analysis_results import (
    RESULTS_PATH,
    UPSTREAM_ARTEFACTS,
    ErrorAnalysisResults,
    build_results,
    file_digest,
    invariant_failures,
    load_results,
    write_results,
)
from churn.modeling.freeze import IntegrityError, model_fingerprint
from churn.modeling.freeze_results import load_decision_policy, verify_or_raise
from churn.modeling.holdout import score_holdout
from churn.modeling.holdout_results import load_results as load_evaluation_results
from churn.modeling.threshold_results import load_results as load_threshold_results
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_holdout
from churn.preprocessing.target import encode_target

logger = logging.getLogger("run_error_analysis")

REPORT_PATH = PROJECT_ROOT / "reports" / "error_analysis_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "error_analysis"

AXIS_LABELS = {
    "Contract": "contract type",
    TENURE_BAND: "tenure band",
    "InternetService": "internet service",
    "PaymentMethod": "payment method",
    "SeniorCitizen": "senior citizen",
    "gender": "gender",
}

#: One figure per slice axis, in the declared order.
SLICE_FIGURE_NAMES = {
    "Contract": "03_fpr_fnr_by_contract.png",
    TENURE_BAND: "04_fpr_fnr_by_tenure_band.png",
    "InternetService": "05_fpr_fnr_by_internet_service.png",
    "PaymentMethod": "06_fpr_fnr_by_payment_method.png",
    "SeniorCitizen": "07_fpr_fnr_by_senior_citizen.png",
}


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def build_figures(
    results: ErrorAnalysisResults,
    probability: np.ndarray,
    outcomes: np.ndarray,
    tables: dict[str, list],
    root: Path,
) -> dict[str, Path]:
    """Render the Phase 9E figures. Slice charts are skipped when unusable."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    threshold = results.frozen_policy["final_threshold"]
    paths: dict[str, Path] = {}

    paths["probability"] = save_figure(
        plot_error_probability_distribution(probability, outcomes, threshold),
        figures_dir / "01_error_probability_distribution.png",
    )
    paths["margin"] = save_figure(
        plot_margin_distribution(results.margin_analysis["counts_by_band"]),
        figures_dir / "02_error_margin_distribution.png",
    )

    for axis, filename in SLICE_FIGURE_NAMES.items():
        figure = plot_error_rates_by_slice(tables[axis], AXIS_LABELS[axis], threshold)
        if figure is None:
            logger.info("Skipped the %s figure: no group has an estimable rate.", axis)
            continue
        paths[axis] = save_figure(figure, figures_dir / filename)

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _probability_table(results: ErrorAnalysisResults) -> str:
    lines = [
        "| Outcome | n | Mean | Median | Q1 | Q3 | Min | Max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in OUTCOME_CLASSES:
        row = results.probability_summaries[name]
        lines.append(
            f"| **{name}** | {row['count']:,} | {row['mean']:.4f} | {row['median']:.4f} | "
            f"{row['q1']:.4f} | {row['q3']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
        )
    return "\n".join(lines)


def _margin_band_table(results: ErrorAnalysisResults) -> str:
    counts = results.margin_analysis["counts_by_band"]
    lines = [
        "| Distance from the threshold | " + " | ".join(OUTCOME_CLASSES) + " | Total |",
        "| --- " * (len(OUTCOME_CLASSES) + 2) + "|",
    ]
    for band in MARGIN_BAND_LABELS:
        row = counts[band]
        cells = " | ".join(f"{row[name]:,}" for name in OUTCOME_CLASSES)
        lines.append(f"| `{band}` | {cells} | {sum(row.values()):,} |")
    return "\n".join(lines)


def _outcome_margin_table(results: ErrorAnalysisResults) -> str:
    summary = results.margin_analysis["absolute_margin_by_outcome"]
    lines = [
        "| Outcome | n | Mean | Median | Q1 | Q3 | Max |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in OUTCOME_CLASSES:
        row = summary[name]
        lines.append(
            f"| **{name}** | {row['count']:,} | {row['mean']:.4f} | {row['median']:.4f} | "
            f"{row['q1']:.4f} | {row['q3']:.4f} | {row['max']:.4f} |"
        )
    return "\n".join(lines)


def _slice_table_markdown(results: ErrorAnalysisResults, axis: str) -> str:
    rows = (
        results.slice_analyses[axis]
        if axis in results.slice_analyses
        else results.demographic_audit["axes"][axis]
    )
    lines = [
        "| Group | n | Pos | Neg | Pred+ | TP | FP | FN | TN | Prevalence | Pred+ rate "
        "| Precision | Recall | Specificity | FPR | FNR |",
        "| --- " * 16 + "|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['n']:,} | {row['positives']:,} | {row['negatives']:,} | "
            f"{row['predicted_positives']:,} | {row['true_positives']:,} | "
            f"{row['false_positives']:,} | {row['false_negatives']:,} | "
            f"{row['true_negatives']:,} | {_percent(row['prevalence'])} | "
            f"{_percent(row['predicted_positive_rate'])} | {_percent(row['precision'])} | "
            f"{_percent(row['recall'])} | {_percent(row['specificity'])} | "
            f"{_percent(row['false_positive_rate'])} | {_percent(row['false_negative_rate'])} |"
        )
    return "\n".join(lines)


def _composition_table(results: ErrorAnalysisResults, error_class: str, axis: str) -> str:
    rows = results.error_composition[error_class][axis]
    eligible = "actual non-churners" if error_class == FALSE_POSITIVE else "actual churners"
    lines = [
        f"| Group | {error_class} count | Share of {error_class} | {eligible} "
        f"| Share of {eligible} |",
        "| --- " * 5 + "|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['errors']:,} | {_percent(row['share_of_errors'])} | "
            f"{row['eligible']:,} | {_percent(row['share_of_eligible'])} |"
        )
    return "\n".join(lines)


def _digest_table(results: ErrorAnalysisResults) -> str:
    lines = ["| Artefact | SHA-256 |", "| --- | --- |"]
    for artefact, digest in results.upstream_artefact_digests.items():
        lines.append(f"| `{artefact}` | `{digest[:16]}…` |")
    return "\n".join(lines)


def _insufficient_table(results: ErrorAnalysisResults) -> str:
    if not results.insufficient_cells:
        return "Every group cleared every guardrail; no rate was withheld."
    lines = ["| Axis | Group | Why a rate was withheld |", "| --- | --- | --- |"]
    for entry in results.insufficient_cells:
        lines.append(f"| `{entry['axis']}` | {entry['group']} | {'; '.join(entry['reasons'])} |")
    return "\n".join(lines)


def _eda_table(results: ErrorAnalysisResults) -> str:
    lines = ["| Axis | Observation recorded in Phase 3 | Source |", "| --- | --- | --- |"]
    for entry in results.prior_eda_observations:
        lines.append(f"| `{entry['axis']}` | {entry['prior_observation']} | `{entry['source']}` |")
    return "\n".join(lines)


def build_report(results: ErrorAnalysisResults, figures: dict[str, Path]) -> str:
    """Render the Phase 9E report. Every number comes from this analysis.

    The report carries **no wall-clock metadata**, per the convention recorded in
    `CLAUDE.md`: it is a pure function of its inputs, so a rerun on an unchanged
    repository reproduces it byte for byte and any diff is a real change.
    """
    provenance = results.provenance
    policy = results.frozen_policy
    sample = results.evaluation_sample
    reproduction = results.confusion_reproduction
    derived = reproduction["derived"]
    confident = results.high_confidence_errors
    marginal = results.margin_analysis["marginal_errors"]
    guardrails = results.guardrails
    threshold = policy["final_threshold"]
    limit = marginal["margin_limit"]

    def slice_section(number: int, axis: str, title: str) -> str:
        figure = (
            f"\n\n{_figure(figures[axis], f'Error rates by {AXIS_LABELS[axis]}')}"
            if axis in figures
            else "\n\n*No figure: no group in this axis has an estimable rate pair.*"
        )
        return f"## {number}. {title}\n\n{_slice_table_markdown(results, axis)}{figure}"

    return f"""# Post-hoc Error Analysis — Telco Customer Churn (Phase 9E)

- Generated by: `scripts/run_error_analysis.py`
- Machine-readable record: `reports/experiments/error_analysis_results.json`
- Describes the evaluation committed in: \
`{provenance["evaluation_commit_short"]}` — {provenance["evaluation_commit_subject"]}
- Model frozen in: `{provenance["freeze_commit_short"]}`

> **This report carries no generation date.** It is a pure function of its
> inputs, so re-running the generator on an unchanged repository reproduces it
> byte for byte and any diff is a real change. Git records when it was produced.

---

## 1. Scope and methodological status

This analysis ran **after** the final evaluation was produced and committed. It
describes where the errors that estimate already counted actually fell.

| Property | Value |
| --- | --- |
| `analysis_type` | **{results.analysis_type}** |
| `post_hoc` | **{results.post_hoc}** |
| `descriptive_only` | **{results.descriptive_only}** |
| `holdout_already_consumed` | **{results.holdout_already_consumed}** |
| `selection_allowed` | **{results.selection_allowed}** |
| `model_change_allowed` | **{results.model_change_allowed}** |
| `selection_after_analysis` | **{results.selection_after_analysis}** |

**Why looking is permitted, and acting is not.** The evaluation sample has been
consumed: it has already served as the **designated final held-out evaluation
under the frozen protocol**, so it can no longer provide a fresh independent
estimate for any decision made after it was examined. That is exactly what makes
a descriptive account of its errors harmless — and what makes acting on that
account destructive. Any change to the model, the preprocessing, the features,
the calibration policy or the threshold justified by something in this report
would leave the Phase 9D figures describing a system that no longer exists, with
no held-out sample left to measure the replacement on.

**What the Phase 9D result is, stated precisely.** It is the final evaluation
estimate produced *after* the model and the decision policy were frozen and
committed, on rows that took no part in any fit, fold, feature decision,
calibration gate or threshold search. It is **not** described here as an unbiased
estimate, and the reason is a limitation this project has carried since Phase 3:
the exploratory analysis ran on all 7,043 rows, including the ones that later
became the evaluation sample. The holdout was protected against fitting, tuning
and selection from Phase 4 onwards, but that earlier **analyst exposure** came
first, and it can carry an optimistic bias that this project cannot quantify.
The protection is real and the limitation is real; both are stated rather than
one being allowed to imply the absence of the other.

The question here is therefore **not** "how do we improve the model". It is:
*under what conditions, and in what manner, does this frozen system get it
wrong.*

### What this phase deliberately did not do

| Not done | Count |
| --- | --- |
| Fits | {results.methodology["fit_calls"]} |
| Alternative models evaluated | {results.methodology["alternative_models_evaluated"]} |
| Alternative thresholds scored | {results.methodology["alternative_thresholds_scored"]} |
| Calibrations | {results.methodology["calibrations_performed"]} |
| Hyperparameter searches | {results.methodology["hyperparameter_searches"]} |
| Feature selections | {results.methodology["feature_selections"]} |
| Significance tests | {results.methodology["significance_tests_run"]} |
| Features swept for interesting slices | {results.methodology["features_swept"]} |
| Explainability computed | {results.methodology["explainability_computed"]} |
| Financial quantities assumed | {results.methodology["financial_costs_assumed"]} |
| Causal claims | {results.methodology["causal_claims"]} |
| Identifiers persisted | {results.methodology["identifiers_persisted"]} |

Explainability is **Phase 10's** subject: this phase asks *where* the decisions
were wrong, not *how* the features drove them. The two are kept apart on purpose.

## 2. Frozen result being diagnosed

| Property | Value |
| --- | --- |
| Estimator | `{policy["estimator"]}` |
| Calibration policy | **{policy["calibration_policy"]}** |
| Threshold policy | **{policy["threshold_policy"]}** |
| `final_threshold` | **{threshold}** |
| Comparison | `{policy["comparison"]}` |
| Positive class | label `{policy["positive_class_label"]}`, column \
`{policy["positive_class_column"]}` |
| Reopened here | {policy["reopened_here"]} |
| Alternative thresholds scored | {policy["alternative_thresholds_scored"]} |
| Model fingerprint verified after load | **{provenance["model_fingerprint_verified"]}** |
| Pipeline unchanged by this analysis | **{provenance["pipeline_unchanged_by_analysis"]}** |
| Refitted here | {provenance["refitted_here"]} |

### Integrity gate: the same classification

The outcomes below were rederived from the frozen pipeline, the frozen threshold
and the evaluation sample, then checked against the Phase 9D record. They agree
exactly, which is what makes this a description of *that* result rather than of a
different one:

| Cell | Rederived | Recorded in Phase 9D |
| --- | --- | --- |
| TN | {derived["TN"]:,} | {reproduction["recorded"]["TN"]:,} |
| FP | {derived["FP"]:,} | {reproduction["recorded"]["FP"]:,} |
| FN | {derived["FN"]:,} | {reproduction["recorded"]["FN"]:,} |
| TP | {derived["TP"]:,} | {reproduction["recorded"]["TP"]:,} |
| **Total** | **{reproduction["total"]:,}** | **{sample["n_samples"]:,}** |

Reproduced: **{reproduction["reproduced"]}**. The run aborts on any disagreement.

{_digest_table(results)}

## 3. Error taxonomy

{sample["n_samples"]:,} customers, {sample["n_positive"]:,} of whom churned
(prevalence {sample["prevalence"]:.4f}). Every row falls in exactly one class:

| Outcome | Count | Meaning |
| --- | --- | --- |
| **TN** | {derived["TN"]:,} | retained and not flagged |
| **FP** | {derived["FP"]:,} | retained but flagged — a contact spent on someone who stayed |
| **FN** | {derived["FN"]:,} | churned and **not** flagged — a churner the system missed |
| **TP** | {derived["TP"]:,} | churned and flagged |

The two error classes are the subject of the next two sections:
**{derived["FP"]:,} false positives** and **{derived["FN"]:,} false negatives**.

### Probability distribution by outcome

{_probability_table(results)}

{_figure(figures["probability"], "Predicted probability by outcome class")}

## 4. False-positive profile

Customers whose true label is *retained* and whose score reached the threshold.

**Composition, with the denominator that makes it readable.** The left pair of
columns says how the false positives are made up; the right pair says how the
population that *could* have produced them — the {sample["n_negative"]:,} actual
non-churners — is made up. A category is over-represented among the errors only
relative to the second pair. Reading the first pair alone is the standard
mistake: the largest category usually supplies the most errors by being largest.

{_composition_table(results, FALSE_POSITIVE, "Contract")}

{_composition_table(results, FALSE_POSITIVE, TENURE_BAND)}

{_composition_table(results, FALSE_POSITIVE, "InternetService")}

{_composition_table(results, FALSE_POSITIVE, "PaymentMethod")}

The conditional rate that answers "which categories err most among their own
non-churners" is the **FPR** column in sections 8 to 12, not any share above.

## 5. False-negative profile

Customers who churned and whose score did not reach the threshold. There are
{derived["FN"]:,} of them out of {sample["n_positive"]:,} churners.

False negatives are **operationally important** in a retention setting because
they are churners the system did not flag: no retention action can reach a
customer the model never surfaced. Their **relative** importance against false
positives cannot be established from this dataset, because no business-cost ratio
is available — there is no cost of losing a customer, no cost of a retention
action, no customer value and no operational capacity anywhere in it. This
project never established that a false negative costs more than a false positive,
and nothing in this section asserts it.

{_composition_table(results, FALSE_NEGATIVE, "Contract")}

{_composition_table(results, FALSE_NEGATIVE, TENURE_BAND)}

{_composition_table(results, FALSE_NEGATIVE, "InternetService")}

{_composition_table(results, FALSE_NEGATIVE, "PaymentMethod")}

The conditional rate here is the **FNR** column in sections 8 to 12.

## 6. Distance from the frozen threshold

```text
margin          = probability − {threshold}
absolute margin = |margin|
```

`margin > 0` is the flagged side, `margin < 0` the unflagged side. The bands were
**fixed before the errors were placed in them**
(`bands_fixed_before_analysis: {results.margin_analysis["bands_fixed_before_analysis"]}`),
and the threshold is the frozen one — it is not re-derived here
(`threshold_re_derived: {results.margin_analysis["threshold_re_derived"]}`).

{_margin_band_table(results)}

{_figure(figures["margin"], "Distance from the frozen threshold by outcome")}

### Absolute margin within each outcome class

{_outcome_margin_table(results)}

### Marginal errors

An error is called **marginal** when its score lies within {limit} of the frozen
decision boundary. That means the binary decision is **boundary-proximate under
the frozen operating point**. It does **not** mean the estimated churn
probability is close to 50%: the boundary sits at {threshold}, so a
boundary-proximate score is near *that* value, not near a coin flip.

| Error class | Marginal | Total | Share |
| --- | --- | --- | --- |
| False positives | {marginal["marginal_false_positives"]:,} | \
{marginal["total_false_positives"]:,} | \
{marginal["marginal_false_positives"] / marginal["total_false_positives"]:.1%} |
| False negatives | {marginal["marginal_false_negatives"]:,} | \
{marginal["total_false_negatives"]:,} | \
{marginal["marginal_false_negatives"] / marginal["total_false_negatives"]:.1%} |

Both classes remain errors of the frozen decision — a false negative is a missed
churner whether its score sat close to the boundary or far from it. The margin
analysis does not divide the errors into real and incidental ones; it distinguishes
only **boundary-proximate errors** from **errors whose score lies farther from the
frozen boundary**.

Boundary proximity is **descriptive only**. Moving the threshold would change the
precision/recall trade-off, but evaluating such alternatives on this already
consumed sample would reopen decision selection and is therefore outside the
scope of this analysis. No alternative threshold was scored
(`alternative_thresholds_scored: {policy["alternative_thresholds_scored"]}`), and
no quantitative statement about the effect of moving the boundary appears
anywhere in this report.

## 7. Extreme-score errors

> **Terminology.** The record calls these *high-confidence* errors and the field
> names keep that wording for compatibility. **"High confidence" is descriptive
> shorthand for an extreme model probability under the frozen score. It is not a
> validated per-case confidence guarantee.** The probabilities are uncalibrated by
> the Phase 9A decision, so an extreme score is an extreme *output* of the model,
> not a demonstrated likelihood for the individual customer. This report says
> **extreme-score errors**.

| Criterion | Value |
| --- | --- |
| False positive with probability ≥ | \
{confident["false_positive_probability_minimum"]} |
| False negative with probability ≤ | \
{confident["false_negative_probability_maximum"]} |
| `diagnostic_only` | **{confident["diagnostic_only"]}** |
| `decision_policy_candidate` | **{confident["decision_policy_candidate"]}** |

| Error class | Extreme score | Total | Share |
| --- | --- | --- | --- |
| False positives | {confident["n_high_confidence_false_positives"]:,} | \
{confident["n_false_positives"]:,} | {_percent(confident["share_of_false_positives"])} |
| False negatives | {confident["n_high_confidence_false_negatives"]:,} | \
{confident["n_false_negatives"]:,} | {_percent(confident["share_of_false_negatives"])} |

These two numbers answer one question — *does the model ever assign an extreme
probability and still get the case wrong?* — and nothing else.
{confident["note"]}. The two cut-offs are **not** thresholds: the frozen decision
boundary is {threshold} and is unchanged, and nothing here reopens the
`calibration_policy = {policy["calibration_policy"]}` decision taken in Phase 9A.

## 8. Guardrails and slice protocol

The slice axes were **fixed before this analysis ran**
(`slice_axes_fixed_before_analysis: \
{results.methodology["slice_axes_fixed_before_analysis"]}`), and each carries a
justification recorded in Phase 3, before the evaluation sample was opened:
{", ".join(f"`{axis}`" for axis in SLICE_AXES)}. No sweep over the 19 features was
performed ({results.methodology["features_swept"]} swept), and no automatic search
over their combinations was performed.

| Guardrail | Value |
| --- | --- |
| Minimum group size | {guardrails["minimum_group_size"]} |
| Minimum actual positives for recall / FNR | \
{guardrails["minimum_actual_positives_for_recall_and_fnr"]} |
| Minimum actual negatives for specificity / FPR | \
{guardrails["minimum_actual_negatives_for_specificity_and_fpr"]} |
| Minimum predicted positives for precision | \
{guardrails["minimum_predicted_positives_for_precision"]} |
| Fixed before interpretation | {guardrails["fixed_before_interpretation"]} |
| Applied uniformly | {guardrails["applied_uniformly"]} |
| Categories merged to clear a threshold | \
**{guardrails["categories_merged_to_clear_a_threshold"]}** |
| Insufficient rate shown as | {guardrails["insufficient_rate_representation"]} |

A rate below its own guardrail is reported as **n/a**, never as zero and never as
a number: recall over a dozen churners is not an estimate of anything, and
printing it would invite a reader to treat it as one.

{_insufficient_table(results)}

{slice_section(9, "Contract", "Error rates by contract")}

{slice_section(10, TENURE_BAND, "Error rates by tenure band")}

The bands are the ones the Phase 3 EDA defined
({", ".join(f"`{label}`" for label in TENURE_BAND_LABELS)}), reused unchanged so
the tables stay comparable with that analysis. No cut was searched for here.

{slice_section(11, "InternetService", "Error rates by internet service")}

{slice_section(12, "PaymentMethod", "Error rates by payment method")}

## 13. Descriptive demographic audit

> **Status: {results.demographic_audit["status"]}.**
> `is_fairness_assessment: {results.demographic_audit["is_fairness_assessment"]}` ·
> `establishes_bias: {results.demographic_audit["establishes_bias"]}` ·
> `inferential_tests_run: {results.demographic_audit["inferential_tests_run"]}`

{results.demographic_audit["note"]}.

### SeniorCitizen

{_slice_table_markdown(results, "SeniorCitizen")}
{
        f"{chr(10)}{_figure(figures['SeniorCitizen'], 'Error rates by senior citizen status')}"
        if "SeniorCitizen" in figures
        else ""
    }

### gender

Included as a **control**: the Phase 3 EDA recorded an essentially null
association between gender and churn, which makes a large difference in error
rates here surprising in a way that a difference on `Contract` would not be.

{_slice_table_markdown(results, "gender")}

Calling this a fairness analysis would overstate what it is. It is a description
of two rates in one sample, with no test, no margin and no protected-attribute
framework behind it. A real fairness assessment would need a stated notion of
fairness, a pre-specified tolerance and data this project does not have.

## 14. Secondary interaction

> **Status: {results.interactions["status"]}.**
> `automatic_combination_search: \
{results.interactions["automatic_combination_search"]}`

{results.interactions["note"]}.

Only `Contract × tenure_band` is reported, because it is the pair the EDA already
crossed. A three-way or four-way breakdown would multiply the cells past the
point where any of them supports a rate, and the exercise would become post-hoc
mining rather than description.

| Cell | n | Pos | Neg | TP | FP | FN | TN | Recall | Specificity | FPR | FNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{
        chr(10).join(
            f"| {row['group']} | {row['n']:,} | {row['positives']:,} | {row['negatives']:,} | "
            f"{row['true_positives']:,} | {row['false_positives']:,} | "
            f"{row['false_negatives']:,} | {row['true_negatives']:,} | "
            f"{_percent(row['recall'])} | {_percent(row['specificity'])} | "
            f"{_percent(row['false_positive_rate'])} | {_percent(row['false_negative_rate'])} |"
            for row in results.interactions["cells"]["Contract x tenure_band"]
        )
    }

## 15. Comparison with earlier EDA observations

{_eda_table(results)}

**Two different things, kept apart.** An *association with churn* is a property
of the data: month-to-month customers churned more often. An *error rate* is a
property of the model's decisions on a group. They are not the same quantity and
one does not imply the other — a group with a high churn rate can be easy for the
model precisely because the signal is strong there, and a group with a low churn
rate can be hard because its few churners look like everyone else.

The prior observations above are quoted as context for reading the tables, not as
an explanation of any rate in them, and no new causal story is constructed from
this sample.

## 16. What this analysis can and cannot support

**It can support:** a description of how the {derived["FP"] + derived["FN"]:,}
errors are distributed across the pre-specified axes; how far each error class
sits from the frozen cut; how much of the error is a boundary effect versus a
confident mistake; and which cells are too small for any of that to be said.

**It cannot support:**

- any change to the model, the preprocessing, the features, the calibration
  policy or the threshold — not because the findings are uninteresting, but
  because the sample that produced them is spent;
- a claim that any group is treated unfairly: no fairness notion was specified,
  no tolerance was set and no test was run;
- a statistical claim of any kind about a subgroup difference. With this many
  descriptive cells examined after the outcome was known, a search for
  significant subgroups would find some whether or not anything is there, and
  **{results.methodology["significance_tests_run"]}** tests were run;
- a causal statement. Nothing here shows that changing an attribute would change
  an outcome;
- a monetary reading. A false negative is a churner the system did not identify
  and a false positive is a flagged customer who stayed; neither carries a value
  this dataset contains.

## 17. Implications for future work

Recorded as **hypotheses for a future development cycle**, not as changes to be
made to the current model. Each would need a fresh, independent validation set
to be assessed honestly — the sample described here can no longer serve that
purpose.

- A future cycle **could investigate** whether the error concentration visible in
  the tables above persists on new data, or whether it is a property of this
  sample.
- A future cycle **could investigate** whether features capturing the conditions
  where false negatives cluster carry signal the present 19 do not, evaluated
  under the same leakage-safe protocol used in Phase 6.
- A future cycle **could investigate** an operating point chosen against real
  costs, if a retention programme ever supplies them — Phase 9B recorded that the
  cost-optimal threshold depends entirely on a ratio this dataset does not
  contain.
- A future cycle **could investigate** whether the extreme-score errors in section
  7 share a describable structure, using the explainability tooling Phase 10
  introduces.

None of these is a recommendation to modify the frozen system, and none may be
applied retroactively: the Phase 9D figures describe the system as it was frozen,
and there is no held-out sample left in this project on which a modified system
could be measured.

## 18. Limitations

- **Post-hoc and descriptive.** Everything here was computed after the outcome was
  known, on a sample that is now spent.
- **One sample, finite groups.** Several cells are too small to support a rate and
  are marked `n/a`; the ones that are large enough are still single-sample
  descriptions with no interval attached.
- **No inference.** No test, no p-value, no interval on any slice. Differences
  between groups are observations about this sample.
- **Pre-specified axes only.** Five axes plus one control and one interaction.
  Error structure outside them was not examined, and its absence here is not
  evidence of its absence in the data.
- **No explainability.** Where the model errs is described; why it errs, in terms
  of feature influence, is Phase 10.
- **No causal claim, no monetary claim, no fairness verdict.**
- **The analyst-exposure limitation from Phase 3 still applies** to the training
  pool and therefore to everything selected on it.
- **Snapshot classification.** The dataset has no time dimension, so "churned"
  means "had churned at the time of the snapshot".

---

`{results.analysis_type}` · `selection_after_analysis: \
{results.selection_after_analysis}` · `model_change_allowed: \
{results.model_change_allowed}`

{results.methodology["statement"]}.
"""


def analyse() -> tuple[ErrorAnalysisResults, np.ndarray, np.ndarray, dict[str, list]]:
    """Run the post-hoc analysis. Gates first, description second."""
    config = get_config()

    manifest = verify_split_manifest(load_split_manifest())
    policy = load_decision_policy()
    evaluation = load_evaluation_results()
    checks, pipeline = verify_or_raise(policy, load_threshold_results(), load_calibration_results())
    logger.info("Freeze verified: %d integrity checks passed.", len(checks))

    pipeline_path = PROJECT_ROOT / policy.artifacts.pipeline_path
    digest_before = file_digest(pipeline_path)
    loaded_fingerprint = model_fingerprint(pipeline)
    if loaded_fingerprint != policy.artifacts.model_fingerprint_sha256:
        raise IntegrityError(
            "The loaded pipeline is not the frozen model; the analysis was not run."
        )

    # The evaluation sample, already consumed by Phase 9D.
    frame = add_tenure_band(load_holdout())
    if len(frame) != manifest.n_rows_holdout:
        raise ErrorAnalysisError(
            f"{len(frame)} rows against {manifest.n_rows_holdout} in the frozen manifest."
        )

    features = build_feature_matrix(frame)
    labels = np.asarray(encode_target(frame[config.target.column]))
    probability = score_holdout(pipeline, features)

    threshold = policy.threshold.final_threshold
    predictions = (probability >= threshold).astype(int)
    outcomes = classify_outcomes(labels, predictions)

    # Integrity gate: the same classification the final estimate came from.
    cells = evaluation.confusion_matrix
    reproduction = verify_confusion_reproduction(
        outcome_counts(outcomes),
        int(cells["true_negatives"]),
        int(cells["false_positives"]),
        int(cells["false_negatives"]),
        int(cells["true_positives"]),
    )

    absolute_margin = np.abs(margins(probability, threshold))
    slices = {axis: slice_table(frame, axis, labels, predictions) for axis in SLICE_AXES}
    audit = {axis: slice_table(frame, axis, labels, predictions) for axis in AUDIT_AXES}
    interactions = {
        f"{first} x {second}": interaction_table(frame, first, second, labels, predictions)
        for first, second in INTERACTION_AXES
    }

    error_masks = {
        FALSE_POSITIVE: (outcomes == FALSE_POSITIVE, labels == 0),
        FALSE_NEGATIVE: (outcomes == FALSE_NEGATIVE, labels == 1),
    }
    composition_tables = {
        error_class: {axis: composition(frame, axis, error_mask, eligible) for axis in SLICE_AXES}
        for error_class, (error_mask, eligible) in error_masks.items()
    }

    digests = {artefact: file_digest(PROJECT_ROOT / artefact) for artefact in UPSTREAM_ARTEFACTS}
    results = build_results(
        policy,
        evaluation,
        reproduction,
        summarise_probabilities(probability, outcomes),
        margin_table(outcomes, absolute_margin),
        outcome_margin_summary(outcomes, absolute_margin),
        marginal_error_counts(outcomes, absolute_margin),
        high_confidence_errors(probability, outcomes),
        composition_tables,
        slices,
        audit,
        interactions,
        insufficient_cells([*slices.values(), *audit.values(), *interactions.values()]),
        digests,
        loaded_fingerprint,
        digest_before,
        file_digest(pipeline_path),
    )
    write_results(results)

    failures = invariant_failures(load_results())
    if failures:
        raise ErrorAnalysisError(
            "The record just written violates its own invariants:\n  " + "\n  ".join(failures)
        )
    logger.info("Record invariants verified.")
    return results, probability, outcomes, {**slices, **audit}


def verify() -> int:
    """Check the existing Phase 9E record. **Read-only.** Returns an exit code."""
    results = load_results()
    failures = invariant_failures(results)

    for failure in failures:
        logger.error("INVARIANT FAILED: %s", failure)
    if failures:
        logger.error("%d invariant(s) failed.", len(failures))
        return 1

    logger.info("All record invariants hold.")
    logger.info(
        "%s: %s, selection_after_analysis=%s",
        results.analysis_type,
        results.confusion_reproduction["derived"],
        results.selection_after_analysis,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the post-hoc analysis, or verify an existing record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the existing analysis record and change nothing",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        if arguments.verify:
            return verify()
        results, probability, outcomes, tables = analyse()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1
    except RawDatasetMismatchError as error:
        logger.error("%s", error)
        return 2
    except SplitManifestMismatchError as error:
        logger.error("%s", error)
        return 3
    except IntegrityError as error:
        logger.error("FREEZE INTEGRITY FAILED — the analysis did not run.\n%s", error)
        return 4
    except ConfusionReproductionError as error:
        logger.error("%s", error)
        return 5
    except ErrorAnalysisError as error:
        logger.error("%s", error)
        return 6

    figures = build_figures(results, probability, outcomes, tables, PROJECT_ROOT)
    Path(REPORT_PATH).write_text(
        build_report(results, figures),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)
    logger.info(
        "Marginal errors within %.3f: %d FP, %d FN",
        results.margin_analysis["marginal_errors"]["margin_limit"],
        results.margin_analysis["marginal_errors"]["marginal_false_positives"],
        results.margin_analysis["marginal_errors"]["marginal_false_negatives"],
    )
    logger.info(
        "Confident errors: %d FP >= %.2f, %d FN <= %.2f",
        results.high_confidence_errors["n_high_confidence_false_positives"],
        results.high_confidence_errors["false_positive_probability_minimum"],
        results.high_confidence_errors["n_high_confidence_false_negatives"],
        results.high_confidence_errors["false_negative_probability_maximum"],
    )
    logger.info("Post-hoc description complete. Nothing was selected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
