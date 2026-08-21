"""Evaluate the frozen churn system on the final holdout (Phase 9D).

Usage (from the repository root):

    uv run python scripts/build_split.py --verify        # prove the split
    uv run python scripts/freeze_model.py --verify       # prove the freeze
    uv run python scripts/evaluate_holdout.py            # the final evaluation
    uv run python scripts/evaluate_holdout.py --verify   # check the record, change nothing

Writes:
    reports/experiments/holdout_results.json
    reports/holdout_evaluation_report.md
    reports/figures/holdout/*.png

**This script opens the final holdout.** It refuses to do so unless the Phase 9C
freeze verifies first: the model, the calibration policy, the threshold, the
decision rule and the source of the whole inference function must all be exactly
what was committed in `9d1db49`, before the holdout existed as anything other
than reserved rows.

Nothing is fitted, searched, calibrated or selected here. One frozen model, one
frozen policy, one frozen holdout, one evaluation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from churn.analysis.plots import save_figure, use_project_style
from churn.config import PROJECT_ROOT, get_config
from churn.data.loader import RawDatasetMismatchError
from churn.modeling.calibration_results import load_results as load_calibration_results
from churn.modeling.freeze import IntegrityError, file_digest, model_fingerprint
from churn.modeling.freeze_results import load_decision_policy, verify_or_raise
from churn.modeling.holdout import (
    BOOTSTRAP_METRICS,
    CONFIDENCE_LEVEL,
    CONFUSION_NAMES,
    N_BOOTSTRAP,
    HoldoutEvaluationError,
    HoldoutMetrics,
    bootstrap_intervals,
    compute_holdout_metrics,
    generalization_check,
    score_holdout,
)
from churn.modeling.holdout_plots import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_probability_distribution,
    plot_roc_curve,
)
from churn.modeling.holdout_results import (
    RESULTS_PATH,
    UPSTREAM_ARTEFACTS,
    HoldoutMismatchError,
    HoldoutResults,
    build_results,
    invariant_failures,
    load_results,
    verify_holdout_partition,
    write_results,
)
from churn.modeling.threshold_results import load_results as load_threshold_results
from churn.preprocessing.contracts import ID_COLUMN, build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import fingerprint_ids, load_holdout
from churn.preprocessing.target import encode_target

logger = logging.getLogger("evaluate_holdout")

REPORT_PATH = PROJECT_ROOT / "reports" / "holdout_evaluation_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "holdout"

METRIC_LABELS = {
    "average_precision": "Average Precision",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall (sensitivity)",
    "specificity": "Specificity",
    "balanced_accuracy": "Balanced accuracy",
    "predicted_positive_rate": "Predicted positive rate",
    "accuracy": "Accuracy",
    "negative_predictive_value": "Negative predictive value",
    "false_positive_rate": "False positive rate",
    "false_negative_rate": "False negative rate",
    "brier_score": "Brier score",
    "log_loss": "Log loss",
}

CONFUSION_LABELS = {
    "true_negatives": "TN",
    "false_positives": "FP",
    "false_negatives": "FN",
    "true_positives": "TP",
}


def build_figures(
    metrics: HoldoutMetrics,
    target: np.ndarray,
    probability: np.ndarray,
    root: Path,
) -> dict[str, Path]:
    """Render the Phase 9D figures and return a name -> path mapping."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    cells = metrics.confusion
    paths: dict[str, Path] = {}

    paths["precision_recall"] = save_figure(
        plot_precision_recall_curve(
            target,
            probability,
            metrics.average_precision,
            metrics.actual_positive_rate,
            (metrics.recall, metrics.precision),
            metrics.threshold,
        ),
        figures_dir / "01_precision_recall_curve.png",
    )
    paths["roc"] = save_figure(
        plot_roc_curve(
            target,
            probability,
            metrics.roc_auc,
            (metrics.false_positive_rate, metrics.recall),
            metrics.threshold,
        ),
        figures_dir / "02_roc_curve.png",
    )
    paths["confusion"] = save_figure(
        plot_confusion_matrix(
            cells.true_negatives,
            cells.false_positives,
            cells.false_negatives,
            cells.true_positives,
            metrics.threshold,
        ),
        figures_dir / "03_confusion_matrix.png",
    )
    paths["distribution"] = save_figure(
        plot_probability_distribution(target, probability, metrics.threshold),
        figures_dir / "04_probability_distribution.png",
    )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _metric_table(results: HoldoutResults, group: str, with_ci: bool = False) -> str:
    header = "| Metric | Value | 95% CI |" if with_ci else "| Metric | Value |"
    divider = "| --- | --- | --- |" if with_ci else "| --- | --- |"
    lines = [header, divider]
    for name, value in results.metrics[group].items():
        label = METRIC_LABELS.get(name, name)
        if with_ci and name in results.confidence_intervals:
            interval = results.confidence_intervals[name]
            lines.append(
                f"| {label} | **{value:.4f}** | [{interval.ci_lower:.4f}, "
                f"{interval.ci_upper:.4f}] |"
            )
        elif with_ci:
            lines.append(f"| {label} | **{value:.4f}** | — |")
        else:
            lines.append(f"| {label} | {value:.4f} |")
    return "\n".join(lines)


def _confusion_table(results: HoldoutResults) -> str:
    cells = results.confusion_matrix
    lines = ["| Cell | Count | Meaning |", "| --- | --- | --- |"]
    meanings = {
        "true_negatives": "retained, correctly not flagged",
        "false_positives": "retained but flagged — a contact spent on a customer who would stay",
        "false_negatives": "churned and **not** flagged — a churner the system missed",
        "true_positives": "churned and flagged — the churners the system caught",
    }
    for name in CONFUSION_NAMES:
        lines.append(
            f"| **{CONFUSION_LABELS[name]}** ({name.replace('_', ' ')}) | "
            f"{int(cells[name]):,} | {meanings[name]} |"
        )
    lines.append(f"| **Total** | {int(cells['total']):,} | every holdout row, exactly once |")
    return "\n".join(lines)


def _interval_table(results: HoldoutResults) -> str:
    lines = [
        "| Metric | Estimate | CI lower | CI upper | Width | Valid | Degenerate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, interval in results.confidence_intervals.items():
        lines.append(
            f"| {METRIC_LABELS.get(name, name)} | **{interval.estimate:.4f}** | "
            f"{interval.ci_lower:.4f} | {interval.ci_upper:.4f} | "
            f"{interval.ci_upper - interval.ci_lower:.4f} | "
            f"{interval.n_valid_replications:,} | {interval.n_degenerate_replications} |"
        )
    return "\n".join(lines)


def _comparison_table(results: HoldoutResults) -> str:
    lines = [
        "| Metric | Holdout | Development | Source | Difference | Directly comparable |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for check in results.comparison_with_development["checks"]:
        lines.append(
            f"| {METRIC_LABELS.get(check['metric'], check['metric'])} | "
            f"**{check['holdout']:.4f}** | {check['development']:.4f} | "
            f"`{check['development_source']}` | {check['difference']:+.4f} | "
            f"{'yes' if check['directly_comparable'] else '**no**'} |"
        )
    return "\n".join(lines)


def _digest_table(results: HoldoutResults) -> str:
    lines = ["| Artefact | SHA-256 |", "| --- | --- |"]
    for artefact, digest in results.upstream_artefact_digests.items():
        lines.append(f"| `{artefact}` | `{digest[:16]}…` |")
    return "\n".join(lines)


def build_report(results: HoldoutResults, figures: dict[str, Path]) -> str:
    """Render the Phase 9D report. Every number comes from this evaluation.

    The report carries **no wall-clock metadata**, per the convention recorded in
    `CLAUDE.md`: it is a pure function of its inputs, so a rerun on an unchanged
    repository reproduces it byte for byte and any diff is a real change.
    """
    provenance = results.frozen_model_provenance
    policy = results.policy
    holdout = results.holdout
    cells = results.confusion_matrix
    discrimination = results.metrics["discrimination_primary"]
    operating = results.metrics["operating_point"]
    degenerate = sum(
        interval.n_degenerate_replications for interval in results.confidence_intervals.values()
    )
    # The trivial reference for accuracy: always predict the majority class.
    # Derived from the recorded cells, so it cannot disagree with them.
    majority_baseline = int(cells["actual_negatives"]) / int(cells["total"])

    # The two directly comparable checks, pulled out so section 8 can state each
    # finding in prose instead of leaving a reader to add up a table.
    outer = {
        check["metric"]: check
        for check in results.comparison_with_development["checks"]
        if "outer-fold mean" in check["development_source"]
    }
    outer_ap, outer_roc = outer["average_precision"], outer["roc_auc"]
    bootstrap = results.bootstrap
    threshold = policy["final_threshold"]
    ap = results.confidence_intervals["average_precision"]
    roc = results.confidence_intervals["roc_auc"]

    return f"""# Final Holdout Evaluation — Telco Customer Churn (Phase 9D)

- Generated by: `scripts/evaluate_holdout.py`
- Machine-readable record: `reports/experiments/holdout_results.json`
- Frozen in commit: `{provenance["freeze_commit_short"]}` — {provenance["freeze_commit_subject"]}
- Raw SHA-256: `{provenance["raw_sha256"]}`
- Holdout partition fingerprint: `{provenance["holdout_ids_sha256"]}`

> **This report carries no generation date.** It is a pure function of its
> inputs, so re-running the generator on an unchanged repository reproduces it
> byte for byte and any diff is a real change. Git records when it was produced.

> **This is the final test result.** Every other report in this repository shows
> cross-validated training-pool performance. The numbers here were computed on
> {holdout["n_samples"]:,} customers that took no part in any fit, any fold, any
> feature decision, the calibration gate or the threshold search.

---

## 1. Executive summary

**One** frozen model-and-decision-policy configuration was evaluated on the
previously untouched holdout, and no selection followed its results.

| Result | Value | 95% CI |
| --- | --- | --- |
| **Average Precision** | **{discrimination["average_precision"]:.4f}** | \
[{ap.ci_lower:.4f}, {ap.ci_upper:.4f}] |
| **ROC-AUC** | **{discrimination["roc_auc"]:.4f}** | [{roc.ci_lower:.4f}, {roc.ci_upper:.4f}] |
| F1 at the frozen threshold | {operating["f1"]:.4f} | \
[{results.confidence_intervals["f1"].ci_lower:.4f}, \
{results.confidence_intervals["f1"].ci_upper:.4f}] |
| Recall at the frozen threshold | {operating["recall"]:.4f} | \
[{results.confidence_intervals["recall"].ci_lower:.4f}, \
{results.confidence_intervals["recall"].ci_upper:.4f}] |
| Precision at the frozen threshold | {operating["precision"]:.4f} | \
[{results.confidence_intervals["precision"].ci_lower:.4f}, \
{results.confidence_intervals["precision"].ci_upper:.4f}] |

Read against the two references that make those numbers interpretable: a
no-skill ranker scores an Average Precision equal to the prevalence
({holdout["prevalence"]:.4f}) and a ROC-AUC of 0.500. The frozen operating point
flags {cells["predicted_positive_rate"]:.1%} of customers and catches
{operating["recall"]:.1%} of the churners among them.

Section 8 compares this with the development estimates. Section 10 states what
the result does not support.

## 2. Frozen evaluation protocol

**The model and the decision policy were frozen and committed before the holdout
was opened**, in commit `{provenance["freeze_commit_short"]}`
(`{provenance["freeze_commit"]}`). That ordering is the entire basis for calling
this an estimate of generalisation, so it is verified rather than asserted:

| Gate | Result |
| --- | --- |
| `build_split.py --verify` | the partition is reproducible from the raw bytes and the config |
| `freeze_model.py --verify` | 29 integrity checks on the freeze, all passing |
| Model fingerprint after load | `{provenance["model_fingerprint_sha256_after_load"][:32]}…` |
| Matches the policy | **{provenance["model_fingerprint_verified"]}** |
| Pipeline SHA-256 before | `{provenance["pipeline_sha256_before_evaluation"][:32]}…` |
| Pipeline SHA-256 after | `{provenance["pipeline_sha256_after_evaluation"][:32]}…` |
| Unchanged by the evaluation | **{provenance["pipeline_unchanged_by_evaluation"]}** |
| Refitted here | {provenance["refitted_here"]} |

The evaluation would have aborted before touching a single holdout row if any of
those had failed. The order was: verify the split, verify the freeze, load the
fitted pipeline, re-verify its fingerprint — **and only then** open the holdout.

### What was evaluated

| Property | Value |
| --- | --- |
| Estimator | `{policy["estimator"]}` |
| Hyperparameters | \
{", ".join(f"`{k}={v}`" for k, v in policy["hyperparameters"].items())} |
| Features | {policy["n_features"]} original, engineered {policy["engineered_features"] or "none"} |
| Calibration policy | **{policy["calibration_policy"]}** |
| Threshold policy | **{policy["threshold_policy"]}** |
| `final_threshold` | **{threshold}** |
| Decision rule | `{policy["decision_rule"]}` |
| Comparison | `{policy["comparison"]}` |
| Positive class | label `{policy["positive_class_label"]}`, column \
`{policy["positive_class_column"]}` (resolved from `classes_`) |
| Feature entry point | `{policy["feature_entry_point"]}` |
| Reopened here | {policy["reopened_here"]} |

`pipeline.predict()` was deliberately **not** used: it applies scikit-learn's
internal 0.5, not the frozen threshold, and would have measured a different
operating point than the one that was frozen.

### Upstream artefacts, listed explicitly

{_digest_table(results)}

Every pre-Phase-9D artefact this evaluation depends on is named individually
above, with its digest. That is deliberate: a directory-level check would be no
evidence at all here, because `reports/experiments/` **gained** a file in this
phase — `holdout_results.json`, the record of this evaluation — and an untracked
addition does not appear in a diff of tracked files. The frozen pipeline is not
in the table because it is covered more strongly, by the before-and-after digest
pair in the previous section.

## 3. Holdout dataset

| Property | Value |
| --- | --- |
| Loader | `{holdout["loader"]}` |
| Rows | **{holdout["n_samples"]:,}** |
| Churned (positive) | {holdout["n_positive"]:,} |
| Retained (negative) | {holdout["n_negative"]:,} |
| Prevalence | **{holdout["prevalence"]:.4f}** |
| Positive label | `{policy["positive_class_label"]}` = {holdout["positive_label_meaning"]} |
| Matches the frozen manifest | **{holdout["verification"]["matches_frozen_manifest"]}** |
| Identifier digest | `{holdout["verification"]["holdout_ids_sha256_observed"][:32]}…` |

The partition is regenerated rather than stored, so before any metric was
computed it was checked against the frozen manifest twice: the row count, and a
SHA-256 over the sorted identifiers — a digest of the *set* of rows, independent
of their order. {holdout["verification"]["note"]}.

**Status: {holdout["status"]}.**

## 4. Final discrimination performance

These are the **primary results**. Both are threshold-independent: they describe
how well the model ranks customers by risk, and no decision rule enters them.

{_metric_table(results, "discrimination_primary", with_ci=True)}

{_figure(figures["precision_recall"], "Precision-recall on the final test set")}

{_figure(figures["roc"], "ROC on the final test set")}

Average Precision is reported rather than "PR-AUC" because the two are not the
same computation: AP is a weighted sum of precisions at each threshold, while a
trapezoidal PR-AUC interpolates between operating points and is optimistically
biased. The interpretation reference for AP is the **prevalence**
({holdout["prevalence"]:.4f}) — a ranking with no signal converges to it — while
for ROC-AUC it is 0.500.

## 5. Frozen operating-point performance

Everything in this section is a consequence of one number: the threshold
**{threshold}**, chosen in Phase 9B on the training pool and not touched since.

{_metric_table(results, "operating_point", with_ci=True)}

### Auxiliary metrics

{_metric_table(results, "auxiliary")}

**Accuracy is auxiliary and was never a criterion.**

| Rule | Accuracy |
| --- | --- |
| Majority class — always predict "retained" | {majority_baseline:.4f} \
({int(cells["actual_negatives"]):,} of {int(cells["total"]):,}) |
| **The frozen rule** | **{results.metrics["auxiliary"]["accuracy"]:.4f}** |
| Difference | \
{100 * (results.metrics["auxiliary"]["accuracy"] - majority_baseline):+.1f} \
percentage points |

That comparison is reported for completeness and **decides nothing**. Accuracy
remains auxiliary because it is dominated by the majority class — most of what it
measures is the model agreeing with "retained" — and because it collapses false
positives and false negatives into one number that cannot show which way the rule
leans. A rule tuned for recall buys churners at the cost of false alarms;
accuracy charges it for the false alarms and credits the churners at the same
rate, so the trade-off that defines this operating point is invisible in it. The
metrics in section 5 are the ones that show it.

### Probability diagnostics

{_metric_table(results, "probability_diagnostics")}

Reported to close the thread Phase 9A opened, and **diagnostic only**: the
calibration policy is `{policy["calibration_policy"]}`, it was decided under
cross-validation before the holdout existed as data, and nothing here reopens it.
A calibration curve on this sample could not be used to justify recalibrating
without destroying the property that makes this evaluation worth anything.

## 6. Confusion matrix

{_confusion_table(results)}

| Property | Value |
| --- | --- |
| Orientation | {cells["orientation"]} |
| Cells sum to the sample | **{cells["cells_sum_to_n_samples"]}** \
({int(cells["total"]):,}) |
| Actual positive rate | {cells["actual_positive_rate"]:.4f} |
| Predicted positive rate | {cells["predicted_positive_rate"]:.4f} |

{_figure(figures["confusion"], "Confusion matrix on the final test set")}

{_figure(figures["distribution"], "Predicted probability by true class, final test set")}

### Prevalence against action rate

These two numbers answer different questions and the gap between them is **not an
error**:

- **{cells["actual_positive_rate"]:.4f}** is the share of holdout customers who
  actually churned — a property of the data;
- **{cells["predicted_positive_rate"]:.4f}** is the share the frozen rule flags —
  a property of the operating point.

{cells["rate_note"]}. A threshold below 0.5 on a model whose probabilities are
roughly calibrated will necessarily flag more customers than churn: that is what
buying recall costs, and it was chosen deliberately in Phase 9B.

Qualitatively, and with no monetary value attached: **FN are churners the system
did not identify**, and **FP are flagged customers who would have stayed**. Which
of the two matters more depends on costs this dataset does not contain — see
section 10.

## 7. Uncertainty

| Property | Value |
| --- | --- |
| Method | `{bootstrap["method"]}` |
| Resampling unit | {bootstrap["resampling_unit"]} |
| Replications | {bootstrap["n_bootstrap"]:,} |
| Confidence level | {bootstrap["confidence_level"]} |
| Percentiles | {", ".join(bootstrap["percentiles"])} |
| Seed | {bootstrap["seed"]} |
| Model refitted per replication | {bootstrap["model_refitted_per_replication"]} |
| Threshold reselected per replication | {bootstrap["threshold_reselected_per_replication"]} |
| Point estimate | {bootstrap["point_estimate_source"]} |

{_interval_table(results)}

**What the procedure is.** {bootstrap["n_bootstrap"]:,} resamples of the
{holdout["n_samples"]:,} holdout observations are drawn with replacement; the
existing probabilities are re-indexed — **no model is refitted**, so this
measures the uncertainty of the *evaluation*, not of the training; each metric is
recomputed on each resample at the **same frozen threshold**; and the interval is
the empirical 2.5th to 97.5th percentile of the valid values. The seed is fixed
and recorded, so every bound above is reproducible to the last digit.

**What it is not.** These are not significance tests. There is no null hypothesis
anywhere in this phase, no p-value, and nothing was compared against a threshold
of evidence. An interval says how far the estimate would move under resampling of
this population; it does not say the model is "significantly" anything.

{bootstrap["degenerate_replication_policy"].capitalize()}. In this run the
degenerate count was {degenerate} across all metrics — expected, since a resample
of {holdout["n_samples"]:,} rows at this prevalence is essentially certain to
contain both classes.

## 8. Comparison with development estimates

{_comparison_table(results)}

{results.comparison_with_development["note"]}.

**Average Precision and ROC-AUC are the honest comparison.** Both phases computed
them on probabilities with no threshold involved, on the same estimator and the
same preprocessing. The difference is a **generalization check**: the observed gap
between a cross-validated estimate on 5,634 training rows and a single-sample
estimate on {holdout["n_samples"]:,} held-out rows.

Stated plainly, and without softening either half:

- **ROC-AUC stayed very close** to the development estimate
  ({outer_roc["difference"]:+.4f}, {outer_roc["holdout"]:.4f} against
  {outer_roc["development"]:.4f});
- **Average Precision was lower on the holdout** ({outer_ap["difference"]:+.4f},
  {outer_ap["holdout"]:.4f} against {outer_ap["development"]:.4f}).

That AP difference is reported, not explained away. Two things are true about it
at once, and neither cancels the other. It is the larger of the two gaps and it
is the metric this project treated as primary throughout. And the holdout
estimate carries real sampling uncertainty — the interval in section 4 spans
[{ap.ci_lower:.4f}, {ap.ci_upper:.4f}] — so a single sample of
{holdout["n_samples"]:,} rows cannot localise the true value tightly enough for
the gap to be attributed with confidence to anything in particular.

**What that does not license.** An interval wide enough to overlap a development
estimate is **not evidence of equivalence**. Establishing equivalence needs a
pre-specified margin and a test designed for it; none was specified, none was
run, and reading overlap as agreement is the same error as reading
non-significance as no effect. What can be said is bounded and is the whole
claim: on this sample, discrimination was somewhat lower than development
cross-validation suggested for Average Precision and nearly unchanged for
ROC-AUC, with the uncertainty stated.

**The threshold-dependent rows are marked "not directly comparable", and the
reason is structural.** In Phase 9B each outer fold received a threshold selected
by F1-maximisation *inside its own training fold*, so the reported F1, precision
and recall estimate a **procedure** — "select a threshold, then apply it". This
phase applies **one fixed threshold** to one sample. Those are different
estimands. A gap between them can come from the change of protocol alone, and
reading it as model decay would be a mistake about what was measured, not an
observation about the model.

There is **no gate here** (`gate: {results.comparison_with_development["gate"]}`,
`cutoff: {results.comparison_with_development["cutoff"]}`). No cutoff exists in
this project that turns a difference into a pass or a failure, and inventing one
after seeing the numbers would be exactly the retrospective gate the protocol
forbids.

## 9. Interpretation

The frozen operating point is a deliberate trade, and the confusion matrix shows
its shape: of {int(cells["actual_positives"]):,} churners in the holdout the
system flags {int(cells["true_positives"]):,} and misses
{int(cells["false_negatives"]):,}; of {int(cells["actual_negatives"]):,} customers
who stayed it also flags {int(cells["false_positives"]):,}.

Restated as the two rates that matter operationally: the system catches
**{operating["recall"]:.1%} of churners**, and **{operating["precision"]:.1%} of
the customers it flags** actually churn. Moving the threshold up would raise the
second at the cost of the first, and down the reverse — that is what the curve in
section 4 shows, and the threshold was fixed at F1-maximum, which weighs the two
symmetrically.

**Symmetric weighting is a defensible default, not a business optimum.** F1 gives
precision and recall equal importance because this dataset provides no basis for
any other weighting: it has no customer lifetime value, no campaign cost and no
record of whether a retention offer works. A real programme would almost
certainly not weigh them equally, and the honest consequence is that this
operating point is a **research** one. Section 10 keeps that limitation attached
to the result rather than mentioned once and forgotten.

The discrimination result is the more transferable finding: it describes the
ranking, which is what any future threshold — chosen with real costs — would be
applied to.

## 10. Limitations

- **One dataset, one snapshot.** A single public telecommunications dataset with
  no time dimension. "Will churn" here means "resembles customers who had
  churned", and nothing establishes that the association holds in another market,
  another period or another operator.
- **A finite holdout.** {holdout["n_samples"]:,} customers,
  {holdout["n_positive"]:,} of them churners. The confidence intervals in section
  7 are the honest width of that limitation; the point estimates alone overstate
  the precision of what is known.
- **The intervals are percentile bootstrap intervals**, not exact coverage
  guarantees, and they inherit any bias the estimator itself has.
- **No external validation.** The model has never been evaluated on data from a
  different source, and no such data exists in this project.
- **No real cost of error.** No `cost_FN`, no `cost_FP`, no customer lifetime
  value, no revenue. The confusion matrix is **not** converted into money
  anywhere in this report, because any such number would be an assumption
  wearing the clothes of a result.
- **The threshold was optimised for F1, not for profit.** See section 9.
- **No causal claim.** The coefficients and the predictions describe association.
  Nothing here supports "changing this attribute would prevent this churn".
- **No subgroup analysis.** A slice battery improvised after opening the holdout
  invites opportunistic narratives. Fairness and monitoring slices belong to the
  later phases, which can pre-specify them.
- **Calibration was not reassessed as a decision.** The policy is
  `{policy["calibration_policy"]}`; the diagnostics in section 5 are reported,
  not acted on.
- **The analyst-exposure limitation from Phase 3 still applies** to the training
  pool, and therefore to everything that was selected on it.
- **The holdout is now consumed.** It has served its single purpose. Any decision
  taken from these numbers from here on would make future evaluations on this
  partition optimistic, and no honest replacement for it exists in this dataset.

## 11. Final status

```text
The final holdout has now been used for evaluation.
No model-selection decision was made from the holdout results.
```

| Flag | Value |
| --- | --- |
| `holdout_touched` | **{results.holdout_touched}** |
| `holdout_purpose` | **{results.holdout_purpose}** |
| `holdout_opened_after_model_freeze` | **{results.holdout_opened_after_model_freeze}** |
| `selection_allowed` | **{results.selection_allowed}** |
| `selection_after_holdout` | **{results.selection_after_holdout}** |
| `evaluations_performed` (frozen configurations) | {results.evaluations_performed} |
| `fit_calls_after_holdout_opened` | {results.methodology["fit_calls_after_holdout_opened"]} |
| `models_evaluated` | {results.methodology["models_evaluated"]} |
| `thresholds_evaluated` | {results.methodology["thresholds_evaluated"]} |
| `calibrations_performed` | {results.methodology["calibrations_performed"]} |
| `hyperparameter_searches` | {results.methodology["hyperparameter_searches"]} |
| `subgroup_slices_evaluated` | {results.methodology["subgroup_slices_evaluated"]} |
| `financial_costs_assumed` | {results.methodology["financial_costs_assumed"]} |
| `causal_claims` | {results.methodology["causal_claims"]} |

{results.methodology["statement"]}.

**What "one evaluation" counts, precisely.** The number that matters
scientifically is `{results.methodology["evaluated_configurations"]}`: one frozen
configuration was evaluated, and nothing was selected, adapted or reconsidered
after its results were seen. It is **not** a claim that the holdout file was
physically read exactly once — the number of physical executions
{results.methodology["physical_executions"]}.

Re-running the identical frozen evaluation cannot constitute a further attempt at
selection, because there is nothing to select between: `models_evaluated` stays
1, `thresholds_evaluated` stays 1, `calibrations_performed` stays 0, and every
run reproduces byte-identical artefacts.

The estimator, the calibration policy, the threshold and the decision rule are
unchanged from commit `{provenance["freeze_commit_short"]}`, and the pipeline file
is byte-identical before and after this evaluation. What comes next — global and
local explainability, the inference interface, the monitoring design — reads this
result; it does not revise the model in response to it.
"""


def evaluate() -> tuple[HoldoutResults, HoldoutMetrics, np.ndarray, np.ndarray]:
    """Run the final evaluation. Gates first, holdout second.

    The ordering in this function *is* the protocol. Every integrity gate runs,
    and the fitted pipeline is loaded and re-fingerprinted, before a single
    holdout row is read.
    """
    config = get_config()

    # --- gates, all of them, before the holdout exists as data ---------------
    manifest = verify_split_manifest(load_split_manifest())
    threshold_reference = load_threshold_results()
    calibration_reference = load_calibration_results()
    policy = load_decision_policy()

    checks, pipeline = verify_or_raise(policy, threshold_reference, calibration_reference)
    logger.info("Freeze verified: %d integrity checks passed.", len(checks))

    pipeline_path = PROJECT_ROOT / policy.artifacts.pipeline_path
    digest_before = file_digest(pipeline_path)
    loaded_fingerprint = model_fingerprint(pipeline)
    if loaded_fingerprint != policy.artifacts.model_fingerprint_sha256:
        raise IntegrityError(
            "The loaded pipeline is not the frozen model: fingerprint "
            f"{loaded_fingerprint} against {policy.artifacts.model_fingerprint_sha256} in the "
            "decision policy. The holdout was NOT opened."
        )
    logger.info("Frozen model verified after load: %s", loaded_fingerprint)

    # --- the holdout is opened here, and only here ---------------------------
    logger.warning(
        "Opening the final holdout to evaluate the one frozen configuration. No selection may "
        "follow these results."
    )
    holdout = load_holdout()
    partition = verify_holdout_partition(holdout, manifest, ID_COLUMN, fingerprint_ids)

    features = build_feature_matrix(holdout)
    target = np.asarray(encode_target(holdout[config.target.column]))
    probability = score_holdout(pipeline, features)

    threshold = policy.threshold.final_threshold
    metrics = compute_holdout_metrics(target, probability, threshold)
    logger.info(
        "Final holdout: AP %.4f, ROC-AUC %.4f, F1 %.4f, precision %.4f, recall %.4f",
        metrics.average_precision,
        metrics.roc_auc,
        metrics.f1,
        metrics.precision,
        metrics.recall,
    )

    intervals = bootstrap_intervals(
        target,
        probability,
        threshold,
        metrics,
        BOOTSTRAP_METRICS,
        N_BOOTSTRAP,
        CONFIDENCE_LEVEL,
        config.seed,
    )

    # --- descriptive comparison with the development estimates ---------------
    development = {record.experiment: record for record in threshold_reference.policies}
    calibration_c0 = next(
        record for record in calibration_reference.policies if record.experiment == "C0"
    )
    nested_caveat = (
        "NOT directly comparable: Phase 9B measured a nested procedure in which each outer fold "
        "used a threshold selected inside its own training fold, while this phase applies one "
        "fixed threshold to one sample. A difference can come from the change of protocol alone"
    )
    comparisons = [
        generalization_check(
            "average_precision",
            metrics.average_precision,
            development["D0"].mean["average_precision"],
            "phase9b outer-fold mean (threshold-independent)",
            True,
            "directly comparable: no threshold enters either estimate",
        ),
        generalization_check(
            "roc_auc",
            metrics.roc_auc,
            development["D0"].mean["roc_auc"],
            "phase9b outer-fold mean (threshold-independent)",
            True,
            "directly comparable: no threshold enters either estimate",
        ),
        generalization_check(
            "average_precision",
            metrics.average_precision,
            calibration_c0.outer_out_of_fold["average_precision"],
            "phase9a pooled out-of-fold (C0)",
            True,
            "directly comparable; pooled out-of-fold rather than a mean of folds",
        ),
        generalization_check(
            "roc_auc",
            metrics.roc_auc,
            calibration_c0.outer_out_of_fold["roc_auc"],
            "phase9a pooled out-of-fold (C0)",
            True,
            "directly comparable; pooled out-of-fold rather than a mean of folds",
        ),
        generalization_check(
            "f1",
            metrics.f1,
            development["D1"].mean["f1"],
            "phase9b nested D1 mean",
            False,
            nested_caveat,
        ),
        generalization_check(
            "precision",
            metrics.precision,
            development["D1"].mean["precision"],
            "phase9b nested D1 mean",
            False,
            nested_caveat,
        ),
        generalization_check(
            "recall",
            metrics.recall,
            development["D1"].mean["recall"],
            "phase9b nested D1 mean",
            False,
            nested_caveat,
        ),
    ]

    digests = {artefact: file_digest(PROJECT_ROOT / artefact) for artefact in UPSTREAM_ARTEFACTS}
    digest_after = file_digest(pipeline_path)

    results = build_results(
        metrics,
        intervals,
        comparisons,
        policy,
        manifest,
        partition,
        digests,
        loaded_fingerprint,
        digest_before,
        digest_after,
        config.seed,
    )
    write_results(results)

    failures = invariant_failures(load_results())
    if failures:
        raise HoldoutEvaluationError(
            "The record just written violates its own invariants:\n  " + "\n  ".join(failures)
        )
    logger.info("Record invariants verified.")
    return results, metrics, target, probability


def verify() -> int:
    """Check the existing Phase 9D record. **Read-only.** Returns an exit code.

    Reads the artefact and re-checks its invariants: the confusion matrix
    accounts for every row, the rates agree with the cells, every metric lies in
    its domain, each interval contains its own estimate, the model fingerprint
    was verified, the pipeline was unchanged, and the protocol flags still say
    that nothing was selected after the holdout was opened.

    It does not re-evaluate, does not open the holdout, and writes nothing.
    """
    results = load_results()
    failures = invariant_failures(results)

    for failure in failures:
        logger.error("INVARIANT FAILED: %s", failure)
    if failures:
        logger.error("%d invariant(s) failed.", len(failures))
        return 1

    logger.info("All record invariants hold.")
    logger.info(
        "Final holdout: AP %.4f, ROC-AUC %.4f at threshold %r",
        results.metrics["discrimination_primary"]["average_precision"],
        results.metrics["discrimination_primary"]["roc_auc"],
        results.policy["final_threshold"],
    )
    logger.info(
        "holdout_touched=%s, selection_after_holdout=%s",
        results.holdout_touched,
        results.selection_after_holdout,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the final evaluation, or verify an existing record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the existing evaluation record and change nothing",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        if arguments.verify:
            return verify()
        results, metrics, target, probability = evaluate()
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
        logger.error("FREEZE INTEGRITY FAILED — the holdout was not opened.\n%s", error)
        return 4
    except HoldoutMismatchError as error:
        logger.error("%s", error)
        return 5
    except HoldoutEvaluationError as error:
        logger.error("%s", error)
        return 6

    figures = build_figures(metrics, target, probability, PROJECT_ROOT)
    Path(REPORT_PATH).write_text(
        build_report(results, figures),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)

    for group, values in results.metrics.items():
        logger.info("%s: %s", group, values)
    logger.info(
        "Confusion: TN %d, FP %d, FN %d, TP %d",
        *(int(results.confusion_matrix[name]) for name in CONFUSION_NAMES),
    )
    logger.info("The final holdout has now been used. No selection followed it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
