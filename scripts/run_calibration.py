"""Run the Phase 9A calibration gate and write its report.

Usage (from the repository root):

    uv run python scripts/build_split.py --verify        # prove the split
    uv run python scripts/run_calibration.py

Writes:
    reports/experiments/calibration_results.json
    reports/calibration_report.md
    reports/figures/calibration/*.png

Only the frozen training pool is loaded. The holdout is never read.

One question is answered here: do the probabilities of the frozen Phase 8B
candidate need an explicit calibration layer? No threshold is selected, no
threshold-dependent metric is computed, the estimator and its hyperparameters
are frozen inputs, and no fitted object is persisted.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

from churn.analysis.plots import save_figure, use_project_style
from churn.config import PROJECT_ROOT, get_config
from churn.data.loader import RawDatasetMismatchError
from churn.modeling.calibration import (
    AVERAGE_PRECISION,
    BRIER,
    CALIBRATION_METRIC_NAMES,
    ECE,
    ELIGIBLE,
    ISOTONIC,
    ISOTONIC_CALIBRATION,
    LOG_LOSS,
    LOWER_IS_BETTER,
    NO_CALIBRATION,
    PROBABILITY_METRICS,
    RANKING_METRICS,
    ROC_AUC,
    SIGMOID,
    SIGMOID_CALIBRATION,
    UNCALIBRATED,
    CalibrationRun,
    build_calibrated_estimator,
    build_calibration_splitter,
    build_frozen_pipeline,
    eligibility,
    pair,
    run_calibration_cv,
    select_policy,
)
from churn.modeling.calibration_plots import plot_probability_distribution, plot_reliability_diagram
from churn.modeling.calibration_results import (
    RESULTS_PATH,
    CalibrationResults,
    FrozenCandidateMismatchError,
    build_eligibility_record,
    build_results,
    build_selection_record,
    file_digest,
    verify_frozen_candidate,
    verify_uncalibrated_reference,
    write_results,
)
from churn.modeling.comparison import collect_warnings
from churn.modeling.comparison_plots import plot_metric_by_family
from churn.modeling.comparison_results import (
    ArtefactReference,
    ReproductionCheck,
    ReproductionError,
)
from churn.modeling.comparison_results import load_results as load_comparison_results
from churn.modeling.evaluation import build_splitter
from churn.modeling.metrics import positive_prevalence
from churn.modeling.representation_results import load_results as load_representation_results
from churn.modeling.results import load_results as load_baseline_results
from churn.modeling.tuning_results import load_results as load_tuning_results
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target

logger = logging.getLogger("run_calibration")

REPORT_PATH = PROJECT_ROOT / "reports" / "calibration_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "calibration"

METRIC_LABELS = {
    BRIER: "Brier score",
    LOG_LOSS: "Log loss",
    ECE: "ECE",
    AVERAGE_PRECISION: "Average Precision",
    ROC_AUC: "ROC-AUC",
}

#: Upstream artefacts this phase reads. Their digests go into the record, so a
#: reader learns which bytes were actually anchored to, not merely which names.
UPSTREAM_ARTEFACTS = (
    "reports/split_manifest.json",
    "reports/experiments/baseline_results.json",
    "reports/experiments/feature_engineering_results.json",
    "reports/experiments/model_comparison_results.json",
    "reports/experiments/hgb_representation_results.json",
    "reports/experiments/tuning_results.json",
)

OOF_CAPTION = "Cross-validated outer out-of-fold probabilities on the training pool"


def _direction(metric: str) -> str:
    return "lower is better" if metric in LOWER_IS_BETTER else "higher is better"


def build_figures(
    results: CalibrationResults,
    target: np.ndarray,
    oof: dict[str, np.ndarray],
    root: Path,
) -> dict[str, Path]:
    """Render the Phase 9A figures and return a name -> path mapping."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    paths: dict[str, Path] = {}

    paths["reliability"] = save_figure(
        plot_reliability_diagram(
            target,
            oof,
            "Reliability — uncalibrated, sigmoid and isotonic, out of fold",
        ),
        figures_dir / "01_reliability_diagram.png",
    )

    for key, metric, filename in (
        ("brier", BRIER, "02_brier_fold_comparison.png"),
        ("log_loss", LOG_LOSS, "03_log_loss_fold_comparison.png"),
    ):
        paths[key] = save_figure(
            plot_metric_by_family(
                {
                    record.model: [fold.metrics[metric] for fold in record.folds]
                    for record in results.policies
                },
                results.policies[0].model,
                f"{METRIC_LABELS[metric]} ({_direction(metric)})",
                f"Outer-fold {METRIC_LABELS[metric]} — {_direction(metric)}",
            ),
            figures_dir / filename,
        )

    paths["distribution"] = save_figure(
        plot_probability_distribution(
            oof,
            {
                record.model: record.n_distinct_oof_probabilities
                for record in results.policies
                if record.model in oof
            },
            "Where each policy puts its probability mass, out of fold",
        ),
        figures_dir / "04_oof_probability_distribution.png",
    )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _absolute_table(results: CalibrationResults) -> str:
    lines = [
        "| Experiment | Policy | "
        + " | ".join(f"{METRIC_LABELS[metric]}" for metric in CALIBRATION_METRIC_NAMES)
        + " |",
        "| --- " * (len(CALIBRATION_METRIC_NAMES) + 2) + "|",
    ]
    for record in results.policies:
        cells = " | ".join(
            f"{record.mean[metric]:.4f} ± {record.std[metric]:.4f}"
            for metric in CALIBRATION_METRIC_NAMES
        )
        lines.append(f"| {record.experiment} | {record.label} | {cells} |")
    return "\n".join(lines)


def _fold_table(results: CalibrationResults, metric: str) -> str:
    lines = [
        "| Experiment | F1 | F2 | F3 | F4 | F5 | Mean ± SD |",
        "| --- " * 7 + "|",
    ]
    for record in results.policies:
        cells = " | ".join(f"{fold.metrics[metric]:.4f}" for fold in record.folds)
        lines.append(
            f"| {record.experiment} | {cells} | "
            f"**{record.mean[metric]:.4f}** ± {record.std[metric]:.4f} |"
        )
    return "\n".join(lines)


def _delta_table(results: CalibrationResults, pairs: tuple[tuple[str, str], ...]) -> str:
    records = {record.experiment: record for record in results.policies}
    lines = [
        "| Comparison | Metric | Direction | F1 | F2 | F3 | F4 | F5 | Mean | SD "
        "| Improved | Tied | Worsened |",
        "| --- " * 13 + "|",
    ]
    for experiment, reference in pairs:
        for metric in CALIBRATION_METRIC_NAMES:
            delta = records[experiment].paired_deltas[reference][metric]
            cells = " | ".join(f"{value:+.4f}" for value in delta.per_fold)
            lines.append(
                f"| {experiment} vs {reference} | {METRIC_LABELS[metric]} | "
                f"{'lower better' if delta.lower_is_better else 'higher better'} | {cells} | "
                f"**{delta.mean:+.4f}** | {delta.std:.4f} | {delta.folds_improved} | "
                f"{delta.folds_tied} | {delta.folds_worsened} |"
            )
    return "\n".join(lines)


def _oof_table(results: CalibrationResults) -> str:
    lines = [
        "| Experiment | Policy | "
        + " | ".join(METRIC_LABELS[metric] for metric in CALIBRATION_METRIC_NAMES)
        + " | Distinct values |",
        "| --- " * (len(CALIBRATION_METRIC_NAMES) + 3) + "|",
    ]
    for record in results.policies:
        cells = " | ".join(
            f"{record.outer_out_of_fold[metric]:.4f}" for metric in CALIBRATION_METRIC_NAMES
        )
        lines.append(
            f"| {record.experiment} | {record.label} | {cells} | "
            f"{record.n_distinct_oof_probabilities:,} |"
        )
    return "\n".join(lines)


def _eligibility_table(results: CalibrationResults) -> str:
    lines = [
        "| Experiment | Method | Mean ΔBrier | Folds Brier improved | Mean ΔLog loss "
        "| Mean ΔAP | Status |",
        "| --- " * 7 + "|",
    ]
    for record in results.eligibility:
        lines.append(
            f"| {record.experiment} | `{record.method}` | {record.mean_delta_brier:+.5f} | "
            f"{record.folds_brier_improved}/5 | {record.mean_delta_log_loss:+.5f} | "
            f"{record.mean_delta_average_precision:+.5f} | **{record.status}** |"
        )
    return "\n".join(lines)


def _reproduction_table(results: CalibrationResults) -> str:
    lines = [
        "| Experiment | Reproduces | Artefact | Tolerance | Largest difference | Reproduced |",
        "| --- " * 6 + "|",
    ]
    for check in results.reproduction:
        lines.append(
            f"| {check.experiment} | `{check.reference_key}` | `{check.reference_artefact}` "
            f"| {check.tolerance:.0e} | {check.max_absolute_difference:.1e} "
            f"| **{check.reproduced}** |"
        )
    return "\n".join(lines)


def _digest_table(results: CalibrationResults) -> str:
    lines = ["| Upstream artefact | SHA-256 |", "| --- | --- |"]
    for artefact, digest in results.upstream_artefact_digests.items():
        lines.append(f"| `{artefact}` | `{digest[:16]}…` |")
    return "\n".join(lines)


def _warning_table(results: CalibrationResults) -> str:
    if not results.warnings:
        return "No warning was raised during the run."
    lines = ["| Category | Origin | Count | Message |", "| --- " * 4 + "|"]
    for warning in results.warnings:
        message = warning.message.replace("|", "\\|")
        lines.append(f"| `{warning.category}` | `{warning.source}` | {warning.count} | {message} |")
    return "\n".join(lines)


def build_report(results: CalibrationResults, figures: dict[str, Path]) -> str:
    """Render the Phase 9A report. Every number comes from this run.

    The report carries **no wall-clock metadata**, per the convention recorded in
    `CLAUDE.md`: it is a pure function of its inputs, so a rerun on an unchanged
    repository reproduces it byte for byte and any diff is a real change.
    """
    records = {record.experiment: record for record in results.policies}
    c0 = records[UNCALIBRATED]
    outer, inner = results.outer_cross_validation, results.calibration_inner_cross_validation
    selection = results.selection
    estimator = results.frozen_estimator
    ece = results.ece_definition

    # "NONE" is a policy, not a record: when no calibrator is adopted the
    # probability carried forward is the uncalibrated one.
    calibrated = selection.selected_policy != NO_CALIBRATION
    selected = records[selection.selected_policy] if calibrated else c0

    # Pulled out of the f-string: a format specifier cannot span lines, and the
    # formatter is free to reflow anything left inline.
    isotonic = records[ISOTONIC_CALIBRATION]
    isotonic_distinct = isotonic.n_distinct_oof_probabilities
    isotonic_ap = isotonic.paired_deltas[UNCALIBRATED][AVERAGE_PRECISION]
    isotonic_ap_mean = isotonic_ap.mean
    isotonic_ap_worsened = isotonic_ap.folds_worsened

    outer_signature = (
        f"{outer['strategy']}(n_splits={outer['n_splits']}, shuffle={outer['shuffle']}, "
        f"random_state={outer['random_state']})"
    )
    inner_signature = (
        f"{inner['strategy']}(n_splits={inner['n_splits']}, shuffle={inner['shuffle']}, "
        f"random_state={inner['random_state']})"
    )
    selected_label = "no calibration layer" if not calibrated else selected.label

    return f"""# Calibration Gate — Telco Customer Churn (Phase 9A)

- Generated by: `scripts/run_calibration.py`
- Machine-readable record: `reports/experiments/calibration_results.json`
- Raw SHA-256: `{results.raw_sha256}`
- Training partition fingerprint: `{results.training_ids_sha256}`

> **This report carries no generation date.** It is a pure function of its
> inputs, so re-running the generator on an unchanged repository reproduces it
> byte for byte and any diff is a real change. Git records when it was produced.

> **Every number here is cross-validated on the training pool.** No holdout row
> was loaded, transformed, calibrated, predicted or measured. There is still no
> test result in this repository.

---

## 1. Objective

One question:

> **Do the probabilities produced by the frozen logistic regression need an
> explicit calibration layer?**

This phase measures **probability quality**. It does not select a threshold, it
does not reopen model selection, and it does not compute a single
threshold-dependent metric.

The distinction matters because a model can rank customers perfectly and still
be systematically wrong about *how likely* churn is. Average Precision and
ROC-AUC are invariant to any monotone transformation of the score; Brier score
and log loss are not. Every earlier phase in this project measured ranking. This
one measures whether a 0.7 means 70%.

## 2. Frozen candidate

| Property | Value |
| --- | --- |
| Estimator | `{estimator["estimator"]}` |
| Builder | `{estimator["builder"]}` |
| Selected in | `{estimator["selected_in"]}` |
| Hyperparameters | \
{", ".join(f"`{k}={v}`" for k, v in estimator["hyperparameters"].items())} |
| Reopened here | {estimator["reopened_here"]} |
| Features | {results.feature_set["n_features"]} original (3 numeric, 16 categorical) |
| Engineered | {results.engineered_features_used or "none"} |
| Preprocessing | \
{" → ".join(results.frozen_preprocessing["steps"])} |
| Transformed columns | {results.frozen_preprocessing["n_transformed_features"]} |
| `class_weight` | `{results.class_weight}` |
| Seed | {results.random_seed} |
| scikit-learn | {results.scikit_learn_version} |

Nothing in that table was chosen here. It is the output of Phase 8B, carried in
unchanged, and the tests assert it against the Phase 8B factory rather than
against a restatement of it.

{_digest_table(results)}

Digests of the artefacts this phase is anchored to. A reference by name says
which file was meant; a digest says which bytes were read.

## 3. Why calibration precedes threshold selection

A calibrator rewrites the probabilities. A threshold is a cut on them.

If the threshold were chosen first, adopting a calibrator afterwards would move
every probability underneath it, and the chosen cut would belong to a score that
no longer exists — 0.42 on raw scores and 0.42 on calibrated scores are
different operating points, selecting different customers, with different recall
and different precision. The threshold has to be chosen on **exactly the score
that will be used in production and later on the holdout**, so the calibration
policy has to be frozen before it.

That order is not a preference. It was fixed in section 20 of
`reports/tuning_report.md` and in `selection.phase9_sequence` of the Phase 8B
record before any Phase 9 work began: **A** calibration gate, **B** threshold
policy, **C** freeze, **D** the first and only holdout evaluation, **E** post-hoc
descriptive error analysis. This report is step A.

## 4. Protected holdout

The {results.n_training_rows:,} training rows are the entire universe of this
phase. `load_holdout()` is not called anywhere in the Phase 9A code, and an AST
audit fails the build if it appears. No holdout row took part in a fit, a
calibration, a metric, a plot, a reliability diagram or the choice of method.

## 5. Evaluation protocol

| Level | Splitter | Role |
| --- | --- | --- |
| Outer | `{outer_signature}` | evaluates the calibration policy; never fits a calibrator |
| Inner | `{inner_signature}` | fits the calibrator, inside one outer training fold |

The outer partition is the identical one used by Phases 5 to 8B, so every paired
delta here is a per-fold difference on the same folds every earlier number was
computed on.

**Cross-fitted, or it measures nothing.** A calibrator fitted on the scores its
own base estimator produced in-sample sees an optimistically distorted score
distribution — the model is overconfident on rows it was trained on — and learns
to correct a distortion that does not exist out of sample. So no row is ever
used to fit the estimator that scores it, to fit the calibrator that transforms
that score, and to evaluate the result:

```text
for each outer fold (the frozen 5-fold partition):
    inside the outer TRAIN only:
        inner 4-fold -> a cross-validated score for every outer-train row
        fit the calibrator on those out-of-sample scores
        refit the base pipeline on the WHOLE outer train
    predict the outer VALIDATION fold exactly once
```

The **complete pipeline** goes inside the calibrator, not just the classifier, so
the cleaner, the scaler and the encoder are refitted inside every inner
calibration fold.

**`ensemble={inner["ensemble"]}`, and that choice is what makes this experiment
answer the question asked.** It is pinned rather than left at the `"auto"`
default, so a future change of that default cannot silently alter the protocol.
The value matters:

- with `ensemble=False`, the inner folds are used **only** to produce
  out-of-sample scores for the calibrator, and the estimator that finally
  predicts is refitted on the entire outer training fold — the same rows C0 is
  fitted on. The arms then differ by the calibration layer **alone**;
- with `ensemble=True`, the calibrator keeps one (estimator, calibrator) pair per
  inner fold and averages four models each fitted on three quarters of the fold.
  That is a variance-reduction technique with its own effect on the Brier score,
  and a delta measured that way could not be attributed to calibration rather
  than to model averaging.

This phase exists to isolate the calibration layer, so `ensemble=False` is the
protocol and the only results reported here.

### Gate: the uncalibrated reference has not drifted

C0 is the frozen pipeline on the frozen outer folds — which is exactly what
Phase 8B's T0 was. The ranking metrics must therefore come out identical, and the
run aborts if they do not: a calibration delta measured against a drifted
reference would mean nothing.

{_reproduction_table(results)}

### Metrics, and which of them decide

| Group | Metrics | Role |
| --- | --- | --- |
| Probability quality | {", ".join(f"`{m}`" for m in PROBABILITY_METRICS)} | \
**decides** — Brier and log loss |
| Ranking guardrails | {", ".join(f"`{m}`" for m in RANKING_METRICS)} | watches only |
| Threshold-dependent | \
{results.metric_protocol["threshold_dependent_metrics_computed"] or "none — not computed"} | \
absent by design |

Brier score and log loss are **losses**: a negative delta is an improvement. That
is the opposite of every earlier phase in this project, so the paired comparison
carries an explicit direction flag rather than relying on a reader to remember.

**What those two losses are, and what they are not.** Both are *proper scoring
rules*: they are minimised by the true probabilities, which is why they can be
trusted to rank probabilistic quality. They are **not** measures of calibration
in isolation. Each decomposes into a calibration term and a
refinement/discrimination term, so a movement in either can come from
probabilities that became sharper rather than better aligned. No conclusion in
this report rests on Brier or log loss alone: the decision reads them **together
with** the reliability diagram, the ranking guardrails and the fold-to-fold
consistency of the effect.

A calibrator is never adopted because a ranking metric improved. The guardrails
exist to catch the opposite case — a calibrator that damaged the ranking. What
each method is expected to do to the ranking is stated in advance:

{
        chr(10).join(
            f"- **{key.replace('_', ' — ', 1)}**: {value}"
            for key, value in results.ranking_expectations.items()
        )
    }

Precision, recall, F1 and accuracy were not computed at all. They need a
threshold, this phase does not choose one, and a metric that is never computed
cannot accidentally decide anything.

### Expected Calibration Error

There is no canonical ECE — implementations differ in binning, weighting and
empty-bin handling — so the one used here is stated in full:

```text
{ece.formula}
```

| Property | Value |
| --- | --- |
| Bins | {ece.n_bins} |
| Strategy | `{ece.strategy}` |
| Empty bins | {ece.empty_bin_behaviour} |
| Duplicate edges | {ece.duplicate_edge_behaviour} |

{ece.caveat}.

**ECE is sensitive to how many rows back each bin, so the per-fold and pooled
figures in this report are not comparable with each other.** A per-fold ECE
splits roughly {results.n_training_rows // outer["n_splits"]:,} validation rows
into {ece.n_bins} quantile bins — about
{results.n_training_rows // outer["n_splits"] // ece.n_bins:,} customers per bin —
and the sampling noise in each bin's observed frequency inflates the average gap,
because ECE averages **absolute** differences and noise cannot cancel. The pooled
out-of-fold ECE in section 16 divides {results.n_training_rows:,} rows into the
same number of bins, so each gap rests on roughly
{results.n_training_rows // ece.n_bins:,} customers and the same model scores a
visibly lower ECE. Both numbers are correct; reading the difference between them
as a calibration effect would not be. This is a further reason ECE decides
nothing here.

## 6-8. The three policies

| Experiment | Method | What it is |
| --- | --- | --- |
{
        chr(10).join(
            f"| {method['experiment']} | `{method['method']}` | {method['description']} |"
            for method in results.calibration_methods
        )
    }

**Whether a logistic regression needs calibration is an empirical question, and
this phase is the experiment that answers it — for this dataset and this
protocol only.** A logistic regression can produce adequately calibrated
probabilities: it has a probabilistic structure and is fitted on log loss.
But that is a tendency, not a guarantee. Regularisation shrinks coefficients and
can compress probabilities towards the base rate; a misspecified model — omitted
interactions, a wrong functional form, unmodelled heterogeneity — can be
miscalibrated in specific regions of the score range; and the characteristics of
a particular dataset can produce miscalibration on their own. Nothing here is
concluded from the family. The conclusion is whatever the measurements below
support, scoped to this dataset and this protocol.

**Isotonic is more flexible, and that is not automatically an advantage.** A
sigmoid fit has two parameters; isotonic fits a free monotone step function. The
training pool holds {results.n_training_rows:,} rows, but no calibrator is fitted
on all of them — each one sees a single inner fold of a single outer training
fold, roughly a fifth of a fifth of the data. Flexibility that pays off on a
large sample can track the particular split it was given on a smaller one. That
is an empirical question, and section 12 answers it rather than assuming it.

{_absolute_table(results)}

{_figure(figures["brier"], "Outer-fold Brier score, lower is better")}

{_figure(figures["log_loss"], "Outer-fold log loss, lower is better")}

## 9. Brier score

{_fold_table(results, BRIER)}

The Brier score is the mean squared error of the probability. It is bounded, it
is a proper scoring rule, and it decomposes into calibration and refinement —
which is why it can improve when the probabilities are corrected even though the
ranking is untouched.

## 10. Log loss

{_fold_table(results, LOG_LOSS)}

Log loss is also proper, and unbounded. It punishes a confident mistake far
harder than Brier does, so the two disagree when a calibrator helps the bulk of
the distribution while hurting its tail. The eligibility rule treats that
disagreement as **inconclusive** rather than picking the metric that agrees.

## 11. Ranking guardrails

{_fold_table(results, AVERAGE_PRECISION)}

{_fold_table(results, ROC_AUC)}

Neither calibrator can reorder customers — both are monotone — but they are not
equally safe here, and the difference is structural.

**C1 (sigmoid)** is *strictly* monotone, so it preserves the ordering exactly.
Its AP and ROC-AUC against C0 are read as an **implementation guardrail**: a
material change would mean something other than calibration happened, and the run
would have to be investigated before being accepted. Small numerical differences
are legitimate and are reported rather than asserted away.

**C2 (isotonic)** is monotone but *not strictly increasing*: it maps many
distinct scores onto one plateau, and customers who had a strict ordering before
become tied afterwards. In this experiment it collapsed the out-of-fold score
from {c0.n_distinct_oof_probabilities:,} distinct values to
{isotonic_distinct:,}, and that **reduction in ranking resolution was
accompanied by a lower Average Precision**: mean delta {isotonic_ap_mean:+.6f}
against C0, worse in {isotonic_ap_worsened}/{outer["n_splits"]} outer folds.

That co-occurrence is what was measured. It is reported as a **trade-off of the
probabilistic representation** — fewer distinct probabilities to rank with — and
not as a calibration error, and no claim is made here about the internal tie
handling of the metric.

These rows are read as "the ranking survived", never as "calibration helped".

## 12. Paired outer-fold results

{
        _delta_table(
            results,
            (
                (SIGMOID_CALIBRATION, UNCALIBRATED),
                (ISOTONIC_CALIBRATION, UNCALIBRATED),
                (ISOTONIC_CALIBRATION, SIGMOID_CALIBRATION),
            ),
        )
    }

Improved, tied and worsened are three separate counts and do not have to sum the
way a two-way split would. "Improved" respects each metric's direction: for the
three losses it counts negative deltas.

The between-fold standard deviation is **not** a bar these deltas must clear. It
describes how one policy varies across partitions, not the uncertainty of a
difference between two policies measured on the same partitions. Five folds
cannot support a significance test and none was run.

## 13. Reliability diagrams

{_figure(figures["reliability"], "Reliability of the three policies, out of fold")}

{_figure(figures["distribution"], "Out-of-fold probability distribution and granularity")}

The diagonal is perfect calibration. A point **below** it means the policy
promised more churn than occurred in that band. Bins are
{results.reliability_diagram["n_bins"]} `{results.reliability_diagram["strategy"]}`
edges, declared before any diagram was drawn — bins chosen after seeing which
curve looks best would be a presentation decision, not a measurement.

The distinct-value counts in the second figure are not decoration. A step
function collapses a continuous score into plateaus, and a threshold chosen in
Phase 9B can only land between them.

## 14. Calibration selection rule

The rule, fixed before any Phase 9A number was read:

{chr(10).join(f"{index}. {step}" for index, step in enumerate(selection.rule, start=1))}

No minimum-gain cutoff exists anywhere in it (`minimum_gain_cutoff`:
{selection.minimum_gain_cutoff}). Every branch is exercised by tests on synthetic
inputs, so the branch the real numbers took is not the only one anybody checked.

{_eligibility_table(results)}

{
        chr(10).join(
            f"- **{record.experiment} ({record.method})** — {record.rationale}"
            for record in results.eligibility
        )
    }

## 15. Selected calibration policy

**Selected: {selection.selected_policy} — {selected_label}.**

{selection.rationale}

Fold-to-fold Brier stability: {
        ", ".join(f"`{key}` {value:.5f}" for key, value in selection.brier_stability.items())
    }.

## 16. Outer-OOF probability quality

Each of the {results.n_training_rows:,} training rows received exactly one
probability, from a policy whose **entire fitting and calibration** ran without
that row:

{_oof_table(results)}

These are cross-validated training-pool results. They are not test performance
and must never be relabelled as such. No per-customer probability was persisted.

## 17. Limitations

- **No holdout estimate.** Everything is training-pool cross-validation.
- **Five outer folds, no significance test.** The deltas and the eligibility
  labels describe direction and consistency, not statistical significance.
- **What was measured is the calibration *procedure***, not one fitted
  calibrator. Each outer fold fitted its own, and the policy carried forward is
  the procedure, not any particular one of them.
- **ECE is definition-dependent** and is reported as a supporting diagnostic. It
  decided nothing on its own.
- **Only two methods were evaluated.** Temperature scaling, beta calibration and
  anything else are out of scope by protocol, not by evidence.
- **Calibration was assessed on the whole population.** A model can be calibrated
  overall and miscalibrated inside a subgroup; that is a fairness-adjacent
  question this phase did not open.
- **`class_weight` stays `None` and remains deferred.** It changes the fit, so
  reopening it would reopen model selection.
- **No threshold was selected**, and no threshold-dependent metric exists here.
- **The analyst-exposure limitation from Phase 3 still applies.**
- **Snapshot classification, not forecasting.** The dataset has no time dimension.

## 18. Frozen probability policy for Phase 9B

| Property | Value |
| --- | --- |
| Estimator | `{estimator["estimator"]}` from `{estimator["builder"]}` |
| Hyperparameters | \
{", ".join(f"`{k}={v}`" for k, v in estimator["hyperparameters"].items())} |
| Calibration policy | **{selection.selected_policy}** — {selected_label} |
| Probability source | {selected.label} ({selected.experiment}) |
| Outer-OOF Brier | {selected.outer_out_of_fold[BRIER]:.4f} |
| Outer-OOF log loss | {selected.outer_out_of_fold[LOG_LOSS]:.4f} |
| Outer-OOF ECE | {selected.outer_out_of_fold[ECE]:.4f} |
| Distinct probabilities | {selected.n_distinct_oof_probabilities:,} |
| Threshold | {selection.threshold_status} |

{selection.phase9b_input}.

Phase 9B selects the decision threshold on the training pool, acting on exactly
this probability. The holdout stays untouched until the threshold is frozen too.

## 19. Warnings

{_warning_table(results)}

Nothing was suppressed. The modern logistic builder does not raise the historical
`penalty` removal warning.
"""


def _log_warnings(captured) -> None:
    for warning in captured:
        logger.warning(
            "%s x%d from %s: %s", warning.category, warning.count, warning.source, warning.message
        )


def main() -> int:
    """Run the calibration gate. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()

    try:
        verify_split_manifest(load_split_manifest())
        baseline_reference = load_baseline_results()
        comparison_reference = load_comparison_results()
        representation_reference = load_representation_results()
        tuning_reference = load_tuning_results()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1
    except RawDatasetMismatchError as error:
        logger.error("%s", error)
        return 2
    except SplitManifestMismatchError as error:
        logger.error("%s", error)
        return 3

    # Gate: what this phase calibrates must be what Phase 8B selected. Calibrating
    # a different estimator than the frozen one would produce a policy that
    # belongs to no model in this repository.
    try:
        frozen = verify_frozen_candidate(tuning_reference)
    except FrozenCandidateMismatchError as error:
        logger.error("%s", error)
        return 4
    logger.info("Frozen candidate verified against Phase 8B: %s", frozen)

    # From here on, only the training pool exists for this process.
    training = load_training_pool()
    features = build_feature_matrix(training)
    target = encode_target(training[config.target.column])
    labels = np.asarray(target)
    logger.info(
        "Training pool: %d rows, positive prevalence %.4f",
        len(features),
        positive_prevalence(labels),
    )

    outer = build_splitter()
    inner = build_calibration_splitter()

    runs: OrderedDict[str, CalibrationRun] = OrderedDict()
    checks: list[ReproductionCheck] = []
    with collect_warnings() as captured:
        runs[UNCALIBRATED] = run_calibration_cv(
            UNCALIBRATED,
            "Uncalibrated logistic regression",
            build_frozen_pipeline(),
            features,
            target,
            outer,
        )
        # Gate: the uncalibrated reference must be the Phase 8B T0 numbers. A
        # drifted reference would make every calibration delta uninterpretable.
        try:
            checks.append(verify_uncalibrated_reference(runs[UNCALIBRATED], tuning_reference))
        except ReproductionError as error:
            logger.error("%s", error)
            return 5

        runs[SIGMOID_CALIBRATION] = run_calibration_cv(
            SIGMOID_CALIBRATION,
            "Sigmoid-calibrated logistic regression",
            build_calibrated_estimator(SIGMOID, inner),
            features,
            target,
            outer,
            method=SIGMOID,
        )
        runs[ISOTONIC_CALIBRATION] = run_calibration_cv(
            ISOTONIC_CALIBRATION,
            "Isotonic-calibrated logistic regression",
            build_calibrated_estimator(ISOTONIC, inner),
            features,
            target,
            outer,
            method=ISOTONIC,
        )

        pair(runs[SIGMOID_CALIBRATION], runs[UNCALIBRATED])
        pair(runs[ISOTONIC_CALIBRATION], runs[UNCALIBRATED])
        pair(runs[ISOTONIC_CALIBRATION], runs[SIGMOID_CALIBRATION])

        # --- the pre-registered decision -------------------------------------
        n_folds = len(runs[UNCALIBRATED].folds)
        statuses: dict[str, str] = {}
        eligibility_records = []
        for experiment, method in (
            (SIGMOID_CALIBRATION, SIGMOID),
            (ISOTONIC_CALIBRATION, ISOTONIC),
        ):
            deltas = runs[experiment].deltas[UNCALIBRATED]
            status, rationale = eligibility(deltas, n_folds)
            statuses[experiment] = status
            eligibility_records.append(
                build_eligibility_record(experiment, method, status, rationale, deltas)
            )
            logger.info("%s (%s): %s", experiment, method, status)

        stability = {
            run.experiment: run.std(BRIER)
            for run in runs.values()
            if run.experiment != UNCALIBRATED
        }
        head_to_head = runs[ISOTONIC_CALIBRATION].deltas[SIGMOID_CALIBRATION]
        policy, rationale = select_policy(
            statuses,
            head_to_head if all(status == ELIGIBLE for status in statuses.values()) else None,
            stability,
            n_folds,
        )
        logger.info("Calibration policy: %s", policy)

    _log_warnings(captured)

    references = {
        "tuning": ArtefactReference(
            artefact="reports/experiments/tuning_results.json",
            schema_version=tuning_reference.schema_version,
            experiment=tuning_reference.experiment,
            raw_sha256=tuning_reference.raw_sha256,
            training_ids_sha256=tuning_reference.training_ids_sha256,
        ),
        "baseline": ArtefactReference(
            artefact="reports/experiments/baseline_results.json",
            schema_version=baseline_reference.schema_version,
            experiment=baseline_reference.experiment,
            raw_sha256=baseline_reference.raw_sha256,
            training_ids_sha256=baseline_reference.training_ids_sha256,
        ),
        "model_comparison": ArtefactReference(
            artefact="reports/experiments/model_comparison_results.json",
            schema_version=comparison_reference.schema_version,
            experiment=comparison_reference.experiment,
            raw_sha256=comparison_reference.raw_sha256,
            training_ids_sha256=comparison_reference.training_ids_sha256,
        ),
        "hgb_representation": ArtefactReference(
            artefact="reports/experiments/hgb_representation_results.json",
            schema_version=representation_reference.schema_version,
            experiment=representation_reference.experiment,
            raw_sha256=representation_reference.raw_sha256,
            training_ids_sha256=representation_reference.training_ids_sha256,
        ),
    }
    digests = {artefact: file_digest(PROJECT_ROOT / artefact) for artefact in UPSTREAM_ARTEFACTS}

    selected_label = "no calibration layer" if policy == NO_CALIBRATION else runs[policy].label
    results = build_results(
        runs,
        checks,
        eligibility_records,
        build_selection_record(policy, selected_label, rationale, stability),
        references,
        digests,
        tuning_reference.raw_sha256,
        tuning_reference.training_ids_sha256,
        labels,
        captured,
    )
    write_results(results)

    oof = {run.model: run.oof_probability for run in runs.values()}
    figures = build_figures(results, labels, oof, PROJECT_ROOT)
    Path(REPORT_PATH).write_text(
        build_report(results, figures),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)

    for experiment, run in runs.items():
        logger.info(
            "%s: Brier %.5f, log loss %.5f, AP %.4f",
            experiment,
            run.mean(BRIER),
            run.mean(LOG_LOSS),
            run.mean(AVERAGE_PRECISION),
        )
    logger.info("Calibration policy frozen for Phase 9B: %s", policy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
