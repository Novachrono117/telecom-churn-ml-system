"""Run the Phase 9B decision-threshold policy and write its report.

Usage (from the repository root):

    uv run python scripts/build_split.py --verify        # prove the split
    uv run python scripts/run_threshold.py

Writes:
    reports/experiments/threshold_results.json
    reports/threshold_report.md
    reports/figures/threshold/*.png

Only the frozen training pool is loaded. The holdout is never read.

One question is answered here: which operating threshold gets frozen so that the
churn probability of the frozen Phase 8B candidate becomes a decision? The
estimator, its hyperparameters, `class_weight`, the preprocessing, the feature
set and the calibration policy are all frozen inputs, no fitted object is
persisted, and no financial cost is invented.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

from churn.analysis.plots import save_figure, use_project_style
from churn.config import PROJECT_ROOT, get_config
from churn.data.loader import RawDatasetMismatchError
from churn.modeling.calibration_results import (
    FrozenCandidateMismatchError,
    verify_frozen_candidate,
)
from churn.modeling.calibration_results import load_results as load_calibration_results
from churn.modeling.comparison import collect_warnings
from churn.modeling.comparison_results import ArtefactReference, ReproductionError
from churn.modeling.evaluation import build_splitter
from churn.modeling.metrics import DEFAULT_THRESHOLD, positive_prevalence
from churn.modeling.threshold import (
    ACCURACY,
    AVERAGE_PRECISION,
    CONFUSION_NAMES,
    DECISION_METRIC_NAMES,
    DEFAULT_EXPERIMENT,
    F1,
    F1_POLICY,
    GUARDRAIL_METRIC_NAMES,
    NESTED_EXPERIMENT,
    PRECISION,
    PREDICTED_POSITIVE_RATE,
    RECALL,
    ROC_AUC,
    RankingInvarianceError,
    build_frozen_pipeline,
    build_threshold_inner_splitter,
    eligibility,
    evaluate_at_threshold,
    operating_curve,
    out_of_fold_probabilities,
    pair,
    recall_scenarios,
    run_threshold_nested_cv,
    select_f1_threshold,
    select_threshold_policy,
    sweep_thresholds,
    threshold_stability,
    top_k_analysis,
    verify_ranking_invariance,
)
from churn.modeling.threshold_plots import (
    plot_f1_vs_threshold,
    plot_precision_recall_vs_threshold,
    plot_threshold_stability,
    plot_top_k_capture,
)
from churn.modeling.threshold_results import (
    RESULTS_PATH,
    CalibrationPolicyMismatchError,
    ThresholdResults,
    build_eligibility_record,
    build_results,
    build_selection_record,
    build_stability_record,
    file_digest,
    verify_calibration_policy,
    verify_probability_source,
    verify_training_pool_probabilities,
    write_results,
)
from churn.modeling.tuning_results import load_results as load_tuning_results
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target

logger = logging.getLogger("run_threshold")

REPORT_PATH = PROJECT_ROOT / "reports" / "threshold_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "threshold"

METRIC_LABELS = {
    PRECISION: "Precision",
    RECALL: "Recall",
    F1: "F1",
    ACCURACY: "Accuracy",
    PREDICTED_POSITIVE_RATE: "Predicted positive rate",
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
    "reports/experiments/calibration_results.json",
)

#: Step of the operating-curve table rendered in the report, as a stride over the
#: uniform grid stored in the artefact. The artefact keeps every grid point.
CURVE_TABLE_STRIDE = 5


def build_figures(
    results: ThresholdResults,
    target: np.ndarray,
    probability: np.ndarray,
    root: Path,
) -> dict[str, Path]:
    """Render the Phase 9B figures and return a name -> path mapping."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    paths: dict[str, Path] = {}

    final_threshold = results.selection.final_threshold
    markers = {
        "default": (results.default_threshold, f"Default ({results.default_threshold:.3f})"),
        "final": (final_threshold, f"Frozen decision ({final_threshold:.4f})"),
    }
    # The curves are drawn over every candidate threshold, not the reporting grid
    # stored in the artefact: a curve is where the resolution is worth having.
    sweep = sweep_thresholds(target, probability)
    prevalence = results.training_positive_prevalence

    paths["precision_recall"] = save_figure(
        plot_precision_recall_vs_threshold(
            sweep,
            markers,
            prevalence,
            "Precision and recall against the decision threshold, out of fold",
        ),
        figures_dir / "01_precision_recall_vs_threshold.png",
    )
    paths["f1"] = save_figure(
        plot_f1_vs_threshold(
            sweep,
            markers,
            "F1 against the decision threshold, out of fold",
        ),
        figures_dir / "02_f1_vs_threshold.png",
    )
    paths["stability"] = save_figure(
        plot_threshold_stability(
            results.threshold_stability.per_fold,
            results.default_threshold,
            final_threshold,
            "Threshold selected inside each outer training fold",
        ),
        figures_dir / "03_outer_threshold_stability.png",
    )
    paths["top_k"] = save_figure(
        plot_top_k_capture(
            top_k_analysis(target, probability, prevalence=prevalence),
            prevalence,
            "Churners reached by a hypothetical contact budget, out of fold",
        ),
        figures_dir / "04_top_k_capture.png",
    )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _digest_table(results: ThresholdResults) -> str:
    lines = ["| Upstream artefact | SHA-256 |", "| --- | --- |"]
    for artefact, digest in results.upstream_artefact_digests.items():
        lines.append(f"| `{artefact}` | `{digest[:16]}…` |")
    return "\n".join(lines)


def _reproduction_table(results: ThresholdResults) -> str:
    lines = [
        "| Check | Reproduces | Artefact | Tolerance | Largest difference | Reproduced |",
        "| --- " * 6 + "|",
    ]
    for check in results.reproduction:
        lines.append(
            f"| {check.experiment} | `{check.reference_key}` | `{check.reference_artefact}` "
            f"| {check.tolerance:.0e} | {check.max_absolute_difference:.1e} "
            f"| **{check.reproduced}** |"
        )
    return "\n".join(lines)


def _outer_threshold_table(results: ThresholdResults) -> str:
    lines = [
        "| Outer fold | Rows searched | Candidates | Tied at max | Selected threshold "
        "| Inner-OOF F1 | Inner-OOF precision | Inner-OOF recall |",
        "| --- " * 8 + "|",
    ]
    for record in results.outer_thresholds:
        point = record.inner_out_of_fold_selection_metrics
        lines.append(
            f"| {record.fold} | {record.n_outer_train:,} | {record.n_candidates:,} | "
            f"{record.n_tied_within_tolerance} | **{record.threshold:.6f}** | {point.f1:.4f} | "
            f"{point.precision:.4f} | {point.recall:.4f} |"
        )
    return "\n".join(lines)


def _fold_table(results: ThresholdResults, metric: str) -> str:
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


def _confusion_table(results: ThresholdResults) -> str:
    lines = [
        "| Experiment | Fold | Threshold | TN | FP | FN | TP | Predicted positive rate |",
        "| --- " * 8 + "|",
    ]
    for record in results.policies:
        for fold in record.folds:
            cells = " | ".join(f"{fold.confusion[name]:,}" for name in CONFUSION_NAMES)
            lines.append(
                f"| {record.experiment} | {fold.fold} | {fold.threshold:.6f} | {cells} | "
                f"{fold.metrics[PREDICTED_POSITIVE_RATE]:.4f} |"
            )
    return "\n".join(lines)


def _guardrail_table(results: ThresholdResults) -> str:
    lines = [
        "| Metric | Experiment | F1 | F2 | F3 | F4 | F5 | Mean |",
        "| --- " * 8 + "|",
    ]
    for metric in GUARDRAIL_METRIC_NAMES:
        for record in results.policies:
            cells = " | ".join(f"{fold.metrics[metric]:.6f}" for fold in record.folds)
            lines.append(
                f"| {METRIC_LABELS[metric]} | {record.experiment} | {cells} | "
                f"{record.mean[metric]:.6f} |"
            )
    return "\n".join(lines)


def _delta_table(results: ThresholdResults) -> str:
    deltas = {
        record.experiment: record.paired_deltas
        for record in results.policies
        if record.paired_deltas
    }[NESTED_EXPERIMENT][DEFAULT_EXPERIMENT]
    lines = [
        "| Metric | F1 | F2 | F3 | F4 | F5 | Mean | SD | Higher | Tied | Lower |",
        "| --- " * 11 + "|",
    ]
    for metric in DECISION_METRIC_NAMES:
        delta = deltas[metric]
        cells = " | ".join(f"{value:+.4f}" for value in delta.per_fold)
        lines.append(
            f"| {METRIC_LABELS[metric]} | {cells} | **{delta.mean:+.4f}** | {delta.std:.4f} | "
            f"{delta.folds_higher} | {delta.folds_tied} | {delta.folds_lower} |"
        )
    return "\n".join(lines)


def _curve_table(results: ThresholdResults) -> str:
    points = results.operating_curve["points"]
    lines = [
        "| Threshold | Precision | Recall | F1 | Predicted positive rate |",
        "| --- " * 5 + "|",
    ]
    for point in points[::CURVE_TABLE_STRIDE]:
        lines.append(
            f"| {point['threshold']:.2f} | {point['precision']:.4f} | {point['recall']:.4f} | "
            f"{point['f1']:.4f} | {point['predicted_positive_rate']:.4f} |"
        )
    return "\n".join(lines)


def _scenario_table(results: ThresholdResults) -> str:
    lines = [
        "| Target | Attainable | Threshold | Precision | Recall | F1 | Predicted positive rate |",
        "| --- " * 7 + "|",
    ]
    for scenario in results.hypothetical_recall_scenarios["scenarios"]:
        if not scenario["attainable"]:
            lines.append(f"| Recall ≥ {scenario['target_recall']:.2f} | no | — | — | — | — | — |")
            continue
        lines.append(
            f"| Recall ≥ {scenario['target_recall']:.2f} | yes | {scenario['threshold']:.6f} | "
            f"{scenario['precision']:.4f} | {scenario['recall']:.4f} | {scenario['f1']:.4f} | "
            f"{scenario['predicted_positive_rate']:.4f} |"
        )
    return "\n".join(lines)


def _top_k_table(results: ThresholdResults) -> str:
    lines = [
        "| Capacity | Customers contacted | Churners reached | Recall | Precision | Lift "
        "| Probability at the cut |",
        "| --- " * 7 + "|",
    ]
    for level in results.top_k_capacity_analysis["levels"]:
        cut = level["probability_at_cut"]
        lines.append(
            f"| Top {level['fraction']:.0%} | {level['n_contacted']:,} | "
            f"{level['churners_captured']:,} | {level['recall']:.4f} | {level['precision']:.4f} | "
            f"×{level['lift']:.2f} | {'—' if cut is None else format(cut, '.4f')} |"
        )
    return "\n".join(lines)


def _warning_table(results: ThresholdResults) -> str:
    if not results.warnings:
        return "No warning was raised during the run."
    lines = ["| Category | Origin | Count | Message |", "| --- " * 4 + "|"]
    for warning in results.warnings:
        message = warning.message.replace("|", "\\|")
        lines.append(f"| `{warning.category}` | `{warning.source}` | {warning.count} | {message} |")
    return "\n".join(lines)


def build_report(results: ThresholdResults, figures: dict[str, Path]) -> str:
    """Render the Phase 9B report. Every number comes from this run.

    The report carries **no wall-clock metadata**, per the convention recorded in
    `CLAUDE.md`: it is a pure function of its inputs, so a rerun on an unchanged
    repository reproduces it byte for byte and any diff is a real change.
    """
    outer = results.outer_cross_validation
    inner = results.inner_threshold_cross_validation
    estimator = results.frozen_estimator
    policy = results.threshold_policy
    selection = results.selection
    stability = results.threshold_stability
    eligible = results.eligibility
    point = selection.full_training_out_of_fold_operating_point
    f1_max = selection.f1_max_on_full_training_out_of_fold
    d0 = {record.experiment: record for record in results.policies}[DEFAULT_EXPERIMENT]

    outer_signature = (
        f"{outer['strategy']}(n_splits={outer['n_splits']}, shuffle={outer['shuffle']}, "
        f"random_state={outer['random_state']})"
    )
    inner_signature = (
        f"{inner['strategy']}(n_splits={inner['n_splits']}, shuffle={inner['shuffle']}, "
        f"random_state={inner['random_state']})"
    )
    adopted = selection.selected_threshold_policy == F1_POLICY
    invariance = results.ranking_invariance

    return f"""# Decision Threshold Policy — Telco Customer Churn (Phase 9B)

- Generated by: `scripts/run_threshold.py`
- Machine-readable record: `reports/experiments/threshold_results.json`
- Raw SHA-256: `{results.raw_sha256}`
- Training partition fingerprint: `{results.training_ids_sha256}`

> **This report carries no generation date.** It is a pure function of its
> inputs, so re-running the generator on an unchanged repository reproduces it
> byte for byte and any diff is a real change. Git records when it was produced.

> **Every number here is cross-validated on the training pool.** No holdout row
> was loaded, transformed, predicted, thresholded or counted. There is still no
> test result in this repository.

---

## 1. Objective

One question:

> **Which operating threshold do we freeze, so that the churn probability of the
> frozen model becomes a binary decision before the holdout is opened?**

The answer has to be fixed *now*, on the training pool, because a threshold
chosen after seeing holdout results would make the holdout part of the selection
and destroy the only untouched estimate this project has.

**The selection metric was pre-registered**: `POLICY_F1`, the threshold
maximising the F1 score of the positive class. Precision and recall both matter
in churn — a missed churner is a lost customer, a false alarm is a wasted contact
— and F1 weighs them symmetrically without requiring a cost ratio.

> **F1-max is a research operating point, not a business-optimal threshold.**

That sentence is the scope of what this phase can claim, and section 16 explains
why nothing stronger is available.

## 2. Frozen probability policy

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
| Preprocessing | {" → ".join(results.frozen_preprocessing["steps"])} |
| Transformed columns | {results.frozen_preprocessing["n_transformed_features"]} |
| `class_weight` | `{results.class_weight}` |
| Calibration policy | **{results.calibration_policy}** (decided in \
`{results.probability_source["decided_in"]}`) |
| Probability source | `{results.probability_source["expression"]}` |
| Seed | {results.random_seed} |
| scikit-learn | {results.scikit_learn_version} |

Nothing in that table was chosen here. Phases 5 to 8B selected the estimator,
Phase 9A settled that its probabilities are used raw, and this phase changes
neither.

{_digest_table(results)}

Digests of the artefacts this phase is anchored to. A reference by name says
which file was meant; a digest says which bytes were read.

## 3. Why the threshold is a separate decision from the fit

A threshold is applied **after** the model is fitted. Moving it changes which
customers are flagged; it does not change a single coefficient. That is precisely
why it can be chosen at this stage without reopening anything.

`class_weight` is the instructive contrast. It also trades precision against
recall, and it is therefore tempting to treat the two as interchangeable. They
are not: `class_weight` changes the objective the model is fitted on, so a
different value produces a different model, which would need its own tuning, its
own calibration gate and its own comparison — model selection, reopened.
`class_weight` stays `{results.class_weight}` and stays closed. The threshold is
the knob that is legitimately still open here.

The order within Phase 9 follows the same logic and was fixed before any Phase 9
work began: **A** calibration gate, **B** threshold policy, **C** freeze, **D**
the first and only holdout evaluation, **E** post-hoc error analysis. Calibration
comes first because a calibrator rewrites the probabilities, and a threshold
chosen beforehand would belong to a score that stopped existing. Phase 9A
selected `{results.calibration_policy}`, so the cut in this report acts directly
on `{results.probability_source["expression"]}`.

## 4. Protected holdout

The {results.n_training_rows:,} training rows are the entire universe of this
phase. `load_holdout()` is not called anywhere in the Phase 9B code, and an AST
audit fails the build if it appears. No holdout row took part in a fit, a
probability, a threshold search, a metric, a curve, a scenario or a count.

`holdout_touched` in the artefact is `{results.holdout_touched}` and is asserted
by the tests rather than merely written.

## 5. Threshold-selection protocol

### The rule

{chr(10).join(f"{index}. {step}" for index, step in enumerate(policy["rule"], start=1))}

**Why the candidate set is the distinct probabilities.** With the rule
`probability >= threshold`, sweeping the threshold upwards changes the
predicted-positive set **exactly** when it crosses one of the observed
probabilities: any value strictly between two observed scores selects the same
customers as the higher of the two. The distinct probabilities are therefore the
complete set of materially different cuts, and a denser grid would only add
duplicates. `{results.default_threshold}` is added unconditionally so the search
can return "the default was already best" rather than being structurally unable
to say it.

### Tie-breaking

{chr(10).join(f"- {step}" for step in results.tie_breaking_rule)}

Preferring the candidate closest to `{results.default_threshold}` is a
**parsimony rule**, not a business conclusion: when the evidence does not
distinguish two operating points, the one that moves less from the default is
taken.

### Numeric precision

| Property | Value |
| --- | --- |
| Metrics rounded to | {results.numeric_precision["metrics_rounded_to"]} decimals |
| Thresholds rounded | {results.numeric_precision["thresholds_rounded"]} |
| Probabilities rounded before the search | \
{results.numeric_precision["probabilities_rounded_before_search"]} |

{results.numeric_precision["note"]}.

### Not `TunedThresholdClassifierCV`

scikit-learn ships a threshold tuner. It was deliberately not used. The candidate
set, the decision rule, the objective and the tie-breaking all had to be visible
and testable rather than delegated to defaults that could change between
versions, and the eligibility rule needs the *procedure* under nested evaluation
rather than a fitted object. An AST audit fails the build if the class appears in
Phase 9B code.

## 6. D0 — the default policy

| Property | Value |
| --- | --- |
| Experiment | `{d0.experiment}` |
| Policy | `{d0.policy}` |
| Threshold | {results.default_threshold} |
| Threshold source | {d0.threshold_source} |

`{results.default_threshold}` is the conventional cut of a probabilistic
classifier and **has not been optimised for anything**. It is the baseline
against which the selection procedure has to prove itself: without it, "we tuned
the threshold and F1 improved" would be a claim with nothing behind it.

## 7. D1 — the nested F1-max policy

| Level | Splitter | Role |
| --- | --- | --- |
| Outer | `{outer_signature}` | {outer["role"]} |
| Inner | `{inner_signature}` | {inner["role"]} |

The outer partition is the identical one used by Phases 5 to 9A, so every paired
delta here is a per-fold difference on the same folds every earlier number was
computed on.

**Choosing a threshold on a set of scores and then reporting the F1 it achieves
on those same scores measures nothing** — the threshold was fitted to them. So
the procedure is evaluated nested:

```text
for each outer fold (the frozen 5-fold partition):
    inside the outer TRAIN only:
        inner 4-fold -> an out-of-sample probability for every outer-train row
        POLICY_F1 on those inner-OOF probabilities -> this fold's threshold
        refit the frozen pipeline on the WHOLE outer train
    predict the outer VALIDATION once, cut it at that threshold
```

No outer-validation row takes part in a fit, in the probabilities the search
reads, or in the F1 the search maximises. What that estimates is **the
threshold-selection procedure**, not one threshold.

### Gate: the probability source has not drifted

The outer loop fits the frozen pipeline on each outer training fold and predicts
the outer validation fold, which is exactly what Phase 9A's C0 did. The
threshold-independent metrics must therefore come out identical, and the run
aborts if they do not.

{_reproduction_table(results)}

The second row is the identity check behind section 11: the training-pool
out-of-fold vector used to freeze the threshold was recomputed independently and
compared against the nested outer vector.

### Gate: D0 and D1 differ by the cut and by nothing else

Both arms read the **same** outer probability vector. Average Precision and
ROC-AUC are invariant to any threshold, so they must be *identical* — not close.

{_guardrail_table(results)}

Largest absolute difference across all folds and both metrics:
**{invariance["max_absolute_difference"]:.1e}** — and {invariance["gate"]}.

## 8. Thresholds selected inside each outer fold

{_outer_threshold_table(results)}

The F1, precision and recall columns are **inner out-of-fold selection
metrics** — the value the search maximised on rows inside the outer training
fold. They are not performance, and the outer-fold results are in section 9.

## 9. Paired D1 vs D0 on the outer folds

### F1 — the selection metric

{_fold_table(results, F1)}

### Precision

{_fold_table(results, PRECISION)}

### Recall

{_fold_table(results, RECALL)}

### Accuracy — auxiliary

{_fold_table(results, ACCURACY)}

At a {results.training_positive_prevalence:.1%} positive rate a rule that never
fires already reaches about {1 - results.training_positive_prevalence:.1%}
accuracy. It cannot rank operating points and it decided nothing.

### Predicted positive rate

{_fold_table(results, PREDICTED_POSITIVE_RATE)}

Neither direction of this row is good or bad on its own: it is how many customers
the rule would put on a contact list.

### Confusion matrices

{_confusion_table(results)}

### Paired deltas, D1 − D0, on the same outer folds

{_delta_table(results)}

"Higher", "tied" and "lower" are three separate counts and do not have to sum the
way a two-way split would. They are named by **direction**, not by merit: for F1,
precision and recall a higher value is better; for the predicted positive rate it
is neither.

The between-fold standard deviation is **not** a bar these deltas must clear. It
describes how one policy varies across partitions, not the uncertainty of a
difference between two policies measured on the same partitions. Five folds
cannot support a significance test and none was run.

## 10. Eligibility decision

The rule, fixed before any Phase 9B number was read:

{chr(10).join(f"{index}. {step}" for index, step in enumerate(eligible.rule, start=1))}

No minimum-gain cutoff exists anywhere in it (`minimum_gain_cutoff`:
{eligible.minimum_gain_cutoff}). Every branch is exercised by tests on synthetic
inputs, so the branch the real numbers took is not the only one anybody checked.

| Quantity | Value |
| --- | --- |
| Mean paired ΔF1 | {eligible.mean_delta_f1:+.5f} |
| Folds F1 improved strictly | {eligible.folds_f1_improved}/{outer["n_splits"]} |
| Folds tied | {eligible.folds_f1_tied} |
| Folds worsened | {eligible.folds_f1_worsened} |
| Required | mean > 0 and ≥ {eligible.min_improved_folds}/{outer["n_splits"]} strict improvements |
| **Status** | **{eligible.status}** |

{eligible.rationale}

## 11. Final threshold selection

**Selected policy: {selection.selected_threshold_policy}.**

{selection.rationale}

| Property | Value |
| --- | --- |
| `final_threshold` | **{selection.final_threshold}** |
| Decision rule | `{selection.decision_rule}` |
| Optimised | {selection.optimised} |
| F1-max on the full training-pool OOF | {f1_max.threshold} |
| F1-max adopted | {selection.f1_max_adopted} |

{
        "The threshold was fixed by applying POLICY_F1 once to the out-of-fold "
        "probabilities of the whole training pool. That application is a **freeze**, "
        "not a new performance estimate: the honest estimate of the procedure is the "
        "nested result in section 9."
        if adopted
        else "The F1-max threshold on the full training-pool out-of-fold probabilities is "
        "reported above **for transparency only**. It was not adopted: the procedure that "
        "would have produced it did not clear the eligibility rule, and reporting it while "
        "hiding it would be worse than reporting it plainly. The frozen threshold is the "
        "unoptimised default."
    }

### Full-training out-of-fold operating point

| Metric | Value |
| --- | --- |
| Threshold | {point.threshold} |
| F1 | {point.f1:.4f} |
| Precision | {point.precision:.4f} |
| Recall | {point.recall:.4f} |
| Accuracy | {point.accuracy:.4f} |
| Predicted positive rate | {point.predicted_positive_rate:.4f} |
| TN | {point.true_negatives:,} |
| FP | {point.false_positives:,} |
| FN | {point.false_negatives:,} |
| TP | {point.true_positives:,} |

> **These are {selection.operating_point_label}.**

They are computed on the training pool, on the same rows any threshold search
read, and they are optimistically biased by construction. Calling them "final
performance" would be exactly the error this whole phase structure exists to
avoid. **Performance will exist only after the holdout is evaluated in Phase 9D.**

## 12. Precision-recall operating curve

{_figure(figures["precision_recall"], "Precision and recall against the threshold, out of fold")}

{_figure(figures["f1"], "F1 against the threshold, out of fold")}

Tabulated on the uniform reporting grid stored in the artefact
({results.operating_curve["n_points"]} points; every
{CURVE_TABLE_STRIDE / (results.operating_curve["n_points"] - 1):.2f} shown here):

{_curve_table(results)}

{results.operating_curve["note"]}.

Above roughly 0.85 no customer is predicted positive at all. Precision is
undefined there (0/0) and is reported as `0.0000` under `zero_division=0`, the
same convention every earlier phase used: a rule that never fires has no
precision to speak of.

## 13. Threshold stability

| Statistic | Value |
| --- | --- |
| Per fold | {", ".join(f"{value:.6f}" for value in stability.per_fold)} |
| Mean | {stability.mean:.6f} |
| Median | {stability.median:.6f} |
| SD | {stability.std:.6f} |
| Min | {stability.minimum:.6f} |
| Max | {stability.maximum:.6f} |
| Spread | {stability.spread:.6f} |
| Distinct values | {stability.n_distinct} |

{_figure(figures["stability"], "Threshold selected inside each outer training fold")}

Role: {stability.role}.

A spread of {stability.spread:.4f} across five partitions of the same pool is a
statement about how much the chosen cut depends on which rows the search happened
to see. It is reported as a limitation in section 17, not converted into a new
criterion.

## 14. Hypothetical recall scenarios

{_scenario_table(results)}

> **Status: {results.hypothetical_recall_scenarios["status"]}.**
> `decisional: {results.hypothetical_recall_scenarios["decisional"]}`.

{results.hypothetical_recall_scenarios["note"]}.

Rule: {results.hypothetical_recall_scenarios["rule"]}.

## 15. Top-k capacity analysis

{_top_k_table(results)}

{_figure(figures["top_k"], "Churners reached by a hypothetical contact budget, out of fold")}

> **Status: {results.top_k_capacity_analysis["status"]}.**
> `decisional: {results.top_k_capacity_analysis["decisional"]}`.

| Property | Value |
| --- | --- |
| Ordering | {results.top_k_capacity_analysis["ordering"]} |
| Customers contacted | `{results.top_k_capacity_analysis["n_contacted_rule"]}` |
| Lift baseline | {results.top_k_capacity_analysis["lift_baseline"]} \
({results.training_positive_prevalence:.4f}) |

{results.top_k_capacity_analysis["note"]}.

This is the **ranking** view of the same probabilities: it asks how many churners
a fixed budget reaches, not where to cut. A capacity constraint and a threshold
answer different questions, and this table did not select `final_threshold`.

## 16. Why no financial optimum is claimed

The cost-optimal threshold is the one minimising

```text
{results.cost_framework.formula}
```

and that function depends **entirely** on the ratio `cost_FN / cost_FP`. This
dataset contains neither quantity: it has no customer lifetime value, no
retention-campaign cost, no margin, no discount rate and no record of what a
retention offer would cost or whether it would work.

| Field | Value |
| --- | --- |
| `cost_false_negative` | {results.cost_framework.cost_false_negative} |
| `cost_false_positive` | {results.cost_framework.cost_false_positive} |
| `optimal_threshold_computed` | {results.cost_framework.optimal_threshold_computed} |

{results.cost_framework.note}.

The conceptual conclusion is worth stating because it is the actionable one: **if
real cost figures existed, they — not F1 — would set the threshold**, and the
curve in section 12 is exactly the object a cost analysis would be applied to.
Producing a number here by assuming a ratio would give an economic-looking answer
whose entire content is the assumption.

## 17. Limitations

- **No holdout estimate.** Everything is training-pool cross-validation. Nothing
  in this report is test performance.
- **The full-training out-of-fold operating point is optimistically biased.** It
  is a selection artefact, reported as one.
- **F1 is a choice, not a discovery.** It weighs precision and recall equally
  because no evidence supports any other weighting here — not because equal
  weighting is correct for a real retention programme.
- **Five outer folds, no significance test.** The deltas and the eligibility
  label describe direction and consistency, not statistical significance.
- **The threshold moved across folds** (spread {stability.spread:.4f}). The
  procedure, not a specific value, is what the nested design evaluated.
- **The recall scenarios and the top-k table are hypothetical** and could not
  select anything here.
- **Ties in the top-k ranking are broken by row position**, which is arbitrary
  though deterministic; with a near-continuous score it affects few customers.
- **`class_weight` stays `{results.class_weight}` and remains closed.** It changes
  the fit, so reopening it would reopen model selection.
- **Calibration stays `{results.calibration_policy}`.** The threshold acts on the
  raw probability, and it would have to be reselected if that ever changed.
- **The analyst-exposure limitation from Phase 3 still applies.**
- **Snapshot classification, not forecasting.** The dataset has no time dimension,
  so "will churn" here means "resembles customers who had churned".

## 18. Frozen decision policy for the final holdout evaluation

| Property | Value |
| --- | --- |
| Estimator | `{estimator["estimator"]}` from `{estimator["builder"]}` |
| Hyperparameters | \
{", ".join(f"`{k}={v}`" for k, v in estimator["hyperparameters"].items())} |
| `class_weight` | `{results.class_weight}` |
| Calibration policy | **{results.calibration_policy}** |
| Probability source | `{results.probability_source["expression"]}` |
| Threshold policy | **{selection.selected_threshold_policy}** |
| `final_threshold` | **{selection.final_threshold}** |
| Decision rule | `{selection.decision_rule}` |

{selection.phase9c_input}.

Everything needed to turn a customer into a decision is now frozen. The holdout
stays untouched until Phase 9C has frozen this policy and Phase 9D evaluates it
exactly once.

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
    """Run the threshold policy phase. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()

    try:
        verify_split_manifest(load_split_manifest())
        tuning_reference = load_tuning_results()
        calibration_reference = load_calibration_results()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1
    except RawDatasetMismatchError as error:
        logger.error("%s", error)
        return 2
    except SplitManifestMismatchError as error:
        logger.error("%s", error)
        return 3

    # Gate: the estimator this phase thresholds must be the Phase 8B candidate,
    # and the probability it emits must be the one Phase 9A froze.
    try:
        frozen = verify_frozen_candidate(tuning_reference)
        calibration_policy = verify_calibration_policy(calibration_reference)
    except FrozenCandidateMismatchError as error:
        logger.error("%s", error)
        return 4
    except CalibrationPolicyMismatchError as error:
        logger.error("%s", error)
        return 5
    logger.info("Frozen candidate %s, calibration policy %s", frozen, calibration_policy)

    # From here on, only the training pool exists for this process.
    training = load_training_pool()
    features = build_feature_matrix(training)
    target = encode_target(training[config.target.column])
    labels = np.asarray(target)
    prevalence = positive_prevalence(labels)
    logger.info("Training pool: %d rows, positive prevalence %.4f", len(features), prevalence)

    outer = build_splitter()
    inner = build_threshold_inner_splitter()

    with collect_warnings() as captured:
        evaluation = run_threshold_nested_cv(
            build_frozen_pipeline(), features, target, outer, inner, DEFAULT_THRESHOLD
        )
        pair(evaluation.selected_run, evaluation.default_run)

        # The procedure is judged before the threshold that goes forward is even
        # computed: the freeze below must not be able to influence the decision.
        delta_f1 = evaluation.selected_run.deltas[DEFAULT_EXPERIMENT][F1]
        n_folds = len(evaluation.default_run.folds)
        status, eligibility_rationale = eligibility(delta_f1, n_folds)
        policy, selection_rationale = select_threshold_policy(status, DEFAULT_THRESHOLD)
        logger.info("Eligibility: %s -> policy %s", status, policy)

        # Independent pass over the same folds, so "the training-pool OOF vector
        # IS the nested outer vector" is demonstrated rather than asserted.
        recomputed, _ = out_of_fold_probabilities(
            build_frozen_pipeline(), features, labels, build_splitter()
        )
        f1_max = select_f1_threshold(labels, evaluation.oof_probability, DEFAULT_THRESHOLD)

    _log_warnings(captured)

    try:
        invariance = verify_ranking_invariance(evaluation)
        checks = [
            verify_probability_source(evaluation.default_run, calibration_reference),
            verify_training_pool_probabilities(evaluation.oof_probability, recomputed),
        ]
    except RankingInvarianceError as error:
        logger.error("%s", error)
        return 6
    except ReproductionError as error:
        logger.error("%s", error)
        return 7

    final_threshold = f1_max.threshold if policy == F1_POLICY else DEFAULT_THRESHOLD
    operating_point = evaluate_at_threshold(labels, evaluation.oof_probability, final_threshold)
    logger.info("Frozen threshold: %.10f (policy %s)", final_threshold, policy)

    references = {
        "calibration": ArtefactReference(
            artefact="reports/experiments/calibration_results.json",
            schema_version=calibration_reference.schema_version,
            experiment=calibration_reference.experiment,
            raw_sha256=calibration_reference.raw_sha256,
            training_ids_sha256=calibration_reference.training_ids_sha256,
        ),
        "tuning": ArtefactReference(
            artefact="reports/experiments/tuning_results.json",
            schema_version=tuning_reference.schema_version,
            experiment=tuning_reference.experiment,
            raw_sha256=tuning_reference.raw_sha256,
            training_ids_sha256=tuning_reference.training_ids_sha256,
        ),
    }
    digests = {artefact: file_digest(PROJECT_ROOT / artefact) for artefact in UPSTREAM_ARTEFACTS}

    results = build_results(
        evaluation,
        evaluation.selections,
        checks,
        invariance,
        build_eligibility_record(status, eligibility_rationale, delta_f1),
        build_selection_record(
            policy,
            final_threshold,
            selection_rationale,
            operating_point,
            f1_max.operating_point,
        ),
        build_stability_record(threshold_stability(evaluation.selections)),
        recall_scenarios(labels, evaluation.oof_probability),
        top_k_analysis(labels, evaluation.oof_probability, prevalence=prevalence),
        operating_curve(labels, evaluation.oof_probability),
        references,
        digests,
        calibration_reference.raw_sha256,
        calibration_reference.training_ids_sha256,
        labels,
        captured,
    )
    write_results(results)

    figures = build_figures(results, labels, evaluation.oof_probability, PROJECT_ROOT)
    Path(REPORT_PATH).write_text(
        build_report(results, figures),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)

    for record in results.policies:
        logger.info(
            "%s: mean F1 %.4f, precision %.4f, recall %.4f",
            record.experiment,
            record.mean[F1],
            record.mean[PRECISION],
            record.mean[RECALL],
        )
    logger.info(
        "Threshold policy frozen for Phase 9C: %s at %.10f",
        policy,
        results.selection.final_threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
