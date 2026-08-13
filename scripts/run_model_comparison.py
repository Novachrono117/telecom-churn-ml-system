"""Run the Phase 7 model-family comparison and write its report.

Usage (from the repository root):

    uv run python scripts/build_split.py --verify        # prove the split
    uv run python scripts/run_model_comparison.py

Writes:
    reports/experiments/model_comparison_results.json
    reports/model_comparison_report.md
    reports/figures/model_comparison/*.png

Only the frozen training pool is loaded. The holdout is never read. The folds,
the representation, the metrics and the 0.5 diagnostic rule are frozen at their
Phase 5 values: this script varies one thing, the estimator — and then, in a
separate secondary analysis, one feature.

Two gates stop the run before anything is interpreted: M0 must reproduce the
Phase 5 baseline, and S0 must reproduce E3 of the Phase 6 ablation.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np

from churn.analysis.plots import save_figure, use_project_style
from churn.config import PROJECT_ROOT, get_config
from churn.data.loader import RawDatasetMismatchError
from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC
from churn.features.groups import CONTRACT_TENURE
from churn.features.results import load_results as load_ablation_results
from churn.modeling.comparison import (
    FamilyRun,
    collect_warnings,
    ranked_by,
    run_main_comparison,
    run_sensitivity,
)
from churn.modeling.comparison_plots import plot_metric_by_family, plot_paired_deltas
from churn.modeling.comparison_results import (
    RESULTS_PATH,
    ModelComparisonResults,
    ReproductionCheck,
    ReproductionError,
    build_results,
    verify_baseline,
    verify_contract_tenure,
    write_results,
)
from churn.modeling.evaluation import build_splitter
from churn.modeling.families import FAMILIES, MAIN_EXPERIMENTS
from churn.modeling.metrics import positive_prevalence
from churn.modeling.plots import (
    OOF_CAPTION,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from churn.modeling.results import load_results as load_baseline_results
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target

logger = logging.getLogger("run_model_comparison")

REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "model_comparison"

METRIC_LABELS = {
    "roc_auc": "ROC-AUC (secondary)",
    "average_precision": "Average Precision (primary)",
    "precision": "Precision @0.5",
    "recall": "Recall @0.5",
    "f1": "F1 @0.5",
    "accuracy": "Accuracy @0.5",
}
DIAGNOSTIC_METRICS = ("precision", "recall", "f1", "accuracy")

#: Relative gain below which a mean delta is called immaterial in the report.
#: It is a *reading* aid applied after the fact, never a selection rule: no
#: experiment was classified, promoted or discarded by this number.
MATERIALITY_REFERENCE = 0.01


def build_figures(
    results: ModelComparisonResults,
    target: np.ndarray,
    oof: dict[str, np.ndarray],
    root: Path,
) -> dict[str, Path]:
    """Render the five Phase 7 figures and return a name -> path mapping."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    main = {record.experiment: record for record in results.main_comparison}
    reference = results.main_comparison[0]
    paths: dict[str, Path] = {}

    paths["ap"] = save_figure(
        plot_metric_by_family(
            {
                record.family: [fold.metrics[PRIMARY_METRIC] for fold in record.folds]
                for record in results.main_comparison
            },
            reference.family,
            "Average Precision",
            "Average Precision per fold, one column per family — 19 original features",
        ),
        figures_dir / "01_cv_average_precision_comparison.png",
    )

    candidates = [record for record in results.main_comparison if record.paired_deltas]
    paths["deltas"] = save_figure(
        plot_paired_deltas(
            {record.family: record.paired_deltas[PRIMARY_METRIC].per_fold for record in candidates},
            f"{reference.experiment} logistic regression",
            "Average Precision",
            "Paired per-fold Average Precision deltas against the logistic baseline",
            "Dots: one per fold.  Vertical bar: mean of the five deltas.  "
            "Right of the line: the family beat the logistic regression on that fold.",
            labels={
                record.family: f"{record.experiment} · {FAMILIES[record.family].label}"
                for record in candidates
            },
        ),
        figures_dir / "02_paired_ap_deltas.png",
    )

    paths["pr"] = save_figure(
        plot_precision_recall_curves(
            target,
            oof,
            {name: main[MAIN_EXPERIMENTS[name]].out_of_fold[PRIMARY_METRIC] for name in oof},
            results.training_positive_prevalence,
            f"Precision-recall — {OOF_CAPTION}",
        ),
        figures_dir / "03_oof_precision_recall.png",
    )

    paths["roc"] = save_figure(
        plot_roc_curves(
            target,
            oof,
            {name: main[MAIN_EXPERIMENTS[name]].out_of_fold[SECONDARY_METRIC] for name in oof},
            f"ROC — {OOF_CAPTION}",
        ),
        figures_dir / "04_oof_roc.png",
    )

    paths["sensitivity"] = save_figure(
        plot_paired_deltas(
            {
                record.family: record.paired_deltas[PRIMARY_METRIC].per_fold
                for record in results.sensitivity_analysis
            },
            "same family without the feature",
            "Average Precision",
            f"Sensitivity to `{CONTRACT_TENURE}` — paired within each family",
            "Each row compares one family against itself: the only difference is the "
            "feature. A family effect and a feature effect are never summed here.",
            labels={
                record.family: f"{record.experiment} · {FAMILIES[record.family].label}"
                for record in results.sensitivity_analysis
            },
        ),
        figures_dir / "05_contract_tenure_sensitivity.png",
    )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _family_table(results: ModelComparisonResults) -> str:
    lines = []
    for definition in results.models:
        parameters = ", ".join(
            f"`{key}={definition.effective_hyperparameters[key]}`"
            for key in _declared_keys(definition.family)
        )
        lines.append(
            f"| {definition.experiment} | `{definition.family}` | `{definition.estimator}` "
            f"| {parameters} |"
        )
    return "\n".join(lines)


def _declared_keys(family: str) -> tuple[str, ...]:
    if family == "logistic_regression":
        return ("C", "penalty", "class_weight", "solver", "max_iter")
    if family == "random_forest":
        return (
            "n_estimators",
            "criterion",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "oob_score",
            "class_weight",
            "random_state",
            "n_jobs",
        )
    return (
        "loss",
        "learning_rate",
        "max_iter",
        "max_leaf_nodes",
        "max_depth",
        "min_samples_leaf",
        "l2_regularization",
        "max_features",
        "early_stopping",
        "class_weight",
        "random_state",
    )


def _absolute_table(records, results: ModelComparisonResults) -> str:
    lines = [
        "| Experiment | Model | Features | Transformed | "
        + " | ".join(METRIC_LABELS[metric] for metric in METRIC_LABELS)
        + " |",
        "| --- " * (len(METRIC_LABELS) + 4) + "|",
    ]
    for record in records:
        cells = " | ".join(
            f"{record.mean[metric]:.4f} ± {record.std[metric]:.4f}" for metric in METRIC_LABELS
        )
        lines.append(
            f"| {record.experiment} | `{record.family}` | {record.n_input_features} | "
            f"{record.n_transformed_features} | {cells} |"
        )
    return "\n".join(lines)


def _fold_table(records, metric: str) -> str:
    lines = [
        "| Experiment | Model | F1 | F2 | F3 | F4 | F5 | Mean ± SD |",
        "| --- " * 8 + "|",
    ]
    for record in records:
        cells = " | ".join(f"{fold.metrics[metric]:.4f}" for fold in record.folds)
        lines.append(
            f"| {record.experiment} | `{record.family}` | {cells} | "
            f"**{record.mean[metric]:.4f}** ± {record.std[metric]:.4f} |"
        )
    return "\n".join(lines)


def _delta_table(records, metric: str) -> str:
    lines = [
        "| Experiment | Model | Reference | F1 | F2 | F3 | F4 | F5 | Mean | SD | +/− folds |",
        "| --- " * 11 + "|",
    ]
    for record in records:
        if metric not in record.paired_deltas:
            continue
        delta = record.paired_deltas[metric]
        cells = " | ".join(f"{value:+.4f}" for value in delta.per_fold)
        lines.append(
            f"| {record.experiment} | `{record.family}` | {delta.reference} | {cells} | "
            f"**{delta.mean:+.4f}** | {delta.std:.4f} | "
            f"{delta.folds_improved}/{delta.folds_worsened} |"
        )
    return "\n".join(lines)


def _oof_table(records) -> str:
    lines = [
        "| Experiment | Model | " + " | ".join(METRIC_LABELS[metric] for metric in METRIC_LABELS),
        "| --- " * (len(METRIC_LABELS) + 2) + "|",
    ]
    lines[0] += " |"
    for record in records:
        cells = " | ".join(f"{record.out_of_fold[metric]:.4f}" for metric in METRIC_LABELS)
        lines.append(f"| {record.experiment} | `{record.family}` | {cells} |")
    return "\n".join(lines)


def _reproduction_table(results: ModelComparisonResults) -> str:
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


def _warning_table(results: ModelComparisonResults) -> str:
    if not results.warnings:
        return "No warning was raised during the run."
    lines = [
        "| Category | Origin | Count | Message |",
        "| --- " * 4 + "|",
    ]
    for warning in results.warnings:
        message = warning.message.replace("|", "\\|")
        lines.append(f"| `{warning.category}` | `{warning.source}` | {warning.count} | {message} |")
    return "\n".join(lines)


def _reading(delta_mean: float, folds_improved: int, n_folds: int, reference: float) -> str:
    """One sentence about direction, consistency and size. No verdict."""
    relative = delta_mean / reference if reference else 0.0
    size = (
        "immaterial at this scale" if abs(relative) < MATERIALITY_REFERENCE else "material in size"
    )
    return (
        f"{delta_mean:+.5f} AP ({relative:+.2%} relative), improving on its reference in "
        f"{folds_improved}/{n_folds} folds — {size}"
    )


def build_report(
    results: ModelComparisonResults,
    figures: dict[str, Path],
    generated_on: str,
) -> str:
    """Render the Phase 7 report. Every number comes from this run."""
    main = results.main_comparison
    reference = main[0]
    candidates = [record for record in main if record.paired_deltas]
    sensitivity = results.sensitivity_analysis
    cv = results.cross_validation
    cv_signature = (
        f"{cv['strategy']}(n_splits={cv['n_splits']}, shuffle={cv['shuffle']}, "
        f"random_state={cv['random_state']})"
    )

    best = max(main, key=lambda record: record.mean[PRIMARY_METRIC])
    ap_readings = "\n".join(
        f"- **{record.experiment} `{record.family}`** — "
        + _reading(
            record.paired_deltas[PRIMARY_METRIC].mean,
            record.paired_deltas[PRIMARY_METRIC].folds_improved,
            len(record.folds),
            reference.mean[PRIMARY_METRIC],
        )
        + "."
        for record in candidates
    )
    sensitivity_readings = "\n".join(
        f"- **{record.experiment} `{record.family}`** — "
        + _reading(
            record.paired_deltas[PRIMARY_METRIC].mean,
            record.paired_deltas[PRIMARY_METRIC].folds_improved,
            len(record.folds),
            next(m.mean[PRIMARY_METRIC] for m in main if m.family == record.family),
        )
        + f", ROC-AUC {record.paired_deltas[SECONDARY_METRIC].mean:+.5f}."
        for record in sensitivity
    )

    n_folds = len(reference.folds)
    gained = [r for r in candidates if r.paired_deltas[PRIMARY_METRIC].mean > 0]
    lost_everywhere = [r for r in candidates if r.paired_deltas[PRIMARY_METRIC].folds_improved == 0]
    if not gained:
        answer_a = (
            f"**No, and not marginally.** Neither non-linear family improved Average "
            f"Precision over the logistic regression on this representation. "
            f"{len(lost_everywhere)} of {len(candidates)} lost in **every one** of the "
            f"{n_folds} folds, so the direction is not a fold artefact — which is exactly "
            "what a paired comparison is for. Note what this does *not* say: it is a "
            "statement about these base configurations on a one-hot matrix, not about the "
            "ceiling of either family."
        )
    else:
        improved = ", ".join(
            f"{r.experiment} ({r.paired_deltas[PRIMARY_METRIC].mean:+.5f} AP in "
            f"{r.paired_deltas[PRIMARY_METRIC].folds_improved}/{n_folds} folds)"
            for r in gained
        )
        answer_a = (
            f"**Partly.** {improved} ended above the logistic baseline on the primary "
            "metric. Whether that is a material advantage, and whether it survives its "
            "cost in interpretability and artefact size, is the judgement section 12 "
            "sets up — not something this ranking settles."
        )

    top_sensitivity = max(sensitivity, key=lambda r: r.paired_deltas[PRIMARY_METRIC].mean)
    top_family_deficit = next(
        (
            r.paired_deltas[PRIMARY_METRIC].mean
            for r in candidates
            if r.family == top_sensitivity.family
        ),
        None,
    )
    answer_b = (
        f"The largest sensitivity gain is **{top_sensitivity.experiment} "
        f"`{top_sensitivity.family}`** at "
        f"{top_sensitivity.paired_deltas[PRIMARY_METRIC].mean:+.5f} AP in "
        f"{top_sensitivity.paired_deltas[PRIMARY_METRIC].folds_improved}/{n_folds} folds. "
    )
    if top_family_deficit is not None and top_family_deficit < 0:
        answer_b += (
            "Read it in context: that family sits "
            f"{abs(top_family_deficit):.4f} AP **below** M0 without the feature, so the "
            "gain recovers part of its own deficit rather than adding anything over the "
            "reference model. A feature helping a weaker model is evidence about that "
            "model, not about the feature's value to the project."
        )
    else:
        answer_b += (
            "Read it against the family's own position in section 7 before treating it as "
            "value added over the reference model."
        )

    return f"""# Model Family Comparison — Telco Customer Churn (Phase 7)

- Generated on: {generated_on}
- Generated by: `scripts/run_model_comparison.py`
- Machine-readable record: `reports/experiments/model_comparison_results.json`
- Raw SHA-256: `{results.raw_sha256}`
- Training partition fingerprint: `{results.training_ids_sha256}`

> **Every number here is cross-validated on the training pool.** No holdout row
> was loaded, transformed, predicted or measured. There is no test result in
> this repository before Phase 9.

---

## 1. Objective

One question:

> Does a non-linear model family improve predictive capability over logistic
> regression when data, features, preprocessing, folds and metrics are held
> constant?

And, separately, a second one:

> Does the manual interaction `{CONTRACT_TENURE}` still contribute once the
> family can represent interactions by itself?

The two are answered apart, in sections 7-10 and 11 respectively. Mixing them
would produce a single number containing both a family effect and a feature
effect, from which neither could be recovered.

This phase **selects nothing and tunes nothing**. It produces evidence for a
review that happens before Phase 8.

## 2. Frozen protocol

| Element | Value |
| --- | --- |
| Data | Frozen Phase 4 training pool, {results.n_training_rows:,} rows |
| Positive prevalence | {results.training_positive_prevalence:.4f} |
| Cross-validation | `{cv_signature}` |
| Identical folds as Phases 5 and 6 | {cv["identical_to_baseline"]} |
| Primary metric | Average Precision |
| Secondary ranking metric | ROC-AUC |
| Diagnostics @ {results.decision_threshold} | Precision, Recall, F1 (Accuracy auxiliary) |
| Threshold optimised | {results.metric_protocol["threshold_optimised"]} |
| Hyperparameter tuning | {results.tuning_performed} |
| Calibration | {results.calibration_performed} |
| Seed | {results.random_seed} |
| scikit-learn | {results.scikit_learn_version} |

**Preprocessing is refitted inside every fold.** Each experiment is a complete
pipeline — `TotalChargesCleaner` → `ColumnTransformer` → classifier — cloned
fresh per fold, so the scaler's statistics, the encoder's categories and the
model's parameters are estimated on the training fold alone.

**Paired comparison.** Every delta is a per-fold difference on the identical
partition:

```text
delta_AP_fold  = AP_candidate_fold  - AP_reference_fold
delta_ROC_fold = ROC_candidate_fold - ROC_reference_fold
```

The between-fold standard deviation is **not** used as a bar a gain must clear:
it describes how one model varies across partitions, not the uncertainty of a
difference between two models measured on the same partitions. With five folds
no significance test is possible and none was run.

## 3. Why the representation is held constant

All three families receive the **same {reference.n_transformed_features}-column
matrix** produced from the {reference.n_input_features} original features by the
same block:

```text
TotalChargesCleaner → StandardScaler (numeric) → OneHotEncoder(handle_unknown="ignore")
```

Holding this block constant is a design decision, not an oversight: it is what
makes the **estimator the only variable** between M0, M1 and M2. If one family
received its own encoding, its delta would measure *estimator plus
representation* against *estimator*, and no part of the difference could be
attributed to the family.

**A common representation is not necessarily the optimal representation for
every family.** Standardisation does not affect where a tree places a split, and
`HistGradientBoostingClassifier` can consume raw categories through
`categorical_features` rather than {reference.n_transformed_features} one-hot
columns. Neither option is exercised here.

What follows is a scope statement about what these numbers estimate: the
performance of **the tested configurations under a common representation**, not
the ceiling of each algorithm. The comparison answers "what does swapping the
estimator do, holding everything else fixed". Whether a different representation
changes the ordering is a separate, unanswered question — a representation
experiment was not run in this phase.

## 4. Model families

| Experiment | Family | Estimator | Declared configuration |
| --- | --- | --- | --- |
{_family_table(results)}

{chr(10).join(f"**{d.experiment} — {d.label}.** {d.hypothesis} {d.notes}" for d in results.models)}

Every configuration above is the family's declared base configuration, fixed
before any result existed. No grid, no random search, no early stopping on an
internal split, no class weighting, no resampling. The full effective parameter
set of each estimator — including the defaults not shown here — is recorded in
`model_comparison_results.json` under `models[].effective_hyperparameters`.

## 5. Baseline reproduction

Two gates run before anything in this report is interpreted:

{_reproduction_table(results)}

M0 rebuilds the frozen Phase 5 logistic regression through the Phase 7 pipeline
builder; S0 rebuilds E3 of the Phase 6 ablation. Both compare **every per-fold
metric** against the stored artefact, and the run aborts if either fails. A
protocol that cannot reproduce its own references cannot attribute a difference
to a family or to a feature — the difference could be an artefact of the
pipeline being assembled differently.

## 6. Primary model-family comparison

Absolute cross-validated performance, mean ± sample standard deviation over
{cv["n_splits"]} folds:

{_absolute_table(main, results)}

{_figure(figures["ap"], "Average Precision per fold, one column per family")}

Per-fold Average Precision:

{_fold_table(main, PRIMARY_METRIC)}

Highest mean Average Precision: **{best.experiment} `{best.family}`**
({best.mean[PRIMARY_METRIC]:.4f}). A ranking is not a verdict — whether that
lead is consistent and whether it is large enough to matter are the next two
sections.

## 7. Paired Average Precision comparison

The primary metric, paired fold by fold against M0:

{_delta_table(candidates, PRIMARY_METRIC)}

{_figure(figures["deltas"], "Paired per-fold Average Precision deltas vs the logistic baseline")}

Read one at a time:

{ap_readings}

**Question A — does a non-linear family improve on the logistic regression?**

{answer_a}

"Material" / "immaterial at this scale" compares the size of a delta against a
baseline AP of {reference.mean[PRIMARY_METRIC]:.4f}, applied after the fact as a
reading aid. It classified nothing: no experiment in this phase was promoted,
discarded or ranked by it, and a small mean delta with a consistent direction is
exactly the case where five folds cannot distinguish a real effect from noise.

## 8. ROC-AUC comparison

The secondary ranking metric, on the same folds:

{_delta_table(candidates, SECONDARY_METRIC)}

Per-fold ROC-AUC:

{_fold_table(main, SECONDARY_METRIC)}

ROC-AUC is reported as a second view of ranking quality, not as a tie-breaker. A
family that improves ROC-AUC while losing Average Precision has moved negatives
around more than it has improved churner retrieval, which on this problem is the
less useful of the two.

## 9. Out-of-fold comparison

Each of the {results.n_training_rows:,} training rows received exactly one
prediction, produced by a model that never saw that row while fitting. Metrics
recomputed over the pooled out-of-fold predictions:

{_oof_table(main)}

{_figure(figures["pr"], "Precision-recall from out-of-fold predictions on the training pool")}

{_figure(figures["roc"], "ROC from out-of-fold predictions on the training pool")}

These are **cross-validated out-of-fold results on the training pool**, not test
metrics, and must never be relabelled as such. Pooled OOF metrics are computed
on one merged score vector produced by five different fitted models, so they are
a slightly different quantity from the mean of the per-fold metrics; both are
reported and neither replaces the other.

## 10. Threshold-0.5 diagnostics

The decision rule was not moved. These describe one operating point:

| Experiment | Model | {" | ".join(METRIC_LABELS[m] for m in DIAGNOSTIC_METRICS)} |
| --- | --- | --- | --- | --- | --- |
{
        chr(10).join(
            f"| {record.experiment} | `{record.family}` | "
            + " | ".join(f"{record.mean[metric]:.4f}" for metric in DIAGNOSTIC_METRICS)
            + " |"
            for record in main
        )
    }

Accuracy is auxiliary and ranks nothing: at a {results.training_positive_prevalence:.2%}
positive rate a model that never predicts churn already reaches ~73%. Recall and
precision at 0.5 are a description of where the default cut-off happens to sit
on each model's score distribution — a model whose probabilities are shifted can
look worse here while ranking identically. Threshold selection is Phase 9.

## 11. Contract-tenure sensitivity

Phase 6 retained `{CONTRACT_TENURE}` as a **candidate**, not as an adopted
feature. The question here is whether the manual interaction still contributes
once a family can represent interactions natively.

Each experiment is paired against **its own family** without the feature, so the
only difference inside each row is `{CONTRACT_TENURE}`:

{_absolute_table(sensitivity, results)}

Paired Average Precision deltas:

{_delta_table(sensitivity, PRIMARY_METRIC)}

Paired ROC-AUC deltas:

{_delta_table(sensitivity, SECONDARY_METRIC)}

{_figure(figures["sensitivity"], "Sensitivity to contract_tenure, paired within each family")}

Read one at a time:

{sensitivity_readings}

S0 is E3 of Phase 6, reproduced exactly (section 5), so the logistic row carries
no new information — it is the control that proves the sensitivity analysis is
measuring the same thing Phase 6 measured. The tree rows are the new evidence.

**Question B — does the manual interaction still contribute when the family can
learn interactions itself?**

{answer_b} The two families that can express an interaction without being told
about it are precisely the two where the answer matters, and they disagree with
each other — which is itself the finding: one explicit interaction column is not
uniformly redundant for a tree ensemble, nor uniformly useful.

The four Phase 6 candidates that did not pass its heuristic —
`protective_services`, `automatic_payment`, `historical_average_charge`,
`charge_intensity` — were **not** reopened. Re-running every rejected candidate
against every new family would turn a controlled comparison into a search over
feature-set × family cells, which is how a leaderboard eventually finds a lucky
combination.

## 12. Complexity and interpretability trade-offs

| Family | Fitted objects | Interpretability | Inference cost | Fitted state to persist |
| --- | --- | --- | --- | --- |
| `logistic_regression` | 1 coefficient vector \
({reference.n_transformed_features} weights) | Direct: one signed coefficient per \
column, odds-ratio readable | One dot product | Smallest |
| `random_forest` | 100 unpruned trees | Indirect: impurity or permutation \
importance, no single signed effect | 100 tree traversals | Largest |
| `hist_gradient_boosting` | 100 boosted trees over binned features | Indirect: \
same, plus an additive stage structure | 100 shallow traversals | Middle |

A non-linear family is not merely "more accurate or not". It costs
interpretability, which this project needs in Phase 10; it costs a larger
artefact to persist, version and monitor; and it costs a fitting procedure with
more parameters that can silently change behaviour between versions. Those costs
are worth paying for a gain that is consistent and large enough to matter
operationally — and the paired deltas in section 7 are the evidence on which
that judgement has to be made, not the ranking in section 6.

## 13. Limitations

- **No holdout estimate.** Everything is training-pool cross-validation.
- **Five folds, no significance test.** The deltas describe direction and
  consistency; they are not statistical claims.
- **Untuned configurations.** Each family ran one declared configuration. A
  family's result here is not that family's ceiling, and boosting in particular
  is the family most sensitive to its learning parameters.
- **One representation, chosen to isolate the estimator.** It is held constant
  by design and is not necessarily optimal for every family, so these results
  estimate the performance of the tested configurations under a common
  representation rather than the ceiling of each algorithm.
- **No calibration.** Probability quality was not measured and was not used to
  rank anything; ranking metrics say nothing about whether a 0.7 means 70%.
- **No class-imbalance handling.** `class_weight` is `None` everywhere.
- **The analyst-exposure limitation from Phase 3 still applies.**
- **Snapshot classification, not forecasting.** The dataset has no time
  dimension.

## 14. What was deliberately not tested

| Not tested | Why | Where it belongs |
| --- | --- | --- |
| Number of trees, depth, learning rate, regularisation | This is a family \
comparison at base configuration | Phase 8 |
| `class_weight`, resampling | A separate lever, and one that interacts with the \
threshold | Phase 8/9 |
| Native categorical handling in `HistGradientBoosting` | Would vary the \
representation together with the estimator | A representation experiment |
| Out-of-bag scoring for the forest | A second, differently-computed number \
invites selecting on it | — |
| Early stopping | Model selection inside a training fold | Phase 8, if at all |
| XGBoost, LightGBM, CatBoost | No dependency was added; three families already \
answer the question | — |
| Threshold movement, calibration, cost framing | Not this phase's question | Phase 9 |
| The four Phase 6 candidates that did not pass its heuristic | Closed by the Phase 6 protocol | — |

## 15. Candidates carried to Phase 8

This section records what the evidence supports carrying forward. It does not
authorise tuning: that decision is taken by a reviewer, after reading sections
7 to 12.

| Family | Status going into Phase 8 |
| --- | --- |
{
        chr(10).join(
            f"| `{record.family}` | mean AP {record.mean[PRIMARY_METRIC]:.4f}"
            + (
                f", paired {record.paired_deltas[PRIMARY_METRIC].mean:+.5f} vs M0 in "
                f"{record.paired_deltas[PRIMARY_METRIC].folds_improved}/{len(record.folds)} folds"
                if record.paired_deltas
                else " (reference)"
            )
            + " |"
            for record in main
        )
    }

`{CONTRACT_TENURE}` remains a **retained candidate**, now with per-family
evidence (section 11) instead of logistic-only evidence. It was not adopted into
any feature set, and the {reference.n_input_features} original features remain
the reference representation.

Nothing here has been tuned, calibrated, thresholded or evaluated on the
holdout, and no family has been selected.
"""


def _log_warnings(captured) -> None:
    for warning in captured:
        logger.warning(
            "%s x%d from %s: %s", warning.category, warning.count, warning.source, warning.message
        )


def main() -> int:
    """Run the comparison. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()

    try:
        verify_split_manifest(load_split_manifest())
        baseline_reference = load_baseline_results()
        ablation_reference = load_ablation_results()
    except FileNotFoundError as error:
        logger.error("%s", error)
        return 1
    except RawDatasetMismatchError as error:
        logger.error("%s", error)
        return 2
    except SplitManifestMismatchError as error:
        logger.error("%s", error)
        return 3

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

    splitter = build_splitter()
    checks: list[ReproductionCheck] = []

    def gate_baseline(run: FamilyRun) -> None:
        checks.append(verify_baseline(run, baseline_reference))

    def gate_contract_tenure(run: FamilyRun) -> None:
        checks.append(verify_contract_tenure(run, ablation_reference))

    with collect_warnings() as captured:
        try:
            comparison = run_main_comparison(features, target, splitter, gate_baseline)
            sensitivity = run_sensitivity(
                features, target, splitter, comparison, gate_contract_tenure
            )
        except ReproductionError as error:
            logger.error("%s", error)
            return 4

    _log_warnings(captured)

    results = build_results(
        comparison,
        sensitivity,
        checks,
        baseline_reference,
        ablation_reference,
        labels,
        captured,
    )
    write_results(results)

    oof = {run.family: run.evaluation.oof_probability for run in comparison.values()}
    figures = build_figures(results, labels, oof, PROJECT_ROOT)
    Path(REPORT_PATH).write_text(
        build_report(results, figures, date.today().isoformat()),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)

    ordered = ranked_by(list(comparison.values()))
    logger.info(
        "Highest mean %s: %s (%.4f). Ranking is not selection.",
        PRIMARY_METRIC,
        ordered[0].experiment,
        ordered[0].evaluation.mean(PRIMARY_METRIC),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
