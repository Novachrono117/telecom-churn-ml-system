"""Run the Phase 8B nested tuning study and write its report.

Usage (from the repository root):

    uv run python scripts/build_split.py --verify        # prove the split
    uv run python scripts/run_tuning.py

Writes:
    reports/experiments/tuning_results.json
    reports/tuning_report.md
    reports/figures/tuning/*.png

Only the frozen training pool is loaded. The holdout is never read.

Two gates stop the run before any tuning result is interpreted:

1. the modern logistic spelling must reproduce the legacy one, per fold and in
   its out-of-fold probabilities — otherwise the migration is a model change;
2. T0 must reproduce the frozen Phase 5 baseline.

The threshold is not optimised, ``class_weight`` is ``None`` everywhere, no
probability is calibrated and no fitted estimator is persisted.
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
from churn.features.ablation import PRIMARY_METRIC, SECONDARY_METRIC
from churn.features.results import load_results as load_ablation_results
from churn.modeling.comparison import collect_warnings
from churn.modeling.comparison_plots import plot_metric_by_family, plot_paired_deltas
from churn.modeling.comparison_results import (
    ArtefactReference,
    ReproductionCheck,
    ReproductionError,
)
from churn.modeling.comparison_results import load_results as load_comparison_results
from churn.modeling.evaluation import build_splitter, evaluate_model
from churn.modeling.families import build_family_pipeline
from churn.modeling.metrics import positive_prevalence
from churn.modeling.models import LOGISTIC_REGRESSION
from churn.modeling.plots import plot_precision_recall_curves
from churn.modeling.representation_results import load_results as load_representation_results
from churn.modeling.results import load_results as load_baseline_results
from churn.modeling.tuning import (
    ELIGIBLE,
    FROZEN_LOGISTIC,
    T1_SEARCH_SPACE,
    T2_SEARCH_SPACE,
    TUNED_HGB,
    TUNED_LOGISTIC,
    NestedRun,
    build_hgb_pipeline,
    build_inner_splitter,
    build_modern_logistic_pipeline,
    eligibility,
    n_candidates,
    pair,
    run_final_search,
    run_nested_cv,
    select_candidate,
)
from churn.modeling.tuning_plots import plot_selection_stability
from churn.modeling.tuning_results import (
    RESULTS_PATH,
    FinalSearchRecord,
    TuningResults,
    build_eligibility_record,
    build_results,
    build_selection_record,
    verify_baseline_reproduction,
    verify_migration,
    write_results,
)
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target

logger = logging.getLogger("run_tuning")

REPORT_PATH = PROJECT_ROOT / "reports" / "tuning_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "tuning"

METRIC_LABELS = {
    "roc_auc": "ROC-AUC (secondary)",
    "average_precision": "Average Precision (primary)",
    "precision": "Precision @0.5",
    "recall": "Recall @0.5",
    "f1": "F1 @0.5",
    "accuracy": "Accuracy @0.5",
}
DIAGNOSTIC_METRICS = ("precision", "recall", "f1", "accuracy")

OUTER_OOF_CAPTION = "Nested cross-validated outer-OOF performance on the training pool"


def build_figures(
    results: TuningResults,
    target: np.ndarray,
    oof: dict[str, np.ndarray],
    root: Path,
) -> dict[str, Path]:
    """Render the four Phase 8B figures and return a name -> path mapping."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    records = {record.experiment: record for record in results.procedures}
    paths: dict[str, Path] = {}

    paths["ap"] = save_figure(
        plot_metric_by_family(
            {
                record.model: [fold.metrics[PRIMARY_METRIC] for fold in record.folds]
                for record in results.procedures
            },
            records[FROZEN_LOGISTIC].model,
            "Average Precision",
            "Outer-fold Average Precision — frozen baseline, tuned logistic, tuned boosting",
        ),
        figures_dir / "01_outer_cv_ap_comparison.png",
    )

    deltas: dict[str, list[float]] = {}
    labels: dict[str, str] = {}
    for experiment, reference in (
        (TUNED_LOGISTIC, FROZEN_LOGISTIC),
        (TUNED_HGB, FROZEN_LOGISTIC),
        (TUNED_HGB, TUNED_LOGISTIC),
    ):
        record = records[experiment]
        key = f"{experiment}_vs_{reference}"
        deltas[key] = record.paired_deltas[reference][PRIMARY_METRIC].per_fold
        labels[key] = f"{experiment} vs {reference}"

    paths["deltas"] = save_figure(
        plot_paired_deltas(
            deltas,
            "its reference, same outer fold",
            "Average Precision",
            "Paired per-outer-fold Average Precision deltas",
            "Dots: one per outer fold.  Vertical bar: mean of the five deltas.  Right of the "
            "line: the procedure beat its reference on that fold. Each row has its own "
            "reference — the top two against the frozen baseline, the third head to head.",
            labels=labels,
        ),
        figures_dir / "02_paired_ap_deltas.png",
    )

    paths["pr"] = save_figure(
        plot_precision_recall_curves(
            target,
            oof,
            {
                record.model: record.outer_out_of_fold[PRIMARY_METRIC]
                for record in results.procedures
                if record.model in oof
            },
            results.training_positive_prevalence,
            f"Precision-recall — {OUTER_OOF_CAPTION}",
        ),
        figures_dir / "03_outer_oof_precision_recall.png",
    )

    frequency = {
        record.experiment: record.hyperparameter_selection_frequency
        for record in results.procedures
        if record.hyperparameter_selection_frequency
    }
    paths["stability"] = save_figure(
        plot_selection_stability(
            frequency,
            results.outer_cross_validation["n_splits"],
            "Hyperparameter values chosen by the inner search, per outer fold",
        ),
        figures_dir / "04_hyperparameter_selection_stability.png",
    )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _absolute_table(results: TuningResults) -> str:
    lines = [
        "| Experiment | Procedure | Tuned | "
        + " | ".join(METRIC_LABELS[metric] for metric in METRIC_LABELS)
        + " |",
        "| --- " * (len(METRIC_LABELS) + 3) + "|",
    ]
    for record in results.procedures:
        cells = " | ".join(
            f"{record.mean[metric]:.4f} ± {record.std[metric]:.4f}" for metric in METRIC_LABELS
        )
        lines.append(f"| {record.experiment} | {record.label} | {record.tuned} | {cells} |")
    return "\n".join(lines)


def _fold_table(results: TuningResults, metric: str) -> str:
    lines = [
        "| Experiment | F1 | F2 | F3 | F4 | F5 | Mean ± SD |",
        "| --- " * 7 + "|",
    ]
    for record in results.procedures:
        cells = " | ".join(f"{fold.metrics[metric]:.4f}" for fold in record.folds)
        lines.append(
            f"| {record.experiment} | {cells} | "
            f"**{record.mean[metric]:.4f}** ± {record.std[metric]:.4f} |"
        )
    return "\n".join(lines)


def _tied(delta) -> str:
    """Render the tied-fold count, or a dash when the artefact does not carry it."""
    return "—" if delta.folds_unchanged is None else str(delta.folds_unchanged)


def _delta_table(results: TuningResults, pairs: tuple[tuple[str, str], ...]) -> str:
    records = {record.experiment: record for record in results.procedures}
    lines = [
        "| Comparison | Metric | F1 | F2 | F3 | F4 | F5 | Mean | SD | Improved | Tied | Worsened |",
        "| --- " * 12 + "|",
    ]
    for experiment, reference in pairs:
        for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
            delta = records[experiment].paired_deltas[reference][metric]
            cells = " | ".join(f"{value:+.4f}" for value in delta.per_fold)
            lines.append(
                f"| {experiment} vs {reference} | {METRIC_LABELS[metric]} | {cells} | "
                f"**{delta.mean:+.4f}** | {delta.std:.4f} | "
                f"{delta.folds_improved} | {_tied(delta)} | {delta.folds_worsened} |"
            )
    return "\n".join(lines)


def _best_params_table(results: TuningResults) -> str:
    lines = [
        "| Experiment | Outer fold | Best inner AP | Selected hyperparameters |",
        "| --- | --- | --- | --- |",
    ]
    for record in results.procedures:
        if not record.tuned:
            continue
        for fold in record.folds:
            parameters = ", ".join(
                f"`{key}={value}`" for key, value in (fold.best_params or {}).items()
            )
            lines.append(
                f"| {record.experiment} | {fold.fold} | "
                f"{fold.best_inner_score:.4f} | {parameters} |"
            )
    return "\n".join(lines)


def _stability_table(results: TuningResults) -> str:
    lines = [
        "| Experiment | Parameter | Values chosen across the 5 outer folds |",
        "| --- | --- | --- |",
    ]
    for record in results.procedures:
        for parameter, values in record.hyperparameter_selection_frequency.items():
            rendered = ", ".join(f"`{value}` ×{count}" for value, count in values.items())
            lines.append(f"| {record.experiment} | `{parameter}` | {rendered} |")
    return "\n".join(lines)


def _oof_table(results: TuningResults) -> str:
    lines = [
        "| Experiment | Procedure | "
        + " | ".join(METRIC_LABELS[metric] for metric in METRIC_LABELS),
        "| --- " * (len(METRIC_LABELS) + 2) + "|",
    ]
    lines[0] += " |"
    for record in results.procedures:
        cells = " | ".join(f"{record.outer_out_of_fold[metric]:.4f}" for metric in METRIC_LABELS)
        lines.append(f"| {record.experiment} | {record.label} | {cells} |")
    return "\n".join(lines)


def _eligibility_table(results: TuningResults) -> str:
    total = results.outer_cross_validation["n_splits"]
    lines = [
        "| Experiment | Mean ΔAP vs T0 | Improved | Tied | Worsened | Improved of "
        f"{total} | Status |",
        "| --- " * 7 + "|",
    ]
    for record in results.eligibility:
        lines.append(
            f"| {record.experiment} | {record.mean_paired_delta_ap:+.5f} | "
            f"{record.outer_folds_improved} | {record.outer_folds_unchanged} | "
            f"{record.outer_folds_worsened} | {record.outer_folds_improved}/{total} | "
            f"**{record.status}** |"
        )
    return "\n".join(lines)


def _final_search_table(results: TuningResults) -> str:
    if not results.final_full_training_searches:
        return (
            "No final search was run. The frozen baseline was selected, so there is no "
            "hyperparameter to determine: `C` stays at 1.0."
        )
    lines = [
        "| Experiment | Best parameters on the full training pool "
        "| Inner CV score (not an estimate) |",
        "| --- | --- | --- |",
    ]
    for record in results.final_full_training_searches:
        parameters = ", ".join(f"`{key}={value}`" for key, value in record.best_params.items())
        lines.append(f"| {record.experiment} | {parameters} | {record.best_cv_score:.4f} |")
    return "\n".join(lines)


def _warning_table(results: TuningResults) -> str:
    if not results.warnings:
        return "No warning was raised during the run."
    lines = ["| Category | Origin | Count | Message |", "| --- " * 4 + "|"]
    for warning in results.warnings:
        message = warning.message.replace("|", "\\|")
        lines.append(f"| `{warning.category}` | `{warning.source}` | {warning.count} | {message} |")
    return "\n".join(lines)


def build_report(results: TuningResults, figures: dict[str, Path]) -> str:
    """Render the Phase 8B report. Every number comes from this run.

    The report carries **no wall-clock metadata**. It is a pure function of the
    results and the figure paths, so re-running the generator on an unchanged
    repository reproduces it byte for byte and any diff is a real change. Git
    already records when a file was produced; a generation date inside the file
    only guarantees a spurious diff on every rerun.
    """
    records = {record.experiment: record for record in results.procedures}
    t0, t1, t2 = records[FROZEN_LOGISTIC], records[TUNED_LOGISTIC], records[TUNED_HGB]
    outer, inner = results.outer_cross_validation, results.inner_cross_validation
    migration = results.compatibility_migration
    selection = results.selection
    spaces = {space.experiment: space for space in results.search_spaces}
    selected = records[selection.selected_experiment]

    outer_signature = (
        f"{outer['strategy']}(n_splits={outer['n_splits']}, shuffle={outer['shuffle']}, "
        f"random_state={outer['random_state']})"
    )
    inner_signature = (
        f"{inner['strategy']}(n_splits={inner['n_splits']}, shuffle={inner['shuffle']}, "
        f"random_state={inner['random_state']})"
    )
    inner_fits = (
        spaces[TUNED_LOGISTIC].n_candidates * inner["n_splits"] * outer["n_splits"],
        spaces[TUNED_HGB].n_candidates * inner["n_splits"] * outer["n_splits"],
    )

    return f"""# Nested Hyperparameter Tuning — Telco Customer Churn (Phase 8B)

- Generated by: `scripts/run_tuning.py`
- Machine-readable record: `reports/experiments/tuning_results.json`
- Raw SHA-256: `{results.raw_sha256}`
- Training partition fingerprint: `{results.training_ids_sha256}`

> **This report carries no generation date.** It is a pure function of its
> inputs, so re-running the generator on an unchanged repository reproduces it
> byte for byte and any diff is a real change. Git records when it was produced.

> **Every number here is cross-validated on the training pool.** No holdout row
> was loaded, transformed, predicted or measured. There is no test result in
> this repository before Phase 9.

---

## 1. Objective

Three questions, answered in order:

1. can the frozen logistic regression be moved to the modern scikit-learn API
   without changing what it computes?
2. does tuning the logistic regression, or tuning histogram gradient boosting,
   improve on the frozen baseline when the tuning **procedure** is evaluated
   honestly?
3. which configuration should be frozen as the candidate carried into Phase 9?

The threshold is not optimised, `class_weight` is `None` everywhere, no
probability is calibrated, and no fitted estimator is persisted.

## 2. Why nested cross-validation

A grid search reports the score of the configuration it selected, on the folds
it used to select it. That number is chosen by maximisation and is therefore
optimistically biased — it answers "how well did the best candidate do on the
data used to pick it", which is not a question anybody needs answered.

Nested cross-validation asks the useful question instead:

```text
for each outer fold (the frozen 5-fold partition):
    run the entire search inside the outer TRAINING fold only
    refit the winning configuration on that outer training fold
    predict the outer VALIDATION fold exactly once
```

What this estimates is **the tuning procedure**, not a single configuration.
That distinction has a visible consequence: the search may choose different
hyperparameters in different outer folds, and that variation is part of the
thing being measured rather than a defect in it. Section 13 reports it.

The inner search performed
{inner_fits[0]:,} fits for T1 and {inner_fits[1]:,} for T2 across the whole
study, none of which ever saw an outer validation row.

## 3. Protected holdout

The {results.n_training_rows:,} training rows are the entire universe of this
phase. `load_holdout()` is not called anywhere in the Phase 8B code, and an AST
audit fails the build if it appears. No metric, distribution, transformation or
prediction involving the protected partition exists.

## 4. Frozen feature representation

| Element | Value |
| --- | --- |
| Features | {results.feature_set["n_features"]} original (3 numeric, 16 categorical) |
| Engineered | {results.engineered_features_used or "none"} |
| Representation | `{results.feature_set["representation"]}` → \
{results.feature_set["n_transformed_features"]} transformed columns |
| Positive prevalence | {results.training_positive_prevalence:.4f} |
| `class_weight` | `{results.class_weight}` for every candidate |
| Threshold | {results.decision_threshold} (diagnostic only; optimised: \
{results.threshold_optimised}) |
| Calibration | {results.calibration_performed} |
| Seed | {results.random_seed} |
| scikit-learn | {results.scikit_learn_version} |

`contract_tenure` is not used: mixing a feature effect into a hyperparameter
study would produce one number containing both. The native categorical
representation is not reopened — Phase 8A closed it as
`REPRESENTATION_NOT_SUPPORTED`.

## 5. Logistic compatibility migration

The frozen baseline passes `penalty="l2"`, deprecated in scikit-learn 1.8 and
scheduled for removal in 1.10. The modern spelling of the same penalty is
`l1_ratio=0.0`.

| | Value |
| --- | --- |
| Legacy | {", ".join(f"`{k}={v}`" for k, v in migration.legacy_configuration.items())} |
| Modern | {", ".join(f"`{k}={v}`" for k, v in migration.modern_configuration.items())} |
| Tolerance | {migration.tolerance:.0e} |
| Largest per-fold metric difference | **{migration.max_metric_difference:.1e}** |
| Largest out-of-fold probability difference | **{migration.max_oof_probability_difference:.1e}** |
| Legacy emits the removal warning | {migration.legacy_emits_deprecation_warning} |
| Modern emits the removal warning | {migration.modern_emits_deprecation_warning} |
| Classification | **{migration.classification}** |

Both the per-fold metrics **and** the out-of-fold probability vectors are
compared. Metrics can agree while probabilities differ in a way that would
surface later, so the stronger check is the one that gates the run — and the run
aborts if either exceeds tolerance.

{migration.note}

**The legacy builder is preserved.** It is what reproduces the historical
artefacts, and those are not regenerated. The modern builder is used for the new
Phase 8B experiments only. This is a compatibility migration, not a model
change; the distinction matters because a model change would silently redefine
the reference every earlier number is measured against.

## 6. Candidate families

| Experiment | Procedure | Tuned |
| --- | --- | --- |
| {FROZEN_LOGISTIC} | {t0.label} | no — the floor tuning has to justify clearing |
| {TUNED_LOGISTIC} | {t1.label} | yes |
| {TUNED_HGB} | {t2.label} | yes |

**Random forest is deliberately out of scope.** Phase 7 measured a mean paired
delta of about −0.0553 AP against the logistic regression with 0/5 folds
positive, and nothing since produced evidence to reopen it. That is a scoping
decision about where to spend a tuning budget, **not** a claim that the family is
universally inferior.

## 7. Search spaces

| Experiment | Estimator | Candidates | Grid |
| --- | --- | --- | --- |
{
        chr(10).join(
            f"| {space.experiment} | `{space.estimator}` | **{space.n_candidates}** | "
            + "; ".join(
                f"`{key.split('__')[-1]}`: {values}" for key, values in space.parameters.items()
            )
            + " |"
            for space in results.search_spaces
        )
    }

Both grids **contain the frozen configuration**
({
        ", ".join(
            f"{space.experiment}: {space.contains_frozen_configuration}"
            for space in results.search_spaces
        )
    }),
so each search is structurally able to answer "the untuned configuration was
already the best" rather than being forced to move away from it.

Fixed for every logistic candidate: `l1_ratio=0.0`, `solver='lbfgs'`,
`class_weight=None`, `max_iter=100`. Fixed for every boosting candidate:
{", ".join(f"`{k}={v}`" for k, v in spaces[TUNED_HGB].fixed_parameters.items())}.

Neither grid was widened after seeing a score, and no adaptive second round was
run. A grid extended towards whichever edge won is a search over search spaces,
and its nested estimate no longer describes the procedure that was actually run.

## 8. Outer/inner CV protocol

| Level | Splitter | Role |
| --- | --- | --- |
| Outer | `{outer_signature}` | estimate of the procedure; never used for selection |
| Inner | `{inner_signature}` | hyperparameter selection, inside one outer training fold |

The outer partition is the identical one used by Phases 5 to 8A, so every paired
delta in this report is a per-fold difference on the same folds every earlier
number was computed on.

**Preprocessing is inside the search.** The estimator handed to `GridSearchCV`
is the complete pipeline — `TotalChargesCleaner` → `ColumnTransformer` →
classifier — so each inner fold refits the scaler and the encoder on its own
training part. Nothing is pre-fitted before the search, which is the leakage a
tuning study most commonly introduces: a scaler fitted once on the outer
training fold would leak inner-validation rows into every inner fit.

Selection is on **{results.metric_protocol["search_selection_metric"]}** alone.
No multi-metric objective was constructed; ROC-AUC and the 0.5 diagnostics
accompany the evaluation and select nothing.

## 9. Frozen baseline T0

T0 is the modern logistic regression at the frozen configuration, evaluated
through the same outer loop as the tuned procedures — so "same folds" is a
structural property of the code, not a claim.

{_reproduction_table(results)}

## 10-11. Outer-fold results

{_absolute_table(results)}

{_figure(figures["ap"], "Outer-fold Average Precision for the three procedures")}

Per-fold Average Precision:

{_fold_table(results, PRIMARY_METRIC)}

What each inner search selected, fold by fold:

{_best_params_table(results)}

The inner scores in that table are **selection scores**, not performance. They
are the mean inner-CV Average Precision of the winning candidate, computed on
the folds used to choose it, and they are systematically higher than the outer
numbers above for exactly that reason.

## 12. Paired outer-fold comparisons

{
        _delta_table(
            results,
            (
                (TUNED_LOGISTIC, FROZEN_LOGISTIC),
                (TUNED_HGB, FROZEN_LOGISTIC),
                (TUNED_HGB, TUNED_LOGISTIC),
            ),
        )
    }

{_figure(figures["deltas"], "Paired per-outer-fold Average Precision deltas")}

**Improved, tied and worsened are three separate counts, and they are reported
separately.** A tuned procedure whose search happens to select the baseline
configuration inside an outer fold fits the identical model there, and the delta
is exactly zero — neither an improvement nor a loss. Collapsing that into a
two-way split would let a tie read as a defeat and would misstate how consistent
a difference was, so the columns above do not have to sum the way a two-way
split would. The eligibility rule in section 15 counts **strictly positive**
deltas only, and a tie never counts towards it.

The between-fold standard deviation is **not** a bar these deltas must clear: it
describes how one procedure varies across partitions, not the uncertainty of a
difference between two procedures measured on the same partitions. Five folds
cannot support a significance test and none was run.

## 13. Hyperparameter stability

{_stability_table(results)}

{_figure(figures["stability"], "Hyperparameter values chosen by the inner search, per outer fold")}

A value chosen in all five outer folds means the search lands in the same region
regardless of the partition. A parameter that moves means the inner objective is
flat enough there that the partition decides — which is information about the
problem, not a defect in the search. **Frequency is not evidence**: five folds
cannot establish that a value is correct, only that the choice was or was not
stable.

## 14. Outer-OOF performance

Each of the {results.n_training_rows:,} training rows received exactly one
prediction, from a procedure whose **entire selection process** — inner folds,
grid search, refit — ran without that row:

{_oof_table(results)}

{_figure(figures["pr"], "Precision-recall from outer out-of-fold predictions")}

These are **nested cross-validated outer-OOF results on the training pool**.
They are not test performance and must never be relabelled as such. No
per-customer prediction was persisted.

## 15. Eligibility to replace the baseline

The rule, fixed before any Phase 8B number was read:

> A tuned procedure is `ELIGIBLE_TO_REPLACE_BASELINE` when its **mean paired
> delta AP against T0 is > 0** and AP improves in **at least
> {selection.eligibility_min_positive_folds} of the {outer["n_splits"]} outer
> folds**, where "improves" means a **strictly positive** delta.

No minimum-gain cutoff exists anywhere in it (`minimum_gain_cutoff`:
{selection.minimum_gain_cutoff}). It is an engineering heuristic about direction
and consistency, not a significance test. A tied fold counts as neither an
improvement nor a loss, so it cannot help a procedure clear the rule — the same
rule that was fixed before any of these numbers existed, applied unchanged.

{_eligibility_table(results)}

{chr(10).join(f"- **{record.experiment}** — {record.rationale}" for record in results.eligibility)}

## 16. Candidate selection

The pre-registered rule, in full:

{chr(10).join(f"{index}. {step}" for index, step in enumerate(selection.rule, start=1))}

Every branch of it is exercised by tests on synthetic inputs, so the branch the
real numbers happened to take is not the only one anybody checked.

**Selected: {selection.selected_experiment} — {selection.selected_label}.**

{selection.rationale}

Frozen hyperparameters carried forward:
{", ".join(f"`{key}={value}`" for key, value in selection.frozen_hyperparameters.items())}.

## 17. Final full-training parameter search

{_final_search_table(results)}

**This is not a generalisation estimate, and its score must never be quoted as
one.** A search over the whole training pool scores each candidate on the same
rows it uses to rank them, so the winner's score is optimistically biased by
construction. Its only job is to answer "which values get frozen" using all
available training data. The honest estimate of what these procedures achieve is
the nested outer result in sections 10-14.

## 18. Complexity trade-offs

| Procedure | Fitted objects | Interpretability | Fitting cost | Parameters chosen by data |
| --- | --- | --- | --- | --- |
| {FROZEN_LOGISTIC} | 1 coefficient vector | direct, odds-ratio readable | one fit | none |
| {TUNED_LOGISTIC} | 1 coefficient vector | direct, odds-ratio readable | \
{spaces[TUNED_LOGISTIC].n_candidates} × inner folds per fit | 1 (`C`) |
| {TUNED_HGB} | 100-200 boosted trees | indirect; importances, no signed effect | \
{spaces[TUNED_HGB].n_candidates} × inner folds per fit | 5 |

Tuning is not free even when it helps. Each searched parameter is a degree of
freedom fitted to the training pool, and a procedure with five of them has more
opportunity to track the partition than one with a single regularisation
constant. That is a reason to require consistency across folds rather than a
higher mean — which is what the eligibility rule encodes.

## 19. Limitations

- **No holdout estimate.** Everything is training-pool cross-validation.
- **Five outer folds, no significance test.** The deltas and the eligibility
  labels describe direction and consistency, not statistical significance.
- **The nested estimate is of the procedure**, not of the single configuration
  that section 17 freezes. Those are different objects, and the frozen one has
  no unbiased estimate anywhere in this repository yet.
- **The grids are small and pre-declared.** A wider search might find more; it
  would also raise the multiplicity that makes a lucky configuration likelier.
- **`class_weight` was not explored, and is deferred rather than postponed to
  Phase 9.** It stays frozen at `None`. Class weights change the fit itself, so
  reopening them would reopen model selection and tuning — this phase's entire
  comparison would have to be rerun to stay valid. The threshold changes only the
  decision rule applied after training, which is why it can be settled on its own
  without invalidating anything here. Cost-sensitive training is explicitly out
  of scope for the main line of this project.
- **The threshold is untouched**, so the 0.5 diagnostics describe one operating
  point and selected nothing.
- **No calibration.** Ranking metrics say nothing about whether a 0.7 means 70%.
- **The analyst-exposure limitation from Phase 3 still applies.**
- **Snapshot classification, not forecasting.** The dataset has no time dimension.

## 20. Frozen candidate carried to Phase 9

| Property | Value |
| --- | --- |
| Experiment | **{selection.selected_experiment}** |
| Procedure | {selection.selected_label} |
| Hyperparameters | \
{", ".join(f"`{key}={value}`" for key, value in selection.frozen_hyperparameters.items())} |
| Feature set | {results.feature_set["n_features"]} original features, \
`{results.feature_set["representation"]}` |
| `class_weight` | `{results.class_weight}` — frozen and deferred |
| Threshold | {selection.threshold_status} |
| Calibration | {selection.calibration_status} |
| Outer-OOF Average Precision | {selected.outer_out_of_fold[PRIMARY_METRIC]:.4f} |
| Outer-OOF ROC-AUC | {selected.outer_out_of_fold[SECONDARY_METRIC]:.4f} |

**The candidate is the modern implementation**, and Phase 9 must instantiate
exactly this:

```python
LogisticRegression(
    C=1.0,
    l1_ratio=0.0,
    solver="lbfgs",
    class_weight=None,
    max_iter=100,
)
```

`churn.modeling.tuning.build_modern_logistic` is that builder. Section 5 proved
it reproduces the legacy `penalty="l2"` configuration with a maximum per-fold
metric difference of **{migration.max_metric_difference:.1f}** and a maximum
out-of-fold probability difference of
**{migration.max_oof_probability_difference:.1f}**, so adopting it changes
nothing about what is computed. The legacy builder `build_logistic_baseline` is
**preserved for reproducing the historical artefacts only** and must not be used
for any new experiment.

### The order Phase 9 has to run in

The holdout stays untouched until **every** development decision is frozen. That
is not a formality. A threshold chosen while looking at holdout performance, or a
calibration adopted because it improved a holdout number, turns the final
evaluation into a selection score — and this project would then have no unbiased
estimate anywhere.

{(chr(10) * 2).join(f"**{step[:2]}** {step[2:].strip()}" for step in selection.phase9_sequence)}

Step E is the one that is easiest to violate by accident. Reading the holdout
errors and then adjusting a threshold, a feature, a hyperparameter or the
calibration would mean the reported holdout number no longer describes the model
that produced it.

## 21. Warnings

{_warning_table(results)}

The legacy logistic regression still raises the `penalty` removal warning, which
is why the migration in section 5 exists. The modern builder used by every
Phase 8B experiment does not raise it. Nothing was suppressed.
"""


def _reproduction_table(results: TuningResults) -> str:
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


def _emits_penalty_warning(captured) -> bool:
    return any("penalty" in warning.message for warning in captured)


def _log_warnings(captured) -> None:
    for warning in captured:
        logger.warning(
            "%s x%d from %s: %s", warning.category, warning.count, warning.source, warning.message
        )


def main() -> int:
    """Run the nested tuning study. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()

    try:
        verify_split_manifest(load_split_manifest())
        baseline_reference = load_baseline_results()
        ablation_reference = load_ablation_results()
        comparison_reference = load_comparison_results()
        representation_reference = load_representation_results()
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

    outer = build_splitter()
    inner = build_inner_splitter()
    logger.info(
        "Search spaces: T1 %d candidates, T2 %d candidates",
        n_candidates(T1_SEARCH_SPACE),
        n_candidates(T2_SEARCH_SPACE),
    )

    # --- gate 1: the compatibility migration ---------------------------------
    with collect_warnings() as legacy_warnings:
        legacy = evaluate_model(
            "legacy_logistic",
            build_family_pipeline(LOGISTIC_REGRESSION, ()),
            features,
            target,
            outer,
        )
    with collect_warnings() as modern_warnings:
        modern = evaluate_model(
            "modern_logistic", build_modern_logistic_pipeline(), features, target, outer
        )

    try:
        migration = verify_migration(
            legacy,
            modern,
            legacy_warned=_emits_penalty_warning(legacy_warnings),
            modern_warned=_emits_penalty_warning(modern_warnings),
        )
    except ReproductionError as error:
        logger.error("%s", error)
        return 4
    logger.info("Compatibility migration: %s", migration.classification)

    # --- the three procedures -------------------------------------------------
    checks: list[ReproductionCheck] = []
    runs: OrderedDict[str, NestedRun] = OrderedDict()
    metrics = (PRIMARY_METRIC, SECONDARY_METRIC)

    with collect_warnings() as captured:
        runs[FROZEN_LOGISTIC] = run_nested_cv(
            FROZEN_LOGISTIC,
            "Frozen logistic regression (C=1.0, untuned)",
            build_modern_logistic_pipeline(),
            features,
            target,
            outer,
        )
        try:
            checks.append(verify_baseline_reproduction(runs[FROZEN_LOGISTIC], baseline_reference))
        except ReproductionError as error:
            logger.error("%s", error)
            return 5

        runs[TUNED_LOGISTIC] = run_nested_cv(
            TUNED_LOGISTIC,
            "Tuned logistic regression (C searched)",
            build_modern_logistic_pipeline(),
            features,
            target,
            outer,
            T1_SEARCH_SPACE,
            inner,
        )
        runs[TUNED_HGB] = run_nested_cv(
            TUNED_HGB,
            "Tuned histogram gradient boosting",
            build_hgb_pipeline(),
            features,
            target,
            outer,
            T2_SEARCH_SPACE,
            inner,
        )

        pair(runs[TUNED_LOGISTIC], runs[FROZEN_LOGISTIC], metrics)
        pair(runs[TUNED_HGB], runs[FROZEN_LOGISTIC], metrics)
        pair(runs[TUNED_HGB], runs[TUNED_LOGISTIC], metrics)

        # --- the pre-registered decision -------------------------------------
        n_folds = len(runs[FROZEN_LOGISTIC].folds)
        statuses: dict[str, str] = {}
        eligibility_records = []
        for experiment in (TUNED_LOGISTIC, TUNED_HGB):
            comparison = runs[experiment].paired_deltas[FROZEN_LOGISTIC][PRIMARY_METRIC]
            status, rationale = eligibility(comparison, n_folds)
            statuses[experiment] = status
            eligibility_records.append(
                build_eligibility_record(experiment, status, rationale, comparison)
            )
            logger.info("%s: %s", experiment, status)

        head_to_head = runs[TUNED_HGB].paired_deltas[TUNED_LOGISTIC][PRIMARY_METRIC]
        selected, rationale = select_candidate(statuses, head_to_head, n_folds)
        logger.info("Selected candidate: %s", selected)

        # --- final search, only once the nested evaluation is complete --------
        final_searches: list[FinalSearchRecord] = []
        for experiment, space, pipeline in (
            (TUNED_LOGISTIC, T1_SEARCH_SPACE, build_modern_logistic_pipeline()),
            (TUNED_HGB, T2_SEARCH_SPACE, build_hgb_pipeline()),
        ):
            if statuses[experiment] != ELIGIBLE and experiment != selected:
                continue
            best_params, best_score = run_final_search(pipeline, space, features, target, outer)
            final_searches.append(
                FinalSearchRecord(
                    experiment=experiment,
                    best_params={
                        key.split("__", 1)[-1]: value for key, value in best_params.items()
                    },
                    best_cv_score=round(best_score, 6),
                    scoring="average_precision",
                    is_generalisation_estimate=False,
                    note=(
                        "Searched over the whole training pool to fix the values carried into "
                        "Phase 9. The score is computed on the same rows used to rank the "
                        "candidates and is optimistically biased by construction; the honest "
                        "estimate of this procedure is its nested outer result."
                    ),
                )
            )
            logger.info("Final search %s: %s", experiment, best_params)

    _log_warnings(captured)

    # T0 is the frozen baseline in its modern spelling; the full configuration is
    # recorded, not just C, so Phase 9 instantiates the estimator from the
    # artefact instead of from a reader's memory of what the defaults are.
    frozen_hyperparameters: dict[str, object] = {
        "C": 1.0,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "class_weight": None,
        "max_iter": 100,
    }
    if selected != FROZEN_LOGISTIC:
        frozen_hyperparameters = next(
            record.best_params for record in final_searches if record.experiment == selected
        )

    references = {
        "baseline": ArtefactReference(
            artefact="reports/experiments/baseline_results.json",
            schema_version=baseline_reference.schema_version,
            experiment=baseline_reference.experiment,
            raw_sha256=baseline_reference.raw_sha256,
            training_ids_sha256=baseline_reference.training_ids_sha256,
        ),
        "feature_engineering": ArtefactReference(
            artefact="reports/experiments/feature_engineering_results.json",
            schema_version=ablation_reference.schema_version,
            experiment=ablation_reference.experiment,
            raw_sha256=ablation_reference.raw_sha256,
            training_ids_sha256=ablation_reference.training_ids_sha256,
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

    results = build_results(
        runs,
        migration,
        checks,
        eligibility_records,
        build_selection_record(
            selected,
            rationale,
            frozen_hyperparameters,
            runs[selected].label,
        ),
        final_searches,
        baseline_reference,
        comparison_reference.metric_protocol,
        references,
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
            "%s: outer AP %.4f, outer-OOF AP %.4f",
            experiment,
            run.mean(PRIMARY_METRIC),
            run.oof_metrics.average_precision,
        )
    logger.info("Frozen candidate for Phase 9: %s %s", selected, frozen_hyperparameters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
