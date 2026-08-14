"""Run the Phase 8A representation gate and write its report.

Usage (from the repository root):

    uv run python scripts/build_split.py --verify        # prove the split
    uv run python scripts/run_hgb_representation.py

Writes:
    reports/experiments/hgb_representation_results.json
    reports/hgb_representation_report.md
    reports/figures/hgb_representation/*.png

Only the frozen training pool is loaded. The holdout is never read. The folds,
the metrics and the 0.5 diagnostic rule are frozen at their Phase 5 values, and
every hyperparameter is frozen at its Phase 7 value: this script varies one
thing, the categorical representation.

Two gates stop the run before anything is interpreted: R0 must reproduce M0 and
R1 must reproduce M2 of the Phase 7 comparison.
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
from churn.modeling.comparison import collect_warnings
from churn.modeling.comparison_plots import plot_metric_by_family, plot_paired_deltas
from churn.modeling.comparison_results import ReproductionCheck, ReproductionError
from churn.modeling.comparison_results import load_results as load_comparison_results
from churn.modeling.evaluation import build_splitter
from churn.modeling.metrics import positive_prevalence
from churn.modeling.plots import OOF_CAPTION, plot_precision_recall_curves
from churn.modeling.representation import (
    CATEGORICAL_MASK,
    NATIVE_HGB,
    REFERENCE_HGB_COMMON,
    REFERENCE_LOGISTIC,
    CategoricalCardinalityError,
    RepresentationRun,
    observed_cardinality,
    run_representation_gate,
)
from churn.modeling.representation_results import (
    RESULTS_PATH,
    RepresentationResults,
    build_results,
    verify_against_comparison,
    write_results,
)
from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_matrix,
)
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target

logger = logging.getLogger("run_hgb_representation")

REPORT_PATH = PROJECT_ROOT / "reports" / "hgb_representation_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "hgb_representation"

METRIC_LABELS = {
    "roc_auc": "ROC-AUC (secondary)",
    "average_precision": "Average Precision (primary)",
    "precision": "Precision @0.5",
    "recall": "Recall @0.5",
    "f1": "F1 @0.5",
    "accuracy": "Accuracy @0.5",
}
DIAGNOSTIC_METRICS = ("precision", "recall", "f1", "accuracy")


def build_figures(
    results: RepresentationResults,
    target: np.ndarray,
    oof: dict[str, np.ndarray],
    root: Path,
) -> dict[str, Path]:
    """Render the three Phase 8A figures and return a name -> path mapping."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    records = {record.experiment: record for record in results.experiments}
    native = records[NATIVE_HGB]
    paths: dict[str, Path] = {}

    paths["delta"] = save_figure(
        plot_paired_deltas(
            {
                native.model: native.paired_deltas[REFERENCE_HGB_COMMON][PRIMARY_METRIC].per_fold,
            },
            f"{REFERENCE_HGB_COMMON} same family, common representation",
            "Average Precision",
            "Representation effect inside HistGradientBoosting — R2 native vs R1 one-hot",
            "Dots: one per fold.  Vertical bar: mean of the five deltas.  Right of the line: "
            "the native representation won that fold. Same estimator, same folds, same "
            "hyperparameters — only the categorical representation differs.",
            labels={native.model: f"{native.experiment} · native categorical"},
        ),
        figures_dir / "01_paired_ap_representation_delta.png",
    )

    paths["absolute"] = save_figure(
        plot_metric_by_family(
            {
                record.model: [fold.metrics[PRIMARY_METRIC] for fold in record.folds]
                for record in results.experiments
            },
            records[REFERENCE_LOGISTIC].model,
            "Average Precision",
            "Average Precision per fold — logistic, HGB one-hot, HGB native categorical",
        ),
        figures_dir / "02_absolute_ap_comparison.png",
    )

    paths["pr"] = save_figure(
        plot_precision_recall_curves(
            target,
            oof,
            {
                record.model: record.out_of_fold[PRIMARY_METRIC]
                for record in results.experiments
                if record.model in oof
            },
            results.training_positive_prevalence,
            f"Precision-recall — {OOF_CAPTION}",
        ),
        figures_dir / "03_oof_precision_recall.png",
    )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _absolute_table(results: RepresentationResults) -> str:
    lines = [
        "| Experiment | Model | Representation | Transformed | "
        + " | ".join(METRIC_LABELS[metric] for metric in METRIC_LABELS)
        + " |",
        "| --- " * (len(METRIC_LABELS) + 4) + "|",
    ]
    for record in results.experiments:
        cells = " | ".join(
            f"{record.mean[metric]:.4f} ± {record.std[metric]:.4f}" for metric in METRIC_LABELS
        )
        lines.append(
            f"| {record.experiment} | `{record.model}` | `{record.representation}` | "
            f"{record.n_transformed_features} | {cells} |"
        )
    return "\n".join(lines)


def _fold_table(results: RepresentationResults, metric: str) -> str:
    lines = [
        "| Experiment | Model | F1 | F2 | F3 | F4 | F5 | Mean ± SD |",
        "| --- " * 8 + "|",
    ]
    for record in results.experiments:
        cells = " | ".join(f"{fold.metrics[metric]:.4f}" for fold in record.folds)
        lines.append(
            f"| {record.experiment} | `{record.model}` | {cells} | "
            f"**{record.mean[metric]:.4f}** ± {record.std[metric]:.4f} |"
        )
    return "\n".join(lines)


def _delta_table(results: RepresentationResults, reference: str) -> str:
    record = next(item for item in results.experiments if item.experiment == NATIVE_HGB)
    lines = [
        "| Metric | Reference | F1 | F2 | F3 | F4 | F5 | Mean | SD | +/− folds |",
        "| --- " * 10 + "|",
    ]
    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
        delta = record.paired_deltas[reference][metric]
        cells = " | ".join(f"{value:+.4f}" for value in delta.per_fold)
        lines.append(
            f"| {METRIC_LABELS[metric]} | {reference} | {cells} | **{delta.mean:+.4f}** | "
            f"{delta.std:.4f} | {delta.folds_improved}/{delta.folds_worsened} |"
        )
    return "\n".join(lines)


def _oof_table(results: RepresentationResults) -> str:
    lines = [
        "| Experiment | Model | " + " | ".join(METRIC_LABELS[metric] for metric in METRIC_LABELS),
        "| --- " * (len(METRIC_LABELS) + 2) + "|",
    ]
    lines[0] += " |"
    for record in results.experiments:
        cells = " | ".join(f"{record.out_of_fold[metric]:.4f}" for metric in METRIC_LABELS)
        lines.append(f"| {record.experiment} | `{record.model}` | {cells} |")
    return "\n".join(lines)


def _reproduction_table(results: RepresentationResults) -> str:
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


def _cardinality_table(results: RepresentationResults) -> str:
    native = results.representations["native_categorical"]
    lines = ["| Feature | Categories in the training pool |", "| --- | --- |"]
    lines.extend(
        f"| `{column}` | {count} |"
        for column, count in native.observed_training_cardinality.items()
    )
    return "\n".join(lines)


def _warning_table(results: RepresentationResults) -> str:
    if not results.warnings:
        return "No warning was raised during the run."
    lines = ["| Category | Origin | Count | Message |", "| --- " * 4 + "|"]
    for warning in results.warnings:
        message = warning.message.replace("|", "\\|")
        lines.append(f"| `{warning.category}` | `{warning.source}` | {warning.count} | {message} |")
    return "\n".join(lines)


def build_report(
    results: RepresentationResults,
    figures: dict[str, Path],
    generated_on: str,
) -> str:
    """Render the Phase 8A report. Every number comes from this run."""
    records = {record.experiment: record for record in results.experiments}
    logistic, common, native = (
        records[REFERENCE_LOGISTIC],
        records[REFERENCE_HGB_COMMON],
        records[NATIVE_HGB],
    )
    cv = results.cross_validation
    cv_signature = (
        f"{cv['strategy']}(n_splits={cv['n_splits']}, shuffle={cv['shuffle']}, "
        f"random_state={cv['random_state']})"
    )
    native_definition = results.representations["native_categorical"]
    declared_explicitly = native_definition.categorical_features_declared_explicitly
    verdict = results.verdict
    n_folds = len(native.folds)

    against_common = native.paired_deltas[REFERENCE_HGB_COMMON][PRIMARY_METRIC]
    against_logistic = native.paired_deltas[REFERENCE_LOGISTIC][PRIMARY_METRIC]
    gap_closed = against_common.mean
    original_gap = common.mean[PRIMARY_METRIC] - logistic.mean[PRIMARY_METRIC]
    # A fraction of a shortfall only reads as one when the movement points at it.
    # Expressing a negative or zero movement as a negative percentage of a gap
    # invites reading it as a partial recovery, which it is not.
    if original_gap < 0 < gap_closed:
        recovered = f"closing about **{gap_closed / abs(original_gap):.0%}** of that distance"
    elif original_gap < 0:
        recovered = (
            "closing **none** of that distance — the movement does not point towards the "
            "reference model"
        )
    else:
        recovered = "measured against a Phase 7 position that was not a shortfall"

    return f"""# HistGradientBoosting Representation Gate — Telco Customer Churn (Phase 8A)

- Generated on: {generated_on}
- Generated by: `scripts/run_hgb_representation.py`
- Machine-readable record: `reports/experiments/hgb_representation_results.json`
- Raw SHA-256: `{results.raw_sha256}`
- Training partition fingerprint: `{results.training_ids_sha256}`

> **Every number here is cross-validated on the training pool.** No holdout row
> was loaded, transformed, predicted or measured. There is no test result in
> this repository before Phase 9.

---

## 1. Objective

One question, and nothing else:

> How much of the difference between HistGradientBoosting and logistic
> regression is sensitive to the representation of the categorical features?

**This is not tuning.** No hyperparameter was searched or changed, no threshold
was moved, no engineered feature was added, no model was selected.

## 2. Why this gate exists

Phase 7 held the representation constant so that the estimator was the only
variable, and recorded that a common representation is not necessarily optimal
for every family — that its results estimate the performance of the tested
configurations under a common representation, not the ceiling of each algorithm.

That leaves one factor unmeasured and one interpretation unavailable. The
{abs(original_gap):.4f} Average Precision by which
`{common.model}` sat below `{logistic.model}` in Phase 7 could be a property of
the family, of the representation, or of both. Tuning a family before knowing
which would be searching hyperparameters to compensate for an encoding — an
expensive way to answer the wrong question.

So this phase changes **one factor**: how the 16 contracted categorical columns
reach the estimator.

## 3. Frozen experimental protocol

| Element | Value |
| --- | --- |
| Data | Frozen Phase 4 training pool, {results.n_training_rows:,} rows |
| Positive prevalence | {results.training_positive_prevalence:.4f} |
| Cross-validation | `{cv_signature}` |
| Identical folds as Phases 5-7 | {cv["identical_to_model_comparison"]} |
| Primary metric | Average Precision |
| Secondary ranking metric | ROC-AUC |
| Diagnostics @ {results.decision_threshold} | Precision, Recall, F1 (Accuracy auxiliary) |
| Threshold optimised | {results.metric_protocol["threshold_optimised"]} |
| Hyperparameter tuning | {results.tuning_performed} |
| Calibration | {results.calibration_performed} |
| Engineered features | {results.engineered_features_used or "none"} |
| Seed | {results.random_seed} |
| scikit-learn | {results.scikit_learn_version} |

**Preprocessing is refitted inside every fold** for all three experiments: the
scaler's statistics, the encoder's categories and the estimator's parameters are
estimated on the training fold alone.

`contract_tenure` is **not** used. It remains the retained candidate Phase 6
documented. Adding it here would put a feature effect and a representation
effect into the same number, and neither could then be recovered.

## 4. Reference reproduction

Two gates run before anything below is interpreted:

{_reproduction_table(results)}

R0 and R1 are not re-declared — they are rebuilt through the **Phase 7 pipeline
builder**, so they are the same objects that produced `model_comparison_results.json`.
Every per-fold metric is compared against that artefact and the run aborts if
either fails. Without this, a difference attributed to the representation could
be an artefact of the pipeline being assembled differently.

## 5. Common representation (R0, R1)

The Phase 4 block, unchanged:

```text
TotalChargesCleaner → StandardScaler (numeric) → OneHotEncoder(handle_unknown="ignore")
```

{logistic.n_transformed_features} transformed columns:
{len(NUMERIC_FEATURES)} scaled numeric and
{logistic.n_transformed_features - len(NUMERIC_FEATURES)} one-hot indicators
derived from the {len(CATEGORICAL_FEATURES)} categorical features.

## 6. Native categorical representation (R2)

```text
prepare_features
  → TotalChargesCleaner
  → cardinality guard (training fold only)
  → ColumnTransformer
       numeric      : {", ".join(NUMERIC_FEATURES)} → StandardScaler
       categorical  : the same {len(CATEGORICAL_FEATURES)} features → OrdinalEncoder
  → HistGradientBoostingClassifier(categorical_features=<explicit mask>)
```

| Property | Value |
| --- | --- |
| Raw features | {native_definition.raw_features} (the same 19; none added, none removed) |
| Transformed columns | **{native_definition.n_transformed_features}** |
| Numeric columns | {native_definition.n_numeric_columns} |
| Categorical columns | {native_definition.n_categorical_columns} |
| `categorical_features` | explicit positional mask ({declared_explicitly}) |
| Mask indices | {native_definition.categorical_indices} |
| Cardinality limit | {native_definition.max_categorical_cardinality} (`max_bins`) |
| Engineered features | {native_definition.engineered_features or "none"} |

The mask is `[False × {native_definition.n_numeric_columns}, True × \
{native_definition.n_categorical_columns}]`, matching the `ColumnTransformer`
output order: the numeric block first, then the categorical one. It is stated
positionally and **never** inferred through `categorical_features="from_dtype"`,
because a representation whose nominality depends on an upstream dtype is not a
guarantee of anything.

**The ordinal codes are not an ordinal assumption.** `OrdinalEncoder` assigns
integers in sorted order and nothing about those integers is meaningful. What
keeps the columns nominal is the second half of the arrangement: the same
{native_definition.n_categorical_columns} positions are declared categorical to
the estimator, so the split finder partitions **sets of categories** instead of
thresholding a number line. The encoder alone would be an ordinal assumption;
the encoder plus the declaration is not.

The estimator configuration is otherwise **identical** to Phase 7's M2 — it is
built from the same frozen parameter dictionary, with `categorical_features`
replaced. `early_stopping` stays `False`, so no model selection happens inside a
training fold.

## 7. Leakage-safe categorical vocabulary

The encoder is fitted **inside each training fold**. A category that appears
only in a validation fold never enters `categories_`; it is encoded as `NaN` and
routed to the estimator's native missing-value handling at split time.

That `NaN` is **serving behaviour for a new but valid category, not blank
handling**. Blanks, `None` and whitespace-only values are rejected earlier by the
Phase 4 feature contract, which has not been relaxed. A new valid category and
invalid input remain different things and are handled at different boundaries.

Cardinality is checked **in the training fold only**, by a guard that raises
rather than repairs. No column is grouped, hashed or truncated to fit the
estimator's {native_definition.max_categorical_cardinality}-category limit:
silently collapsing levels would change the representation under test while the
experiment kept claiming only the encoding had changed.

Observed in the training pool — every column is far below the limit, so the
guard did not fire:

{_cardinality_table(results)}

## 8. HGB representation comparison

Absolute cross-validated performance, mean ± sample standard deviation over
{cv["n_splits"]} folds:

{_absolute_table(results)}

{_figure(figures["absolute"], "Average Precision per fold — the three experiments")}

Per-fold Average Precision:

{_fold_table(results, PRIMARY_METRIC)}

## 9. Paired Average Precision deltas — R2 vs R1 (question A)

The representation effect **inside one family**. Same estimator, same
hyperparameters, same folds; only the categorical representation differs:

{_delta_table(results, REFERENCE_HGB_COMMON)}

{_figure(figures["delta"], "Paired per-fold deltas, native categorical vs one-hot")}

**Classification: {verdict.representation_classification}.**
{verdict.representation_rationale}

The labels — `REPRESENTATION_HELPFUL`, `REPRESENTATION_INCONCLUSIVE`,
`REPRESENTATION_NOT_SUPPORTED` — are engineering heuristics about direction and
consistency, fixed before any result was seen. They carry **no minimum-gain
cutoff**, and with five folds no significance test is possible or was run.

Read against Phase 7: the native representation moves this family by
{gap_closed:+.5f} AP, against a Phase 7 shortfall of
{original_gap:+.5f} AP versus the logistic regression — {recovered}.

That is the direct answer to the question this phase asks. **Under this
protocol, on this dataset, the categorical representation accounts for very
little of the distance between the two families.**

A **plausible mechanism, which this experiment did not measure**: the highest
cardinality among the {len(CATEGORICAL_FEATURES)} categorical features is
{max(native_definition.observed_training_cardinality.values())} (section 7).
Native categorical handling is generally expected to pay off when a feature has
many levels, because one-hot then scatters one variable across many sparse
indicator columns that a tree must recombine through depth. That condition is
not present in this dataset. Stated as a hypothesis consistent with the result —
testing it would require varying cardinality, which is not this gate.

## 10. Comparison against logistic regression (question B)

A different question, deliberately not mixed with the one above: the delta below
compares R2 against **R0**, not against R1.

{_delta_table(results, REFERENCE_LOGISTIC)}

**Standing: {verdict.standing_against_logistic}.** {verdict.standing_rationale}

A representation change can help a family measurably and still leave it behind
the reference model. Those two statements are compatible, and keeping them in
separate sections is what stops the first from being read as the second.

## 11. Out-of-fold results

Each of the {results.n_training_rows:,} training rows received exactly one
prediction, produced by a model that never saw that row while fitting:

{_oof_table(results)}

{_figure(figures["pr"], "Precision-recall from out-of-fold predictions on the training pool")}

These are **cross-validated out-of-fold results on the training pool**, not test
metrics. No per-customer prediction was persisted: the identifier is not part of
the feature matrix and no `customerID`-to-probability mapping is written
anywhere.

Threshold-0.5 diagnostics, which selected nothing:

| Experiment | Model | {" | ".join(METRIC_LABELS[m] for m in DIAGNOSTIC_METRICS)} |
| --- | --- | --- | --- | --- | --- |
{
        chr(10).join(
            f"| {record.experiment} | `{record.model}` | "
            + " | ".join(f"{record.mean[metric]:.4f}" for metric in DIAGNOSTIC_METRICS)
            + " |"
            for record in results.experiments
        )
    }

## 12. Complexity implications

| Property | Common representation | Native categorical |
| --- | --- | --- |
| Transformed columns | {common.n_transformed_features} | \
{native.n_transformed_features} |
| Fitted encoder state | category lists + one-hot layout | category lists only |
| Unseen category at serving | all-zero block (silent) | `NaN`, routed to the \
estimator's missing branch |
| Estimator coupling | none — any estimator accepts the matrix | the mask must \
match the column order, and only some estimators consume it |
| Interpretability | one column per level | one column per feature, split sets \
must be decoded |

The native representation is **narrower** but more **coupled**: the mask is
positional, so any future change to the `ColumnTransformer` order silently
changes what the estimator treats as categorical. That is a real maintenance
cost and is the reason the mask is asserted in the tests rather than assumed.

Neither representation is "simpler" outright. One trades width for coupling.

## 13. Limitations

- **No holdout estimate.** Everything is training-pool cross-validation.
- **Five folds, no significance test.** The deltas and the classification
  describe direction and consistency; they are not statistical claims.
- **Still untuned.** Both HGB experiments run the frozen base configuration, so
  neither result is that family's ceiling under either representation.
- **One family.** The gate was run for HistGradientBoosting only. Nothing here
  says what a native representation would do for a random forest — scikit-learn's
  forest does not consume one — or for any other family.
- **One representation change.** Ordinal-plus-declaration is not the only
  alternative encoding; target and frequency encodings exist and were not tested,
  the first being a leakage pattern this project rejects outright.
- **No calibration.** Probability quality was not measured and ranked nothing.
- **No class-imbalance handling.** `class_weight` is `None` everywhere.
- **The analyst-exposure limitation from Phase 3 still applies.**
- **Snapshot classification, not forecasting.** The dataset has no time
  dimension.

## 14. Decision intentionally deferred

**No tuning candidate was selected here, and no tuning was performed.**
{verdict.decision_note}

Selecting a candidate inside the same run that produced the evidence would make
the selection a function of a single result, which is precisely the failure mode
the phase separation exists to prevent. What follows is evidence for a reviewer,
not a recommendation the script is authorised to act on.

## 15. Evidence to inform tuning eligibility

Stated as measurements, in the order a reviewer needs them:

| Question | Measurement |
| --- | --- |
| Does the representation move the family? | {against_common.mean:+.5f} AP, \
{against_common.folds_improved}/{n_folds} folds → {verdict.representation_classification} |
| Does the representation move the ranking metric? | \
{native.paired_deltas[REFERENCE_HGB_COMMON][SECONDARY_METRIC].mean:+.5f} ROC-AUC |
| Where does R2 stand against the reference model? | {against_logistic.mean:+.5f} AP, \
{against_logistic.folds_improved}/{n_folds} folds → {verdict.standing_against_logistic} |
| Best mean AP in this phase | \
{max(results.experiments, key=lambda r: r.mean[PRIMARY_METRIC]).experiment} \
({max(record.mean[PRIMARY_METRIC] for record in results.experiments):.4f}) |
| Was anything tuned, thresholded or calibrated? | No |

The reviewer's decision is which models, if any, justify a tuning budget — and
that decision should weigh the size of these deltas against the interpretability
and coupling costs in section 12, not the ordering alone.

Nothing here has been tuned, calibrated, thresholded or evaluated on the
holdout, and no model has been selected.
"""


def _log_warnings(captured) -> None:
    for warning in captured:
        logger.warning(
            "%s x%d from %s: %s", warning.category, warning.count, warning.source, warning.message
        )


def main() -> int:
    """Run the representation gate. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()

    try:
        verify_split_manifest(load_split_manifest())
        comparison_reference = load_comparison_results()
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

    try:
        cardinality = observed_cardinality(features)
    except CategoricalCardinalityError as error:
        logger.error("%s", error)
        return 5
    logger.info(
        "Categorical cardinality (training pool): max %d over %d columns, mask %d/%d categorical",
        max(cardinality.values()),
        len(cardinality),
        sum(CATEGORICAL_MASK),
        len(CATEGORICAL_MASK),
    )

    splitter = build_splitter()
    checks: list[ReproductionCheck] = []

    def gate(run: RepresentationRun) -> None:
        checks.append(verify_against_comparison(run, comparison_reference))

    with collect_warnings() as captured:
        try:
            runs = run_representation_gate(
                features,
                target,
                splitter,
                (PRIMARY_METRIC, SECONDARY_METRIC),
                gate,
            )
        except ReproductionError as error:
            logger.error("%s", error)
            return 4

    _log_warnings(captured)

    results = build_results(runs, checks, comparison_reference, cardinality, labels, captured)
    write_results(results)

    oof = {run.model: run.evaluation.oof_probability for run in runs.values()}
    figures = build_figures(results, labels, oof, PROJECT_ROOT)
    Path(REPORT_PATH).write_text(
        build_report(results, figures, date.today().isoformat()),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)
    logger.info(
        "Representation: %s. Standing vs logistic: %s. No tuning candidate selected.",
        results.verdict.representation_classification,
        results.verdict.standing_against_logistic,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
