"""Cross-validate the Phase 5 baselines and write the report.

Usage (from the repository root):

    uv run python scripts/build_split.py --verify   # first: prove the split
    uv run python scripts/run_baselines.py

Writes:
    reports/experiments/baseline_results.json
    reports/baseline_report.md
    reports/figures/baselines/*.png

Only the frozen training pool is loaded. The holdout is never read, and no
metric, distribution or prediction is computed over it. No model is persisted:
Phase 5 measures how much signal the 19 original features carry, it does not
produce a deliverable model.
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
from churn.modeling.evaluation import build_splitter, evaluate_models
from churn.modeling.metrics import DEFAULT_THRESHOLD, confusion_at_threshold, positive_prevalence
from churn.modeling.models import LOGISTIC_REGRESSION, MAJORITY, STRATIFIED_RANDOM, build_baselines
from churn.modeling.plots import (
    OOF_CAPTION,
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from churn.modeling.results import RESULTS_PATH, BaselineResults, build_results, write_results
from churn.preprocessing.contracts import FEATURE_COLUMNS, build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target

logger = logging.getLogger("run_baselines")

REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "baselines"

#: Metrics shown in the comparison figure. Accuracy is left out on purpose: it
#: would put the majority baseline on top and invite exactly the misreading the
#: figure exists to prevent.
COMPARISON_METRICS = ("roc_auc", "average_precision", "precision", "recall", "f1")
METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Average\nPrecision",
    "precision": "Precision\n@0.5",
    "recall": "Recall\n@0.5",
    "f1": "F1\n@0.5",
    "accuracy": "Accuracy\n@0.5",
}
REPORT_METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "average_precision": "Average Precision",
    "precision": "Precision @0.5",
    "recall": "Recall @0.5",
    "f1": "F1 @0.5",
    "accuracy": "Accuracy @0.5",
}


def build_figures(
    results: BaselineResults,
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
    root: Path,
) -> dict[str, Path]:
    """Render every baseline figure and return a name -> path mapping."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    by_name = {model.name: model for model in results.models}
    summaries = {
        name: {
            metric: {"mean": model.mean[metric], "std": model.std[metric]}
            for metric in METRIC_LABELS
        }
        for name, model in by_name.items()
    }
    paths: dict[str, Path] = {}

    paths["comparison"] = save_figure(
        plot_metric_comparison(
            summaries,
            COMPARISON_METRICS,
            METRIC_LABELS,
            "Only the logistic regression separates churners — 5-fold CV on the training pool",
        ),
        figures_dir / "01_cv_metric_comparison.png",
    )

    paths["roc"] = save_figure(
        plot_roc_curves(
            target,
            probabilities,
            {name: model.out_of_fold["roc_auc"] for name, model in by_name.items()},
            f"ROC — {OOF_CAPTION}",
        ),
        figures_dir / "02_oof_roc_curves.png",
    )

    paths["pr"] = save_figure(
        plot_precision_recall_curves(
            target,
            probabilities,
            {name: model.out_of_fold["average_precision"] for name, model in by_name.items()},
            results.training_positive_prevalence,
            f"Precision-Recall — {OOF_CAPTION}",
        ),
        figures_dir / "03_oof_precision_recall_curves.png",
    )

    paths["confusion"] = save_figure(
        plot_confusion_matrix(
            confusion_at_threshold(target, probabilities[LOGISTIC_REGRESSION], DEFAULT_THRESHOLD),
            "Logistic regression at the 0.5 reference rule",
            f"{OOF_CAPTION} — the threshold is not tuned",
        ),
        figures_dir / "04_oof_confusion_matrix_logistic.png",
    )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _fold_table(results: BaselineResults, metric: str) -> str:
    header = "| Model | " + " | ".join(f"Fold {index}" for index in range(1, 6)) + " | Mean ± SD |"
    divider = "| --- " * 7 + "|"
    lines = [header, divider]
    for model in results.models:
        cells = " | ".join(f"{fold.metrics[metric]:.4f}" for fold in model.folds)
        lines.append(
            f"| `{model.name}` | {cells} | **{model.mean[metric]:.4f}** ± {model.std[metric]:.4f} |"
        )
    return "\n".join(lines)


def _summary_table(results: BaselineResults) -> str:
    lines = [
        "| Model | " + " | ".join(REPORT_METRIC_LABELS[m] for m in REPORT_METRIC_LABELS) + " |",
        "| --- " * (len(REPORT_METRIC_LABELS) + 1) + "|",
    ]
    for model in results.models:
        cells = " | ".join(
            f"{model.mean[metric]:.4f} ± {model.std[metric]:.4f}" for metric in REPORT_METRIC_LABELS
        )
        lines.append(f"| `{model.name}` | {cells} |")
    return "\n".join(lines)


def _oof_table(results: BaselineResults) -> str:
    lines = [
        "| Model | " + " | ".join(REPORT_METRIC_LABELS[m] for m in REPORT_METRIC_LABELS) + " |",
        "| --- " * (len(REPORT_METRIC_LABELS) + 1) + "|",
    ]
    for model in results.models:
        cells = " | ".join(f"{model.out_of_fold[metric]:.4f}" for metric in REPORT_METRIC_LABELS)
        lines.append(f"| `{model.name}` | {cells} |")
    return "\n".join(lines)


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def build_report(
    results: BaselineResults,
    figures: dict[str, Path],
    n_protected_rows: int,
    generated_on: str,
) -> str:
    """Render the Phase 5 report. Every number comes from this run."""
    by_name = {model.name: model for model in results.models}
    majority = by_name[MAJORITY]
    random_model = by_name[STRATIFIED_RANDOM]
    logistic = by_name[LOGISTIC_REGRESSION]
    prevalence = results.training_positive_prevalence
    cv = results.cross_validation
    cv_signature = (
        f"{cv.strategy}(n_splits={cv.n_splits}, shuffle={cv.shuffle}, "
        f"random_state={cv.random_state})"
    )
    random_config = (
        f'`strategy="{random_model.hyperparameters["strategy"]}"`, '
        f"`random_state={random_model.hyperparameters['random_state']}`"
    )
    logistic_config = ", ".join(
        f"`{key}={logistic.hyperparameters[key]!r}`"
        for key in ("C", "penalty", "class_weight", "solver", "max_iter")
    )
    recalls = [fold.metrics["recall"] for fold in logistic.folds]
    recall_low, recall_high = min(recalls), max(recalls)
    protocol = results.metric_protocol
    confusion = logistic.out_of_fold_confusion
    caught = confusion["true_positive"] + confusion["false_negative"]
    flagged = confusion["true_positive"] + confusion["false_positive"]
    convergence = (
        f"converged in every fold (at most {logistic.max_iterations_used} lbfgs iterations "
        f"against a `max_iter` of {logistic.hyperparameters['max_iter']})"
        if logistic.converged
        else "**did not converge in at least one fold**"
    )

    return f"""# Baseline Report — Telco Customer Churn (Phase 5)

- Generated on: {generated_on}
- Generated by: `scripts/run_baselines.py`
- Machine-readable record: `reports/experiments/baseline_results.json`
- Raw SHA-256: `{results.raw_sha256}`
- Training partition fingerprint: `{results.training_ids_sha256}`

> **Every number in this report is cross-validated on the training pool.**
> There is no holdout metric here, and there will be none before Phase 9. Curves
> and confusion matrices are out-of-fold predictions on the same
> {results.n_training_rows:,} training rows — never call them test results.

---

## 1. Objective

Measure how much predictive signal the **{results.feature_set.n_features} original
features** carry under the leakage-safe protocol frozen in Phase 4, and fix the
references every later phase has to beat:

1. a majority classifier — the floor;
2. a stratified random classifier — the no-skill reference;
3. an untuned logistic regression — the trainable baseline.

Nothing here is a candidate model. No feature was engineered, no hyperparameter
searched, no threshold moved, no imbalance technique applied.

## 2. Experimental protocol

| Element | Value |
| --- | --- |
| Data | Frozen Phase 4 training pool, {results.n_training_rows:,} rows |
| Features | {results.feature_set.n_features} original \
({len(results.feature_set.numeric)} numeric, {len(results.feature_set.categorical)} categorical) |
| Engineered features | none |
| Cross-validation | `{cv_signature}` |
| Shared folds | {results.cross_validation.shared_across_models} — the comparison is paired |
| Decision rule | {results.decision_threshold} (operational reference, not tuned) |
| Seed | {results.random_seed} (`configs/base.toml`) |
| scikit-learn | {results.scikit_learn_version} |

**Preprocessing is refitted inside every fold.** Each model is a complete
pipeline — `TotalChargesCleaner` → `ColumnTransformer` → classifier — cloned
fresh for each fold. The scaler's mean and standard deviation, the encoder's
category lists and the classifier's coefficients are all estimated on the
training fold alone; the validation fold is only transformed and predicted. No
`fit_transform` was ever called on the whole training pool before splitting,
which would have leaked every validation fold into every other fold.

All three models consume the identical `StratifiedKFold` partition, so a
difference between them cannot be an artefact of one drawing an easier split.

## 3. Holdout protection

The {results.n_training_rows:,} training rows are the entire universe of this
phase. The {n_protected_rows:,} protected rows were not loaded:
`load_training_pool()` discards them before returning, and the loader that would
read them is called nowhere in this repository. The only facts about that
partition that exist anywhere are the structural metadata already published in
`reports/split_manifest.json` — row count and identifier fingerprint — which
this phase did not change.

An automated audit enforces this: a test parses every Phase 5 module and the
baseline script with `ast` and fails if any of them so much as *names* the
protected loader or reads that partition. It inspects code, not prose, so this
paragraph does not trip it.

## 4. Models

| Name | Estimator | Effective configuration |
| --- | --- | --- |
| `{majority.name}` | `{majority.estimator}` | `strategy="{majority.hyperparameters["strategy"]}"` |
| `{random_model.name}` | `{random_model.estimator}` | {random_config} |
| `{logistic.name}` | `{logistic.estimator}` | {logistic_config} |

The two dummies carry the same preprocessing block even though they ignore the
features entirely. That is deliberate: the comparison is between protocols, and
giving one model a different pipeline shape would introduce a difference that
has nothing to do with the model.

`class_weight=None` and `C=1.0` are scikit-learn defaults, kept on purpose. This
is the reference point, not a candidate; making either of them a decision is
Phase 8's job. No `random_state` is passed to the logistic regression because
lbfgs is deterministic and does not consume one — supplying it would suggest a
source of randomness that does not exist.

**Convergence.** The logistic regression {convergence}. Convergence warnings are
captured explicitly per fold and recorded in the JSON artefact; none was
suppressed.

## 5. Metrics

**Threshold-independent** — how well the model *ranks* customers by risk:

- **ROC-AUC** (`roc_auc_score`);
- **Average Precision** (`average_precision_score`). Reported under that name
  and never as "PR-AUC": AP is a weighted sum of precisions at each threshold,
  while a trapezoidal PR-AUC interpolates between operating points and is
  optimistically biased. Naming it removes the ambiguity.

**Threshold-dependent** — what happens when the ranking becomes a decision at
the {results.decision_threshold} reference rule: Precision, Recall, F1.

**Accuracy is auxiliary.** It is computed and shown, but it ranks the majority
baseline first and was not used to select or promote anything. Where the
majority baseline predicts no positives at all, precision is undefined (0/0) and
is reported as `0.0` — a model that never fires has no precision to speak of.

### Metric hierarchy for the phases that follow

Recorded here and, as structured data, in `metric_protocol` inside the JSON
artefact, so Phase 6 reads the protocol instead of re-deciding it:

| Role | Metric |
| --- | --- |
| **Primary development metric** | **Average Precision** |
| Secondary ranking metric | ROC-AUC |
| Diagnostic at the {results.decision_threshold} rule | Precision, Recall, F1 |
| Auxiliary only | Accuracy |

**Why Average Precision leads.** The positive class is the minority and the task
is to discriminate churners. AP is computed entirely on positive-class
retrieval, so it degrades visibly when a model finds fewer churners. ROC-AUC is
dominated by the abundant negatives and moves less for the same loss of
churn-finding ability — which is why a model can look strong on ROC-AUC while
being mediocre at the thing the project exists to do. ROC-AUC stays as the
secondary view because it is prevalence-independent and therefore comparable
across samples.

**No single metric decides anything.** A metric hierarchy orders attention; it
does not replace judgement about trade-offs, added complexity and
interpretability. A later feature or model **does not have to improve every
metric at once** — a candidate that lifts AP while leaving ROC-AUC flat can still
be the right choice, and one that lifts both by a hair while doubling the
complexity may not be.

**The threshold is not optimised**, so the diagnostic trio is read as a
description of the {results.decision_threshold} operating point, not as a
performance ceiling.

## 6. Positive-class prevalence reference

Positive prevalence in the **training pool**: **{prevalence:.4f}**
({prevalence:.2%}).

This number exists only to make Average Precision readable. A ranking with no
signal converges to an AP equal to the positive prevalence, so an AP of
{prevalence:.3f} means "no better than chance", not "bad but non-zero". An AP
must always be read as a distance above this floor, never as an absolute score.
The equivalent floor for ROC-AUC is 0.500 and does not depend on prevalence —
which is exactly why the two metrics disagree about how impressive a model is on
imbalanced data.

No prevalence from the holdout was computed or used.

## 7. Cross-validation results

Mean ± sample standard deviation over the {results.cross_validation.n_splits} folds:

{_summary_table(results)}

{_figure(figures["comparison"], "Cross-validated metrics per model, with fold-to-fold spread")}

Per-fold ROC-AUC:

{_fold_table(results, "roc_auc")}

Per-fold Average Precision:

{_fold_table(results, "average_precision")}

Per-fold Recall at {results.decision_threshold}:

{_fold_table(results, "recall")}

### What the standard deviations do and do not mean

They describe how performance **varies across partitions** — the logistic
regression's recall ranges from {recall_low:.4f} to {recall_high:.4f} across the
five folds, so a single fold's number is not a stable estimate on its own.

They are **not** a margin of error for the difference between two models, and
must not be used as a bar a future gain has to clear. The two quantities answer
different questions: a between-fold SD measures dispersion of one model's
performance over different data, while a comparison of two models on the *same*
folds is a paired measurement in which the fold-to-fold difficulty that drives
most of that dispersion cancels out. A candidate can be consistently better than
the baseline by less than the baseline's own SD.

**Protocol for later phases.** Compare on the identical folds, per fold:

```text
delta_AP_fold = AP_candidate_fold - AP_baseline_fold
```

and judge the candidate on the **consistency and magnitude** of those deltas —
whether the difference points the same way in every fold, and how large it is.
No such comparison is performed here: this phase produces the reference values,
nothing more. No significance test was run either.

## 8. Out-of-fold results

Each of the {results.n_training_rows:,} training rows received exactly one
prediction, produced by a model that never saw that row during fitting. Metrics
recomputed over the pooled out-of-fold predictions:

{_oof_table(results)}

These are **cross-validated out-of-fold performance on the training pool**. They
are not test metrics and must never be relabelled as such.

{_figure(figures["roc"], "ROC curves from out-of-fold predictions on the training pool")}

{_figure(figures["pr"], "Precision-recall from out-of-fold predictions")}

The two dummies are **not drawn** as PR curves. Each assigns the same score to
every customer, so each has a single achievable operating point; joining the two
points `precision_recall_curve` returns would draw a long straight segment that
reads as a series of trade-offs that do not exist — and would make the majority
baseline appear to beat random over most of the range. The dashed prevalence
line already carries what they contribute. Their ROC curves *are* shown above,
because the diagonal is genuinely what a constant score produces there.

## 9. Dummy baseline interpretation

**Majority** (`{majority.name}`) — predicts "no churn" for everyone.

- Accuracy {majority.mean["accuracy"]:.4f}: it "gets {majority.mean["accuracy"]:.1%} right"
  while finding zero churners. This is the concrete reason accuracy is
  disqualified as a selection metric in this project.
- ROC-AUC {majority.mean["roc_auc"]:.4f}: a constant score cannot rank anything.
- Average Precision {majority.mean["average_precision"]:.4f}, i.e. the prevalence.
- Recall {majority.mean["recall"]:.4f} and F1 {majority.mean["f1"]:.4f}: it never fires.

**Stratified random** (`{random_model.name}`) — guesses at the prevalence
observed in each training fold.

- ROC-AUC {random_model.mean["roc_auc"]:.4f} ± {random_model.std["roc_auc"]:.4f} and Average
  Precision {random_model.mean["average_precision"]:.4f} ± \
{random_model.std["average_precision"]:.4f}
  confirm the two no-skill floors empirically rather than by assertion.
- Precision {random_model.mean["precision"]:.4f} ≈ the prevalence: of the customers it
  flags, roughly the base rate actually churn — which is what "no information"
  looks like.
- Recall {random_model.mean["recall"]:.4f} ≈ the prevalence as well, because it fires on
  a random ~{prevalence:.0%} of everyone.

Together they establish that any ROC-AUC near 0.5 or any AP near
{prevalence:.3f} is worthless regardless of how the accuracy reads.

## 10. Logistic Regression baseline

Untuned, on the {results.feature_set.n_features} original features:

| Metric | Cross-validated | Distance above the no-skill floor |
| --- | --- | --- |
| ROC-AUC | {logistic.mean["roc_auc"]:.4f} ± {logistic.std["roc_auc"]:.4f} \
| +{logistic.mean["roc_auc"] - 0.5:.4f} over 0.500 |
| Average Precision | {logistic.mean["average_precision"]:.4f} ± \
{logistic.std["average_precision"]:.4f} \
| +{logistic.mean["average_precision"] - prevalence:.4f} over {prevalence:.4f} |
| Precision @0.5 | {logistic.mean["precision"]:.4f} ± {logistic.std["precision"]:.4f} \
| +{logistic.mean["precision"] - prevalence:.4f} over {prevalence:.4f} |
| Recall @0.5 | {logistic.mean["recall"]:.4f} ± {logistic.std["recall"]:.4f} | — |
| F1 @0.5 | {logistic.mean["f1"]:.4f} ± {logistic.std["f1"]:.4f} | — |

**OBSERVATION.** A plain linear model on the raw features already ranks
substantially better than chance on both threshold-independent metrics. The
signal the EDA described — contract type, tenure, internet tier, payment method
— is largely linearly accessible; it does not require a complex model to be
picked up.

**OBSERVATION.** Ranking quality and decision quality diverge. ROC-AUC
{logistic.mean["roc_auc"]:.3f} sounds strong, yet at the 0.5 rule recall is only
{logistic.mean["recall"]:.3f}. Both describe the same model: it orders customers
well, but the default cut-off sits far out on that ordering, so most churners
fall below it. This gap is the entire reason Phase 9 exists.

**This is a floor, not a result.** Its value is as a reference point for the
engineered features of Phase 6 and the models of Phase 7. Those comparisons are
governed by the protocol in section 5 and the paired method in section 7: a
candidate is judged on the consistency and magnitude of its per-fold deltas
against these numbers, with **Average Precision as the primary development
metric**, ROC-AUC secondary, and complexity and interpretability weighed
alongside. It does not have to improve every metric simultaneously, and the
numbers above are not a bar defined by any standard deviation.

## 11. Error and trade-off interpretation at {results.decision_threshold}

Out-of-fold confusion matrix for the logistic regression:

{_figure(figures["confusion"], "Out-of-fold confusion matrix at the 0.5 reference rule")}

| | Predicted retain | Predicted churn |
| --- | --- | --- |
| **Actual retained** | {confusion["true_negative"]:,} | {confusion["false_positive"]:,} |
| **Actual churned** | {confusion["false_negative"]:,} | {confusion["true_positive"]:,} |

Read operationally, at this rule and on this data:

- of the {caught:,} customers who churned, the model flags
  {confusion["true_positive"]:,} and misses {confusion["false_negative"]:,};
- of the {flagged:,} customers it flags, {confusion["true_positive"]:,} genuinely
  churned and {confusion["false_positive"]:,} would have stayed.

The asymmetry is structural, not a defect of the model. A default 0.5 cut-off on
an imbalanced problem is conservative about firing: it trades recall for
precision. Whether that trade is the right one **cannot be decided here** — it
depends on the cost of a retention offer, the cost of losing a customer and the
capacity of whoever acts on the list. The dataset contains none of those
numbers, so no business claim is made, no cost scenario is invented and the
threshold is left exactly where it is.

## 12. Limitations

- **No holdout estimate exists.** Every number is training-pool cross-validation.
  Cross-validated performance is an optimistic-leaning estimate of generalisation,
  and the honest measurement waits for Phase 9.
- **The 0.5 rule is arbitrary** for this problem. It was not chosen by looking at
  metrics, and it is not a business decision.
- **Untuned logistic regression is not "the linear model's ceiling".** Regularisation
  strength, penalty and class weighting are untouched; a tuned version may differ.
- **No significance testing.** With five folds, the reported standard deviations
  describe variability but do not support a formal claim that one model beats
  another.
- **The analyst-exposure limitation from Phase 3 still applies.** The exploratory
  hypotheses that shaped this protocol were formed with the complete dataset in
  view, holdout rows included. The final estimate may carry optimistic bias that
  cannot be quantified with the data available.
- **Snapshot classification, not forecasting.** The dataset has no time dimension,
  so nothing here validates prospective churn prediction over a horizon.

## 13. Decisions explicitly deferred

| Decision | Phase |
| --- | --- |
| Engineered features from the EDA candidate list | 6 |
| Model comparison beyond a linear baseline | 7 |
| Class-imbalance handling (weights, resampling) | 7 |
| Hyperparameter tuning, including `C` and penalty | 8 |
| Decision threshold, calibration, cost framing, error analysis | 9 |
| Holdout evaluation | 9 |
| Global and local explainability | 10 |

Nothing in this phase constrains those decisions except by providing the number
they must beat.

## 14. Recommended next experiment

Phase 6 — feature engineering — with the EDA's hypothesis-driven candidates:
protective-service count, tenure × contract, charge intensity relative to the
internet tier, `TotalCharges / tenure`, and the automatic-payment flag.

Each must be evaluated **against the logistic-regression baseline recorded
here**, on the same folds and with the same pipeline, using **paired per-fold
differences**:

```text
delta_AP_fold = AP_candidate_fold - AP_baseline_fold
```

The decision rests on whether those deltas point the same way across folds and
how large they are, read under the metric hierarchy of section 5 —
`{protocol.primary_development_metric}` primary,
`{protocol.secondary_ranking_metric}` secondary — and weighed against the
complexity and loss of interpretability the feature introduces. A candidate does
not need to improve every metric at once.
"""


def main() -> int:
    """Run the baseline experiment. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()

    try:
        manifest = verify_split_manifest(load_split_manifest())
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
        "Training pool: %d rows, %d features, positive prevalence %.4f",
        len(features),
        len(FEATURE_COLUMNS),
        positive_prevalence(labels),
    )

    models = build_baselines()
    evaluations = evaluate_models(models, features, target, build_splitter())

    results = build_results(
        evaluations,
        models,
        labels,
        raw_sha256=manifest.raw_sha256,
        training_ids_sha256=manifest.training_ids_sha256,
    )
    write_results(results)

    probabilities = {name: evaluation.oof_probability for name, evaluation in evaluations.items()}
    figures = build_figures(results, labels, probabilities, PROJECT_ROOT)

    Path(REPORT_PATH).write_text(
        build_report(results, figures, manifest.n_rows_holdout, date.today().isoformat()),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
