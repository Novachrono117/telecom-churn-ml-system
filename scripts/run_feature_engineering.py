"""Run the Phase 6 feature-engineering ablation and write its report.

Usage (from the repository root):

    uv run python scripts/build_split.py --verify        # prove the split
    uv run python scripts/run_feature_engineering.py

Writes:
    reports/experiments/feature_engineering_results.json
    reports/feature_engineering_report.md
    reports/figures/feature_engineering/*.png

Only the frozen training pool is loaded. The holdout is never read. The model,
the folds, the threshold and every hyperparameter are frozen at their Phase 5
values: this script varies one thing, the feature set.

If E0 fails to reproduce the frozen baseline the run stops. A protocol that no
longer reproduces its own baseline cannot attribute a delta to a feature.
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
from churn.features.ablation import (
    PRIMARY_METRIC,
    PROMISING,
    PROMISING_MIN_POSITIVE_FOLDS,
    SECONDARY_METRIC,
    run_ablation,
)
from churn.features.groups import FEATURE_GROUPS
from churn.features.plots import (
    plot_absolute_metrics,
    plot_oof_precision_recall,
    plot_paired_deltas,
)
from churn.features.results import (
    RESULTS_PATH,
    AblationResults,
    BaselineReproductionError,
    build_results,
    verify_baseline_reproduction,
    write_results,
)
from churn.modeling.evaluation import build_splitter
from churn.modeling.metrics import positive_prevalence
from churn.modeling.results import load_results as load_baseline_results
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_training_pool
from churn.preprocessing.target import encode_target

logger = logging.getLogger("run_feature_engineering")

REPORT_PATH = PROJECT_ROOT / "reports" / "feature_engineering_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "feature_engineering"

METRIC_LABELS = {
    "average_precision": "Average Precision (primary)",
    "roc_auc": "ROC-AUC (secondary)",
    "precision": "Precision @0.5",
    "recall": "Recall @0.5",
    "f1": "F1 @0.5",
    "accuracy": "Accuracy @0.5",
}
DIAGNOSTIC_METRICS = ("precision", "recall", "f1")

#: Candidates that are exact functions of columns the baseline already receives
#: after one-hot encoding. Their experiments are valid — a redundant aggregate
#: still moves the solution of a penalised model — but their deltas cannot be
#: read as evidence about signal, gained or missing.
REPARAMETRISED_GROUPS = ("protective_services", "automatic_payment")

REPARAMETRISATION_NOTE = (
    "**How to read this delta.** `{columns}` is a deterministic function of columns the "
    "baseline already receives after one-hot encoding, so this experiment introduces no "
    "new information: it tests a redundant re-parametrisation of information the model "
    "already had. The delta is therefore not evidence that the underlying quantity does "
    "or does not carry signal, and not evidence that new signal was or was not found. It "
    "measures one thing only — what the coarser parametrisation did to a regularised "
    "logistic regression on these folds."
)


def _label(record) -> str:
    """Short chart label: experiment id plus the group it adds."""
    group = record.feature_groups[0] if len(record.feature_groups) == 1 else "combined"
    return f"{record.experiment} · {group}"


def build_figures(
    results: AblationResults,
    target: np.ndarray,
    oof: dict[str, np.ndarray],
    root: Path,
) -> dict[str, Path]:
    """Render the ablation figures and return a name -> path mapping."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    candidates = [record for record in results.experiments if record.paired_deltas]
    paths: dict[str, Path] = {}

    paths["deltas"] = save_figure(
        plot_paired_deltas(
            {
                _label(record): record.paired_deltas[PRIMARY_METRIC].per_fold
                for record in candidates
            },
            {_label(record): record.classification or "" for record in candidates},
            "Average Precision",
            "Paired per-fold Average Precision deltas against the E0 baseline",
        ),
        figures_dir / "01_paired_ap_deltas.png",
    )

    paths["absolute"] = save_figure(
        plot_absolute_metrics(
            {record.experiment: record.mean for record in results.experiments},
            {record.experiment: record.std for record in results.experiments},
            (PRIMARY_METRIC, SECONDARY_METRIC),
            METRIC_LABELS,
            "Absolute cross-validated performance per experiment (training pool)",
        ),
        figures_dir / "02_absolute_metrics.png",
    )

    if len(oof) > 1:
        by_name = {record.experiment: record for record in results.experiments}
        paths["pr"] = save_figure(
            plot_oof_precision_recall(
                target,
                oof,
                {
                    name: by_name[name].out_of_fold[PRIMARY_METRIC]
                    for name in oof
                    if by_name[name].out_of_fold
                },
                results.training_positive_prevalence,
                "Precision-Recall — out-of-fold on the training pool",
            ),
            figures_dir / "03_oof_precision_recall.png",
        )

    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _experiment_rows(results: AblationResults) -> str:
    lines = []
    for record in results.experiments:
        groups = ", ".join(f"`{group}`" for group in record.feature_groups) or "— (baseline)"
        lines.append(f"| {record.experiment} | {groups} |")
    return "\n".join(lines)


def _candidate_rows(results: AblationResults) -> str:
    lines = []
    for candidate in results.candidates:
        added = "`" + "`, `".join(candidate.added_numeric) + "`"
        lines.append(
            f"| `{candidate.group}` | {added} | {candidate.stateful} | {candidate.hypothesis} |"
        )
    return "\n".join(lines)


def _complexity_rows(results: AblationResults, baseline_width: int) -> str:
    lines = []
    for record in results.experiments:
        if len(record.feature_groups) != 1:
            continue
        group = FEATURE_GROUPS[record.feature_groups[0]]
        shape = "one monotone coefficient" if len(group.numeric) == 1 else "two slope terms"
        delta = record.paired_deltas[PRIMARY_METRIC]
        lines.append(
            f"| `{group.name}` | {record.n_transformed_features - baseline_width} | {shape} "
            f"| {group.stateful} | {delta.folds_improved}/5 | {delta.mean:+.4f} |"
        )
    return "\n".join(lines)


def _delta_table(results: AblationResults, metric: str) -> str:
    header = "| Experiment | Feature group | " + " | ".join(f"F{i}" for i in range(1, 6))
    lines = [
        header + " | Mean | SD | +/− folds | Class |",
        "| --- " * 11 + "|",
    ]
    for record in results.experiments:
        if metric not in record.paired_deltas:
            continue
        delta = record.paired_deltas[metric]
        cells = " | ".join(f"{value:+.4f}" for value in delta.per_fold)
        groups = ", ".join(record.feature_groups)
        lines.append(
            f"| {record.experiment} | `{groups}` | {cells} | **{delta.mean:+.4f}** | "
            f"{delta.std:.4f} | {delta.folds_improved}/{delta.folds_worsened} | "
            f"{record.classification} |"
        )
    return "\n".join(lines)


def _absolute_table(results: AblationResults) -> str:
    lines = [
        "| Experiment | Input features | Transformed | "
        + " | ".join(METRIC_LABELS[m] for m in METRIC_LABELS)
        + " |",
        "| --- " * (len(METRIC_LABELS) + 4) + "|",
    ]
    for record in results.experiments:
        cells = " | ".join(
            f"{record.mean[metric]:.4f} ± {record.std[metric]:.4f}" for metric in METRIC_LABELS
        )
        lines.append(
            f"| {record.experiment} | {record.n_input_features} | "
            f"{record.n_transformed_features} | {cells} |"
        )
    return "\n".join(lines)


def _candidate_section(results: AblationResults, number: int, group_name: str) -> str:
    group = FEATURE_GROUPS[group_name]
    record = next(r for r in results.experiments if r.feature_groups == [group_name])
    primary = record.paired_deltas[PRIMARY_METRIC]
    secondary = record.paired_deltas[SECONDARY_METRIC]
    definition = next(c for c in results.candidates if c.group == group_name)
    diagnostics = ", ".join(
        f"{METRIC_LABELS[metric]} {record.mean[metric]:.4f}" for metric in DIAGNOSTIC_METRICS
    )
    folds = ", ".join(f"{value:+.4f}" for value in primary.per_fold)
    caveat = (
        "\n\n" + REPARAMETRISATION_NOTE.format(columns="`, `".join(definition.added_numeric))
        if group_name in REPARAMETRISED_GROUPS
        else ""
    )

    return f"""## {number}. {group_name} — {record.experiment}

**Hypothesis.** {group.hypothesis}

**Definition.** Adds `{"`, `".join(definition.added_numeric)}` \
({"stateful" if definition.stateful else "deterministic, stateless"}). \
{definition.notes}

| | Value |
| --- | --- |
| Paired delta {PRIMARY_METRIC} per fold | {folds} |
| Mean / SD | **{primary.mean:+.4f}** / {primary.std:.4f} |
| Folds improved / worsened | {primary.folds_improved} / {primary.folds_worsened} |
| Paired delta {SECONDARY_METRIC} (mean) | {secondary.mean:+.4f} \
({secondary.folds_improved}/{secondary.folds_worsened} folds) |
| Absolute {PRIMARY_METRIC} | {record.mean[PRIMARY_METRIC]:.4f} ± {record.std[PRIMARY_METRIC]:.4f} |
| Absolute {SECONDARY_METRIC} | {record.mean[SECONDARY_METRIC]:.4f} ± \
{record.std[SECONDARY_METRIC]:.4f} |
| Transformed width | {record.n_transformed_features} columns |
| Diagnostic @0.5 (not used for selection) | {diagnostics} |

**Classification: {record.classification}.** {record.rationale}{caveat}
"""


def build_report(
    results: AblationResults,
    figures: dict[str, Path],
    generated_on: str,
) -> str:
    """Render the Phase 6 report. Every number comes from this run."""
    baseline = results.experiments[0]
    reproduction = results.baseline_reproduction
    candidates = [record for record in results.experiments if record.paired_deltas]
    promising = [r for r in candidates if r.classification == PROMISING]
    combined = next((r for r in results.experiments if r.experiment == "E6"), None)
    cv = results.cross_validation
    cv_signature = (
        f"{cv['strategy']}(n_splits={cv['n_splits']}, shuffle={cv['shuffle']}, "
        f"random_state={cv['random_state']})"
    )
    gains = [r.paired_deltas[PRIMARY_METRIC].mean for r in candidates]
    largest_gain = max(gains)
    largest_relative = largest_gain / baseline.mean[PRIMARY_METRIC]

    supported = (
        "\n".join(
            f"- **{r.feature_groups[0]}** ({r.experiment}) — "
            f"{r.paired_deltas[PRIMARY_METRIC].mean:+.4f} AP, "
            f"{r.paired_deltas[PRIMARY_METRIC].folds_improved}/5 folds."
            for r in promising
        )
        or "- None. No candidate improved Average Precision consistently across folds."
    )
    unsupported = "\n".join(
        f"- **{r.feature_groups[0]}** ({r.experiment}, {r.classification}) — {r.rationale}"
        for r in candidates
        if r.classification != PROMISING
    )

    if combined is not None:
        combined_delta = combined.paired_deltas[PRIMARY_METRIC]
        combined_block = f"""E6 combines the candidates classified PROMISING:
`{"`, `".join(combined.feature_groups)}`.

| | Value |
| --- | --- |
| Paired delta {PRIMARY_METRIC} per fold | \
{", ".join(f"{v:+.4f}" for v in combined_delta.per_fold)} |
| Mean / SD | **{combined_delta.mean:+.4f}** / {combined_delta.std:.4f} |
| Folds improved / worsened | {combined_delta.folds_improved} / {combined_delta.folds_worsened} |
| Absolute {PRIMARY_METRIC} | {combined.mean[PRIMARY_METRIC]:.4f} ± \
{combined.std[PRIMARY_METRIC]:.4f} |
| Transformed width | {combined.n_transformed_features} columns |

**Classification: {combined.classification}.** {combined.rationale}"""
    elif len(promising) == 1:
        combined_block = (
            f"**Not built.** Exactly one candidate was classified PROMISING "
            f"(`{promising[0].feature_groups[0]}`), so a combination would be identical to "
            f"{promising[0].experiment}. Building it anyway would produce a duplicate row and "
            "the appearance of an extra result."
        )
    else:
        combined_block = (
            "**Not built.** No candidate was classified PROMISING, and the protocol forbids "
            "assembling a combination out of INCONCLUSIVE or NOT_SUPPORTED features. "
            "Combining candidates that did not individually hold up would be a search for a "
            "lucky subset, which is exactly what an ablation study is meant to avoid."
        )

    retained = [(group, record) for record in promising for group in record.feature_groups]
    if retained:
        retained_lines = "\n".join(
            f"- **`{group}`** ({record.experiment}) — mean ΔAP "
            f"{record.paired_deltas[PRIMARY_METRIC].mean:+.5f}, "
            f"{record.paired_deltas[PRIMARY_METRIC].folds_improved}/"
            f"{len(record.paired_deltas[PRIMARY_METRIC].per_fold)} folds positive. Retained as a "
            "**candidate feature**, to be re-examined in the Phase 7 sensitivity analysis. Not "
            "adopted."
            for group, record in retained
        )
        retained_block = f"""**Reference feature set: the original \
{baseline.n_input_features} features.** They are unchanged, and they remain the
feature set the Phase 7 model comparison is run on. No candidate replaced them,
and no candidate was adopted as *the* Phase 7 feature set.

**Retained candidates**

{retained_lines}

**What `PROMISING` means, and what it does not.** The label was assigned by the
heuristic fixed in section 4 *before* any result was seen — mean ΔAP > 0 and at
least {PROMISING_MIN_POSITIVE_FOLDS} of 5 folds improving — and it was not
revised once the numbers were in. Revising a classification because the surviving
effect looks small would mean choosing the rule to fit the result, which is the
failure mode a pre-registered rule exists to prevent. But the label certifies
exactly what the rule tests, and no more: **direction and consistency**. It makes
no claim about magnitude, and the magnitudes here are small — the largest mean
gain is {largest_gain:+.5f} AP against a baseline of \
{baseline.mean[PRIMARY_METRIC]:.4f} ({largest_relative:.2%} relative), with five
folds and no significance test behind it. **Passing the pre-defined heuristic is
not the same as establishing a material benefit.** Nothing in this phase
establishes one."""
        sensitivity = "`" + "`, `".join(group for group, _ in retained) + "`"
    else:
        retained_block = (
            f"**Reference feature set: the original {baseline.n_input_features} features, "
            "unchanged.** No candidate satisfied the pre-defined heuristic, so none is retained "
            "as a candidate and none is carried into Phase 7."
        )
        sensitivity = ""

    rejected = [r.experiment for r in candidates if r.classification != PROMISING]
    rejected_ids = (
        " and ".join([", ".join(rejected[:-1]), rejected[-1]])
        if len(rejected) > 1
        else "".join(rejected)
    )
    protocol_items = [
        f"1. **Primary comparison — model families on the original "
        f"{baseline.n_input_features} features**, on the same folds and under the same "
        "metric protocol. This is the comparison that selects a model family.",
    ]
    if sensitivity:
        protocol_items.append(
            f"2. **Secondary analysis — sensitivity to {sensitivity}.** The retained "
            "candidate is re-run as a sensitivity check on top of that comparison, to ask "
            "whether a small, consistent gain measured on a linear model survives a family "
            "that captures interactions natively. A tree ensemble may make an explicit "
            "interaction column redundant, in which case the candidate is dropped."
        )
    if rejected_ids:
        protocol_items.append(
            f"{len(protocol_items) + 1}. **{rejected_ids} are not automatically reopened for "
            "every new model family.** Re-running rejected candidates against each family "
            "would turn a controlled comparison into a search over feature-set × "
            "model-family cells, which is how a leaderboard eventually finds a lucky "
            "combination. They may be revisited deliberately, with a stated reason — never "
            "by default, and never promoted on the strength of the numbers in this report."
        )
    protocol_block = "\n".join(protocol_items)

    return f"""# Feature Engineering Ablation — Telco Customer Churn (Phase 6)

- Generated on: {generated_on}
- Generated by: `scripts/run_feature_engineering.py`
- Machine-readable record: `reports/experiments/feature_engineering_results.json`
- Raw SHA-256: `{results.raw_sha256}`
- Training partition fingerprint: `{results.training_ids_sha256}`

> **Every number here is cross-validated on the training pool.** No holdout row
> was loaded, transformed, predicted or measured. There is no test result in
> this repository before Phase 9.

---

## 1. Objective

Answer one question, five times:

> Does a derived representation consistently improve the frozen logistic-regression
> baseline under *exactly* the same validation protocol?

This is an **ablation study**, not a search for features that happen to score
well. Each experiment differs from the baseline by one hypothesis. Nothing about
the model, the folds, the threshold or the hyperparameters was changed to
benefit a candidate — the only variable is what the model sees.

## 2. Frozen baseline

Inherited from Phase 5 and not modified:

| Element | Value |
| --- | --- |
| Model | `LogisticRegression` |
| `C` / `penalty` / `class_weight` | 1.0 / l2 / None |
| `solver` / `max_iter` | lbfgs / 100 |
| Cross-validation | `{cv_signature}` |
| Identical folds as Phase 5 | {results.cross_validation["identical_to_baseline"]} |
| Primary metric | Average Precision |
| Secondary metric | ROC-AUC |
| Threshold | {results.decision_threshold} (unchanged, not optimised) |
| Training rows | {results.n_training_rows:,} |
| Positive prevalence | {results.training_positive_prevalence:.4f} |

The classifier is built by the *same function* the Phase 5 baseline uses, so its
configuration cannot drift between baseline and candidate: there is one
definition, not two copies.

**Maintenance note — a deprecation warning, deliberately not silenced.**
scikit-learn {results.scikit_learn_version} emits a `FutureWarning` for the
explicit `penalty="l2"` argument: it was deprecated in 1.8 and is scheduled for
removal in 1.10, in favour of `l1_ratio`. The warning is **not** a correctness
problem here — `penalty="l2"` is still honoured, and section 6 shows E0
reproducing the Phase 5 per-fold metrics to the last recorded digit. It is not
fixed in this phase either, because the model configuration is frozen and
changing it to quiet a warning would silently redefine the baseline every later
number is measured against.

It is recorded as a **maintenance item**, to be executed as a controlled
migration in whichever phase next has a mandate to touch the estimator, and in
any case before scikit-learn 1.10. The migration is not a text substitution: it
must ship with a numerical-equivalence test proving that `l1_ratio=0` reproduces
the per-fold metrics of the frozen `penalty="l2"` baseline within the same
tolerance section 6 uses. Until that test exists and passes, the argument stays
as it is and the warning stays visible.

The warning surfaced for the first time in this phase. It had always been raised,
but the Phase 5 cross-validation loop wrapped each `fit` in
`warnings.catch_warnings(record=True)` to detect convergence failures, and that
context manager intercepts *every* warning, not only the one being looked for. It
now re-emits anything that is not a `ConvergenceWarning`. That change altered no
number — the artefacts of both phases were regenerated and are byte-identical —
but it removed a place where a real warning could have gone unnoticed.

## 3. Experimental protocol

Every experiment is a complete pipeline, cloned fresh for each fold:

```text
TotalChargesCleaner  →  derived feature columns  →  ColumnTransformer  →  LogisticRegression
```

The cleaner runs first because `historical_average_charge` divides by a
`TotalCharges` that must already be numeric. The encoder runs last so that every
derived column is standardised with the **fold's own** statistics, exactly like
the original ones. Nothing is fitted outside a training fold.

| Experiment | Feature groups added |
| --- | --- |
{_experiment_rows(results)}

## 4. Paired-comparison methodology

For each candidate and each fold:

```text
delta_AP_fold  = AP_candidate_fold  - AP_baseline_fold
delta_ROC_fold = ROC_candidate_fold - ROC_baseline_fold
```

**Why paired.** Most of the spread in a per-fold metric comes from the folds
themselves — some partitions are simply harder — and that component cancels when
both models are measured on the same partition. Consequently the baseline's
between-fold standard deviation is **not** used as a bar a gain must clear: it
describes dispersion of one model across partitions, not the uncertainty of a
difference between two models. A candidate can be consistently better by less
than that figure.

**Classification.** An engineering heuristic about direction and consistency,
not inference:

| Label | Rule |
| --- | --- |
| `NOT_SUPPORTED` | mean delta AP ≤ 0, or the candidate loses in a majority of folds |
| `INCONCLUSIVE` | fewer than 4 of 5 folds improve, or AP improves while ROC-AUC moves against it |
| `PROMISING` | mean delta AP > 0, at least 4 of 5 folds improve, and ROC-AUC \
does not move against it |

**No minimum-gain cutoff exists anywhere in the rule.** Magnitude,
interpretability and complexity are reported separately (section 14) and weighed
by a reader. Five folds cannot support a significance test and none was run.

## 5. Candidate definitions

| Group | Adds | Stateful | Hypothesis |
| --- | --- | --- | --- |
{_candidate_rows(results)}

Every derived column is routed to the **numeric** block. These are ordinal or
binary quantities whose hypotheses are monotone, so a single coefficient states
exactly the claim being tested; one-hot encoding them would spend several
coefficients on a shape nobody hypothesised.

No original feature was removed. This phase tests *added representation*.
Feature selection is a different question and is not mixed in here.

**Informational status of E1 and E2 — what these two experiments actually test.**
`protective_service_count` is a deterministic combination of four service
indicators that the baseline already receives as one-hot columns.
`automatic_payment` is a deterministic aggregation of two of the four
`PaymentMethod` categories the encoder already emits. Both are exact functions of
what the model already has, so **neither introduces new information in the
informational sense**. What they test is a *re-parametrisation*: a coarser,
redundant aggregation of information that is already available.

That is still a legitimate experiment. A regularised logistic regression does not
pick an arbitrary member of an equivalence class of equivalent fits — the L2
penalty selects one specific solution, and adding a collinear aggregate changes
how the penalty distributes weight across the correlated columns. The learned
coefficients, and therefore the fitted probabilities, can change. The experiment
asks whether that shift helps.

It follows that the E1 and E2 deltas **must not be read as evidence of a gain, of
new signal, or of the absence of signal** in protective services or payment
method. Those associations were characterised in the EDA and are unaffected by
this phase. The deltas measure only what a redundant parametrisation did to one
regularised linear model on these five folds.

## 6. Baseline reproduction

E0 rebuilds the Phase 5 baseline through the Phase 6 pipeline builder and
compares every per-fold metric against `{reproduction.reference_artefact}`
(schema v{reproduction.reference_schema_version}).

| | Value |
| --- | --- |
| Reproduced | **{reproduction.reproduced}** |
| Tolerance | {reproduction.tolerance:.0e} |
| Largest absolute difference | {reproduction.max_absolute_difference:.1e} |

The run aborts if this check fails. A protocol that no longer reproduces its own
baseline cannot attribute a delta to a feature — the delta could be an artefact
of the pipeline being assembled differently.

## 7. Individual ablation results

Paired **Average Precision** deltas (the primary metric):

{_delta_table(results, PRIMARY_METRIC)}

Paired **ROC-AUC** deltas (secondary):

{_delta_table(results, SECONDARY_METRIC)}

{_figure(figures["deltas"], "Paired per-fold Average Precision deltas against the E0 baseline")}

Absolute cross-validated performance:

{_absolute_table(results)}

{_figure(figures["absolute"], "Absolute cross-validated performance per experiment")}

{_candidate_section(results, 8, "protective_services")}
{_candidate_section(results, 9, "automatic_payment")}
{_candidate_section(results, 10, "contract_tenure")}
{_candidate_section(results, 11, "historical_average_charge")}
{_candidate_section(results, 12, "charge_intensity")}
## 13. Combined promising features

{combined_block}

## 14. Complexity versus gain

Magnitude, consistency, interpretability and complexity are separate axes and
are stated separately here on purpose — collapsing them into a single score is
how an ablation study turns into a leaderboard.

| Candidate | Columns added | Interpretability | Stateful | Folds improved | Mean delta AP |
| --- | --- | --- | --- | --- | --- |
{_complexity_rows(results, baseline.n_transformed_features)}

A stateful feature costs more than a column: it adds fitted state that must be
persisted, versioned and monitored, and it introduces a failure mode — an unseen
group at prediction time — that a deterministic feature does not have. That cost
is worth paying only for a gain that is both consistent and large enough to
matter operationally.

**Magnitude in context — the headline of this phase.** The largest mean gain
observed is {largest_gain:+.4f} Average Precision against a baseline of
{baseline.mean[PRIMARY_METRIC]:.4f} — a relative change of
{largest_relative:.2%}. Read plainly: **no candidate produced a material change in
the cross-validated performance of this linear model.** That is a legitimate
result, not a failed experiment.

Two of the five were never able to produce one for informational reasons.
`protective_services` and `automatic_payment` are deterministic aggregations of
columns the model already receives one-hot (section 5), so E1 and E2 only ever
tested whether a redundant re-parametrisation shifts a regularised fit. Their
near-zero deltas describe that shift and nothing else — in particular they are
not a verdict on whether protective services or payment method carry signal.
`contract_tenure` is the one candidate that expresses something the additive
baseline genuinely cannot state, which is consistent with it being the only one
that moved in a stable direction — but it moved very little.

## 15. Supported and unsupported hypotheses

**Held up under the protocol**

{supported}

"Held up" means the pre-defined heuristic was satisfied — a positive mean delta
and a consistent direction across folds. It is a statement about direction and
consistency only, not about effect size, and not a demonstration that the feature
is worth adopting. Section 17 states what that implies for Phase 7.

**Did not hold up**

{unsupported}

For E1 and E2 this sentence is about the re-parametrisation, not about the
underlying quantity: both are exact functions of columns the model already had
(section 5), so "did not hold up" means the coarser encoding did not help a
regularised linear fit — it does not mean protective services or payment method
lack association with churn.

**What this does and does not mean.** A feature that improves prediction has
demonstrated a better *representation* of information already in the dataset. It
has not demonstrated that the quantity it measures **causes** churn, and it has
not established a new association beyond what the EDA reported — the derived
columns are functions of the original ones, so no new information entered the
model. The dataset remains observational: every relationship here is association
within this sample.

Equally, a candidate classified `NOT_SUPPORTED` has not been shown to be
irrelevant. It has been shown that *this* encoding of *that* hypothesis does not
help *this* linear model on *these* folds. A non-linear model in Phase 7 may
extract the same interaction without an explicit column.

## 16. Limitations

- **No holdout estimate.** Everything is training-pool cross-validation.
- **Five folds, no significance test.** The classification labels describe
  direction and consistency; they are not statistical claims, and a
  `PROMISING`/`INCONCLUSIVE` boundary case should be read as such.
- **One model.** Every conclusion is conditional on an untuned logistic
  regression. A feature that helps a linear model may be redundant for a tree
  ensemble, and vice versa.
- **The threshold is untouched**, so the diagnostic trio at
  {results.decision_threshold} describes one operating point and was not used to
  select anything.
- **Candidates were chosen from the EDA**, which ran on the complete dataset. The
  analyst-exposure limitation recorded in Phase 3 therefore still applies to the
  hypotheses tested here.
- **Multiplicity.** Five candidates were evaluated against one baseline. No
  correction was applied because no inferential claim is made, but the more
  candidates are tried, the more likely one looks good by chance.

## 17. Feature set status going into Phase 7

{retained_block}

**Phase 7 protocol, fixed here — before any Phase 7 result exists.**

{protocol_block}

Fixing this now is the point: deciding after seeing Phase 7 numbers which feature
set to compare model families on would make the feature set another tuned
parameter. No candidate is carried into Phase 7 on the basis of its EDA
hypothesis alone, and none is carried as an established improvement.
"""


def main() -> int:
    """Run the ablation. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = get_config()

    try:
        verify_split_manifest(load_split_manifest())
        reference = load_baseline_results()
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

    experiments = run_ablation(features, target, build_splitter())

    try:
        reproduction = verify_baseline_reproduction(experiments["E0"], reference)
    except BaselineReproductionError as error:
        logger.error("%s", error)
        return 4

    best = max(
        (r for r in experiments.values() if r.comparisons),
        key=lambda r: r.comparisons[PRIMARY_METRIC].mean,
        default=None,
    )
    oof_names: tuple[str, ...] = ("E0",)
    oof = {"E0": experiments["E0"].evaluation.oof_probability}
    if best is not None and best.classification == PROMISING:
        oof_names = ("E0", best.experiment)
        oof[best.experiment] = best.evaluation.oof_probability

    results = build_results(experiments, reference, reproduction, labels, oof_names)
    write_results(results)

    figures = build_figures(results, labels, oof, PROJECT_ROOT)
    Path(REPORT_PATH).write_text(
        build_report(results, figures, date.today().isoformat()),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
