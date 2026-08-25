"""Interpret the frozen churn model exactly (Phase 10).

Usage (from the repository root):

    uv run python scripts/build_split.py --verify
    uv run python scripts/freeze_model.py --verify
    uv run python scripts/evaluate_holdout.py --verify
    uv run python scripts/run_error_analysis.py --verify
    uv run python scripts/run_model_interpretation.py            # the interpretation
    uv run python scripts/run_model_interpretation.py --verify   # check the record only

Writes:
    reports/experiments/model_interpretation_results.json
    reports/model_interpretation_report.md
    reports/figures/interpretability/*.png

This phase explains a model that already exists. It is **not** model selection:
nothing is fitted, compared, tuned, calibrated, added, removed or re-thresholded,
and nothing discovered here may change any of those. The frozen model is
`9d1db49`, its final estimate is `8c8e266` and its error analysis is `5e9c4e9`.

The global interpretation is computed on the **training pool**. The holdout was
consumed by Phase 9D and is never loaded here: this module does not import
``load_holdout``, and ranking features on a spent sample would be post-hoc mining.

No SHAP, no LIME, no permutation importance and no partial dependence. The model
is additive and linear after a fixed transformation, so its decomposition is
exact, and the run aborts unless that decomposition reproduces the pipeline's own
``decision_function`` and ``predict_proba``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from churn.analysis.plots import save_figure, use_project_style
from churn.config import PROJECT_ROOT
from churn.data.loader import RawDatasetMismatchError
from churn.modeling.calibration_results import load_results as load_calibration_results
from churn.modeling.freeze import (
    CONFIGURATION_FILES,
    INFERENCE_SOURCE_FILES,
    IntegrityError,
    model_fingerprint,
    source_digests,
)
from churn.modeling.freeze_results import load_decision_policy, verify_or_raise
from churn.modeling.interpretation import (
    RANKING_METRIC,
    SUPPORTING_METRICS,
    CategoricalTerm,
    InterpretationError,
    NumericTerm,
    ReconstructionError,
    categorical_terms,
    check_structural_dependencies,
    contribution_dispersion,
    coupled_column_blocks,
    coupled_levels_from,
    extract_terms,
    feature_groups,
    group_contributions,
    logit_of,
    numeric_terms,
    rank_by_dispersion,
    transform_features,
    verify_reconstruction,
)
from churn.modeling.interpretation_plots import (
    plot_categorical_contrasts,
    plot_contribution_direction,
    plot_contribution_dispersion,
    plot_numeric_coefficients,
)
from churn.modeling.interpretation_results import (
    RESULTS_PATH,
    UPSTREAM_ARTEFACTS,
    InterpretationResults,
    build_results,
    file_digest,
    invariant_failures,
    load_results,
    write_results,
)
from churn.modeling.threshold_results import load_results as load_threshold_results
from churn.preprocessing.contracts import build_feature_matrix
from churn.preprocessing.manifest import (
    SplitManifestMismatchError,
    load_split_manifest,
    verify_split_manifest,
)
from churn.preprocessing.splitting import load_training_pool

logger = logging.getLogger("run_model_interpretation")

REPORT_PATH = PROJECT_ROOT / "reports" / "model_interpretation_report.md"
FIGURES_SUBDIR = Path("reports") / "figures" / "interpretability"

#: Human label of the ranking metric, used in figure axes and report prose.
METRIC_LABEL = "Standard deviation"

#: How many raw features the "strongest modelled contributions" section names on
#: each side. Fixed here rather than chosen after reading the table.
N_HIGHLIGHTED = 5


def build_figures(
    results: InterpretationResults,
    numeric: list[NumericTerm],
    categorical: list[CategoricalTerm],
    dispersions: list,
    root: Path,
) -> dict[str, Path]:
    """Render the Phase 10 figures."""
    use_project_style()
    figures_dir = root / FIGURES_SUBDIR
    ranking = results.contribution_dispersion["ranking"]

    paths = {
        "numeric": save_figure(
            plot_numeric_coefficients(numeric),
            figures_dir / "01_numeric_standardized_coefficients.png",
        ),
        "categorical": save_figure(
            plot_categorical_contrasts(categorical),
            figures_dir / "02_categorical_coefficient_contrasts.png",
        ),
        "dispersion": save_figure(
            plot_contribution_dispersion(ranking, RANKING_METRIC, METRIC_LABEL),
            figures_dir / "03_raw_feature_contribution_dispersion.png",
        ),
        "direction": save_figure(
            plot_contribution_direction(dispersions, [row["feature"] for row in ranking]),
            figures_dir / "04_modeled_contribution_direction_summary.png",
        ),
    }
    logger.info("Rendered %d figures into %s", len(paths), figures_dir)
    return paths


def _figure(path: Path, caption: str) -> str:
    relative = path.as_posix().split("reports/")[-1]
    return f"![{caption}]({relative})\n\n*{caption}*"


def _direction(value: float) -> str:
    return "higher" if value >= 0 else "lower"


def _numeric_table(results: InterpretationResults) -> str:
    lines = [
        "| Feature | `standardized_coefficient` | `scaler_mean` | `scaler_scale` "
        "| `raw_logit_coefficient` | `odds_ratio_per_1_sd` | Raw unit |",
        "| --- " * 7 + "|",
    ]
    for term in results.numeric_terms:
        lines.append(
            f"| `{term['feature']}` | {term['standardized_coefficient']:+.6f} | "
            f"{term['scaler_mean']:.6f} | {term['scaler_scale']:.6f} | "
            f"{term['raw_logit_coefficient']:+.8f} | {term['odds_ratio_per_1_sd']:.6f} | "
            f"{term['raw_unit']} |"
        )
    return "\n".join(lines)


def _categorical_summary_table(results: InterpretationResults) -> str:
    lines = [
        "| Feature | Levels | Lowest β level | Highest β level | `spread` "
        "| `max_pairwise_modeled_odds_ratio` | Structurally coupled level |",
        "| --- " * 7 + "|",
    ]
    for term in results.categorical_terms:
        coupled = ", ".join(f"`{level}`" for level in term["structurally_coupled_levels"]) or "—"
        lines.append(
            f"| `{term['feature']}` | {term['n_levels']} | "
            f"`{term['lowest_coefficient_level']}` ({term['minimum']:+.4f}) | "
            f"`{term['highest_coefficient_level']}` ({term['maximum']:+.4f}) | "
            f"{term['spread']:.4f} | {term['max_pairwise_modeled_odds_ratio']:.4f} | {coupled} |"
        )
    return "\n".join(lines)


def _levels_table(term: dict) -> str:
    coupled = set(term["structurally_coupled_levels"])
    lines = ["| Level | Coefficient (log-odds) | Structurally coupled |", "| --- " * 3 + "|"]
    for level, value in term["coefficient_per_level"].items():
        mark = "**yes**" if level in coupled else "no"
        lines.append(f"| `{level}` | {value:+.6f} | {mark} |")

    contrast = term["strongest_pairwise_contrast"]
    heading = (
        "Widest within-feature **algebraic** contrast"
        if contrast["structurally_coupled"]
        else "Widest within-feature contrast"
    )
    caveat = (
        (
            "\n\n> One of these levels is **structurally coupled** to another raw feature "
            "(section 6.1). The difference above is exact, but it is an *algebraic* contrast: "
            "changing only this feature between those two levels would describe a row that does "
            "not occur in the training pool. "
            "`raw_single_feature_counterfactual_supported: false`."
        )
        if contrast["structurally_coupled"]
        else ""
    )
    return (
        "\n".join(lines) + f"\n\n{heading}: `{contrast['level_a']}` against "
        f"`{contrast['level_b']}`, Δ log-odds **{contrast['delta_log_odds']:+.4f}**, "
        f"modelled odds ratio **{term['max_pairwise_modeled_odds_ratio']:.4f}**." + caveat
    )


def _structural_rules_table(results: InterpretationResults) -> str:
    lines = [
        "| Verified equivalence | Rows checked | Rows on both sides | Left only | Right only "
        "| Violations | Deterministic |",
        "| --- " * 7 + "|",
    ]
    for rule in results.structural_dependencies["rules"]:
        lines.append(
            f"| `{rule['relationship']}` | {rule['n_rows_checked']:,} | {rule['n_both']:,} | "
            f"{rule['left_only']:,} | {rule['right_only']:,} | **{rule['violations']}** | "
            f"**{rule['deterministic_in_training_pool']}** |"
        )
    return "\n".join(lines)


def _coupled_blocks_section(results: InterpretationResults) -> str:
    blocks = results.structural_dependencies["coupled_column_blocks"]
    if not blocks:
        return "No block of transformed columns was found to be identical across raw features."

    parts: list[str] = []
    for block in blocks:
        members = "\n".join(
            f"| `{member['feature']}` | `{member['level']}` | {column} | {value:+.10f} |"
            for member, column, value in zip(
                block["members"],
                block["transformed_columns"],
                block["coefficient_per_column"],
                strict=True,
            )
        )
        parts.append(
            f"**Block of {block['n_columns']} identical columns** — active on "
            f"{block['n_active_rows']:,} rows.\n\n"
            "| Raw feature | Level | Column | Coefficient |\n"
            "| --- | --- | --- | --- |\n"
            f"{members}\n\n"
            f"- Largest difference between any two of these coefficients: "
            f"**{block['max_pairwise_coefficient_difference']:.1e}**\n"
            f"- Aggregate the model actually applies when the state is active: "
            f"**{block['aggregate_coefficient']:+.10f}**\n"
            f"- Even split across {block['n_columns']} columns: "
            f"**{block['equal_split_share']:+.10f}** each"
        )
    return "\n\n".join(parts)


def _dispersion_table(results: InterpretationResults) -> str:
    by_feature = {row["feature"]: row for row in results.contribution_dispersion["by_feature"]}
    lines = [
        "| Rank | Feature | Kind | `std` | `mean_absolute_centered_contribution` | `mean` "
        "| `median` | `q1` | `q3` | `iqr` |",
        "| --- " * 10 + "|",
    ]
    for row in results.contribution_dispersion["ranking"]:
        record = by_feature[row["feature"]]
        lines.append(
            f"| {row['rank']} | `{row['feature']}` | {record['kind']} | **{record['std']:.4f}** | "
            f"{record['mean_absolute_centered_contribution']:.4f} | {record['mean']:+.4f} | "
            f"{record['median']:+.4f} | {record['q1']:+.4f} | {record['q3']:+.4f} | "
            f"{record['iqr']:.4f} |"
        )
    return "\n".join(lines)


def _mapping_table(results: InterpretationResults) -> str:
    lines = ["| Raw feature | Kind | Transformed columns | Levels |", "| --- " * 4 + "|"]
    for group in results.feature_mapping["groups"]:
        columns = group["transformed_columns"]
        span = f"{columns[0]}" if len(columns) == 1 else f"{columns[0]}–{columns[-1]}"
        levels = ", ".join(f"`{level}`" for level in group["levels"]) or "—"
        lines.append(
            f"| `{group['feature']}` | {group['kind']} | {span} "
            f"({group['n_transformed_columns']}) | {levels} |"
        )
    return "\n".join(lines)


def _digest_table(results: InterpretationResults) -> str:
    lines = ["| Artefact | SHA-256 |", "| --- | --- |"]
    for artefact, digest in results.upstream_artefact_digests.items():
        lines.append(f"| `{artefact}` | `{digest[:16]}…` |")
    return "\n".join(lines)


def _eda_table(results: InterpretationResults) -> str:
    lines = [
        "| Feature | Marginal association recorded in Phase 3 | Modelled conditional behaviour "
        "| Source |",
        "| --- " * 4 + "|",
    ]
    numeric = {term["feature"]: term for term in results.numeric_terms}
    categorical = {term["feature"]: term for term in results.categorical_terms}

    for entry in results.eda_comparison:
        feature = entry["feature"]
        if feature in numeric:
            value = numeric[feature]["standardized_coefficient"]
            modelled = (
                f"β = {value:+.4f} per 1 SD — pushes the modelled log-odds toward a "
                f"**{_direction(value)}** churn score"
            )
        else:
            term = categorical[feature]
            modelled = (
                f"highest β at `{term['highest_coefficient_level']}` "
                f"({term['maximum']:+.4f}), lowest at `{term['lowest_coefficient_level']}` "
                f"({term['minimum']:+.4f}); spread {term['spread']:.4f}"
            )
        lines.append(
            f"| `{feature}` | {entry['marginal_observation']} | {modelled} | `{entry['source']}` |"
        )
    return "\n".join(lines)


def _highlight_table(results: InterpretationResults) -> str:
    """Name the widest within-feature contrasts, as contrasts and never as effects."""
    rows = sorted(
        results.categorical_terms,
        key=lambda term: (-float(term["spread"]), term["feature"]),
    )[:N_HIGHLIGHTED]
    lines = [
        "| Feature | Encoded level | against | Δ model log-odds | Modelled odds ratio "
        "| Single-feature counterfactual supported |",
        "| --- " * 6 + "|",
    ]
    for term in rows:
        contrast = term["strongest_pairwise_contrast"]
        supported = "**no** — structurally coupled" if contrast["structurally_coupled"] else "yes"
        lines.append(
            f"| `{term['feature']}` | `{term['highest_coefficient_level']}` | "
            f"`{term['lowest_coefficient_level']}` | {term['spread']:+.4f} | "
            f"{term['max_pairwise_modeled_odds_ratio']:.4f} | {supported} |"
        )
    return "\n".join(lines)


def _limitations(results: InterpretationResults) -> str:
    return "\n".join(f"- {item}" for item in results.limitations)


def build_report(results: InterpretationResults, figures: dict[str, Path]) -> str:
    """Render the Phase 10 report. Every number comes from this analysis.

    The report carries **no wall-clock metadata**, per the convention recorded in
    `CLAUDE.md`: it is a pure function of its inputs, so a rerun on an unchanged
    repository reproduces it byte for byte and any diff is a real change.
    """
    provenance = results.provenance
    model = results.model
    reconstruction = results.reconstruction
    decision = results.frozen_decision
    dispersion = results.contribution_dispersion
    population = results.interpretation_population
    ranking = dispersion["ranking"]
    top = ranking[0]

    categorical_sections = "\n\n".join(
        f"#### {term['feature']}\n\n{_levels_table(term)}" for term in results.categorical_terms
    )

    return f"""# Model Interpretation — Telco Customer Churn (Phase 10)

- Generated by: `scripts/run_model_interpretation.py`
- Machine-readable record: `reports/experiments/model_interpretation_results.json`
- Model frozen in: `{provenance["freeze_commit_short"]}` — {provenance["freeze_commit_subject"]}
- Final estimate in: `{provenance["evaluation_commit_short"]}` — \
{provenance["evaluation_commit_subject"]}
- Error analysis in: `{provenance["error_analysis_commit_short"]}` — \
{provenance["error_analysis_commit_subject"]}

> **This report carries no generation date.** It is a pure function of its
> inputs, so re-running the generator on an unchanged repository reproduces it
> byte for byte and any diff is a real change. Git records when it was produced.

---

## 1. Scope

This phase answers one question:

> **How does the frozen model turn the {model["n_raw_features"]} contracted features into an
> estimated churn score?**

It does **not** answer "how could the model be better". That question is closed:
the estimator, the preprocessing, the feature set, the calibration policy and the
threshold were frozen in `{provenance["freeze_commit_short"]}`, measured once in
`{provenance["evaluation_commit_short"]}` and diagnosed in
`{provenance["error_analysis_commit_short"]}`. Nothing in this report may change
any of them, and nothing in it was chosen because of what the earlier phases
found.

| Property | Value |
| --- | --- |
| `analysis_type` | **{results.analysis_type}** |
| `selection_allowed` | **{results.selection_allowed}** |
| `model_change_allowed` | **{results.model_change_allowed}** |
| `fit_calls` | **{results.fit_calls}** |
| `holdout_used_for_global_interpretation` | \
**{results.holdout_used_for_global_interpretation}** |
| Interpretation population | **{population["partition"]}**, {population["n_rows"]:,} rows |

### What this phase deliberately did not do

| Not done | Count / flag |
| --- | --- |
| Fits of any kind | {results.methodology["fit_calls"]} |
| Alternative models evaluated | {results.methodology["alternative_models_evaluated"]} |
| Thresholds scored | {results.methodology["thresholds_scored"]} |
| Calibrations | {results.methodology["calibrations_performed"]} |
| Hyperparameter searches | {results.methodology["hyperparameter_searches"]} |
| Feature selections | {results.methodology["feature_selections"]} |
| Features added / removed | {results.methodology["features_added"]} / \
{results.methodology["features_removed"]} |
| New performance estimates | {results.methodology["new_performance_estimates_produced"]} |
| New dependencies added | {results.methodology["new_dependencies_added"]} |
| SHAP used | {results.methodology["shap_used"]} |
| LIME / ELI5 used | {results.methodology["lime_used"]} / {results.methodology["eli5_used"]} |
| Permutation importance used | {results.methodology["permutation_importance_used"]} |
| Partial dependence / ICE used | {results.methodology["pdp_used"]} / \
{results.methodology["ice_used"]} |
| Holdout loaded | {results.methodology["holdout_loaded"]} |
| Identifiers persisted | {results.methodology["identifiers_persisted"]} |
| Individual explanations persisted | \
{results.methodology["individual_explanations_persisted"]} |
| Causal claims | {results.methodology["causal_claims"]} |

**Why the training pool and not the holdout.** The holdout was consumed by the
Phase 9D evaluation. Ranking features on it — deciding, after the fact, which
inputs "matter" using the sample that produced the final estimate — is post-hoc
mining of a spent partition, and it would contaminate the one number this project
still relies on. The global interpretation therefore runs on the
{population["n_rows"]:,} rows of the training pool, which is also the population
the coefficients and the scaler statistics were learned from. The Phase 10
modules do not import `load_holdout` at all.

**Why the Phase 9E findings did not steer this report.** All
{model["n_raw_features"]} raw features and all {model["n_transformed_features"]}
transformed columns are reported. None was included, excluded, highlighted or
suppressed because of an error pattern observed in the holdout.

## 2. Frozen model being interpreted

| Property | Value |
| --- | --- |
| Estimator | `{model["estimator"]}` |
| Hyperparameters | `C = {model["hyperparameters"]["C"]}`, \
`l1_ratio = {model["hyperparameters"]["l1_ratio"]}`, \
`solver = {model["hyperparameters"]["solver"]}`, \
`class_weight = {model["hyperparameters"]["class_weight"]}`, \
`max_iter = {model["hyperparameters"]["max_iter"]}` |
| `classes_` | `{model["classes"]}` |
| Positive class | label `{model["positive_class_label"]}`, column \
`{model["positive_class_column"]}` |
| Positive class resolution | {model["positive_class_resolution"]} |
| `intercept_` | **{model["intercept"]}** |
| `n_iter_` | `{model["n_iter"]}` (converged: {model["converged"]}, \
max {model["max_iter"]}) |
| Original features | **{model["n_raw_features"]}** \
({model["n_numeric_features"]} numeric, {model["n_categorical_features"]} categorical) |
| Transformed features | **{model["n_transformed_features"]}** |
| Coefficients | **{model["n_coefficients"]}** |
| Engineered features | `{model["engineered_features"]}` |
| `calibration_policy` | **{decision["calibration_policy"]}** |
| `threshold_policy` | **{decision["threshold_policy"]}** |
| `final_threshold` | **{decision["threshold_probability"]}** |
| Model fingerprint verified after load | **{provenance["model_fingerprint_verified"]}** |
| Pipeline unchanged by this phase | **{provenance["pipeline_unchanged_by_analysis"]}** |
| Refitted here | {provenance["refitted_here"]} |

{_digest_table(results)}

## 3. Exact mathematical model

The frozen estimator is a logistic regression, so its response surface is known
in closed form rather than approximated:

```text
logit(P(Churn = 1 | x)) = intercept + Σ_j  coefficient_j × transformed_feature_j

P(Churn = 1 | x)        = sigmoid(logit) = 1 / (1 + exp(−logit))
```

with `intercept = {model["intercept"]}` and {model["n_coefficients"]} coefficients,
one per transformed column.

Regrouping the {model["n_transformed_features"]} transformed columns into the
{model["n_raw_features"]} raw features they came from is exact, because the columns
partition:

```text
logit = intercept + Σ_g  contribution_g(x)

contribution_g(x) = Σ_{{j ∈ g}}  coefficient_j × z_j
```

- **numeric:** `contribution = coefficient × (raw_value − scaler_mean) / scaler_scale`
- **categorical:** `contribution =` the coefficient of the **active level**; the
  feature's other columns are zero, and an unseen category is all-zeros under
  `handle_unknown="ignore"`.

### The frozen decision boundary, in log-odds

The threshold is the frozen one. Expressing it in log-odds is a change of
coordinates on the **same** rule, not a new threshold:

```text
threshold_logit = log(threshold / (1 − threshold))
                = log({decision["threshold_probability"]} / \
(1 − {decision["threshold_probability"]}))
                = {decision["threshold_logit"]}
```

A customer is classified positive when

```text
model_logit >= {decision["threshold_logit"]}      (equivalently  probability \
{decision["comparison"]} {decision["threshold_probability"]})
```

which closes the chain this phase exists to make visible:

```text
feature contributions → logit → probability → frozen threshold → decision
```

`threshold_changed_here: {decision["threshold_changed_here"]}` ·
`calibration_changed_here: {decision["calibration_changed_here"]}` ·
`alternative_thresholds_scored: {decision["alternative_thresholds_scored"]}`

## 4. Reconstruction proof

An interpretation of a model that does not reproduce that model's own decision
function is not an interpretation of it. Three comparisons run over all
{reconstruction["n_rows"]:,} rows of the {reconstruction["population"]}, and the
phase **aborts** on any failure:

| Check | Reconstructed as | Compared against | Max absolute error |
| --- | --- | --- | --- |
| Logit | `intercept + Z @ coefficients` | `{reconstruction["logit_reference"]}` | \
**{reconstruction["max_abs_logit_error"]:.3e}** |
| Probability | `sigmoid(manual_logit)` | `{reconstruction["probability_reference"]}` | \
**{reconstruction["max_abs_probability_error"]:.3e}** |
| Grouped logit | `intercept + Σ` of the {model["n_raw_features"]} contributions | \
`{reconstruction["grouped_reference"]}` | \
**{reconstruction["max_grouped_reconstruction_error"]:.3e}** |

Tolerance: **{reconstruction["tolerance"]:.0e}** · within tolerance:
**{reconstruction["within_tolerance"]}**.

{reconstruction["tolerance_justification"]}.

The third row is the one that matters for everything below section 6: it proves
that regrouping the {model["n_transformed_features"] - model["n_numeric_features"]}
one-hot columns into their {model["n_categorical_features"]} parent features loses
nothing and counts nothing twice.

### How the {model["n_transformed_features"]} columns map onto the \
{model["n_raw_features"]} features

The mapping is derived from the **fitted encoder's own structure** —
`textual_parsing_used: {results.feature_mapping["textual_parsing_used"]}`. The
`ColumnTransformer` emits its numeric block first, one column per scaled feature,
then `len(categories_[i])` consecutive columns for the i-th categorical feature.
Splitting a name such as `categorical__PaymentMethod_Mailed check` on `_` would be
the fragile alternative, since both the separator and the level values contain the
delimiter. The generated names are used only to cross-check the result
(`cross_checked_against_generated_names: \
{results.feature_mapping["cross_checked_against_generated_names"]}`), and the
partition is verified explicitly
(`partition_verified: {results.feature_mapping["partition_verified"]}`).

{_mapping_table(results)}

## 5. Numeric features

The three numeric features pass through `StandardScaler`, so the fitted
coefficient belongs to the **standardised** column: it is the change in the
model's log-odds per one standard deviation of that feature *as measured on the
training pool*.

{_numeric_table(results)}

{_figure(figures["numeric"], "Standardised numeric coefficients of the frozen model")}

**How to read `odds_ratio_per_1_sd`.** It is `exp(β)`, and it means:

> the multiplicative change in the **odds predicted by the model** for an increase
> of one standard deviation in that standardised feature, holding every other
> transformed feature constant.

It is a property of the frozen predictive function. It is **not** a statement that
changing the underlying attribute of a real customer would change their behaviour.

**How to read `raw_logit_coefficient`.** It is `β / scaler_scale`, an exact
algebraic re-expression of the same linear term — the transformed value is
`(x − mean) / scale`, so the derivative of the term with respect to the raw `x` is
`β / scale`. It is the same model in different units, not a second estimate, and no
increment was chosen after looking at the results to make an effect look larger:
the units are one month of tenure and one monetary unit of the dataset's own
charge scales.

## 6. Categorical features

**The one-hot encoding has no dropped baseline.** Every level of every categorical
feature has its own column, and the model also has an intercept. That
parameterisation is redundant: a constant can be added to every level of one
feature and subtracted from the intercept without changing a single prediction.
L2 regularisation picks one point in that family.

The consequence is concrete and governs this whole section:

> **`exp(β)` of a single level is NOT an odds ratio against a reference category**,
> because there is no reference category. `reference_level_declared: \
{results.methodology["reference_category_declared"]}`.

What **is** identified — invariant to that shift — is the **within-feature
contrast**:

```text
Δ log-odds(A vs B) = β_A − β_B
modelled_odds_ratio(A vs B) = exp(β_A − β_B)
```

> This is a contrast **inside the frozen predictive function**, holding every other
> transformed feature constant. It is not a causal estimate.

{_categorical_summary_table(results)}

`spread` is **not** called importance: it is the width of a feature's coefficient
range, which bounds how much that feature could move the logit *if* a customer's
level changed. It says nothing about how often levels differ in any population —
that is what section 7 adds.

{_figure(figures["categorical"], "Categorical coefficients, faceted by raw feature")}

### 6.1 Structural coupling between raw features

A contrast `β_A − β_B` is always an exact statement about the linear function. It
is **not** automatically a statement about a change a customer could undergo, and
in this dataset the two come apart for a specific, checkable reason: some encoded
states are tied to each other.

**The relationships were declared first, then verified — not assumed.** The
candidates come from the product semantics recorded in the Phase 2 data
dictionary (a sentinel such as `No internet service` exists *because* the customer
has no internet), never from the model's coefficients and never from the holdout
(`holdout_used: {results.structural_dependencies["holdout_used"]}`). Each one was
then counted row by row on the {results.structural_dependencies["n_rows_checked"]:,}
rows of the {results.structural_dependencies["population"]}:

{_structural_rules_table(results)}

{results.structural_dependencies["n_deterministic"]} of
{results.structural_dependencies["n_rules_checked"]} rules hold on **every** row;
{results.structural_dependencies["n_with_violations"]} have violations. A rule's
`deterministic_in_training_pool` is *derived* from its violation count, so a rule
with even one violation cannot report itself as structural, and only deterministic
rules are allowed to qualify a contrast.

### What that does to the transformed matrix

If two raw states are the same state, their one-hot columns are the **same
vector**. That was checked directly, column against column, on the same
population:

{_coupled_blocks_section(results)}

**How to read the repeated coefficients.** The equal values are not six or seven
independent findings. Those columns are perfectly collinear, so the model's
parameters are not identified along that direction: any redistribution of the
block's total among its columns produces *identical predictions*. The L2 penalty
resolves the tie by splitting the shared weight evenly, which is exactly what the
pairwise differences above show — they are zero to the last bit. Stated plainly:

> **The repeated `No internet service` coefficients must not be read as
> independent evidence from several services.** Those encoded states are
> structurally coupled in this dataset, so the regularised model distributes one
> underlying no-internet state across redundant indicators. The quantity the model
> actually applies when that state is active is the block **aggregate**; the
> per-column values are one point in a family of equivalent parameterisations.

**What this does not change.** The decomposition remains exactly correct for *this*
frozen pipeline — section 4's three gates are unaffected, and every coefficient and
every contrast is still reported in full
(`contrasts_removed_or_altered: \
{results.structural_dependencies["contrasts_removed_or_altered"]}`,
`coefficients_removed_or_altered: \
{results.structural_dependencies["coefficients_removed_or_altered"]}`). What
changes is the *attribution*: how the total is split among raw features depends on
the parameterisation the model happened to use, not on the world.

### Levels and coefficients, feature by feature

{categorical_sections}

## 7. Contribution dispersion by original feature

Section 5 compares three standardised coefficients with one another and section 6
compares levels within a feature. Neither gives a view across all
{model["n_raw_features"]} raw features, and the obvious shortcut is wrong:

> **No `importance = abs(coefficient)` ranking is produced here**
> (`naive_abs_coefficient_ranking_produced: \
{results.methodology["naive_abs_coefficient_ranking_produced"]}`). Pooling
> standardised numeric coefficients with 0/1 dummy coefficients on one axis would
> compare quantities that do not share an operational scale.

The comparable view is empirical: for every customer in the training pool, each raw
feature contributes an additive term to the logit, and those terms sum exactly to
the logit (section 4). Describing the **distribution** of each feature's term puts
all {model["n_raw_features"]} on the same axis — log-odds — in the population the
model was fitted on.

| Property | Value |
| --- | --- |
| Population | {dispersion["population"]}, {dispersion["n_rows"]:,} rows |
| Unit | {dispersion["unit"]} |
| Primary metric | **`{dispersion["ranking_metric"]}`** |
| Definition | {dispersion["ranking_metric_definition"]} |
| Fixed before any value was computed | \
**{dispersion["ranking_metric_fixed_before_computing_values"]}** |
| Supporting metrics | {", ".join(f"`{name}`" for name in dispersion["supporting_metrics"])} |
| `mean_absolute_centered_contribution` | \
{dispersion["mean_absolute_centered_contribution_definition"]} |

### Ranking by {dispersion["ranking_name"]}

{_dispersion_table(results)}

{_figure(figures["dispersion"], "Raw features ranked by empirical contribution dispersion")}

**What this ranking is, stated precisely.** It is the
*{dispersion["ranking_name"]}*. It is **not**
{", ".join(f"*{name}*" for name in dispersion["ranking_is_not"])}. It depends on:

{chr(10).join(f"- {item};" for item in dispersion["depends_on"])}

Change any of those and the ordering can change without a single prediction of the
frozen model changing.

**The ranking is representation-dependent where features are redundant**
(`representation_dependent: {dispersion["representation_dependent"]}`). It is an
exact decomposition of the frozen function *under the current encoding*, but the
structurally coupled features of section 6.1 **split between them the variability
of one underlying state**. The features carrying a coupled level are
{", ".join(f"`{name}`" for name in dispersion["features_with_a_structurally_coupled_level"])}.
Four consequences follow, and none of them is optional:

- **Do not sum ranks or dispersion values** across features
  (`ranks_may_be_summed: {dispersion["ranks_may_be_summed"]}`). The coupled part of
  their variability is the same quantity counted more than once.
- **Do not read the coupled service features as that many independent signals**
  (`coupled_features_are_independent_signals: \
{dispersion["coupled_features_are_independent_signals"]}`).
- **Do not call any position importance without the qualifier.** The only sanctioned
  phrase is *{dispersion["ranking_name"]}*.
- **A different representation could redistribute the same variability.** An encoding
  that dropped the redundant sentinels, or that merged them into one indicator, could
  preserve very similar predictive behaviour while moving contributions between
  columns — and would produce a different ordering here.

**One arithmetic caveat, stated rather than left for a reader to notice.** For the
three numeric features the metric reduces almost exactly to `|standardised
coefficient|`, because the scaler was fitted on this same pool and the standardised
columns therefore have unit variance on it. That is an identity of the population
being described, not independent corroboration of the coefficient.

{_figure(figures["direction"], "Direction and width of each feature's modelled contribution")}

## 8. Strongest modelled contributions

Read as statements about the **frozen function**, never as effects.

### Numeric

{
        chr(10).join(
            f"- `{term['feature']}`: β = {term['standardized_coefficient']:+.4f} per 1 SD. A "
            f"higher standardised value is associated with a "
            f"**{_direction(term['standardized_coefficient'])}** modelled churn score, holding "
            f"the other transformed features fixed (modelled odds ratio per 1 SD: "
            f"{term['odds_ratio_per_1_sd']:.4f})."
            for term in results.numeric_terms
        )
    }

### The {N_HIGHLIGHTED} widest within-feature categorical contrasts

{_highlight_table(results)}

Two different readings, and the last column decides which one applies.

- **Counterfactual supported.** *Within the frozen model, encoding that feature at
  the first level instead of the second — holding every other transformed feature
  fixed — changes the model's log-odds by that amount.* Still not a claim that
  moving a customer between those states would change their behaviour.
- **Counterfactual not supported.** *The algebraic contrast between those two
  encoded levels is that amount.* The arithmetic is identical and exact, but one of
  the levels is structurally coupled (section 6.1), so "change only this feature"
  describes a row that does not occur in the training pool.

### Odds are not probability

An `odds_ratio` of 2 does **not** mean the probability doubles. It multiplies the
odds `p / (1 − p)` by 2, and how much `p` itself moves depends entirely on where it
started:

```text
p = 0.10  →  odds 0.111  →  ×2 = 0.222  →  p = 0.182   (+8.2 points)
p = 0.33  →  odds 0.493  →  ×2 = 0.985  →  p = 0.496   (+16.7 points)
p = 0.80  →  odds 4.000  →  ×2 = 8.000  →  p = 0.889   (+8.9 points)
```

The effect on the probability is largest near the middle of the scale and small in
either tail. Near the frozen threshold of
{decision["threshold_probability"]:.4f} it is close to its largest.

The three lines above are arithmetic on the definition of odds, not results
computed from this dataset.

## 9. Comparison with the Phase 3 EDA

Two different quantities, and they are kept apart:

- **EDA** measured a *marginal, descriptive association* — one feature against
  churn, nothing held fixed.
- **A coefficient here** describes *conditional behaviour of the frozen model* —
  what the fitted function does with one encoded input while every other encoded
  input is held constant.

They can disagree, and a disagreement does **not** automatically mean either is
wrong.

{_eda_table(results)}

### The instructive disagreement: `MonthlyCharges`

The Phase 3 EDA recorded a **positive** marginal association: churners had higher
monthly charges (median 79.65 against 64.43, rank-biserial +0.242). The frozen
model's standardised coefficient for `MonthlyCharges` is
**{
        next(
            t["standardized_coefficient"]
            for t in results.numeric_terms
            if t["feature"] == "MonthlyCharges"
        ):+.4f}** — the opposite sign.

That is not a contradiction, and the EDA itself predicted it: the same report
recorded that **the marginal association reverses under conditioning on
`InternetService`** — inside every internet tier, the churners' median monthly
charge was not above that of the customers retained. The marginal association is
carried substantially by tier composition, since churners are over-represented in
the expensive fiber tier. A model that also holds `InternetService` fixed is
answering a different question than a marginal comparison, and it answers it with
a different sign.

`TotalCharges` shows the mirror image: marginally **lower** among churners
(rank-biserial −0.303), but with a **positive** conditional coefficient
({
        next(
            t["standardized_coefficient"]
            for t in results.numeric_terms
            if t["feature"] == "TotalCharges"
        ):+.4f}) once `tenure` is in the model —
and the EDA recorded Spearman 0.889 between the two, plus 0.9996 against
`tenure × MonthlyCharges`. When two inputs carry an overlapping signal, how the fit
divides it between them is a property of the fit, not of the world.

This is the concrete reason the report never presents a coefficient as "the effect
of" anything.

## 10. Interpretation limitations

{_limitations(results)}

## 11. Engineering status

| Confirmation | Value |
| --- | --- |
| Model changed | **{results.methodology["model_changed"]}** |
| Threshold changed | **{results.methodology["threshold_changed"]}** |
| Calibration changed | **{results.methodology["calibration_changed"]}** |
| Selection performed | **{results.methodology["selection_after_analysis"]}** |
| Holdout-based feature ranking | **{results.methodology["post_hoc_holdout_selection"]}** |
| Fits | **{results.methodology["fit_calls"]}** |
| Models loaded | {results.methodology["models_loaded"]} |
| Pipeline byte-identical after the run | \
**{provenance["pipeline_unchanged_by_analysis"]}** |
| Model fingerprint | `{provenance["model_fingerprint_sha256_after_load"][:16]}…` (verified: \
{provenance["model_fingerprint_verified"]}) |
| Pipeline SHA-256 | `{provenance["pipeline_sha256_after_analysis"][:16]}…` |
| New dependencies | **{results.methodology["new_dependencies_added"]}** |

```text
no model change
no selection
no holdout-based feature ranking
```

---

`{results.analysis_type}` · `selection_after_analysis: \
{results.methodology["selection_after_analysis"]}` · \
`holdout_used_for_global_interpretation: \
{results.holdout_used_for_global_interpretation}`

{results.methodology["statement"]}.

Top of the ranking by {dispersion["ranking_name"]}: `{top["feature"]}`
({dispersion["ranking_metric"]} = {top["ranking_value"]:.4f}). That phrase is the
whole claim — not "the most important factor".
"""


def interpret() -> tuple[InterpretationResults, list, list, list]:
    """Run the interpretation. Gates first, decomposition second."""
    manifest = verify_split_manifest(load_split_manifest())
    policy = load_decision_policy()
    checks, pipeline = verify_or_raise(policy, load_threshold_results(), load_calibration_results())
    logger.info("Freeze verified: %d integrity checks passed.", len(checks))

    pipeline_path = PROJECT_ROOT / policy.artifacts.pipeline_path
    digest_before = file_digest(pipeline_path)
    loaded_fingerprint = model_fingerprint(pipeline)
    if loaded_fingerprint != policy.artifacts.model_fingerprint_sha256:
        raise IntegrityError(
            "The loaded pipeline is not the frozen model; the interpretation was not run."
        )

    # The training pool. The holdout is never loaded in this phase: it was
    # consumed by Phase 9D, and ranking features on it would be post-hoc mining.
    frame = load_training_pool()
    if len(frame) != manifest.n_rows_training:
        raise InterpretationError(
            f"{len(frame)} rows against {manifest.n_rows_training} in the frozen manifest."
        )
    features = build_feature_matrix(frame)

    terms = extract_terms(pipeline, policy.decision_rule.positive_class_label)
    groups = feature_groups(terms)
    if terms.n_transformed_features != policy.feature_contract.n_transformed_features:
        raise InterpretationError(
            f"{terms.n_transformed_features} transformed features against "
            f"{policy.feature_contract.n_transformed_features} in the frozen contract."
        )
    if len(groups) != policy.feature_contract.n_features:
        raise InterpretationError(
            f"{len(groups)} raw feature groups against {policy.feature_contract.n_features} in "
            "the frozen contract."
        )

    reconstruction = verify_reconstruction(pipeline, terms, groups, features)

    transformed = transform_features(pipeline, features)
    contributions = group_contributions(terms, groups, transformed)
    dispersions = list(contribution_dispersion(groups, contributions))
    ranking = rank_by_dispersion(dispersions, RANKING_METRIC)

    # Structural dependencies, verified on this same population. The rules are
    # declared in the module from the Phase 2 product semantics; what holds and
    # what does not is decided here, by counting.
    structural_checks = list(check_structural_dependencies(features))
    coupled_levels = coupled_levels_from(structural_checks)
    coupled_blocks = list(coupled_column_blocks(terms, groups, transformed))

    numeric = list(numeric_terms(terms, groups))
    categorical = list(categorical_terms(terms, groups, coupled_levels))

    digests = {artefact: file_digest(PROJECT_ROOT / artefact) for artefact in UPSTREAM_ARTEFACTS}
    results = build_results(
        policy,
        terms,
        groups,
        reconstruction,
        logit_of(policy.threshold.final_threshold),
        numeric,
        categorical,
        dispersions,
        ranking,
        structural_checks,
        coupled_blocks,
        coupled_levels,
        {
            "partition": "training pool",
            "loader": "churn.preprocessing.splitting.load_training_pool",
            "n_rows": int(len(features)),
            "holdout_loaded": False,
            "why": (
                "the holdout was consumed by the Phase 9D evaluation; ranking features on it "
                "would be post-hoc mining of a spent sample. The training pool is also the "
                "population the coefficients and the scaler statistics were learned from"
            ),
            "target_used": False,
            "target_note": (
                "no label is read in this phase: the decomposition describes the function, and no "
                "performance metric is computed anywhere"
            ),
        },
        digests,
        source_digests(INFERENCE_SOURCE_FILES),
        source_digests(CONFIGURATION_FILES),
        loaded_fingerprint,
        digest_before,
        file_digest(pipeline_path),
    )
    write_results(results)

    failures = invariant_failures(load_results())
    if failures:
        raise InterpretationError(
            "The record just written violates its own invariants:\n  " + "\n  ".join(failures)
        )
    logger.info("Record invariants verified.")
    return results, numeric, categorical, dispersions


def verify() -> int:
    """Check the existing Phase 10 record. **Read-only.** Returns an exit code."""
    results = load_results()
    failures = invariant_failures(results)

    for failure in failures:
        logger.error("INVARIANT FAILED: %s", failure)
    if failures:
        logger.error("%d invariant(s) failed.", len(failures))
        return 1

    logger.info("All record invariants hold.")
    logger.info(
        "%s: %d raw features, %d transformed, max reconstruction error %.3g, "
        "holdout_used_for_global_interpretation=%s",
        results.analysis_type,
        results.model["n_raw_features"],
        results.model["n_transformed_features"],
        max(
            float(results.reconstruction["max_abs_logit_error"]),
            float(results.reconstruction["max_abs_probability_error"]),
            float(results.reconstruction["max_grouped_reconstruction_error"]),
        ),
        results.holdout_used_for_global_interpretation,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the interpretation, or verify an existing record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the existing interpretation record and change nothing",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        if arguments.verify:
            return verify()
        results, numeric, categorical, dispersions = interpret()
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
        logger.error("FREEZE INTEGRITY FAILED — the interpretation did not run.\n%s", error)
        return 4
    except ReconstructionError as error:
        logger.error("RECONSTRUCTION FAILED — no report was produced.\n%s", error)
        return 5
    except InterpretationError as error:
        logger.error("%s", error)
        return 6

    figures = build_figures(results, numeric, categorical, dispersions, PROJECT_ROOT)
    Path(REPORT_PATH).write_text(
        build_report(results, figures),
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote %s", REPORT_PATH)
    logger.info("Wrote %s", RESULTS_PATH)
    logger.info(
        "Reconstruction: logit %.3g, probability %.3g, grouped %.3g (tolerance %.0e)",
        results.reconstruction["max_abs_logit_error"],
        results.reconstruction["max_abs_probability_error"],
        results.reconstruction["max_grouped_reconstruction_error"],
        results.reconstruction["tolerance"],
    )
    logger.info(
        "Ranking by %s (%s): %s",
        results.contribution_dispersion["ranking_name"],
        RANKING_METRIC,
        ", ".join(
            f"{row['rank']}. {row['feature']}"
            for row in results.contribution_dispersion["ranking"][:5]
        ),
    )
    logger.info(
        "Supporting metrics recorded: %s. Nothing was fitted, selected or re-thresholded.",
        ", ".join(SUPPORTING_METRICS),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
