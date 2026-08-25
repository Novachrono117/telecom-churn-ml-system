"""Machine-readable record of the Phase 10 interpretation of the frozen model.

A **description of a function**, not a new experiment. The model, the calibration
policy and the threshold were frozen in `9d1db49`, evaluated in `8c8e266` and
diagnosed in `5e9c4e9`; this artefact writes down what that frozen function
computes, in terms of the 19 contracted features.

Four properties are written into the record rather than left to a reader's
charity, because they are what separates an interpretation from a second round of
modelling:

* nothing was fitted (``fit_calls: 0``) and the pipeline is byte-identical
  afterwards;
* the decomposition **reproduces** the pipeline's own ``decision_function`` and
  ``predict_proba`` to a recorded numerical tolerance, both per transformed
  column and after regrouping into the 19 raw features;
* the global interpretation was computed on the **training pool**, never on the
  holdout (``holdout_used_for_global_interpretation: false``);
* no approximate explainer was introduced — no SHAP, no permutation importance,
  no partial dependence — because the model's response surface is known in
  closed form.

No individual is persisted. Per-row contributions exist only inside the process;
this file contains coefficients, contrasts and distribution summaries.

Same deliberate absence as every earlier phase: **no timestamp**, so a rerun on
an unchanged repository reproduces the file byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from churn.config import PROJECT_ROOT
from churn.modeling.interpretation import (
    ANALYSIS_TYPE,
    MEAN_ABSOLUTE_CENTERED_DEFINITION,
    RANKING_IS_NOT,
    RANKING_METRIC,
    RANKING_NAME,
    STD_DEFINITION,
    SUPPORTING_METRICS,
    CategoricalTerm,
    ContributionDispersion,
    CoupledBlock,
    LinearTerms,
    NumericTerm,
    Reconstruction,
    StructuralCheck,
)

logger = logging.getLogger(__name__)

RESULTS_PATH = PROJECT_ROOT / "reports" / "experiments" / "model_interpretation_results.json"

SCHEMA_VERSION = 1
EXPERIMENT_NAME = "phase10-frozen-model-interpretation"

#: The three commits this phase interprets, and does not revise.
FREEZE_COMMIT = "9d1db4962769093a617f1db2db66354536acffb3"
FREEZE_COMMIT_SHORT = "9d1db49"
FREEZE_COMMIT_SUBJECT = "feat: freeze final churn model and decision policy"

EVALUATION_COMMIT = "8c8e2663a2756a0fbb95188e96d7771bd9816265"
EVALUATION_COMMIT_SHORT = "8c8e266"
EVALUATION_COMMIT_SUBJECT = "feat: evaluate frozen churn model on final holdout"

ERROR_ANALYSIS_COMMIT = "5e9c4e95a48bad4c9e6f4428fc2daac63032bdc5"
ERROR_ANALYSIS_COMMIT_SHORT = "5e9c4e9"
ERROR_ANALYSIS_COMMIT_SUBJECT = "feat: analyze final churn prediction errors"

#: Artefacts this phase reads and must not modify.
UPSTREAM_ARTEFACTS: tuple[str, ...] = (
    "reports/split_manifest.json",
    "reports/decision_policy.json",
    "reports/model_freeze_report.md",
    "reports/holdout_evaluation_report.md",
    "reports/error_analysis_report.md",
    "reports/experiments/baseline_results.json",
    "reports/experiments/feature_engineering_results.json",
    "reports/experiments/model_comparison_results.json",
    "reports/experiments/hgb_representation_results.json",
    "reports/experiments/tuning_results.json",
    "reports/experiments/calibration_results.json",
    "reports/experiments/threshold_results.json",
    "reports/experiments/holdout_results.json",
    "reports/experiments/error_analysis_results.json",
)

#: Observations recorded in the Phase 3 EDA, quoted so the report can set a
#: **marginal, descriptive** association beside a **conditional, modelled**
#: coefficient without inventing either. ``model_relation`` says what the
#: comparison is; it never says which of the two is correct, because they are
#: different quantities and disagreement is not by itself an error in either.
EDA_COMPARISONS: tuple[dict[str, str], ...] = (
    {
        "feature": "tenure",
        "marginal_observation": (
            "low tenure was associated with higher churn (median 10 months for churners against "
            "38 for those retained; rank-biserial -0.480)"
        ),
        "source": "reports/eda_report.md, reports/figures/eda/03_churn_rate_by_tenure_band.png",
    },
    {
        "feature": "MonthlyCharges",
        "marginal_observation": (
            "churners showed HIGHER monthly charges marginally (median 79.65 against 64.43; "
            "rank-biserial +0.242), but the EDA also recorded that this association REVERSES once "
            "InternetService is held fixed: inside every tier the churners' median monthly charge "
            "was not above that of those retained"
        ),
        "source": (
            "reports/eda_report.md, reports/figures/eda/11_internet_monthly_charges_by_churn.png"
        ),
    },
    {
        "feature": "TotalCharges",
        "marginal_observation": (
            "churners showed LOWER total charges marginally (median 703.55 against 1683.60; "
            "rank-biserial -0.303). The EDA also recorded Spearman 0.889 between TotalCharges and "
            "tenure, and 0.9996 against tenure x MonthlyCharges"
        ),
        "source": "reports/eda_report.md",
    },
    {
        "feature": "Contract",
        "marginal_observation": (
            "month-to-month contracts showed a far higher churn rate than one-year or two-year "
            "contracts, and the gap survived inside every tenure band"
        ),
        "source": "reports/eda_report.md, reports/figures/eda/04_churn_rate_by_contract.png",
    },
    {
        "feature": "InternetService",
        "marginal_observation": "fiber optic showed a higher churn rate than DSL or no internet",
        "source": "reports/eda_report.md, reports/figures/eda/06_service_churn_rates.png",
    },
    {
        "feature": "PaymentMethod",
        "marginal_observation": "electronic check was associated with higher churn",
        "source": (
            "reports/eda_report.md, reports/figures/eda/09_churn_rate_by_payment_method.png"
        ),
    },
    {
        "feature": "SeniorCitizen",
        "marginal_observation": "a descriptive difference in churn rate was recorded",
        "source": "reports/eda_report.md",
    },
    {
        "feature": "gender",
        "marginal_observation": "association with churn was essentially null",
        "source": "reports/eda_report.md, reports/figures/eda/15_association_ranking.png",
    },
)


def _round(value: float | None) -> float | None:
    """Return the value as a plain float, or ``None`` if it is absent or degenerate.

    **Deliberately not rounded**, unlike the earlier phases' records. Those
    persist estimates, where a sixth decimal is noise; this one persists the
    *parameters of the function being described*, and a rounded coefficient
    describes a slightly different function than the frozen one. Rounding
    ``TotalCharges``' raw coefficient to ten places would already cost it four
    significant digits, and the reconstruction gate in section 4 asserts equality
    to 1e-12 — a claim the artefact would then fail to support on its own numbers.

    Reproducibility does not need rounding here: the values come from a frozen
    joblib artefact and :func:`json.dumps` writes the shortest decimal string that
    reads back as the same double, so the file is byte-identical across runs.
    """
    if value is None:
        return None
    number = float(value)
    return None if not np.isfinite(number) else number


def _round_deep(value: object) -> object:
    """Normalise every float inside a nested structure, leaving everything else."""
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return _round(value)
    if isinstance(value, Mapping):
        return {key: _round_deep(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_round_deep(item) for item in value]
    return value


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in binary so line endings cannot alter it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InterpretationResults(BaseModel):
    """The complete Phase 10 record."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    experiment: str
    phase: str
    analysis_type: str
    selection_allowed: bool
    model_change_allowed: bool
    fit_calls: int
    holdout_used_for_global_interpretation: bool
    interpretation_population: dict[str, object]
    provenance: dict[str, object]
    upstream_artefact_digests: dict[str, str]
    source_digests: dict[str, str]
    configuration_digests: dict[str, str]
    model: dict[str, object]
    exact_functional_form: dict[str, object]
    reconstruction: dict[str, object]
    frozen_decision: dict[str, object]
    feature_mapping: dict[str, object]
    structural_dependencies: dict[str, object]
    numeric_terms: list[dict[str, object]]
    categorical_terms: list[dict[str, object]]
    contribution_dispersion: dict[str, object]
    intercept_interpretation: dict[str, object]
    eda_comparison: list[dict[str, str]]
    methodology: dict[str, object]
    limitations: list[str]
    environment: dict[str, str]
    methodological_notes: list[str]


METHODOLOGICAL_NOTES: tuple[str, ...] = (
    "This phase INTERPRETS a frozen model. It is not model selection. No estimator, "
    "hyperparameter, feature, preprocessing step, calibration policy or threshold was chosen, "
    "compared, removed, added or changed, and nothing recorded here may be used to change one.",
    "NOTHING WAS FITTED. The pipeline was loaded from the frozen artefact, its model fingerprint "
    "was verified before use, only `.transform`, `.decision_function` and `.predict_proba` were "
    "called, and the file is byte-identical afterwards. No fit, fit_transform, GridSearchCV, "
    "RandomizedSearchCV or CalibratedClassifierCV appears anywhere in this phase.",
    "THE GLOBAL INTERPRETATION USES THE TRAINING POOL, NEVER THE HOLDOUT. The holdout was consumed "
    "by the Phase 9D evaluation; ranking features on it would be post-hoc mining of a spent "
    "sample. The Phase 10 modules do not call load_holdout at all.",
    "NO SHAP, NO LIME, NO ELI5, NO PERMUTATION IMPORTANCE, NO PARTIAL DEPENDENCE AND NO ICE. Those "
    "are approximations for models whose response surface is unknown. This model is additive and "
    "linear after a fixed transformation, so its decomposition is exact and closed-form; an "
    "approximate estimator would add a dependency and an approximation error in exchange for "
    "nothing.",
    "THE DECOMPOSITION REPRODUCES THE PIPELINE EXACTLY, and that is a gate rather than a claim. "
    "The manual logit is checked against decision_function, its sigmoid against predict_proba, and "
    "the sum of the 19 regrouped contributions against decision_function again. A failure raises "
    "and the phase does not produce a report.",
    "THE ONE-HOT PARAMETERISATION IS REDUNDANT AND NO BASELINE CATEGORY WAS DROPPED. Every level "
    "of every categorical feature has its own column and the model also has an intercept, so the "
    "parameters are not identified: a constant can be moved between a feature's levels and the "
    "intercept without changing a single prediction. exp(beta) OF A SINGLE LEVEL IS THEREFORE NOT "
    "AN ODDS RATIO AGAINST A REFERENCE CATEGORY, and no reference category is declared anywhere.",
    "THE IDENTIFIED CATEGORICAL QUANTITY IS THE WITHIN-FEATURE CONTRAST beta_a - beta_b, whose "
    "value is invariant to that shift, and its exponential exp(beta_a - beta_b) is the modelled "
    "odds ratio between two levels of the SAME feature holding every other transformed feature "
    "fixed. It is a statement about the frozen predictive function, not a causal effect.",
    "NO NAIVE abs(coefficient) RANKING IS PRODUCED. Mixing standardised numeric coefficients with "
    "0/1 dummy coefficients on one axis would compare quantities that do not share an operational "
    "scale. The comparable view across the 19 raw features is the empirical contribution "
    "dispersion, which is a joint property of the coefficients, the preprocessing AND the training "
    "pool's composition.",
    "THE RANKING METRIC WAS FIXED IN CODE BEFORE ANY VALUE WAS COMPUTED: the standard deviation of "
    "each raw feature's additive contribution to the logit over the training pool. The other "
    "dispersion measures are reported beside it as supporting columns, not as alternatives chosen "
    "after seeing the ordering.",
    "THE RANKING IS NOT AN IMPORTANCE RANKING IN ANY UNIVERSAL SENSE. It depends on the fitted "
    "coefficients, on the preprocessing, on the distribution of the training pool and on the "
    "correlations among features. It is not causal importance, not business importance and not a "
    "property of telecom churn.",
    "L2 REGULARISATION IS ACTIVE (C = 1.0). It shrinks coefficients toward zero and, when features "
    "are correlated, distributes a shared association among them rather than assigning it to one. "
    "tenure, TotalCharges and Contract are correlated in this dataset, so their individual "
    "coefficients must not be read as if each variable were experimentally independent.",
    "THE INTERCEPT IS NOT THE RISK OF AN AVERAGE CUSTOMER. The numeric features are centred, so "
    "their zero is the training-pool mean, but every categorical feature always has exactly one "
    "level active, and the intercept alone corresponds to no such customer. It is the constant of "
    "the linear function, nothing more.",
    "ODDS ARE NOT PROBABILITY. An odds ratio of 2 does NOT mean the probability doubles: it "
    "multiplies p/(1-p) by 2, and the resulting change in p depends on where p started. Near the "
    "frozen threshold the effect is largest; in either tail it is small.",
    "THE PROBABILITIES ARE UNCALIBRATED BY THE PHASE 9A DECISION (calibration_policy = NONE). The "
    "sigmoid of the logit is the model's own output, not a validated likelihood for an individual "
    "customer.",
    "NO CAUSAL CLAIM IS MADE ANYWHERE. A coefficient describes how the frozen function responds to "
    "a change in an encoded input, holding the other encoded inputs fixed. It does not establish "
    "that changing the underlying attribute of a real customer would change their behaviour.",
    "THE PHASE 9E ERROR ANALYSIS DID NOT SELECT WHAT IS EXPLAINED HERE. All 19 raw features and "
    "all 46 transformed columns are reported; none was included or excluded because of anything "
    "observed in the holdout errors.",
    "NO INDIVIDUAL EXPLANATION IS PERSISTED AND NO IDENTIFIER IS PERSISTED. Per-row contributions "
    "exist only inside the process, and no customer was selected for a case study. A future "
    "serving interface can explain a single prediction when it receives a real payload.",
    "NO UNCERTAINTY INTERVAL IS ATTACHED TO ANY COEFFICIENT. Standard errors for a penalised "
    "logistic regression are not the textbook maximum-likelihood ones, and none was computed. "
    "Every coefficient here is a point value of one fitted object.",
    "NO NEW PERFORMANCE ESTIMATE WAS PRODUCED. This phase reports no accuracy, precision, recall, "
    "F1, ROC-AUC or average precision. The final estimate is the Phase 9D one and is unchanged.",
    "SOME RAW FEATURES ARE STRUCTURALLY COUPLED IN THIS DATASET, and the coupling was VERIFIED on "
    "the training pool rather than assumed. Where an equivalence holds on every row, the two "
    "encoded states are the same state, several one-hot columns become the SAME VECTOR, and those "
    "columns are perfectly collinear. Every such relationship is recorded in "
    "structural_dependencies with its own row counts and violation count.",
    "REPEATED EQUAL COEFFICIENTS ACROSS COUPLED COLUMNS ARE NOT INDEPENDENT EVIDENCE FROM SEVERAL "
    "FEATURES. When columns are identical, the L2 penalty is minimised by splitting the shared "
    "weight evenly among them, so equal coefficients are a property of the penalty and the "
    "parameterisation, not a demonstration of one effect per column. The quantity the model "
    "actually applies is the block AGGREGATE, which is recorded beside the per-column values.",
    "AN ALGEBRAIC CONTRAST IS NOT AUTOMATICALLY A RAW SINGLE-FEATURE COUNTERFACTUAL. beta_a - "
    "beta_b is always the exact difference the linear function assigns to two encoded levels. But "
    "when a level participates in a verified structural dependency, moving only that one-hot while "
    "holding the other raw features fixed describes a combination that does not occur in the "
    "population, so the contrast is flagged structurally_coupled and "
    "raw_single_feature_counterfactual_supported is false. NO CONTRAST IS REMOVED OR ALTERED — "
    "only qualified.",
    "THE CONTRIBUTION DISPERSION RANKING IS REPRESENTATION-DEPENDENT WHERE FEATURES ARE REDUNDANT. "
    "It is an exact decomposition of the frozen function under the current encoding, but "
    "structurally redundant features can SPLIT BETWEEN THEM the variability of one underlying "
    "state. Ranks must therefore not be summed, coupled service features must not be read as that "
    "many independent signals, and a different representation with similar predictive behaviour "
    "could redistribute the same variability across different columns.",
)

LIMITATIONS: tuple[str, ...] = (
    "Analyst exposure: the Phase 3 exploratory analysis ran on all 7,043 rows, including those "
    "that later formed the holdout. The protection against fitting, tuning and selection began at "
    "Phase 4, so an optimistic bias this project cannot quantify precedes it.",
    "L2 regularisation (C = 1.0) shrinks every coefficient toward zero, so every magnitude "
    "reported here is a penalised magnitude, not a maximum-likelihood one.",
    "Correlated features share their association. tenure and TotalCharges have Spearman 0.889 in "
    "this dataset, and TotalCharges tracks tenure x MonthlyCharges at 0.9996, so the split of a "
    "shared signal between them is a property of the fit, not of the world.",
    "The one-hot parameterisation is redundant and no baseline was dropped, so individual dummy "
    "coefficients are not identified and only within-feature contrasts are.",
    "Structural coupling between raw features, verified on the training pool, means some one-hot "
    "columns are the same vector. Their coefficients are one point in a family of equivalent "
    "parameterisations, repeated equal values are not independent evidence from several features, "
    "and contrasts touching a coupled level are algebraic contrasts rather than single-feature "
    "counterfactuals a customer could undergo.",
    "The contribution dispersion ranking is representation-dependent wherever features are "
    "structurally redundant: coupled features split the variability of one underlying state "
    "between them, so ranks must not be summed and a different encoding with similar predictive "
    "behaviour could redistribute the same variability.",
    "No uncertainty interval is attached to any coefficient.",
    "No causal claim can be supported. The dataset is a cross-sectional snapshot with no time "
    "ordering and no counterfactual.",
    "The contribution dispersion is a property of THIS training pool's composition as well as of "
    "the model. A population with a different mix of contracts or tenures would produce a "
    "different dispersion from the same coefficients.",
    "For the three numeric features the dispersion metric reduces almost exactly to the absolute "
    "standardised coefficient, because the scaler was fitted on this same pool and the "
    "standardised columns therefore have unit variance on it. That is an arithmetic identity of "
    "the population being described, not an independent confirmation of the coefficient.",
    "The interpretation is valid for THIS frozen model on THIS dataset snapshot. It is not a "
    "general account of telecom churn and does not transfer to another operator, another period or "
    "another model.",
    "The probabilities are uncalibrated by the Phase 9A decision, so a modelled probability is the "
    "model's own output rather than a validated likelihood.",
)


def build_results(
    policy: object,
    terms: LinearTerms,
    groups: Sequence[object],
    reconstruction: Reconstruction,
    threshold_logit: float,
    numeric: Sequence[NumericTerm],
    categorical: Sequence[CategoricalTerm],
    dispersions: Sequence[ContributionDispersion],
    ranking: Sequence[Mapping[str, object]],
    structural_checks: Sequence[StructuralCheck],
    coupled_blocks: Sequence[CoupledBlock],
    coupled_levels: Mapping[str, Sequence[str]],
    population: Mapping[str, object],
    digests: Mapping[str, str],
    source_digests: Mapping[str, str],
    configuration_digests: Mapping[str, str],
    loaded_model_fingerprint: str,
    pipeline_digest_before: str,
    pipeline_digest_after: str,
) -> InterpretationResults:
    """Assemble the Phase 10 record."""
    numeric_groups = [group for group in groups if group.kind == "numeric"]
    categorical_groups = [group for group in groups if group.kind == "categorical"]

    return InterpretationResults(
        schema_version=SCHEMA_VERSION,
        experiment=EXPERIMENT_NAME,
        phase="10",
        analysis_type=ANALYSIS_TYPE,
        selection_allowed=False,
        model_change_allowed=False,
        fit_calls=0,
        holdout_used_for_global_interpretation=False,
        interpretation_population=dict(population),
        provenance={
            "freeze_commit": FREEZE_COMMIT,
            "freeze_commit_short": FREEZE_COMMIT_SHORT,
            "freeze_commit_subject": FREEZE_COMMIT_SUBJECT,
            "evaluation_commit": EVALUATION_COMMIT,
            "evaluation_commit_short": EVALUATION_COMMIT_SHORT,
            "evaluation_commit_subject": EVALUATION_COMMIT_SUBJECT,
            "error_analysis_commit": ERROR_ANALYSIS_COMMIT,
            "error_analysis_commit_short": ERROR_ANALYSIS_COMMIT_SHORT,
            "error_analysis_commit_subject": ERROR_ANALYSIS_COMMIT_SUBJECT,
            "decision_policy": "reports/decision_policy.json",
            "decision_policy_sha256": digests["reports/decision_policy.json"],
            "split_manifest_sha256": digests["reports/split_manifest.json"],
            "model_fingerprint_sha256_recorded": policy.artifacts.model_fingerprint_sha256,
            "model_fingerprint_sha256_after_load": loaded_model_fingerprint,
            "model_fingerprint_verified": (
                loaded_model_fingerprint == policy.artifacts.model_fingerprint_sha256
            ),
            "pipeline_sha256_recorded": policy.artifacts.pipeline_sha256,
            "pipeline_sha256_before_analysis": pipeline_digest_before,
            "pipeline_sha256_after_analysis": pipeline_digest_after,
            "pipeline_unchanged_by_analysis": pipeline_digest_before == pipeline_digest_after,
            "refitted_here": False,
            "reopened_here": False,
        },
        upstream_artefact_digests=dict(digests),
        source_digests=dict(source_digests),
        configuration_digests=dict(configuration_digests),
        model={
            "estimator": terms.estimator,
            "classes": list(terms.classes),
            "intercept": _round(terms.intercept),
            "n_iter": list(terms.n_iter),
            "converged": all(
                value < policy.model.hyperparameters["max_iter"] for value in terms.n_iter
            ),
            "max_iter": policy.model.hyperparameters["max_iter"],
            "hyperparameters": dict(policy.model.hyperparameters),
            "n_transformed_features": terms.n_transformed_features,
            "n_coefficients": int(terms.coefficients.size),
            "n_raw_features": len(groups),
            "n_numeric_features": len(numeric_groups),
            "n_categorical_features": len(categorical_groups),
            "engineered_features": list(policy.feature_contract.engineered),
            "positive_class_label": terms.positive_class_label,
            "positive_class_column": terms.positive_class_column,
            "positive_class_resolution": (
                "resolved from classifier.classes_, never assumed. decision_function scores "
                "classes_[1], which is the positive class here"
            ),
        },
        exact_functional_form={
            "logit": (
                "logit(P(Churn=1|x)) = intercept + sum_j coefficient_j * transformed_feature_j"
            ),
            "probability": "P(Churn=1|x) = sigmoid(logit) = 1 / (1 + exp(-logit))",
            "grouped": (
                "logit = intercept + sum over the 19 raw features of contribution_g(x), where "
                "contribution_g = sum over that feature's transformed columns of coefficient_j * "
                "z_j"
            ),
            "numeric_contribution": "coefficient * (raw_value - scaler_mean) / scaler_scale",
            "categorical_contribution": (
                "coefficient of the ACTIVE level; the feature's other columns are zero, and an "
                "unseen category is all-zeros under handle_unknown='ignore'"
            ),
            "approximation_used": False,
            "explainer_library_used": None,
        },
        reconstruction={
            **reconstruction.as_dict(),
            "population": population["partition"],
            "logit_reference": "pipeline.decision_function(X)",
            "probability_reference": "pipeline.predict_proba(X)[:, positive_class_column]",
            "grouped_reference": "pipeline.decision_function(X)",
            "tolerance_justification": (
                "Both quantities are order 1 and are computed as a dot product of "
                f"{terms.n_transformed_features} float64 terms, so the accumulated rounding error "
                "is bounded by a few multiples of 2**-52 ~ 2.2e-16. The tolerance sits about four "
                "orders of magnitude above that noise floor and far below any error that would "
                "signal a wrong column, a missed intercept or a transposed coefficient"
            ),
            "gate": "the phase raises and produces no report if any error exceeds the tolerance",
        },
        frozen_decision={
            "threshold_probability": policy.threshold.final_threshold,
            "threshold_logit": _round(threshold_logit),
            "threshold_logit_definition": "log(threshold / (1 - threshold))",
            "comparison": policy.decision_rule.comparison,
            "positive_class_label": policy.decision_rule.positive_class_label,
            "equivalent_rule": (
                "predicted positive when model_logit >= threshold_logit, which is the SAME frozen "
                "rule as probability >= threshold, not a new one"
            ),
            "threshold_policy": policy.threshold.threshold_policy,
            "calibration_policy": policy.calibration.calibration_policy,
            "threshold_changed_here": False,
            "calibration_changed_here": False,
            "alternative_thresholds_scored": 0,
        },
        feature_mapping={
            "n_raw_features": len(groups),
            "n_transformed_columns": terms.n_transformed_features,
            "derivation": (
                "index arithmetic over the fitted encoder's own structure: the numeric block emits "
                "one column per scaled feature in feature_names_in_ order, then the categorical "
                "block emits len(categories_[i]) consecutive columns for the i-th categorical "
                "feature"
            ),
            "textual_parsing_used": False,
            "cross_checked_against_generated_names": True,
            "partition_verified": True,
            "groups": [
                {
                    "feature": group.feature,
                    "kind": group.kind,
                    "n_transformed_columns": group.n_columns,
                    "transformed_columns": [int(column) for column in group.columns],
                    "levels": list(group.levels),
                }
                for group in groups
            ],
        },
        structural_dependencies={
            "status": "verified empirically on the interpretation population",
            "population": population["partition"],
            "n_rows_checked": population["n_rows"],
            "holdout_used": False,
            "rules_declared_before_checking": True,
            "rules_source": (
                "the product semantics recorded in the Phase 2 data dictionary — a sentinel "
                "category such as 'No internet service' exists because the customer has no "
                "internet. NOT derived from the model's coefficients and NOT derived from the "
                "holdout"
            ),
            "n_rules_checked": len(structural_checks),
            "n_deterministic": sum(check.deterministic for check in structural_checks),
            "n_with_violations": sum(not check.deterministic for check in structural_checks),
            "rules": [check.as_dict(str(population["partition"])) for check in structural_checks],
            "coupled_levels": {feature: list(levels) for feature, levels in coupled_levels.items()},
            "coupled_column_blocks": [_round_deep(block.as_dict()) for block in coupled_blocks],
            "n_coupled_column_blocks": len(coupled_blocks),
            "detection_method": (
                "two independent passes. The rules above compare RAW CATEGORY VALUES row by row; "
                "the blocks below compare the TRANSFORMED COLUMNS for exact vector equality on the "
                "same population. Neither is a general constraint solver, and neither declares a "
                "coupling the data does not show"
            ),
            "interpretation_consequence": (
                "a contrast whose level participates in a deterministic rule is flagged "
                "structurally_coupled and carries raw_single_feature_counterfactual_supported = "
                "false: the algebraic difference beta_a - beta_b is still exact, but 'change only "
                "this feature and hold the rest fixed' would describe a row that does not occur. "
                "Repeated equal coefficients across a coupled block are one shared weight split by "
                "the L2 penalty, not one effect per feature"
            ),
            "contrasts_removed_or_altered": False,
            "coefficients_removed_or_altered": False,
        },
        numeric_terms=[_round_deep(term.as_dict()) for term in numeric],
        categorical_terms=[_round_deep(term.as_dict()) for term in categorical],
        contribution_dispersion={
            "population": population["partition"],
            "n_rows": population["n_rows"],
            "unit": "log-odds (the additive term this feature contributes to the logit)",
            "definition": (
                "contribution_g(x) is that raw feature's additive term in the logit, so intercept "
                "+ sum of the 19 contributions reproduces the logit exactly"
            ),
            "ranking_metric": RANKING_METRIC,
            "ranking_metric_definition": STD_DEFINITION,
            "ranking_metric_fixed_before_computing_values": True,
            "supporting_metrics": list(SUPPORTING_METRICS),
            "mean_absolute_centered_contribution_definition": MEAN_ABSOLUTE_CENTERED_DEFINITION,
            "ranking_name": RANKING_NAME,
            "ranking_is_not": list(RANKING_IS_NOT),
            "depends_on": [
                "the fitted coefficients",
                "the fitted preprocessing (scaler statistics and encoder categories)",
                "the composition of the training pool",
                "the correlations among the features",
                "the chosen representation, wherever features are structurally redundant",
            ],
            "representation_dependent": True,
            "redundancy_caveat": (
                "the ranking is an EXACT decomposition of the frozen function under the CURRENT "
                "encoding, but structurally redundant features split between them the variability "
                "of one underlying state. Consequently: do NOT sum ranks; do NOT read the coupled "
                "service features as that many independent signals; do NOT call any position "
                "importance without the qualifier; and note that a different representation with "
                "similar predictive behaviour could redistribute the same variability across "
                "different columns"
            ),
            "ranks_may_be_summed": False,
            "coupled_features_are_independent_signals": False,
            "features_with_a_structurally_coupled_level": sorted(coupled_levels),
            "by_feature": [_round_deep(record.as_dict()) for record in dispersions],
            "ranking": [_round_deep(dict(row)) for row in ranking],
        },
        intercept_interpretation={
            "value": _round(terms.intercept),
            "is_the_risk_of_an_average_customer": False,
            "why_not": (
                "the numeric features are centred, so their zero is the training-pool mean, but "
                "every categorical feature always has exactly one level active. A vector of all "
                "zeros in the transformed space is not a customer: it is a row with no gender, no "
                "contract and no internet service. The intercept is the constant of the linear "
                "function and nothing more"
            ),
            "redundant_parameterisation": True,
            "reference_level_declared": False,
        },
        eda_comparison=[dict(entry) for entry in EDA_COMPARISONS],
        methodology={
            "post_hoc_holdout_selection": False,
            "shap_used": False,
            "lime_used": False,
            "eli5_used": False,
            "permutation_importance_used": False,
            "pdp_used": False,
            "ice_used": False,
            "identifiers_persisted": False,
            "individual_explanations_persisted": False,
            "causal_claims": False,
            "naive_abs_coefficient_ranking_produced": False,
            "reference_category_declared": False,
            "structural_dependencies_verified_empirically": True,
            "structural_dependencies_assumed_without_checking": False,
            "coupled_contrasts_flagged": True,
            "coupled_contrasts_removed": False,
            "repeated_coefficients_read_as_independent_evidence": False,
            "contribution_dispersion_representation_dependent": True,
            "fit_calls": 0,
            "models_loaded": 1,
            "alternative_models_evaluated": 0,
            "thresholds_scored": 0,
            "alternative_thresholds_scored": 0,
            "calibrations_performed": 0,
            "hyperparameter_searches": 0,
            "feature_selections": 0,
            "features_added": 0,
            "features_removed": 0,
            "new_performance_estimates_produced": 0,
            "uncertainty_intervals_computed": 0,
            "new_dependencies_added": 0,
            "holdout_loaded": False,
            "holdout_used_for_global_interpretation": False,
            "selection_after_analysis": False,
            "model_changed": False,
            "threshold_changed": False,
            "calibration_changed": False,
            "statement": (
                "The analysis decomposes the frozen predictive function exactly and describes it "
                "over the training pool. No selection, fit, calibration, tuning, threshold choice, "
                "feature change or model change preceded or followed it"
            ),
        },
        limitations=list(LIMITATIONS),
        environment={
            "python": ".".join(str(part) for part in sys.version_info[:2]),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        methodological_notes=list(METHODOLOGICAL_NOTES),
    )


def write_results(results: InterpretationResults, path: Path | None = None) -> Path:
    """Write the record as indented JSON."""
    destination = path or RESULTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("Wrote model interpretation results: %s", destination)
    return destination


def load_results(path: Path | None = None) -> InterpretationResults:
    """Read and validate the Phase 10 record.

    Raises:
        FileNotFoundError: If the interpretation has not been run yet.
    """
    source = path or RESULTS_PATH
    if not source.is_file():
        raise FileNotFoundError(
            f"Model interpretation results not found at {source}. Produce them with "
            "`uv run python scripts/run_model_interpretation.py`."
        )
    return InterpretationResults.model_validate_json(source.read_text(encoding="utf-8"))


def invariant_failures(results: InterpretationResults) -> list[str]:
    """Return every violated invariant of a loaded record. Read-only.

    What ``--verify`` checks: that the decomposition still reproduces the frozen
    pipeline, that the 46 columns still partition into the 19 raw features
    exactly once each, that the ranking still uses the pre-declared metric, and
    that the protocol flags still say nothing was fitted, selected or explained
    with an approximate estimator.
    """
    failures: list[str] = []

    reconstruction = results.reconstruction
    tolerance = float(reconstruction["tolerance"])
    for key in (
        "max_abs_logit_error",
        "max_abs_probability_error",
        "max_grouped_reconstruction_error",
    ):
        if float(reconstruction[key]) > tolerance:
            failures.append(f"reconstruction.{key} exceeds the recorded tolerance")
    if not reconstruction["within_tolerance"]:
        failures.append("the reconstruction is not marked as within tolerance")

    model = results.model
    n_transformed = int(model["n_transformed_features"])
    if int(model["n_coefficients"]) != n_transformed:
        failures.append("the number of coefficients does not match the transformed feature count")
    if int(model["n_numeric_features"]) + int(model["n_categorical_features"]) != int(
        model["n_raw_features"]
    ):
        failures.append("numeric plus categorical does not equal the raw feature count")
    if model["engineered_features"]:
        failures.append("the record lists engineered features, which the freeze declares empty")
    if int(model["positive_class_column"]) != 1 or int(model["positive_class_label"]) != 1:
        failures.append("the positive class is not the one the frozen policy declares")

    mapping = results.feature_mapping
    if int(mapping["n_transformed_columns"]) != n_transformed:
        failures.append("the mapping covers a different number of transformed columns")
    if int(mapping["n_raw_features"]) != int(model["n_raw_features"]):
        failures.append("the mapping covers a different number of raw features")
    if mapping["textual_parsing_used"]:
        failures.append("the mapping claims to have been derived by textual parsing")

    assigned: list[int] = []
    for group in mapping["groups"]:
        columns = [int(column) for column in group["transformed_columns"]]
        assigned.extend(columns)
        if len(columns) != int(group["n_transformed_columns"]):
            failures.append(f"group {group['feature']} disagrees with its own column count")
        if group["kind"] == "categorical" and len(group["levels"]) != len(columns):
            failures.append(f"group {group['feature']} has a level/column mismatch")
        if group["kind"] == "numeric" and len(columns) != 1:
            failures.append(f"numeric group {group['feature']} does not own exactly one column")
    if sorted(assigned) != list(range(n_transformed)):
        failures.append("the raw feature groups are not a partition of the transformed columns")

    if len(results.numeric_terms) != int(model["n_numeric_features"]):
        failures.append("the numeric term count disagrees with the model record")
    if len(results.categorical_terms) != int(model["n_categorical_features"]):
        failures.append("the categorical term count disagrees with the model record")

    for term in results.numeric_terms:
        scale = float(term["scaler_scale"])
        if scale <= 0:
            failures.append(f"numeric term {term['feature']} has a non-positive scaler scale")
            continue
        expected = float(term["standardized_coefficient"]) / scale
        if abs(expected - float(term["raw_logit_coefficient"])) > 1e-9:
            failures.append(f"numeric term {term['feature']} has an inconsistent raw coefficient")
        if (
            abs(
                np.exp(float(term["standardized_coefficient"])) - float(term["odds_ratio_per_1_sd"])
            )
            > 1e-9
        ):
            failures.append(f"numeric term {term['feature']} has an inconsistent odds ratio")

    for term in results.categorical_terms:
        coefficients = [float(value) for value in term["coefficient_per_level"].values()]
        if len(coefficients) != int(term["n_levels"]):
            failures.append(f"categorical term {term['feature']} has a level count mismatch")
            continue
        spread = max(coefficients) - min(coefficients)
        if abs(spread - float(term["spread"])) > 1e-9:
            failures.append(f"categorical term {term['feature']} has an inconsistent spread")
        if abs(np.exp(spread) - float(term["max_pairwise_modeled_odds_ratio"])) > 1e-9:
            failures.append(f"categorical term {term['feature']} has an inconsistent odds ratio")
        contrast = term["strongest_pairwise_contrast"]
        if abs(float(contrast["delta_log_odds"]) - spread) > 1e-9:
            failures.append(f"categorical term {term['feature']} has an inconsistent contrast")
        if term["reference_level_removed"] or term["reference_level"] is not None:
            failures.append(f"categorical term {term['feature']} declares a reference level")

    structural = results.structural_dependencies
    if structural["population"] != "training pool":
        failures.append("the structural dependencies were not checked on the training pool")
    if structural["holdout_used"]:
        failures.append("the structural check claims to have used the holdout")
    if int(structural["n_rows_checked"]) != int(results.interpretation_population["n_rows"]):
        failures.append("the structural check covered a different number of rows")
    if structural["contrasts_removed_or_altered"] or structural["coefficients_removed_or_altered"]:
        failures.append("the record claims a contrast or a coefficient was removed or altered")

    declared_coupled: set[tuple[str, str]] = set()
    for rule in structural["rules"]:
        violations = int(rule["violations"])
        if violations != int(rule["left_only"]) + int(rule["right_only"]):
            failures.append(f"rule {rule['relationship']} miscounts its own violations")
        # The load-bearing invariant: determinism is a *derived* claim, and a rule
        # with even one violation may never assert it.
        if bool(rule["deterministic_in_training_pool"]) != (violations == 0):
            failures.append(
                f"rule {rule['relationship']} reports determinism inconsistent with "
                f"{violations} violation(s)"
            )
        if int(rule["n_rows_checked"]) != int(structural["n_rows_checked"]):
            failures.append(f"rule {rule['relationship']} checked a different number of rows")
        if rule["deterministic_in_training_pool"]:
            if int(rule["n_left"]) != int(rule["n_right"]) != int(rule["n_both"]):
                failures.append(
                    f"rule {rule['relationship']} is deterministic but its sides differ"
                )
            declared_coupled.add((rule["left"]["feature"], rule["left"]["level"]))
            declared_coupled.add((rule["right"]["feature"], rule["right"]["level"]))

    recorded_coupled = {
        (feature, level)
        for feature, levels in structural["coupled_levels"].items()
        for level in levels
    }
    if recorded_coupled != declared_coupled:
        failures.append("coupled_levels does not match the deterministic rules")

    for block in structural["coupled_column_blocks"]:
        coefficients = [float(value) for value in block["coefficient_per_column"]]
        if len(coefficients) != int(block["n_columns"]):
            failures.append("a coupled block miscounts its columns")
            continue
        if not block["spans_multiple_raw_features"]:
            failures.append("a coupled block does not span multiple raw features")
        if abs(sum(coefficients) - float(block["aggregate_coefficient"])) > 1e-9:
            failures.append("a coupled block's aggregate is not the sum of its coefficients")
        if (
            abs(
                (max(coefficients) - min(coefficients))
                - float(block["max_pairwise_coefficient_difference"])
            )
            > 1e-9
        ):
            failures.append("a coupled block misreports its coefficient spread")

    for term in results.categorical_terms:
        flagged = set(term["structurally_coupled_levels"])
        expected = {
            level for level in term["levels"] if (term["feature"], level) in declared_coupled
        }
        if flagged != expected:
            failures.append(f"categorical term {term['feature']} flags the wrong coupled levels")
        contrast = term["strongest_pairwise_contrast"]
        should_flag = bool(flagged & {contrast["level_a"], contrast["level_b"]})
        if bool(contrast["structurally_coupled"]) != should_flag:
            failures.append(f"the strongest contrast of {term['feature']} is misflagged")
        if bool(contrast["raw_single_feature_counterfactual_supported"]) is should_flag:
            failures.append(
                f"the counterfactual flag of {term['feature']} contradicts its coupling flag"
            )

    dispersion = results.contribution_dispersion
    if not dispersion["representation_dependent"]:
        failures.append("the ranking does not declare itself representation-dependent")
    if dispersion["ranks_may_be_summed"]:
        failures.append("the record permits summing ranks")
    if dispersion["coupled_features_are_independent_signals"]:
        failures.append("the record calls coupled features independent signals")
    if sorted(dispersion["features_with_a_structurally_coupled_level"]) != sorted(
        structural["coupled_levels"]
    ):
        failures.append("the ranking and the structural section disagree on which features couple")
    if dispersion["ranking_metric"] != RANKING_METRIC:
        failures.append("the ranking does not use the pre-declared metric")
    if not dispersion["ranking_metric_fixed_before_computing_values"]:
        failures.append("the record does not assert the metric was fixed in advance")
    if dispersion["population"] != "training pool":
        failures.append("the contribution dispersion was not computed on the training pool")
    if len(dispersion["by_feature"]) != int(model["n_raw_features"]):
        failures.append("the dispersion table does not cover every raw feature")
    ranking = dispersion["ranking"]
    if len(ranking) != int(model["n_raw_features"]):
        failures.append("the ranking does not cover every raw feature")
    if [int(row["rank"]) for row in ranking] != list(range(1, len(ranking) + 1)):
        failures.append("the ranks are not a dense 1..n sequence")
    values = [float(row["ranking_value"]) for row in ranking]
    if any(later > earlier + 1e-12 for earlier, later in zip(values[:-1], values[1:], strict=True)):
        failures.append("the ranking is not sorted by the declared metric")
    if {row["feature"] for row in ranking} != {row["feature"] for row in dispersion["by_feature"]}:
        failures.append("the ranking and the dispersion table cover different features")

    decision = results.frozen_decision
    threshold = float(decision["threshold_probability"])
    expected_logit = float(np.log(threshold / (1.0 - threshold)))
    if abs(expected_logit - float(decision["threshold_logit"])) > 1e-9:
        failures.append("the threshold logit is not log(p / (1 - p)) of the frozen threshold")
    if decision["comparison"] != ">=":
        failures.append("the decision comparison is not the frozen one")
    if decision["threshold_changed_here"] or decision["calibration_changed_here"]:
        failures.append("the record claims the threshold or the calibration changed")
    if int(decision["alternative_thresholds_scored"]) != 0:
        failures.append("an alternative threshold was scored")

    provenance = results.provenance
    if not provenance["model_fingerprint_verified"]:
        failures.append("the loaded model fingerprint did not match the decision policy")
    if not provenance["pipeline_unchanged_by_analysis"]:
        failures.append("the pipeline file changed during the interpretation")
    if provenance["refitted_here"] or provenance["reopened_here"]:
        failures.append("the record claims a refit or a reopened decision")
    for key, expected_commit in (
        ("freeze_commit", FREEZE_COMMIT),
        ("evaluation_commit", EVALUATION_COMMIT),
        ("error_analysis_commit", ERROR_ANALYSIS_COMMIT),
    ):
        if provenance[key] != expected_commit:
            failures.append(f"provenance.{key} is not the expected commit")

    methodology = results.methodology
    for flag in (
        "post_hoc_holdout_selection",
        "shap_used",
        "lime_used",
        "eli5_used",
        "permutation_importance_used",
        "pdp_used",
        "ice_used",
        "identifiers_persisted",
        "individual_explanations_persisted",
        "causal_claims",
        "naive_abs_coefficient_ranking_produced",
        "reference_category_declared",
        "holdout_loaded",
        "holdout_used_for_global_interpretation",
        "selection_after_analysis",
        "model_changed",
        "threshold_changed",
        "calibration_changed",
        "structural_dependencies_assumed_without_checking",
        "coupled_contrasts_removed",
        "repeated_coefficients_read_as_independent_evidence",
    ):
        if methodology[flag] is not False:
            failures.append(f"methodology.{flag} is not False")
    for flag in (
        "structural_dependencies_verified_empirically",
        "coupled_contrasts_flagged",
        "contribution_dispersion_representation_dependent",
    ):
        if methodology[flag] is not True:
            failures.append(f"methodology.{flag} is not True")
    for counter in (
        "fit_calls",
        "alternative_models_evaluated",
        "thresholds_scored",
        "alternative_thresholds_scored",
        "calibrations_performed",
        "hyperparameter_searches",
        "feature_selections",
        "features_added",
        "features_removed",
        "new_performance_estimates_produced",
        "uncertainty_intervals_computed",
        "new_dependencies_added",
    ):
        if methodology[counter] != 0:
            failures.append(f"methodology.{counter} is not zero")
    if methodology["models_loaded"] != 1:
        failures.append("methodology.models_loaded is not 1")

    if results.selection_allowed or results.model_change_allowed:
        failures.append("the record permits selection or a model change")
    if results.fit_calls != 0:
        failures.append("the record claims a fit occurred")
    if results.holdout_used_for_global_interpretation:
        failures.append("the record claims the holdout drove the global interpretation")
    if results.interpretation_population["partition"] != "training pool":
        failures.append("the interpretation population is not the training pool")
    if results.analysis_type != ANALYSIS_TYPE:
        failures.append("the analysis type is not the declared one")
    if results.exact_functional_form["approximation_used"] is not False:
        failures.append("the record claims an approximation was used")
    if results.exact_functional_form["explainer_library_used"] is not None:
        failures.append("the record names an explainer library")

    return failures
