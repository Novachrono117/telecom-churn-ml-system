# Methodology

The experimental protocol, in the order it ran. Every stage below produced a report and,
where it produced an estimate, a machine-readable record.

![Experimental protocol: every decision, from the baselines to the frozen threshold, is made on the
5,634-row training pool while the 1,409-row holdout stays locked; after the freeze the holdout is
evaluated once and leads only to a post-hoc error analysis, while the global interpretation
branches off the freeze over the training distribution without loading the holdout, and no arrow
returns to the decisions.](diagrams/experimental-protocol.svg)

The rule the diagram encodes: **every decision is made on the 5,634-row training pool**,
under paired 5-fold cross-validation with fixed folds and seed 42; the 1,409-row holdout is
opened once, after the freeze, and no arrow returns from it to a decision.

## Analyst exposure

One limitation qualifies the whole protocol, and it belongs next to the diagram rather
than in a footnote.

> The exploratory analysis ran **before** the holdout protocol was in force and covered all
> 7,043 rows, including the rows that later became the holdout.

So: the holdout was protected from fitting, selection, feature decisions, calibration and
tuning after the split — but it was **not literally never seen by the analyst**. The final
estimate may therefore carry an optimistic bias that cannot be quantified from inside this
study. It is recorded in the artifact rather than omitted; see
[`holdout_results.json`](../reports/experiments/holdout_results.json) and
[error_analysis_report.md](../reports/error_analysis_report.md).

## The chain

| Stage | What it did | Report | Record |
| --- | --- | --- | --- |
| **Data understanding** | Verified the target, the 7,043 rows, the identifier, duplicates and the 11 whitespace-only `TotalCharges` cells; screened column names for leakage. | [data_understanding.md](../reports/data_understanding.md) · [data_dictionary.md](../reports/data_dictionary.md) | — |
| **EDA** | Answered questions about tenure, contract, services, billing and the `TotalCharges` anomaly, and stated the hypotheses later stages had to test. | [eda_report.md](../reports/eda_report.md) | — |
| **Split** | Stratified 80/20 split as the *first* operation on the raw data, seed 42; the partition is regenerated, never stored. | [preprocessing_report.md](../reports/preprocessing_report.md) | [split_manifest.json](../reports/split_manifest.json) |
| **Preprocessing** | Declared the 19-feature contract and packaged cleaning, scaling and one-hot encoding as a pipeline refitted inside every fold. | [preprocessing_report.md](../reports/preprocessing_report.md) | — |
| **Baselines** | Fixed the references every later stage had to beat: majority, stratified-random and an untuned logistic regression. | [baseline_report.md](../reports/baseline_report.md) | [baseline_results.json](../reports/experiments/baseline_results.json) |
| **Feature experiments** | Five hypothesis-driven candidates, one variable each, paired ablation against the baseline. | [feature_engineering_report.md](../reports/feature_engineering_report.md) | [feature_engineering_results.json](../reports/experiments/feature_engineering_results.json) |
| **Model comparison** | Logistic regression, random forest and HistGradientBoosting under identical folds and one shared representation. | [model_comparison_report.md](../reports/model_comparison_report.md) | [model_comparison_results.json](../reports/experiments/model_comparison_results.json) |
| **Representation experiment** | Isolated one factor — how the 16 categorical columns reach the estimator — so the booster's gap could not be blamed on an encoding. | [hgb_representation_report.md](../reports/hgb_representation_report.md) | [hgb_representation_results.json](../reports/experiments/hgb_representation_results.json) |
| **Nested tuning** | Hyperparameter search inside the outer folds only, for the two surviving families, under an adoption rule fixed before any tuned result existed. | [tuning_report.md](../reports/tuning_report.md) | [tuning_results.json](../reports/experiments/tuning_results.json) |
| **Calibration** | Measured probability quality — Brier score, log loss — and tested sigmoid and isotonic against a pre-registered eligibility rule. | [calibration_report.md](../reports/calibration_report.md) | [calibration_results.json](../reports/experiments/calibration_results.json) |
| **Threshold selection** | Selected an operating point on out-of-fold training probabilities, evaluated nested against the 0.5 default. | [threshold_report.md](../reports/threshold_report.md) | [threshold_results.json](../reports/experiments/threshold_results.json) |
| **Freeze** | Fitted the chosen configuration on the whole training pool and wrote it down with fingerprints. No metric was computed here. | [model_freeze_report.md](../reports/model_freeze_report.md) | [decision_policy.json](../reports/decision_policy.json) |
| **Holdout evaluation** | One frozen model, one frozen policy, one holdout, **one** evaluation — and it refused to open the holdout until the freeze verified. | [holdout_evaluation_report.md](../reports/holdout_evaluation_report.md) | [holdout_results.json](../reports/experiments/holdout_results.json) |
| **Error analysis** | Post-hoc and descriptive: where the errors the estimate already counted actually fell. Nothing found here may change the model. | [error_analysis_report.md](../reports/error_analysis_report.md) | [error_analysis_results.json](../reports/experiments/error_analysis_results.json) |
| **Interpretation** | Exact global and local decomposition of the frozen model, computed on the training pool because the holdout was spent. | [model_interpretation_report.md](../reports/model_interpretation_report.md) | [model_interpretation_results.json](../reports/experiments/model_interpretation_results.json) |

Calibration is settled **before** the threshold, deliberately: calibration rewrites the
probabilities a threshold would act on, so a threshold chosen first would belong to a score
that no longer exists. Both were decided on the training pool alone.

## Decision gates

The interesting results here are the negative ones. Each was a pre-registered rule that
additional complexity failed to clear, not a preference:

| Gate | Outcome | Evidence |
| --- | --- | --- |
| **Engineered features** | **None adopted.** Four of five candidates failed the paired ablation; the fifth moved Average Precision by an immaterial amount. | [feature_engineering_report.md](../reports/feature_engineering_report.md) |
| **Tuned HistGradientBoosting** | **Not eligible to replace the baseline.** It had a *positive* mean paired ΔAP and still improved only 3 of the 5 outer folds, below the rule's 4-of-5 requirement. | [tuning_report.md](../reports/tuning_report.md) |
| **Tuned logistic regression** | **Not eligible.** Negative mean paired ΔAP; the untuned baseline stood. | [tuning_report.md](../reports/tuning_report.md) |
| **Calibration** | **`NONE`.** Neither sigmoid nor isotonic improved the Brier score consistently across the outer folds. | [calibration_report.md](../reports/calibration_report.md) |
| **Default threshold 0.5** | **Not retained.** The nested F1-maximisation policy beat it in 5 of 5 outer folds; the trade it makes is stated in [decision-policy.md](decision-policy.md). | [threshold_report.md](../reports/threshold_report.md) |

Four of these five gates closed against the more complex option, and the fifth replaced a
default with an argued choice. That is what writing the rule down first buys: a rule fixed
before the numbers exist is the only thing that can turn "it did not win" into a decision
rather than a disappointment.
