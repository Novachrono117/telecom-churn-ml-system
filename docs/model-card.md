# Model Card

## Model summary

| | |
| --- | --- |
| Task | Supervised binary classification — telecom customer churn |
| Estimator | `LogisticRegression` (`C=1.0`, `solver=lbfgs`, `class_weight=None`, `max_iter=100`) |
| Positive class | `Churn = Yes`, encoded `1` |
| Primary output | Churn probability |
| Secondary output | A class, under an explicit frozen decision rule |
| Input | 19 raw features (3 numeric, 16 categorical) → 46 transformed columns |
| Calibration policy | `NONE` |
| Semantic identity | `model_fingerprint_sha256 = a57568e1…` |

An ML model, not an AI system: one linear estimator behind a fixed preprocessing pipeline.
Source of truth: [`decision_policy.json`](../reports/decision_policy.json).

## Intended use

A **benchmark and engineering demonstration** of churn-risk ranking and the decision
infrastructure around it: how a score is produced, how it becomes a decision, how that
decision is explained, served and observed. It exists to be audited and reproduced.

## Out-of-scope use

- Causal explanation of churn — nothing here identifies a cause.
- Automatic retention action, or any decision taken without a human.
- Recommending a treatment for a customer; targeting and persuadability are different
  problems, and nothing here measures uplift.
- Monitoring performance without labels — see [monitoring.md](monitoring.md).
- Populations this dataset does not cover: another operator, market or period.
- Any high-stakes automated decision about a person.

## Data

One public dataset — **Telco Customer Churn**, 7,043 rows, 21 columns,
`raw_sha256 = 88be4b93…`. The raw file is versioned unchanged and its digest is checked
before the split runs. Stratified 80/20 split, seed 42: 5,634 training rows, 1,409 holdout.
See [methodology.md](methodology.md) and
[`split_manifest.json`](../reports/split_manifest.json).

## Features

The 19 contracted features are the original columns, minus the identifier and the target.
**No engineered feature is in the contract** — five candidates were tested and none cleared
the ablation gate. Numeric features are standardised; categorical features are one-hot
encoded with `handle_unknown="ignore"`, so an unseen category is absorbed as all-zeros
rather than rejected. `TotalCharges` blanks are cleaned by a pipeline step, not by hand.

## Decision policy

```text
positive = probability >= 0.3272694566222328
```

`F1_MAXIMIZATION`, selected on the training pool alone, comparison `>=`, threshold stored
unrounded. Details and trade-offs: [decision-policy.md](decision-policy.md).

## Evaluation

One frozen configuration, evaluated **once** on the 1,409-row holdout (374 positives,
prevalence 0.2654). Values read from
[`portfolio_metadata.json`](../reports/portfolio/portfolio_metadata.json), which is derived
from [`holdout_results.json`](../reports/experiments/holdout_results.json); CIs are
percentile bootstrap, 2,000 replications.

| Metric | Value | 95% CI |
| --- | --- | --- |
| Average Precision | 0.6337 | [0.5777, 0.6864] |
| ROC-AUC | 0.8420 | [0.8183, 0.8633] |
| Recall | 0.7219 | [0.6760, 0.7652] |
| Precision | 0.5378 | [0.4937, 0.5791] |
| F1 | 0.6164 | [0.5762, 0.6534] |
| *Accuracy (auxiliary)* | *0.7615* | — |

Anchors: a no-skill ranker scores AP equal to the prevalence (0.2654) and ROC-AUC 0.500;
always predicting "no churn" scores accuracy 0.7346 while flagging nobody. Accuracy is
reported for completeness and was never a criterion. Confusion matrix at the frozen
threshold: 803 / 232 / 104 / 270 (TN / FP / FN / TP).

## Explainability

Exact, not approximate: `logit(P) = intercept + Σ contribution(feature)`, verified over all
5,634 training rows at a maximum absolute probability error of 1.11e-16 against a 1e-12
tolerance. Global coefficients and local per-customer decomposition come from the same
identity. See [explainability.md](explainability.md).

## Monitoring

Data quality, feature drift and prediction drift are computed per window without labels;
**performance degradation is never inferred**. See [monitoring.md](monitoring.md).

## Limitations

- **Analyst exposure.** Exploratory analysis preceded the holdout protocol and covered all
  7,043 rows, so the final estimate may carry an unquantifiable optimistic bias.
- **A single public dataset.** One operator, one market, one vintage; nothing establishes
  that the model transfers.
- **No temporal validation.** The split is random and stratified, not by time, so it never
  simulated the drift a real deployment faces.
- **No business costs or capacity.** None were available, so the threshold optimises F1
  rather than an economic objective, and no cost figure is assumed anywhere.
- **Calibration policy is `NONE`.** Treat the output as a score and a ranking position, not
  a literal likelihood.
- **Associations, not causes.** No causal identification strategy is used anywhere.
- **No intervention or uplift evidence.** Nothing shows that acting on a prediction changes
  an outcome.
- **Label-free monitoring cannot measure performance degradation**, and no drift signal is
  offered as a proxy for one.
- **Monitoring is local and in-memory**, with no persistence, rotation or alert routing.
- **No production deployment.** The architecture is production-oriented; it has never
  served real traffic.

## Reproducibility

Python 3.12, `uv`, explicit seeds, relative paths, pinned dependencies. Every report,
figure and record is regenerated by code and checked by one of 11 `--verify` gates. See
[reproducibility.md](reproducibility.md).

## Provenance

Three digests, checked by 34 startup gates before the service will answer a request:
`model_fingerprint_sha256` (semantic identity, authoritative), `pipeline_sha256`
(serialised bytes) and the whole-file source digests of the code that runs around the
model. On any mismatch the process refuses to serve. See
[architecture.md](architecture.md#provenance-and-integrity) and
[model_freeze_report.md](../reports/model_freeze_report.md).
