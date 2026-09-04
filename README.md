# Customer Churn — Production-Oriented ML System

End-to-end ML engineering system for telecom customer churn: a frozen, fingerprinted
model behind an explicit decision policy, exact per-customer explanations, an
integrity-gated inference API and label-free drift monitoring.

![Python](https://img.shields.io/badge/python-3.12-3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E)
![License](https://img.shields.io/badge/license-MIT-green)

![Churn Risk Console running against the frozen model: the API ready, the model artifact verified
and monitoring healthy, the 19-feature request form, and a model card carrying the frozen threshold
next to the held-out metrics.](docs/screenshots/churn-risk-console.png)

## What this is

- **Problem.** Rank telecom customers by churn risk, and turn that score into a
  retention decision a team could operate.
- **Built.** A reproducible path from raw data to a served, monitored artifact:
  leakage-safe splitting, model comparison, a decision policy, exact explanations, an
  HTTP boundary, drift monitoring.
- **Different.** The model is a closed artifact, not a notebook cell. Every number comes
  from a run its own code executed, and every adoption decision was made against a rule
  written down before the results were seen.

**Looking for something specific?** The [documentation map](docs/README.md) answers
"where is the evidence for X, and which command proves it" without opening every report.

## Results

One frozen configuration, evaluated **once** on a holdout of **1,409 customers**
(374 churners, prevalence 0.2654) that took no part in any decision.

| Metric | Value | 95% CI (percentile bootstrap, 2,000 replications) |
| --- | --- | --- |
| **Average Precision** | **0.6337** | [0.5777, 0.6864] |
| **ROC-AUC** | **0.8420** | [0.8183, 0.8633] |
| Recall | 0.7219 | [0.6760, 0.7652] |
| Precision | 0.5378 | [0.4937, 0.5791] |
| F1 | 0.6164 | [0.5762, 0.6534] |
| *Accuracy (auxiliary)* | *0.7615* | — |

**Anchors.** A no-skill ranker scores Average Precision equal to the prevalence
(0.2654) and ROC-AUC 0.500. Always predicting "no churn" scores accuracy 0.7346 — 2.7
points below this model — while flagging nobody. That is why accuracy is reported for
completeness and never used as a criterion.

| | Predicted: retained | Predicted: churn |
| --- | --- | --- |
| **Actually retained** (1,035) | 803 | 232 |
| **Actually churned** (374) | 104 | 270 |

The rule flags 35.6% of customers and catches 72.2% of the churners: 232 false positives
are retention contacts spent on customers who would have stayed, 104 false negatives are
churners never flagged. Held-out Average Precision sits 0.0278 below the cross-validated
development estimate and ROC-AUC 0.0041 below it — a generalization check that gated
nothing.

> **Analyst exposure.** The holdout was never used for fitting, selection or tuning, but
> exploratory analysis earlier in the project had already exposed the analyst to all
> 7,043 rows. The estimate may therefore carry an optimistic bias that cannot be
> quantified from inside the study; it is recorded in the artifact rather than omitted.

The protocol behind that single number — which partition was allowed to influence what:

![Experimental protocol: every decision, from the baselines to the frozen threshold, is made on the
5,634-row training pool while the 1,409-row holdout stays locked; after the freeze the holdout is
evaluated once and leads only to a post-hoc error analysis, while the global interpretation
branches off the freeze over the training distribution without loading the holdout, and no arrow
returns to the decisions.](docs/diagrams/experimental-protocol.svg)

## System at a glance

![System architecture: API consumers post to the prediction endpoints and the Portfolio UI sends
exactly one explanation request per submission, both through the same canonical scoring call; the
request crosses the feature contract, then the frozen pipeline, which emits one probability feeding
the frozen decision policy and the exact log-odds decomposition, while the monitoring observer sits
off the causal path behind a read-only endpoint.](docs/diagrams/system-architecture.svg)

The three boundaries are the point: what the contract guarantees before the model sees anything,
what the freeze closes, and what the serving process may add around it without touching either.

## Why Logistic Regression?

Not because it topped a leaderboard — because nothing else earned the right to replace
it. Families compared under identical 5-fold cross-validation on the training pool:

| Family | Mean Average Precision |
| --- | --- |
| **Logistic regression** | **0.6615** |
| HistGradientBoosting | 0.6471 |
| Random forest | 0.6062 |

![Paired per-fold Average Precision deltas against the logistic baseline: every one of the five
folds is negative for random forest and for histogram gradient boosting, so neither family beat the
logistic regression on a single fold.](reports/figures/model_comparison/02_paired_ap_deltas.png)

Random forest was clearly worse. HistGradientBoosting was competitive, so it went to
nested hyperparameter tuning alongside the logistic regression — under an adoption
rule fixed **before** any tuned result existed:

> A tuned procedure replaces the frozen baseline only if its mean paired ΔAP is greater
> than zero **and** AP improves in at least 4 of the 5 outer folds. A tie counts as
> neither.

| Procedure | Mean paired ΔAP | Outer folds improved | Verdict |
| --- | --- | --- | --- |
| Tuned logistic regression | −0.000195 | 2 / 5 (1 tie, 2 worse) | not eligible |
| Tuned HistGradientBoosting | **+0.003842** | 3 / 5 (0 ties, 2 worse) | not eligible |

The tuned booster had a *positive* mean advantage and still was not adopted: three of
five folds is insufficient evidence under the rule, not a win. Tuning had to justify
replacing the baseline; when it did not, the simpler model remained the default.

Two further negative results held. **No engineered feature was adopted**: five
hypothesis-driven candidates went through paired ablations, four failed, and the fifth
moved logistic regression by an immaterial +0.0017 AP. **No calibration was adopted**:
neither sigmoid nor isotonic improved the Brier score consistently across the outer
folds, so the policy is `NONE` — a gate outcome, not an omission.

Deep dive: [model comparison](reports/model_comparison_report.md) ·
[tuning](reports/tuning_report.md) · [features](reports/feature_engineering_report.md)

## Probability ≠ Decision

The model emits a probability. Turning it into an action is a separate, explicit
choice — and 0.5 is a default, not a policy. The threshold was selected on the training
pool alone under an `F1_MAXIMIZATION` policy, evaluated nested so no row helped choose
its own threshold. Against the 0.5 default, across the same 5 outer folds:

| | Default 0.5 | Selected policy | Paired Δ (folds in favour) |
| --- | --- | --- | --- |
| Recall | 0.5438 | **0.7672** | **+0.2234** (5 / 5) |
| Precision | 0.6521 | 0.5396 | −0.1125 (0 / 5) |
| F1 | 0.5924 | **0.6333** | **+0.0409** (5 / 5) |
| Predicted-positive rate | 0.2213 | 0.3777 | +0.1564 |

The trade is deliberate and one-directional: substantially more churners caught, at a
materially lower precision and a predicted-positive rate that grows by roughly half.

**Frozen decision rule** — `positive = probability >= 0.3272694566222328`. The
comparison is `>=`, not `>`: a customer whose probability equals the threshold exactly
is predicted positive. The value is stored unrounded because a threshold rounded in the
sixth decimal moves customers across the boundary.

This is **not** a business-optimal threshold. No retention cost, contact capacity or
customer value was available, so none was assumed. Choosing an operating point against
real costs remains open work.

![Out-of-fold F1 across the whole range of decision thresholds, peaking at 0.6381 on the frozen
threshold 0.3273 and falling away well before the 0.500 default, which is marked separately.](reports/figures/threshold/02_f1_vs_threshold.png)

A real customer scored through the live API, on the band where the default and the policy disagree:

![Result panel from the running console: a churn score of 34.0%, full precision
0.34000536953235566, decision "churn", against the frozen threshold 0.3273 under the rule
score &gt;= 0.3273.](docs/screenshots/decision-threshold.png)

At 34.0% the 0.5 default would have called this customer retained and never contacted them; the
frozen policy calls them churn. An illustrative public training-pool record chosen to show where
the boundary sits — not a typical customer, not evidence of performance, never from the holdout.

Deep dive: [calibration gate](reports/calibration_report.md) ·
[threshold policy](reports/threshold_report.md) · [frozen policy](reports/decision_policy.json)

## Explainability

The explanation is an algebraic identity, not an approximation. No SHAP, no LIME, no
sampling, no seed — for a linear model in log-odds none of those are needed:

```text
logit(P) = intercept + sum of contribution(feature)
P        = sigmoid(logit)
```

Each of the 19 raw features contributes one number in log-odds; the contributions and
the intercept reconstruct the served score exactly. Verified over all 5,634 training
rows: maximum absolute logit error **0.0** and maximum absolute probability error
**1.1e-16**, against a tolerance of 1e-12. The endpoint fails closed — if the
decomposition ever stopped reproducing the served probability it returns
`EXPLANATION_UNAVAILABLE` instead of an explanation the prediction would not support.

**Structural coupling is handled explicitly.** Seven product equivalences hold with zero
violations across the training pool: `InternetService=No` forces `No internet service`
in six add-on features, and `PhoneService=No` forces `No phone service` in
`MultipleLines`. Those columns are not independent evidence — they are one customer
property re-encoded up to seven times, which a naive per-column ranking would count
that many times — so the interpretation groups them instead. And they are associations:
no coefficient here licenses a claim about what *would* happen if a contract changed.

The same customer, explained by the live endpoint:

![Contributing factors for the scored customer: Contract Month-to-month +0.5860 and MonthlyCharges
+0.3138 pushing the score up against InternetService DSL −0.6436 pushing it down, the two phone
columns grouped as one property, and a reconstruction closing to a maximum error of
1.11e-16.](docs/screenshots/local-explanation.png)

Deep dive: [model interpretation](reports/model_interpretation_report.md)

## Monitoring without labels

Four questions that are routinely collapsed into one, kept separate here:

| Signal | Answers | Available without labels |
| --- | --- | --- |
| Data quality | is the input well-formed? | yes |
| Feature drift | does the input population still look like the reference? | yes |
| Prediction drift | is the system deciding differently? | yes |
| Performance degradation | is the model still right? | **no** |

Production ground truth does not exist here, so **performance degradation is never
inferred**: no accuracy, recall or precision is reported for live traffic, and no drift
signal is presented as a proxy for one.

Computed per window: PSI for numeric features and for the model score, TVD over known
categorical levels plus an `__UNSEEN__` bucket, unseen-category rates, the seven
structural equivalences, and the shift in predicted-positive rate. Windows below 100
records report `INSUFFICIENT_DATA` rather than a number, and aggregation is
privacy-aware: raw category values are never retained, only bounded digests, never
exposed. PSI and TVD are descriptive distances, not tests — crossing a cutoff means
"this window no longer resembles the reference", never "the model degraded".

![Monitoring panel of the running console: status WARNING, 101 window records against a minimum
window size of 100, data drift WARNING, prediction drift WARNING, structural consistency
OK.](docs/screenshots/monitoring-status.png)

**Demonstration window.** The first 101 scored observations of the frozen training pool, taken
deterministically in frozen order — the reference population itself. It reports `WARNING`: PSI
0.155 on `tenure` and 0.152 on the model score against a heuristic 0.10 cutoff, while the mean of
`tenure` moved −0.10 reference standard deviations and the predicted-positive rate moved −0.015 and
stayed `OK`. What this demonstrates is that the operational cutoff can fire on a small window drawn
from the reference population. It is **not** evidence of production drift, and **not** evidence of
performance degradation — which nothing here can measure, because there are no labels. The window
is published as captured rather than replaced with a greener one: the cutoffs are heuristics fixed
before any traffic existed, and this is exactly the kind of window that says they need re-tuning
once real ones exist.

Deep dive: [monitoring](reports/monitoring_report.md) ·
[policy and cutoffs](configs/monitoring.toml)

## Frozen artifact & provenance

Two fingerprints answer different questions. **`model_fingerprint_sha256`** is the
*semantic* identity of the learned state, hashed under a canonical encoding that
round-trips every double exactly and invariant to how the object was serialised — this
one is authoritative. **`pipeline_sha256`** is the identity of the *serialised bytes*:
it catches a corrupted or swapped file, but is a property of the file, not the model.

Neither is sufficient alone, because code running *outside* the pickle can still change
a prediction. `prepare_features` is the clearest case: it validates and orders the input
**before** the persisted pipeline sees it, so editing it would change what the model
receives while both fingerprints, the feature names, the threshold and the
hyperparameters stayed identical. Its source is therefore hashed separately, alongside
the eight source files and three configuration files that materially determine the
inference function. Hashing is whole-file and fail-closed: an unrelated edit invalidates
the freeze, deliberately — a freeze that survives an edit it should have caught is worse
than one that fails on an edit it need not have caught.

**34 startup integrity gates** enforce this at boot, checking the artifact digest, the
model fingerprint, the decision-policy digest and the exact pinned versions of
scikit-learn, numpy, pandas and joblib; on any mismatch the service refuses to serve.
The pipeline is also *regenerable*: rebuilding it from the frozen split, builder and
training pool must reproduce both fingerprints — a stronger guarantee than the committed
binary alone.

Deep dive: [freeze report](reports/model_freeze_report.md) ·
[serving](reports/serving_report.md)

## Interactive demo

**Churn Risk Console** — score a customer, see the probability against the frozen
threshold, inspect the exact per-feature contributions behind it. Vanilla HTML, modern
CSS and ES6 with inline SVG over the FastAPI backend: no frontend framework, no build
step, no CDN, no web fonts. It runs offline, is served by the API process itself, and is
off by default behind an environment flag. It is pictured at the top of this page, and the result
and explanation panels above are screenshots of it running against the frozen artifact.

## Running locally

Requires [uv](https://docs.astral.sh/uv/); Python 3.12 is installed automatically and
pinned in `.python-version`. The serving dependencies live in their own project because
the root `pyproject.toml` and `uv.lock` are part of the frozen provenance — a web
framework must not be able to invalidate a model freeze.

```bash
uv sync                      # root: data, modeling, analysis, tests
uv sync --project serving    # serving: FastAPI + the pinned ML runtime

# API on http://127.0.0.1:8000 — add either flag, or both, to enable more
uv run --project serving python -m churn.serving
CHURN_SERVING_PORTFOLIO_UI=1 uv run --project serving python -m churn.serving  # + /demo
CHURN_SERVING_MONITORING=1 uv run --project serving python -m churn.serving    # + drift
CHURN_SERVING_PORTFOLIO_UI=1 CHURN_SERVING_MONITORING=1 \
  uv run --project serving python -m churn.serving
```

PowerShell has no inline `VAR=value command` form — set the variables first:

```powershell
$env:CHURN_SERVING_PORTFOLIO_UI = "1"
$env:CHURN_SERVING_MONITORING = "1"
uv run --project serving python -m churn.serving
```

Endpoints: `/health/live`, `/health/ready`, `/api/v1/model`, `/api/v1/predict`,
`/api/v1/predict/batch`, `/api/v1/explain`, `/api/v1/portfolio`, `/openapi.json`, plus
`/api/v1/monitoring` and `/demo` when enabled. `GET /health/ready` reports what was
verified — `"startup_gates_passed": 34`, the served `model_fingerprint`, and the
monitoring state — or `503` if any gate failed.

The persisted pipeline starts **after** the feature contract, so a caller that skips the
service produces the validated, canonically ordered matrix itself:

```python
import joblib
import pandas as pd

from churn.preprocessing import build_feature_matrix

pipeline = joblib.load("artifacts/model/churn_pipeline.joblib")
raw = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv").head(3)
features = build_feature_matrix(raw)
column = list(pipeline.classes_).index(1)
probability = pipeline.predict_proba(features)[:, column]
```

```bash
uv run pytest -rs                                   # 1330 tests
uv run --project serving pytest -rs serving/tests   # 391 tests
```

## Reproducing every result

Every report, figure and machine-readable record here is generated by code and
verifiable without rerunning training. Each generator has a `--verify` mode that
rebuilds its artifact in memory and compares it, writing nothing.

**11 gates, all exit 0.** The commands, and what each one actually proves, are in
[docs/reproducibility.md](docs/reproducibility.md#artifact-verification). Drop `--verify` to
regenerate. `freeze_model.py` rebuilds the pipeline from the frozen split, builder and
training pool; both fingerprints must come out unchanged.

Neither partition is stored on disk: the split is regenerated from the raw file plus
`configs/base.toml`, and `reports/split_manifest.json` records the raw SHA-256, every
split parameter and a digest over the sorted identifiers of each partition — enough to
prove the partition is the same one without keeping it.

**Independent reproduction.**
[`notebooks/02_academic_delivery.ipynb`](notebooks/02_academic_delivery.ipynb) rebuilds
the frozen protocol without cloning this repository: it downloads the dataset from a
public mirror with no credentials, canonicalises line endings, verifies the SHA-256
against the frozen value and **aborts on mismatch**, then reproduces the split,
preprocessing, baselines, model comparison, threshold and holdout evaluation, checking
every number on screen against the committed artifact. Committed without outputs;
execute to reproduce. *(Narrative in Portuguese.)*

## Project structure

```text
src/churn/      data, analysis, preprocessing, features, modeling, serving,
                monitoring, portfolio  -  all reusable logic
serving/        isolated uv project for the HTTP boundary (FastAPI) and its tests
portfolio/      Churn Risk Console: static HTML/CSS/JS demo assets
configs/        base.toml (frozen provenance) and monitoring.toml (alert policy)
scripts/        one entry point per stage; 11 of them support --verify
tests/          contract, transformation, modeling, serving and monitoring tests
reports/        generated reports, figures and machine-readable experiment records
notebooks/      exploratory analysis and the independent reproduction notebook
data/raw/       the untouched source CSV, versioned and fingerprinted
artifacts/      the frozen inference pipeline (8 KB, regenerable, fingerprinted)
academic/       historical / extended reproducibility material (Portuguese)
```

## Limitations

Stated plainly, because a result you cannot bound is not a result.

- **Analyst exposure.** Exploratory analysis preceded the holdout protocol and covered
  every row, so the estimate may carry an unquantifiable optimistic bias.
- **A single public dataset.** One telecom snapshot, one market, one vintage; nothing
  here establishes that the model transfers.
- **No temporal validation.** The split is stratified at random, not by time, so it
  never simulated the drift a real deployment faces.
- **Calibration policy is `NONE`.** Probabilities rank well but were not shown to be
  calibrated; treat them as scores, not literal likelihoods.
- **No business costs or capacity.** None were available, so the threshold optimises F1
  rather than an economic objective.
- **Associations, not causes.** No causal identification strategy is used anywhere, and
  nothing shows that acting on a prediction changes an outcome — targeting and
  persuadability are different problems.
- **Monitoring is local and in-memory**, with no persistence, rotation or alert routing.
- **No production deployment.** The architecture is production-oriented; it has never
  served real traffic.

## Technical documentation

Start at the **[documentation map](docs/README.md)**: a question, the page that answers it,
the artifact that is authoritative, and the command that verifies it.

| Page | Covers |
| --- | --- |
| [Architecture](docs/architecture.md) | Boundaries — feature contract, frozen pipeline, decision policy, HTTP layer, monitoring observer, environment isolation |
| [Methodology](docs/methodology.md) | The experimental protocol end to end, analyst exposure, and the decisions that went the other way |
| [Model card](docs/model-card.md) | Intended and out-of-scope use, evaluated performance, limitations |
| [Decision policy](docs/decision-policy.md) | Calibration, threshold selection, and the trade actually made |
| [Explainability](docs/explainability.md) | Why the per-customer explanation is an exact identity |
| [Monitoring](docs/monitoring.md) | Four signals kept separate, and what labels would be needed to say more |
| [Reproducibility](docs/reproducibility.md) | Clone, run, test, lint, verify — every documented command |

Raw reports and machine-readable records remain under [`reports/`](reports): 18 generated
reports, the experiment records in [`reports/experiments/`](reports/experiments), and 58
figures in [`reports/figures/`](reports/figures).

## Project origin

This project began as a postgraduate Machine Learning Engineering exercise and was
subsequently expanded into a production-oriented ML engineering system. The original
methodological material is preserved under `academic/` and `reports/academic/` as
extended documentation.

## License

[MIT](LICENSE) — applies to the original code and documentation in this project.
Third-party datasets retain their own respective source terms; see
[`data/README.md`](data/README.md) for dataset provenance.
