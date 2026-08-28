# Phase 13 — Portfolio Product / Interactive ML Demo

**Component:** `PORTFOLIO_PRODUCT` · **Model changed:** no · **Fit calls:** 0 ·
**Holdout loaded:** no

Built on the frozen model of Phase 9C (`9d1db49`), the serving boundary of Phase 11
(`71a7f3a`) and the monitoring layer of Phase 12 (`f1a9fec`).

---

## 1. Scope

Phase 13 makes an already-validated system **demonstrable**. It adds no modelling:
no fit, no tuning, no feature engineering, no recalibration, no threshold selection,
no new experiment, and no reopening of the holdout.

| | |
|---|---|
| Added | a local-explanation endpoint, a versioned-metadata endpoint, one HTML page and two static assets |
| Reused | the Phase 11 scoring path, the Phase 10 decomposition primitives, the Phase 12 monitoring endpoint |
| Changed in the model | nothing |
| Rewritten from earlier phases | nothing |

Everything the product displays comes from a frozen artefact or from the request the
visitor just submitted.

---

## 2. Product objective

A person who has never seen this repository should be able to, in one screen:

1. understand what problem the model solves;
2. describe a customer;
3. get the frozen model's churn score;
4. see **why** that score came out as it did;
5. see which decision threshold the score was compared against;
6. read the model's identity and its held-out evaluation;
7. see the operational monitoring state, when monitoring is on.

And nothing on that screen should overstate what the system knows.

---

## 3. Architecture

```text
portfolio/                               static assets, no build step
├── index.html
├── css/app.css
└── js/app.js

src/churn/portfolio/                     the core — no web framework, no dataset
├── __init__.py
├── coupling.py       which raw features move together (derived from Phase 12 rules)
├── explanation.py    the exact decomposition and its reconstruction gate
├── metadata.py       the versioned offline summary of Phase 9D
└── results.py        the Phase 13 machine-readable record

src/churn/serving/portfolio.py           the HTTP seam: service, routes, static mount

scripts/build_portfolio_metadata.py      build / --verify the versioned summary
scripts/build_portfolio_record.py        build / --verify the Phase 13 record

reports/portfolio/portfolio_metadata.json
reports/experiments/portfolio_results.json
reports/portfolio_product_report.md      this document
```

Layering is one-directional: `metadata` and `coupling` → `explanation` → the serving
seam. The core imports no web framework and reads no dataset, which is what lets the
whole portfolio test suite run in the root environment.

### Files modified from earlier phases

| File | Why |
|---|---|
| `src/churn/serving/settings.py` | three operational fields: UI on/off, asset path, metadata path |
| `src/churn/serving/schemas.py` | the explanation and metadata response models |
| `src/churn/serving/api.py` | opt-in route registration, the explanation error handler |
| `src/churn/serving/service.py` | `score_with_features` extracted — see §5 |
| `src/churn/serving/errors.py` | one new code, deliberately outside the Phase 11 set — see §12 |
| `serving/tests/test_settings.py` | the pinned field-set test |

`reports/experiments/serving_results.json` and `reports/experiments/monitoring_results.json`
are **unchanged** and still verify byte for byte.

---

## 4. Stack decision

**HTML5, modern CSS, vanilla ES6 JavaScript, inline SVG. No framework, no build
step, no new dependency, no external asset.**

The environment was inspected before deciding. The serving process already runs
FastAPI over the frozen artifacts; adding React or Vite would introduce a second
runtime, a second lockfile and a build artefact to keep in sync with a Python
service — for a page with one form, one result panel and four cards. Streamlit would
have replaced the API boundary this project spent Phase 11 building.

No technical reason was found to add a framework, so none was added. The value of
this portfolio is the ML engineering; the frontend's job is to not get in the way of
seeing it.

```text
new_runtime_dependencies : []
frontend_framework       : none
frontend_build_step      : no
external_runtime_assets  : no
cdn_used                 : no
web_fonts                : none — system font stack
runs_offline             : yes
```

`test_the_page_requests_nothing_from_the_internet` parses every `src` and `href` in
the page and requires each to be a local `/demo/static/` path.

---

## 5. Exact local explanation

### The identity

The model is a logistic regression, so its output is known in closed form:

```text
logit(P) = intercept + sum over the 19 raw features of contribution_f
P        = sigmoid(logit)
```

Everything the explanation panel shows is an algebraic rearrangement of that. There
is **no SHAP, no LIME, no permutation importance, no sampling and no random seed** —
those exist to probe models whose response surface is unknown, and using one here
would add a dependency, a seed and an approximation error in exchange for a worse
answer to a question already answered exactly.

### The primitives are reused, not reimplemented

Phase 10 established this decomposition. Phase 13 imports it:

```text
churn.modeling.interpretation.extract_terms
churn.modeling.interpretation.feature_groups
churn.modeling.interpretation.transform_features
churn.modeling.interpretation.group_contributions
churn.modeling.interpretation.sigmoid
churn.modeling.interpretation.logit_of
```

Two implementations of one identity are two chances to drift, so there is one.
Importing that module was checked against the runtime graph first: the serving
process already imports `churn.modeling.freeze`, which pulls in the same transitive
set, so the import adds exactly one module and no dependency. **No Phase 10 source
and no Phase 10 artefact was modified.**
`test_the_phase_10_module_is_reused_and_not_reimplemented` asserts both halves: the
primitives are imported, and the sigmoid is not spelled out a second time.

### The explanation never scores

`churn.portfolio.explanation` receives a prepared feature row **and the probability
the serving boundary already produced**, and decomposes that row. It has no second
inference path. `test_the_explanation_core_never_calls_predict_proba` arms
`predict_proba`, `predict` and `decision_function` to raise, and the explanation is
still produced.

That is the same contract Phase 12 gave the monitoring collector: results are handed
in, never derived a second time.

---

## 6. Mathematical reconstruction — the gate

Every explained request re-derives the identity and checks it against what was
served, at Phase 10's tolerance (`1e-12`):

```text
model_logit         = intercept + sum(contributions)
logit_error         = |model_logit - logit(served probability)|
probability_error   = |sigmoid(model_logit) - served probability|
```

A representative request:

```text
intercept          -0.530048
sum(contributions) +1.907477
model_logit         1.377429
logit_error         6.66e-16
probability_error   0.00e+00
tolerance           1.00e-12
```

**If the reconstruction fails, no explanation is returned.** The endpoint answers
`500 EXPLANATION_UNAVAILABLE` with a sanitised message; the prediction endpoints are
untouched and remain authoritative. A decomposition that does not reproduce the model
it claims to describe is not an explanation of that model, and showing it with a
caveat would be worse than showing nothing —
`test_a_decomposition_that_does_not_close_is_refused` perturbs the served probability
and requires the failure.

---

## 7. `predict` and `explain` return the same number

Not "agree to six decimals" — the same float.

`PortfolioService.explain` calls
`ChurnInferenceService.score_with_features(record)`, which is the same primitive
`predict_one` and `predict_batch` call. The probability, prediction, decision,
threshold, comparison and calibration policy in an explanation are **copied** from
that result, never recomputed.

`score_with_features` is Phase 11's `_score` made public and returning the canonical
matrix alongside the answer — the change is an extraction, not a new path. `_score`
now delegates to it, so there is still exactly one implementation.

That has a second consequence worth naming: an explained record is **observed by
monitoring exactly as a predicted one is**, because the observer fires inside that
one primitive. Without it, `/explain` would have incremented `records_total` while
never producing a `successful_record`.

```text
single PredictionResponse identical : True   (JSON and bytes)
batch response identical            : True
predict vs explain, per field       : True   across 6 payload variants
churn_probability, repr-for-repr    : True
canonical_scoring_strategy          : row-wise
single_batch_probability_contract   : exact
```

`test_the_explanation_goes_through_the_canonical_scoring_path` counts calls into the
shared primitive: one per predict, one per explain, and the same float out.

---

## 7b. One submission is one observation

`/api/v1/explain` already returns `churn_probability`, `prediction`, `decision`,
`threshold`, `comparison` and `calibration_policy` alongside the contributions, so
there is **no mathematical reason** to score the same form twice. The page posts
once:

```text
network calls for one analysis = [ POST /api/v1/explain, GET /api/v1/monitoring ]
model scoring calls            = 1
monitoring observations        = 1
```

The `GET` is the window-counter refresh that runs after a successful analysis. It is
read-only and scores nothing, which is asserted separately.

**No change of behaviour was needed.** The audit was run to establish the property,
not to repair it: `submit()` contained exactly one `postJSON` call, to `API.explain`,
and the string `/api/v1/predict` does not appear in `app.js` at all. What did change
is how the record describes this — see below. A double-scoring frontend would have
doubled the latency, doubled the monitoring records, made the window count analyses
as two, and left room for the two answers to diverge one day.

`POST /api/v1/predict` is untouched and remains the endpoint for ordinary API
consumers. A client that deliberately calls both endpoints gets **two** observations,
which is correct — that is two requests. What must not happen is the UI generating
both for a single action, and
`test_one_submission_makes_exactly_one_scoring_call` extracts the `submit()` body by
brace matching and asserts it.

Measured over the boundary with monitoring on:

| Action | `n_records` before | after | `successful_records` |
|---|---|---|---|
| one `POST /api/v1/explain` | 0 | **1** | 1 |
| one `POST /api/v1/predict` | 0 | **1** | 1 |
| both, deliberately | 0 | **2** | 2 |
| five `GET /api/v1/monitoring` after one analysis | 1 | **1** | 1 |

### Record wording corrected

The record previously carried `prediction_endpoint_reused = true`. Read literally
that says the frontend calls `/api/v1/predict`, which it never did. The claim it was
meant to express is that the explanation goes through the one canonical scoring
implementation. Both facts are now stated unambiguously, and the false-sounding name
is gone rather than preserved:

| Retired | Replaced by |
|---|---|
| `prediction_endpoint_reused` | `canonical_prediction_service_reused` — the explanation scores through `ChurnInferenceService.score_with_features` |
| — | `prediction_endpoint_unchanged` — `/api/v1/predict` is byte-for-byte the Phase 11 endpoint |
| — | `ui_analysis_endpoint = "/api/v1/explain"` |
| — | `ui_scoring_calls_per_submission = 1` |
| — | `ui_monitoring_observations_per_submission = 1` |

---

## 8. Contribution schema

Per raw feature:

| Field | Meaning |
|---|---|
| `feature` | the contracted feature name |
| `kind` | `numeric` or `categorical` |
| `value_display` | the value as the model received it (`(blank)` for a structural zero) |
| `active_level` | the encoder level that was active, or `null` for a numeric feature |
| `is_unseen_level` | `true` when the category was not in the training contract |
| `contribution_log_odds` | this feature's term in the linear predictor |
| `direction` | `increases` / `decreases` / `neutral` |

**An unseen category is flagged, not silently zero.** `handle_unknown="ignore"`
encodes it as an all-zero block, so its contribution is exactly `0.0`. Reporting a
bare zero would be true and useless: a reader cannot tell "this level happens to have
no effect" from "the model never saw this level".

The UI groups them as **"pushes the model score higher"** and **"pushes the model
score lower"**, ordered by `|contribution|` **for this request**. The response says
so explicitly: `ordering_is_global_feature_importance: false`. Because the one-hot
parameterisation is redundant with the intercept, an individual level's coefficient
is a property of this fitted parameterisation rather than a transferable quantity —
a global ranking across mixed column types would be misleading, and Phase 10 already
declined to produce one.

---

## 9. Structural coupling

Phase 10 proved seven product equivalences; Phase 12 turned them into data-quality
rules. Phase 13 uses the same declaration for presentation, and derives the blocks
from `churn.monitoring.structural.STRUCTURAL_RULES` rather than restating them.

| Block | Trigger | Members |
|---|---|---|
| No internet service | `InternetService = No` | 7 (trigger + 6 add-ons) |
| No phone service | `PhoneService = No` | 2 (trigger + `MultipleLines`) |

**The problem this solves is a reading error, not a computation error.** A customer
without internet produces seven correct terms. Listed as seven rows they read as
seven independent reasons the score moved; they are one fact about how the product
encodes an account, recorded in seven columns.

So the block is one row whose value is the **exact sum** of its members:

```text
No internet service   value = -2.050685
                members_sum = -2.050685      exact match
intercept + groups + ungrouped = -5.545903
model_logit                    = -5.545903
```

Grouping changes the layout and never the arithmetic. The flat 19-term list is still
returned in full beside the grouped view, and both reconstruct the logit.

**A record that breaks the equivalence is not grouped.** `InternetService = No` with
`OnlineSecurity = No` is accepted by the API, scored, and counted by Phase 12 as a
structural violation. Here it means the seven columns are *not* saying one thing, so
they are shown individually — presenting them as a block would assert a relationship
the input just contradicted.

Nothing about `prepare_features` or the API contract changed.

---

## 10. UI design

Direction: a clean internal tool a Customer Success team would actually use. One
neutral ramp plus two semantic accents; density from spacing and rules rather than
colour; no gradients beyond a 26px brand mark, no neon, no emoji as language, and no
animation except a 13px spinner that respects `prefers-reduced-motion`.

### Screens

One page, two columns on desktop, stacked below 1080px:

* **Top bar** — brand, and three live status pills (API, model artifact, monitoring),
  every one of them read from `/health/ready` and `/api/v1/monitoring`. No badge is
  green because the HTML says so.
* **Customer panel** — the 19 features in five fieldsets, two synthetic examples and
  a clear button.
* **Result panel** — score, decision, threshold, rule, calibration, and the threshold
  visualisation.
* **Contributing factors** — two columns, higher/lower, with a collapsible
  reconstruction block.
* **Model details** — from `/api/v1/model`.
* **Held-out evaluation** — from `/api/v1/portfolio`.
* **Monitoring** — from `/api/v1/monitoring`.

### The 19-feature form

Exactly the contracted features, in product-shaped groups: Account (`tenure`,
`MonthlyCharges`, `TotalCharges`), Demographics, Phone, Internet & add-ons, Contract
& billing. `customerID` and `Churn` are not fields and are rejected by the schema.
No engineered field is offered — none exists in the frozen contract.

### Categorical inputs

Dropdowns are built from `known_levels`, which the metadata endpoint reads from the
**fitted encoder** — not from a dataset and not from a hand-written list.

This is a **convenience layer for the UI only**. The API still accepts any non-blank
category, because the frozen encoder was fitted with `handle_unknown="ignore"`
precisely so a new contract term or payment method can still be scored. Every
dropdown therefore carries a final **"Custom / unseen value…"** option that opens a
text input, so the unseen-category behaviour stays demonstrable rather than being
quietly closed off by the front end.

### Structural relationships in the form

Setting `InternetService = No` fills the six add-ons with `No internet service`;
setting `PhoneService = No` fills `MultipleLines` with `No phone service`.

**This is UI assistance and nothing more.** The user can override any of them, the
API accepts the result, and Phase 12 counts it as a structural inconsistency exactly
as before. `prepare_features` is untouched, and no validation rule was added to the
backend.

### TotalCharges

The field is a text input, sent as typed — including blank. The help text explains
that a blank is admissible only at `tenure = 0`, that the frozen preprocessing treats
it as a structural zero, and that at positive tenure the API rejects it.

**The rule is not reimplemented in JavaScript.** The backend is the authority; the
page improves the wording of the rejection, not the decision. Confirmed live:
`tenure=0` blank → `200`; `tenure=5` blank → `422 INVALID_FEATURE_VALUE`.

---

## 11. Prediction experience and the threshold visual

The result panel shows, after one request to `POST /api/v1/explain`:

```text
Model churn score          79.8%      (full precision 0.7985776966898797)
Decision                   churn
Frozen decision threshold  0.3273
Rule                       score >= 0.3273
Calibration                NONE
```

The percentage is **formatting**; the full-precision value is displayed beside it and
the decision is never recomputed from the rounded number. The decision comes from the
backend — the script has no threshold comparison anywhere, which
`test_the_script_never_recomputes_a_decision` enforces.

### The scale

An inline SVG axis from 0 to 1: a light track below the threshold, a warm track above
it, a dashed line at the threshold, and a dot at the customer's score. It exists to
make one point visible to a non-specialist — **the classification does not use 0.5**.
It uses `0.3272694566222328`, selected in Phase 9B by a pre-registered
F1-maximisation policy, with `>=` so a customer exactly on the boundary is predicted
positive.

The only arithmetic the JavaScript performs is converting those two numbers into
positions on that axis.

### No certainty language

`calibration_policy` is `NONE`, so the page says:

> Model-estimated churn probability; no post-hoc calibration was adopted, so the
> score is a ranking position rather than a calibrated frequency.

`test_the_page_does_not_claim_certainty` forbids "certainty", "certain to", "will
definitely", "guaranteed" and "% accurate" anywhere in the page or the script.

### No risk bands, no recommendations

One threshold was justified, so the product reports **two sides**: *below threshold*
and *at/above threshold*. No low/medium/high cuts were invented.

No retention action is generated. The dataset carries no costs, no capacity and no
treatment effects, so a discount or a call list would be fabricated. The footer says
what is true instead: the output *can support retention prioritisation when combined
with business costs and operational capacity*.

### Local explanation is not causal

Carried in the response and rendered under the factor lists:

> Contributions explain the frozen model's calculation for this input. They are not
> causal effects or recommendations for changing customer behaviour.

---

## 12. API surface

### The demo is opt-in

```text
CHURN_SERVING_PORTFOLIO_UI   default: 0 (off)
```

With the UI **off**, the published route list is byte-identical to Phase 11's:

```text
/health/live  /health/ready  /api/v1/model  /api/v1/predict  /api/v1/predict/batch
```

`/demo`, `/demo/static/*`, `/api/v1/explain` and `/api/v1/portfolio` all return
`404`. With it **on**, those are added and nothing existing changes.

The reason is the same one that made monitoring opt-in: Phase 11 published a route
list and recorded it, so a demo that added routes by default would silently alter a
published contract. `reports/experiments/serving_results.json` is unchanged.

### Endpoints added

| Route | Method | Purpose |
|---|---|---|
| `POST /api/v1/explain` | POST | the same prediction, plus its exact decomposition |
| `GET /api/v1/portfolio` | GET | versioned metadata: policy, evaluation, known levels |
| `GET /demo` | GET | the page (not in the OpenAPI schema — it is a page, not an API) |
| `GET /demo/static/*` | GET | the stylesheet and the script |

`PredictionResponse` was **not** modified. `/api/v1/model` was **not** overloaded;
the versioned metadata got its own read-only endpoint.

### One new error code, deliberately outside the frozen set

`EXPLANATION_UNAVAILABLE` is emitted when the reconstruction gate fails. It is **not**
a member of `ERROR_CODES`, whose contents `serving_results.json` records and verifies
byte for byte. It lives in `PORTFOLIO_ERROR_CODES`, and the union is `ALL_ERROR_CODES`.

That is not a loophole — it is accurate. The code belongs to a route that exists only
when the demo is enabled, so on a Phase 11 deployment it genuinely cannot be emitted
and the recorded set remains exactly true. Extending the recorded set would have
changed a published contract to make a later phase more convenient.

---

## 13. Model card

Read live from `GET /api/v1/model` — nothing is hardcoded in the HTML:

```text
Estimator             LogisticRegression
Frozen model          9d1db49
Model fingerprint     a57568e18ec0333e…
Input features        19
Transformed features  46
Calibration           NONE
Threshold policy      F1_MAXIMIZATION
Threshold             0.3272694566222328
Positive class        Churn = Yes
Serving version       (from the API)
```

---

## 14. Evaluation metrics shown

Copied from `reports/experiments/holdout_results.json` by an **offline** build step
into `reports/portfolio/portfolio_metadata.json`. The serving process reads the small
versioned summary, never the evaluation record — so nothing in the runtime is in a
position to re-evaluate. `evaluations_performed` stays **1**.

### Headline

| Metric | Value | 95% CI |
|---|---|---|
| Average precision | 0.633702 | 0.577681 – 0.686356 |
| ROC-AUC | 0.842034 | 0.818325 – 0.863266 |
| Recall | 0.721925 | — |
| Precision | 0.537849 | — |
| F1 | 0.616438 | — |

Intervals are Phase 9D's bootstrap percentiles (2 000 replications, seed 42), copied
value for value; `test_every_interval_equals_the_phase_9d_artefact` checks each one.

### Auxiliary, never highlighted

`accuracy` 0.761533 · `specificity` 0.775845 · `balanced_accuracy` 0.748885 ·
no-skill AP 0.265436.

**Accuracy is structurally prevented from being a headline metric.**
`verify_portfolio_metadata` fails if `accuracy` appears in the headline list, and
`test_accuracy_is_never_a_headline_metric` fails with it. On a population where
73.5 % of customers do not churn, a model that predicts nobody churns scores close to
it while finding no one.

The metric names are used correctly: ROC-AUC 0.842 is **ranking quality**, not "84 %
accurate" and not certainty. That sentence ships with the numbers:

> Average precision and ROC-AUC are ranking quality, not accuracy and not certainty.
> Precision, recall and F1 describe the single frozen operating point. Accuracy is
> reported for completeness only […]

### The analyst-exposure caveat

> The final evaluation used the frozen holdout protocol. Earlier exploratory analysis
> had already exposed the analyst to all rows, so the estimate may carry optimistic
> bias that cannot be quantified here.

It is a **field of the metadata artefact**, not a line of HTML, so the metrics cannot
be rendered without it. A portfolio page is exactly where an inconvenient limitation
goes missing; this one cannot.

---

## 15. Monitoring integration

The page reads `/api/v1/monitoring` and displays what it returns.

* Monitoring **off** → the endpoint is absent and the card says
  *"Monitoring disabled for this deployment."*
* Monitoring **on**, window below the minimum → `INSUFFICIENT_DATA`, and the card
  explains that the per-feature distributions are suppressed by the privacy policy.
* Monitoring **on**, window at or above the minimum → status, window size, data
  drift, prediction drift and structural consistency.

**The frontend cannot bypass the small-window policy.** It never reaches for
`bin_counts`, `count_by_level` or `n_distinct_unseen_observed`, and it reconstructs
nothing that was withheld — `test_the_page_cannot_bypass_the_small_window_policy`
enforces that by inspecting the script.

---

## 16. Health integration

Three pills, all backend-sourced:

| Pill | Source |
|---|---|
| API | `/health/ready` → `status` |
| Model artifact | `/health/ready` → verified when ready |
| Monitoring | `/health/ready` → `monitoring`, and `/api/v1/monitoring` |

No badge is green because the HTML says so.

---

## 17. Error UX

The API's stable codes are mapped to something a person can act on:

| Code | Shown |
|---|---|
| `INVALID_REQUEST_SCHEMA` | "Check the highlighted input fields." + the offending `loc`/`msg` pairs |
| `INVALID_FEATURE_VALUE` / `INVALID_FEATURE_PAYLOAD` | the backend's sanitised message |
| `EXPLANATION_UNAVAILABLE` | "The explanation could not be verified and was withheld." |
| `SERVICE_NOT_READY` | "The service is not ready." |

Offending fields are marked with `aria-invalid`. The error box is
`role="alert"` + `aria-live="assertive"`.

No traceback, no filesystem path and no Python exception ever reaches the page —
`test_no_error_response_carries_a_traceback_or_a_path` checks the response body. The
Phase 11 error envelopes were **not** modified to make JavaScript easier.

### Loading and double-submit

The submit button is disabled and shows a small spinner while a request is in flight,
and an `inFlight` guard drops a second submit outright.

---

## 18. Synthetic examples

Two, both invented:

* **Synthetic example A** — short tenure, fiber, month-to-month, electronic check.
* **Synthetic example B** — long tenure, no internet, two-year contract, bank transfer.

Neither is a row from the training pool or the holdout, neither carries an
identifier, and neither is labelled by outcome before the model has been asked. The
names are deliberately neutral: calling one "high-risk customer" would be asserting
the answer before running the model.
`test_the_synthetic_examples_are_labelled_as_synthetic` forbids "high-risk customer"
and "loyal customer" anywhere in the page.

The form starts **empty**.

---

## 19. Privacy and security

| Property | Status |
|---|---|
| Payload persisted | **no** |
| Prediction / explanation persisted | **no** |
| Identifier accepted or stored | **no** |
| `localStorage` / `sessionStorage` / IndexedDB / cookies | **none** |
| Analytics or telemetry | **none** |
| Third-party requests | **none** |
| External fonts or CDN | **none** |
| Real customer data embedded | **no** |
| Static root from operator configuration | **yes** |
| User-supplied file paths accepted | **no** |
| Upload or write endpoints | **none** |
| Traceback or path in an error body | **no** |
| Dangerous DOM sinks in `app.js` | **none** |
| User-controlled values rendered as text | **yes** |
| Sanitiser library added | **no** |
| Portfolio metadata digest pinned in source | **yes** |

Static files are served by Starlette's `StaticFiles` from a directory resolved once
at startup from settings. **No route takes a filename**, so no request parameter ever
reaches a filesystem call. `test_the_static_mount_refuses_to_escape_its_root` tries
three traversal shapes, including percent-encoded, and requires that
`pyproject.toml` never comes back.

### Untrusted input reaches the DOM as text, and only as text

The form offers **Custom / unseen value…** on every categorical, so a feature can
carry arbitrary text the visitor typed. That text comes back in `value_display`, can
appear in `active_level`, and can be quoted inside an error message. It is data.

The approach is not "escape carefully at each call site" but **construct no HTML
string at all**. `app.js` builds nodes with `createElement`, writes content through
`textContent` or `value`, empties containers with `replaceChildren()` and fills them
with `appendChild`. There is exactly one node factory:

```js
function el(tag, className, text) {
  var node = document.createElement(tag);
  if (className) { node.className = className; }
  if (text !== undefined && text !== null) { node.textContent = String(text); }
  return node;
}
```

A node built anywhere in the file therefore carries its value as text by
construction, and there is no second path to get wrong.

| Sink | Occurrences in `app.js` | Occurrences in `index.html` |
|---|---|---|
| `innerHTML` | **0** | **0** |
| `outerHTML` | **0** | **0** |
| `insertAdjacentHTML` | **0** | **0** |
| `document.write` | **0** | **0** |
| `eval` | **0** | **0** |
| `new Function` | **0** | **0** |
| `createContextualFragment` / `DOMParser` | **0** | **0** |
| inline `on*=` handler | — | **0** |
| inline `<script>` | — | **0** (one `src=` tag, local) |

The audit found **no sink to remove**: the file was already written this way. One
change was made for the sake of the invariant rather than to fix a defect —
`clear()` emptied its container with a `removeChild` loop and now calls
`replaceChildren()`, so "this file constructs no HTML string" is a property of the
whole file rather than of each call site.

**No sanitiser was added.** A value that is never parsed as HTML has nothing to
sanitise, and a sanitiser would be a dependency guarding a sink that does not exist.

Two proofs, one per side of the boundary:

* **Backend** — `test_a_markup_shaped_unseen_value_is_returned_as_data` posts five
  payloads (`<img src=x onerror="alert(1)">`, `<script>alert(1)</script>`,
  `</script><svg onload=alert(1)>`, `"><svg/onload=confirm(document.domain)>`,
  `javascript:alert(1)`) as categorical values. Each is accepted as an ordinary
  unseen category, contributes exactly `0.0` (the frozen encoder was fitted with
  `handle_unknown="ignore"`), and comes back in `value_display` **verbatim** — not
  stripped, not escaped, not rewritten. Escaping belongs to the medium that renders
  the value, and the medium is JSON followed by `textContent`. `predict` and
  `explain` still agree exactly on each of them.
* **Frontend** — `test_the_script_uses_no_dangerous_dom_sink` and six companions read
  the delivered assets and assert the absence statically. No browser automation: the
  property is a property of the source, and a headless browser would add a dependency
  and a flake surface without checking it better.

### The metadata the page presents as measured is pinned

`CHURN_SERVING_PORTFOLIO_METADATA_PATH` lets an operator name the file the metrics,
intervals and analyst-exposure caveat come from. Parsing that file and validating its
shape answers *"is this shaped like the summary"*; it cannot answer *"is this the
summary"* — a file whose average precision was edited satisfies every structural
invariant while displaying a number nobody measured.

So the summary's digest is compared at startup against
`churn.portfolio.metadata.EXPECTED_PORTFOLIO_METADATA_SHA256`, an expectation
recorded **in source rather than in the file being checked**, exactly as Phase 12
pins its reference profile with `EXPECTED_REFERENCE_PROFILE_SHA256`.

```text
expected_sha256           : e535e67d8038624f2e09ccdeb35ba9265ac0c5a53adf5355ca805e2228dd16cc
actual_sha256             : computed from the loaded summary's canonical serialisation
source_of_expected_sha256 : churn.portfolio.metadata.EXPECTED_PORTFOLIO_METADATA_SHA256
```

On a mismatch the process **does not start**: `PortfolioStartupError`, with all three
values in the message. It does not fall back to empty metadata and it does not show
the numbers unverified — a portfolio page whose figures cannot be traced is worse
than one that is briefly unavailable. With the UI off, no metadata is loaded at all
and nothing is verified, because nothing is displayed.

The pin is enforced twice. `scripts/build_portfolio_metadata.py --verify` fails on a
stale constant, so rebuilding the summary and forgetting to move the pin fails
offline, in the same command that caused the drift, rather than at the next
deployment. Building deliberately does *not* rewrite the constant: a script that
updated its own expectation would be no expectation at all.

Three contexts meet a stale pin, and they are deliberately **not** equivalent:

| Context | Stale pin | Why |
| --- | --- | --- |
| `build_portfolio_metadata.py` | writes the candidate, **warns**, exits `0` | producing a new summary is how a legitimate change begins; refusing here would make the pin unmovable |
| `build_portfolio_metadata.py --verify` | **fails**, exits `1` | this is the gate — it refuses until the constant is moved deliberately |
| Serving startup, UI on | **fails**, `PortfolioStartupError` | nothing unpinned is ever displayed |

Changing the summary is therefore a deliberate sequence: build the candidate, review
it, move the constant consciously, run `--verify`, and only then serve. Building is
the draft step; `--verify` and the runtime are the fail-closed ones.

`GET /api/v1/portfolio` carries the comparison it passed:

```json
"integrity": {
  "verified": true,
  "actual_sha256": "e535e67d…",
  "expected_sha256": "e535e67d…",
  "source_of_expected_sha256": "churn.portfolio.metadata.EXPECTED_PORTFOLIO_METADATA_SHA256",
  "digest_scope": "canonical serialisation of the summary"
}
```

The digest is taken over the **canonical serialisation**, not over the file's bytes,
so a summary built in memory and one read back from disk hash identically. Because
the model forbids extra fields, the parsed value determines the content completely:
any change to any displayed number, interval, label or caveat moves the digest. Byte
identity of the committed file is a separate and stricter guarantee, enforced offline
by the same `--verify`.

Four tamperings are exercised, each structurally valid and each caught only by the
digest: an inflated average precision, an inflated ROC-AUC, a softened
analyst-exposure caveat that keeps the words the invariant looks for, and a relabelled
metric. In every case `verify_portfolio_metadata` passes **all twelve structural
checks** and fails exactly one — `metadata_digest_matches`. No test edits a real file;
each writes its own copy under `tmp_path`.

---

## 20. Accessibility and responsiveness

`lang="en"`, a skip link, semantic headings, `<fieldset>`/`<legend>` grouping, a
`<label for>` on every input, real `<button>` elements, visible `:focus-visible`
states, `aria-live="polite"` on the result and status regions, `aria-live="assertive"`
+ `role="alert"` on errors, `aria-invalid` on rejected fields, and an SVG with
`<title>` and `<desc>`. `prefers-reduced-motion` disables the spinner animation.

No formal WCAG audit was performed, and the record says so.

Breakpoints at 1080px (columns collapse, form first) and 720px (single-column
fields, tighter type). Desktop is the priority target. No pixel-perfect visual tests.

---

## 21. Tests

| Suite | Before Phase 13 | After the build | After the closing gates |
|---|---|---|---|
| Root | 1170 | 1270 (+100) | **1291** (+121) |
| Serving | 293 | 348 (+55) | **391** (+98) |
| Combined (informational) | 1463 | 1618 | **1682** |

### Root modules added

| Module | What it establishes |
|---|---|
| `test_portfolio_explanation.py` | the identity reconstructs, over six payload shapes; 19 features exactly once; unseen levels flagged; the gate refuses a bad decomposition; determinism |
| `test_portfolio_coupling.py` | blocks derived from Phase 12's rules; group value is the exact member sum; a broken equivalence is not grouped; the total still reconstructs |
| `test_portfolio_metadata.py` | every metric equals the Phase 9D artefact; no holdout, no scoring; accuracy never headline; the caveat travels with the numbers |
| `test_portfolio_record.py` | reproducible, and every claim cites a test that exists |
| `test_portfolio_isolation.py` | no fit, no loader, no SHAP/LIME, no randomness, no web framework, no threshold literal |

### Serving modules added

| Module | What it establishes |
|---|---|
| `test_serving_portfolio.py` | predict == explain on every shared field, bit for bit; one shared scoring primitive; contract edges; fail-closed startup; explained records are observed; **pinned metadata starts, tampered metadata refuses**; **markup-shaped values return as data**; **one analysis is one observation** |
| `test_portfolio_ui.py` | opt-in routes; zero external assets; no storage or analytics; 19 features present; no hardcoded model constant; no risk bands; the small-window policy is respected; **no dangerous DOM sink**; **exactly one scoring call per submission** |

### Added at the closing gates

| Test | Gate |
|---|---|
| `test_the_committed_summary_matches_the_pinned_digest` | the artefact and the constant move together |
| `test_a_tampered_summary_still_passes_every_structural_invariant` | the premise: structure cannot detect this |
| `test_a_tampered_summary_is_caught_by_the_digest` | …and exactly one check does |
| `test_intact_metadata_starts_the_demo` | integrity verified → startup PASS |
| `test_tampered_metadata_stops_the_demo` | AP, ROC-AUC or caveat edited → startup FAIL |
| `test_a_tampered_summary_never_reaches_a_response` | no silent degradation to unverified metrics |
| `test_metadata_is_not_read_when_the_demo_is_off` | UI OFF loads nothing |
| `test_the_script_uses_no_dangerous_dom_sink` | static frontend audit |
| `test_a_markup_shaped_unseen_value_is_returned_as_data` | five malicious payloads, returned verbatim as data |
| `test_one_submission_makes_exactly_one_scoring_call` | `submit()` posts once, to `/api/v1/explain` |
| `test_one_analysis_is_observed_exactly_once` | N → N + 1, never N + 2 |
| `test_a_direct_prediction_call_is_observed_exactly_once` | `/predict` keeps its Phase 12 semantics |

---

## 22. How to run locally

```bash
# 1. API only — the Phase 11 contract, unchanged
uv run --project serving python -m churn.serving

# 2. API + portfolio demo
CHURN_SERVING_PORTFOLIO_UI=1 uv run --project serving python -m churn.serving
#    then open http://127.0.0.1:8000/demo

# 3. API + demo + monitoring
CHURN_SERVING_PORTFOLIO_UI=1 CHURN_SERVING_MONITORING=1 \
  uv run --project serving python -m churn.serving

# regenerate the Phase 13 artefacts
uv run python scripts/build_portfolio_metadata.py [--verify]
uv run python scripts/build_portfolio_record.py   [--verify]
```

On Windows PowerShell, set the variable first:

```powershell
$env:CHURN_SERVING_PORTFOLIO_UI = "1"
uv run --project serving python -m churn.serving
```

The three combinations are independent: monitoring and the demo are separate flags,
and either can be on without the other.

---

## 23. Limitations

1. **The demo is a single-process, local product.** No authentication, no rate
   limiting, no TLS and no deployment. It is meant to be run and shown, not exposed.

2. **The unauthenticated-endpoint limitation carries over from Phase 12.** Enabling
   the demo publishes two more read-only endpoints on the same unauthenticated
   surface. Neither exposes a row, but both are readable by anyone who can reach the
   port.

3. **The explanation is local and additive by construction.** It says what the frozen
   model computed for one input. It is not a causal effect, not a counterfactual, and
   not transferable to another customer.

4. **Individual one-hot coefficients are parameterisation-dependent.** The encoding is
   redundant with the intercept, so a single level's contribution is a property of
   this fit. The ordering shown is explicitly local to the request; no global
   importance ranking is produced, here or in Phase 10.

5. **The metrics inherit the analyst-exposure caveat** (§14), which cannot be
   quantified.

6. **Displayed percentages are formatting, not calibration.** 79.8 % is
   `0.7985776966898797` rounded for reading; it is not a calibrated frequency, and
   `calibration_policy` is `NONE`.

7. **Coupled-block grouping is presentational.** It helps a reader avoid
   double-counting, but the underlying coefficients remain individually
   non-identified (limitation 4).

8. **No browser-level testing.** The UI tests are static and HTTP-level. Rendering,
   layout and keyboard traversal in a real browser are unverified by automation.

9. **No formal accessibility audit.** The basics are in place and tested; WCAG
   conformance is not claimed.

10. **The demo assumes the repository layout.** Static assets are located relative to
    the project root (overridable by setting). A non-editable install would need the
    path configured.

11. **Deployment is out of scope.** No cloud, no domain, no container. Documented as
    a future extension.

---

## 24. Boundary with what comes next

Phase 13 produces a product. It does **not** produce the academic deliverable
(Phase 14) or the GitHub polish (Phase 15). `README.md` was not touched, no
Colab/notebook/slides were created, and no deployment was performed.
