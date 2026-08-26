# Phase 11 — Production Inference API / Serving Boundary

Machine-readable record: [`reports/experiments/serving_results.json`](experiments/serving_results.json)

---

## 1. Scope

Phase 11 turns the object frozen in Phase 9C into something that can be called over
HTTP. It is a packaging phase, and the constraint that defines it is what it is
**not** allowed to do:

| | |
|---|---|
| Trains a model | no — `fit_calls = 0`, asserted behaviourally |
| Selects, tunes, calibrates or re-thresholds | no — every decision comes from the frozen policy |
| Reads the training pool, the holdout or the raw CSV | no — no dataset is reachable from this package |
| Persists a payload, a prediction or an identifier | no — inference is read-only |
| Implements monitoring or drift detection | no — Phase 12 |
| Ships a Dockerfile or a frontend | no — later phases |

What it does add: a verified startup, an inference service callable without a
server, two prediction endpoints, two health endpoints, a metadata endpoint, an
OpenAPI contract, and an isolated dependency environment that leaves the frozen
provenance byte-identical.

**Phase 11 versions the serving application. It does not version the model.**

---

## 2. Frozen model identity

Nothing below was produced in this phase. All of it was read from
`reports/decision_policy.json` and from the artefact it describes.

```text
freeze commit             9d1db4962769093a617f1db2db66354536acffb3  (9d1db49, Phase 9C)
estimator                 LogisticRegression
model_fingerprint_sha256  a57568e18ec0333e1c85b590f98e75da93ebbfafff063860ec5daf4e460d939f
pipeline_sha256           574fde36c6e2e991de7c3dfdab21981dc41eae2810d003ecbfb179dd504dc3d8
decision_policy_sha256    bd7aa800ae36ac04ecf951f7377ce71d16d084faeee771b01eb6c9655a4bc714

final_threshold           0.3272694566222328   (unrounded)
comparison                >=
threshold_policy          F1_MAXIMIZATION      (Phase 9B)
calibration_policy        NONE                 (Phase 9A)
positive_class_label      1                    (Churn = Yes)
positive_class_column     1                    (resolved from classes_, not assumed)

raw features              19
transformed features      46
```

**Model version vs. serving version.** The model's version is the freeze commit
`9d1db49`. This application's version is `1.0.0`. They move independently, and the
commit that lands Phase 11 is **not** a new model version. `GET /api/v1/model`
returns both as separate fields, and a test asserts they differ.

**The policy SHA pin, and its consequence.** The serving code pins
`bd7aa800…` — the byte digest of `reports/decision_policy.json`. Its
consequence is deliberate and worth stating plainly: **this serving application
version is compatible with one specific frozen policy.** A future re-freeze will
change the policy's bytes and will therefore require a deliberate update to the
serving code and manifest. That is the intended behaviour, not a defect — a serving
release that silently accepted a new policy would be a serving release that could
silently serve a different model. The pin is a source constant; it is not
configurable by request, by environment variable, or by any other runtime input.

---

## 3. Runtime architecture

```text
HTTP layer            churn.serving.api         FastAPI, routes, status codes, error envelope
       │                                        (the only module that needs FastAPI)
       ▼
inference service     churn.serving.service     ChurnInferenceService.predict_one / predict_batch
       │              churn.serving.inference   canonical one-row scoring primitive
       ▼
frozen artefacts      churn.serving.artifacts   startup gates, one load per process
                      churn.serving.provenance  runtime source classification + gate
                      churn.serving.runtime     library-version gate
                      churn.serving.settings    operational configuration only
                      churn.serving.schemas     typed request/response contract
                      churn.serving.errors      stable error codes
                      churn.serving.record      the machine-readable record
```

The predictive path does not import FastAPI, and importing `churn.serving` does not
either. `ChurnInferenceService.predict_one(...)` is callable, and asserted on,
without a server — most of the test suite does exactly that.

### The function, in the order it actually runs

```text
raw JSON payload
  → PredictionRequest            pydantic, extra="forbid", exactly 19 fields
  → dict                         plain mapping, no framework type crosses the boundary
  → pandas.DataFrame             ONE ROW, always — see §9
  → prepare_features(...)        FROZEN contract: validates, rejects, canonicalises order
  → pipeline.predict_proba(...)  frozen preprocessor → frozen classifier
  → column from classes_         resolved, never assumed to be [:, 1]
  → P(Churn = 1)
  → probability >= 0.3272694566222328
  → prediction ∈ {0, 1} → decision ∈ {retained, churn}
```

**Nothing frozen is reimplemented here.** The positive-column resolution
(`positive_class_column`), the probability read (`positive_probability`) and the
comparison itself (`apply_decision_rule`) are imported from
`churn.modeling.freeze`, whose source is hashed into `code_provenance` in the
decision policy — and, as of this phase, gated against that hash at startup. A
local copy of `probability >= threshold` would be a second decision rule, free to
drift while every fingerprint kept matching.

---

## 4. Artifact integrity at startup

The model is located, verified and loaded **once per process**, in the FastAPI
lifespan. Thirty-four gates run before the service is allowed to exist:

| Group | Gates |
|---|---|
| Policy identity | `decision_policy_digest_matches`, `policy_is_the_phase9c_freeze` |
| Decision rule | `threshold_is_finite_and_a_probability`, `decision_comparison_is_greater_or_equal`, `threshold_policy_recognised`, `policy_classes_are_the_encoded_binary_target`, `policy_positive_class_is_unambiguous` |
| Calibration | `calibration_policy_recognised`, `calibration_policy_is_the_frozen_one`, `loaded_classifier_has_no_calibration_wrapper` |
| Feature contract | `inference_entry_point_is_prepare_features`, `feature_contract_matches_the_installed_contract`, `no_engineered_feature_in_the_contract`, `feature_names_digest_matches`, `prepare_features_source_unchanged` |
| **Runtime source provenance** | `runtime_source_classification_covers_the_policy`, `runtime_source_churn_preprocessing_contracts_unchanged`, `runtime_source_churn_preprocessing_transformers_unchanged`, `runtime_source_churn_modeling_freeze_unchanged`, `runtime_source_churn_modeling_models_unchanged`, `runtime_source_churn_preprocessing_pipeline_unchanged` |
| Pipeline artefact | `pipeline_file_digest_matches`, `pipeline_loads`, `pipeline_has_the_frozen_structure`, `model_fingerprint_matches`, `preprocessor_step_present` |
| Loaded object | `loaded_classes_match_the_policy`, `positive_class_column_resolves_from_classes_`, `transformed_feature_names_match` |
| Runtime versions | `runtime_python_matches`, `runtime_scikit_learn_matches`, `runtime_numpy_matches`, `runtime_pandas_matches`, `runtime_joblib_matches` |

Three of these deserve their reasoning stated here; the runtime source group gets
its own section (§5).

**`decision_policy_digest_matches`.** Every other policy gate asks whether the
policy agrees with *itself* — is the threshold in range, does the positive column
match the class list. None of them would notice a hand-edited threshold, because a
hand-edited policy is still internally consistent. So the serving layer pins the
SHA-256 of `reports/decision_policy.json` itself, exactly as the policy pins the
pipeline's bytes. The pinned value is not invented here: Phases 9D, 9E and 10 each
independently recorded the digest of the policy they read, and all three agree.

**`prepare_features_source_unchanged`.** `prepare_features` runs *before* the
persisted pipeline, so it sits outside everything the model's own fingerprints can
see. Phase 9C recorded a digest of that function's own source, and this is the
narrower, function-level companion to the module-level gate in §5.

**The runtime version gates.** A `.joblib` pickles every class **by reference**, so
unpickling imports whatever scikit-learn, numpy and pandas currently define. A minor
release that changes a private attribute or a dtype promotion rule can change what
the loaded object computes while both model fingerprints stay identical. Versions
are checked, and a mismatch **fails the startup**.

---

## 5. Runtime source provenance

### Three things are protected, and only two of them were protected before

| Fingerprint | What it covers | What it cannot see |
|---|---|---|
| `pipeline_sha256` | the artefact's **bytes** | anything about code |
| `model_fingerprint_sha256` | the **learned parameters** — coefficients, scaler statistics, encoder categories | anything about code |
| `code_provenance.source_digests` | the **source of the code that will execute** | — |

The gap the first two leave is not theoretical. A `.joblib` pickles every class **by
reference**: the stream carries the string `churn.preprocessing.transformers`, and
unpickling binds whatever that module currently defines. Edit
`TotalChargesCleaner.transform` and every customer's `TotalCharges` becomes a
different number — while the file's bytes are untouched, the learned parameters are
untouched, the transformed feature names are untouched, the classes are untouched
and the threshold is untouched. Every check the model has about itself passes.

> **Pickle integrity protects serialized state; source provenance additionally
> protects the behavior of custom Python code resolved at runtime.**

Phase 9C recorded those digests for exactly this reason. Phase 11 is what finally
*acts* on them, at startup, before a single request is answered.

### Which files are gated, and on what evidence

Gating all eight recorded files would be the easy answer and the wrong one: three of
them are never executed by this process, and refusing to start over an edit to a
training builder that cannot reach a prediction spends availability for nothing.

So each file was **traced, not assumed**. Three arms, run against the real running
boundary:

1. a walk of the unpickled pipeline's object graph, collecting the defining module of
   every object inside it — this is what finds pickle-by-reference exposure;
2. a call-level profile (`sys.setprofile`) over a full startup *and* a real
   prediction, with every module already imported, so that "imported" and "called"
   are distinguished;
3. an AST scan of every module the profile showed executing, looking for
   `from <recorded module> import ...` bindings — because a module can determine
   behaviour through a constant that another module reads, without any function of
   its own ever being called.

A file is **runtime-effective** if changing it could change (i) the feature matrix
handed to the pipeline, (ii) what the unpickled pipeline computes, (iii) which
probability column is read, (iv) the decision rule applied, or (v) the outcome of a
startup integrity gate — that is, what this boundary accepts as the frozen model.

| # | File | Role | Traced evidence | Why it can (or cannot) change a served prediction |
|---|---|---|---|---|
| 1 | `src/churn/preprocessing/contracts.py` | **runtime-effective** | called on every request: `prepare_features`, `validate_feature_matrix`, `_blank_mask` | **(i)** it *is* the input side of the inference function — which columns reach the model, in which order, which values are rejected as blank |
| 2 | `src/churn/preprocessing/transformers.py` | **runtime-effective** | called on every request, **and the only module owning a class inside the unpickled pipeline** (`TotalChargesCleaner`) | **(ii)** the canonical pickle-by-reference exposure. `transform` is resolved from source at runtime; editing it changes what the loaded pipeline computes with both model fingerprints unmoved |
| 3 | `src/churn/modeling/freeze.py` | **runtime-effective** | called on every request (`positive_probability`, `positive_class_column`, `apply_decision_rule`) and throughout startup (`load_pipeline`, `model_fingerprint`, `transformed_feature_names`) | **(iii)(iv)(v)** it implements the decision rule. `>=` → `>` here moves every customer sitting on the threshold; a changed positive-column resolution inverts every decision |
| 4 | `src/churn/modeling/models.py` | **runtime-effective** | not called, but `CLASSIFIER_STEP` and `PREPROCESSOR_STEP` are bound by `freeze.py` and by `serving/artifacts.py` | **(iii)** `CLASSIFIER_STEP` is dereferenced on the request path, inside `positive_class_column`: `pipeline.named_steps[CLASSIFIER_STEP].classes_`. A different value resolves a different step, so the column read is no longer the one `classes_` describes |
| 5 | `src/churn/preprocessing/pipeline.py` | **runtime-effective** | not called, but `CATEGORICAL_STEP`, `CLEANER_STEP`, `ENCODER_STEP` and `NUMERIC_STEP` are bound by `freeze.py` | **(v)** those constants select which sub-transformer `model_state()` and `transformed_feature_names()` read, and both run as startup gates. A changed value changes what the model fingerprint is computed over, so a model that is not the frozen one could pass verification |
| 6 | `src/churn/modeling/tuning.py` | training-only | not called. `build_modern_logistic_pipeline` is *bound* by `freeze.py` but is only ever invoked from `fit_final_pipeline`, which this process never reaches | it builds an **unfitted** estimator. The served object was fitted in Phase 9C and is read from disk; no builder runs. Gating it would refuse startup over an edit that cannot reach a prediction |
| 7 | `src/churn/features/pipeline.py` | training-only | not called, and not bound by any module the profile showed executing; reachable only through `tuning.py`, itself never called | `build_feature_preprocessor` assembles a preprocessor at fit time. At serving the assembled object is unpickled already configured, so the assembler never runs |
| 8 | `src/churn/preprocessing/target.py` | training-only | not called, and not imported by any module the profile showed executing | `encode_target` maps `Yes`/`No` onto `1`/`0` during training. Serving never sees a label: the positive class it acts on is read from the fitted `classes_` and cross-checked against the policy, never recomputed here |

**5 runtime-effective, 3 training-only.** The classification is *not* taken on trust.
`serving/tests/test_provenance.py` re-runs all three arms of the trace at test time
and asserts the declaration still agrees with what it finds — including the negative
half, that no training-only file executes and none owns an object inside the
pipeline. Move a function between modules and the test fails.

### How the gate resolves a file

Through `sys.modules[<module>].__file__` — never by joining a repository root with a
relative path. Two consequences:

* the digest describes the code **this interpreter has actually bound**, wherever the
  package is installed. Nothing depends on `C:\…`, `/home/…` or `/workspace/…`; what
  is compared is source *content*, and no absolute path is stored, logged or asserted
  anywhere;
* the serving package keeps **zero dynamic-import surface**. The runtime-effective
  modules are all pulled in by `churn.serving.artifacts` before any gate runs, so
  they are looked up rather than imported, and its static audit enforces the absence
  of `import_module`. A module that is *not* loaded is a failed gate, never a skip.

The digest is `source_digest` — SHA-256 of the source text with newlines normalised
to `\n` — the same function and the same convention the policy's digests were
produced with, so a CRLF checkout cannot fail the gate for a reason unrelated to the
code.

### Behaviour on mismatch

Startup fails. Not a warning, not a degraded mode, not a repair. The failing gate is
named, every other failing gate is named alongside it, and the process does not come
up. The expected digest is never regenerated, the policy is never rewritten, and no
artefact is rebuilt.

---

## 6. Fail-closed behaviour

Startup errors are a separate hierarchy from request errors, and none of them is
ever turned into an HTTP response: a process that cannot prove it holds the frozen
model does not start.

| Scenario | Result |
|---|---|
| Policy absent | `ArtifactNotFoundError` — process does not start |
| Policy is not a decision policy | `ArtifactIntegrityError` — schema violation named |
| Policy bytes altered (threshold, calibration, anything) | `ArtifactIntegrityError` — `decision_policy_digest_matches` |
| Pipeline absent | `ArtifactNotFoundError` — process does not start |
| Pipeline bytes altered | `ArtifactIntegrityError` — `pipeline_file_digest_matches` |
| Model swapped, policy updated to accept the new bytes | `ArtifactIntegrityError` — `model_fingerprint_matches` |
| **`contracts.py` altered** | `ArtifactIntegrityError` — `runtime_source_churn_preprocessing_contracts_unchanged` |
| **`transformers.py` altered** | `ArtifactIntegrityError` — `runtime_source_churn_preprocessing_transformers_unchanged` |
| **`freeze.py` altered** | `ArtifactIntegrityError` — `runtime_source_churn_modeling_freeze_unchanged` |
| **A new provenance entry left unclassified** | `ArtifactIntegrityError` — `runtime_source_classification_covers_the_policy` |
| A training-only file altered | **starts** — it cannot reach a prediction |
| Calibration wrapper present | `ArtifactIntegrityError` — `loaded_classifier_has_no_calibration_wrapper` |
| Feature contract diverged | `ArtifactIntegrityError` — contract and digest gates |
| Library version diverged | `RuntimeCompatibilityError` — process does not start |
| `CHURN_THRESHOLD` (or any equivalent) set | `ServingConfigurationError` — refused, not ignored |

**No failure path rebuilds anything.** A missing pipeline is an error, never a
trigger to run `scripts/freeze_model.py`. A service that can regenerate its own
model can also regenerate a *different* one, and nothing downstream would know.
Every failed gate is reported at once rather than one restart at a time.

Two of these tests are worth naming.

`test_a_swapped_model_is_caught_by_the_fingerprint_even_when_the_bytes_agree` alters
the classifier's coefficients, re-serialises the pipeline, and then updates the
policy to accept the new bytes. The file-digest gate passes — and the model
fingerprint, which describes what the model *does* rather than how it was stored,
still refuses it.

`test_a_changed_total_charges_cleaner_stops_the_service_before_any_request` first
proves the premise on the real artefact — `TotalChargesCleaner` genuinely is inside
the loaded pipeline, and its `__module__` genuinely is
`churn.preprocessing.transformers` — then simulates a divergent source digest for
that module and asserts the startup refuses. No real file is edited: divergence is
injected through the digest reader, and a final test re-hashes all eight recorded
files and asserts the repository is untouched.

---

## 7. Request contract

Exactly the 19 raw features. `customerID` and `Churn` are **rejected**, not dropped:
dropping an unexpected field is how a client comes to believe the model reads
something it never sees.

| Field | Type | Note |
|---|---|---|
| `tenure` | number | months with the company |
| `MonthlyCharges` | number | |
| `TotalCharges` | number \| string \| null | may arrive blank; see §8 |
| `SeniorCitizen` | **integer** | see below |
| `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` | string | free-text category, no enum |

Three decisions in this schema could easily have been made the other way.

**Categories are typed, not enumerated.** The frozen encoder was fitted with
`handle_unknown="ignore"` so that a category which did not exist at training time —
a new contract term, a new payment method — can still be scored. Freezing today's
observed values into an `Enum` would revoke that Phase 4 decision by the back door:
the request would be rejected at the schema and the encoder would never be reached.

**No range constraints on the numeric features.** The training data spans some
interval of `MonthlyCharges`; that is a fact about a sample, not a definition of a
valid input. `le=118.75` would confuse the training distribution with input validity
and reject a real, scoreable customer. Whether a record is *far from the training
distribution* is a monitoring question — Phase 12 — not a schema question. A test
asserts no `minimum`/`maximum` appears in the published schema.

**`SeniorCitizen` is an integer, and that is a type decision, not a domain one.**
The frozen encoder learned it as the integers `0`/`1`. A JSON string `"0"` would not
match the integer `0`, so `handle_unknown="ignore"` would encode it as an all-zero
block and the record would be scored **as if the field had never been sent** — no
error, a quietly different prediction. Typing it as an integer is what prevents that.

### Blanks

The authority on "blank" is the frozen feature contract, one layer down — the HTTP
schema does not re-implement it.

| Input | Result |
|---|---|
| `"Contract": ""` | 422 `INVALID_FEATURE_VALUE` |
| `"Contract": "   "` | 422 `INVALID_FEATURE_VALUE` |
| `"Contract": null` | 422 `INVALID_REQUEST_SCHEMA` (not a string) |
| `"Contract": "Three year prepaid"` | **200** — unseen but non-blank; the encoder absorbs it |

That last row is the point of §16 below: an unseen category is *accepted*, and that
is a statement about the boundary, not about whether the score should be trusted.

### Key order

JSON has no order semantics. `prepare_features` normalises to `FEATURE_COLUMNS`
after validating, and a test sends the same payload with its keys reversed and
asserts a byte-identical response — plus a second test that captures the frame
handed to `predict_proba` and asserts its columns are the canonical order.

---

## 8. `TotalCharges`

The rule is unchanged and is **not** duplicated at the HTTP layer. The schema only
has to let the raw value survive, which is why the field accepts a number, a string
or `null`.

| Input | `tenure` | Result |
|---|---|---|
| blank / whitespace / `null` | `0` | **accepted** — structural zero: a customer who has not completed a billing cycle |
| blank / whitespace / `null` | `> 0` | 422 `INVALID_FEATURE_VALUE` — a blank here means "unknown", not "never billed" |
| `"n/a"`, any unreadable text | any | 422 `INVALID_FEATURE_VALUE` |
| `"845.50"` or `845.50` | any | accepted |

The decision lives in `clean_total_charges`, inside the frozen pipeline. The serving
layer's only job is to not corrupt the value on the way there — and, as of §5, to
refuse to start if that decision's source has changed.

---

## 9. Endpoint invariance, and the canonical scoring path

### The contract

**For any record, `/api/v1/predict` and `/api/v1/predict/batch` return the same
probability — the same `float`, by `==`, not within a tolerance.** The decision, the
label, the threshold and the comparison are identical too. A caller never has to know
how a request was batched in order to reproduce a result.

### How it was originally implemented, and what that revealed

The first implementation scored a batch as one N-row matrix. That is the obvious and
efficient thing to do, every step of the frozen pipeline is row-wise, and it was not
wrong. The equivalence tests still caught a difference, and diagnosing it before
changing anything mattered:

```text
preprocessing output (batch vs. row-by-row)   BIT-IDENTICAL   (numpy.array_equal)
decision_function output                      DIVERGENT
max |Δ| on decision_function                  4.44e-16
max |Δ| on the probability                    1.11e-16   (1 ULP)
max relative Δ                                1.40e-16
decisions                                     identical
```

**Root cause.** `decision_function` is a matrix product. BLAS dispatches a 1×46
product to a different kernel (GEMV) than an N×46 one (GEMM); the two accumulate the
same 46 terms in a different order, and floating-point addition is not associative.
The scaler is elementwise and the encoder is a lookup, which is why everything before
the classifier was bit-identical.

### Why it was changed anyway

The divergence was one bit, and it could only have changed a decision for a record
sitting within one ULP of the threshold. It was still the wrong contract to publish:
the same customer could come back with two different probabilities depending on how
the caller happened to group the request, and "1 ULP" is not something an API
consumer should have to reason about.

**Rounding the score was not an option.** Phase 9C stored the threshold unrounded
precisely because a value rounded in the sixth decimal can move customers across the
boundary. Rounding the probability instead would have been that same mistake
mirrored, and would have hidden the effect rather than removed it.

So the arithmetic was made canonical instead. `predict_one` and `predict_batch` both
call one primitive, `ChurnInferenceService._score`, once per record, and that
primitive always scores a **one-row** frame — the shape the single endpoint always
used. GEMV/GEMM is no longer an observable difference, because only one of them is
ever reached.

```text
predict_one(record)      → _score(record)                          → canonical one-row inference
predict_batch(records)   → [_score(r) for r in records]            → the same primitive, N times
```

The pipeline is still loaded **once per process**. What repeats per record is
arithmetic, not I/O: nothing is unpickled, re-read or re-verified per row, and there
is no second copy of the validation.

### The trade-off, chosen consciously

* **Cost.** Batch scoring no longer amortises one matrix product over N rows, so its
  potential throughput is lower than a vectorised implementation's. **No benchmark is
  claimed here and the cost is not zero.**
* **Gain.** Exact endpoint invariance: a probability is a property of a record, not
  of the request that carried it.
* **Why it was judged acceptable.** The model is small — a logistic regression over
  46 transformed columns — and the batch endpoint is bounded at 500 records, so the
  ceiling on what is given up is known and modest. Semantic consistency at a public
  boundary was judged worth more than unmeasured throughput.
* **One behavioural consequence.** A batch containing an invalid record now reports
  the **first** offending record rather than every offending row at once. The
  exception type is preserved, so the error code a client sees is unchanged, and the
  message is prefixed with the record's index. Validation is not duplicated to
  recover the old aggregate report.

### What is proved

| Claim | Test |
|---|---|
| `single == batch` in every field, exact | `test_batch_and_single_are_exactly_equal_in_every_field` |
| Batch size does not move a float (1, 2, 21, 100, 500) | `test_the_probability_is_bit_identical_at_every_batch_size` |
| Position in the batch does not move a float | `test_position_inside_the_batch_does_not_move_a_probability` |
| `A, B, A, C` — order preserved, both `A`s identical | `test_a_repeated_record_is_scored_identically_at_both_positions` |
| Same, over HTTP, on the decoded JSON | `test_a_batch_with_a_repeated_record_preserves_order_and_repeats_the_answer` |
| One record, both routes, one float | `test_the_same_record_through_both_endpoints_is_numerically_identical` |
| Corresponding `PredictionResponse` objects equal | `test_batch_and_single_return_identical_json` |

The shared assertion helper compares **every** field with `==` and has no ULP budget.
When a probability assertion fails it reports the ULP distance, because "3 ULP apart"
and "a different number entirely" call for very different investigations.

---

## 10. Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/predict` | score one customer |
| `POST` | `/api/v1/predict/batch` | score several, answering in input order |
| `GET` | `/api/v1/model` | identity and policy of the frozen model |
| `GET` | `/health/live` | liveness — answers without touching the model |
| `GET` | `/health/ready` | readiness — 200 only if a verified model is loaded |
| `GET` | `/openapi.json`, `/docs` | the published contract |

Request is `{"records": [ ... ]}`; response is `{"count": n, "predictions": [ ... ]}`
in input order. Nothing is persisted. One invalid record fails the **whole** batch —
a caller must not have to guess which rows were scored.

The limit is **500 records** by default, configurable through
`CHURN_SERVING_MAX_BATCH_SIZE` (hard ceiling 5 000), and it lives in the serving
layer only: `configs/base.toml` belongs to the frozen provenance and was not
touched. **The limit bounds the request, never the model.**

---

## 11. Prediction semantics

```json
{
  "churn_probability": 0.6962685943547531,
  "prediction": 1,
  "decision": "churn",
  "threshold": 0.3272694566222328,
  "comparison": ">=",
  "calibration_policy": "NONE",
  "model_fingerprint": "a57568e18ec0333e1c85b590f98e75da93ebbfafff063860ec5daf4e460d939f"
}
```

**`churn_probability` is not a calibrated probability.** The frozen calibration
policy is `NONE`, decided in Phase 9A because neither sigmoid nor isotonic
calibration cleared the pre-registered eligibility rule. The score orders customers
by risk on a 0–1 scale; it does not promise that 0.30 means 30 % of such customers
churn. The OpenAPI description says this explicitly, and a test asserts the wording
is there.

**`prediction` and `decision` are the same fact twice**, and both follow from
`churn_probability >= threshold` and nothing else. `pipeline.predict` is never
called — it would apply scikit-learn's own 0.5 cut, which is not the frozen policy.
That is enforced twice: statically (the identifier `predict` does not appear in the
serving source) and behaviourally (`Pipeline.predict` is monkeypatched to raise and
real requests are driven through the real app).

**The `>=` boundary** is tested on the decision primitive directly, at
`probability == threshold` → `1`, and at both neighbouring doubles via
`np.nextafter`. Constructing a customer whose score lands exactly on a 16-digit
threshold would be an exercise in luck, and a test that depends on luck proves
nothing about the comparison.

**Determinism.** The response carries no timestamp, no request id, and nothing else
that is not a pure function of the payload and the frozen artefacts. Two identical
requests return byte-identical bodies.

---

## 12. Error behaviour

| Status | Code | Cause |
|---|---|---|
| 422 | `INVALID_REQUEST_SCHEMA` | body is not a valid feature payload (missing, extra, wrong type) |
| 422 | `INVALID_FEATURE_PAYLOAD` | `FeatureContractError` from the frozen contract |
| 422 | `INVALID_FEATURE_VALUE` | `DataQualityError` — blank or unreadable where no rule justifies it |
| 413 | `BATCH_TOO_LARGE` | batch beyond the deployment's operational limit |
| 503 | `SERVICE_NOT_READY` | no verified model on this process |
| 500 | `INTERNAL_ERROR` | anything unexpected |

One envelope: `{"error": {"code": ..., "message": ..., "details": [...]}}`.

No traceback, no filesystem path, no internal message reaches a client — asserted
by a test that raises `RuntimeError("secret detail /etc/passwd")` inside the service
and checks neither the string nor the word `Traceback` appears in the response.
Validation `details` carry only `loc` and `msg`; pydantic's `input` field is stripped,
because it would otherwise echo submitted feature values back and from there into any
client-side log.

---

## 13. Dependency and environment isolation

**FastAPI and Uvicorn were not added to the root project, and could not have been.**

`pyproject.toml` and `uv.lock` are part of the Phase 9C provenance: their SHA-256
digests are recorded in `reports/decision_policy.json` under
`code_provenance.configuration_digests`, and `scripts/freeze_model.py --verify`
fails — correctly — the moment either one moves. A web framework must not be able to
invalidate a model freeze.

So the serving dependencies live in an isolated uv project:

```text
serving/
├── pyproject.toml     churn-serving 1.0.0, package = false
├── uv.lock            44 packages
└── tests/             the serving suite
```

It depends on the root project as an **editable path dependency**, so the serving
code itself stays in `src/churn/serving/` alongside the rest of the package, while
the *environment* is what is separated.

**The four ML libraries are pinned with `==`** to exactly the versions recorded in
`reports/decision_policy.json` → `environment`. That is not a preference; it is the
freeze. The service re-checks them at startup and refuses to serve on a mismatch.

```text
python        3.12.13            fastapi     0.141.1
scikit-learn  1.9.0   (pinned)   starlette   1.6.0
numpy         2.5.1   (pinned)   uvicorn     0.52.4
pandas        3.0.5   (pinned)   httpx2      2.12.0  (test transport only)
joblib        1.5.3   (pinned)
```

### Proof the frozen provenance is intact

```text
pyproject.toml     d6eaa68766005bc601792a7dc75aa02091d836ecf8af1aac8797f00bf0b72155  == 47d6dd3
uv.lock            a262860cfc50bf7eddf05d5e2592c75cadf1f5993ea561ec40dcc121b146c59c  == 47d6dd3
configs/base.toml  c69ccbcb84c4f6b3197e7749e384146c2ccbee7afe8d67555cf3e3ccce88faa5  == 47d6dd3
```

All 31 tracked Phase 1–10 artefacts — manifests, policy, reports, experiment
records, figures and the pipeline binary — are byte-for-byte identical before and
after this phase, and all five upstream `--verify` gates still pass.

### Deployment footprint limitation

The serving environment carries the root project's analysis dependencies —
matplotlib, seaborn, scipy — because it depends on the root project and the root
project declares them. A production inference container has no use for them.

Removing them would mean adding optional-dependency groups to the root
`pyproject.toml`, which would change its digest and fail `freeze_model.py --verify`.
This is recorded as a **deployment footprint limitation of this serving version**,
not as a problem with the model: nothing about the frozen artefact, the decision
policy or the served prediction is affected. Resolving it requires a deliberate
re-freeze, which is out of scope here.

---

## 14. Security and privacy

| Property | Status |
|---|---|
| Extra fields | forbidden (`additionalProperties: false` in the published schema) |
| `customerID` accepted | **no** — not a field; there is no identifier in the process to leak |
| Body / batch bounded | yes, 500 records by default |
| Model path from a request | no — trusted process configuration only |
| Policy SHA pin overridable at runtime | **no** — a source constant, not a setting |
| Pickle path from a client | no |
| Traceback exposed | no |
| `eval` / `exec` / dynamic import | no — static audit, including `import_module` |
| `pickle` / `dill` / `marshal` imported | no — static audit |
| Payload logged | **no** |
| Probability logged | **no** |
| Prediction persisted | **no** |
| Repository written to during serving | **no** — hashes and directory listings compared before/after |

Operational logs carry shape only: method, path, status code, latency, batch size.
A test scores a record and asserts that neither its feature values nor its
probability appear anywhere in the captured log output.

The threshold and the calibration policy are **not** deployment settings. There is
no environment variable for either, and setting one — `CHURN_THRESHOLD`,
`CHURN_SERVING_CALIBRATION` and nine other spellings — **fails the startup**.
Silently ignoring `CHURN_THRESHOLD=0.5` would be worse than rejecting it: the
operator would believe the threshold had changed, and every decision would be
attributed to a rule nobody applied.

---

## 15. Tests and quality gates

Two suites, in two different environments. They are reported separately because they
run in separate environments, and a combined number would hide that.

| Suite | Command | Result |
|---|---|---|
| Root (Phases 1–10) | `uv run pytest -rs` | **1007 passed** |
| Serving (Phase 11) | `uv run --project serving pytest -rs serving/tests` | **246 passed** |
| Combined (informational) | — | 1253 |

The serving suite is 11 modules:

| Module | What it establishes |
|---|---|
| `test_settings.py` | operational config works; a frozen decision cannot be overridden from the environment |
| `test_runtime.py` | this environment *is* the frozen one; any divergence fails closed |
| `test_artifacts.py` | every startup gate, provoked one at a time; every fail-closed scenario |
| `test_provenance.py` | the source classification, re-derived by trace; the gate, per file; the `TotalChargesCleaner` case |
| `test_inference.py` | `>=` at the boundary; `prepare_features` mandatory; blanks; structural zero |
| `test_service.py` | the service without a server; exact endpoint invariance; order; batch limit |
| `test_api.py` | every endpoint, every rejection, determinism, key order, unseen category |
| `test_openapi.py` | 19 features, no `customerID`/`Churn`, no closed enums, no invented ranges |
| `test_isolation.py` | behavioural: no `fit`, no dataset, no `pipeline.predict`, one load, no writes |
| `test_static_audit.py` | AST: the forbidden calls are not in the source, on any path |
| `test_record.py` | the record is reproducible and every claim it makes names a real test |

**Behavioural and static are both present because neither replaces the other.** A
behavioural test cannot cover a branch that only fires in production; a static audit
cannot see through an indirection.

### The README / Ruff audit

Phase 10 reported `ruff format --check .` → `130 files already formatted`. Phase 11
observed a failure on `README.md`. The discrepancy was investigated rather than
patched over, and it resolves cleanly.

| Question | Answer |
|---|---|
| Root ruff version | `ruff 0.16.2` (pinned in root `uv.lock`) |
| Serving ruff version | `ruff 0.16.4` (pinned in `serving/uv.lock`) |
| Which ruff produced the observation | **both** — the failure reproduces identically under each |
| `README.md` working tree, raw SHA-256 | `b5465cbb5213fdebe4bb321422a4b8c3aba7c072a9b57b8cc7cd9e52f3b4ff37` |
| `README.md` working tree, LF-normalised | `939cb619c58000ab9809563d99e16a889dcf4b43d1507f70c891d77199b41051` |
| `README.md` at `HEAD` (`82b68ba`) | `939cb619c58000ab9809563d99e16a889dcf4b43d1507f70c891d77199b41051` |
| `README.md` at `47d6dd3` (Phase 10) | `3726f1f9aefbe08c9a1cda7ffca439c814c542e95ccb3a117bd22f19c83b02a6` |
| Working tree vs `HEAD` | **byte-identical in content**; the checkout is CRLF, the blob is LF, and `git diff HEAD -- README.md` is empty |
| Working tree vs `47d6dd3` | **different** — the file changed after Phase 10 |
| Exact failing command | `uv run ruff format --check README.md` |
| Exit code | `1` |
| Exact message | `unformatted: File would be reformatted` at `README.md:136:71`, reformatting the Python block: a `json.loads(...)` line rewrapped, two aligned trailing comments collapsed to a single space, and a long `print(f"...")` split across lines |
| Same content in a temp file | same failure, same exit code — under **both** ruff versions |
| `47d6dd3` content in a temp file | **passes**, exit `0`, under both ruff versions |
| Whole tree at `47d6dd3` (git worktree) | `130 files already formatted`, exit `0` |
| Whole tree at `HEAD` (git worktree) | `1 file would be reformatted, 129 files already formatted`, exit `1` |

**Conclusion: the Phase 10 report was correct, and this is not a Phase 11
regression.** The debt was introduced by commit `82b68ba`
(*docs: update README to the frozen-model state*) — a documentation commit that lands
**after** the Phase 10 tip `47d6dd3` and **before** Phase 11 began. It added a Python
example block that `ruff format` would rewrap. The file count is unchanged at 130 in
both worktrees; the difference is entirely one file's contents.

Neither the ruff version nor the CRLF checkout is the cause: the failure reproduces
under 0.16.2 and 0.16.4 alike, and on both the CRLF working copy and the LF blob.

`README.md` was **not** touched. It is a Phase 1–10 deliverable, fixing it is not in
this phase's scope, and doing so to obtain a green check without review is exactly
what the audit was asked to prevent. It is carried as **proven pre-existing
formatting debt**, `LOW`, and it is the only reason
`uv run ruff format --check .` exits non-zero.

### Gate results

| Command | Exit code | Output |
|---|---|---|
| `uv run python scripts/build_split.py --verify` | `0` | manifest reproducible |
| `uv run python scripts/freeze_model.py --verify` | `0` | all 29 integrity checks passed |
| `uv run python scripts/evaluate_holdout.py --verify` | `0` | AP 0.6337, ROC-AUC 0.8420 |
| `uv run python scripts/run_error_analysis.py --verify` | `0` | TN 803 / FP 232 / FN 104 / TP 270 |
| `uv run python scripts/run_model_interpretation.py --verify` | `0` | 19 raw, 46 transformed |
| `uv run --project serving python scripts/build_serving_record.py --verify` | `0` | reproducible, every invariant holds |
| `uv run pytest -rs` | `0` | 1007 passed |
| `uv run --project serving pytest -rs serving/tests` | `0` | 246 passed |
| `uv run ruff check .` | `0` | All checks passed |
| `uv run ruff format --check .` | **`1`** | **`README.md` only** — pre-existing, proven above |
| `uv lock --check` | `0` | Resolved 78 packages |
| `uv run --project serving ruff check .` | `0` | All checks passed |
| `uv run --project serving ruff format --check .` | **`1`** | **`README.md` only** — same file, same cause |
| `uv lock --check --project serving` | `0` | Resolved 44 packages |

Every Python file added by Phase 11 passes `ruff check` and `ruff format --check` in
both environments. The only non-zero exit is the historical README, and it is not
hidden.

---

## 16. Limitations

1. **Batch throughput.** Canonical row-wise scoring gives up the vectorisation a
   single N-row matrix product would provide. No benchmark is claimed, and the cost
   is not zero. Accepted deliberately for exact endpoint invariance (§9), on a small
   model with a bounded batch.

2. **Batch validation reports the first offending record**, not all of them, because
   validation is not duplicated. The error type and code are unchanged and the
   record's index is included in the message.

3. **Deployment footprint** (§13): the serving environment carries matplotlib,
   seaborn and scipy transitively. Recorded as a limitation of this serving version;
   resolving it requires a re-freeze of the root manifests.

4. **The policy SHA pin ties this serving version to one frozen policy** (§2). A
   future re-freeze requires a deliberate serving-code update. Intended, not a defect.

5. **`FREEZE_COMMIT` is a constant in the serving source**, not a policy field. A
   freeze record cannot contain the hash of the commit that writes it. A deliberate
   re-freeze must move this constant, and nothing automated enforces that.

6. **Three recorded source files are not gated** (§5). That is the traced,
   tested conclusion that they cannot reach a prediction — not an omission — but it
   is a narrower guarantee than gating all eight, and it depends on the trace staying
   accurate. `test_provenance.py` is what keeps it accurate.

7. **Importing `churn.serving.artifacts` transitively imports** `churn.data.loader`
   and `churn.preprocessing.splitting`, because `churn.modeling.freeze_results`
   imports `churn.modeling.tuning`. That is an import-graph fact of Phases 1–9, and
   *importing* is not *loading*: the behavioural guard proves no loader is ever
   called. Re-implementing the policy schema inside the serving layer to avoid the
   import would duplicate the contract and invite drift, which is worse.

8. **A `DeprecationWarning` from `joblib.numpy_pickle`** (`Setting the shape on a
   NumPy array has been deprecated in NumPy 2.5`) is emitted on every pipeline load.
   It comes from joblib 1.5.3 against numpy 2.5.1 — both pinned by the freeze — so it
   cannot be fixed without changing the frozen environment. No serving code triggers
   it and no deprecated FastAPI or Starlette API is used.

9. **`README.md` formatting debt**, proven pre-existing in §15. `LOW`. Untouched.

10. **No load testing, no concurrency testing, no TLS, no auth, no rate limiting.**
    The service is single-process and synchronous. Nothing here has been validated
    under production traffic — which also means the throughput trade-off in (1) has
    not been measured under load.

---

## 17. Running it locally

```bash
# using the configured host and port
uv run --project serving python -m churn.serving

# or with uvicorn directly, via the app factory
uv run --project serving uvicorn churn.serving.api:create_app --factory --host 127.0.0.1 --port 8000
```

Interactive contract at `http://127.0.0.1:8000/docs`.

Operational settings, all optional, none of which can move a frozen decision:

```text
CHURN_SERVING_HOST             default 127.0.0.1
CHURN_SERVING_PORT             default 8000
CHURN_SERVING_MAX_BATCH_SIZE   default 500, ceiling 5000
CHURN_SERVING_POLICY_PATH      default reports/decision_policy.json
CHURN_SERVING_MODEL_PATH       default: the path the policy records
```

Regenerate the machine-readable record:

```bash
uv run --project serving python scripts/build_serving_record.py [--verify]
```

---

## 18. Operational boundary with Phase 12

Phase 11 stops where observation begins. It deliberately contains **no** drift
metric, no PSI or KS computation, no feature-distribution baseline, no prediction-
drift dashboard, no Prometheus business metric, no alert rule and no retraining
trigger.

The seams Phase 12 will need already exist and are clean:

* a structured access log carrying shape only, with no payload to strip out later;
* a `model_fingerprint` on every response and in the metadata endpoint, so an
  observation can be attributed to a specific frozen model;
* an unseen-category path that is *known to be reachable* — the Phase 11 test proves
  only that such a record is **not blocked**. Whether the resulting score should be
  trusted, and how often such records arrive, is exactly the Phase 12 question. An
  unseen category is encoded as an all-zero block, which is a silent degradation
  Phase 11 can detect nothing about;
* an uncalibrated score, documented as such, so that any future calibration
  monitoring starts from an accurate description of what is being monitored;
* a probability that is now a property of the record alone, which is what makes a
  drift baseline comparable across differently batched traffic.
