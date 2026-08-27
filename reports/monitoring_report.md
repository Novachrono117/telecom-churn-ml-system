# Phase 12 — Monitoring, Data Quality & Drift Detection

Machine-readable record: [`reports/experiments/monitoring_results.json`](experiments/monitoring_results.json)
Reference profile: [`reports/monitoring/reference_profile.json`](monitoring/reference_profile.json)

---

## 1. Scope

Phase 12 answers one question: **do the records arriving in production still look
like the population this system was built for?**

| | |
|---|---|
| Trains a model | no — `fit_calls = 0`, asserted behaviourally |
| Moves the threshold or recalibrates | no — both read from the frozen policy |
| Reopens the holdout | no — unreachable from every monitoring module |
| Looks for a better model | no |
| Persists a payload, an identifier or a row-level score | no |
| Claims model performance degraded | **no — and cannot, without labels** |
| Adds a dependency | no — numpy, pandas, pydantic, standard library |
| Ships a dashboard, Prometheus, or an SaaS integration | no |

What it adds: a frozen reference profile, six families of drift and quality signal,
a privacy-safe incremental collector, window semantics with an explicit minimum, an
opt-in serving endpoint, three scripts and 130 root tests.

---

## 2. Monitoring versus model evaluation

Four phenomena, kept apart everywhere in the code, the record and the API response.
Conflating them is the classic monitoring failure, and this phase is largely an
exercise in not doing it.

| Phenomenon | Question | Answered here | How |
|---|---|---|---|
| **Data quality** | Is the input well formed? | yes | counters: schema rejections, contract rejections, blanks, structural violations, unseen categories |
| **Data drift** | Has `P(X)` moved? | yes | PSI per numeric feature, TVD per categorical |
| **Prediction drift** | Has the score distribution moved? | yes | PSI over the score, change in predicted-positive rate |
| **Performance degradation** | Has the model got worse? | **NO** | requires production ground truth, which does not exist |

**The fourth row is the point of the phase.** Without labels, no accuracy, recall,
precision, F1, ROC-AUC, average precision or confusion matrix can be computed — and
none is, anywhere. `test_no_label_based_metric_is_computed` enforces this statically:
no metric function is imported into the monitoring package, no metric name is
written in it, and `sklearn.metrics` is a forbidden import.

Every monitoring response carries this sentence:

> Drift means the population stopped resembling the reference. It is **NOT** evidence
> that the model degraded: that is a claim about outcomes and requires ground-truth
> labels, which this phase does not have. A change in `predicted_positive_rate` means
> the system is assigning positive decisions to a different share of requests, not
> that a different share of customers is churning.

A window can be `CRITICAL` on every signal while the model still makes exactly the
decisions the business wants; it can be `OK` on all of them while the model quietly
rots. Neither statement is available from what is measured here.

---

## 3. Reference population

```text
reference_population   training_pool
n_reference            5634
holdout_used           false
target_used            false
random_seed            42
```

**The training pool, not the holdout.** The holdout was consumed by Phase 9D. An
operational reference should describe the population the system was built for, and
the pool is both larger (5 634 vs. 1 409) and the partition the model actually saw.
Using the holdout would also spend, for a monitoring baseline, a partition reserved
for one thing.

**The target was never read**, and not merely by convention:
`build_reference_profile` has no parameter through which a label could arrive, its
body never names `Churn`, and the frame it receives is immediately reduced to the 19
contracted features. `test_the_target_is_never_read` asserts all three.

**No monitoring module can load anything.** The offline script loads the pool and
*hands it in*, exactly as Phase 9C's freeze modules required. An AST test asserts
that `load_holdout`, `load_split`, `load_training_pool` and `read_csv` appear in no
module of `churn/monitoring/`.

---

## 4. Reference profile

```text
reference_profile_sha256  6fef9f81a6dd03c94d4f2b7ebbd143b24c4e225eb2c6001ee1be68c4db0c531b
training_ids_sha256       a553196dd46b672f6344867707144fbf56a8abe14b338a225662c7208a1450dd
model_fingerprint         a57568e18ec0333e1c85b590f98e75da93ebbfafff063860ec5daf4e460d939f
pipeline_sha256           574fde36c6e2e991de7c3dfdab21981dc41eae2810d003ecbfb179dd504dc3d8
decision_policy_sha256    bd7aa800ae36ac04ecf951f7377ce71d16d084faeee771b01eb6c9655a4bc714
freeze_commit             9d1db4962769093a617f1db2db66354536acffb3
serving_commit            71a7f3a785c13a052e4f8ab3a6780842de60110a
```

The digest is computed from the profile's **value** under a canonical serialisation,
so a profile built in memory and one read back from disk hash identically.

### The trust anchor

Computing the digest of the file that was just loaded and then logging it is **not**
integrity validation. It answers *what did I load*, never *did I load the right
thing*. A digest becomes a gate only when it is compared against an expectation
recorded independently of the file being checked.

That expectation is pinned in source:

```text
expected_sha           6fef9f81a6dd03c94d4f2b7ebbd143b24c4e225eb2c6001ee1be68c4db0c531b
actual_sha             SHA-256 of the canonical serialisation of the profile loaded
source_of_expected_sha churn.monitoring.settings.EXPECTED_REFERENCE_PROFILE_SHA256
```

`ServingMonitor.from_settings` passes the constant to `verify_reference_profile` as
`expected_digest`, so `profile_digest_matches` is checked alongside every structural
invariant. On mismatch it raises `MonitoringStartupError`, the process does not come
up, and **monitoring never reaches `HEALTHY` on an unpinned baseline** — the same
fail-closed posture the frozen model gets.

`test_a_reference_whose_digest_is_not_the_pinned_one_stops_startup` proves it with
the hard case: a profile edited in a way that breaks *no* invariant — a note appended
— which every structural check would pass. Only the digest moves, and startup fails.

Both sides are reported by the endpoint, as `reference_profile_sha256`,
`expected_reference_profile_sha256` and `reference_profile_sha256_source`, so nobody
reading a monitoring response has to guess what the comparison was against.

Rebuilding the reference is therefore deliberately a two-line change: the artefact
and the constant move together in one commit, or the process refuses to start.
`decision_policy.json` is untouched by any of this.

The profile SHA is deliberately **not** mixed into `decision_policy.json`. That file
is Phase 9C provenance and stays byte-identical.

**Aggregates only.** Counts, order statistics, fixed bin edges, level frequency
tables and a score histogram. No row survives, and none could be reconstructed: the
largest list in the file is a ten-element histogram.

**Deterministic.** No timestamp, no hostname, no run id.
`build_monitoring_reference.py --verify` rebuilds it in memory and compares it byte
for byte against the file — reproduced exactly on every run.

---

## 5. Numeric drift

Three features: `tenure`, `MonthlyCharges`, `TotalCharges`. Recorded per feature:
`count`, `mean`, `std`, `min`, `q01`, `q05`, `q25`, `median`, `q75`, `q95`, `q99`,
`max`, plus fixed bin edges and counts.

### Binning

**Computed once, from the reference, and never again.** A window that recomputed its
own bins would compare two different partitions of the real line, and the resulting
"drift" would partly measure the binning.

* **Strategy:** deciles of the reference — nine interior edges from evenly spaced
  quantiles.
* **Interval convention:** `(-inf, e₁], (e₁, e₂], …, (e_{k-1}, +inf)`. Left-open,
  right-closed, resolved by `numpy.searchsorted(side="left")`.
* **Totality:** the outermost bins are unbounded, so every finite value falls in
  exactly one bin. No mass is ever lost or clipped into a neighbour.
* **Duplicate edges are dropped.** A concentrated feature — `tenure` has a large
  spike at its minimum — yields repeated quantiles, and a zero-width bin can never
  receive a value while still contributing a term to a PSI sum.
* **Values beyond the reference range** are *not* hidden inside the extreme bins.
  They are reported separately as `out_of_reference_range_rate`, computed against the
  reference `[min, max]`, so "unseen territory" is a named signal.

### Metric — Population Stability Index

```text
PSI = Σ_i (actual_i - expected_i) * ln(actual_i / expected_i)
```

over the fixed reference bins, with `actual_i` and `expected_i` the *shares* of each
population in bin `i`.

**Zeros.** The formula is undefined when a bin is empty, and empty bins are normal.
Every share is floored at **ε = 1e-6** before the logarithm. The consequence is
stated rather than hidden: one empty bin against a decile of reference mass
contributes `(1e-6 − 0.1) · ln(1e-6 / 0.1) ≈ 1.15`, already far past any cutoff. That
is intended — a bin that emptied out *should* dominate — but it means a PSI over a
window with several empty bins is driven by the epsilon, not by the data. This is
exactly why no drift verdict is claimed below the minimum window size.

**PSI is not a test statistic.** No null distribution, no p-value, no significance
level. It is symmetric, and zero only when the two histograms are identical.

### Also reported per numeric feature

* `mean_shift_in_reference_sd` — signed, `(actual_mean − reference_mean) / reference_std`,
  so a shift in `tenure` and one in `MonthlyCharges` are comparable. Returns `0.0` for
  a constant reference rather than manufacturing a large number from a degenerate
  feature.
* `out_of_reference_range_rate` — share outside the reference `[min, max]`.

---

## 6. Categorical drift

Sixteen features. Recorded per feature: `known_levels`, `count_by_level`,
`frequency_by_level`, and a `__UNSEEN__` bucket with reference count `0` — kept in
the table so the reference and a window always share a support.

### Metric — Total Variation Distance

```text
TVD = 0.5 * Σ_c |p_actual(c) - p_reference(c)|
```

over the known levels plus `__UNSEEN__`. Bounded in `[0, 1]`: zero when the two
distributions agree exactly, one when their supports are disjoint.

Chosen over a log-based divergence because **no epsilon is needed** — the formula is
defined at zero — so an empty level cell does not distort it the way it distorts PSI.
Categorical features have many rarely-populated levels, which is where that matters.

**TVD is a distance, not a probability.** "TVD = 0.3" does not mean a 30 % chance of
anything.

---

## 7. Unseen-category monitoring

The Phase 11 API accepts a new, non-blank category, and the frozen encoder absorbs
it via `handle_unknown="ignore"`. Phase 12 does not change that — it makes it
**visible**.

Per feature: `unseen_count`, `unseen_rate`, `n_distinct_unseen_observed` (with its
two qualifiers, below). Globally: `unseen_category_records`.

**The values themselves are never stored.** Cardinality is maintained with a set of
SHA-256 digests: enough to recognise that the same new category recurred. Keeping the
strings would be unbounded in cardinality and would be customer-supplied free text —
the two properties that turn a monitoring store into a data-protection incident.

**The SHA-256 is not anonymisation, and is not claimed to be.** A categorical feature
has a small, guessable domain, so an unsalted digest of one of its values is reversed
by hashing the candidates — a rainbow table with a few dozen entries. What the digest
buys is narrower: the cleartext is never retained as a string, so it cannot reach a
heap dump, a debugger's locals or a log line by accident.

The property the design actually relies on is **containment**, not the hash:

| Property | Status |
|---|---|
| `unseen_values_persisted` | **false** |
| `unseen_value_digests_exposed` | **false** — the endpoint returns `len()` and nothing else |
| `unseen_value_digests_persisted` | **false** — not in the profile, the record, a report or a snapshot |
| `unseen_value_digests_logged` | **false** |

Because the digests never cross a boundary, a keyed or salted construction would
harden nothing that is exposed; it would add a key to manage against a threat model
with no reader in it. If a later phase ever needed to publish or persist unseen-value
identity, the unsalted digest would be inadequate and the design would have to change
with it. `test_no_unseen_digest_reaches_a_snapshot_a_report_or_a_log` checks all four
surfaces at once; `test_every_digest_in_a_committed_artefact_is_a_named_artefact_fingerprint`
checks the committed files by shape, so an escaped digest would be caught as an
unexplained 64-hex string rather than only as a known one.

### Memory safety — a different problem from privacy

Privacy is **containment**: the values are never retained and the digests never
leave the process. That property holds at any size. It says nothing about *how many*
digests there may be, and that turned out to be the real remaining risk.

The serving contract accepts an unseen category on purpose. So traffic of the form

```text
value_000001
value_000002
…
value_1000000
```

would have grown the digest set without limit — inside a process that is stateful in
memory, whose window has no automatic rotation, whose `reset()` is an internal
administrative primitive, and whose monitoring endpoint is unauthenticated. A
component whose job is to observe must not be the thing that exhausts what it
observes, however private its contents are.

**The set is now bounded**, by
`max_distinct_unseen_tracked_per_feature = 1024` in `configs/monitoring.toml`.

| Phase | `unseen_count` / `unseen_rate` | `n_distinct_unseen_observed` | `distinct_unseen_tracking_saturated` | `distinct_unseen_is_lower_bound` |
|---|---|---|---|---|
| Below the cap | exact | **exact** | `false` | `false` |
| At or above it | exact | **lower bound** | `true` | `true` |

> Distinct unseen cardinality is exact until the configured tracking cap is reached.
> After saturation, the reported value is a lower bound; occurrence counts and unseen
> rates remain exact.

Three design points worth stating:

**Occurrence counting never saturates.** `unseen_count` is incremented *before* the
cap is consulted, so a record is never lost to the bound. `unseen_rate` is a ratio of
two exact integers. That matters because it is the quantity the alert policy grades —
**no status is derived from the distinct count**, so saturating the diagnostic cannot
move a verdict in either direction. `test_saturation_never_moves_an_alert_status`
runs the same traffic at cap 4 and cap 100 000 and requires identical overall status,
identical section statuses, and identical per-feature unseen rates and statuses.

**Nothing is evicted to make room.** An evicting set would report a number that is
neither the true cardinality nor a bound on it, and `1024` would quietly mean
something different in every window. Saturation refuses new entries instead.

**The bound is per feature and per window.** Saturating `Contract` does not blind the
tracker to `PaymentMethod`, and `reset()` clears every digest set and every saturation
flag — saturation is a property of one window, never of the process.

**What the cap guarantees, stated precisely.** The configured cap bounds each
categorical feature to at most 1,024 retained unseen digests per monitoring window.
That is a bound on the *number of retained entries*, and it is the property that
matters: collector state stops growing with traffic and becomes a function of
(features × cap) rather than of what a caller sends.

**Actual Python heap usage is implementation-dependent and larger than the raw digest
payload — `str` objects, the set's hash table, references and allocator behaviour all
contribute — and it was not benchmarked in this phase.** No figure for it is claimed
here or anywhere else in the repository. What is claimed, and tested, is that the
retained state is bounded and does not scale with traffic; for a deployment of this
size that state is small.

**Why 1024 rather than a smaller number.** It is two orders of magnitude above any
plausible legitimate value. The widest reference feature has four known levels, so a
window carrying a thousand distinct new ones is already a finding on its own — at
which point the exact cardinality has stopped being the actionable part. It is an
operational choice, tunable in `configs/monitoring.toml`, and deliberately **not** in
`configs/base.toml`: a memory limit has no business invalidating a model freeze.

Note that below the minimum window size none of `unseen_count`, `unseen_rate`,
`n_distinct_unseen_observed` or the saturation flags is reported at all — see §10. The
global `unseen_category_records` counter is what a small window gives an operator.

The operational meaning is worth stating plainly: an unseen category is encoded as an
**all-zero block**, so the record is scored as if that feature had never been sent.
The prediction is produced, no error is raised, and the only way to know it happened
is this counter.

---

## 8. Structural consistency

The seven product equivalences Phase 10 verified. Re-verified on the reference at
build time: **0 violations across 5 634 records**, which is what makes them usable as
data-quality rules.

```text
InternetService=No  <->  OnlineSecurity     = No internet service
InternetService=No  <->  OnlineBackup       = No internet service
InternetService=No  <->  DeviceProtection   = No internet service
InternetService=No  <->  TechSupport        = No internet service
InternetService=No  <->  StreamingTV        = No internet service
InternetService=No  <->  StreamingMovies    = No internet service
PhoneService=No     <->  MultipleLines      = No phone service
```

Per window: `violations_by_rule`, `records_with_any_violation`, `violation_rate`.
A record breaking six internet rules counts as **one** affected record — summing the
per-rule counts would overstate how much traffic is affected.

**Watched, not enforced.** Phase 11's schema accepts a violating record, and Phase 12
does not retroactively make it invalid: the record is still scored, still returned,
and counted. Tightening input validity after an API has been published is a breaking
change dressed up as a bug fix.

The rules are declared in `churn/monitoring/structural.py` rather than imported from
`churn.modeling.interpretation`, which transitively pulls in the dataset loaders and
would widen the import surface of a package that runs inside the serving process.
`test_the_monitoring_rules_are_the_phase_10_rules` asserts the two declarations are
identical, rule for rule, so the duplication cannot drift.

---

## 9. Prediction drift

The frozen pipeline applied to the reference population, through the same primitive
the serving boundary uses — `churn.modeling.freeze.positive_probability`, positive
column resolved from `classes_`. `pipeline.predict` is not used here or anywhere in
the package.

Reference score distribution:

```text
score mean               0.265436
predicted_positive_rate  0.351793
threshold                0.3272694566222328   (frozen, from the decision policy)
comparison               >=
calibration_policy       NONE
```

Per window: score PSI over the fixed reference bins, `mean_delta`,
`predicted_positive_rate`, `predicted_positive_rate_delta` (signed — a rate that
halved and one that doubled are different situations).

**Interpretation, stated in the response itself.** A rise in
`predicted_positive_rate` means *the system is assigning positive decisions to a
different share of requests*. It does **not** mean more customers are churning. And
because `calibration_policy` is `NONE`, a score of 0.70 is a ranking position, not a
70 % chance of churn — the response carries that note too.

---

## 10. Window semantics

A window is a **counter generation**, not "everything since the process started"
and not a time series.

### The primitive

```text
MonitoringCollector.snapshot()   read the open window, leave it open
MonitoringCollector.reset()      snapshot AND reset, under one lock acquisition
MonitoringService.close_window() reset(), then compare the closing window
```

`reset()` is deliberately *snapshot-and-reset* rather than a bare clear: it returns
the snapshot of the window it closes, so nothing observed before the call is lost and
nothing observed after it is attributed to the old window. Snapshot-then-clear as two
operations would silently drop every update that arrived in between — data loss
inside the component whose job is to notice things.

`test_a_reset_closes_one_window_and_opens_an_empty_one` runs exactly the sequence the
phase asked for: window A of 100 records reports a verdict, the window is closed,
window B starts at `n = 0`, 25 records go in, and B reports `n = 25` with
`INSUFFICIENT_DATA`.

**Atomicity.** Every mutating path and every read takes the same `RLock`, and
`observe()` applies all of one call's updates inside a single acquisition — so an
observation lands entirely in the window that was open when it took the lock, never
split across a reset. `test_concurrent_observation_and_reset_lose_and_duplicate_nothing`
runs eight writers against a rotating resetter and asserts the arithmetic closes
exactly: no exception, no negative counter, and the closed windows plus the final open
one sum to precisely what was written. This is single-process consistency — one
collector, one address space, one lock. No distributed claim is made or needed.

**A reset is not reachable over HTTP.** Closing a window destroys the evidence a
drift investigation runs on, this service has no authentication, and an anonymous
caller able to do that is a worse trade than restarting the process. It changes no
model, no policy, no threshold, no prediction and no reference profile; a test
asserts the reference object is the same instance afterwards.

### Minimum window size

**100 records.** Below it the status is `INSUFFICIENT_DATA` — deliberately *not* the
lowest rung of the OK/WARNING/CRITICAL ladder, but the absence of a rung. No drift
verdict is claimed, and no detailed distribution is reported.

100 is chosen so that a decile histogram has roughly ten observations per bin on
average, and low enough that a modest deployment still produces a verdict in
reasonable time. It is an operational choice, documented in `configs/monitoring.toml`.

### Small-window privacy policy

This is a privacy control, and the first version of it was **not strong enough**.

The obvious leak is an order statistic: *a mean over one record is that record's
value*, and so are its min and its max. Those were withheld from the start. But the
review that followed found the same leak in a different shape — **any per-feature
breakdown of a one-record window is that record**:

| Output | What it reveals at `n = 1` |
|---|---|
| `count_by_level = {"Month-to-month": 1}` | the customer's contract, exactly |
| `bin_counts = [0, 1, 0, …]` | their tenure, to a decile |
| `unseen_count = 1`, `n_distinct_unseen_observed = 1` | that *their* category was the unknown one |
| score `bin_counts` | their score, to a decile |
| `predicted_positive_rate = 1.0` | their decision |
| `violations_by_rule = {…: 1}` | which product equivalence they broke |

None of those is an aggregate. Each is a single record wearing a histogram's
clothing, and a low-traffic deployment whose endpoint is polled after every request
would reconstruct customers one at a time.

**The policy is therefore a threshold, not a per-field judgement call:**

```text
n_records < minimum_window_size
  → status = INSUFFICIENT_DATA
  → details_suppressed = true
  → no detailed distribution is COMPUTED at all
```

Not suppressed after the fact — never built. `compare()` returns from
`_suppressed_report()` before any per-feature, per-level, per-bin or per-rule
quantity exists, so there is no code path on which a detail could survive into the
response by accident.

#### Visible below 100

| Field | Why it is safe |
|---|---|
| `status`, `section_status` | a verdict about the window, not a value |
| `n_records`, `minimum_window_size` | the window's size and the policy |
| `details_suppressed`, `small_window_note` | why the sections are absent |
| `reference_profile_sha256` and the two anchor fields | about the *reference*, not the traffic |
| `collector_health`, `collector_failures` | about the collector |
| `data_quality.requests_total` | |
| `data_quality.records_total` | |
| `data_quality.successful_records` | |
| `data_quality.invalid_schema_records` | |
| `data_quality.invalid_feature_records` | |
| `data_quality.unseen_category_records` | |
| `data_quality.structural_violation_records` | |
| `performance_degradation`, `interpretation`, `calibration` | static prose |

Those global counters describe the **deployment** — traffic is arriving, *n* of it was
rejected and in which class — and none can be narrowed to a feature, a level, a bin or
a score. Without them a suppressed window would be indistinguishable from a dead
process, which is its own operational failure.

#### Suppressed below 100

`feature_drift`, `prediction_drift` and `structural_consistency` are **`null`** —
explicitly null rather than empty or partial, so there is no shape to probe and a
reader cannot mistake absence for zero. That removes, in one stroke:

* numeric histograms and order statistics (`mean`, `std`, `min`, `max`);
* per-feature PSI, mean-shift and out-of-range rates;
* categorical `count_by_level` and level frequencies;
* per-feature `unseen_count`, `unseen_rate`, `n_distinct_unseen_observed` and the
  saturation qualifiers;
* the score histogram, its summary statistics and `mean_delta`;
* `predicted_positive_count` and `predicted_positive_rate`;
* per-rule structural details (`violations_by_rule`).

#### At and above 100

Everything above is reported normally. A histogram over a hundred records is an
aggregate, and 100 is also roughly the point at which a decile histogram has ten
observations per bin — the size at which these numbers become *interpretable* is the
size at which they stop being *identifying*, which is why one cut serves both.

#### Evidence

| Test | Window |
|---|---|
| `test_a_one_record_window_cannot_be_read_back_out_of_the_report` | n = 1 |
| `test_a_one_record_window_leaks_nothing_through_the_endpoint` | n = 1, over HTTP |
| `test_no_counter_in_a_one_record_window_can_be_narrowed_to_a_feature` | n = 1 |
| `test_the_window_just_below_the_minimum_is_suppressed_too` | n = 99 |
| `test_a_window_of_ninety_nine_is_suppressed_exactly_like_a_window_of_one` | n = 99, over HTTP |
| `test_at_exactly_the_minimum_the_distributions_appear` | n = 100 |

The n = 1 tests do not check named fields for named leaks. They build a window from a
record with unmistakable values, serialise the whole response, and walk it — every
key, every value, every list, at any depth — asserting that nothing in it identifies
the record, and that outside the global counters no numeric leaf is non-zero. A
field-by-field allow-list would pass the day someone adds a new distribution and
forgets to suppress it; a recursive sweep fails.

The same policy applies to `run_monitoring_check.py`. A file on an operator's disk is
not a safer place to print a one-record profile than an HTTP body.

---

## 11. Alert policy

`configs/monitoring.toml` — a **separate file**, deliberately not `configs/base.toml`,
which is Phase 9C provenance and whose digest is checked by
`scripts/freeze_model.py --verify`. An alert cutoff has no business invalidating a
model freeze.

| Signal | WARNING | CRITICAL |
|---|---|---|
| numeric PSI | 0.10 | 0.25 |
| numeric out-of-range rate | 0.01 | 0.05 |
| categorical TVD | 0.10 | 0.25 |
| categorical unseen rate | 0.01 | 0.05 |
| score PSI | 0.10 | 0.25 |
| \|predicted-positive-rate delta\| | 0.05 | 0.10 |
| structural violation rate | 0.001 | 0.01 |

The same file carries one setting that is **not** a cutoff:
`max_distinct_unseen_tracked_per_feature = 1024` (§7). It bounds how much state the
collector may retain, and nothing grades against it — there is no warning/critical
pair to call, and `test_saturation_never_moves_an_alert_status` proves the verdicts
are identical on either side of it. It lives here because an operator tunes it, not
because it decides anything.

### These numbers are operational policy, not statistics

Stated in the config file, in the module docstring, in `policy.as_record()` and in
the machine-readable record, which carries
`classification: OPERATIONAL_MONITORING_POLICY`,
`is_statistical_significance: false` and `estimated_from_data: false`.

They are **heuristics about when a human should look**, chosen before any production
data existed. The 0.10 / 0.25 pair for PSI is the value most quoted in credit-scoring
practice; it is reproduced because operators recognise it, **not** because it has been
validated for this dataset, this model or this traffic. It should be re-tuned once
real production windows exist.

The structural cutoffs are low on purpose: the equivalences hold *exactly* on the
reference, so any violation is worth noticing, and a violated equivalence is a
data-quality signal rather than a distribution shift.

`MonitoringPolicy` has no field for a threshold, a calibration policy, a positive
class or a feature list — asserted with an exact field-set test, so a future addition
forces a reviewer to look.

---

## 12. Serving integration

### Monitoring is opt-in, and that is a deliberate compatibility decision

`CHURN_SERVING_MONITORING=1`. **Default: off.**

Phase 11 published a serving contract and recorded its route list in
`reports/experiments/serving_results.json`. Enabling monitoring adds
`GET /api/v1/monitoring`, which changes that published surface. Making it opt-in means
deploying Phase 12 code **cannot silently alter the Phase 11 API contract** — and it
means the Phase 11 record was not rewritten to pretend monitoring had always been
there. `build_serving_record.py --verify` still passes byte for byte, and
`serving_results.json` is unchanged.

A later phase can flip the default deliberately, updating the serving record at the
same time. That is a decision with a diff, which is the point.

### Endpoints

| Method | Path | When |
|---|---|---|
| `GET` | `/api/v1/monitoring` | monitoring enabled |
| `GET` | `/health/ready` | always — now also reports `monitoring: HEALTHY / DEGRADED / DISABLED` |

**There is no HTTP reset.** Closing a window is an administrative operation, this
service has no authentication, and an unauthenticated endpoint that erases the
evidence a drift investigation depends on is a worse trade than restarting the
process. `MonitoringCollector.reset()` remains available as an internal primitive.

### Monitoring cannot change a prediction

The observer is called **after** the answer exists, from
`ChurnInferenceService._score`, and its return value is discarded. The `Prediction`
returned is fully determined by the line above the call.

Two tests establish it end to end: the same payload through a monitoring-off process
and a monitoring-on process returns **byte-identical** responses, single and batch.

The Phase 11 contracts are re-run under monitoring, not assumed to survive:
`single_batch_probability_contract = exact`, batch order preserved, the **34 startup
gates** still passing.

### Readiness is not a drift signal

Drift never makes the service NOT READY. Readiness is a claim about the **artefact** —
is the frozen model loaded and verified. A shifted population is not a corrupt model,
and conflating them would pull a healthy instance out of rotation because customers
changed. A test drives 150 drifted records through the boundary, confirms the
monitoring status goes `CRITICAL`, and confirms `/health/ready` still answers `200 ready`.

Artefact corruption still makes the service not ready. Both halves are tested.

---

## 13. Privacy

| Property | Status |
|---|---|
| Raw request payload persisted | **no** |
| `customerID` | not accepted by the schema; never in the process |
| Row-level features persisted | **no** |
| Row-level probability persisted | **no** |
| Row-level prediction persisted | **no** |
| Unseen category values persisted | **no** — process-local SHA-256 digests only |
| Unseen value digests exposed / persisted / logged | **no** — only a count derived from them leaves memory |
| Unseen digest retention bounded | **yes** — 1024 per feature per window (§7); a *memory* property, not a privacy one |
| Individual customer lists | **no** |
| Collector state grows with traffic | **no** — fixed-size counters |
| Distributions exposed below `minimum_window_size` | **no** — `details_suppressed = true` (§10) |
| Exception messages or tracebacks logged | **no** — event name and exception type only (§14) |

The strongest form of "no payload is persisted" is that there is nowhere to persist
one. `test_the_collector_keeps_no_record` enumerates the collector's entire state —
counters, count tables, running moments, bin vectors — and asserts its serialised size
is independent of how many records were observed.

Server logs carry method, path, status, latency and batch size. The smoke test greps
the live server's log for feature values and finds none.

Two leaks that a payload-focused audit misses are handled separately, because neither
involves the collector storing anything: a **small window's own distributions** (§10)
and a **monitoring failure's exception message** (§14).

**Privacy and memory safety are not the same property, and the report keeps them
apart.** Privacy is containment — digests stay in memory and are never exposed, which
holds at any set size. Memory safety is boundedness — the number of retained digests
is capped (§7), which would matter even if the contents were public. Neither
substitutes for the other, and `unseen_digest_is_anonymisation` stays **false**: an
unsalted SHA-256 of a small categorical domain is reversible by anyone who can read
it, and the design relies on there being no such reader.

### Cumulative counters and active differencing

The small-window policy prevents direct disclosure through distribution breakdowns,
but cumulative operational counters are **not a substitute for access control**. An
observer able to poll the unauthenticated monitoring endpoint around individual
requests may infer request-level operational events from counter deltas: polling
before and after a single request and seeing `unseen_category_records` rise by one
reveals that *that* request carried an unknown category, and the same differencing
works for `structural_violation_records` and the two rejection classes.

This is not fixed by hiding individual counters. Removing them would blind operators
to the only signal a low-traffic deployment has, and any remaining counter can be
differenced the same way — including `records_total`. The correct mitigation is
**authentication, authorisation or network isolation** on the endpoint, which the
unauthenticated-endpoint limitation (§17) already records and which this phase
deliberately does not implement. It is stated here so that nobody reads the
small-window suppression as a claim it does not make.

---

## 14. Failure isolation

**Policy: the prediction path is authoritative.** A monitoring failure costs an
observation, never an answer.

* the observer is called after the prediction exists;
* `ServingMonitor` catches, counts and logs any collector exception — the *inner*
  catch, whose job is to make the failure reportable;
* `ChurnInferenceService._observe` catches again — the *outer* catch, whose job is to
  guarantee the prediction. The two are not redundant: neither can be removed without
  losing something;
* the request still returns **200** with the correct prediction;
* the monitoring status turns `DEGRADED`, in `/health/ready` and in the monitoring
  endpoint, and stays there for the life of the window.

**Nothing is swallowed silently.** The failure is logged, and a test asserts the log
line is emitted. Choosing fail-open here is deliberate: failing a prediction because
a histogram could not be updated trades a correct answer for an observation, which is
backwards.

### What the failure log may contain

The first version logged a traceback. That was wrong, and the reason is worth being
precise about: **both guards run holding feature values and a probability**, so an
exception raised beneath them can carry either into its message. A traceback renders
`str(error)`, so

```text
ValueError: 'Instant transfer (PIX)' is not a known level
```

is the payload leaking through the observability layer — the one surface nobody
audits for it, and the one most likely to be shipped to a third-party log aggregator.

The log line is now a fixed event name, the stage, and the exception's **type**:

```text
monitoring_observer_failure stage=observe exception_type=RuntimeError failures=1
monitoring_observation_failure exception_type=RuntimeError; the prediction is unaffected.
```

Never `str(error)`, never `logger.exception`, never `exc_info`. The in-memory tally
follows the same rule: `ServingMonitor.failures.last_error` holds a class name, not a
message.

`test_a_collector_failure_never_logs_anything_derived_from_the_payload` makes the
observer raise `RuntimeError("SECRET_CUSTOMER_VALUE_ABC123")`, captures every log
record at `DEBUG` including `exc_text`, and asserts the string does not appear — while
the prediction is byte-identical to the monitoring-off response, still `200`, and the
monitoring status turns `DEGRADED`. Two more tests cover the rejection-counting path
and the in-memory tally.

The stack trace is traded, not lost. The failure is counted, the status degrades
visibly, and a developer reproducing the bug locally gets the full traceback from the
collector's own unit tests, where the records are synthetic.

Monitoring **startup**, by contrast, is fail-closed: an absent, unparseable or
tampered reference profile stops the process — including tampering that breaks no
invariant, because the digest is checked against a pinned constant (§4). A drift
number computed against an unverified baseline is worse than no number, because it
looks like one.

---

## 15. Tests

| Suite | Command | Before | After |
|---|---|---|---|
| Root | `uv run pytest -rs` | 1007 | **1170** (+163) |
| Serving | `uv run --project serving pytest -rs serving/tests` | 246 | **293** (+47) |
| Combined (informational) | — | 1253 | **1463** |

Root modules added:

| Module | What it establishes |
|---|---|
| `test_monitoring_reference.py` | population, no target, no holdout, n = 5634, provenance, determinism, tamper detection |
| `test_monitoring_selfcheck.py` | the reference does not drift from itself — every distance exactly zero |
| `test_monitoring_drift.py` | PSI and TVD properties; synthetic numeric, categorical, unseen, structural and score drift; window minimum |
| `test_monitoring_collector.py` | counters, privacy, and thread safety under 8 concurrent writers |
| `test_monitoring_policy.py` | the policy cannot reach the model; cutoffs labelled heuristic |
| `test_monitoring_structural.py` | the seven rules equal Phase 10's, rule for rule |
| `test_monitoring_isolation.py` | no fit, no dataset, no `pipeline.predict`, **no label-based metric** |
| `test_monitoring_record.py` | the record is reproducible and every claim names a real test |
| `test_monitoring_privacy.py` | a one-record window cannot be read back out of the report; the n = 1 / n = 99 / n = 100 boundary; unseen digests never leave memory |
| `test_monitoring_window.py` | the snapshot-and-reset primitive, a fresh window, and atomicity under concurrent rotation |
| `test_monitoring_cardinality.py` | the distinct-unseen set is bounded, saturation is labelled, occurrence counts stay exact, and no status moves with the cap |

Serving module added: `test_serving_monitoring.py` — ON/OFF prediction equivalence,
exact single/batch under monitoring, batch order, the endpoint, counters, privacy,
readiness separation, failure isolation, fail-closed reference startup, the pinned
reference digest, sanitised failure logging, the small-window suppression policy over
the real HTTP boundary, and bounded unseen-cardinality reporting.

### The reference self-check

The strongest single test in the phase. The reference population is scored and pushed
through the **production** path — the collector, not the builder — and every distance
must be exactly zero:

```text
n_records                        5634
max numeric PSI                  0.0
max out_of_reference_range_rate  0.0
max categorical TVD              0.0
max unseen rate                  0.0
score PSI                        0.0
predicted_positive_rate delta    0.0
structural violations            0
```

It is strong because the two sides are computed by *different code*: the reference
histogram by a whole-column `searchsorted`, the window histogram by a running counter
accumulating batch by batch. A disagreement about an edge, an inclusivity convention,
a dtype or the treatment of `TotalCharges` would show up here — and a monitoring
system whose reference does not match itself would report drift that is not there.

---

## 16. Delayed-label monitoring

Not implemented, and deliberately not stubbed with anything that executes.

Churn labels arrive late: whether a customer scored today churns is knowable only
after the observation window the business defines has passed. When that feedback
exists, the metrics that become available are the ones Phase 9D already used on the
holdout, applied to a labelled production window:

* **Average Precision** and **ROC-AUC** — ranking quality, threshold-free;
* **Precision, Recall, F1** at the frozen threshold — what the decision rule actually
  achieved;
* **confusion matrix** — the operational shape of the errors;
* **calibration diagnostics** — a reliability curve, and whether the Phase 9A decision
  to leave the score uncalibrated still holds.

Three things would have to be settled before any of that is trustworthy, and none is
a coding problem:

1. **The label definition and its horizon** — "churned within N days of scoring" — must
   match what Phase 2 defined, or the numbers describe a different target.
2. **Joining scores to outcomes requires an identifier**, which this boundary
   deliberately does not accept or retain. A labelled-performance pipeline is
   therefore a *different* system with a different privacy posture, not an extension
   of this collector.
3. **Selection effects.** If a retention campaign acts on the model's positive
   predictions, the observed outcome for those customers is no longer the outcome the
   model predicted. Measuring performance on intervened traffic without accounting for
   that measures the campaign as much as the model.

Until those are answered, the honest report is `performance_degradation: {evaluated: false}`.

---

## 17. Limitations

1. **PSI over small or sparse windows is epsilon-driven.** An empty bin contributes
   ≈1.15 regardless of how much data is behind it. Mitigated by the minimum window
   size, not eliminated.

2. **The cutoffs are unvalidated heuristics.** No production data existed to tune
   them. They are labelled as operational policy everywhere, and they should be
   revisited once real windows accumulate.

3. **A schema rejection is counted as one record.** A body that failed to parse
   carries no record count, so a rejected batch of 50 is counted as one. The response
   carries a `rejection_counting_note` saying so.

4. **Windows are per-process and in-memory.** A restart loses the current window;
   several replicas each keep their own. There is no aggregation across processes, no
   persistence and no time-based rollover — a deployment that needs those needs a
   metrics backend, which is out of scope here.

5. **`InternetService` and its six add-ons drift together.** They are structurally
   linked, so a single upstream change lights up seven signals. That is faithful
   rather than wrong, but the signals are not independent and should not be counted as
   seven pieces of evidence.

6. **`TotalCharges` is monitored after the frozen cleaner runs**, so a structural zero
   enters the histogram as `0.0`. That is what the model sees, which is the right
   thing to monitor, but it means the numeric summary cannot distinguish "never billed"
   from "billed zero".

7. **No time dimension.** A window has no start, no end and no rate-per-hour. Counts
   are cumulative within a window, and `reset()` (§10) is the only thing that closes
   one. Nothing rotates windows on a schedule: an operator or an embedding process has
   to call the primitive, and it is not reachable over HTTP.

8. **Reference staleness has no expiry.** Nothing warns that a profile built against
   a population from months ago may itself be the thing that has moved. Deciding when
   to re-reference is a judgement this phase does not automate.

9. **The monitoring endpoint is unauthenticated**, like the rest of the service. It
   exposes aggregates only, and windows below the minimum expose no distribution at
   all, but a deployment on an untrusted network needs the same auth story the
   prediction endpoints need.

10. **No performance monitoring** — §2 and §16. This is a boundary, not a gap.

11. **A large window is still a disclosure surface, in principle.** Suppression below
    100 stops the one-record reconstruction; it does not make a 100-record histogram
    a formal privacy guarantee. There is no differential privacy here, no noise and
    no k-anonymity claim — a homogeneous window of 100 near-identical records would
    describe those records fairly precisely. Mitigating that properly means noise or a
    much larger minimum, and both trade away the operational usefulness the phase
    exists for. Stated rather than solved.

12. **`request_total` and `observe` are counted under separate lock acquisitions.**
    Each is individually atomic, but a request counted just before a reset can have
    its records observed just after it, so across a window boundary the global
    counters and the record count can disagree by the traffic in flight. It is a
    counter-attribution artefact of rotating a live window, not a lost update — the
    sums still close across all windows, which is what the concurrency test asserts.

13. **The unseen-value digest is not anonymisation** (§7). The design relies on the
    digests never leaving memory, which is tested on four surfaces, rather than on the
    hash being irreversible — it is not, for a small categorical domain.

14. **Distinct unseen cardinality saturates** (§7). Past 1024 distinct values in one
    window for one feature, the reported count is a lower bound and says so. The exact
    cardinality of an unbounded-cardinality attack is not recoverable from the report,
    by design; `unseen_count` and `unseen_rate` remain exact and are what the alert
    policy reads.

15. **Cumulative counters are differenceable** (§13). An observer polling the
    unauthenticated endpoint around a single request can infer request-level
    operational events from counter deltas. The small-window policy stops disclosure
    through distributions; it is not access control, and the correct mitigation is
    authentication or network isolation rather than hiding counters.

---

## 18. Engineering status

### Files

```text
configs/monitoring.toml                     operational policy, separate from the freeze
src/churn/monitoring/                       __init__, build, collector, distributions, drift,
                                            reference, results, service, settings, structural
src/churn/serving/monitoring.py             the serving-side observer and its fail-closed startup
scripts/build_monitoring_reference.py       build / --verify the reference profile
scripts/run_monitoring_check.py             offline window check, --input required
scripts/build_monitoring_record.py          build / --verify the Phase 12 record
reports/monitoring/reference_profile.json   the frozen baseline
reports/experiments/monitoring_results.json the machine-readable record
reports/monitoring_report.md                this document
```

Phase 11 files modified for the integration — each listed with its reason in the
delivery summary: `api.py` (opt-in route, counters, readiness field),
`service.py` (observer hook, failure isolation), `settings.py` (two operational
fields), `schemas.py` (monitoring response), and `serving/tests/test_settings.py`
(the pinned field-set test).

`reports/experiments/serving_results.json` was **not** modified.

### Running it

```bash
# build the reference (offline; loads the training pool, never the holdout)
uv run python scripts/build_monitoring_reference.py [--verify]

# check a window of records offline
uv run python scripts/run_monitoring_check.py --input <records.csv|.json> [--json]

# serve with monitoring enabled
CHURN_SERVING_MONITORING=1 uv run --project serving python -m churn.serving
curl http://127.0.0.1:8000/api/v1/monitoring

# regenerate the Phase 12 record
uv run python scripts/build_monitoring_record.py [--verify]
```

### Boundary with what comes next

Phase 12 produces signals. It does **not** decide what to do about them: there is no
retraining trigger, no automatic re-reference, no alert routing and no schedule. A
retraining criterion needs the labelled feedback loop of §16, and prescribing a
cadence without evidence would be exactly the kind of arbitrary rule this project has
avoided at every earlier phase.
