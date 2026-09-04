# Monitoring

Four questions that are routinely collapsed into one, kept apart everywhere in the code, the
record and the API response:

```text
Data quality  !=  Feature drift  !=  Prediction drift  !=  Performance degradation
```

| Signal | The question | Available without labels |
| --- | --- | --- |
| **Data quality** | Is the input well-formed? | yes |
| **Feature drift** | Does the input population still resemble the reference? | yes |
| **Prediction drift** | Is the system *deciding* differently? | yes |
| **Performance degradation** | Is the model still right? | **no** |

Production ground truth does not exist here, so the fourth is **never inferred**: no
accuracy, recall, precision, F1, ROC-AUC or confusion matrix is computed for live traffic
anywhere, and no drift signal is offered as a proxy for one. A static test enforces it —
`sklearn.metrics` is a forbidden import inside the monitoring package.

Source of truth: [monitoring_report.md](../reports/monitoring_report.md) ·
[`monitoring.toml`](../configs/monitoring.toml) ·
[`monitoring_results.json`](../reports/experiments/monitoring_results.json) ·
[`reference_profile.json`](../reports/monitoring/reference_profile.json)

## What is computed per window

| Signal | Measure |
| --- | --- |
| Numeric drift | **PSI** per feature over fixed reference bins, plus out-of-range share |
| Categorical drift | **TVD** over known levels plus an `__UNSEEN__` bucket |
| Unseen categories | Count and rate per feature — the frozen encoder absorbs them, so they are scored, not rejected |
| Prediction drift | **PSI over the model score**, signed `mean_delta`, and the change in **predicted-positive rate** |
| Structural checks | The seven product equivalences, violated only if an upstream system changed |

Two window rules shape everything above. **Minimum window size is 100**: below it the window
reports `INSUFFICIENT_DATA` rather than a number, because PSI over a sparse window is
epsilon-driven — an empty bin contributes ≈1.15 regardless of how much data is behind it.
And **aggregation is privacy-aware**: raw category values are never retained, only bounded
digests, capped at 1,024 distinct unseen entries per feature per window so that an
observability component cannot exhaust the process it observes. The cap bounds a diagnostic;
`unseen_count` and `unseen_rate` stay exact, and they are what drives the status.

## What the cutoffs are — and are not

> **PSI and TVD cutoffs are operational heuristics.** They are not hypothesis tests, they
> produce no p-value, none was estimated from data, and crossing one means *"this window no
> longer resembles the reference population by this much"* — never *"the model degraded"*.

The 0.10 / 0.25 PSI pair is the value most widely quoted in credit-scoring practice,
reproduced because it is familiar to operators, **not** because it was validated for this
dataset, this model or this traffic. It is a starting point to be re-tuned once real
production windows exist. All cutoffs live in
[`configs/monitoring.toml`](../configs/monitoring.toml), deliberately separate from
`configs/base.toml` — an alert cutoff has no business invalidating a model freeze, and
nothing in that file can reach the model.

## The demonstration window

The console screenshot ([`screenshots/monitoring-status.png`](screenshots/monitoring-status.png))
shows `WARNING`: 101 records, data drift `WARNING`, prediction drift `WARNING`, structural
consistency `OK`. That window is the first 101 scored observations of the frozen training
pool, taken deterministically in frozen order — **the reference population itself** — and it
crosses the heuristic cutoffs anyway, at PSI 0.155 on `tenure` and 0.152 on the model score.

What that demonstrates is a property of the cutoff, not a property of the data: an
operational threshold fixed before any traffic existed can fire on a small window drawn from
the population it was built from. It is **not** evidence of drift, and it cannot be evidence
of performance degradation, because nothing here measures that. It is published as captured
rather than replaced with a greener one, because it is exactly the kind of window that says
the cutoffs need re-tuning once real ones exist.

## Boundaries

- **Monitoring cannot change a prediction.** The observer sits off the causal path behind a
  read-only endpoint; it reads what was decided. It is opt-in, off by default.
- **Readiness is not a drift signal.** A drifting window does not make the service unready.
- **State is local and in-memory** — no persistence, no rotation, no alert routing.

## What labels would unlock

When outcomes arrive, the metrics that become available are the ones the holdout evaluation
already used: Average Precision, ROC-AUC, precision/recall/F1 at the frozen threshold, the
confusion matrix, and a calibration diagnostic. Three things would have to be settled first,
and none is a coding problem:

1. **The label definition and its horizon** — "churned within N days of scoring" must match
   the target this model was trained on, or the numbers describe something else.
2. **Joining scores to outcomes needs an identifier**, which this boundary deliberately does
   not accept or retain. That is a different system with a different privacy posture.
3. **Selection effects.** If a retention campaign acts on the model's positive predictions,
   the observed outcome for those customers is no longer the outcome the model predicted;
   measuring on intervened traffic measures the campaign as much as the model.

Until then the honest report is `performance_degradation: {evaluated: false}`, and that is
what it says. This layer produces signals and stops there: **no retraining trigger, no
automatic re-reference, no alert routing and no schedule.** A retraining criterion needs the
labelled feedback loop above, and prescribing a cadence without evidence would be an
arbitrary rule dressed as a policy.
