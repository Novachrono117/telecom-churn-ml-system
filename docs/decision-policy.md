# Decision policy

> **A probability is not a decision.** The model emits a score; turning it into an action
> is a separate, explicit choice — and `0.5` is a default, not a policy.

Source of truth: [`decision_policy.json`](../reports/decision_policy.json) ·
[threshold_report.md](../reports/threshold_report.md) ·
[calibration_report.md](../reports/calibration_report.md) ·
[`threshold_results.json`](../reports/experiments/threshold_results.json)

## What the model emits

A logistic regression produces `predict_proba(X)[:, positive_class_column]`, with the
column resolved from `classes_` rather than assumed. **Calibration policy is `NONE`** — the
outcome of a gate, not an omission: neither sigmoid nor isotonic calibration cleared the
pre-registered eligibility rule, so no calibration layer wraps the model. The practical
consequence: `0.70` is a ranking position, not a 70 % chance of churn.

Calibration was decided **before** the threshold on purpose. A calibration layer rewrites
the probabilities a threshold would act on, so a threshold chosen first would belong to a
score that no longer exists.

## The frozen rule

```text
positive = probability >= 0.3272694566222328
```

- **The comparison is `>=`, not `>`.** They differ on exactly the customers whose
  probability equals the threshold, and a rule that leaves the boundary implicit is not a
  frozen rule. A customer at exactly the threshold is predicted positive.
- **The threshold is stored unrounded.** A threshold rounded in the sixth decimal moves
  customers across the boundary.
- **It is not configurable.** No environment variable, request field or runtime input may
  redefine it; the serving layer refuses to start if one tries.

## How it was selected

Policy `F1_MAXIMIZATION`, applied **on the training pool alone**:

1. Candidates are every distinct predicted probability, plus `0.5` added unconditionally so
   the procedure is able to return "the default was already best".
2. Each candidate is scored by the F1 of the positive class, and by nothing else.
3. Ties within 1e-12 are broken toward the threshold closest to `0.5` — a parsimony rule,
   not a claim that `0.5` is correct.
4. Probabilities are never rounded before the search.

`TunedThresholdClassifierCV` was deliberately not used: the candidate set, the rule, the
objective and the tie-breaking had to be visible and testable, and an AST audit fails the
build if the class appears in the threshold code.

**Nested evaluation.** The *procedure* — not a fitted threshold — was evaluated across 5
outer folds, so no row ever helped choose the threshold it was then judged under. The
adoption rule was fixed before any of it was read: eligible only if the mean paired ΔF1 is
greater than zero **and** F1 improves strictly in at least 4 of the 5 outer folds, with a
tie counting as neither.

Once eligible, the same policy is applied once to the out-of-fold probabilities of the whole
training pool to fix the number. **That application is a freeze, not an estimate.**

## The trade it makes

Nested, against the `0.5` default, on the same 5 outer folds:

| | Default 0.5 | Selected policy | Paired Δ (folds in favour) |
| --- | --- | --- | --- |
| Recall | 0.5438 | **0.7672** | **+0.2234** (5 / 5) |
| Precision | 0.6521 | 0.5396 | −0.1125 (0 / 5) |
| F1 | 0.5924 | **0.6333** | **+0.0409** (5 / 5) |
| Predicted-positive rate | 0.2213 | 0.3777 | +0.1564 |

The trade is deliberate and one-directional: substantially more churners caught, at a
materially lower precision, and a predicted-positive rate that grows by roughly half. In
operational terms the predicted-positive rate *is* the contact volume — moving the threshold
down buys recall with capacity.

On the holdout the frozen rule flags 35.6 % of customers and catches 72.2 % of the churners:
232 false positives are retention contacts spent on customers who would have stayed, 104
false negatives are churners never flagged.

## Why this is not business-optimal

**No retention cost, contact capacity or customer value was available, so none was assumed.**
F1 weights precision and recall equally, which is a defensible default and not an economic
argument. A cost-optimal operating point would need:

- the cost of a retention contact, and its success rate;
- the value of a retained customer over the horizon that matters;
- the team's actual contact capacity per period.

Given those, the selection procedure is the same and only the objective changes. Until they
exist, inventing them would produce a number that looks like an economic conclusion and is
not one. Choosing an operating point against real costs remains open work.
