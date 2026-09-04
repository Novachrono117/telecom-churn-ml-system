# Explainability

The explanation here is an **algebraic identity, not an approximation** — no sampling, no
surrogate model, no seed, and therefore nothing to be uncertain about.

Source of truth: [model_interpretation_report.md](../reports/model_interpretation_report.md) ·
[`model_interpretation_results.json`](../reports/experiments/model_interpretation_results.json)

## The identity

A logistic regression's response surface is known in closed form:

```text
logit(P(Churn = 1 | x)) = intercept + Σ_j  coefficient_j × transformed_feature_j
P(Churn = 1 | x)        = sigmoid(logit)
```

The 46 transformed columns **partition** the 19 raw features, so regrouping them is exact:

```text
logit = intercept + Σ_g  contribution_g(x)

numeric      contribution = coefficient × (raw_value − scaler_mean) / scaler_scale
categorical  contribution = the coefficient of the active level
```

Each of the 19 raw features contributes exactly one number in log-odds. Nothing is left
over, and nothing is counted twice — the partition is derived from the fitted encoder's own
structure rather than by parsing generated column names, and is verified explicitly.

## Global and local, from the same equation

- **Global.** The coefficients themselves, plus the empirical dispersion of each feature's
  contribution across the training pool — a feature with a large coefficient that barely
  varies moves few customers.
- **Local.** For one customer, the same decomposition, listed as the factors pushing the
  score up and down. The threshold expressed in log-odds is
  `log(θ / (1 − θ)) = −0.7205610102182292`, which is the *same* rule in different
  coordinates, not a second threshold.

The global interpretation is computed on the **training pool**, never the holdout: the
holdout was consumed by the final evaluation, and ranking features on a spent sample is
post-hoc mining.

## Exact score reconstruction

An interpretation that does not reproduce the model's own decision function is not an
interpretation of it. Three comparisons run over all 5,634 training rows, and the analysis
**aborts** on any failure:

| Reconstructed | Compared against | Max absolute error |
| --- | --- | --- |
| `intercept + Z @ coefficients` | `pipeline.decision_function(X)` | 0.0 |
| `sigmoid(manual_logit)` | `pipeline.predict_proba(X)[:, positive_column]` | 1.110e-16 |
| `intercept + Σ` of the 19 grouped contributions | `pipeline.decision_function(X)` | 1.776e-15 |

Tolerance **1e-12**. A dot product of 46 float64 terms has a rounding floor of a few
multiples of 2⁻⁵², about 2.2e-16, so the tolerance sits roughly four orders of magnitude
above the noise and far below any error that would signal a wrong column, a missed intercept
or a transposed coefficient. The serving endpoint applies the same check per request and
returns `EXPLANATION_UNAVAILABLE` rather than an explanation the prediction would not
support.

## Structural coupling

Some encoded states are the same state. Seven product equivalences were declared from the
data dictionary — never from the coefficients — then counted row by row on the training
pool, with **zero violations**:

- `InternetService=No` ⇔ `No internet service` in six add-on features (1,214 rows);
- `PhoneService=No` ⇔ `MultipleLines=No phone service` (559 rows).

When two raw states are identical, their one-hot columns are the same vector, so the model's
parameters are not identified along that direction: any redistribution of the block's weight
among its columns yields identical predictions, and the L2 penalty splits it evenly. The
pairwise coefficient differences inside each block are zero to the last bit.

> The repeated `No internet service` coefficients are **not** six or seven independent
> findings. They are one underlying customer property re-encoded up to seven times, which a
> naive per-column ranking would count that many times. The quantity the model actually
> applies is the **block aggregate**; the per-column values are one point in a family of
> equivalent parameterisations.

The decomposition stays exactly correct either way — what changes is *attribution*, so the
interpretation groups these columns instead of ranking them separately.

## Why no SHAP, no LIME

SHAP and LIME exist to approximate attributions for models whose response surface is not
available in closed form. This model's is. For an additive linear model in log-odds the
per-feature contributions *are* the decomposition, reproducing the served score to 1e-16 —
so an approximation would add a sampling procedure, a seed and an error term to answer a
question that already has an exact answer. That is the reason they are absent: not that they
would be wrong, but that they would be redundant for this specific question. A non-linear
model here would change that argument entirely.

## What an explanation is not

These are **associations, not causes**. No coefficient here licenses a claim about what
*would* happen if a customer's contract changed: nothing in this project uses a causal
identification strategy, and nothing measures whether acting on a prediction changes an
outcome. A contrast `β_A − β_B` is an exact statement about the linear function — it is not
automatically a statement about a change a customer could undergo.
