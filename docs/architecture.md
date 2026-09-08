# Architecture

What the system is made of, and — more usefully — where its boundaries are. Each
boundary exists because something on one side of it must not be able to change
something on the other.

Deep dive: [serving_report.md](../reports/serving_report.md) ·
[model_freeze_report.md](../reports/model_freeze_report.md) ·
[serving_results.json](../reports/experiments/serving_results.json)

![System architecture: API consumers post to the prediction endpoints and the Portfolio UI sends
exactly one explanation request per submission, both through the same canonical scoring call; the
request crosses the feature contract, then the frozen pipeline, which emits one probability feeding
the frozen decision policy and the exact log-odds decomposition, while the monitoring observer sits
off the causal path behind a read-only endpoint.](diagrams/system-architecture.svg)

## The path a request takes

| # | Stage | What it guarantees |
| --- | --- | --- |
| 1 | **Raw input contract** — `PredictionRequest`, pydantic, `extra="forbid"` | Exactly the 19 contracted features, nothing else accepted. |
| 2 | **`prepare_features`** — `churn.preprocessing.contracts` | Validates, rejects and canonicalises column order **before** the persisted pipeline sees anything. It runs *outside* the pickle, so its source is fingerprinted separately. |
| 3 | **Frozen pipeline** — `TotalChargesCleaner` → `ColumnTransformer` → `LogisticRegression` | 19 raw features → 46 transformed columns → one score. Nothing is fitted at serve time; `fit_calls = 0` is asserted behaviourally. |
| 4 | **Positive-class probability** | The column is resolved from `classes_`, never assumed to be `[:, 1]`. |
| 5 | **Frozen decision policy** — `probability >= 0.3272694566222328` | The comparison and the unrounded threshold come from [`decision_policy.json`](../reports/decision_policy.json) and from nowhere else. No environment variable may redefine them. |
| 6 | **Exact explanation** — log-odds decomposition | Reconstructs the served score algebraically; returns `EXPLANATION_UNAVAILABLE` rather than an explanation the prediction would not support. See [explainability.md](explainability.md). |
| 7 | **FastAPI boundary** — `churn.serving.api` | The only module that needs a web framework. The predictive path is callable without a server. |
| 8 | **Monitoring observer** | Off the causal path: it reads what was decided and cannot change a prediction. See [monitoring.md](monitoring.md). |
| 9 | **Portfolio UI** — `/demo` | Static HTML/CSS/JS over the same endpoints, opt-in behind an environment flag. |

Endpoints: `/health/live`, `/health/ready`, `/api/v1/model`, `/api/v1/predict`,
`/api/v1/predict/batch`, `/api/v1/explain`, `/api/v1/portfolio`, `/openapi.json`, plus
`/api/v1/monitoring` and `/demo` when enabled.

This is a single local in-memory process, not a distributed system. The separations
below are inside one program.

## Provenance and integrity

![Provenance chain: the raw dataset digest gates the split manifest, which gates the training
identifiers and the frozen configuration; from there three independent digests — the semantic model
fingerprint, the serialised pipeline bytes, and the source and runtime provenance of the code that
executes around the model — plus the frozen decision-policy digest feed 34 startup gates that
either serve or fail closed.](diagrams/provenance-chain.svg)

Each link is verifiable on its own, and the two ends are regenerable rather than stored:
the partition is rebuilt from the raw file plus [`configs/base.toml`](../configs/base.toml),
and the pipeline is rebuilt from the frozen split, builder and training pool.

Three digests answer three different questions, and none of them is sufficient alone:

- **`model_fingerprint_sha256`** — the *semantic* identity of the learned state, invariant
  to serialisation. Authoritative.
- **`pipeline_sha256`** — the identity of the serialised *bytes*. Catches a corrupted or
  swapped file; it is a property of the file, not of the model.
- **`code_provenance.source_digests`** — whole-file digests of the 8 source files and 3
  configuration files that materially determine the inference function. A `.joblib`
  pickles classes *by reference*, so editing `TotalChargesCleaner.transform` changes what
  the loaded pipeline computes while both fingerprints stay identical.

**34 startup integrity gates** check all of this before the service exists — artifact
digest, model fingerprint, decision-policy digest, feature contract, runtime source
digests and the exact pinned versions of scikit-learn, numpy, pandas and joblib. On any
mismatch the process refuses to serve; no failure path rebuilds anything.
`GET /health/ready` reports `"startup_gates_passed": 34`, or `503`.

> **The freeze records describe the repository as it was at freeze time.**
> [`decision_policy.json`](../reports/decision_policy.json) records
> `artifacts.versioned_in_git = false`, and
> [`model_freeze_report.md`](../reports/model_freeze_report.md) says the same, because the
> pipeline binary had not been committed yet when the model was frozen. It was committed
> later, without changing its bytes, its semantic fingerprint or the decision policy — the
> three digests above still verify. Those records are deliberately not rewritten: a frozen
> record edited to match a newer repository state stops being evidence of what was frozen.

## Why `serving/` is its own uv project

`pyproject.toml` and `uv.lock` are part of the freeze: their digests are recorded in
[`decision_policy.json`](../reports/decision_policy.json) under
`code_provenance.configuration_digests`, and `scripts/freeze_model.py --verify` fails the
moment either moves. Adding FastAPI and Uvicorn to the root project would therefore have
invalidated the model freeze.

> A web-serving dependency must not be able to invalidate the frozen model environment
> merely because the interface layer was added later.

So the serving **environment** is separated while the serving **code** is not: `serving/`
holds its own `pyproject.toml`, `uv.lock` and test suite, and depends on the root project
as an editable path dependency. `src/churn/serving/` still lives alongside the rest of the
package. The four ML libraries are pinned with `==` to the versions recorded in the policy,
and re-checked at startup.

A consequence, recorded rather than hidden: the serving environment inherits the root
project's analysis dependencies (matplotlib, seaborn, scipy), which a production inference
container has no use for. Removing them would change the root project's digest and fail the
freeze, so it needs a deliberate re-freeze. See
[serving_report.md](../reports/serving_report.md) §13.

## What lives where

The directory layout is in the README's
[project structure](../README.md#project-structure). Two entries matter architecturally:
`artifacts/` holds the frozen pipeline (8,218 bytes), and that binary **is** versioned:
`.gitignore` excludes `*.joblib` and then re-includes `artifacts/model/*.joblib`, so a clone
can serve without retraining. Shipping the bytes is a convenience, not the guarantee — the
authoritative identity of the model is `model_fingerprint_sha256`, `pipeline_sha256` verifies
the serialised bytes, and [`freeze_model.py`](../scripts/freeze_model.py) reconstructs and
re-verifies the artefact from code, so the pipeline is regenerable as well as stored. And
`scripts/` holds one entry point per stage, 11 of which support `--verify`.

> **On the screenshots.** The images under [`screenshots/`](screenshots) are documentation
> evidence captured from the real local application. They are not part of the
> reproducibility contract, and no tooling for regenerating them is versioned.
