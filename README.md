# Churn ML Engineering — Telecom Customer Churn

End-to-end **machine learning engineering** solution for telecom customer churn
prediction. The goal is not a trained classifier in a notebook, but a
reproducible system: validated data understanding, leakage-free preprocessing,
a controlled model comparison, an analyzed decision threshold, explainability,
a packaged inference pipeline and a monitoring design.

> ## Project status — Phase 10 (Global interpretation of the frozen model)
>
> The model **is frozen** and the holdout **has been opened, once**. Model
> selection, feature engineering, tuning, the calibration policy and the decision
> threshold were all settled on the training pool; the configuration was frozen in
> `9d1db49`; the untouched holdout was evaluated exactly once in `8c8e266`; the
> errors were diagnosed in `5e9c4e9` and the frozen function was decomposed in
> `47d6dd3`.
>
> **No selection followed the holdout result, and none may.** That partition is
> spent: it can no longer provide an unbiased estimate for any decision taken from
> here on. Metrics from Phases 5–9B are cross-validated out-of-fold performance on
> the training pool, not test results, and are labelled as such throughout. Every
> number in this repository comes from a run actually executed by its code.
>
> Still open: local (per-customer) explanations, the public prediction interface
> and the monitoring design.

---

## Problem

| Item | Definition |
| --- | --- |
| Domain | Telecommunications |
| Task | Supervised binary classification |
| Dataset | Telco Customer Churn — [provenance](data/README.md) |
| Target | `Churn` |
| Positive class | `Yes` |
| Negative class | `No` |
| Primary output | Churn probability |
| Secondary output | Class label from a justified decision threshold |

The project must answer *who is at risk*, *which factors are associated with the
predictions*, *why an individual prediction was made*, *which threshold is
appropriate* and *how the model would be monitored*. Accuracy alone is never a
model-selection criterion.

> Verified in Phase 2 against the acquired file: `Churn` holds exactly
> `{No, Yes}`, with 26.54% positives. A contract test fails if a future dataset
> version breaks this.

---

## Final result

One frozen configuration, evaluated **once** on the previously untouched holdout
(1,409 customers, 26.54% churners). Full report:
[`reports/holdout_evaluation_report.md`](reports/holdout_evaluation_report.md).

| Metric | Value | 95% CI (bootstrap, 2,000 replications) |
| --- | --- | --- |
| **Average Precision** | **0.6337** | [0.5777, 0.6864] |
| **ROC-AUC** | **0.8420** | [0.8183, 0.8633] |
| F1 at the frozen threshold | 0.6164 | [0.5762, 0.6534] |
| Recall at the frozen threshold | 0.7219 | [0.6760, 0.7652] |
| Precision at the frozen threshold | 0.5378 | [0.4937, 0.5791] |

Those numbers only mean something against the references that bound them: a
no-skill ranker scores an Average Precision equal to the prevalence (0.2654) and
a ROC-AUC of 0.500. Against those, AP 0.6337 and ROC-AUC 0.8420 are real signal.
Accuracy, by contrast, says almost nothing here: this model scores 0.7615 while
always predicting "no churn" scores 0.7346 — 2.7 points apart, for a rule that
would never flag a single customer. That is why accuracy is not the criterion.

The frozen rule flags 35.6% of customers and catches 72.2% of the churners among
them. Confusion matrix at the frozen threshold:

| | Predicted: retained | Predicted: churn |
| --- | --- | --- |
| **Actually retained** (1,035) | 803 | 232 |
| **Actually churned** (374) | 104 | 270 |

That trade is deliberate. 232 false positives are retention contacts spent on
customers who would have stayed; 104 false negatives are churners the system
never flags. The threshold that produces it was chosen by a pre-registered
policy, not tuned to this table — see below.

### The frozen decision policy

Every field is recorded in
[`reports/decision_policy.json`](reports/decision_policy.json).

| Decision | Value | Decided in |
| --- | --- | --- |
| Estimator | `LogisticRegression` (`C=1.0`, `lbfgs`, `class_weight=None`) | Phase 8B |
| Features | 19 contracted → 46 transformed; **0 engineered** | Phase 6 |
| Calibration | **NONE** | Phase 9A |
| Threshold | **0.3272694566222328** (F1-maximisation) | Phase 9B |
| Decision rule | `positive = probability >= threshold` | Phase 9C |

Three of those decisions deserve reading closely, and two of them are negative
results — the kind a leaderboard write-up quietly omits:

- **Logistic regression won.** Cross-validated on the training pool, it reached
  AP 0.6615 against 0.6471 for HistGradientBoosting and 0.6062 for Random Forest.
  The simplest family was not a fallback.
- **No engineered feature was adopted.** Five hypothesis-driven candidates were
  tested with paired ablations in Phase 6. Four did not pass; the fifth,
  `contract_tenure`, was carried forward as a *candidate* and re-examined per
  family in Phase 7, where it moved logistic regression by +0.0017 AP —
  immaterial. Features were designed, measured and discarded, not kept to inflate
  the count.
- **No calibration was adopted.** Neither sigmoid nor isotonic improved the Brier
  score consistently across the outer folds, so no calibration layer was added.
  `NONE` is a gate outcome, not an omission.

The threshold *was* adopted: the F1-maximisation procedure beat the 0.5 default in
5 of 5 outer folds under nested evaluation.

---

## Serving a prediction

The persisted pipeline starts **after** the feature contract. Producing a
validated, canonically ordered matrix is the caller's responsibility —
`build_feature_matrix` does it and rejects anything that breaks the contract:

```python
import json
from pathlib import Path

import joblib
import pandas as pd

from churn.preprocessing import build_feature_matrix

threshold = json.loads(Path("reports/decision_policy.json").read_text(
    encoding="utf-8"))["threshold"]["final_threshold"]
pipeline = joblib.load("artifacts/model/churn_pipeline.joblib")

raw = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv").head(3)
features = build_feature_matrix(raw)          # drops customerID and Churn, validates, orders

positive_column = list(pipeline.classes_).index(1)   # resolved, never assumed to be 1
probability = pipeline.predict_proba(features)[:, positive_column]

for identifier, p in zip(raw["customerID"], probability):
    print(f"{identifier}  churn_probability={p:.4f}  prediction={'Yes' if p >= threshold else 'No'}")
```

```text
7590-VHVEG  churn_probability=0.6145  prediction=Yes
5575-GNVDE  churn_probability=0.0444  prediction=No
3668-QPYBK  churn_probability=0.2992  prediction=No
```

This is the raw inference path, not a public interface — that is Phase 11. The
comparison is `>=`, not `>`: a customer whose probability equals the threshold
exactly is predicted positive.

---

## Requirements

- [uv](https://docs.astral.sh/uv/) (project and environment manager)
- Python 3.12 — installed automatically by uv, pinned in `.python-version`

Conda is not used. `pyproject.toml` is the single source of dependencies and
`uv.lock` is versioned so the environment is byte-for-byte reproducible.

## Environment reproduction

```powershell
# from the repository root
uv sync                      # creates .venv from uv.lock and installs the project
uv run pytest                # run the test suite
uv run ruff check .          # lint
uv run ruff format .         # format
```

`uv run <command>` uses the project environment without a manual `activate`.
To open a Python session with the package importable:

```powershell
uv run python -c "from churn import get_config; print(get_config())"
```

---

## Configuration

All shared settings live in [`configs/base.toml`](configs/base.toml) — random
seed, data-layer paths, target definition and the train/test split policy.
They are read and validated by [`src/churn/config.py`](src/churn/config.py)
using `tomllib` (standard library) plus `pydantic` models.

```python
from churn import get_config

config = get_config()
config.seed  # 42
config.target.column  # "Churn"
config.split.test_size  # 0.2
config.data.raw_dir  # <repo>/data/raw  (absolute, resolved from the repo root)
```

Data paths are written relative to the repository root in the TOML file and
resolved to absolute paths at load time, so no module depends on the current
working directory and no absolute path is hardcoded.

---

## Repository structure

Current state (directories are created only when a phase actually needs them —
no empty placeholders):

```text
.
├── CLAUDE.md              # project contract: rules, phases, quality bar
├── README.md
├── pyproject.toml         # dependencies and tooling (source of truth)
├── uv.lock                # locked environment (versioned)
├── .python-version        # 3.12
├── configs/
│   └── base.toml          # seed, paths, target, split
├── data/
│   ├── README.md          # acquisition + provenance
│   └── raw/               # untouched source CSV (versioned: <1 MB, immutable)
├── artifacts/
│   └── model/churn_pipeline.joblib  # the frozen inference pipeline (8,218 bytes)
├── notebooks/
│   └── 01_eda.ipynb       # Phase 3 narrative (committed without outputs)
├── reports/               # one report per phase, all generated
│   ├── data_understanding.md · data_dictionary.md      # Phase 2
│   ├── eda_report.md                                   # Phase 3
│   ├── preprocessing_report.md · split_manifest.json   # Phase 4
│   ├── baseline_report.md                              # Phase 5
│   ├── feature_engineering_report.md                   # Phase 6
│   ├── model_comparison_report.md                      # Phase 7
│   ├── hgb_representation_report.md · tuning_report.md # Phase 8
│   ├── calibration_report.md · threshold_report.md     # Phase 9A / 9B
│   ├── model_freeze_report.md · decision_policy.json   # Phase 9C
│   ├── holdout_evaluation_report.md                    # Phase 9D
│   ├── error_analysis_report.md                        # Phase 9E
│   ├── model_interpretation_report.md                  # Phase 10
│   ├── experiments/       # 10 machine-readable experiment records
│   └── figures/           # 58 figures across 11 subdirectories
├── scripts/               # one entry point per phase; see the table below
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── config.py
│       ├── data/          # loader, inspection, provenance, reporting
│       ├── analysis/      # descriptive stats, association measures, plots
│       ├── preprocessing/ # split, contract, transformers, pipeline, manifest
│       ├── features/      # candidate features, feature groups, ablation protocol
│       └── modeling/      # comparison, tuning, calibration, threshold, freeze,
│                          # holdout, error analysis, interpretation
└── tests/                 # 1,007 tests
```

Planned as the project advances: `data/{interim,processed}/` and
`src/churn/{explainability,monitoring}/`.

The raw CSV (`data/raw/`) and the frozen inference pipeline
(`artifacts/model/churn_pipeline.joblib`) **are** versioned: together under 1 MB,
they let a fresh clone reproduce every result and serve predictions without a
Kaggle account or a retraining run. Derived data (`data/{interim,processed}/`),
virtual environments and caches are not — `uv sync` rebuilds the environment
from `uv.lock`.

## Regenerating the analysis artifacts

Every report, figure and JSON record in this repository is regenerated by one of
these commands. They are ordered: each phase consumes the artefacts of the ones
above it and verifies their digests.

```powershell
uv run python scripts/inspect_raw_data.py         # Phase 2: provenance, dictionary, validation
uv run python scripts/run_eda.py                  # Phase 3: EDA report and figures
uv run python scripts/build_split.py              # Phase 4: freeze the split manifest
uv run python scripts/build_split.py --verify     # Phase 4: prove the split is reproducible
uv run python scripts/run_preprocessing.py        # Phase 4: preprocessing report
uv run python scripts/run_baselines.py            # Phase 5: baseline CV, figures, report
uv run python scripts/run_feature_engineering.py  # Phase 6: paired feature ablations
uv run python scripts/run_model_comparison.py     # Phase 7: model-family comparison
uv run python scripts/run_hgb_representation.py   # Phase 8A: native categorical representation
uv run python scripts/run_tuning.py               # Phase 8B: nested hyperparameter tuning
uv run python scripts/run_calibration.py          # Phase 9A: calibration gate
uv run python scripts/run_threshold.py            # Phase 9B: decision-threshold policy
uv run python scripts/freeze_model.py             # Phase 9C: freeze model + decision policy
uv run python scripts/evaluate_holdout.py         # Phase 9D: the single holdout evaluation
uv run python scripts/run_error_analysis.py       # Phase 9E: post-hoc error analysis
uv run python scripts/run_model_interpretation.py # Phase 10: interpret the frozen model
uv run jupyter lab notebooks/01_eda.ipynb         # the EDA narrative, interactively
```

`freeze_model.py` rebuilds `artifacts/model/churn_pipeline.joblib` from the frozen
split, the frozen builder and the frozen training pool. Both fingerprints
(`model_fingerprint_sha256` and `pipeline_sha256`) must come out unchanged — that
is a stronger guarantee than the committed binary itself.

Neither partition is written to disk. The split is regenerated from the raw file
plus `configs/base.toml`, and `reports/split_manifest.json` records the raw
SHA-256, every split parameter and a SHA-256 over the sorted identifiers of each
partition — enough to prove the partition is the same one without storing it.

The notebook is committed **without executed outputs** to keep the repository
small and its diffs readable; it is validated by executing it end to end
(`jupyter nbconvert --to notebook --execute`).

---

## Phases

| # | Phase | Status |
| --- | --- | --- |
| 1 | Foundation — repository, environment, configuration | ✅ Done |
| 2 | Data understanding — acquisition, data dictionary, validation | ✅ Done — [report](reports/data_understanding.md), [dictionary](reports/data_dictionary.md) |
| 3 | EDA — analysis and hypotheses | ✅ Done — [report](reports/eda_report.md), [notebook](notebooks/01_eda.ipynb) |
| 4 | Preprocessing — split, contract, encoding, scaling, pipeline | ✅ Done — [report](reports/preprocessing_report.md), [split manifest](reports/split_manifest.json) |
| 5 | Baselines — `DummyClassifier` and Logistic Regression | ✅ Done — [report](reports/baseline_report.md), [results](reports/experiments/baseline_results.json) |
| 6 | Feature engineering — hypothesis-driven ablation study | ✅ Done — [report](reports/feature_engineering_report.md), [results](reports/experiments/feature_engineering_results.json) |
| 7 | Modeling — controlled model comparison | ✅ Done — [report](reports/model_comparison_report.md), [results](reports/experiments/model_comparison_results.json) |
| 8A | Representation — native categoricals for HistGradientBoosting | ✅ Done — [report](reports/hgb_representation_report.md), [results](reports/experiments/hgb_representation_results.json) |
| 8B | Tuning — nested hyperparameter search | ✅ Done — [report](reports/tuning_report.md), [results](reports/experiments/tuning_results.json) |
| 9A | Calibration gate — decided **before** the threshold | ✅ Done — [report](reports/calibration_report.md), [results](reports/experiments/calibration_results.json) |
| 9B | Decision threshold — nested policy evaluation | ✅ Done — [report](reports/threshold_report.md), [results](reports/experiments/threshold_results.json) |
| 9C | Freeze — persisted pipeline and decision policy | ✅ Done — [report](reports/model_freeze_report.md), [policy](reports/decision_policy.json) |
| 9D | Final evaluation — the single holdout measurement | ✅ Done — [report](reports/holdout_evaluation_report.md), [results](reports/experiments/holdout_results.json) |
| 9E | Error analysis — post-hoc diagnosis, no selection | ✅ Done — [report](reports/error_analysis_report.md), [results](reports/experiments/error_analysis_results.json) |
| 10 | Explainability — global and local | 🟡 Global done — [report](reports/model_interpretation_report.md), [results](reports/experiments/model_interpretation_results.json). Local (per-customer) explanations pending |
| 11 | Persistence & inference — prediction interface | 🟡 Pipeline persisted and fingerprinted in 9C; the public interface is pending |
| 12 | Monitoring design — quality, drift, performance, retraining triggers | ⬜ Not started |
| 13 | Portfolio product — optional API/UI | ⬜ Not started |
| 14 | Academic delivery — notebook, report, figures, presentation | ⬜ Not started |
| 15 | GitHub polish — documentation and reproduction instructions | ⬜ Not started |

## Methodological commitments

- The holdout was separated before any transformation was fitted, stayed
  untouched through Phase 9C, and was opened exactly **once**, in Phase 9D, after
  the model and the decision rule were frozen. It is now spent: no decision taken
  after that measurement may use it, and none has.
- Preprocessing is learned **only** on training data, through
  `Pipeline` / `ColumnTransformer`.
- The feature contract is order-agnostic on input and canonical on output, so a
  caller's column order can never change what the model sees.
- Nothing is imputed. Missing, empty or whitespace-only feature values are
  rejected; the single exception is a blank `TotalCharges` at `tenure == 0`,
  which is a structural zero, not an estimate.
- Model selection, feature ablations, tuning, the calibration gate and the
  threshold policy all ran on the training pool only, each against a
  pre-registered eligibility rule written down before the comparison — so a
  candidate that merely looked better could not be adopted.
- Calibration is decided **before** the threshold. It rewrites the probabilities a
  threshold acts on, so a threshold chosen first would belong to a score that no
  longer exists.
- The frozen inference function is fingerprinted by whole-file SHA-256 over every
  source file that materially determines it, plus the configuration and the
  dependency pins. An unrelated edit to one of those files invalidates the freeze:
  that is deliberate and fail-closed.
- Evaluation reports precision, recall, F1, ROC-AUC, PR-AUC, the confusion
  matrix, calibration and threshold sensitivity — never accuracy alone.
- Any cost-based threshold discussion without real business figures is labeled
  explicitly as a hypothetical scenario.
- Feature importance and SHAP values are treated as associations, not causal
  evidence.

## Reproducing this repository from scratch

```powershell
git clone https://github.com/Novachrono117/ML-Crunch.git
cd ML-Crunch
uv sync --frozen
uv run pytest -q
```

The raw dataset and the frozen pipeline are versioned, so no Kaggle account and
no retraining run are needed. On Windows, clone into a **short** path
(e.g. `C:\dev\ML-Crunch`): a deep path pushes the compiled extension modules in
`.venv` past the 260-character `MAX_PATH` limit and scipy then fails to import.

## License

Not defined yet.
