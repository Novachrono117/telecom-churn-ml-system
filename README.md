# Churn ML Engineering — Telecom Customer Churn

End-to-end **machine learning engineering** solution for telecom customer churn
prediction. The goal is not a trained classifier in a notebook, but a
reproducible system: validated data understanding, leakage-free preprocessing,
a controlled model comparison, an analyzed decision threshold, explainability,
a packaged inference pipeline and a monitoring design.

> ## ⚠️ Project status — Phase 5 (Baselines)
>
> **There is no holdout estimate in this repository, and there will be none
> before Phase 9.** The dataset has been acquired, inspected and explored; the
> train/holdout split is frozen; the preprocessing pipeline is built and tested;
> and three reference models have been cross-validated on the training pool.
>
> Every metric published here is **cross-validated out-of-fold performance on the
> training pool** — not a test result. Every number comes from a run actually
> executed by code in this repository. The holdout partition is reserved for the
> final evaluation and no statistic of it is reported anywhere.

---

## Problem

| Item | Definition |
| --- | --- |
| Domain | Telecommunications |
| Task | Supervised binary classification |
| Planned dataset | Telco Customer Churn (to be acquired in Phase 2) |
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
│   ├── README.md          # acquisition + provenance (the only tracked file here)
│   └── raw/               # untouched source CSV, never committed
├── notebooks/
│   └── 01_eda.ipynb       # Phase 3 narrative (committed without outputs)
├── reports/
│   ├── data_understanding.md   # generated: findings, quality, leakage
│   ├── data_dictionary.md      # generated: one row per observed column
│   ├── eda_report.md           # generated: Phase 3 analysis
│   ├── preprocessing_report.md # generated: Phase 4 protocol
│   ├── split_manifest.json     # generated: frozen split + fingerprints
│   ├── baseline_report.md      # generated: Phase 5 baselines
│   ├── experiments/            # generated: machine-readable experiment records
│   └── figures/eda/            # generated: 16 figures
├── scripts/
│   ├── inspect_raw_data.py     # regenerates the Phase 2 reports
│   ├── run_eda.py              # regenerates the Phase 3 report and figures
│   ├── build_split.py          # freezes / verifies the train-holdout split
│   ├── run_preprocessing.py    # regenerates the Phase 4 report
│   └── run_baselines.py        # regenerates the Phase 5 baselines
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── config.py
│       ├── data/          # loader, inspection, provenance, reporting
│       ├── analysis/      # descriptive stats, association measures, plots
│       ├── preprocessing/ # split, contract, transformers, pipeline, manifest
│       └── modeling/      # baseline builders, paired CV, metrics, results
└── tests/
```

Planned as the project advances: `data/{interim,processed}/`,
`src/churn/{features,modeling,explainability,monitoring}/` and `artifacts/`.
Raw data and model artifacts are never committed.

## Regenerating the analysis artifacts

```powershell
uv run python scripts/inspect_raw_data.py     # Phase 2: provenance, dictionary, validation
uv run python scripts/run_eda.py              # Phase 3: EDA report and figures
uv run python scripts/build_split.py          # Phase 4: freeze the split manifest
uv run python scripts/build_split.py --verify # Phase 4: prove the split is reproducible
uv run python scripts/run_preprocessing.py    # Phase 4: preprocessing report
uv run python scripts/run_baselines.py        # Phase 5: baseline CV, figures, report
uv run jupyter lab notebooks/01_eda.ipynb     # the EDA narrative, interactively
```

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
| 6 | Feature engineering — hypothesis-driven features | ⬜ Not started |
| 7 | Modeling — controlled model comparison | ⬜ Not started |
| 8 | Tuning — only promising candidates | ⬜ Not started |
| 9 | Final evaluation — held-out test set, threshold, calibration, errors | ⬜ Not started |
| 10 | Explainability — global and local | ⬜ Not started |
| 11 | Persistence & inference — trained pipeline and prediction interface | ⬜ Not started |
| 12 | Monitoring design — quality, drift, performance, retraining triggers | ⬜ Not started |
| 13 | Portfolio product — optional API/UI | ⬜ Not started |
| 14 | Academic delivery — notebook, report, figures, presentation | ⬜ Not started |
| 15 | GitHub polish — documentation and reproduction instructions | ⬜ Not started |

## Methodological commitments

- The test set is separated before any transformation is fitted and stays
  untouched until Phase 9.
- Preprocessing is learned **only** on training data, through
  `Pipeline` / `ColumnTransformer`.
- The feature contract is order-agnostic on input and canonical on output, so a
  caller's column order can never change what the model sees.
- Nothing is imputed. Missing, empty or whitespace-only feature values are
  rejected; the single exception is a blank `TotalCharges` at `tenure == 0`,
  which is a structural zero, not an estimate.
- Model selection uses cross-validation on training data only.
- Evaluation reports precision, recall, F1, ROC-AUC, PR-AUC, the confusion
  matrix, calibration and threshold sensitivity — never accuracy alone.
- Any cost-based threshold discussion without real business figures is labeled
  explicitly as a hypothetical scenario.
- Feature importance and SHAP values are treated as associations, not causal
  evidence.

## License

Not defined yet.
