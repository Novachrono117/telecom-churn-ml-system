# Data — acquisition and provenance

This directory holds the project's data layers. **No data file is versioned**:
only this document is tracked (see `.gitignore`). Everything else is acquired or
regenerated reproducibly by code.

```text
data/
├── README.md      <- tracked
└── raw/           <- untouched source file, never edited by hand
```

`interim/` and `processed/` are created by the phase that first writes to them
(preprocessing onward); they are not pre-created as empty placeholders.

## Dataset

| Field | Value |
| --- | --- |
| Name | Telco Customer Churn |
| Source | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| Kaggle identifier | `blastchar/telco-customer-churn` |
| Expected file | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Expected location | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| License | Per the Kaggle dataset page (IBM sample data) |

The multi-table IBM Cognos variant — the enriched sample containing `Churn Score`,
`CLTV`, `Churn Reason` and similar fields — is **deliberately not used**. Those
columns are recorded after the churn outcome and would introduce target leakage.

## Acquisition

The file must end up at exactly:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Option A — Kaggle CLI (authenticated, reproducible)

Requires a Kaggle account and an API token (`kaggle.json` from *Account →
Create New API Token*), placed at `%USERPROFILE%\.kaggle\kaggle.json` on Windows
or `~/.kaggle/kaggle.json` on Linux/macOS.

```powershell
# uvx runs the CLI in a throwaway environment: no permanent project dependency
uvx --from kaggle kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
```

### Option B — manual download

Download the CSV from the dataset page above while signed in to Kaggle and move
it into `data/raw/` without renaming or editing it.

### After acquisition

```powershell
uv run python scripts/inspect_raw_data.py
```

This computes the SHA-256, fills in the provenance table below, and regenerates
`reports/data_understanding.md` and `reports/data_dictionary.md`.

## Rules

- The raw file is **read-only**: never edit, re-save or re-encode it. Every
  correction happens downstream, in code.
- Any change of source, version or file must be reflected in the provenance
  table below, because recorded results are only traceable to a specific hash.

## Provenance

<!-- provenance:start -->
| Field | Value |
| --- | --- |
| File name | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Size (bytes) | 977501 |
| SHA-256 | `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a` |
| Source | https://www.kaggle.com/datasets/blastchar/telco-customer-churn |
| Dataset identifier | `blastchar/telco-customer-churn` |
| Recorded on | 2026-08-09 |
<!-- provenance:end -->

> The values above are filled in automatically by
> `scripts/inspect_raw_data.py`. They stay marked as pending until the real file
> is inspected — no hash or size is ever written by hand.
