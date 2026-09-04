# Reproducibility

Clone, run, test and verify. Every command on this page was executed on this tree.

## Requirements

- **Python 3.12** — pinned in `.python-version` and installed automatically by `uv`. No
  other version was tested, and the frozen artifact pins the interpreter's library
  versions.
- **[uv](https://docs.astral.sh/uv/)** — the only prerequisite you install yourself.

No Kaggle account and no retraining run are needed: the raw dataset is versioned and
fingerprinted, and the pipeline is regenerable from it.

## Clone and sync

```bash
# from the repository root, after cloning
uv sync                      # root: data, modeling, analysis, verification, tests
uv sync --project serving    # serving: FastAPI + the pinned ML runtime
```

## Root vs serving environment

Two environments, one source tree — they are *not* independent projects:

| | Root | `serving/` |
| --- | --- | --- |
| Purpose | Data, experimentation, analysis, verification | The HTTP boundary and its tests |
| Holds | pandas, numpy, scikit-learn, matplotlib, pytest | FastAPI, Uvicorn, plus the four ML libraries pinned with `==` |
| Code lives in | `src/churn/` | `src/churn/serving/` — the same package |
| Relationship | — | Depends on the root project as an **editable path dependency** |

Only the *environment* is separated. It has to be: `pyproject.toml` and `uv.lock` are part
of the frozen provenance, so adding a web framework to the root project would invalidate the
model freeze. See [architecture.md](architecture.md#why-serving-is-its-own-uv-project).

## Run the API

```bash
uv run --project serving python -m churn.serving      # http://127.0.0.1:8000
```

Both extras are off by default, behind environment flags.

**Portfolio UI** — adds `/demo`:

```bash
CHURN_SERVING_PORTFOLIO_UI=1 uv run --project serving python -m churn.serving
```

```powershell
$env:CHURN_SERVING_PORTFOLIO_UI = "1"
uv run --project serving python -m churn.serving
```

**Monitoring** — adds `/api/v1/monitoring`:

```bash
CHURN_SERVING_MONITORING=1 uv run --project serving python -m churn.serving
```

```powershell
$env:CHURN_SERVING_MONITORING = "1"
uv run --project serving python -m churn.serving
```

**Both:**

```bash
CHURN_SERVING_PORTFOLIO_UI=1 CHURN_SERVING_MONITORING=1 \
  uv run --project serving python -m churn.serving
```

```powershell
$env:CHURN_SERVING_PORTFOLIO_UI = "1"
$env:CHURN_SERVING_MONITORING = "1"
uv run --project serving python -m churn.serving
```

PowerShell has no inline `VAR=value command` form, which is why the variables are set first.
`GET /health/ready` reports what startup verified, or `503` if any gate failed.

## Tests

```bash
uv run pytest -rs                                   # root
uv run --project serving pytest -rs serving/tests   # serving
```

## Lint and formatting

```bash
uv run ruff check .
uv run --project serving ruff check serving
uv run ruff format --check .
uv run --project serving ruff format --check serving
```

## Dependency locks

```bash
uv lock --check
uv lock --check --project serving
```

## Artifact verification

Eleven gates. Each rebuilds its artifact in memory or re-checks its invariants, writes
nothing, and fails closed — a check that repaired what it found missing could never fail.
Drop `--verify` to regenerate.

```bash
uv run python scripts/build_split.py --verify
uv run python scripts/freeze_model.py --verify
uv run python scripts/evaluate_holdout.py --verify
uv run python scripts/run_error_analysis.py --verify
uv run python scripts/run_model_interpretation.py --verify
uv run python scripts/build_monitoring_reference.py --verify
uv run python scripts/build_monitoring_record.py --verify
uv run python scripts/build_portfolio_metadata.py --verify
uv run python scripts/build_portfolio_record.py --verify
uv run python scripts/build_academic_record.py --verify
```

**The eleventh runs in the serving project**, not the root one — it builds the FastAPI
application in order to read the published routes from it rather than from a hand-kept list,
and FastAPI is deliberately absent from the root environment:

```bash
uv run --project serving python scripts/build_serving_record.py --verify
```

### What each gate protects

| Command | What it proves |
| --- | --- |
| `build_split.py` | The partition is regenerated from the raw bytes plus the config and reproduces the recorded manifest field for field. |
| `freeze_model.py` | The freeze still holds: pipeline digest, model fingerprint, decision rule, calibration policy, feature contract and the whole-file source and configuration digests. Strictly read-only. |
| `evaluate_holdout.py` | The committed final-evaluation record's invariants — the confusion matrix accounts for every row, every metric lies in its domain, each interval contains its estimate, and nothing was selected after the holdout was opened. |
| `run_error_analysis.py` | The error-analysis record's invariants, including that the frozen confusion matrix is reproduced and no selection followed the analysis. |
| `run_model_interpretation.py` | The interpretation record's invariants — the exact decomposition reproduced the pipeline within tolerance, and the holdout was not used. |
| `build_serving_record.py` | The committed serving record still describes the real application, byte for byte, against the real frozen artifacts. |
| `build_monitoring_reference.py` | The reference profile's invariants and provenance, and that rebuilding it reproduces the file byte for byte. |
| `build_monitoring_record.py` | The monitoring record rebuilt from the reference profile and the operational policy, byte for byte. |
| `build_portfolio_metadata.py` | The metadata the demo may display, rebuilt from the holdout record and the decision policy, byte for byte — plus the digest constant the serving layer enforces at startup. |
| `build_portfolio_record.py` | The portfolio record rebuilt from that metadata and the demo assets, byte for byte. |
| `build_academic_record.py` | The record of the academic deliverables, rebuilt from the deliverables that actually exist, byte for byte. |

Byte comparison is only meaningful because these artifacts carry no clock: a file with a
generation timestamp would differ on every run, and the check would have to be weakened to
something that proves less.

## Independent reproduction notebook

[`notebooks/02_academic_delivery.ipynb`](../notebooks/02_academic_delivery.ipynb) rebuilds
the frozen protocol **without cloning this repository**: it downloads the dataset from a
public mirror with no credentials, canonicalises line endings, verifies the SHA-256 against
the frozen value and aborts on mismatch, then reproduces the split, preprocessing,
baselines, model comparison, threshold and holdout evaluation, checking every number against
the committed artifact. Committed without outputs; execute to reproduce. *(Narrative in
Portuguese.)*

## Windows: long paths

Clone into a **short** path — `C:\dev\churn`, not a deep one. A deep checkout pushes the
compiled extension modules inside `.venv` past the 260-character `MAX_PATH` limit, and
scipy then fails to import. This is the only Windows-specific issue observed here.
