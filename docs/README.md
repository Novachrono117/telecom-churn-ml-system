# Documentation map

You have a question about this system. This page says where to go, what the
authoritative artifact is, and which command proves it.

Seven pages sit between the [README](../README.md) and the generated evidence under
[`reports/`](../reports). They **synthesise and navigate**; they do not restate the
reports, and they introduce no number that is not read from an artifact.

## Where to go

| Question | Start here | Source of truth | Verification |
| --- | --- | --- | --- |
| How is the system put together? | [architecture.md](architecture.md) | [serving_report.md](../reports/serving_report.md) · [serving_results.json](../reports/experiments/serving_results.json) | `uv run --project serving python scripts/build_serving_record.py --verify` |
| How was the dataset split? | [methodology.md](methodology.md) | [preprocessing_report.md](../reports/preprocessing_report.md) · [split_manifest.json](../reports/split_manifest.json) | `uv run python scripts/build_split.py --verify` |
| Why logistic regression? | [methodology.md](methodology.md) | [model_comparison_report.md](../reports/model_comparison_report.md) · [tuning_report.md](../reports/tuning_report.md) · [tuning_results.json](../reports/experiments/tuning_results.json) | — (development experiments; not re-run by a gate) |
| Why was nothing else adopted? | [methodology.md](methodology.md#decision-gates) | [feature_engineering_report.md](../reports/feature_engineering_report.md) · [calibration_report.md](../reports/calibration_report.md) | — |
| What does the model do, and where does it fail? | [model-card.md](model-card.md) | [portfolio_metadata.json](../reports/portfolio/portfolio_metadata.json) · [holdout_results.json](../reports/experiments/holdout_results.json) | `uv run python scripts/build_portfolio_metadata.py --verify` |
| How was the final estimate produced? | [methodology.md](methodology.md) | [holdout_evaluation_report.md](../reports/holdout_evaluation_report.md) | `uv run python scripts/evaluate_holdout.py --verify` |
| Why threshold `0.3272694566222328`? | [decision-policy.md](decision-policy.md) | [threshold_report.md](../reports/threshold_report.md) · [decision_policy.json](../reports/decision_policy.json) | `uv run python scripts/freeze_model.py --verify` |
| How are explanations computed? | [explainability.md](explainability.md) | [model_interpretation_report.md](../reports/model_interpretation_report.md) · [model_interpretation_results.json](../reports/experiments/model_interpretation_results.json) | `uv run python scripts/run_model_interpretation.py --verify` |
| How is a prediction served? | [architecture.md](architecture.md) | [serving_report.md](../reports/serving_report.md) | `uv run --project serving python scripts/build_serving_record.py --verify` |
| What is frozen, and what proves it? | [architecture.md](architecture.md#provenance-and-integrity) | [model_freeze_report.md](../reports/model_freeze_report.md) · [decision_policy.json](../reports/decision_policy.json) | `uv run python scripts/freeze_model.py --verify` |
| How does monitoring work? | [monitoring.md](monitoring.md) | [monitoring_report.md](../reports/monitoring_report.md) · [monitoring.toml](../configs/monitoring.toml) | `uv run python scripts/build_monitoring_record.py --verify` |
| How do I run and reproduce this? | [reproducibility.md](reproducibility.md) | [`scripts/`](../scripts) · [`configs/`](../configs) | the 11 verification gates |

## The seven pages

| Page | Covers |
| --- | --- |
| [architecture.md](architecture.md) | Boundaries: feature contract, frozen pipeline, decision policy, HTTP layer, monitoring observer, environment isolation. |
| [methodology.md](methodology.md) | The experimental protocol end to end, the analyst-exposure limitation, and the decisions that went the other way. |
| [model-card.md](model-card.md) | Intended and out-of-scope use, evaluated performance, limitations. |
| [decision-policy.md](decision-policy.md) | Probability is not a decision: calibration, threshold selection, the trade actually made. |
| [explainability.md](explainability.md) | Why the per-customer explanation here is an exact identity rather than an approximation. |
| [monitoring.md](monitoring.md) | Four signals kept separate, what the cutoffs are, and what labels would be needed to say more. |
| [reproducibility.md](reproducibility.md) | Clone, run, test, lint and verify — every documented command. |

## Deeper evidence

Everything above is a summary of material that stays where it was generated:

- [`reports/`](../reports) — 18 generated reports, one per stage.
- [`reports/experiments/`](../reports/experiments) — machine-readable records.
- [`reports/figures/`](../reports/figures) — 58 generated figures.
- [`configs/`](../configs) — [`base.toml`](../configs/base.toml) (frozen provenance) and
  [`monitoring.toml`](../configs/monitoring.toml) (alert policy).
- [`diagrams/`](diagrams) and [`screenshots/`](screenshots) — visual evidence.

Reports are the depth; these pages are the index. Where the two ever disagree, the
report and the machine-readable record are authoritative.
