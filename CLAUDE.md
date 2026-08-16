# CLAUDE.md — Churn ML Engineering

## Mission
Build an end-to-end **Machine Learning Engineering solution for telecom customer churn prediction**.
Goals:
- **Academic:** satisfy the postgraduate supervised-learning assignment.
- **Portfolio:** demonstrate real ML Engineering beyond a Kaggle-style notebook.
The result must cover data understanding, preprocessing, experimentation, validation, explainability, inference and monitoring.

## Role
Act as a **senior Machine Learning Engineer and technical reviewer**.
Priority: methodological correctness > reproducibility > clarity > engineering quality > explainability > academic value > portfolio value > sophistication.
Prefer simple, robust solutions. Explicitly flag leakage, invalid validation, misleading metrics and unnecessary complexity.

## Workflow
**Do not implement the whole project at once.**
For substantial work:
1. inspect the repository;
2. identify the current phase;
3. state briefly what will change;
4. modify only the requested scope;
5. run relevant validation/tests;
6. summarize changed files and technical decisions.
Do not implement future features unless requested.

## Problem
- Domain: telecommunications.
- Task: supervised binary classification.
- Positive class: `Churn = Yes`.
- Negative class: `Churn = No`.
- Primary output: churn probability.
- Secondary output: class based on a justified threshold.
- Planned dataset: **Telco Customer Churn**.
The project must answer who is at risk, what factors are associated with predictions, why an individual prediction was made, what threshold is appropriate, and how the model could be monitored.
Never reduce the objective to maximizing accuracy.

## Data rules
- Preserve raw data unchanged.
- Make all transformations reproducible through code.
- Document dataset source/version and a Data Dictionary.
- Validate types, missing values, duplicates, invalid domains and target balance.
- Investigate potential leakage before modeling.

## Leakage prevention
Data leakage is a critical failure.
Never:
- fit preprocessing before train/test separation;
- use test data for feature selection or tuning;
- learn transformations from test data;
- create features using future information;
- use information unavailable at prediction time.
Use `Pipeline` and `ColumnTransformer` where appropriate.

## Experimental protocol
- Keep an isolated final test set.
- Prefer stratified splitting and fixed random seeds.
- Perform model selection and CV only on training data.
- Use the test set only for final candidate evaluation.
Baselines:
1. `DummyClassifier`;
2. Logistic Regression.
Candidate models should be limited and justified, e.g. Logistic Regression, Random Forest, HistGradientBoosting/Gradient Boosting, and optionally XGBoost/LightGBM.
Do not benchmark dozens of algorithms without a hypothesis.

## Metrics
Never choose a model using accuracy alone.
Evaluate as appropriate:
- Precision, Recall and F1;
- ROC-AUC;
- PR-AUC / Average Precision;
- confusion matrix;
- ROC and Precision-Recall curves;
- probability calibration;
- performance across thresholds.
False negatives matter in churn, but maximum recall is not automatically optimal.
Threshold selection must discuss retention cost, customer-loss cost, operational capacity, precision and recall.
If real business costs are unavailable, label hypothetical scenarios explicitly.
Never fabricate business numbers.

## EDA
EDA must answer questions, not merely generate charts.
Investigate target distribution, contract type, tenure, payment method, charges, subscribed services, missing/invalid values, important relationships and possible leakage.
Important visualizations need clear labels and interpretation.
Do not treat association, feature importance or SHAP values as causal evidence.

## Feature engineering
Every engineered feature needs a hypothesis.
For each feature: document motivation, verify no leakage, inspect behavior and measure impact.
Do not create arbitrary features just to increase feature count.

## Explainability
Final work must contain:
- **Global:** coefficients, permutation importance and/or SHAP.
- **Local:** explanation for selected individual predictions.
A future prediction response may contain `churn_probability`, `prediction`, `risk_level`, `threshold` and major contributing factors.

## Reproducibility
Use:
- Python 3.12 unless a documented reason requires another version;
- explicit seeds;
- relative paths;
- versioned dependencies;
- centralized configuration where useful;
- reusable preprocessing pipelines;
- persisted complete inference pipeline;
- documented executable commands.
Never invent metrics, plots, experiments or conclusions.

**Deterministic artefacts.** New generated artefacts — reports, machine-readable
records, figures — must be pure functions of their inputs, so that re-running a
generator on an unchanged repository reproduces them byte for byte and any diff
is a real change. Do not write wall-clock metadata (generation dates, timestamps,
run durations, hostnames) into them: Git already records when a file was
produced, and a clock inside the file guarantees a spurious diff on every rerun.
Reports committed before this convention are not retrofitted.

## Preferred stack
Core: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, joblib, shap, pydantic, pytest.
Use `pyproject.toml` as the primary project configuration/dependency source.
Add dependencies only when they provide clear value.
FastAPI/Streamlit are later portfolio extensions, not initial requirements.

## Code quality
Favor clear names, small focused functions, type hints in reusable code, useful docstrings, explicit error handling, logging instead of `print()` in reusable modules, low coupling and minimal duplication.
Do not create classes when simple functions are sufficient.
Use `pytest`, prioritizing data contracts, transformations, preprocessing, inference, model loading and output schemas.

## Repository architecture
Create directories incrementally, not as empty placeholders.
```text
churn-ml-engineering/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/
├── data/{raw,interim,processed}/
├── notebooks/
├── src/churn/{data,features,modeling,explainability,monitoring}/
├── tests/
├── artifacts/
├── reports/
└── scripts/
```
Notebooks are for exploration, visualization, experimentation and academic narrative.
Reusable logic belongs under `src/`.

## Official phases
1. Foundation — scope, repository, environment, config.
2. Data understanding — acquisition, dictionary, validation.
3. EDA — analysis and hypotheses.
4. Preprocessing — split, imputation, encoding, scaling, pipeline.
5. Baselines — DummyClassifier and Logistic Regression.
6. Feature engineering — hypothesis-driven features.
7. Modeling — controlled model comparison.
8. Tuning — optimize only promising candidates.
9. Final evaluation — untouched test set, threshold, calibration, error analysis.
10. Explainability — global and local.
11. Persistence & inference — trained pipeline and prediction interface.
12. Monitoring design — quality, drift, performance, retraining criteria.
13. Portfolio product — optional API/UI after ML validation.
14. Academic delivery — Colab/notebook, report, figures, presentation/video.
15. GitHub polish — README, architecture, results and reproduction instructions.
Always identify the current phase before expanding scope.

## Monitoring
The final proposal must cover schema/data quality, missing values, unseen categories, feature drift, prediction drift, churn-rate changes, performance once labels arrive, and investigation/retraining triggers.
Do not prescribe arbitrary retraining schedules without evidence.

## Git and security
Never commit secrets, `.env`, virtual environments, caches or unnecessary large artifacts.
Use `.env.example` when needed.
Prefer semantic commits such as:
- `feat: add preprocessing pipeline`
- `fix: prevent leakage during imputation`
- `test: add inference schema tests`
- `docs: document evaluation strategy`

## Academic integrity
All reported results must come from executed experiments.
Never fabricate metrics, visualizations, experiments, dataset characteristics or conclusions.
Academic deliverables must be traceable to repository code and generated artifacts.

## Definition of Done
A relevant task is done when applicable:
- implementation works;
- relevant tests/checks pass;
- no known leakage was introduced;
- output is reproducible;
- docs/config were updated;
- artifacts are stored correctly;
- changed files and decisions are summarized.
For substantial work, respond with: `Implemented` → `Files changed` → `Validation` → `Technical decisions` → `Recommended next step`.

## Final principle
The goal is not: **"I trained a churn classifier."**
The goal is: **"I built a reproducible churn prediction ML system, validated it correctly, analyzed decision thresholds, explained its predictions, packaged inference, and designed monitoring."**
Every technical decision should move the repository toward that outcome.
