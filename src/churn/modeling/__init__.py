"""Baseline modeling: model builders, paired cross-validation and result records.

Phase 5 only. There is no tuning, no model registry and no deployment concern
here — those belong to later phases and would be speculative infrastructure now.

The layering is deliberate: :mod:`churn.modeling.evaluation` never loads data.
It receives ``X`` and ``y`` already prepared by
:func:`churn.preprocessing.splitting.load_training_pool`, so no module in this
package can reach the holdout, by accident or otherwise.
"""

from churn.modeling.evaluation import (
    CV_STRATEGY,
    N_SPLITS,
    SHUFFLE,
    FoldResult,
    ModelEvaluation,
    build_splitter,
    evaluate_model,
    evaluate_models,
)
from churn.modeling.metrics import (
    DEFAULT_THRESHOLD,
    METRIC_NAMES,
    THRESHOLD_INDEPENDENT,
    ClassificationMetrics,
    compute_metrics,
    confusion_at_threshold,
    positive_prevalence,
)
from churn.modeling.models import (
    LOGISTIC_REGRESSION,
    MAJORITY,
    STRATIFIED_RANDOM,
    build_baselines,
    build_logistic_baseline,
    build_majority_baseline,
    build_stratified_baseline,
)
from churn.modeling.results import (
    RESULTS_PATH,
    SCHEMA_VERSION,
    BaselineResults,
    build_results,
    load_results,
    write_results,
)

__all__ = [
    "CV_STRATEGY",
    "DEFAULT_THRESHOLD",
    "LOGISTIC_REGRESSION",
    "MAJORITY",
    "METRIC_NAMES",
    "N_SPLITS",
    "RESULTS_PATH",
    "SCHEMA_VERSION",
    "SHUFFLE",
    "STRATIFIED_RANDOM",
    "THRESHOLD_INDEPENDENT",
    "BaselineResults",
    "ClassificationMetrics",
    "FoldResult",
    "ModelEvaluation",
    "build_baselines",
    "build_logistic_baseline",
    "build_majority_baseline",
    "build_results",
    "build_splitter",
    "build_stratified_baseline",
    "compute_metrics",
    "confusion_at_threshold",
    "evaluate_model",
    "evaluate_models",
    "load_results",
    "positive_prevalence",
    "write_results",
]
