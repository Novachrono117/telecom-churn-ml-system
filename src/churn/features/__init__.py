"""Phase 6 feature engineering: candidate features and the ablation protocol.

The layer is deliberately small. It offers a declarative registry of feature
groups, transformers that express them, and a paired ablation runner — not a
feature store, not a generic transformation framework.

Nothing here loads data. The ablation receives ``X`` and ``y`` already prepared
by :func:`churn.preprocessing.splitting.load_training_pool`, so no module in
this package can reach the holdout.
"""

from churn.features.ablation import (
    INCONCLUSIVE,
    NOT_SUPPORTED,
    PRIMARY_METRIC,
    PROMISING,
    SECONDARY_METRIC,
    ExperimentResult,
    PairedComparison,
    classify,
    compare,
    promising_groups,
    run_ablation,
    run_experiment,
)
from churn.features.deterministic import (
    AUTOMATIC_PAYMENT,
    HISTORICAL_AVERAGE_CHARGE,
    PROTECTIVE_COUNT,
    AutomaticPaymentFlag,
    ContractTenureInteraction,
    HistoricalAverageCharge,
    ProtectiveServiceCount,
)
from churn.features.groups import (
    ABLATION_ORDER,
    FEATURE_GROUPS,
    FeatureGroup,
    added_categorical,
    added_numeric,
    resolve,
)
from churn.features.intensity import CHARGE_INTENSITY, ChargeIntensityByTier
from churn.features.pipeline import (
    build_feature_preprocessor,
    build_model_pipeline,
    transformed_width,
)
from churn.features.results import (
    RESULTS_PATH,
    SCHEMA_VERSION,
    AblationResults,
    BaselineReproductionError,
    build_results,
    load_results,
    verify_baseline_reproduction,
    write_results,
)

__all__ = [
    "ABLATION_ORDER",
    "AUTOMATIC_PAYMENT",
    "CHARGE_INTENSITY",
    "FEATURE_GROUPS",
    "HISTORICAL_AVERAGE_CHARGE",
    "INCONCLUSIVE",
    "NOT_SUPPORTED",
    "PRIMARY_METRIC",
    "PROMISING",
    "PROTECTIVE_COUNT",
    "RESULTS_PATH",
    "SCHEMA_VERSION",
    "SECONDARY_METRIC",
    "AblationResults",
    "AutomaticPaymentFlag",
    "BaselineReproductionError",
    "ChargeIntensityByTier",
    "ContractTenureInteraction",
    "ExperimentResult",
    "FeatureGroup",
    "HistoricalAverageCharge",
    "PairedComparison",
    "ProtectiveServiceCount",
    "added_categorical",
    "added_numeric",
    "build_feature_preprocessor",
    "build_model_pipeline",
    "build_results",
    "classify",
    "compare",
    "load_results",
    "promising_groups",
    "resolve",
    "run_ablation",
    "run_experiment",
    "transformed_width",
    "verify_baseline_reproduction",
    "write_results",
]
