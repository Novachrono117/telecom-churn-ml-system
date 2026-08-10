"""Leakage-safe preprocessing: split, feature contract, transformers, pipeline.

``load_holdout`` is intentionally **not** exported. Every phase before the final
evaluation works from :func:`~churn.preprocessing.splitting.load_training_pool`;
reaching the holdout requires importing ``churn.preprocessing.splitting``
explicitly, which makes it visible in review.
"""

from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    STRUCTURAL_BLANK_FEATURE,
    build_feature_matrix,
    prepare_features,
    target_column,
    validate_feature_matrix,
    validate_no_unexpected_blanks,
    validate_source_frame,
)
from churn.preprocessing.exceptions import DataQualityError, FeatureContractError
from churn.preprocessing.manifest import (
    MANIFEST_PATH,
    MANIFEST_VERSION,
    SplitManifest,
    SplitManifestMismatchError,
    build_split_manifest,
    load_split_manifest,
    verify_split_manifest,
    write_split_manifest,
)
from churn.preprocessing.pipeline import build_preprocessor, feature_names_out
from churn.preprocessing.splitting import (
    DatasetSplit,
    SplitIntegrityError,
    fingerprint_ids,
    load_training_pool,
    split_dataset,
)
from churn.preprocessing.target import UnknownTargetLabelError, encode_target
from churn.preprocessing.transformers import TotalChargesCleaner, clean_total_charges

__all__ = [
    "CATEGORICAL_FEATURES",
    "FEATURE_COLUMNS",
    "ID_COLUMN",
    "MANIFEST_PATH",
    "MANIFEST_VERSION",
    "NUMERIC_FEATURES",
    "STRUCTURAL_BLANK_FEATURE",
    "DataQualityError",
    "DatasetSplit",
    "FeatureContractError",
    "SplitIntegrityError",
    "SplitManifest",
    "SplitManifestMismatchError",
    "TotalChargesCleaner",
    "UnknownTargetLabelError",
    "build_feature_matrix",
    "build_preprocessor",
    "build_split_manifest",
    "clean_total_charges",
    "encode_target",
    "feature_names_out",
    "fingerprint_ids",
    "load_split_manifest",
    "load_training_pool",
    "prepare_features",
    "split_dataset",
    "target_column",
    "validate_feature_matrix",
    "validate_no_unexpected_blanks",
    "validate_source_frame",
    "verify_split_manifest",
    "write_split_manifest",
]
