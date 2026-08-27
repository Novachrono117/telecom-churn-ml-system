"""Building the reference profile from a population that is handed to it.

Nothing here loads data. The training pool arrives as an argument, exactly as it
did for Phase 9C's freeze modules, and for the same reason: a module that cannot
reach a partition cannot read the wrong one. The holdout is unreachable from this
file, and so is ``Churn``.

The score half of the profile applies the frozen pipeline to the reference
population. That is a **monitoring reference distribution** and not a performance
estimate: no label is read, no metric is computed against one, and nothing in the
resulting artefact licenses a statement of the form "production should reach this
AP". What it licenses is "the frozen model assigned scores shaped like this to the
population it was fitted on", which is the only thing a later window can be
compared against.

The probability is read through the same frozen primitive the serving boundary
uses — ``churn.modeling.freeze.positive_probability``, which resolves the column
from ``classes_`` — so the reference and production sides of every score comparison
come out of one implementation. ``pipeline.predict`` is not used here or anywhere
else in this package.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn.modeling.freeze import apply_decision_rule, positive_probability
from churn.monitoring.distributions import (
    DEFAULT_N_BINS,
    REFERENCE_QUANTILES,
    UNSEEN_LEVEL,
    bin_counts,
    frequencies,
    level_counts,
    quantile_bin_edges,
    summarise_numeric,
)
from churn.monitoring.drift import EPSILON
from churn.monitoring.reference import (
    PROFILE_NAME,
    REFERENCE_POPULATION,
    SCHEMA_VERSION,
    CategoricalReference,
    NumericReference,
    ProfileConfiguration,
    ReferenceProfile,
    ReferenceProvenance,
    ScoreReference,
    StructuralReference,
)
from churn.monitoring.structural import (
    STRUCTURAL_RULES,
    count_records_with_any_violation,
    count_violations,
)
from churn.preprocessing.contracts import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
)
from churn.preprocessing.transformers import clean_total_charges

logger = logging.getLogger(__name__)

#: Number of bins for the model score. Deciles of the reference score, like the
#: numeric features, so one PSI implementation covers both.
SCORE_N_BINS = DEFAULT_N_BINS

NOTES: tuple[str, ...] = (
    "This profile is a MONITORING REFERENCE DISTRIBUTION. It is not a performance "
    "estimate, and no metric in it was computed against a label. Comparing a production "
    "window with it can show that the population changed; it cannot show that the model "
    "got worse, which is a claim about outcomes that requires ground truth.",
    "The reference population is the frozen training pool. The holdout was consumed by "
    "Phase 9D and is deliberately not used: an operational reference should describe the "
    "population the system was built for, and the training pool is both larger and the "
    "one the model actually saw.",
    "The target was never read. target_used is false, and the build function has no "
    "parameter through which a label could be passed.",
    "Bin edges are computed ONCE here, from the reference, and are stored. A window that "
    "recomputed its own bins would produce a number partly measuring the binning.",
    "The bin partition is (-inf, e1], (e1, e2], ..., (e_last, +inf) — total, so no value "
    "can fall outside it and no mass is clipped. Whether a value lies outside the "
    "reference [min, max] is reported separately as out_of_reference_range_rate.",
    "Duplicate quantile edges are dropped. A concentrated feature such as tenure yields "
    "repeated quantiles, and a zero-width bin can never receive a value while still "
    "contributing a term to a PSI sum.",
    "The score comes from churn.modeling.freeze.positive_probability, the same frozen "
    "primitive the serving boundary uses, with the positive column resolved from "
    "classes_. pipeline.predict is not used.",
    "calibration_policy is NONE, so the score is the model's raw output. Read the score "
    "distribution as a ranking distribution, not as a set of calibrated probabilities.",
    "Every number here is an aggregate. No row of the training pool is stored, and none "
    "could be reconstructed from what is.",
)


def build_reference_profile(
    frame: pd.DataFrame,
    pipeline: Pipeline,
    *,
    threshold: float,
    comparison: str,
    calibration_policy: str,
    raw_sha256: str,
    training_ids_sha256: str,
    model_fingerprint_sha256: str,
    pipeline_sha256: str,
    decision_policy_sha256: str,
    freeze_commit: str,
    serving_commit: str,
    random_seed: int,
    n_bins: int = DEFAULT_N_BINS,
) -> ReferenceProfile:
    """Return the monitoring reference profile for a population.

    Args:
        frame: The reference population, raw-shaped. Only the contracted features
            are selected; anything else it carries — including the target — is
            dropped by :func:`prepare_features` and never read.
        pipeline: The verified frozen pipeline.
        threshold: The frozen decision threshold, from the decision policy.
        comparison: The frozen comparison, from the decision policy.
        calibration_policy: The frozen calibration policy, from the decision policy.
        n_bins: Requested histogram bins.

    Returns:
        A validated :class:`ReferenceProfile`.

    Raises:
        FeatureContractError: If the population does not satisfy the frozen contract.
        DataQualityError: If a value is blank or unreadable where no rule justifies it.
    """
    features = prepare_features(frame.loc[:, list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)])
    n_reference = len(features)
    logger.info("Building the monitoring reference from %d records.", n_reference)

    # TotalCharges arrives as text and is made numeric by the frozen cleaner, which
    # is the same object the pipeline applies. Monitoring must summarise the value
    # the model actually sees, so it reuses that function rather than restating it.
    numeric_frame = clean_total_charges(features)

    numeric: dict[str, NumericReference] = {}
    for feature in NUMERIC_FEATURES:
        values = numeric_frame[feature].astype(float)
        summary = summarise_numeric(values)
        edges = quantile_bin_edges(values, n_bins=n_bins)
        numeric[feature] = NumericReference(
            **summary.as_record(),
            bin_edges=list(edges),
            bin_counts=list(bin_counts(values, edges)),
        )

    categorical: dict[str, CategoricalReference] = {}
    for feature in CATEGORICAL_FEATURES:
        column = features[feature].astype("string")
        known = sorted(str(level) for level in column.dropna().unique())
        counts = level_counts(column, known)
        # __UNSEEN__ is 0 by construction on the population that defined the levels.
        # It is kept in the table so the reference and a window share a support.
        categorical[feature] = CategoricalReference(
            count=n_reference,
            known_levels=known,
            count_by_level=counts,
            frequency_by_level=frequencies(counts),
        )

    scores = np.asarray(positive_probability(pipeline, features), dtype=float)
    score_summary = summarise_numeric(scores)
    score_edges = quantile_bin_edges(scores, n_bins=SCORE_N_BINS)
    positives = int(apply_decision_rule(scores, threshold).sum())
    score = ScoreReference(
        **score_summary.as_record(),
        bin_edges=list(score_edges),
        bin_counts=list(bin_counts(scores, score_edges)),
        threshold=float(threshold),
        comparison=comparison,
        predicted_positive_count=positives,
        predicted_positive_rate=positives / n_reference,
        calibration_policy=calibration_policy,
    )

    violations = count_violations(features)
    any_violation = count_records_with_any_violation(features)
    structural = StructuralReference(
        rules={rule.name: rule.relationship for rule in STRUCTURAL_RULES},
        violations_by_rule=violations,
        records_with_any_violation=any_violation,
        violation_rate=any_violation / n_reference,
    )

    return ReferenceProfile(
        schema_version=SCHEMA_VERSION,
        profile=PROFILE_NAME,
        built_in_phase="12",
        provenance=ReferenceProvenance(
            reference_population=REFERENCE_POPULATION,
            n_reference=n_reference,
            holdout_used=False,
            target_used=False,
            raw_sha256=raw_sha256,
            training_ids_sha256=training_ids_sha256,
            model_fingerprint_sha256=model_fingerprint_sha256,
            pipeline_sha256=pipeline_sha256,
            decision_policy_sha256=decision_policy_sha256,
            freeze_commit=freeze_commit,
            serving_commit=serving_commit,
            random_seed=random_seed,
        ),
        configuration=ProfileConfiguration(
            n_bins_requested=n_bins,
            binning_strategy="reference quantiles, duplicates dropped",
            bin_interval_convention="(-inf, e1], (e1, e2], ..., (e_last, +inf)",
            quantiles_recorded=list(REFERENCE_QUANTILES),
            unseen_level=UNSEEN_LEVEL,
            numeric_features=list(NUMERIC_FEATURES),
            categorical_features=list(CATEGORICAL_FEATURES),
            psi_epsilon=EPSILON,
            numeric_drift_metric="population_stability_index",
            categorical_drift_metric="total_variation_distance",
            score_drift_metric="population_stability_index",
        ),
        numeric=numeric,
        categorical=categorical,
        score=score,
        structural=structural,
        notes=list(NOTES),
    )


__all__ = ["NOTES", "SCORE_N_BINS", "build_reference_profile"]
