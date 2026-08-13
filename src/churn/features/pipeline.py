"""Model pipelines parameterised by feature group.

The whole ablation runs through one builder. Passing no groups must reproduce
the Phase 5 baseline exactly — same steps, same classifier, same everything —
because otherwise a measured delta could be an artefact of two pipelines being
assembled differently rather than of the feature under test.

Step order matters and is fixed::

    TotalChargesCleaner   deterministic; makes TotalCharges numeric
    └── feature groups    appended derived columns, in registry order
        └── ColumnTransformer   StandardScaler + OneHotEncoder, widened
            └── LogisticRegression   frozen Phase 5 configuration

The cleaner runs first because ``historical_average_charge`` divides by a
``TotalCharges`` that must already be numeric. The encoder runs last because
every derived column has to be scaled with the fold's own statistics, exactly
like the original ones.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sklearn.pipeline import Pipeline

from churn.features.groups import FeatureGroup, added_categorical, added_numeric, resolve
from churn.modeling.models import CLASSIFIER_STEP, build_logistic_baseline
from churn.preprocessing.contracts import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from churn.preprocessing.pipeline import CLEANER_STEP, ENCODER_STEP, build_column_encoder
from churn.preprocessing.transformers import TotalChargesCleaner

logger = logging.getLogger(__name__)

#: Prefix of the derived-feature steps, so they are recognisable in a repr.
#: Single underscore: scikit-learn reserves ``__`` for parameter routing and
#: rejects step names containing it.
FEATURE_STEP_PREFIX = "derive_"


def build_feature_preprocessor(groups: Sequence[FeatureGroup] = ()) -> Pipeline:
    """Build the preprocessing pipeline widened by ``groups``.

    Args:
        groups: Feature groups whose columns are appended before encoding.

    Returns:
        The unfitted preprocessing pipeline.
    """
    steps: list[tuple[str, object]] = [(CLEANER_STEP, TotalChargesCleaner())]
    steps.extend((f"{FEATURE_STEP_PREFIX}{group.name}", group.build()) for group in groups)
    steps.append(
        (
            ENCODER_STEP,
            build_column_encoder(
                (*NUMERIC_FEATURES, *added_numeric(groups)),
                (*CATEGORICAL_FEATURES, *added_categorical(groups)),
            ),
        )
    )
    return Pipeline(steps=steps)


def build_model_pipeline(feature_groups: Sequence[str] = ()) -> Pipeline:
    """Build the frozen baseline classifier on top of a widened feature set.

    The classifier is taken from :func:`churn.modeling.models.build_logistic_baseline`,
    so ``C``, ``penalty``, ``class_weight``, ``solver`` and ``max_iter`` cannot
    drift between the baseline and a candidate: there is one definition.

    Args:
        feature_groups: Names of the groups to add. Empty reproduces Phase 5.

    Returns:
        The unfitted pipeline.

    Raises:
        KeyError: If a group name is not registered.
    """
    groups = resolve(feature_groups)
    classifier = build_logistic_baseline().named_steps[CLASSIFIER_STEP]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_feature_preprocessor(groups)),
            (CLASSIFIER_STEP, classifier),
        ]
    )
    logger.debug(
        "Built pipeline with %d feature group(s): %s",
        len(groups),
        ", ".join(group.name for group in groups) or "none (baseline)",
    )
    return pipeline


def transformed_width(pipeline: Pipeline) -> int:
    """Return the number of columns a fitted pipeline feeds to the classifier."""
    encoder = pipeline.named_steps["preprocessor"].named_steps[ENCODER_STEP]
    return len(encoder.get_feature_names_out())
