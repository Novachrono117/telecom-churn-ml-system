"""The three model families compared in Phase 7.

One question is being asked — *does a non-linear family rank churners better
than the linear baseline?* — so exactly one thing may differ between the
experiments: the estimator. Everything else is inherited, not rebuilt.

**The representation is held constant on purpose.** All three families receive
the identical Phase 4 block::

    TotalChargesCleaner -> StandardScaler (numeric) + OneHotEncoder (categorical)

This common representation is deliberate: it is what makes the estimator the
only thing that varies between M0, M1 and M2. Giving one family a different
encoding would mean measuring *estimator + representation* against *estimator*,
and no delta could then be attributed to the family.

A common representation is not necessarily the optimal one for every family.
``StandardScaler`` does not affect tree split points, and
``HistGradientBoostingClassifier`` can consume raw categories through
``categorical_features`` rather than one-hot columns. Neither option is
exercised here. The consequence is stated rather than judged: these experiments
estimate the performance of the tested configurations **under a common
representation**, not the ceiling of each algorithm. Representation is a
separate experiment, and it is not this one.

**No configuration here is tuned.** The hyperparameters are the declared base
configuration of each family, frozen before any result was seen. Nothing in this
module may become a search: no grid, no random search, no early stopping on a
validation split, no class weighting. Phase 8 is where a hyperparameter becomes
a decision.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from churn.config import get_config
from churn.features.groups import resolve
from churn.features.pipeline import build_feature_preprocessor
from churn.modeling.models import (
    CLASSIFIER_STEP,
    LOGISTIC_REGRESSION,
    PREPROCESSOR_STEP,
    build_logistic_baseline,
)

logger = logging.getLogger(__name__)

RANDOM_FOREST = "random_forest"
HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"

#: Experiment identifiers of the main comparison.
MAIN_EXPERIMENTS: dict[str, str] = {
    LOGISTIC_REGRESSION: "M0",
    RANDOM_FOREST: "M1",
    HIST_GRADIENT_BOOSTING: "M2",
}

#: Experiment identifiers of the contract_tenure sensitivity analysis.
SENSITIVITY_EXPERIMENTS: dict[str, str] = {
    LOGISTIC_REGRESSION: "S0",
    RANDOM_FOREST: "S1",
    HIST_GRADIENT_BOOSTING: "S2",
}

#: Random forest base configuration. Declared, not searched: 100 trees,
#: unlimited depth, ``sqrt`` features per split, bootstrap on. ``n_jobs=1``
#: because the comparison values reproducibility over speed — parallel tree
#: fitting is deterministic here, but keeping a single worker removes the
#: question entirely. ``oob_score=False``: an out-of-bag estimate is a second,
#: differently-computed performance number, and mixing it into a protocol that
#: measures everything by paired cross-validation would invite selecting on it.
RANDOM_FOREST_PARAMS: dict[str, object] = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "oob_score": False,
    "class_weight": None,
    "n_jobs": 1,
}

#: Histogram gradient boosting base configuration — scikit-learn's defaults for
#: every learning parameter. ``early_stopping=False`` is explicit and load
#: bearing: with the default ``"auto"`` this estimator would carve an internal
#: validation split out of each training fold and stop on it, which is model
#: selection performed inside a fold. That would make the number it reports
#: incomparable with the other two families, which see the whole training fold.
#: ``categorical_features=None`` is set explicitly rather than left at the
#: scikit-learn 1.9 default of ``"from_dtype"``. The default would enable native
#: categorical handling for any column arriving with a pandas categorical dtype.
#: Nothing does today — the ``ColumnTransformer`` hands over an all-float array,
#: so the two settings are numerically identical here — but the protocol says
#: native categorical support is not used in this phase, and a setting that
#: depends on an upstream dtype is not a guarantee of that.
HIST_GRADIENT_BOOSTING_PARAMS: dict[str, object] = {
    "loss": "log_loss",
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "max_features": 1.0,
    "early_stopping": False,
    "class_weight": None,
    "categorical_features": None,
}


def build_logistic_classifier() -> BaseEstimator:
    """Return the frozen Phase 5 classifier.

    Taken from :func:`churn.modeling.models.build_logistic_baseline` rather than
    re-declared, so ``C``, ``penalty``, ``class_weight``, ``solver`` and
    ``max_iter`` cannot drift between the baseline and M0. The deprecated
    ``penalty="l2"`` argument is deliberately **not** modernised here: M0 has to
    reproduce the Phase 5 per-fold metrics, and changing the estimator to quiet
    a warning would redefine the reference this phase is measured against.
    """
    return build_logistic_baseline().named_steps[CLASSIFIER_STEP]


def build_random_forest(seed: int | None = None) -> RandomForestClassifier:
    """Return the untuned random forest, unfitted."""
    random_state = get_config().seed if seed is None else seed
    return RandomForestClassifier(random_state=random_state, **RANDOM_FOREST_PARAMS)


def build_hist_gradient_boosting(seed: int | None = None) -> HistGradientBoostingClassifier:
    """Return the untuned histogram gradient boosting model, unfitted."""
    random_state = get_config().seed if seed is None else seed
    return HistGradientBoostingClassifier(
        random_state=random_state, **HIST_GRADIENT_BOOSTING_PARAMS
    )


@dataclass(frozen=True)
class ModelFamily:
    """One family, its estimator factory and why it is in the comparison."""

    name: str
    label: str
    factory: Callable[[], BaseEstimator]
    hypothesis: str
    notes: str = ""

    def build(self) -> BaseEstimator:
        """Return a fresh, unfitted classifier for this family."""
        return self.factory()


FAMILIES: OrderedDict[str, ModelFamily] = OrderedDict(
    (
        (
            LOGISTIC_REGRESSION,
            ModelFamily(
                name=LOGISTIC_REGRESSION,
                label="Logistic regression",
                factory=build_logistic_classifier,
                hypothesis=(
                    "Additive log-odds in the encoded features. It is the reference, not a "
                    "candidate: every delta in this phase is measured against it."
                ),
                notes=(
                    "Frozen at the Phase 5 configuration and reproduced fold by fold before "
                    "any other family is interpreted."
                ),
            ),
        ),
        (
            RANDOM_FOREST,
            ModelFamily(
                name=RANDOM_FOREST,
                label="Random forest",
                factory=build_random_forest,
                hypothesis=(
                    "Averaging deep, decorrelated trees captures interactions and "
                    "non-monotone tenure and charge effects that a single additive slope "
                    "cannot express."
                ),
                notes=(
                    "Fully grown trees on a 46-column one-hot matrix: variance is "
                    "controlled by averaging alone, since no depth or leaf-size limit is "
                    "imposed in this phase."
                ),
            ),
        ),
        (
            HIST_GRADIENT_BOOSTING,
            ModelFamily(
                name=HIST_GRADIENT_BOOSTING,
                label="Histogram gradient boosting",
                factory=build_hist_gradient_boosting,
                hypothesis=(
                    "Sequentially fitting shallow trees to the residual gradient targets "
                    "the errors a linear model leaves behind, which is a different way of "
                    "reaching the same interactions the forest averages over."
                ),
                notes=(
                    "Boosting is the family most sensitive to its learning parameters, so "
                    "its untuned result is the weakest evidence of the three about what the "
                    "family can do — and the strongest about what it does for free."
                ),
            ),
        ),
    )
)

#: Order in which the families are run and reported.
FAMILY_ORDER: tuple[str, ...] = tuple(FAMILIES)


def build_family_pipeline(family: str, feature_groups: Sequence[str] = ()) -> Pipeline:
    """Build the complete estimator for one family.

    The preprocessing block is
    :func:`churn.features.pipeline.build_feature_preprocessor` — the same builder
    the Phase 6 ablation used — so a Phase 7 pipeline with no feature group is
    structurally the Phase 5 baseline, and one with ``contract_tenure`` is
    structurally E3. That is what makes the two reproduction checks meaningful
    rather than approximate.

    Args:
        family: Registered family name.
        feature_groups: Phase 6 feature groups to add. Empty is the main
            comparison's representation: the 19 original features.

    Returns:
        The unfitted pipeline.

    Raises:
        KeyError: If the family or a feature group is not registered.
    """
    if family not in FAMILIES:
        raise KeyError(f"Unknown model family: {family!r}. Known: {sorted(FAMILIES)}.")

    groups = resolve(feature_groups)
    pipeline = Pipeline(
        steps=[
            (PREPROCESSOR_STEP, build_feature_preprocessor(groups)),
            (CLASSIFIER_STEP, FAMILIES[family].build()),
        ]
    )
    logger.debug(
        "Built %s pipeline with feature group(s): %s",
        family,
        ", ".join(group.name for group in groups) or "none (19 original features)",
    )
    return pipeline
