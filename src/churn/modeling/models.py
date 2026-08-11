"""The three Phase 5 baselines.

Each one is a **complete** estimator: the Phase 4 preprocessing pipeline
followed by a classifier. Nothing is fitted here — the objects are returned
unfitted so that cross-validation refits every learned component inside each
fold.

The two ``DummyClassifier`` baselines carry the same preprocessing block even
though they ignore ``X`` entirely. That is deliberate: the comparison is between
*protocols*, and giving one model a different pipeline shape would introduce a
difference that has nothing to do with the model.

What the baselines answer
-------------------------
* **majority** — what a model that never predicts churn achieves. It fixes the
  floor for any threshold-dependent metric and exposes accuracy as useless here.
* **stratified random** — what guessing at the observed prevalence achieves. It
  fixes the reference for ROC-AUC (0.5) and Average Precision (the prevalence).
* **logistic regression** — how much of the signal a simple, interpretable
  linear model extracts from the 19 original features. Every engineered feature
  proposed in Phase 6 will have to beat this number.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import get_config
from churn.preprocessing.pipeline import build_preprocessor

logger = logging.getLogger(__name__)

MAJORITY = "majority"
STRATIFIED_RANDOM = "stratified_random"
LOGISTIC_REGRESSION = "logistic_regression"

#: Name of the classifier step inside every baseline pipeline.
CLASSIFIER_STEP = "classifier"
PREPROCESSOR_STEP = "preprocessor"

#: scikit-learn's default. Measured on the training pool, lbfgs converges in 42
#: iterations, so the default already carries a comfortable margin and there is
#: no reason to inflate it. Convergence is re-checked per fold at run time.
LOGISTIC_MAX_ITER = 100

#: lbfgs is the scikit-learn default for this problem shape: a dense, moderately
#: sized, L2-penalised binary problem. It is deterministic and does **not**
#: consume ``random_state``, so none is passed — supplying one would suggest a
#: source of randomness that does not exist.
LOGISTIC_SOLVER = "lbfgs"


def _wrap(name: str, classifier: object) -> Pipeline:
    return Pipeline(
        steps=[
            (PREPROCESSOR_STEP, build_preprocessor()),
            (CLASSIFIER_STEP, classifier),
        ]
    )


def build_majority_baseline() -> Pipeline:
    """Baseline A — always predict the majority class."""
    return _wrap(MAJORITY, DummyClassifier(strategy="most_frequent"))


def build_stratified_baseline(seed: int | None = None) -> Pipeline:
    """Baseline B — guess at the prevalence observed in each training fold."""
    random_state = get_config().seed if seed is None else seed
    return _wrap(
        STRATIFIED_RANDOM,
        DummyClassifier(strategy="stratified", random_state=random_state),
    )


def build_logistic_baseline() -> Pipeline:
    """Baseline C — plain logistic regression, untuned.

    ``C=1.0``, ``penalty="l2"`` and ``class_weight=None`` are scikit-learn's
    defaults and are kept on purpose: this is the reference point, not a
    candidate. Any of them becoming a decision belongs to Phase 8.
    """
    return _wrap(
        LOGISTIC_REGRESSION,
        LogisticRegression(
            C=1.0,
            penalty="l2",
            class_weight=None,
            solver=LOGISTIC_SOLVER,
            max_iter=LOGISTIC_MAX_ITER,
        ),
    )


def build_baselines(seed: int | None = None) -> OrderedDict[str, Pipeline]:
    """Return the three baselines, unfitted, in reporting order.

    Args:
        seed: Random state for the stratified dummy. Defaults to the configured
            project seed.

    Returns:
        An ordered mapping of model name to unfitted pipeline.
    """
    models: OrderedDict[str, Pipeline] = OrderedDict()
    models[MAJORITY] = build_majority_baseline()
    models[STRATIFIED_RANDOM] = build_stratified_baseline(seed)
    models[LOGISTIC_REGRESSION] = build_logistic_baseline()
    logger.debug("Built %d baseline pipelines", len(models))
    return models
