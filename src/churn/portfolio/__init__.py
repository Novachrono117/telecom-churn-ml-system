"""Phase 13 — the demonstrable surface of a system that was already validated.

This package adds **no modelling**. It exists so that a person can put a customer
in front of the frozen model and see, in one screen, what the model estimated, what
threshold that estimate was compared against, and which parts of the input moved the
score. Everything it reports is read out of artefacts the earlier phases froze.

Three boundaries define it, and each one is enforced rather than described.

**It never scores.** :mod:`churn.portfolio.explanation` receives a prepared feature
row and the probability the serving boundary already produced, and decomposes that
row. It has no second inference path, so ``/api/v1/explain`` cannot disagree with
``/api/v1/predict`` about a probability — the number is the same object, not a
recomputation that happens to match.

**It approximates nothing.** A logistic regression's response surface is known in
closed form, so the local explanation is an algebraic identity, not an estimate::

    logit(P) = intercept + sum_j coefficient_j * z_j
    P        = sigmoid(logit)

No SHAP, no LIME, no permutation importance, no sampling and no random seed. The
primitives come from :mod:`churn.modeling.interpretation`, which established that
decomposition in Phase 10 — reused, not reimplemented, because two implementations
of one identity are two chances to drift.

**It publishes an explanation only if the arithmetic closes.** Every explained
request re-derives ``intercept + sum(contributions)`` and checks it against the
probability that was served, at Phase 10's tolerance. A decomposition that does not
reproduce the model it claims to describe is not an explanation of that model, so a
failure raises instead of being shown with a caveat.

Layering::

    metadata     churn.portfolio.metadata      versioned offline summary of Phase 9D
    coupling     churn.portfolio.coupling      which raw features move together
       ↓
    explanation  churn.portfolio.explanation   exact decomposition + the gate
       ↓
    (serving)    churn.serving.portfolio       the HTTP seam and the static UI

Importing this package pulls in no web framework, reads no dataset and touches no
holdout.
"""

from churn.portfolio.coupling import (
    COUPLING_BLOCKS,
    CouplingBlock,
    active_blocks,
    coupled_features,
)
from churn.portfolio.explanation import (
    EXPLANATION_METHOD,
    Contribution,
    ExplanationError,
    GroupedContribution,
    LocalExplanation,
    Reconstruction,
    explain_prepared_row,
)
from churn.portfolio.metadata import (
    PORTFOLIO_METADATA_RELATIVE_PATH,
    PortfolioMetadata,
    load_portfolio_metadata,
    verify_portfolio_metadata,
)

__all__ = [
    "COUPLING_BLOCKS",
    "EXPLANATION_METHOD",
    "PORTFOLIO_METADATA_RELATIVE_PATH",
    "Contribution",
    "CouplingBlock",
    "ExplanationError",
    "GroupedContribution",
    "LocalExplanation",
    "PortfolioMetadata",
    "Reconstruction",
    "active_blocks",
    "coupled_features",
    "explain_prepared_row",
    "load_portfolio_metadata",
    "verify_portfolio_metadata",
]
