"""Reusable exploratory-analysis helpers.

Descriptive tables, association measures and small plotting functions shared by
`notebooks/01_eda.ipynb` and `scripts/run_eda.py`. Nothing here fits or persists
a preprocessing step.
"""

from churn.analysis.association import (
    association_ranking,
    categorical_association,
    cramers_v,
    interpret_cramers_v,
    numeric_comparison,
)
from churn.analysis.descriptive import (
    churn_rate_by,
    churn_rate_matrix,
    numeric_summary_by_target,
    overall_churn_rate,
    value_share,
)
from churn.analysis.frames import (
    CHURN_FLAG,
    TENURE_BAND,
    TENURE_BAND_EDGES,
    TENURE_BAND_LABELS,
    add_churn_flag,
    add_tenure_band,
    coerce_total_charges,
    load_eda_frame,
)

__all__ = [
    "CHURN_FLAG",
    "TENURE_BAND",
    "TENURE_BAND_EDGES",
    "TENURE_BAND_LABELS",
    "add_churn_flag",
    "add_tenure_band",
    "association_ranking",
    "categorical_association",
    "churn_rate_by",
    "churn_rate_matrix",
    "coerce_total_charges",
    "cramers_v",
    "interpret_cramers_v",
    "load_eda_frame",
    "numeric_comparison",
    "numeric_summary_by_target",
    "overall_churn_rate",
    "value_share",
]
