"""Column semantics documented by the dataset source.

These descriptions come from the published documentation of the Kaggle dataset
``blastchar/telco-customer-churn`` (IBM sample "Telco customer churn"). They are
**not** derived from the data itself and are labeled as such in every generated
artifact. Columns absent from this mapping are reported as undocumented rather
than guessed.
"""

from __future__ import annotations

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "customerID": "Unique customer identifier.",
    "gender": "Whether the customer is male or female.",
    "SeniorCitizen": "Whether the customer is a senior citizen (1) or not (0).",
    "Partner": "Whether the customer has a partner.",
    "Dependents": "Whether the customer has dependents.",
    "tenure": "Number of months the customer has stayed with the company.",
    "PhoneService": "Whether the customer has phone service.",
    "MultipleLines": "Whether the customer has multiple lines.",
    "InternetService": "Customer's internet service provider (DSL, Fiber optic, No).",
    "OnlineSecurity": "Whether the customer has online security.",
    "OnlineBackup": "Whether the customer has online backup.",
    "DeviceProtection": "Whether the customer has device protection.",
    "TechSupport": "Whether the customer has tech support.",
    "StreamingTV": "Whether the customer has streaming TV.",
    "StreamingMovies": "Whether the customer has streaming movies.",
    "Contract": "Contract term of the customer (Month-to-month, One year, Two year).",
    "PaperlessBilling": "Whether the customer uses paperless billing.",
    "PaymentMethod": "The customer's payment method.",
    "MonthlyCharges": "The amount charged to the customer monthly.",
    "TotalCharges": "The total amount charged to the customer.",
    "Churn": "Whether the customer churned.",
}

UNDOCUMENTED = "_Not documented by the source; semantics not inferred._"


def describe(column: str) -> str:
    """Return the documented description of a column, or an explicit placeholder."""
    return COLUMN_DESCRIPTIONS.get(column, UNDOCUMENTED)
