# Data Dictionary — Telco Customer Churn

Generated from `WA_Fn-UseC_-Telco-Customer-Churn.csv` (SHA-256 `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`).
7043 rows x 21 columns.

`dtype` and `domain / range` are **observed in the file**. `description` comes
from the dataset source documentation and is not inferred from the data.

| # | Column | dtype | Likely role | Description (source doc) | Domain / range | Missing | Blank | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `customerID` | `str` | identifier | Unique customer identifier. | 7043 distinct values | 0 | 0 | non-numeric with too many distinct values to enumerate |
| 2 | `gender` | `str` | categorical feature | Whether the customer is male or female. | `Female`, `Male` | 0 | 0 | — |
| 3 | `SeniorCitizen` | `int64` | numeric feature | Whether the customer is a senior citizen (1) or not (0). | [0, 1] | 0 | 0 | — |
| 4 | `Partner` | `str` | categorical feature | Whether the customer has a partner. | `No`, `Yes` | 0 | 0 | — |
| 5 | `Dependents` | `str` | categorical feature | Whether the customer has dependents. | `No`, `Yes` | 0 | 0 | — |
| 6 | `tenure` | `int64` | numeric feature | Number of months the customer has stayed with the company. | [0, 72] | 0 | 0 | — |
| 7 | `PhoneService` | `str` | categorical feature | Whether the customer has phone service. | `No`, `Yes` | 0 | 0 | — |
| 8 | `MultipleLines` | `str` | categorical feature | Whether the customer has multiple lines. | `No`, `No phone service`, `Yes` | 0 | 0 | — |
| 9 | `InternetService` | `str` | categorical feature | Customer's internet service provider (DSL, Fiber optic, No). | `DSL`, `Fiber optic`, `No` | 0 | 0 | — |
| 10 | `OnlineSecurity` | `str` | categorical feature | Whether the customer has online security. | `No`, `No internet service`, `Yes` | 0 | 0 | — |
| 11 | `OnlineBackup` | `str` | categorical feature | Whether the customer has online backup. | `No`, `No internet service`, `Yes` | 0 | 0 | — |
| 12 | `DeviceProtection` | `str` | categorical feature | Whether the customer has device protection. | `No`, `No internet service`, `Yes` | 0 | 0 | — |
| 13 | `TechSupport` | `str` | categorical feature | Whether the customer has tech support. | `No`, `No internet service`, `Yes` | 0 | 0 | — |
| 14 | `StreamingTV` | `str` | categorical feature | Whether the customer has streaming TV. | `No`, `No internet service`, `Yes` | 0 | 0 | — |
| 15 | `StreamingMovies` | `str` | categorical feature | Whether the customer has streaming movies. | `No`, `No internet service`, `Yes` | 0 | 0 | — |
| 16 | `Contract` | `str` | categorical feature | Contract term of the customer (Month-to-month, One year, Two year). | `Month-to-month`, `One year`, `Two year` | 0 | 0 | — |
| 17 | `PaperlessBilling` | `str` | categorical feature | Whether the customer uses paperless billing. | `No`, `Yes` | 0 | 0 | — |
| 18 | `PaymentMethod` | `str` | categorical feature | The customer's payment method. | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` | 0 | 0 | — |
| 19 | `MonthlyCharges` | `float64` | numeric feature | The amount charged to the customer monthly. | [18.25, 118.75] | 0 | 0 | — |
| 20 | `TotalCharges` | `str` | numeric feature | The total amount charged to the customer. | [18.8, 8684.8] | 0 | 11 | 11 whitespace-only cell(s) |
| 21 | `Churn` | `str` | target | Whether the customer churned. | `No`, `Yes` | 0 | 0 | — |
