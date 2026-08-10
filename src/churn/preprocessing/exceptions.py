"""Errors raised by the preprocessing layer.

They live in their own module because both the feature contract and the
``TotalCharges`` cleaner need to raise them, and neither should have to import
the other to do so.

The distinction is deliberate:

* :class:`FeatureContractError` — the *schema* is wrong (a column is missing,
  undeclared, or has an impossible dtype). No amount of data would fix it.
* :class:`DataQualityError` — the schema is right but a *value* cannot be used,
  and repairing it silently would invent information.
"""

from __future__ import annotations


class FeatureContractError(ValueError):
    """A frame does not satisfy the declared feature contract."""


class DataQualityError(ValueError):
    """A value cannot be read and cannot be repaired without inventing information."""
