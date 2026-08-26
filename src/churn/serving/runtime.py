"""Runtime compatibility gate: is this interpreter the one the model came from?

A ``.joblib`` is not self-contained. It stores fitted *state* and pickles every
class **by reference**, so unpickling imports whatever ``scikit-learn``,
``numpy`` and ``pandas`` currently define. A minor release that changes an
estimator's private attributes, a default, or an array's dtype promotion rules
can therefore change what the loaded object computes while both model
fingerprints stay identical — the fingerprints describe the stored values, not
the code that will interpret them.

So the versions are checked, and a mismatch **fails the startup**. It is not a
warning: a warning that the runtime differs would be logged once, scrolled past,
and every prediction afterwards would carry an unproven claim.

The authority is ``environment`` in ``reports/decision_policy.json``. Nothing is
invented here — the recorded strings are the expectation, whatever they are.

``python`` is compared at ``major.minor`` because that is the precision the
policy recorded (``"3.12"``); the four libraries are compared exactly, because
the policy recorded them exactly.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn.serving.errors import RuntimeCompatibilityError

logger = logging.getLogger(__name__)

#: Key in ``policy.environment`` compared at ``major.minor`` rather than exactly.
PYTHON_KEY = "python"

#: Keys compared exactly. The names are the policy's, not the import names.
PINNED_LIBRARY_KEYS: tuple[str, ...] = ("scikit_learn", "numpy", "pandas", "joblib")


@dataclass(frozen=True)
class VersionCheck:
    """One component, what the freeze recorded and what is installed."""

    component: str
    expected: str
    actual: str
    compatible: bool


def python_version() -> str:
    """Return the running interpreter as ``major.minor``."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def runtime_versions() -> dict[str, str]:
    """Return the installed versions, keyed as ``policy.environment`` keys them."""
    return {
        PYTHON_KEY: python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def check_runtime_compatibility(
    recorded: Mapping[str, str],
    actual: Mapping[str, str] | None = None,
) -> list[VersionCheck]:
    """Compare the recorded environment with the running one, component by component.

    Returns the full list rather than raising, so a caller can report every
    divergence at once instead of one restart at a time.

    Args:
        recorded: ``environment`` from the decision policy.
        actual: Installed versions. Defaults to :func:`runtime_versions`.

    Returns:
        One :class:`VersionCheck` per component the policy recorded.
    """
    installed = dict(runtime_versions() if actual is None else actual)
    checks: list[VersionCheck] = []
    for component in (PYTHON_KEY, *PINNED_LIBRARY_KEYS):
        if component not in recorded:
            continue
        expected = str(recorded[component])
        found = installed.get(component, "<absent>")
        checks.append(
            VersionCheck(
                component=component,
                expected=expected,
                actual=found,
                compatible=found == expected,
            )
        )
    return checks


def verify_runtime_compatibility(
    recorded: Mapping[str, str],
    actual: Mapping[str, str] | None = None,
) -> list[VersionCheck]:
    """Return the checks, or raise naming every component that diverged.

    Raises:
        RuntimeCompatibilityError: If any recorded component does not match.
    """
    checks = check_runtime_compatibility(recorded, actual)
    if not checks:
        raise RuntimeCompatibilityError(
            "The decision policy records no environment, so the runtime cannot be "
            "checked against the environment the model was produced in."
        )
    divergent = [check for check in checks if not check.compatible]
    if divergent:
        raise RuntimeCompatibilityError(
            "The serving runtime is not the environment the frozen model was produced in:\n  "
            + "\n  ".join(
                f"{check.component}: installed {check.actual}, freeze recorded {check.expected}"
                for check in divergent
            )
            + "\nThe pipeline unpickles its classes by reference, so a different library "
            "version can change what the loaded object computes while every stored "
            "fingerprint stays identical. Startup fails closed."
        )
    logger.info(
        "Runtime matches the freeze: %s",
        ", ".join(f"{check.component}={check.actual}" for check in checks),
    )
    return checks
