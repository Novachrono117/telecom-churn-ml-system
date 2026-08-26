"""The runtime gate: a different library version is a different predictive function."""

from __future__ import annotations

import pytest

from churn.serving.errors import RuntimeCompatibilityError
from churn.serving.runtime import (
    PINNED_LIBRARY_KEYS,
    PYTHON_KEY,
    check_runtime_compatibility,
    python_version,
    runtime_versions,
    verify_runtime_compatibility,
)
from churn.serving.service import ChurnInferenceService


def test_this_environment_is_the_environment_the_model_was_frozen_in(
    service: ChurnInferenceService,
) -> None:
    """The serving environment is isolated from the root one, so this is a real check."""
    recorded = service.artifacts.policy.environment

    checks = verify_runtime_compatibility(recorded)

    assert {check.component for check in checks} == {PYTHON_KEY, *PINNED_LIBRARY_KEYS}
    assert all(check.compatible for check in checks)


def test_the_recorded_versions_are_the_ones_installed(service: ChurnInferenceService) -> None:
    recorded = service.artifacts.policy.environment
    installed = runtime_versions()

    assert installed[PYTHON_KEY] == recorded[PYTHON_KEY] == python_version()
    for key in PINNED_LIBRARY_KEYS:
        assert installed[key] == recorded[key], key


@pytest.mark.parametrize("component", [PYTHON_KEY, *PINNED_LIBRARY_KEYS])
def test_any_divergent_component_fails_closed(component: str) -> None:
    recorded = dict(runtime_versions())
    actual = dict(recorded)
    actual[component] = "0.0.0-not-the-frozen-build"

    with pytest.raises(RuntimeCompatibilityError) as error:
        verify_runtime_compatibility(recorded, actual)

    assert component in str(error.value)
    # A divergence is reported, never warned about and continued past.
    assert "fails closed" in str(error.value)


def test_every_divergence_is_reported_at_once() -> None:
    recorded = dict(runtime_versions())
    actual = {key: "0.0.0" for key in recorded}

    checks = check_runtime_compatibility(recorded, actual)

    assert len(checks) == len(recorded)
    assert not any(check.compatible for check in checks)


def test_an_empty_recorded_environment_is_not_treated_as_compatible() -> None:
    """ "Nothing to compare" is a missing guarantee, not a satisfied one."""
    with pytest.raises(RuntimeCompatibilityError):
        verify_runtime_compatibility({})
