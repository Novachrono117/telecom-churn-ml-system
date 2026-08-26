"""The serving configuration may move the deployment. It may not move the model."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from churn.serving.errors import ServingConfigurationError
from churn.serving.settings import (
    DEFAULT_MAX_BATCH_SIZE,
    FORBIDDEN_ENV_VARS,
    MAX_ALLOWED_BATCH_SIZE,
    ServingSettings,
    default_policy_path,
    load_settings,
)


def test_defaults_point_at_the_repository_artefacts(repo_root: Path) -> None:
    settings = load_settings(env={}, root=repo_root)

    assert settings.policy_path == repo_root / "reports" / "decision_policy.json"
    assert settings.policy_path.is_file()
    # None means "use the path the policy itself records", resolved at startup.
    assert settings.pipeline_path is None
    assert settings.max_batch_size == DEFAULT_MAX_BATCH_SIZE


def test_operational_values_are_configurable(repo_root: Path, tmp_path: Path) -> None:
    settings = load_settings(
        env={
            "CHURN_SERVING_HOST": "0.0.0.0",
            "CHURN_SERVING_PORT": "9001",
            "CHURN_SERVING_MAX_BATCH_SIZE": "25",
            "CHURN_SERVING_MODEL_PATH": str(tmp_path / "model.joblib"),
        },
        root=repo_root,
    )

    assert (settings.host, settings.port, settings.max_batch_size) == ("0.0.0.0", 9001, 25)
    assert settings.pipeline_path == tmp_path / "model.joblib"


@pytest.mark.parametrize("name", FORBIDDEN_ENV_VARS)
def test_a_frozen_decision_cannot_be_overridden_from_the_environment(name: str) -> None:
    """Refused, not ignored.

    Ignoring the variable would be worse than failing on it: the operator would
    believe the threshold had moved, and every decision would be attributed to a
    rule nobody applied.
    """
    with pytest.raises(ServingConfigurationError) as error:
        load_settings(env={name: "0.5"})

    assert name in str(error.value)


def test_the_threshold_is_not_a_settings_field() -> None:
    """There is no field to set, so there is nothing to forget to guard."""
    fields = set(ServingSettings.model_fields)

    assert not fields & {"threshold", "calibration", "calibration_policy", "positive_class"}
    assert fields == {"policy_path", "pipeline_path", "max_batch_size", "host", "port"}


def test_unknown_settings_fields_are_rejected(repo_root: Path) -> None:
    with pytest.raises(ValidationError):
        ServingSettings(policy_path=default_policy_path(repo_root), threshold=0.5)


def test_settings_are_immutable(repo_root: Path) -> None:
    """Read by every request, rebuilt by none."""
    settings = load_settings(env={}, root=repo_root)

    with pytest.raises(ValidationError):
        settings.max_batch_size = 1


@pytest.mark.parametrize("value", ["0", str(MAX_ALLOWED_BATCH_SIZE + 1)])
def test_the_batch_limit_is_bounded_on_both_sides(value: str) -> None:
    with pytest.raises(ValidationError):
        load_settings(env={"CHURN_SERVING_MAX_BATCH_SIZE": value})


def test_a_non_numeric_batch_limit_fails_loudly() -> None:
    with pytest.raises(ServingConfigurationError):
        load_settings(env={"CHURN_SERVING_MAX_BATCH_SIZE": "many"})
