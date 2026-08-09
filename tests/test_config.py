"""Contract tests for the centralized project configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from churn.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_config, load_config


def test_default_config_file_exists() -> None:
    assert DEFAULT_CONFIG_PATH.is_file()


def test_project_root_points_to_repository_root() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


def test_load_config_exposes_seed_and_target() -> None:
    config = load_config()

    assert config.seed == 42
    assert config.target.column == "Churn"
    assert config.target.positive_label == "Yes"
    assert config.target.negative_label == "No"
    assert config.target.positive_label != config.target.negative_label


def test_split_policy_is_a_valid_holdout() -> None:
    split = load_config().split

    assert 0.0 < split.test_size < 1.0
    assert split.stratify is True


def test_data_paths_are_absolute_and_under_project_root() -> None:
    data = load_config().data

    for path in (data.raw_dir, data.interim_dir, data.processed_dir):
        assert path.is_absolute()
        assert PROJECT_ROOT in path.parents


def test_config_is_immutable() -> None:
    config = load_config()

    with pytest.raises(ValidationError):
        config.seed = 7


def test_get_config_is_cached() -> None:
    assert get_config() is get_config()


def test_invalid_test_size_is_rejected(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text(
        DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
            "test_size = 0.2", "test_size = 1.5"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(path=bad_config)
