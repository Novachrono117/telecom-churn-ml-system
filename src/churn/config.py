"""Central configuration for the churn project.

Values live in ``configs/base.toml`` so that the random seed, data paths, target
definition and split policy have a single source of truth shared by notebooks,
scripts and ``src`` modules.

Paths are declared relative to the repository root in the TOML file and resolved
to absolute paths here, so no module depends on the current working directory.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

#: Repository root, resolved from this file's location (``src/churn/config.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Default configuration file shipped with the repository.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.toml"


class DataPaths(BaseModel):
    """Absolute locations of the data layers. Directories may not exist yet."""

    model_config = ConfigDict(frozen=True)

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path


class TargetConfig(BaseModel):
    """Definition of the supervised target and its class labels."""

    model_config = ConfigDict(frozen=True)

    column: str = Field(min_length=1)
    positive_label: str = Field(min_length=1)
    negative_label: str = Field(min_length=1)


class SplitConfig(BaseModel):
    """Train/test separation policy for the held-out evaluation set."""

    model_config = ConfigDict(frozen=True)

    test_size: float = Field(gt=0.0, lt=1.0)
    stratify: bool


class Config(BaseModel):
    """Validated project configuration."""

    model_config = ConfigDict(frozen=True)

    project_name: str = Field(min_length=1)
    seed: int = Field(ge=0)
    data: DataPaths
    target: TargetConfig
    split: SplitConfig


def load_config(
    path: Path | None = None,
    project_root: Path | None = None,
) -> Config:
    """Read and validate the TOML configuration.

    Args:
        path: Configuration file to read. Defaults to ``configs/base.toml``.
        project_root: Base used to resolve the relative data paths. Defaults to
            the repository root.

    Returns:
        The validated :class:`Config`.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError: If a required section or key is missing.
        pydantic.ValidationError: If a value violates the expected contract.
    """
    config_path = path or DEFAULT_CONFIG_PATH
    root = project_root or PROJECT_ROOT

    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    data = raw["data"]
    return Config(
        project_name=raw["project"]["name"],
        seed=raw["random"]["seed"],
        data=DataPaths(
            raw_dir=root / data["raw_dir"],
            interim_dir=root / data["interim_dir"],
            processed_dir=root / data["processed_dir"],
        ),
        target=TargetConfig(**raw["target"]),
        split=SplitConfig(**raw["split"]),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the default project configuration, loaded once per process."""
    return load_config()
