"""Repo-root resolution and YAML config access.

Every module resolves paths through here rather than relative to the current
working directory, so `pytest`, `make`, and ad-hoc scripts all agree on where
`data/` lives regardless of where they were invoked from.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# src/fir/config.py -> src/fir -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"


def resolve(path: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute one."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


@functools.lru_cache(maxsize=None)
def load_config(name: str) -> dict[str, Any]:
    """Load and cache `configs/<name>.yaml`."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def datasets() -> dict[str, Any]:
    return load_config("datasets")


def models() -> dict[str, Any]:
    return load_config("models")


def pipeline() -> dict[str, Any]:
    return load_config("pipeline")
