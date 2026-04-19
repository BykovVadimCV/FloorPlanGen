"""Shared pytest fixtures."""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import pytest

from floorplangen import GeneratorConfig


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def default_config() -> GeneratorConfig:
    cfg = GeneratorConfig()
    cfg.icon_pack_dir = REPO_ROOT / "icons"
    return cfg


@pytest.fixture(scope="session")
def icon_pack_dir() -> Path:
    return REPO_ROOT / "icons"
