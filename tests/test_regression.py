"""Regression test against pinned reference fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from floorplangen import GeneratorConfig, generate_sample


FIXTURES = Path(__file__).parent / "fixtures" / "reference"
MANIFEST = FIXTURES / "manifest.json"


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip(f"no manifest at {MANIFEST} — run scripts/gen_reference_fixtures.py")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.mark.parametrize("stem", sorted([
    f"{p}_{e}" for p in ["RECT", "L", "T", "U", "Z", "STAIR", "BEVEL"]
    for e in ["scan", "digital", "soviet"]
]))
def test_mask_matches_reference(stem: str, icon_pack_dir: Path) -> None:
    manifest = _load_manifest()
    expected = manifest.get(stem)
    if not expected:
        pytest.skip(f"no reference entry for {stem}")
    primitive, era = stem.split("_", 1)
    cfg = GeneratorConfig()
    cfg.icon_pack_dir = icon_pack_dir
    cfg.forced_primitive = primitive
    sample = generate_sample(
        seed=42, image_size=(512, 512), era=era,
        config=cfg, icon_pack_dir=str(icon_pack_dir),
    )
    assert _hash(sample.mask) == expected["mask_sha"], (
        f"mask regression for {stem}"
    )
