"""Regenerate pinned reference fixtures for regression tests.

Produces a 21-image reference set (7 primitives × 3 eras) at seed=42 so that
CI can diff any future generator changes against a known-good baseline.

Usage:
    python scripts/gen_reference_fixtures.py        # regenerate all
    python scripts/gen_reference_fixtures.py --verify  # compare against stored
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from floorplangen import GeneratorConfig, generate_sample  # noqa: E402

PRIMITIVES = ["RECT", "L", "T", "U", "Z", "STAIR", "BEVEL"]
ERAS = ["soviet", "transitional", "modern"]
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "reference"
SEED = 42
IMAGE_SIZE = (512, 512)


def _hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _generate(primitive: str, era: str):
    cfg = GeneratorConfig()
    cfg.icon_pack_dir = ROOT / "icons"
    cfg.forced_primitive = primitive
    return generate_sample(
        seed=SEED, image_size=IMAGE_SIZE, era=era,
        config=cfg, icon_pack_dir=str(cfg.icon_pack_dir),
    )


def regenerate() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for prim in PRIMITIVES:
        for era in ERAS:
            stem = f"{prim}_{era}"
            sample = _generate(prim, era)
            cv2.imwrite(str(FIXTURES_DIR / f"{stem}_mask.png"), sample.mask)
            cv2.imwrite(str(FIXTURES_DIR / f"{stem}_image.png"), sample.image)
            manifest[stem] = {
                "mask_sha": _hash(sample.mask),
                "class_mask_sha": _hash(sample.class_mask),
                "image_shape": list(sample.image.shape),
                "mask_shape": list(sample.mask.shape),
                "wall_count": sample.metadata["wall_count"],
                "room_count": sample.metadata["room_count"],
            }
    (FIXTURES_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Regenerated {len(manifest)} fixtures at {FIXTURES_DIR}")


def verify() -> int:
    manifest_path = FIXTURES_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}; run without --verify first")
        return 2
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for prim in PRIMITIVES:
        for era in ERAS:
            stem = f"{prim}_{era}"
            sample = _generate(prim, era)
            got = _hash(sample.mask)
            exp = stored.get(stem, {}).get("mask_sha")
            if got != exp:
                mismatches.append(f"{stem}: mask {got} != {exp}")
    if mismatches:
        print("MISMATCH:")
        for m in mismatches:
            print(f"  {m}")
        return 1
    print(f"All {len(PRIMITIVES) * len(ERAS)} fixtures match.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args(argv)
    if args.verify:
        return verify()
    regenerate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
