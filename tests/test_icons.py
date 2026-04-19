"""Chapter 7 acceptance tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floorplangen import GeneratorConfig
from floorplangen.footprint import generate_footprint
from floorplangen.icons import place_icons
from floorplangen.icons.loader import get_cached_pack, load_icon_pack
from floorplangen.rng import bundle_from_seed
from floorplangen.subdivision import subdivide
from floorplangen.walls import build_wall_graph


IMG = (512, 512)


def test_icon_pack_loads(icon_pack_dir: Path) -> None:
    pack = load_icon_pack(icon_pack_dir)
    assert len(pack.by_category) > 0
    for cat, assets in pack.by_category.items():
        for a in assets:
            assert a.rgba.ndim == 3 and a.rgba.shape[2] == 4
            assert a.rgba.dtype == np.uint8


def test_icon_pack_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_icon_pack(tmp_path / "does_not_exist")


@pytest.mark.parametrize("seed", [1, 7, 42, 100])
def test_icons_placed_within_rooms(seed: int, icon_pack_dir: Path) -> None:
    rng = bundle_from_seed(seed)
    fp = generate_footprint(IMG, 0.5, rng["footprint"], margin=16)
    cfg = GeneratorConfig()
    cfg.icon_pack_dir = icon_pack_dir
    rooms = subdivide(fp, cfg, rng["subdivision"])
    walls = build_wall_graph(fp, rooms, cfg, rng["walls"], rng["diagonal"], era="scan")
    icons = place_icons(fp, rooms, walls, cfg, rng["icons"], era="scan")

    for icon in icons:
        assert icon.footprint_polygon.is_valid
        # Must intersect at least one room by majority
        best = max(rooms, key=lambda r: r.polygon.intersection(icon.footprint_polygon).area)
        overlap = best.polygon.intersection(icon.footprint_polygon).area
        assert overlap / max(icon.footprint_polygon.area, 1.0) > 0.75
