"""Icon pack loader (DESIGN §7.2–§7.4)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False


@dataclass
class IconAsset:
    category: str
    stem: str
    rgba: np.ndarray           # (h, w, 4), straight alpha
    sidecar: dict[str, Any] = field(default_factory=dict)

    @property
    def anchor(self) -> tuple[float, float]:
        return tuple(self.sidecar.get("anchor", [0.5, 0.5]))  # type: ignore[return-value]

    @property
    def allow_rotation(self) -> list[int]:
        return list(self.sidecar.get("allow_rotation", [0, 90, 180, 270]))

    @property
    def era_compatible(self) -> list[str]:
        return list(self.sidecar.get("era_compatible",
                                     ["soviet", "transitional", "modern"]))

    @property
    def min_room_area_frac(self) -> float:
        return float(self.sidecar.get("min_room_area_frac", 0.05))

    @property
    def wall_snap(self) -> str | None:
        return self.sidecar.get("wall_snap")


@dataclass
class IconPack:
    root: Path
    by_category: dict[str, list[IconAsset]] = field(default_factory=dict)

    def eligible(self, category: str, era: str,
                 min_area_frac: float = 0.0) -> list[IconAsset]:
        out = []
        for asset in self.by_category.get(category, []):
            if era not in asset.era_compatible:
                continue
            if asset.min_room_area_frac > min_area_frac:
                continue
            out.append(asset)
        return out


def load_icon_pack(root: Path) -> IconPack:
    if not _HAS_PIL:  # pragma: no cover
        raise ImportError("Pillow is required to load the icon pack")
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Icon pack directory missing: {root}")

    pack = IconPack(root=root)
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name in {"fonts"}:
            continue
        cat = sub.name
        assets: list[IconAsset] = []
        for png in sorted(sub.glob("*.png")):
            with Image.open(png) as im:
                im = im.convert("RGBA")
                rgba = np.asarray(im, dtype=np.uint8)
            sidecar_path = png.with_suffix(".json")
            sidecar: dict[str, Any] = {}
            if sidecar_path.exists():
                try:
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                except Exception:
                    sidecar = {}
            assets.append(IconAsset(category=cat, stem=png.stem, rgba=rgba,
                                    sidecar=sidecar))
        pack.by_category[cat] = assets
    return pack


# Lazy global cache keyed by absolute path
_CACHE: dict[str, IconPack] = {}


def get_cached_pack(root: Path) -> IconPack:
    key = str(Path(root).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_icon_pack(Path(root))
    return _CACHE[key]
