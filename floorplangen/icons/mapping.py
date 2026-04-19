"""Room-type -> icon-category mapping.

Per Guidelines §3.7 / §5.7, BTI plans show only fixed fixtures (sanitary,
built-in wardrobes, stairs) + structural elements — free-standing furniture
(bed, sofa, TV) does not appear on any era. Radiators appear inconsistently
on transitional-era plans only.

To avoid discarding the existing icon pack, we keep the category list wide
but filter at placement time via `ROOM_ICON_CATEGORIES_BY_ERA`.
"""
from __future__ import annotations

from ..linework import LineworkSpec


# Categories the icon pack actually ships. Fixed fixtures only, then optional
# tiers. Order = priority order inside a room.
_FIXED_CATEGORIES: dict[str, list[str]] = {
    "bedroom":  ["misc"],            # built-in wardrobe placeholders live in "misc"
    "living":   ["misc"],
    "kitchen":  ["kitchen"],         # stove + sink are fixtures
    "bathroom": ["sanitary"],
    "hall":     ["misc"],
    "storage":  ["misc"],
    "balcony":  [],                  # no fixtures
    "stair":    ["stair"],
}

# Optional "movable" / soft-fixture categories — allowed only on the
# transitional era to match real archive practice.
_TRANSITIONAL_EXTRAS: dict[str, list[str]] = {
    "bedroom": ["bedroom", "radiator"],
    "living":  ["living", "radiator"],
    "kitchen": ["radiator"],
}


def room_icon_categories(room_type: str, spec: LineworkSpec) -> list[str]:
    base = list(_FIXED_CATEGORIES.get(room_type, []))
    if spec.furniture.allow_free_furniture:
        extras = _TRANSITIONAL_EXTRAS.get(room_type, [])
        base = base + [c for c in extras if c not in base]
    elif spec.furniture.allow_radiators:
        if "radiator" not in base:
            base = base + ["radiator"]
    return base
