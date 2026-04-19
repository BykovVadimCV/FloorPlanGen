"""Room-type → icon-category mapping (DESIGN §7.5)."""
from __future__ import annotations

# Each room type draws icons from the listed categories, in priority order.
ROOM_ICON_CATEGORIES: dict[str, list[str]] = {
    "bedroom":  ["bedroom", "radiator"],
    "living":   ["living", "radiator"],
    "kitchen":  ["kitchen", "radiator"],
    "bathroom": ["sanitary"],
    "hall":     ["misc"],
    "storage":  ["misc"],
    "balcony":  ["misc"],
    "stair":    ["stair"],
}
