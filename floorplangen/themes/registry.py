"""Theme plugin registry (DESIGN §9.6)."""
from __future__ import annotations

from .base import EraTheme, EraThemePlugin

_REGISTRY: dict[str, EraThemePlugin] = {}


def register_theme(plugin: EraThemePlugin) -> None:
    _REGISTRY[plugin.era_id] = plugin


def get_theme(era_id: str, rng, image_size: tuple[int, int]) -> EraTheme:
    if era_id not in _REGISTRY:
        raise KeyError(f"Unknown era: {era_id!r}. "
                       f"Registered eras: {sorted(_REGISTRY)}")
    return _REGISTRY[era_id].build_theme(rng, image_size)


def list_themes() -> list[str]:
    return sorted(_REGISTRY)
