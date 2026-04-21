"""Top-level generator API (DESIGN §1.3).

    result = generate_sample(seed=42, image_size=(512, 512))
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import GeneratorConfig
from .mask import MaskInputs, encode_class_mask, validate_mask, write_mask
from .rng import bundle_from_seed
from .types import GeneratorOutput, WallGraph


def generate_sample(
    seed: int = 0,
    image_size: tuple[int, int] = (512, 512),
    era: str | None = None,
    era_mix: dict[str, float] | None = None,
    monochrome_prob: float = 0.70,
    aggression: float | None = None,
    icon_pack_dir: str | None = None,
    augmentation_preset: str | None = None,
    config: GeneratorConfig | None = None,
    _max_retries: int = 4,
) -> GeneratorOutput:
    """Generate one synthetic floor-plan sample per DESIGN §1.3 contract."""
    # ── Resolve config ──────────────────────────────────────────────────────
    if config is None:
        config = GeneratorConfig()
    if image_size is not None:
        config.image_size = image_size
    if era is not None:
        config.era = era  # type: ignore[assignment]
    if era_mix is not None:
        config.era_mix = dict(era_mix)
    config.monochrome_prob = monochrome_prob
    if aggression is not None:
        config.aggression = aggression
    if icon_pack_dir is not None:
        from pathlib import Path
        config.icon_pack_dir = Path(icon_pack_dir)
    if augmentation_preset is not None:
        config.augmentation_preset = augmentation_preset  # type: ignore[assignment]

    discarded: list[int] = []
    current_seed = int(seed)
    last_err: list[str] = []
    for _attempt in range(_max_retries):
        try:
            out = _try_generate(current_seed, config)
            errs = validate_mask(out.mask, None, require_wall_fraction=False)
            if errs:
                discarded.append(current_seed)
                last_err = errs
                current_seed = (current_seed * 2654435761 + 1) & 0xFFFFFFFF
                continue
            if discarded:
                out.metadata["discarded_seeds"] = discarded
            return out
        except Exception as ex:  # noqa: BLE001 — regenerate with a fresh seed
            discarded.append(current_seed)
            last_err = [f"exception: {ex!r}"]
            current_seed = (current_seed * 2654435761 + 1) & 0xFFFFFFFF
            continue

    # All retries exhausted — return a minimal-but-valid output
    return _degenerate_output(seed, config, discarded, last_err)


def _try_generate(seed: int, cfg: GeneratorConfig) -> GeneratorOutput:
    from .footprint import generate_footprint
    from .subdivision import subdivide
    from .walls import build_wall_graph
    from .openings import place_openings
    from .icons import place_icons
    from .annotations import place_dimensions
    from .themes import get_theme
    from .linework import get_linework
    from .rendering import (
        render_canvas, render_rooms, render_walls, render_openings,
        render_icons, render_annotations, render_era_overlays,
    )
    from .rendering.text import render_room_labels
    from .augmentation import apply_augmentation
    from .augmentation.spatial import apply_spatial, yolo_from_mask
    from .types import PIXEL_WINDOW, PIXEL_DOOR

    rng = bundle_from_seed(seed)
    era = cfg.resolved_era(rng["footprint"])
    spec = get_linework(era)
    preset_name = cfg.resolved_preset(era)
    aggression = cfg.resolved_aggression(rng["footprint"])

    # Chapter 2
    footprint = generate_footprint(cfg.image_size, aggression, rng["footprint"],
                                   margin=cfg.canvas_margin,
                                   forced_primitive=cfg.forced_primitive)
    # Chapter 3
    rooms = subdivide(footprint, cfg, rng["subdivision"])
    # Chapter 5 (plus Chapter 4 diagonals injected inside)
    walls = build_wall_graph(footprint, rooms, cfg, rng["walls"], rng["diagonal"], era=era)
    # Chapter 6
    openings = place_openings(footprint, rooms, walls, cfg, rng["openings"], era=era)
    # Chapter 7
    icons = place_icons(footprint, rooms, walls, cfg, rng["icons"], era=era)
    # Chapter 8
    annotations = place_dimensions(footprint, rooms, walls, cfg, rng["annotations"], era=era)

    # Chapter 11 — mask written once
    inputs = MaskInputs(
        footprint=footprint, rooms=rooms, walls=walls,
        openings=openings, icons=icons, annotations=annotations,
    )
    mask = write_mask(cfg.image_size, inputs)

    # Chapter 9/10 — image rendering
    theme = get_theme(era, rng["theme"], cfg.image_size)
    force_mono = bool(rng["theme"].random() < cfg.monochrome_prob)
    canvas = render_canvas(cfg.image_size, theme, force_mono)
    render_rooms(canvas, rooms, theme, rng["theme"])

    # Pastel-wall dice: ~5% of plans ship with coloured walls/openings.
    pastel_mode: str | None = None
    pastel_rgb: tuple[int, int, int] | None = None
    if not force_mono and rng["theme"].random() < cfg.pastel_wall_prob:
        from .rendering.pastel import pick_pastel
        pastel_mode, pastel_rgb = pick_pastel(rng["theme"])

    render_walls(canvas, walls, theme, rng["theme"], spec=spec,
                 openings=openings, pastel_mode=pastel_mode,
                 pastel_rgb=pastel_rgb)
    render_openings(canvas, openings, walls, theme, rng["theme"], spec=spec,
                    pastel_mode=pastel_mode, pastel_rgb=pastel_rgb)
    render_era_overlays(canvas, rooms, theme, spec)
    render_icons(canvas, icons, theme, rng["theme"])
    render_annotations(canvas, annotations, theme, rng["theme"])

    # Text labels (room names + area)
    if rng["theme"].random() < cfg.text_prob:
        render_room_labels(canvas, rooms, theme, rng["theme"],
                           fonts_dir=cfg.fonts_dir,
                           language=cfg.text_language)

    # Spatial augmentation — flip + rotation applied to BOTH image and mask
    canvas, mask, spatial_meta = apply_spatial(
        canvas, mask, rng["augment"],
        flip_prob=0.5, vflip_prob=0.3, rotate_max_deg=2.0,
    )

    # Degradation augmentation — image only
    canvas = apply_augmentation(canvas, preset_name, rng["augment"], force_mono=force_mono)

    # Freeze mask after spatial transforms
    mask.setflags(write=False)
    class_mask = encode_class_mask(mask)

    # YOLO labels recomputed from the (possibly flipped/rotated) mask
    yolo = yolo_from_mask(mask, {0: PIXEL_WINDOW, 1: PIXEL_DOOR}, cfg.image_size)

    meta: dict[str, Any] = {
        "seed": int(seed),
        "era": era,
        "aggression": float(aggression),
        "augmentation_preset": preset_name,
        "primitive_id": footprint.primitive_id,
        "primitive_fallback": footprint.primitive_fallback,
        "n_cuts_applied": footprint.n_cuts_applied,
        "n_extrusions_applied": footprint.n_extrusions_applied,
        "room_count": len(rooms),
        "wall_count": len(walls.segments),
        "window_count": sum(1 for o in openings if o.kind == "window"),
        "door_count": sum(1 for o in openings if o.kind == "door"),
        "icon_count": len(icons),
        "annotation_count": len(annotations),
        "monochrome": force_mono,
        "image_size": list(cfg.image_size),
        **{k: v for k, v in spatial_meta.items()},
    }

    return GeneratorOutput(
        image=canvas,
        mask=np.array(mask),  # writable copy for downstream users
        class_mask=class_mask,
        yolo_labels=yolo,
        metadata=meta,
    )


def _degenerate_output(seed: int, cfg: GeneratorConfig,
                       discarded: list[int], err: list[str]) -> GeneratorOutput:
    h, w = cfg.image_size
    from .types import PIXEL_BACKGROUND
    image = np.full((h, w), 255, dtype=np.uint8)
    mask = np.full((h, w), PIXEL_BACKGROUND, dtype=np.uint8)
    return GeneratorOutput(
        image=image,
        mask=mask,
        class_mask=encode_class_mask(mask),
        yolo_labels=[],
        metadata={"seed": int(seed), "degenerate": True,
                  "discarded_seeds": discarded, "last_error": err},
    )
