"""Procedurally synthesise a minimal RGBA icon pack (DESIGN §7.11 bootstrap).

Each icon is a simple vector-style black glyph on a transparent background,
paired with a JSON sidecar declaring category, era compatibility, and
allowed rotations. Not production-grade — intended to let Chapter 7 run
without external assets.

Usage:
    python -m scripts.gen_icon_pack --out D:/FloorPLanGen/icons
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CANVAS = 96
INK = (40, 40, 40, 255)
TRANSPARENT = (0, 0, 0, 0)


def _new() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGBA", (CANVAS, CANVAS), TRANSPARENT)
    return im, ImageDraw.Draw(im)


def _save(im: Image.Image, path: Path, sidecar: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PNG")
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2),
                                         encoding="utf-8")


# ── Bedroom ────────────────────────────────────────────────────────────────
def bed_single() -> Image.Image:
    im, d = _new()
    d.rounded_rectangle((8, 12, 88, 84), outline=INK, width=3, radius=6)
    d.rectangle((12, 16, 84, 36), outline=INK, width=2)  # pillow area
    d.line((12, 38, 84, 38), fill=INK, width=2)
    return im


def bed_double() -> Image.Image:
    im, d = _new()
    d.rounded_rectangle((4, 12, 92, 84), outline=INK, width=3, radius=6)
    d.rectangle((8, 16, 46, 36), outline=INK, width=2)
    d.rectangle((50, 16, 88, 36), outline=INK, width=2)
    d.line((8, 38, 88, 38), fill=INK, width=2)
    return im


def nightstand() -> Image.Image:
    im, d = _new()
    d.rectangle((24, 24, 72, 72), outline=INK, width=3)
    d.line((24, 48, 72, 48), fill=INK, width=2)
    return im


def wardrobe() -> Image.Image:
    im, d = _new()
    d.rectangle((12, 8, 84, 88), outline=INK, width=3)
    d.line((48, 8, 48, 88), fill=INK, width=2)
    d.ellipse((42, 46, 46, 50), fill=INK)
    d.ellipse((50, 46, 54, 50), fill=INK)
    return im


# ── Kitchen ────────────────────────────────────────────────────────────────
def stove() -> Image.Image:
    im, d = _new()
    d.rectangle((8, 8, 88, 88), outline=INK, width=3)
    for cx, cy in [(30, 30), (66, 30), (30, 66), (66, 66)]:
        d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline=INK, width=2)
    return im


def sink() -> Image.Image:
    im, d = _new()
    d.rectangle((6, 14, 90, 82), outline=INK, width=3)
    d.rounded_rectangle((16, 24, 80, 72), outline=INK, width=2, radius=6)
    d.ellipse((46, 8, 50, 12), fill=INK)
    return im


def fridge() -> Image.Image:
    im, d = _new()
    d.rectangle((16, 6, 80, 90), outline=INK, width=3)
    d.line((16, 34, 80, 34), fill=INK, width=2)
    d.rectangle((70, 14, 74, 24), fill=INK)
    d.rectangle((70, 54, 74, 70), fill=INK)
    return im


def counter() -> Image.Image:
    im, d = _new()
    d.rectangle((4, 28, 92, 68), outline=INK, width=3)
    return im


# ── Living ─────────────────────────────────────────────────────────────────
def sofa() -> Image.Image:
    im, d = _new()
    d.rounded_rectangle((6, 28, 90, 80), outline=INK, width=3, radius=8)
    d.rounded_rectangle((6, 16, 90, 44), outline=INK, width=2, radius=6)
    d.line((30, 28, 30, 80), fill=INK, width=2)
    d.line((66, 28, 66, 80), fill=INK, width=2)
    return im


def armchair() -> Image.Image:
    im, d = _new()
    d.rounded_rectangle((16, 28, 80, 84), outline=INK, width=3, radius=6)
    d.rounded_rectangle((16, 18, 80, 40), outline=INK, width=2, radius=6)
    return im


def tv() -> Image.Image:
    im, d = _new()
    d.rectangle((8, 30, 88, 72), outline=INK, width=3)
    d.line((30, 72, 66, 72), fill=INK, width=2)
    d.line((48, 72, 48, 82), fill=INK, width=2)
    d.line((30, 82, 66, 82), fill=INK, width=2)
    return im


def table() -> Image.Image:
    im, d = _new()
    d.rounded_rectangle((10, 22, 86, 74), outline=INK, width=3, radius=10)
    return im


# ── Sanitary ───────────────────────────────────────────────────────────────
def toilet() -> Image.Image:
    im, d = _new()
    d.rounded_rectangle((28, 6, 68, 26), outline=INK, width=3, radius=4)
    d.ellipse((20, 28, 76, 84), outline=INK, width=3)
    return im


def bathtub() -> Image.Image:
    im, d = _new()
    d.rounded_rectangle((6, 18, 90, 78), outline=INK, width=3, radius=12)
    d.rounded_rectangle((14, 26, 82, 70), outline=INK, width=2, radius=8)
    d.ellipse((76, 40, 82, 46), fill=INK)
    return im


def shower() -> Image.Image:
    im, d = _new()
    d.rectangle((8, 8, 88, 88), outline=INK, width=3)
    d.ellipse((44, 44, 52, 52), fill=INK)
    for dx, dy in [(-14, -14), (14, -14), (-14, 14), (14, 14)]:
        d.line((48, 48, 48 + dx, 48 + dy), fill=INK, width=1)
    return im


def basin() -> Image.Image:
    im, d = _new()
    d.rounded_rectangle((10, 24, 86, 74), outline=INK, width=3, radius=8)
    d.ellipse((44, 44, 52, 52), fill=INK)
    d.line((48, 24, 48, 12), fill=INK, width=2)
    return im


# ── Stair ──────────────────────────────────────────────────────────────────
def stair_flight() -> Image.Image:
    im, d = _new()
    for i in range(8):
        y = 8 + i * 10
        d.line((8, y, 88, y), fill=INK, width=2)
    d.line((8, 8, 8, 88), fill=INK, width=3)
    d.line((88, 8, 88, 88), fill=INK, width=3)
    return im


# ── Radiator ──────────────────────────────────────────────────────────────
def radiator() -> Image.Image:
    im, d = _new()
    d.rectangle((8, 32, 88, 64), outline=INK, width=3)
    for x in range(14, 88, 8):
        d.line((x, 32, x, 64), fill=INK, width=2)
    return im


# ── Misc ──────────────────────────────────────────────────────────────────
def plant() -> Image.Image:
    im, d = _new()
    d.polygon([(38, 78), (58, 78), (54, 88), (42, 88)], outline=INK, width=3)
    for cx, cy in [(48, 36), (34, 50), (62, 50), (40, 64), (56, 64)]:
        d.ellipse((cx - 10, cy - 10, cx + 10, cy + 10), outline=INK, width=2)
    return im


def dot() -> Image.Image:
    im, d = _new()
    d.ellipse((36, 36, 60, 60), outline=INK, width=3)
    return im


# ── Catalogue ─────────────────────────────────────────────────────────────
CATALOGUE: dict[str, list[tuple[str, callable, dict]]] = {
    "bedroom": [
        ("bed_single",  bed_single,  {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.04}),
        ("bed_double",  bed_double,  {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.06}),
        ("nightstand",  nightstand,  {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.01}),
        ("wardrobe",    wardrobe,    {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.03}),
    ],
    "kitchen": [
        ("stove",   stove,   {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.02}),
        ("sink",    sink,    {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.015}),
        ("fridge",  fridge,  {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.025}),
        ("counter", counter, {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.02}),
    ],
    "living": [
        ("sofa",     sofa,     {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.05}),
        ("armchair", armchair, {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.02}),
        ("tv",       tv,       {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.02}),
        ("table",    table,    {"allow_rotation": [0],                "min_room_area_frac": 0.03}),
    ],
    "sanitary": [
        ("toilet",   toilet,   {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.015}),
        ("bathtub",  bathtub,  {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.03}),
        ("shower",   shower,   {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.02}),
        ("basin",    basin,    {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.015}),
    ],
    "stair": [
        ("stair_flight", stair_flight, {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.04}),
    ],
    "radiator": [
        ("radiator", radiator, {"allow_rotation": [0, 90, 180, 270], "min_room_area_frac": 0.005,
                                "wall_snap": "exterior"}),
    ],
    "misc": [
        ("plant", plant, {"allow_rotation": [0], "min_room_area_frac": 0.01}),
        ("dot",   dot,   {"allow_rotation": [0], "min_room_area_frac": 0.005}),
    ],
}


def build(out: Path) -> None:
    for category, items in CATALOGUE.items():
        for stem, fn, sidecar_extra in items:
            im = fn()
            sidecar = {
                "category": category,
                "anchor": [0.5, 0.5],
                "era_compatible": ["scan", "digital", "soviet"],
            }
            sidecar.update(sidecar_extra)
            _save(im, out / category / f"{stem}.png", sidecar)
    print(f"Built {sum(len(v) for v in CATALOGUE.values())} icons across "
          f"{len(CATALOGUE)} categories at {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent / "icons")
    args = p.parse_args(argv)
    build(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
