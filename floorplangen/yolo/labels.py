"""YOLO label emission — v6.0-compatible pass-through (DESIGN §1.3.2, §1.6)."""
from __future__ import annotations

from ..types import Opening

# Class indices: 0=window, 1=door — compatible with v6.0 detection head
YOLO_CLASS_MAP: dict[str, int] = {"window": 0, "door": 1}


def emit_yolo_labels(image_size: tuple[int, int],
                     openings: list[Opening]) -> list[str]:
    h, w = image_size
    fw = float(w)
    fh = float(h)
    lines: list[str] = []
    for op in openings:
        if op.kind not in YOLO_CLASS_MAP:
            continue
        minx, miny, maxx, maxy = op.class_polygon.bounds
        cx = (minx + maxx) * 0.5 / fw
        cy = (miny + maxy) * 0.5 / fh
        bw = (maxx - minx) / fw
        bh = (maxy - miny) / fh
        if bw <= 0 or bh <= 0:
            continue
        cls = YOLO_CLASS_MAP[op.kind]
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines
