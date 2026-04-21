"""Opening symbol rendering.

Window line count (2 / 3 / 4 parallel frame lines) and door leaf angle
(30° drafting-open or 90° fully-closed) come from the per-opening metadata
populated during placement. Line weights scale with the LineworkSpec.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..linework import LineworkSpec, px_per_mm
from ..themes.base import EraTheme
from ..types import Opening, WallGraph


def _ink_color(theme: EraTheme, channels: int):
    if channels == 1:
        v = int(round(0.299 * theme.ink_rgb[0]
                      + 0.587 * theme.ink_rgb[1]
                      + 0.114 * theme.ink_rgb[2]))
        return int(v)
    return (int(theme.ink_rgb[2]), int(theme.ink_rgb[1]), int(theme.ink_rgb[0]))


def _paper_color(theme: EraTheme, channels: int):
    if channels == 1:
        v = int(round(0.299 * theme.paper_rgb[0]
                      + 0.587 * theme.paper_rgb[1]
                      + 0.114 * theme.paper_rgb[2]))
        return int(v)
    return (int(theme.paper_rgb[2]), int(theme.paper_rgb[1]), int(theme.paper_rgb[0]))


def _draw_window_lines(canvas: np.ndarray, rect: np.ndarray, n_lines: int,
                       color, thickness: int) -> None:
    """Draw `n_lines` evenly-spaced parallel lines along the window long axis."""
    xs = rect[:, 0]
    ys = rect[:, 1]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    horizontal = (x1 - x0) >= (y1 - y0)
    if n_lines < 1:
        return
    if horizontal:
        for k in range(n_lines):
            t = (k + 1) / (n_lines + 1)
            y = int(round(y0 + t * (y1 - y0)))
            cv2.line(canvas, (x0, y), (x1, y),
                     color=color, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        for k in range(n_lines):
            t = (k + 1) / (n_lines + 1)
            x = int(round(x0 + t * (x1 - x0)))
            cv2.line(canvas, (x, y0), (x, y1),
                     color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_door_leaf(canvas: np.ndarray, op: Opening, walls: WallGraph,
                    color, thickness: int) -> None:
    """Draw the door leaf line at its era-sampled angle.

    The hinge sits on the wall centreline at one end of the opening; the leaf
    line rotates away from the wall by `leaf_angle_deg` (30° = partly open
    drafting convention, 90° = fully closed against the wall).
    """
    if op.wall_index < 0 or op.wall_index >= len(walls.segments):
        return
    seg = walls.segments[op.wall_index]
    coords = list(seg.centreline.coords)
    (x0, y0), (x1, y1) = coords[0], coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    # Infer the along-wall extent of the opening by projecting its polygon.
    op_coords = np.asarray(list(op.polygon.exterior.coords), dtype=float)
    ts = (op_coords[:, 0] - x0) * ux + (op_coords[:, 1] - y0) * uy
    t_lo = float(ts.min())
    t_hi = float(ts.max())
    leaf_w = max(1.0, t_hi - t_lo)

    # Pick the arc's side: use the swing arc's centroid relative to the wall.
    side = 1.0
    if op.swing_arc is not None and not op.swing_arc.is_empty:
        cx = op.swing_arc.centroid.x
        cy = op.swing_arc.centroid.y
        if (cx - x0) * nx + (cy - y0) * ny < 0:
            side = -1.0
    nxs, nys = nx * side, ny * side

    hinge = (x0 + ux * t_lo, y0 + uy * t_lo)
    theta = math.radians(op.leaf_angle_deg)
    tip = (
        hinge[0] + (ux * math.cos(theta) + nxs * math.sin(theta)) * leaf_w,
        hinge[1] + (uy * math.cos(theta) + nys * math.sin(theta)) * leaf_w,
    )
    cv2.line(canvas,
             (int(round(hinge[0])), int(round(hinge[1]))),
             (int(round(tip[0])), int(round(tip[1]))),
             color=color, thickness=max(thickness, 1), lineType=cv2.LINE_AA)


def _draw_door_arc(canvas: np.ndarray, op: Opening, color, thickness: int) -> None:
    if op.swing_arc is None or op.swing_arc.is_empty:
        return
    try:
        arc_coords = np.array(list(op.swing_arc.exterior.coords), dtype=np.int32)
        cv2.polylines(canvas, [arc_coords], isClosed=False,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)
    except Exception:
        pass


def render_openings(canvas: np.ndarray, openings: list[Opening],
                    walls: WallGraph, theme: EraTheme, rng,
                    spec: LineworkSpec) -> None:
    channels = 1 if canvas.ndim == 2 else canvas.shape[2]
    ink = _ink_color(theme, channels)
    paper = _paper_color(theme, channels)
    h, w = canvas.shape[:2]
    ppm = px_per_mm((h, w))
    contour_thk = max(1, int(round(spec.wall.outline_px * ppm)))
    arc_thk = max(1, int(round(spec.door.arc_weight_px * ppm)))
    frame_thk = max(1, int(round(spec.window.frame_weight_px * ppm)))

    for op in openings:
        coords = np.array(list(op.polygon.exterior.coords), dtype=np.int32)
        cv2.fillPoly(canvas, [coords], color=paper)
        cv2.polylines(canvas, [coords], isClosed=True,
                      color=ink, thickness=contour_thk, lineType=cv2.LINE_AA)

        if op.kind == "window":
            _draw_window_lines(canvas, coords, op.window_line_count,
                               color=ink, thickness=frame_thk)
        elif op.kind == "door":
            _draw_door_arc(canvas, op, color=ink, thickness=arc_thk)
            _draw_door_leaf(canvas, op, walls, color=ink, thickness=arc_thk)
