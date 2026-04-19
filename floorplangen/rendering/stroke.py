"""Variable-width hand-drawn stroke engine (DESIGN §9.5 soviet era).

Unlike cv2.line which renders a uniform-thickness anti-aliased stroke, hand
strokes must vary in thickness along their length, overshoot their endpoints,
and wobble slightly off the idealised centreline. Implementation:

    1. Resample the centreline at ~2px intervals.
    2. Add a low-frequency Perlin-like jitter to each sample (smoothed white
       noise); apply the same jitter to a parallel "thickness envelope" so
       the width oscillates smoothly along the stroke.
    3. Extend the endpoints by overshoot_px so the line "leaks past" the
       junction — a signature of hand drafting.
    4. Draw each segment between resampled points with its local thickness
       using cv2.line; cap with a small filled circle at every sample to
       keep the envelope continuous.

This engine is O(L) where L is the centreline length in pixels and is
intended for soviet-era wall/opening rendering.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def _smooth_noise(n: int, rng: np.random.Generator, scale: float = 1.0,
                  smoothing: int = 5) -> np.ndarray:
    """Return length-n array of smoothed noise in [-scale, +scale]."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    raw = rng.uniform(-1.0, 1.0, size=n + smoothing).astype(np.float32)
    k = np.ones(smoothing, dtype=np.float32) / smoothing
    smoothed = np.convolve(raw, k, mode="valid")
    if len(smoothed) < n:
        smoothed = np.pad(smoothed, (0, n - len(smoothed)), mode="edge")
    return smoothed[:n] * scale


def draw_hand_stroke(canvas: np.ndarray,
                     p0: tuple[float, float],
                     p1: tuple[float, float],
                     thickness: float,
                     color,
                     rng: np.random.Generator,
                     overshoot_px: float = 0.0,
                     sample_step: float = 2.0,
                     wobble_px: float = 0.8,
                     thickness_jitter: float = 0.35) -> None:
    """Draw a single hand-style segment from p0 to p1."""
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux

    # Extend endpoints by overshoot
    ox = ux * overshoot_px
    oy = uy * overshoot_px
    ax, ay = p0[0] - ox, p0[1] - oy
    bx, by = p1[0] + ox, p1[1] + oy
    extended_L = L + 2.0 * overshoot_px

    n_samples = max(3, int(round(extended_L / sample_step)))
    ts = np.linspace(0.0, 1.0, n_samples)
    cx = ax + ux * extended_L * ts
    cy = ay + uy * extended_L * ts

    wobble = _smooth_noise(n_samples, rng, scale=wobble_px, smoothing=5)
    cx = cx + nx * wobble
    cy = cy + ny * wobble

    thick_env = thickness * (1.0 + _smooth_noise(
        n_samples, rng, scale=thickness_jitter, smoothing=3))
    thick_env = np.clip(thick_env, 0.5, thickness * 1.8)

    for i in range(n_samples - 1):
        t = max(1, int(round((thick_env[i] + thick_env[i + 1]) / 2.0)))
        cv2.line(canvas,
                 (int(round(cx[i])), int(round(cy[i]))),
                 (int(round(cx[i + 1])), int(round(cy[i + 1]))),
                 color=color, thickness=t, lineType=cv2.LINE_AA)


def draw_hand_polyline(canvas: np.ndarray,
                       points: list[tuple[float, float]],
                       thickness: float,
                       color,
                       rng: np.random.Generator,
                       overshoot_px: float = 0.0,
                       **kwargs) -> None:
    """Draw a multi-segment polyline, overshooting only at the ends."""
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        over = overshoot_px if (i == 0 or i == len(points) - 2) else 0.0
        # Only apply overshoot to the true endpoints
        p0 = points[i]
        p1 = points[i + 1]
        over_start = overshoot_px if i == 0 else 0.0
        over_end = overshoot_px if i == len(points) - 2 else 0.0
        # Asymmetric overshoot: extend each endpoint individually
        _draw_hand_segment_asymmetric(canvas, p0, p1, thickness, color, rng,
                                      over_start, over_end, **kwargs)


def _draw_hand_segment_asymmetric(canvas, p0, p1, thickness, color, rng,
                                   over_start, over_end,
                                   sample_step: float = 2.0,
                                   wobble_px: float = 0.8,
                                   thickness_jitter: float = 0.35) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    L = math.hypot(dx, dy)
    if L < 1e-3:
        return
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    ax = p0[0] - ux * over_start
    ay = p0[1] - uy * over_start
    bx = p1[0] + ux * over_end
    by = p1[1] + uy * over_end
    ext_L = L + over_start + over_end
    n_samples = max(3, int(round(ext_L / sample_step)))
    ts = np.linspace(0.0, 1.0, n_samples)
    cx = ax + ux * ext_L * ts
    cy = ay + uy * ext_L * ts
    wobble = _smooth_noise(n_samples, rng, scale=wobble_px, smoothing=5)
    cx = cx + nx * wobble
    cy = cy + ny * wobble
    thick_env = thickness * (1.0 + _smooth_noise(
        n_samples, rng, scale=thickness_jitter, smoothing=3))
    thick_env = np.clip(thick_env, 0.5, thickness * 1.8)
    for i in range(n_samples - 1):
        t = max(1, int(round((thick_env[i] + thick_env[i + 1]) / 2.0)))
        cv2.line(canvas,
                 (int(round(cx[i])), int(round(cy[i]))),
                 (int(round(cx[i + 1])), int(round(cy[i + 1]))),
                 color=color, thickness=t, lineType=cv2.LINE_AA)
