"""Pen and ink rendering.

Strokes (pixel coordinates + per-point pen width in pixels) are drawn into a single
"ink coverage" canvas with anti-aliased variable-width segments and round joints. The
coverage is then given ink texture and multiplied onto the paper background, so ink
over printed lines looks the way real ink does.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from handwriter.strokes.skeleton import resample

_SHIFT = 4
_F = 1 << _SHIFT

INK_COLORS: dict[str, tuple[int, int, int]] = {
    "blue": (24, 46, 128),
    "dark-blue": (20, 30, 82),
    "black": (28, 28, 34),
    "gel-blue": (16, 52, 168),
    "red": (170, 28, 36),
    "green": (20, 105, 55),
    "pencil": (80, 80, 84),
}


@dataclass
class PxStroke:
    """A stroke ready to draw: pixel coordinates and per-point pen width in pixels."""

    points: np.ndarray  # (N, 2)
    widths: np.ndarray  # (N,)
    darkness: float = 1.0  # 0..1 multiplier on ink coverage


def draw_strokes(canvas: np.ndarray, strokes: Iterable[PxStroke], spacing: float = 1.6) -> None:
    """Draw strokes into a uint8 coverage canvas (255 = full ink), in place."""
    h, w = canvas.shape[:2]
    for s in strokes:
        if len(s.points) == 0:
            continue
        val = int(round(255 * float(np.clip(s.darkness, 0, 1))))
        pts, wid = s.points, s.widths
        if len(pts) > 1:
            pts, wid = resample(pts, wid, spacing)
        r = np.maximum(wid * 0.5, 0.35)
        # Cull strokes entirely off-canvas.
        if (pts[:, 0].max() < -10 or pts[:, 1].max() < -10 or pts[:, 0].min() > w + 10
                or pts[:, 1].min() > h + 10):
            continue
        if len(pts) == 1:
            c = tuple(int(v) for v in np.round(pts[0] * _F))
            cv2.circle(canvas, c, int(round(r[0] * _F)), val, -1, cv2.LINE_AA, _SHIFT)
            continue
        d = np.diff(pts, axis=0)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n < 1e-9] = 1e-9
        normal = np.column_stack([-d[:, 1], d[:, 0]]) / n
        p0, p1 = pts[:-1], pts[1:]
        r0, r1 = r[:-1, None], r[1:, None]
        off0, off1 = normal * r0, normal * r1
        quads = np.stack([p0 + off0, p1 + off1, p1 - off1, p0 - off0], 1)
        quads = np.round(quads * _F).astype(np.int32)
        centers = [tuple(int(v) for v in c) for c in np.round(pts * _F)]
        radii = [int(v) for v in np.round(r * _F)]
        aa = cv2.LINE_AA
        cv2.circle(canvas, centers[0], radii[0], val, -1, aa, _SHIFT)
        for i in range(len(quads)):
            cv2.fillConvexPoly(canvas, quads[i], val, aa, _SHIFT)
            cv2.circle(canvas, centers[i + 1], radii[i + 1], val, -1, aa, _SHIFT)


def ink_texture(coverage: np.ndarray, rng: np.random.Generator, grain: float = 0.12,
                blotch: float = 0.10) -> np.ndarray:
    """Convert a coverage canvas into ink alpha (0..1) with subtle grain and density variation."""
    a = coverage.astype(np.float32) / 255.0
    if grain <= 0 and blotch <= 0:
        return a
    h, w = a.shape
    # Fine grain (paper fibres / ballpoint skipping), only visible on ink.
    fine = rng.normal(0, 1, (h // 2 + 1, w // 2 + 1)).astype(np.float32)
    fine = cv2.resize(cv2.GaussianBlur(fine, (0, 0), 0.8), (w, h), interpolation=cv2.INTER_LINEAR)
    # Large-scale density variation (ink flow).
    coarse = rng.normal(0, 1, (h // 64 + 2, w // 64 + 2)).astype(np.float32)
    coarse = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    density = 1.0 - blotch * (0.5 + 0.5 * np.tanh(coarse)) - grain * np.clip(fine, 0, None) * 0.5
    return np.clip(a * density, 0, 1) ** 0.9


def composite(background: np.ndarray, alpha: np.ndarray,
              ink_rgb: tuple[int, int, int]) -> np.ndarray:
    """Multiply-blend ink onto an RGB uint8 background."""
    bg = background.astype(np.float32)
    ink = np.asarray(ink_rgb, dtype=np.float32) / 255.0
    a = alpha[..., None]
    out = bg * (1.0 - a + a * ink)
    return np.clip(out, 0, 255).astype(np.uint8)
