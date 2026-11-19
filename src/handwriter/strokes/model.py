"""Stroke-based glyph representation.

Coordinate system for glyphs ("x-height units"):
    * x grows to the right, y grows DOWN (image convention)
    * the baseline is y = 0, the x-height line is y = -1
    * ascenders reach about y = -1.6, descenders about y = +0.6
    * 1 unit = the x-height of the style, so all styles/sources share one scale
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Vertical band (in x-height units) where cursive joins may attach.
JOIN_BAND = (-1.3, 0.3)


@dataclass
class Stroke:
    points: np.ndarray  # (N, 2) float
    widths: np.ndarray  # (N,) relative pen width, 1.0 = nominal pen
    closed: bool = False

    def copy(self) -> Stroke:
        return Stroke(self.points.copy(), self.widths.copy(), self.closed)


@dataclass(frozen=True)
class Anchor:
    """A point on a glyph where a cursive connection can attach."""

    stroke: int
    index: int
    is_end: bool  # True when the point is a free stroke end (vs. a mid-stroke point)


@dataclass
class Glyph:
    char: str
    strokes: list[Stroke]
    advance: float
    entry: Anchor | None = None
    exit: Anchor | None = None
    source: str = ""
    meta: dict = field(default_factory=dict)

    def bbox(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) of stroke centre-lines."""
        if not self.strokes:
            return (0.0, 0.0, self.advance, 0.0)
        pts = np.concatenate([s.points for s in self.strokes])
        return (*pts.min(0), *pts.max(0))

    def copy(self) -> Glyph:
        return Glyph(
            self.char,
            [s.copy() for s in self.strokes],
            self.advance,
            self.entry,
            self.exit,
            self.source,
            dict(self.meta),
        )


def find_anchors(strokes: list[Stroke]) -> tuple[Anchor | None, Anchor | None]:
    """Pick the entry (left) and exit (right) attachment points for cursive joins.

    Preference: free stroke ends inside the join band on the correct side of the
    glyph; fallback: the left-most / right-most centre-line point inside the band.
    """
    if not strokes:
        return None, None
    lo, hi = JOIN_BAND
    allpts = np.concatenate([s.points for s in strokes])
    xmin, xmax = allpts[:, 0].min(), allpts[:, 0].max()
    width = max(xmax - xmin, 1e-6)

    ends: list[tuple[Anchor, np.ndarray]] = []
    mids: list[tuple[Anchor, np.ndarray]] = []
    for si, s in enumerate(strokes):
        n = len(s.points)
        if n == 0:
            continue
        # Skip tiny dots (i/j dots, periods): not join targets.
        span = np.ptp(s.points, axis=0).max() if n > 1 else 0.0
        if span < 0.15:
            continue
        if not s.closed:
            for idx in (0, n - 1):
                p = s.points[idx]
                if lo <= p[1] <= hi:
                    ends.append((Anchor(si, idx, True), p))
        for idx in range(0, n, max(1, n // 40)):
            p = s.points[idx]
            if lo <= p[1] <= hi:
                mids.append((Anchor(si, idx, False), p))

    def pick(side: str) -> Anchor | None:
        if side == "left":
            good = [(a, p) for a, p in ends if p[0] <= xmin + 0.4 * width]
            key = lambda ap: ap[1][0]  # noqa: E731
            pool = good or mids
            return min(pool, key=key)[0] if pool else None
        good = [(a, p) for a, p in ends if p[0] >= xmin + 0.6 * width]
        key = lambda ap: ap[1][0]  # noqa: E731
        pool = good or mids
        return max(pool, key=key)[0] if pool else None

    return pick("left"), pick("right")


def anchor_point_and_dir(
    strokes: list[Stroke], anchor: Anchor, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the anchor position and the travel direction of the pen there.

    kind == "exit": direction the pen is moving when it LEAVES the glyph.
    kind == "entry": direction the pen is moving when it ARRIVES at the glyph.
    Directions are biased to point rightwards, as handwriting flows left-to-right.
    """
    s = strokes[anchor.stroke]
    pts = s.points
    i = anchor.index
    p = pts[i]
    k = max(1, min(4, len(pts) - 1))
    if len(pts) < 2:
        d = np.array([1.0, 0.0])
    elif anchor.is_end:
        inner = pts[min(i + k, len(pts) - 1)] if i == 0 else pts[max(i - k, 0)]
        outward = p - inner
        d = outward if kind == "exit" else -outward
    else:
        a, b = pts[max(i - k, 0)], pts[min(i + k, len(pts) - 1)]
        d = b - a
        if d[0] < 0:
            d = -d
    n = np.linalg.norm(d)
    d = d / n if n > 1e-9 else np.array([1.0, 0.0])
    d = d + np.array([0.8, 0.0])  # bias rightwards to avoid loops in joins
    return p.copy(), d / np.linalg.norm(d)
