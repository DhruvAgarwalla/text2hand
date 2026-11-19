"""Find text lines and words on an ink mask, and measure baseline, x-height and slant."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class Comp:
    label: int
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    area: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass
class TextLine:
    comps: list[int] = field(default_factory=list)
    baseline: tuple[float, float] = (0.0, 0.0)  # y = a * x + b
    xheight: float = 0.0
    center: float = 0.0

    def base_y(self, x: np.ndarray | float) -> np.ndarray | float:
        a, b = self.baseline
        return a * np.asarray(x) + b


@dataclass
class WordBox:
    line: int
    comps: list[int]
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0


@dataclass
class Segmentation:
    labels: np.ndarray
    comps: dict[int, Comp]
    lines: list[TextLine]
    words: list[WordBox]  # reading order
    slant_deg: float
    word_gap_threshold: float  # in x-height units


def components(ink: np.ndarray) -> tuple[np.ndarray, dict[int, Comp]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    comps = {}
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        comps[i] = Comp(i, int(x), int(y), int(x + w), int(y + h), int(a))
    return labels, comps


def segment_page(ink: np.ndarray, rules_y: list[float] | None = None) -> Segmentation:
    labels, comps = components(ink)
    if not comps:
        return Segmentation(labels, comps, [], [], 0.0, 0.6)
    heights = np.array([c.h for c in comps.values() if c.area > 30])
    unit = float(np.median(heights)) if len(heights) else 20.0
    # Minimum distance between text lines. Printed rules give it directly; otherwise
    # use the typical ink-piece height (whole words in cursive, so be conservative).
    min_dist = 0.6 * unit
    if rules_y is not None and len(rules_y) >= 4:
        spacing = float(np.median(np.diff(sorted(rules_y))))
        if spacing > 0.5 * unit:
            min_dist = 0.75 * spacing
    lines = _find_lines(ink, comps, unit, min_dist)
    for ln in lines:
        _line_metrics(ln, labels, comps)
    lines = [ln for ln in lines if ln.xheight > 0]
    if lines:
        # ALL-CAPS or digit-only lines have no x-height letters: use the page's typical one.
        med = float(np.median([ln.xheight for ln in lines]))
        for ln in lines:
            if not 0.75 * med <= ln.xheight <= 1.3 * med:
                ln.xheight = med
    lines.sort(key=lambda ln: ln.center)
    slant = estimate_slant(labels, comps, lines)
    words, thr = _find_words(lines, labels, comps, slant)
    return Segmentation(labels, comps, lines, words, slant, thr)


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


def _find_lines(ink: np.ndarray, comps: dict[int, Comp], unit: float,
                min_dist: float) -> list[TextLine]:
    sigma = max(2.0, min(unit, min_dist) * 0.3)
    prof = gaussian_filter1d(ink.sum(1).astype(np.float64), sigma=sigma)
    if prof.max() <= 0:
        return []
    peaks, _ = find_peaks(prof, distance=max(4, int(min_dist)), prominence=prof.max() * 0.03,
                          height=prof.max() * 0.04)
    if len(peaks) == 0:
        return []
    lines = [TextLine(center=float(p)) for p in peaks]
    centers = np.array([ln.center for ln in lines])
    big_area = np.median([c.area for c in comps.values()])
    for c in comps.values():
        if c.area < 0.25 * big_area and c.h < 0.6 * unit:
            # Dots/accents/small marks belong to the line just below them, if close.
            below = centers[centers >= c.cy - 0.2 * unit]
            if len(below) and below.min() - c.cy < 1.4 * unit:
                idx = int(np.argmin(np.abs(centers - below.min())))
            else:
                idx = int(np.argmin(np.abs(centers - c.cy)))
        else:
            idx = int(np.argmin(np.abs(centers - c.cy)))
        lines[idx].comps.append(c.label)
    return [ln for ln in lines if ln.comps]


def _line_metrics(ln: TextLine, labels: np.ndarray, comps: dict[int, Comp]) -> None:
    cs = [comps[i] for i in ln.comps]
    x0, x1 = min(c.x0 for c in cs), max(c.x1 for c in cs)
    y0, y1 = min(c.y0 for c in cs), max(c.y1 for c in cs)
    sub = np.isin(labels[y0:y1, x0:x1], ln.comps)
    width = x1 - x0
    chunks = max(1, min(6, width // 300))
    xs, bases, tops = [], [], []
    for k in range(chunks):
        a, b = k * width // chunks, (k + 1) * width // chunks
        part = sub[:, a:b]
        rows = part.sum(1).astype(np.float64)
        if rows.max() <= 0:
            continue
        rows = gaussian_filter1d(rows, 1.0)
        core = np.nonzero(rows > 0.4 * rows.max())[0]
        if len(core) == 0:
            continue
        xs.append(x0 + (a + b) / 2)
        tops.append(y0 + core[0])
        bases.append(y0 + core[-1] + 1)
    if not xs:
        return
    xs, bases, tops = map(np.asarray, (xs, bases, tops))
    if len(xs) >= 3:
        a, b = np.polyfit(xs, bases, 1)
        if abs(a) > 0.05:  # implausible tilt -> flat
            a, b = 0.0, float(np.median(bases))
    else:
        a, b = 0.0, float(np.median(bases))
    ln.baseline = (float(a), float(b))
    ln.xheight = float(np.median(bases - tops))
    ln.center = float(np.median(bases) - ln.xheight / 2)


# ---------------------------------------------------------------------------
# Slant
# ---------------------------------------------------------------------------


def estimate_slant(labels: np.ndarray, comps: dict[int, Comp], lines: list[TextLine]) -> float:
    """Dominant writing slant in degrees (positive = leaning right).

    Measured from the edges of near-vertical strokes only (down-strokes define slant;
    cursive connecting strokes and horizontal bars are ignored).
    """
    ink = (labels > 0).astype(np.float32)
    ink = cv2.GaussianBlur(ink, (0, 0), 1.5)
    gx = cv2.Sobel(ink, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ink, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    sel = mag > 0.25 * mag.max()
    gx, gy, mag = gx[sel], gy[sel], mag[sel]
    flip = gx < 0
    gx[flip], gy[flip] = -gx[flip], -gy[flip]
    theta = np.degrees(np.arctan2(gy, gx))  # 0 = vertical stroke, + = leaning right
    near_vertical = np.abs(theta) < 40
    if near_vertical.sum() < 50:
        return 0.0
    th, w = theta[near_vertical], mag[near_vertical]
    hist, edges = np.histogram(th, bins=80, range=(-40, 40), weights=w)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    peak = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    close = np.abs(th - peak) < 10
    return float(np.round(np.average(th[close], weights=w[close]), 1))


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------


def _find_words(lines: list[TextLine], labels: np.ndarray, comps: dict[int, Comp],
                slant: float) -> tuple[list[WordBox], float]:
    t = np.tan(np.deg2rad(slant))
    per_line_blobs = []
    gaps_all = []
    for li, ln in enumerate(lines):
        items = []
        for i in ln.comps:
            c = comps[i]
            # deslanted horizontal extent of the actual ink pixels
            yy, xx = np.nonzero(labels[c.y0:c.y1, c.x0:c.x1] == i)
            xs, ys = xx + c.x0, yy + c.y0
            ds = xs + t * (ys - ln.base_y(xs))
            items.append((float(ds.min()), float(ds.max()) + 1, i))
        items.sort()
        blobs = []  # merged overlapping components: [d0, d1, [labels]]
        for d0, d1, i in items:
            if blobs and d0 <= blobs[-1][1] + 0.05 * ln.xheight:
                blobs[-1][1] = max(blobs[-1][1], d1)
                blobs[-1][2].append(i)
            else:
                blobs.append([d0, d1, [i]])
        per_line_blobs.append((li, blobs))
        for a, b in zip(blobs, blobs[1:], strict=False):
            gaps_all.append((b[0] - a[1]) / max(ln.xheight, 1))
    thr = _gap_threshold(np.array(gaps_all))
    words: list[WordBox] = []
    for li, blobs in per_line_blobs:
        xh = max(lines[li].xheight, 1)
        group = [blobs[0]] if blobs else []
        for prev, cur in zip(blobs, blobs[1:], strict=False):
            if (cur[0] - prev[1]) / xh > thr:
                words.append(_word_from(li, group, comps))
                group = []
            group.append(cur)
        if group:
            words.append(_word_from(li, group, comps))
    return words, thr


def _gap_threshold(gaps: np.ndarray) -> float:
    gaps = gaps[gaps > 0]
    if len(gaps) < 4:
        return 0.6
    lg = np.log(gaps + 0.02)
    lo, hi = np.percentile(lg, 25), np.percentile(lg, 90)
    for _ in range(20):  # 2-means in log space
        mid = (lo + hi) / 2
        a, b = lg[lg <= mid], lg[lg > mid]
        if len(a) == 0 or len(b) == 0:
            break
        lo, hi = a.mean(), b.mean()
    thr = float(np.exp((lo + hi) / 2) - 0.02)
    return float(np.clip(thr, 0.3, 1.3))


def _word_from(li: int, blobs: list, comps: dict[int, Comp]) -> WordBox:
    labels = [i for b in blobs for i in b[2]]
    cs = [comps[i] for i in labels]
    return WordBox(li, labels, min(c.x0 for c in cs), max(c.x1 for c in cs),
                   min(c.y0 for c in cs), max(c.y1 for c in cs))
