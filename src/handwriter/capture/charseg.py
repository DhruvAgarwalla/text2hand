"""Split a handwritten word image into its letters, given the word's text.

Work in *deslanted* x coordinates (x shifted by slant x height above the baseline) so a
slanted letter occupies a vertical band. Find cut positions with dynamic programming:
cuts prefer columns with little or no ink, and each letter's width should be close to its
expected width (from a reference handwriting font, scaled to this word). Then:

* print style: whole connected pieces of ink are assigned to the letter whose band
  contains their centre (so letters are never cut), unless a piece clearly spans several
  letters (touching letters), in which case it is cut at the band edges;
* cursive: ink is cut at the band edges; the connecting strokes become the letters'
  entry/exit stubs, and joins are redrawn smoothly when rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from handwriter.capture.segment import Segmentation, TextLine, WordBox


@dataclass
class CharCut:
    char: str
    mask: np.ndarray  # bool, crop of the page
    x0: int  # crop origin on the page
    y0: int
    origin_x: float  # page x of the glyph origin (left edge of band at the baseline)
    advance_px: float
    cost: float


def segment_word(seg: Segmentation, box: WordBox, text: str, cursive: bool,
                 char_width: Callable[[str], float]) -> list[CharCut]:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return []
    line: TextLine = seg.lines[box.line]
    t = np.tan(np.deg2rad(seg.slant_deg))
    labels = seg.labels[box.y0:box.y1, box.x0:box.x1]
    mask = np.isin(labels, box.comps)
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return []
    px, py = xx + box.x0, yy + box.y0
    base = line.base_y(px)
    ds = px + t * (py - base)  # deslanted x per ink pixel
    d_min = float(ds.min())
    width = int(np.ceil(ds.max() - d_min)) + 1
    col = (ds - d_min).astype(np.int64)
    prof = np.bincount(col, minlength=width).astype(np.float64)

    expected = np.array([max(char_width(c), 0.15) for c in chars])
    expected = expected / expected.sum() * width
    if cursive:
        cut_cost = _cursive_cut_cost(col, py - base, width, line.xheight)
    else:
        cut_cost = _print_cut_cost(prof)
    bounds, cost = _dp_cuts(cut_cost, expected)

    # Assign ink to letters.
    seg_of_pixel = np.clip(np.searchsorted(bounds, ds - d_min, side="right") - 1, 0,
                           len(chars) - 1)
    if not cursive:
        seg_of_pixel = _assign_components(labels[yy, xx], ds - d_min, bounds, seg_of_pixel)

    cuts = []
    for k, ch in enumerate(chars):
        sel = seg_of_pixel == k
        if not sel.any():
            continue
        cx, cy = xx[sel], yy[sel]
        x0, x1, y0, y1 = cx.min(), cx.max() + 1, cy.min(), cy.max() + 1
        m = np.zeros((y1 - y0, x1 - x0), bool)
        m[cy - y0, cx - x0] = True
        cuts.append(CharCut(ch, m, int(box.x0 + x0), int(box.y0 + y0),
                            origin_x=d_min + bounds[k], advance_px=bounds[k + 1] - bounds[k],
                            cost=cost / len(chars)))
    return cuts


def _print_cut_cost(prof: np.ndarray) -> np.ndarray:
    """Print: cut through as little ink as possible, ideally through blank gaps."""
    pmax = max(prof.max(), 1.0)
    smooth = np.convolve(prof, np.ones(3) / 3, mode="same")
    return 1.5 * smooth / pmax - 0.8 * (smooth < 0.5)


def _cursive_cut_cost(col: np.ndarray, rel_y: np.ndarray, width: int,
                      xheight: float) -> np.ndarray:
    """Cursive: cut where a single connecting stroke passes, low in the x-height zone.

    rel_y: ink y relative to the baseline (negative = above).
    """
    xh = max(xheight, 1.0)
    # Rasterise the deslanted word: rows = height above baseline, columns = deslanted x.
    row = np.clip(np.round(-rel_y).astype(np.int64), -int(1.5 * xh), int(3 * xh)) + int(1.5 * xh)
    img = np.zeros((int(4.5 * xh) + 2, width), np.uint8)
    img[row, col] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 2), np.uint8))
    runs = (np.diff(img, axis=0, prepend=0) == 1).sum(0)  # strokes crossing each column
    ink = img.sum(0).astype(np.float64)
    row_idx = np.arange(img.shape[0])[:, None]
    mean_row = (img * row_idx).sum(0) / np.maximum(ink, 1)
    heights = np.where(ink > 0, mean_row, 0) - int(1.5 * xh)  # mean height above baseline
    cost = 0.9 * np.maximum(runs - 1, 0) + 0.4 * ink / max(ink.max(), 1)
    single_low = (runs == 1) & (heights < 0.65 * xh)
    cost = cost - 0.5 * single_low - 0.9 * (ink == 0)
    return np.convolve(cost, np.ones(3) / 3, mode="same")


def _dp_cuts(cut_cost: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, float]:
    """Return K+1 boundaries (0 = first, W = last) minimising cut cost + width deviation."""
    w = len(cut_cost)
    k_total = len(expected)
    if k_total == 1:
        return np.array([0.0, float(w)]), 0.0
    inf = 1e18
    dp = np.full((k_total + 1, w + 1), inf)
    back = np.zeros((k_total + 1, w + 1), np.int64)
    dp[0, 0] = 0.0
    for k in range(1, k_total + 1):
        e = expected[k - 1]
        lo, hi = max(1, int(0.3 * e)), max(2, int(2.6 * e) + 1)
        end_cost = np.append(cut_cost, 0.0)  # cutting at the very end is free
        if k < k_total:
            end_cost = end_cost.copy()
            end_cost[w] = inf  # interior cuts must be inside the word
        for d in range(lo, min(hi, w) + 1):
            prev = dp[k - 1, : w + 1 - d]
            cand = prev + ((d - e) / e) ** 2 + end_cost[d:]
            better = cand < dp[k, d:]
            dp[k, d:][better] = cand[better]
            back[k, d:][better] = d
    if dp[k_total, w] >= inf:  # fallback: proportional cuts
        return np.concatenate([[0.0], np.cumsum(expected)]), 5.0 * k_total
    bounds = [w]
    pos = w
    for k in range(k_total, 0, -1):
        pos -= back[k, pos]
        bounds.append(pos)
    return np.array(bounds[::-1], dtype=np.float64), float(dp[k_total, w])


def _assign_components(comp_ids: np.ndarray, ds: np.ndarray, bounds: np.ndarray,
                       seg_of_pixel: np.ndarray) -> np.ndarray:
    """Print style: move whole components into a single letter where sensible."""
    out = seg_of_pixel.copy()
    for cid in np.unique(comp_ids):
        sel = comp_ids == cid
        d = ds[sel]
        lo, hi = d.min(), d.max()
        segs, counts = np.unique(seg_of_pixel[sel], return_counts=True)
        if len(segs) == 1:
            continue
        main = segs[np.argmax(counts)]
        band = bounds[main + 1] - bounds[main]
        # A piece mostly inside one band, not much wider than it: keep it whole.
        if counts.max() / counts.sum() > 0.6 and (hi - lo) < 1.4 * band:
            out[sel] = main
    return out
