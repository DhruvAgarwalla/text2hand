"""Synthetic training data for the letter recogniser.

Letters are drawn from the bundled handwriting fonts through the same imperfection engine
the app uses for writing (random warp, rotation, slant, scale, pressure), rendered as ink
at a random scan resolution and pen width, then normalised exactly like real cut-outs.
Extra realism for segmentation: neighbouring-letter fragments at the edges, connecting
stubs (cursive), and a JUNK class made of half letters / two half letters / connectors.
"""

from __future__ import annotations

import numpy as np

from handwriter.imperfections.params import HandParams
from handwriter.imperfections.word import glyph_instance
from handwriter.ml.features import CHARSET, CLASS_OF, INPUT, JUNK, raster
from handwriter.render.pen import PxStroke, draw_strokes
from handwriter.strokes.glyphset import font_provider
from handwriter.strokes.model import Stroke

TRAIN_FONTS = ["kalam", "architects-daughter", "homemade-apple", "cedarville-cursive"]


def _ink_pixels(strokes: list[Stroke], xh_px: float, pen_px: float, rng):
    pts = np.concatenate([s.points for s in strokes]) if strokes else np.zeros((1, 2))
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    pad = 4 + pen_px
    w = int((x1 - x0) * xh_px + 2 * pad) + 1
    h = int((y1 - y0) * xh_px + 2 * pad) + 1
    canvas = np.zeros((max(h, 1), max(w, 1)), np.uint8)
    ox, oy = pad - x0 * xh_px, pad - y0 * xh_px
    draw_strokes(canvas, [PxStroke(s.points * xh_px + (ox, oy), s.widths * pen_px)
                          for s in strokes], spacing=1.2)
    yy, xx = np.nonzero(canvas > 110)
    # return pixel coords in x-height-scaled glyph space (origin = glyph origin)
    return (xx - ox), (yy - oy)


def _variant(font: str, ch: str, rng: np.random.Generator, messiness: float):
    g = font_provider(font).source.glyph(ch)
    if g is None or not g.strokes:
        return None
    p = HandParams(messiness=messiness, slant_deg=float(rng.normal(0, 3)))
    strokes, adv = glyph_instance(g, p, rng, cursive=font in ("homemade-apple",
                                                               "cedarville-cursive"))
    return strokes, adv


def _shift(strokes: list[Stroke], dx: float) -> list[Stroke]:
    return [Stroke(s.points + (dx, 0.0), s.widths, s.closed) for s in strokes]


def _clip_x(strokes: list[Stroke], lo: float, hi: float) -> list[Stroke]:
    out = []
    for s in strokes:
        keep = (s.points[:, 0] >= lo) & (s.points[:, 0] <= hi)
        if keep.sum() >= 2:
            out.append(Stroke(s.points[keep], s.widths[keep]))
    return out


def sample(ch: str, rng: np.random.Generator) -> np.ndarray | None:
    """One normalised training image for character ``ch`` (or JUNK)."""
    font = TRAIN_FONTS[int(rng.integers(len(TRAIN_FONTS)))]
    mess = float(rng.uniform(0.2, 2.2))
    xh_px = float(rng.uniform(18, 42))
    pen_px = xh_px * float(rng.uniform(0.08, 0.32))

    if ch == JUNK:
        strokes, center = _junk(font, rng, mess)
        if not strokes:
            return None
    else:
        v = _variant(font, ch, rng, mess)
        if v is None:
            return None
        strokes, adv = v
        center = adv / 2
        # Imperfect cuts: bits of neighbouring letters at the band edges.
        if rng.random() < 0.35:
            nb = _variant(font, str(rng.choice(list("aeimnorsuc"))), rng, mess)
            if nb is not None:
                left = rng.random() < 0.5
                ns, nadv = nb
                if left:
                    piece = _clip_x(_shift(ns, -nadv), -rng.uniform(0.05, 0.3), 0.0)
                else:
                    piece = _clip_x(_shift(ns, adv), adv, adv + rng.uniform(0.05, 0.3))
                strokes = strokes + piece
    xs, ys = _ink_pixels(strokes, xh_px, pen_px, rng)
    if len(xs) == 0:
        return None
    xh_est = xh_px * float(rng.normal(1.0, 0.07))  # x-height measurement error
    jitter = float(rng.normal(0, 0.06)) * xh_px
    return raster(xs.astype(np.float64), ys.astype(np.float64) + rng.normal(0, 0.05) * xh_px,
                  center * xh_px + jitter, xh_est)


def _junk(font: str, rng: np.random.Generator, mess: float):
    kind = rng.integers(3)
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHKMNRSW"
    a = _variant(font, str(rng.choice(list(chars))), rng, mess)
    if a is None:
        return [], 0.0
    sa, adv_a = a
    if kind == 0:  # left or right part of a letter
        cut = adv_a * rng.uniform(0.3, 0.6)
        if rng.random() < 0.5:
            return _clip_x(sa, -1, cut), cut / 2
        return _shift(_clip_x(sa, cut, 9), -cut), (adv_a - cut) / 2
    if kind == 1:  # right half of one letter + left half of the next
        b = _variant(font, str(rng.choice(list(chars))), rng, mess)
        if b is None:
            return [], 0.0
        sb, adv_b = b
        ca, cb = adv_a * rng.uniform(0.4, 0.6), adv_b * rng.uniform(0.4, 0.6)
        left = _shift(_clip_x(sa, ca, 9), -ca)
        right = _shift(_clip_x(sb, -1, cb), adv_a - ca)
        return left + right, (adv_a - ca + cb) / 2
    # a bare connecting stroke
    t = np.linspace(0, 1, 20)[:, None]
    p0 = np.array([0.0, rng.uniform(-0.2, 0.0)])
    p1 = np.array([rng.uniform(0.3, 0.8), rng.uniform(-1.0, -0.3)])
    pts = p0 + (p1 - p0) * t + np.column_stack([np.zeros(20), 0.1 * np.sin(np.pi * t[:, 0])])
    return [Stroke(pts, np.ones(20))], float(p1[0] / 2)


def make_dataset(per_class: int, seed: int = 0, junk_factor: float = 4.0,
                 progress=None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    total = len(CHARSET)
    for k, ch in enumerate(CHARSET):
        n = int(per_class * (junk_factor if ch == JUNK else 1))
        got, tries = 0, 0
        while got < n and tries < n * 3:
            tries += 1
            img = sample(ch, rng)
            if img is not None:
                xs.append(img)
                ys.append(CLASS_OF[ch])
                got += 1
        if progress:
            progress((k + 1) / total)
    return np.stack(xs).reshape(-1, 1, INPUT, INPUT), np.asarray(ys, np.int64)
