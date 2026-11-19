"""Build one handwritten word: per-letter variation, then cursive joins.

Every letter occurrence becomes a fresh *instance* of its glyph: elastic warp,
rotation, scale, slant change, baseline offset and pen-pressure changes. In cursive
styles, consecutive letters are then connected by a smooth pen path from the exit point
of one letter to the entry point of the next. Joins are computed AFTER the per-letter
variation, so letters stay connected however much they vary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from handwriter.imperfections.noise import Smooth1D, Warp2D
from handwriter.imperfections.params import HandParams
from handwriter.strokes.glyphset import GlyphSet
from handwriter.strokes.model import Anchor, Glyph, Stroke, anchor_point_and_dir

# Superscript / subscript: (scale, baseline shift in x-height units)
SCRIPT_LEVELS = {0: (1.0, 0.0), 1: (0.62, -0.95), -1: (0.62, 0.38)}

_JOINABLE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class Run:
    text: str
    level: int = 0  # 0 normal, 1 superscript, -1 subscript


@dataclass
class WordShape:
    strokes: list[Stroke]  # x-height units; origin = baseline at the word's left edge
    width: float
    text: str
    missing: list[str]


@dataclass
class _Placed:
    char: str
    strokes: list[Stroke]
    entry: Anchor | None
    exit: Anchor | None
    level: int


def glyph_instance(g: Glyph, p: HandParams, rng: np.random.Generator, m: float = 1.0,
                   cursive: bool = False) -> tuple[list[Stroke], float]:
    """Return a randomly varied copy of a glyph's strokes and its varied advance width."""
    rot_k = 0.5 if cursive else 1.0
    rot = np.deg2rad(_clip_normal(rng, p.jitter("rotation_jitter_deg", m * rot_k)))
    scale = 1.0 + _clip_normal(rng, p.jitter("scale_jitter", m))
    slant = np.deg2rad(p.slant_deg + _clip_normal(rng, p.jitter("slant_jitter_deg", m)))
    dy = _clip_normal(rng, p.jitter("baseline_jitter", m))
    warp = Warp2D(rng, p.jitter("warp", m), wavelength=rng.uniform(0.8, 1.6))
    pressure = 1.0 + _clip_normal(rng, p.jitter("pressure_jitter", m))

    cx, cy = g.advance / 2, -0.5
    c, s = np.cos(rot), np.sin(rot)
    rotm = np.array([[c, -s], [s, c]])
    tan_sl = np.tan(slant)
    contrast = float(np.clip(p.thickness_contrast, 0, 1))

    out: list[Stroke] = []
    for st in g.strokes:
        pts = warp(st.points)
        pts = (pts - (cx, cy)) @ rotm.T * scale + (cx, cy)
        pts[:, 0] -= pts[:, 1] * tan_sl  # y is negative above the baseline
        pts[:, 1] += dy
        w = (1 - contrast) + contrast * st.widths
        w = w * pressure * _along_stroke_variation(len(pts), rng, p.jitter("pressure_jitter", m))
        if not st.closed:
            w = w * _taper(pts, p.end_taper)
        out.append(Stroke(pts, w, st.closed))
    return out, g.advance * scale


def build_word(runs: list[Run], gs: GlyphSet, p: HandParams, rng: np.random.Generator,
               m: float = 1.0) -> WordShape:
    """Lay out the letters of one word (no spaces inside) with variation and joins."""
    placed: list[_Placed] = []
    missing: list[str] = []
    x = 0.0
    for run in runs:
        scale, shift = SCRIPT_LEVELS[run.level]
        for ch in run.text:
            g = gs.get(ch, rng)
            if g is None:
                missing.append(ch)
                g = gs.get("?", rng)
                if g is None:
                    continue
            strokes, adv = glyph_instance(g, p, rng, m, cursive=gs.cursive)
            for st in strokes:
                st.points = st.points * scale + (x, shift)
                st.widths = st.widths * (0.85 + 0.15 * scale)
            placed.append(_Placed(ch, strokes, g.entry, g.exit, run.level))
            x += adv * scale + p.letter_spacing * scale
            x += _clip_normal(rng, p.jitter("letter_spacing_jitter", m)) * scale

    all_strokes: list[Stroke] = [st for pl in placed for st in pl.strokes]
    if gs.cursive:
        for a, b in zip(placed, placed[1:], strict=False):
            if _should_join(a, b, p, rng, m):
                bridge = _bridge(a, b)
                if bridge is not None:
                    all_strokes.append(bridge)
    text = "".join(r.text for r in runs)
    return WordShape(all_strokes, max(x, 0.0), text, missing)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clip_normal(rng: np.random.Generator, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return float(np.clip(rng.normal(0, sigma), -2.5 * sigma, 2.5 * sigma))


def _along_stroke_variation(n: int, rng: np.random.Generator, amount: float) -> np.ndarray:
    if n < 3 or amount <= 0:
        return np.ones(n)
    f = Smooth1D(rng, wavelength=max(n / 2.0, 8.0), n=3)
    return 1.0 + 0.5 * amount * f(np.arange(n))


def _taper(pts: np.ndarray, amount: float) -> np.ndarray:
    n = len(pts)
    if n < 4 or amount <= 0:
        return np.ones(n)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    if total < 0.2:  # dots and tiny marks keep full width
        return np.ones(n)
    ramp = 0.12
    start = np.clip(s / ramp, 0, 1)
    end = np.clip((total - s) / ramp, 0, 1)
    return 1.0 - amount * (1 - np.minimum(start, end))


def _should_join(a: _Placed, b: _Placed, p: HandParams, rng: np.random.Generator,
                 m: float) -> bool:
    if a.level != b.level or a.exit is None or b.entry is None:
        return False
    if a.char not in _JOINABLE or b.char not in _JOINABLE or not b.char.islower():
        return False
    if a.char.isupper() and not p.join_capitals:
        return False
    return rng.random() >= min(0.9, p.jitter("pen_lift_prob", m))


def _bridge(a: _Placed, b: _Placed) -> Stroke | None:
    pa, da = anchor_point_and_dir(a.strokes, a.exit, "exit")
    pb, db = anchor_point_and_dir(b.strokes, b.entry, "entry")
    delta = pb - pa
    dist = float(np.linalg.norm(delta))
    if dist < 0.05 or dist > 1.8 or delta[0] < -0.15:
        return None  # already touching, implausibly far, or overlapping letters
    k = 0.4 * dist
    c1, c2 = pa + da * k, pb - db * k
    t = np.linspace(0, 1, max(6, int(dist / 0.025)))[:, None]
    pts = (1 - t) ** 3 * pa + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * pb
    wa = a.strokes[a.exit.stroke].widths[a.exit.index]
    wb = b.strokes[b.entry.stroke].widths[b.entry.index]
    w = np.linspace(wa, wb, len(pts)) * 0.92
    return Stroke(pts, w)
