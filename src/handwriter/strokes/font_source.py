"""Turn TrueType font glyphs into stroke glyphs (x-height units)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from handwriter.strokes.model import Glyph, Stroke, find_anchors
from handwriter.strokes.skeleton import mask_to_strokes

_REFERENCE_CHARS = "aenosuxcmrvwz"


class FontGlyphSource:
    """Extracts centre-line stroke glyphs from a font, rendered at a fixed x-height."""

    def __init__(self, path: str | Path, target_xheight_px: int = 64, name: str | None = None):
        self.path = Path(path)
        self.name = name or self.path.stem
        probe = ImageFont.truetype(str(self.path), 200)
        xh200 = _xheight(probe)
        size = max(20, round(200 * target_xheight_px / xh200))
        self.font = ImageFont.truetype(str(self.path), size)
        self.xh_px = float(_xheight(self.font))
        cmap = TTFont(str(self.path), lazy=True).getBestCmap() or {}
        self._codepoints = set(cmap)
        self._cache: dict[str, Glyph | None] = {}
        self.space_advance = self.font.getlength(" ") / self.xh_px
        self.ref_radius = self._reference_radius()
        self.baseline_shift = self._baseline_shift()

    # -- public API ----------------------------------------------------------
    def has(self, ch: str) -> bool:
        return len(ch) == 1 and ord(ch) in self._codepoints

    def glyph(self, ch: str) -> Glyph | None:
        if ch not in self._cache:
            self._cache[ch] = self._extract(ch) if self.has(ch) else None
        return self._cache[ch]

    # -- internals -----------------------------------------------------------
    def _render_mask(self, ch: str) -> tuple[np.ndarray, float, float]:
        left, top, right, bottom = self.font.getbbox(ch, anchor="ls")
        pad = 8
        w = max(1, int(right - left) + 2 * pad)
        h = max(1, int(bottom - top) + 2 * pad)
        img = Image.new("L", (w, h), 0)
        ox, oy = pad - left, pad - top  # baseline-left origin inside the image
        ImageDraw.Draw(img).text((ox, oy), ch, font=self.font, fill=255, anchor="ls")
        return np.asarray(img) > 127, float(ox), float(oy)

    def _raw_strokes(self, ch: str):
        mask, ox, oy = self._render_mask(ch)
        return mask_to_strokes(mask), ox, oy

    def _reference_radius(self) -> float:
        radii = []
        for ch in _REFERENCE_CHARS:
            if self.has(ch):
                raw, _, _ = self._raw_strokes(ch)
                radii.extend(r for s in raw for r in s.radius)
        return float(np.median(radii)) if radii else self.xh_px * 0.06

    def _baseline_shift(self) -> float:
        """Where the centre-lines of short letters actually end, relative to the baseline.

        Usually about one ink radius above it, but some script fonts sit letters below
        their own baseline. Shifting by the measured value puts every style's letters
        exactly on the line (the pen radius is added back at layout).
        """
        bottoms = []
        for ch in "acemnorsuvwxz":
            if self.has(ch):
                raw, _, oy = self._raw_strokes(ch)
                if raw:
                    bottoms.append(max(s.points[:, 1].max() for s in raw) - oy)
        return float(np.median(bottoms)) if bottoms else -self.ref_radius

    def _extract(self, ch: str) -> Glyph:
        raw, ox, oy = self._raw_strokes(ch)
        xh = self.xh_px
        oy += self.baseline_shift  # put the centre-line baseline at y = 0
        strokes = [
            Stroke(
                points=np.column_stack([(s.points[:, 0] - ox) / xh, (s.points[:, 1] - oy) / xh]),
                widths=np.clip(s.radius / self.ref_radius, 0.5, 2.0),
                closed=s.closed,
            )
            for s in raw
        ]
        if ch in _ASCENDER_LETTERS:
            _stretch_ascender(strokes, target=_MIN_ASCENDER)
        advance = self.font.getlength(ch) / xh
        entry, exit_ = find_anchors(strokes)
        return Glyph(ch, strokes, advance, entry, exit_, source=self.name)


_ASCENDER_LETTERS = set("bdfhkl")  # not 't': it is naturally shorter
_MIN_ASCENDER = 1.45


def _stretch_ascender(strokes: list[Stroke], target: float) -> None:
    """Some script fonts draw 'l', 'b', 'k'... barely taller than 'e', which hurts
    legibility. Stretch the part above mid x-height so the ascender reaches ``target``."""
    top = -min(s.points[:, 1].min() for s in strokes)
    pivot = 0.5
    if top >= target - 0.08 or top <= pivot + 0.1:
        return
    k = (target - pivot) / (top - pivot)
    for s in strokes:
        y = s.points[:, 1]
        above = y < -pivot
        y[above] = -pivot + (y[above] + pivot) * k


def _xheight(font: ImageFont.FreeTypeFont) -> float:
    _, top, _, _ = font.getbbox("x", anchor="ls")
    xh = -top
    if xh <= 0:  # fonts without a sensible 'x'
        _, top, _, _ = font.getbbox("H", anchor="ls")
        xh = -top * 0.66
    return float(max(xh, 1))
