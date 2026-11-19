"""Glyph sets: a handwriting style = primary glyph source + symbol fallbacks.

A ``GlyphSet`` answers "give me a glyph for this character". The primary source is
either a built-in handwriting font or a user handwriting profile (which may hold several
variants per character). Characters missing from the primary source (most maths and
science symbols) come from fallback fonts, re-stroked with the same pen so they blend in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Protocol

import numpy as np

from handwriter.config import FONTS_DIR
from handwriter.strokes.font_source import FontGlyphSource
from handwriter.strokes.model import Glyph


class GlyphProvider(Protocol):
    name: str
    space_advance: float

    def has(self, ch: str) -> bool: ...

    def variants(self, ch: str) -> list[Glyph]: ...


class FontProvider:
    """Adapter exposing a FontGlyphSource as a provider with a single variant per char."""

    def __init__(self, source: FontGlyphSource):
        self.source = source
        self.name = source.name
        self.space_advance = source.space_advance

    def has(self, ch: str) -> bool:
        return self.source.has(ch)

    def variants(self, ch: str) -> list[Glyph]:
        g = self.source.glyph(ch)
        return [g] if g is not None and (g.strokes or ch.isspace()) else []


@dataclass
class GlyphSet:
    name: str
    primary: GlyphProvider
    fallbacks: list[GlyphProvider] = field(default_factory=list)
    cursive: bool = False
    # Default handwriting parameters that suit this style (merged into HandParams).
    defaults: dict = field(default_factory=dict)

    @property
    def space_advance(self) -> float:
        # Handwritten word gaps are roughly one x-height; some fonts use far less.
        return max(self.primary.space_advance, 0.95)

    def provider_for(self, ch: str) -> GlyphProvider | None:
        for p in [self.primary, *self.fallbacks]:
            if p.has(ch) and p.variants(ch):
                return p
        return None

    def get(self, ch: str, rng: np.random.Generator | None = None) -> Glyph | None:
        p = self.provider_for(ch)
        if p is None:
            return None
        options = p.variants(ch)
        if rng is None or len(options) == 1:
            return options[0]
        return options[int(rng.integers(len(options)))]

    def missing(self, chars: str) -> list[str]:
        return sorted({c for c in chars if not c.isspace() and self.provider_for(c) is None})


# ---------------------------------------------------------------------------
# Built-in styles
# ---------------------------------------------------------------------------

_FONT_FILES = {
    "kalam": "kalam/Kalam-Regular.ttf",
    "architects-daughter": "architects-daughter/ArchitectsDaughter-Regular.ttf",
    "homemade-apple": "homemade-apple/HomemadeApple-Regular.ttf",
    "cedarville-cursive": "cedarville-cursive/Cedarville-Cursive.ttf",
    "noto-sans-math": "noto-sans-math/NotoSansMath-Regular.ttf",
    "noto-sans": "noto-sans/NotoSans-Variable.ttf",
}

BUILTIN_STYLES = {
    # key: (label, font key, cursive?, style defaults)
    "print": ("Print (natural)", "kalam", False, {"slant_deg": 0.0}),
    "print-neat": ("Print (rounded)", "architects-daughter", False, {"slant_deg": 2.0}),
    "cursive": ("Cursive (joined)", "homemade-apple", True, {"slant_deg": 0.0}),
    "cursive-school": ("Cursive (school)", "cedarville-cursive", True, {"slant_deg": 0.0}),
}


@cache
def font_provider(key: str) -> FontProvider:
    return FontProvider(FontGlyphSource(FONTS_DIR / _FONT_FILES[key], name=key))


def symbol_fallbacks() -> list[GlyphProvider]:
    return [font_provider("noto-sans-math"), font_provider("noto-sans")]


def builtin_glyphset(style: str) -> GlyphSet:
    if style not in BUILTIN_STYLES:
        raise KeyError(f"Unknown style {style!r}. Choose from: {', '.join(BUILTIN_STYLES)}")
    _, font_key, cursive, defaults = BUILTIN_STYLES[style]
    return GlyphSet(style, font_provider(font_key), symbol_fallbacks(), cursive, dict(defaults))
