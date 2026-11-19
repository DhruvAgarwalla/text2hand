"""High-level API: text + settings -> handwritten A4 page images."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from handwriter.config import DEFAULT_DPI, a4_size_px
from handwriter.imperfections.params import HandParams
from handwriter.layout.engine import layout_text
from handwriter.render.pen import INK_COLORS, composite, draw_strokes, ink_texture
from handwriter.strokes.glyphset import GlyphSet, builtin_glyphset
from handwriter.templates.pages import PageSpec, make_page


class RenderSettings(BaseModel):
    style: str = "print"  # built-in style key or "profile:<name>"
    page: str = "ruled-college"  # built-in template key (ignored with custom backgrounds)
    messiness: float = Field(1.0, ge=0.0, le=3.0)
    size: float = Field(0.40, ge=0.15, le=0.7, description="x-height / line spacing")
    pen_mm: float = Field(0.38, ge=0.1, le=1.5)
    ink: str = "blue"
    slant_deg: float | None = None
    word_spacing: float = 1.0
    seed: int | None = None
    dpi: int = DEFAULT_DPI
    # Optional page geometry overrides (mm)
    line_spacing_mm: float | None = None
    top_mm: float | None = None
    left_mm: float | None = None
    right_mm: float | None = None


@dataclass
class RenderResult:
    images: list[Image.Image]
    missing: list[str]
    seed: int


PageFactory = Callable[[int], PageSpec]


def resolve_glyphset(style: str) -> GlyphSet:
    if style.startswith("profile:"):
        from handwriter.capture.profile import load_profile_glyphset

        return load_profile_glyphset(style.split(":", 1)[1])
    return builtin_glyphset(style)


def builtin_page_factory(s: RenderSettings) -> PageFactory:
    def factory(i: int) -> PageSpec:
        return make_page(
            s.page,
            s.dpi,
            line_spacing_mm=s.line_spacing_mm,
            top_mm=s.top_mm,
            left_mm=s.left_mm,
            right_mm=s.right_mm,
            seed=i,
        )

    return factory


def render_pages(
    text: str,
    settings: RenderSettings | None = None,
    *,
    page_factory: PageFactory | None = None,
    glyphs: GlyphSet | None = None,
) -> RenderResult:
    s = settings or RenderSettings()
    seed = s.seed if s.seed is not None else int(np.random.default_rng().integers(1, 2**31))
    rng = np.random.default_rng(seed)
    gs = glyphs or resolve_glyphset(s.style)

    hand_kwargs = dict(gs.defaults)
    hand_kwargs["messiness"] = s.messiness
    hand_kwargs["word_spacing"] = s.word_spacing
    if s.slant_deg is not None:
        hand_kwargs["slant_deg"] = s.slant_deg
    params = HandParams(**hand_kwargs)

    factory = page_factory or builtin_page_factory(s)
    pen_px = s.pen_mm / 25.4 * s.dpi
    size = min(s.size, gs.max_size())
    result = layout_text(text, gs, params, factory, size=size, pen_px=pen_px, rng=rng)

    ink_rgb = INK_COLORS.get(s.ink, INK_COLORS["blue"])
    images = []
    for laid in result.pages:
        bg = laid.page.background
        h, w = bg.shape[:2]
        canvas = np.zeros((h, w), np.uint8)
        draw_strokes(canvas, laid.strokes, spacing=max(1.2, pen_px * 0.4))
        grain = 0.25 if s.ink == "pencil" else 0.12
        alpha = ink_texture(canvas, rng, grain=grain)
        images.append(Image.fromarray(composite(bg, alpha, ink_rgb)))
    return RenderResult(images, result.missing, seed)


def a4_check(img: Image.Image, dpi: int = DEFAULT_DPI) -> bool:
    return img.size == a4_size_px(dpi)
