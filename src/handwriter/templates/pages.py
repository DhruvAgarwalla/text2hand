"""A4 page templates.

A ``PageSpec`` is everything the layout engine needs to know about a page: the
background image, where each writing line (baseline) is, and the writable horizontal
range on each line. Built-in templates are drawn procedurally; scanned pages produce
the same structure via line detection (see ``templates.scanned``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from handwriter.config import DEFAULT_DPI, a4_size_px, mm_to_px

PAPER_RGB = (252, 251, 246)
RULE_RGB = (150, 184, 222)
MARGIN_RGB = (222, 120, 132)


@dataclass
class LineSlot:
    y: float  # baseline (rule) y position in px
    x_left: float
    x_right: float


@dataclass
class PageSpec:
    background: np.ndarray  # (H, W, 3) uint8 RGB
    lines: list[LineSlot]
    line_spacing: float  # px
    ruled: bool  # True: text must sit on visible rules; False: virtual baselines
    dpi: int = DEFAULT_DPI
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.background.shape[:2]
        return w, h


@dataclass(frozen=True)
class TemplateDef:
    label: str
    ruled: bool
    spacing_mm: float
    margin_line: bool = False
    top_mm: float = 30.0


TEMPLATES: dict[str, TemplateDef] = {
    "blank": TemplateDef("Blank", ruled=False, spacing_mm=8.5, top_mm=25.0),
    "ruled-wide": TemplateDef("Ruled - wide (8.7 mm)", ruled=True, spacing_mm=8.7),
    "ruled-college": TemplateDef("Ruled - college (7.1 mm)", ruled=True, spacing_mm=7.1),
    "ruled-narrow": TemplateDef("Ruled - narrow (6.35 mm)", ruled=True, spacing_mm=6.35),
    "ruled-margin": TemplateDef(
        "Ruled with margin (7.1 mm)", ruled=True, spacing_mm=7.1, margin_line=True
    ),
}

MARGIN_LINE_MM = 32.0
BOTTOM_MARGIN_MM = 14.0
RIGHT_MARGIN_MM = 12.0


def paper(dpi: int = DEFAULT_DPI, seed: int = 0, color=PAPER_RGB) -> np.ndarray:
    """Off-white paper with a very faint fibre texture."""
    w, h = a4_size_px(dpi)
    rng = np.random.default_rng(seed)
    base = np.empty((h, w, 3), np.float32)
    base[:] = color
    tex = rng.normal(0, 1, (h // 4 + 1, w // 4 + 1)).astype(np.float32)
    tex = cv2.resize(cv2.GaussianBlur(tex, (0, 0), 1.2), (w, h), interpolation=cv2.INTER_LINEAR)
    base += tex[..., None] * 1.6
    return np.clip(base, 0, 255).astype(np.uint8)


def make_page(
    template: str = "ruled-college",
    dpi: int = DEFAULT_DPI,
    *,
    line_spacing_mm: float | None = None,
    top_mm: float | None = None,
    left_mm: float | None = None,
    right_mm: float | None = None,
    seed: int = 0,
) -> PageSpec:
    """Create one of the built-in A4 page templates."""
    if template not in TEMPLATES:
        raise KeyError(f"Unknown page template {template!r}. Choose from: {', '.join(TEMPLATES)}")
    t = TEMPLATES[template]
    spacing_mm = line_spacing_mm or t.spacing_mm
    top = top_mm if top_mm is not None else t.top_mm
    w, h = a4_size_px(dpi)
    img = paper(dpi, seed)

    spacing_f = max(spacing_mm, 2.0) / 25.4 * dpi
    ys = []
    y = top / 25.4 * dpi
    bottom = h - BOTTOM_MARGIN_MM / 25.4 * dpi
    while y <= bottom:
        ys.append(y)
        y += spacing_f

    if t.ruled:
        thick = max(1, mm_to_px(0.22, dpi))
        for yy in ys:
            yi = int(round(yy))
            cv2.line(img, (0, yi), (w - 1, yi), RULE_RGB, thick, cv2.LINE_AA)
    if t.margin_line:
        x = mm_to_px(MARGIN_LINE_MM, dpi)
        cv2.line(img, (x, 0), (x, h - 1), MARGIN_RGB, max(1, mm_to_px(0.3, dpi)), cv2.LINE_AA)

    if left_mm is not None:
        x_left = left_mm / 25.4 * dpi
    elif t.margin_line:
        x_left = (MARGIN_LINE_MM + 3.0) / 25.4 * dpi
    else:
        x_left = (20.0 if not t.ruled else 14.0) / 25.4 * dpi
    x_right = w - (right_mm if right_mm is not None else RIGHT_MARGIN_MM) / 25.4 * dpi

    lines = [LineSlot(yy, x_left, x_right) for yy in ys]
    return PageSpec(img, lines, spacing_f, t.ruled, dpi, template)
