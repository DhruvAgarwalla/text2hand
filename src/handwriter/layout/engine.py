"""Layout engine: flow words onto page lines like a person writing.

* Words are placed left-to-right with natural word gaps; a word that does not fit
  moves to the next line (words are never hyphenated; over-long words are split).
* On ruled pages each line's baseline sits ON the printed rule (lifted by half the pen
  width plus a tiny random lift), so letters rest on the line.
* On blank pages lines get a slight random slope.
* Each line wanders slowly up/down (baseline drift), starts at a slightly different x,
  and handwriting gets a bit messier towards the bottom of each page (fatigue).
* Text that does not fit on a page continues on a new page from ``page_factory``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from handwriter.imperfections.noise import Smooth1D
from handwriter.imperfections.params import HandParams
from handwriter.imperfections.word import Run, WordShape, build_word
from handwriter.layout.text import Word, parse_text
from handwriter.render.pen import PxStroke
from handwriter.strokes.glyphset import GlyphSet
from handwriter.templates.pages import PageSpec


@dataclass
class LaidOutPage:
    page: PageSpec
    strokes: list[PxStroke] = field(default_factory=list)


@dataclass
class LayoutResult:
    pages: list[LaidOutPage]
    missing: list[str]
    xheight_px: float


def layout_text(
    text: str,
    glyphs: GlyphSet,
    params: HandParams,
    page_factory: Callable[[int], PageSpec],
    *,
    size: float = 0.40,
    pen_px: float = 4.0,
    rng: np.random.Generator | None = None,
) -> LayoutResult:
    """Flow ``text`` over as many pages as needed.

    size: x-height as a fraction of the page's line spacing.
    pen_px: nominal pen width in pixels.
    """
    rng = rng or np.random.default_rng()
    paragraphs = parse_text(text)
    st = _State(page_factory, rng)
    xh = st.page.line_spacing * size
    missing: set[str] = set()
    space = glyphs.space_advance * params.word_spacing

    for para in paragraphs:
        if not para.words:
            st.next_line()
            continue
        x = st.line_start(params, xh) + para.indent * 0.5 * space * xh
        line_words: list[tuple[WordShape, float]] = []
        queue = list(para.words)
        while queue:
            word = queue.pop(0)
            ws = build_word(word.runs, glyphs, params, rng, st.fatigue(params))
            missing.update(ws.missing)
            w_px = ws.width * xh
            slot = st.slot
            if x + w_px > slot.x_right and line_words:
                _emit_line(st, line_words, params, xh, pen_px)
                st.next_line()
                line_words = []
                x = st.line_start(params, xh)
                slot = st.slot
            if x + w_px > slot.x_right and not line_words:
                head, tail = _split_word(word, ws, (slot.x_right - x) / xh)
                if tail is not None:
                    queue.insert(0, tail)
                    ws = build_word(head.runs, glyphs, params, rng, st.fatigue(params))
                    w_px = ws.width * xh
            line_words.append((ws, x))
            gap = space * (1 + float(np.clip(rng.normal(0, params.jitter("word_spacing_jitter")),
                                             -0.5, 0.6)))
            x += w_px + gap * xh
        _emit_line(st, line_words, params, xh, pen_px)
        st.next_line()

    st.finish()
    return LayoutResult(st.done, sorted(missing), xh)


class _State:
    def __init__(self, factory: Callable[[int], PageSpec], rng: np.random.Generator):
        self.factory = factory
        self.rng = rng
        self.page_index = 0
        self.page = factory(0)
        self.current = LaidOutPage(self.page)
        self.done: list[LaidOutPage] = []
        self.slot_index = 0
        self.used = False

    @property
    def slot(self):
        return self.page.lines[self.slot_index]

    def fatigue(self, p: HandParams) -> float:
        n = max(1, len(self.page.lines) - 1)
        return 1.0 + p.fatigue * self.slot_index / n

    def line_start(self, p: HandParams, xh: float) -> float:
        return self.slot.x_left + abs(self.rng.normal(0, p.jitter("margin_jitter"))) * xh

    def next_line(self):
        self.slot_index += 1
        if self.slot_index >= len(self.page.lines):
            self.done.append(self.current)
            self.page_index += 1
            self.page = self.factory(self.page_index)
            self.current = LaidOutPage(self.page)
            self.slot_index = 0

    def finish(self):
        if self.current.strokes or not self.done:
            self.done.append(self.current)


def _emit_line(st: _State, words: list[tuple[WordShape, float]], p: HandParams, xh: float,
               pen_px: float) -> None:
    if not words:
        return
    rng = st.rng
    page, slot = st.page, st.slot
    m = st.fatigue(p)
    x0 = words[0][1]
    drift = Smooth1D(rng, wavelength=14 * xh)
    drift_amp = p.jitter("baseline_drift", m) * xh
    if page.ruled:
        lift = pen_px * 0.5 + abs(rng.normal(0, 0.02 * p.messiness)) * xh
        slope = 0.0
        drift_amp = min(drift_amp, 0.12 * xh)
    else:
        lift = 0.0
        slope = np.tan(np.deg2rad(rng.normal(0, p.jitter("line_slope_deg"))))
    base_y = slot.y - lift

    for ws, wx in words:
        dy_word = float(np.clip(rng.normal(0, p.jitter("word_baseline_jitter", m)), -0.1, 0.1)) * xh
        darkness = 1.0 - abs(rng.normal(0, p.jitter("ink_variation")))
        for s in ws.strokes:
            px = wx + s.points[:, 0] * xh
            dx = px - x0
            if page.ruled:
                # On a rule people float slightly above the line, rarely through it.
                wander = -drift_amp * (0.5 + 0.5 * drift(dx)) - abs(dy_word)
            else:
                wander = drift_amp * drift(dx) + dy_word + slope * dx
            py = base_y + s.points[:, 1] * xh + wander
            st.current.strokes.append(
                PxStroke(np.column_stack([px, py]), s.widths * pen_px, darkness)
            )


def _split_word(word: Word, ws: WordShape, avail_units: float) -> tuple[Word, Word | None]:
    """Split an over-long word so that the head roughly fits ``avail_units``."""
    total = len(word.text)
    if total <= 1 or ws.width <= 0:
        return word, None
    keep = max(1, min(total - 1, int(total * avail_units / ws.width)))
    head: list[Run] = []
    tail: list[Run] = []
    count = 0
    for r in word.runs:
        if count + len(r.text) <= keep:
            head.append(r)
        elif count >= keep:
            tail.append(r)
        else:
            cut = keep - count
            head.append(Run(r.text[:cut], r.level))
            tail.append(Run(r.text[cut:], r.level))
        count += len(r.text)
    return Word(head), (Word(tail) if tail else None)
