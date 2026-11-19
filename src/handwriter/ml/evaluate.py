"""Quality metrics for handwriting styles (`handwriter evaluate`).

* line fit     - gap between each line's lowest ink edge and its rule (x-height units);
                 0 = resting exactly on the line, + = floating above, - = dipping below.
                 Measured on text without descenders.
* variation    - mean pixel difference between repeated instances of the same letter
                 (0 = identical copies like a font; higher = more natural variation)
* legibility   - share of individually rendered letters the CNN recogniser reads correctly
                 (top-1 and top-3). The recogniser was trained on the built-in fonts, so
                 for built-in styles this is an optimistic upper bound
* speed        - seconds to render one full A4 page at 300 DPI
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from handwriter.imperfections.params import HandParams
from handwriter.imperfections.word import glyph_instance
from handwriter.layout.engine import layout_text
from handwriter.ml.features import CLASS_OF, raster
from handwriter.pipeline import RenderSettings, render_pages, resolve_glyphset
from handwriter.render.pen import PxStroke, draw_strokes
from handwriter.templates.pages import make_page

TEXT = ("The quick brown fox jumps over the lazy dog. Sphinx of black quartz, judge my vow! "
        "How vexingly quick daft zebras jump; pack my box with five dozen liquor jugs. ") * 6
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# No descenders (g j p q y and commas hang below the line by design).
LINE_TEXT = "We met on the main street in the summer then rode the old tram home at noon. " * 8


@dataclass
class StyleReport:
    style: str
    line_fit_mean: float
    line_fit_worst: float
    variation: float
    legibility_top1: float | None
    legibility_top3: float | None
    seconds_per_page: float


def _instance_image(strokes, adv: float, xh_px: float = 30.0) -> np.ndarray:
    pts = np.concatenate([s.points for s in strokes])
    pad = 8
    ox, oy = pad - pts[:, 0].min() * xh_px, pad - pts[:, 1].min() * xh_px
    w = int(np.ptp(pts[:, 0]) * xh_px + 2 * pad) + 1
    h = int(np.ptp(pts[:, 1]) * xh_px + 2 * pad) + 1
    canvas = np.zeros((h, w), np.uint8)
    draw_strokes(canvas, [PxStroke(s.points * xh_px + (ox, oy), s.widths * 4.0)
                          for s in strokes])
    yy, xx = np.nonzero(canvas > 110)
    return raster((xx - ox).astype(float), (yy - oy).astype(float), adv * xh_px / 2, xh_px)


def evaluate_style(style: str, messiness: float = 1.0, seed: int = 0) -> StyleReport:
    rng = np.random.default_rng(seed)
    gs = resolve_glyphset(style)
    p = HandParams(messiness=messiness, **gs.defaults)

    # line fit on a ruled page
    pen_px = 2.2
    res = layout_text(LINE_TEXT, gs, p, lambda i: make_page("ruled-college", 150, seed=i),
                      size=min(0.4, gs.max_size()), pen_px=pen_px, rng=rng)
    page = res.pages[0]
    ys = np.concatenate([s.points[:, 1] for s in page.strokes])
    xh = res.xheight_px
    gaps = []
    for slot in page.page.lines:
        on = ys[(ys > slot.y - 1.2 * xh) & (ys < slot.y + 0.5 * xh)]
        if len(on):  # lowest ink edge (centre-line + pen radius) vs. the rule
            gaps.append((slot.y - on.max() - pen_px / 2) / xh)
    gaps = np.array(gaps)

    # variation and legibility from individual letter instances
    imgs, labels, var = [], [], []
    for ch in LETTERS:
        g = gs.get(ch)
        if g is None or not g.strokes:
            continue
        inst = [_instance_image(*glyph_instance(g, p, rng, cursive=gs.cursive))
                for _ in range(4)]
        var.append(np.mean([np.abs(inst[i] - inst[j]).mean()
                            for i in range(4) for j in range(i + 1, 4)]))
        imgs.extend(inst)
        labels.extend([CLASS_OF[ch]] * 4)
    top1 = top3 = None
    try:
        from handwriter.ml.recognizer import get_recognizer

        rec = get_recognizer()
    except ImportError:
        rec = None
    if rec is not None and imgs:
        lp = rec.log_probs(np.stack(imgs))
        lab = np.array(labels)
        order = np.argsort(-lp, 1)
        top1 = float((order[:, 0] == lab).mean())
        top3 = float((order[:, :3] == lab[:, None]).any(1).mean())

    t0 = time.time()
    render_pages(TEXT[:900], RenderSettings(style=style, seed=seed, messiness=messiness))
    secs = time.time() - t0
    return StyleReport(style, float(gaps.mean()), float(np.abs(gaps).max()), float(np.mean(var)),
                       top1, top3, secs)
