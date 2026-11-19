"""Learn handwriting from a page + transcript: the full capture pipeline.

    page image --preprocess--> ink mask --segment--> lines / words
    words <--align--> transcript words --charseg--> letter masks
    letter masks --skeleton--> stroke glyphs --validate--> profile samples
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from handwriter.capture.align import align_words
from handwriter.capture.charseg import CharCut, segment_word
from handwriter.capture.preprocess import InkPage, extract_ink, load_page_image
from handwriter.capture.segment import Segmentation, segment_page
from handwriter.layout.text import parse_text
from handwriter.render.pen import PxStroke, draw_strokes
from handwriter.strokes.glyphset import builtin_glyphset
from handwriter.strokes.model import Glyph, Stroke, find_anchors
from handwriter.strokes.skeleton import mask_to_strokes

SUGGESTED_TEXT = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs!\n"
    "How vexingly quick daft zebras jump; sphinx of black quartz, judge my vow?\n"
    "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.\n"
    "0123456789 (a+b) = c - d * e / f, 50% & #1 @ 3:45 \"yes\" 'no' [x] {y} <z>"
)

# Letters whose shape tells us where they must sit vertically (x-height units, y down).
_XHEIGHT = set("acemnorsuvwxz")
_ASCENDERS = set("bdhkl")
_DESCENDERS = set("gjpqy")


@dataclass
class CapturedGlyph:
    id: int
    glyph: Glyph
    word: str
    suspect: bool  # failed a plausibility check: excluded unless the user keeps it
    reason: str = ""
    confidence: float | None = None  # recogniser's P(letter) when available


@dataclass
class CaptureResult:
    style: str
    glyphs: list[CapturedGlyph]
    stats: dict
    seg: Segmentation
    page: InkPage
    cuts_by_word: list[tuple[int, list[CharCut], str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def accepted(self) -> list[Glyph]:
        return [c.glyph for c in self.glyphs if not c.suspect]

    def coverage(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.glyphs:
            if not c.suspect:
                out[c.glyph.char] = out.get(c.glyph.char, 0) + 1
        return out


@cache
def _width_prior(style: str):
    gs = builtin_glyphset("print" if style == "print" else "cursive-school")

    def width(ch: str) -> float:
        g = gs.get(ch)
        return g.advance if g is not None else 0.6

    return width


def analyze_page(path: str | Path, transcript: str, style: str = "print",
                 page_index: int = 0, use_ml: bool = True) -> CaptureResult:
    rgb = load_page_image(path, page_index)
    return analyze_image(rgb, transcript, style, use_ml)


def analyze_image(rgb: np.ndarray, transcript: str, style: str = "print",
                  use_ml: bool = True) -> CaptureResult:
    """Learn letter samples. use_ml: use the letter recogniser if it is installed/trained."""
    if style not in ("print", "cursive"):
        raise ValueError("style must be 'print' or 'cursive'")
    page = extract_ink(rgb)
    seg = segment_page(page.ink, page.rules_y)
    warnings: list[str] = []
    if not seg.lines:
        raise ValueError("No handwriting found on the page. Is the photo sharp and well lit?")

    words = [w.text for p in parse_text(transcript) for w in p.words]
    if not words:
        raise ValueError("The typed text is empty.")
    width = _width_prior(style)
    matches = align_words(seg, words, width)
    if len(seg.words) < 0.6 * len(words) or len(seg.words) > 1.6 * len(words):
        warnings.append(f"Found {len(seg.words)} written words but the text has {len(words)}. "
                        "Check that the typed text matches the page exactly.")

    r_ref = _ink_radius(page.ink)
    cursive = style == "cursive"
    rec = None
    if use_ml:
        from handwriter.ml.recognizer import get_recognizer

        rec = get_recognizer()
    # First pass: geometry-only cuts (fast) to measure this writer's letter widths.
    captured, cuts_by_word = _learn(seg, matches, words, width, cursive, r_ref, None, rec)
    # Second pass: re-split words with the writer's own widths, guided by the recogniser.
    refined = _refined_widths(captured, width)
    captured2, cuts2 = _learn(seg, matches, words, refined, cursive, r_ref, rec, rec)
    if sum(not c.suspect for c in captured2) >= sum(not c.suspect for c in captured):
        captured, cuts_by_word = captured2, cuts2

    stats = _stats(seg, r_ref)
    res = CaptureResult(style, captured, stats, seg, page, cuts_by_word, warnings)
    if not res.accepted():
        warnings.append("No letters could be learned. Try a clearer photo, or check the text.")
    return res


def _refined_widths(captured: list[CapturedGlyph], prior):
    obs: dict[str, list[float]] = {}
    for c in captured:
        if not c.suspect:
            obs.setdefault(c.glyph.char, []).append(c.glyph.advance)
    ratios = [np.median(v) / prior(ch) for ch, v in obs.items() if prior(ch) > 0.1]
    scale = float(np.median(ratios)) if ratios else 1.0

    def width(ch: str) -> float:
        base = prior(ch) * scale
        v = obs.get(ch)
        if not v:
            return base
        n = len(v)
        return (n * float(np.median(v)) + 2 * base) / (n + 2)

    return width


def _learn(seg: Segmentation, matches, words: list[str], width, cursive: bool, r_ref: float,
           rec=None, verifier=None):
    """Cut matched words into letters. rec guides the cuts; verifier checks the results."""
    t = np.tan(np.deg2rad(seg.slant_deg))
    captured: list[CapturedGlyph] = []
    cuts_by_word = []
    next_id = 1
    for match in matches:
        if len(match.boxes) != 1 or len(match.words) != 1:
            continue  # only learn from confident 1:1 word matches
        box = seg.words[match.boxes[0]]
        text = words[match.words[0]]
        if match.cost > 0.45:
            continue
        cuts = segment_word(seg, box, text, cursive, width, rec, r_ref)
        cuts_by_word.append((match.boxes[0], cuts, text))
        line = seg.lines[box.line]
        word_glyphs = []
        kept_cuts = []
        for cut in cuts:
            g = _cut_to_glyph(cut, line, r_ref, t)
            if g is None:
                continue
            reason = _implausible(g)
            word_glyphs.append(CapturedGlyph(next_id, g, text, bool(reason), reason))
            kept_cuts.append(cut)
            next_id += 1
        if verifier is not None:
            _verify(verifier, word_glyphs, kept_cuts, line, r_ref, t)
        bad = sum(c.suspect for c in word_glyphs)
        if word_glyphs and bad / len(word_glyphs) > 0.4:
            for c in word_glyphs:  # probably misaligned: distrust the whole word
                if not c.suspect:
                    c.suspect, c.reason = True, "word looked misaligned"
        captured.extend(word_glyphs)

    _flag_outliers(captured)
    return captured, cuts_by_word


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _verify(rec, glyphs: list[CapturedGlyph], cuts: list[CharCut], line, r_ref: float,
            t: float) -> None:
    """Ask the recogniser whether each cut-out looks like its letter."""
    from handwriter.ml.features import CLASS_OF, raster

    if not glyphs:
        return
    images = []
    for cut in cuts:
        yy, xx = np.nonzero(cut.mask)
        x, y = xx + cut.x0, yy + cut.y0
        base = line.base_y(x) - r_ref
        ds = x + t * (y - base)
        images.append(raster(ds, y - base, cut.origin_x + cut.advance_px / 2, line.xheight))
    lp = rec.log_probs(np.stack(images))
    for c, row in zip(glyphs, lp, strict=True):
        cls = CLASS_OF.get(c.glyph.char)
        if cls is None:
            continue
        c.confidence = float(np.exp(row[cls]))
        rank = int((row > row[cls]).sum())
        if not c.suspect and rank >= 5 and c.confidence < 0.01:
            c.suspect, c.reason = True, "does not look like this letter"


def _ink_radius(ink: np.ndarray) -> float:
    sk = skeletonize(ink)
    if not sk.any():
        return 2.0
    return float(np.median(ndi.distance_transform_edt(ink)[sk]))


def _cut_to_glyph(cut: CharCut, line, r_ref: float, t: float) -> Glyph | None:
    raw = mask_to_strokes(cut.mask, spur_length=2.0 * r_ref + 2.0)
    if not raw:
        return None
    xh = max(line.xheight, 1.0)
    strokes = []
    for s in raw:
        x = s.points[:, 0] + cut.x0
        y = s.points[:, 1] + cut.y0
        base = line.base_y(x) - r_ref  # centre-line baseline
        gx = (x - cut.origin_x) / xh
        gy = (y - base) / xh
        strokes.append(Stroke(np.column_stack([gx, gy]), np.clip(s.radius / r_ref, 0.5, 2.0),
                              s.closed))
    entry, exit_ = find_anchors(strokes)
    return Glyph(cut.char, strokes, cut.advance_px / xh, entry, exit_, source="capture")


def _implausible(g: Glyph) -> str:
    _, top, _, bottom = g.bbox()
    top, bottom = -top, bottom  # heights above / below the baseline
    c = g.char
    if c in _XHEIGHT and (top > 1.5 or bottom > 0.45 or top < 0.45):
        return "wrong height for a short letter"
    if c in _ASCENDERS and top < 1.1:
        return "too short for a tall letter"
    if c in _DESCENDERS and bottom < 0.2:
        return "missing its tail below the line"
    if (c.isupper() or c.isdigit()) and top < 0.9:
        return "too short for a capital/digit"
    if g.advance > 3.0 or g.advance < 0.08:
        return "implausible width"
    return ""


def _flag_outliers(captured: list[CapturedGlyph]) -> None:
    by_char: dict[str, list[CapturedGlyph]] = {}
    for c in captured:
        if not c.suspect:
            by_char.setdefault(c.glyph.char, []).append(c)
    for items in by_char.values():
        if len(items) < 3:
            continue
        adv = np.array([c.glyph.advance for c in items])
        med = np.median(adv)
        for c, a in zip(items, adv, strict=True):
            if a > 1.8 * med or a < 0.5 * med:
                c.suspect, c.reason = True, "much wider/narrower than other samples"


def _stats(seg: Segmentation, r_ref: float) -> dict:
    xh = float(np.median([ln.xheight for ln in seg.lines]))
    gaps = []
    for a, b in zip(seg.words, seg.words[1:], strict=False):
        if a.line == b.line:
            gaps.append((b.x0 - a.x1) / max(seg.lines[a.line].xheight, 1))
    return {
        "slant_deg": float(seg.slant_deg),
        "xheight_px": xh,
        "pen_ratio": float(2 * r_ref / max(xh, 1)),
        "word_gap": float(np.clip(np.median(gaps), 0.5, 2.5)) if gaps else 0.95,
        "lines": len(seg.lines),
        "words": len(seg.words),
    }


# ---------------------------------------------------------------------------
# Visualisation for the review screen
# ---------------------------------------------------------------------------


def segmentation_overlay(res: CaptureResult, max_width: int = 1200) -> Image.Image:
    img = res.page.rgb.copy()
    img = (img * 0.55 + 255 * 0.45).astype(np.uint8)  # fade the page
    img[res.page.ink] = (40, 40, 40)
    seg = res.seg
    t = np.tan(np.deg2rad(seg.slant_deg))
    th = max(2, img.shape[1] // 800)
    for ln in seg.lines:
        xs = [seg.comps[i].x0 for i in ln.comps] + [seg.comps[i].x1 for i in ln.comps]
        x0, x1 = min(xs), max(xs)
        cv2.line(img, (x0, int(ln.base_y(x0))), (x1, int(ln.base_y(x1))), (0, 170, 0), th)
    for b in seg.words:
        cv2.rectangle(img, (b.x0, b.y0), (b.x1, b.y1), (60, 110, 230), th)
    for _, cuts, _ in res.cuts_by_word:
        for cut in cuts[1:]:
            xh = 1.3 * seg.lines[0].xheight if seg.lines else 40
            y_base = cut.y0 + cut.mask.shape[0]
            p1 = (int(cut.origin_x), int(y_base))
            p2 = (int(cut.origin_x + t * xh * 1.2), int(y_base - xh * 1.2))
            cv2.line(img, p1, p2, (220, 40, 40), th)
    out = Image.fromarray(img)
    if out.width > max_width:
        out = out.resize((max_width, int(out.height * max_width / out.width)))
    return out


def glyph_sheet(glyphs: list[CapturedGlyph], cols: int = 12, cell: int = 90,
                show_suspect: bool = True) -> Image.Image:
    items = [c for c in glyphs if show_suspect or not c.suspect]
    items.sort(key=lambda c: (c.glyph.char, c.id))
    rows = max(1, (len(items) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    xh = cell * 0.32
    for k, c in enumerate(items):
        cx, cy = (k % cols) * cell, (k // cols) * (cell + 18)
        canvas = np.zeros((cell, cell), np.uint8)
        x0, _, x1, _ = c.glyph.bbox()
        off_x = (cell - (x1 + x0) * xh) / 2
        strokes = [PxStroke(s.points * xh + (off_x, cell * 0.68), s.widths * 2.2)
                   for s in c.glyph.strokes]
        draw_strokes(canvas, strokes)
        col = np.array([200, 40, 40] if c.suspect else [20, 30, 90], np.float32)
        tile = 255 - (canvas[..., None] / 255.0) * (255 - col)
        sheet.paste(Image.fromarray(tile.astype(np.uint8)), (cx, cy))
        draw.line([(cx, cy + int(cell * 0.68)), (cx + cell, cy + int(cell * 0.68))],
                  fill=(200, 220, 240))
        label = f"#{c.id} '{c.glyph.char}'" + (" ✗" if c.suspect else "")
        draw.text((cx + 4, cy + cell + 2), label, fill=(200, 40, 40) if c.suspect else
                  (60, 60, 60), font=font)
    return sheet
