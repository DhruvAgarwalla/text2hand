import numpy as np
import pytest

from handwriter.render.pen import PxStroke, draw_strokes
from handwriter.strokes.glyphset import BUILTIN_STYLES, builtin_glyphset
from handwriter.strokes.skeleton import mask_to_strokes


def _disk_ring(size=80, r_out=30, r_in=22):
    yy, xx = np.mgrid[:size, :size]
    d = np.hypot(yy - size / 2, xx - size / 2)
    return (d <= r_out) & (d >= r_in)


def test_horizontal_bar_is_one_stroke():
    m = np.zeros((40, 120), bool)
    m[15:25, 10:110] = True
    strokes = mask_to_strokes(m)
    assert len(strokes) == 1
    s = strokes[0]
    assert np.ptp(s.points[:, 0]) > 80  # spans most of the bar
    assert abs(np.median(s.points[:, 1]) - 19.5) < 1.5  # runs down the middle
    assert 3 < np.median(s.radius) < 7


def test_ring_is_single_closed_loop():
    strokes = mask_to_strokes(_disk_ring())
    assert len(strokes) == 1
    assert strokes[0].closed


def test_cross_gives_spokes_without_spurs():
    m = np.zeros((101, 101), bool)
    m[45:56, 5:96] = True
    m[5:96, 45:56] = True
    strokes = mask_to_strokes(m)
    # A plus sign: 4 arms meeting at a junction (or 2 merged strokes), never tiny spurs.
    assert 2 <= len(strokes) <= 4
    for s in strokes:
        assert np.linalg.norm(np.diff(s.points, axis=0), axis=1).sum() > 25


def test_empty_mask():
    assert mask_to_strokes(np.zeros((10, 10), bool)) == []


@pytest.mark.parametrize("style", list(BUILTIN_STYLES))
def test_builtin_styles_cover_required_characters(style):
    gs = builtin_glyphset(style)
    ascii_chars = "".join(chr(c) for c in range(33, 127))
    math = "±×÷≈≠≤≥∞√∑∏∫∂∆∇∈⊂∪∩∀∃→⇒°µπαβγθλσω²³₂½…"
    assert gs.missing(ascii_chars + math) == []


def test_glyph_units_are_xheight_normalised():
    gs = builtin_glyphset("print")
    x = gs.get("x")
    xmin, ymin, xmax, ymax = x.bbox()
    # Centre-line of 'x' spans roughly baseline (0) to x-height (-1), inset by the pen.
    assert -1.05 < ymin < -0.75
    assert -0.25 < ymax < 0.05
    assert x.advance > 0.4


def test_cursive_glyphs_have_join_anchors():
    gs = builtin_glyphset("cursive")
    for ch in "acemnu":
        g = gs.get(ch)
        assert g.entry is not None and g.exit is not None, ch


def test_draw_strokes_marks_ink():
    canvas = np.zeros((50, 100), np.uint8)
    draw_strokes(canvas, [PxStroke(np.array([[10.0, 25.0], [90.0, 25.0]]), np.array([6.0, 6.0]))])
    assert canvas[25, 50] == 255
    assert canvas[5, 50] == 0
    assert canvas[25, 4] == 0
