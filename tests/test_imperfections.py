import numpy as np
from scipy import ndimage as ndi

from handwriter.imperfections.params import PRESETS, HandParams
from handwriter.imperfections.word import Run, build_word
from handwriter.render.pen import PxStroke, draw_strokes
from handwriter.strokes.glyphset import builtin_glyphset


def _render(ws, xh=30, pen=4.0):
    canvas = np.zeros((int(4 * xh), int((ws.width + 2) * xh)), np.uint8)
    draw_strokes(
        canvas,
        [PxStroke(s.points * xh + (xh, 2.5 * xh), s.widths * pen) for s in ws.strokes],
    )
    return canvas


def test_same_seed_is_reproducible():
    gs = builtin_glyphset("print")
    p = HandParams()
    a = build_word([Run("hello")], gs, p, np.random.default_rng(7))
    b = build_word([Run("hello")], gs, p, np.random.default_rng(7))
    assert all(np.allclose(x.points, y.points) for x, y in zip(a.strokes, b.strokes, strict=True))


def test_letters_vary_between_occurrences():
    gs = builtin_glyphset("print")
    rng = np.random.default_rng(0)
    a = build_word([Run("a")], gs, HandParams(), rng)
    b = build_word([Run("a")], gs, HandParams(), rng)
    assert not np.allclose(a.strokes[0].points, b.strokes[0].points)


def test_zero_messiness_matches_font():
    gs = builtin_glyphset("print")
    p = HandParams(messiness=0.0, end_taper=0.0)
    ws = build_word([Run("o")], gs, p, np.random.default_rng(0))
    g = gs.get("o")
    assert np.allclose(ws.strokes[0].points, g.strokes[0].points)


def test_presets_scale_variation():
    assert PRESETS["neat"] < PRESETS["natural"] < PRESETS["hurried"]
    assert HandParams.preset("hurried").jitter("warp") > HandParams.preset("neat").jitter("warp")


def test_cursive_word_is_connected():
    gs = builtin_glyphset("cursive-school")
    p = HandParams(pen_lift_prob=0.0)
    ws = build_word([Run("summer")], gs, p, np.random.default_rng(3))
    _, n = ndi.label(_render(ws) > 64)
    assert n <= 2, f"cursive word broke into {n} pieces"


def test_print_word_has_separate_letters():
    gs = builtin_glyphset("print")
    ws = build_word([Run("summer")], gs, HandParams(), np.random.default_rng(3))
    _, n = ndi.label(_render(ws) > 64)
    assert n >= 4


def test_superscript_is_raised_and_smaller():
    gs = builtin_glyphset("print")
    p = HandParams(messiness=0.0)
    ws = build_word([Run("x"), Run("2", level=1)], gs, p, np.random.default_rng(0))
    x_pts = np.concatenate([s.points for s in ws.strokes[:2]])
    two = ws.strokes[-1].points
    assert two[:, 1].max() < x_pts[:, 1].max() - 0.3  # bottom of '2' well above baseline


def test_unknown_character_reported():
    gs = builtin_glyphset("print")
    ws = build_word([Run("a中")], gs, HandParams(), np.random.default_rng(0))
    assert ws.missing == ["中"]
