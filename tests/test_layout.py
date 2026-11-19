import numpy as np
import pytest

from handwriter.config import a4_size_px
from handwriter.imperfections.params import HandParams
from handwriter.layout.engine import layout_text
from handwriter.layout.text import parse_text
from handwriter.pipeline import RenderSettings, render_pages
from handwriter.strokes.glyphset import builtin_glyphset
from handwriter.templates.pages import TEMPLATES, make_page

DPI = 100  # keep tests fast


# --- text parsing -----------------------------------------------------------


def _levels(text):
    return [(r.text, r.level) for w in parse_text(text)[0].words for r in w.runs]


@pytest.mark.parametrize(
    "src, expected",
    [
        ("x^2", [("x", 0), ("2", 1)]),
        ("x^{n+1}", [("x", 0), ("n+1", 1)]),
        ("H_2O", [("H", 0), ("2", -1), ("O", 0)]),
        ("a_{ij}", [("a", 0), ("ij", -1)]),
        ("x_n", [("x", 0), ("n", -1)]),
        ("snake_case", [("snake_case", 0)]),
        (r"a\^b", [("a^b", 0)]),
    ],
)
def test_math_markup(src, expected):
    assert _levels(src) == expected


def test_paragraphs_and_indent():
    paras = parse_text("hello world\n\n    indented line")
    assert [len(p.words) for p in paras] == [2, 0, 2]
    assert paras[2].indent == 4


# --- templates --------------------------------------------------------------


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_templates_are_a4_with_lines(name):
    page = make_page(name, DPI)
    assert page.background.shape[:2][::-1] == a4_size_px(DPI)
    assert len(page.lines) > 20
    gaps = np.diff([s.y for s in page.lines])
    assert np.allclose(gaps, page.line_spacing)


def test_ruled_template_draws_rules_at_baselines():
    page = make_page("ruled-college", 300)
    y = int(round(page.lines[3].y))
    mid = int(round(page.lines[3].y + page.line_spacing / 2))
    x = page.background.shape[1] // 2
    assert page.background[y, x].mean() < page.background[mid, x].mean() - 20


# --- layout -----------------------------------------------------------------


def _layout(text, template="ruled-college", style="print"):
    gs = builtin_glyphset(style)
    return layout_text(
        text, gs, HandParams(), lambda i: make_page(template, DPI, seed=i),
        size=0.4, pen_px=1.5, rng=np.random.default_rng(0),
    )


def test_text_stays_inside_margins():
    res = _layout("lorem ipsum dolor sit amet " * 60)
    page = res.pages[0].page
    xs = np.concatenate([s.points[:, 0] for s in res.pages[0].strokes])
    assert xs.min() >= page.lines[0].x_left - 2
    assert xs.max() <= page.lines[0].x_right + res.xheight_px * 0.6


def test_letters_sit_on_rules():
    res = _layout("mmmm nnnn uuuu " * 10)
    page = res.pages[0].page
    xh = res.xheight_px
    ys = np.concatenate([s.points[:, 1] for s in res.pages[0].strokes])
    gaps = []
    for rule in (s.y for s in page.lines):
        on_line = ys[(ys > rule - 1.2 * xh) & (ys < rule + 0.5 * xh)]
        if len(on_line):
            gaps.append((rule - on_line.max()) / xh)
    assert len(gaps) >= 3
    # The lowest ink of every line rests on its rule: not through it, not floating.
    assert all(-0.15 < g < 0.25 for g in gaps), gaps


def test_overflow_creates_new_pages():
    res = _layout("word " * 3000)
    assert len(res.pages) >= 2
    assert all(p.strokes for p in res.pages)


def test_render_pages_end_to_end():
    r = render_pages("Hello world, E = mc^2", RenderSettings(dpi=DPI, seed=1))
    assert len(r.images) == 1
    assert r.images[0].size == a4_size_px(DPI)
    assert r.missing == []
    # Same seed -> identical output
    r2 = render_pages("Hello world, E = mc^2", RenderSettings(dpi=DPI, seed=1))
    assert np.array_equal(np.asarray(r.images[0]), np.asarray(r2.images[0]))
