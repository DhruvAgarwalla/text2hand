"""Capture pipeline tests on synthetic 'handwritten' pages rendered by the app itself.

The page is written in one built-in style while the segmenter's width priors come from a
different font, so the test exercises real alignment and segmentation, not a lookup.
"""

import numpy as np
import pytest

from handwriter.capture import profile as profile_mod
from handwriter.capture.analyze import analyze_image
from handwriter.capture.profile import Profile, load_profile, profile_glyphset, save_profile
from handwriter.layout.text import parse_text
from handwriter.pipeline import RenderSettings, render_pages

TEXT = (
    "The quick brown fox jumps over the lazy dog.\n"
    "Pack my box with five dozen liquor jugs!\n"
    "How vexingly quick daft zebras jump."
)


@pytest.fixture(scope="module")
def print_capture():
    page = render_pages(TEXT, RenderSettings(style="print-neat", seed=11, ink="black",
                                             pen_mm=0.45)).images[0]
    rgb = np.asarray(page).astype(np.float32)
    rgb += np.random.default_rng(0).normal(0, 4, rgb.shape[:2])[..., None]
    return analyze_image(np.clip(rgb, 0, 255).astype(np.uint8), TEXT, "print")


def test_finds_lines_and_words(print_capture):
    res = print_capture
    n_words = sum(len(p.words) for p in parse_text(TEXT))
    assert len(res.seg.lines) == 3
    assert abs(len(res.seg.words) - n_words) <= 2


def test_learns_most_letters(print_capture):
    cov = print_capture.coverage()
    lowercase = set("abcdefghijklmnopqrstuvwxyz")
    assert len(lowercase & set(cov)) >= 20
    assert len(print_capture.accepted()) >= 60


def test_stats_are_sensible(print_capture):
    st = print_capture.stats
    assert abs(st["slant_deg"]) < 8
    assert 0.5 <= st["word_gap"] <= 2.5
    assert 0.05 < st["pen_ratio"] < 0.6


def test_learned_glyphs_are_normalised(print_capture):
    for g in print_capture.accepted():
        if g.char in "acemnorsuvwxz":
            _, top, _, bottom = g.bbox()
            assert -1.5 < top < -0.45 and bottom < 0.45


def test_profile_roundtrip_and_render(print_capture, tmp_path, monkeypatch):
    monkeypatch.setattr(profile_mod, "PROFILES_DIR", tmp_path)
    p = Profile("unit_test", "print", stats=print_capture.stats, pages=1)
    p.add(print_capture.accepted())
    save_profile(p)
    loaded = load_profile("unit_test")
    assert loaded.sample_count() == p.sample_count()
    gs = profile_glyphset(loaded)
    assert gs.pen_ratio and not gs.cursive
    r = render_pages("quick brown fox 123 ∑", RenderSettings(dpi=100, seed=1), glyphs=gs)
    assert r.missing == []  # unlearned characters fall back to built-in glyphs


def test_cursive_capture_runs():
    page = render_pages(TEXT, RenderSettings(style="cursive", seed=3, ink="black")).images[0]
    res = analyze_image(np.asarray(page), TEXT, "cursive")
    assert res.stats["slant_deg"] > 5  # this font leans right
    assert len(res.accepted()) >= 25


def test_rejects_empty_transcript(print_capture):
    with pytest.raises(ValueError):
        analyze_image(print_capture.page.rgb, "   ", "print")
