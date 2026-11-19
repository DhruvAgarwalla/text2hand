"""User handwriting profiles: storage and use as a glyph source.

A profile lives in ``profiles/<name>/profile.json`` and holds every accepted letter
sample (as strokes in x-height units) plus measured style statistics. Pages can be added
to a profile over time; more samples = more natural variety.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

from handwriter.config import PROFILES_DIR
from handwriter.strokes.glyphset import GlyphSet, font_provider, symbol_fallbacks
from handwriter.strokes.model import Glyph, Stroke, find_anchors

VERSION = 1


@dataclass
class Profile:
    name: str
    style: str  # "print" or "cursive"
    glyphs: dict[str, list[Glyph]] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    pages: int = 0

    def add(self, glyphs: list[Glyph]) -> None:
        for g in glyphs:
            self.glyphs.setdefault(g.char, []).append(g)

    @property
    def chars(self) -> str:
        return "".join(sorted(self.glyphs))

    def sample_count(self) -> int:
        return sum(len(v) for v in self.glyphs.values())


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip().replace(" ", "_")
    if not name:
        raise ValueError("Profile name must contain letters or digits.")
    return name[:40]


def profile_path(name: str) -> Path:
    return PROFILES_DIR / safe_name(name) / "profile.json"


# ---------------------------------------------------------------------------
# (de)serialisation
# ---------------------------------------------------------------------------


def _glyph_to_json(g: Glyph) -> dict:
    return {
        "a": round(float(g.advance), 4),
        "s": [
            {"p": np.round(s.points, 4).tolist(), "w": np.round(s.widths, 3).tolist(),
             "c": bool(s.closed)}
            for s in g.strokes
        ],
    }


def _glyph_from_json(ch: str, d: dict, source: str) -> Glyph:
    strokes = [Stroke(np.asarray(s["p"], np.float64).reshape(-1, 2),
                      np.asarray(s["w"], np.float64), bool(s.get("c", False)))
               for s in d["s"]]
    entry, exit_ = find_anchors(strokes)
    return Glyph(ch, strokes, float(d["a"]), entry, exit_, source=source)


def save_profile(p: Profile) -> Path:
    path = profile_path(p.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": VERSION,
        "name": p.name,
        "style": p.style,
        "pages": p.pages,
        "updated": datetime.now().isoformat(timespec="seconds"),
        "stats": p.stats,
        "glyphs": {ch: [_glyph_to_json(g) for g in gs] for ch, gs in sorted(p.glyphs.items())},
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _load_cached.cache_clear()
    return path


def load_profile(name: str) -> Profile:
    return _load_cached(safe_name(name))


@lru_cache(maxsize=8)
def _load_cached(name: str) -> Profile:
    path = profile_path(name)
    if not path.exists():
        raise FileNotFoundError(f"No handwriting profile named {name!r}")
    data = json.loads(path.read_text(encoding="utf-8"))
    glyphs = {ch: [_glyph_from_json(ch, d, f"profile:{name}") for d in lst]
              for ch, lst in data["glyphs"].items()}
    return Profile(data["name"], data["style"], glyphs, data.get("stats", {}),
                   data.get("pages", 0))


def merge_stats(old: dict, new: dict, old_weight: int) -> dict:
    out = dict(new)
    for k, v in old.items():
        if isinstance(v, (int, float)) and isinstance(new.get(k), (int, float)):
            out[k] = (v * old_weight + new[k]) / (old_weight + 1)
    return out


# ---------------------------------------------------------------------------
# Glyph providers
# ---------------------------------------------------------------------------


class ProfileProvider:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.name = f"profile:{profile.name}"
        self.space_advance = float(profile.stats.get("word_gap", 0.95))

    def has(self, ch: str) -> bool:
        return ch in self.profile.glyphs

    def variants(self, ch: str) -> list[Glyph]:
        return self.profile.glyphs.get(ch, [])


class SlantedProvider:
    """Wraps a provider and shears its glyphs to match a user's slant."""

    def __init__(self, inner, slant_deg: float):
        self.inner = inner
        self.name = f"{inner.name}~{slant_deg:.0f}"
        self.space_advance = inner.space_advance
        self.t = float(np.tan(np.deg2rad(slant_deg)))
        self._cache: dict[str, list[Glyph]] = {}

    def has(self, ch: str) -> bool:
        return self.inner.has(ch)

    def variants(self, ch: str) -> list[Glyph]:
        if ch not in self._cache:
            out = []
            for g in self.inner.variants(ch):
                strokes = []
                for s in g.strokes:
                    pts = s.points.copy()
                    pts[:, 0] -= pts[:, 1] * self.t
                    strokes.append(Stroke(pts, s.widths.copy(), s.closed))
                entry, exit_ = find_anchors(strokes)
                out.append(Glyph(g.char, strokes, g.advance, entry, exit_, g.source))
            self._cache[ch] = out
        return self._cache[ch]


def profile_glyphset(profile: Profile) -> GlyphSet:
    slant = float(profile.stats.get("slant_deg", 0.0))
    base_font = "kalam" if profile.style == "print" else "cedarville-cursive"
    fallbacks = [SlantedProvider(font_provider(base_font), slant)]
    fallbacks += [SlantedProvider(p, slant) for p in symbol_fallbacks()]
    return GlyphSet(f"profile:{profile.name}", ProfileProvider(profile), fallbacks,
                    cursive=profile.style == "cursive", defaults={},
                    pen_ratio=float(profile.stats.get("pen_ratio", 0.0)) or None)


def load_profile_glyphset(name: str) -> GlyphSet:
    return profile_glyphset(load_profile(name))
