"""Listing of all available handwriting styles (built-in + user profiles)."""

from __future__ import annotations

from handwriter.config import PROFILES_DIR
from handwriter.strokes.glyphset import BUILTIN_STYLES


def list_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.name for p in PROFILES_DIR.iterdir() if (p / "profile.json").exists())


def available_styles() -> list[tuple[str, str]]:
    """(label, key) pairs for every style the user can pick."""
    items = [(label, key) for key, (label, *_rest) in BUILTIN_STYLES.items()]
    items += [(f"My handwriting: {name}", f"profile:{name}") for name in list_profiles()]
    return items
