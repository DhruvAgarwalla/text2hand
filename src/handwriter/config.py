"""Global constants and settings shared across modules."""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
PROFILES_DIR = PROJECT_ROOT / "profiles"
MODELS_DIR = PROJECT_ROOT / "models"

# --- Page geometry (A4 only for now) -----------------------------------------
MM_PER_INCH = 25.4
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DEFAULT_DPI = 300
PREVIEW_DPI = 100


def mm_to_px(mm: float, dpi: int = DEFAULT_DPI) -> int:
    """Convert millimetres to pixels at the given DPI."""
    return round(mm / MM_PER_INCH * dpi)


def a4_size_px(dpi: int = DEFAULT_DPI) -> tuple[int, int]:
    """A4 page size in pixels (width, height) at the given DPI."""
    return mm_to_px(A4_WIDTH_MM, dpi), mm_to_px(A4_HEIGHT_MM, dpi)
