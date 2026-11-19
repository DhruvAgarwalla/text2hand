"""Use the user's own scanned page (PDF or image) as the paper.

Steps: load each page at the target DPI -> fit to A4 -> straighten small rotations ->
detect printed rules and the margin line -> build a PageSpec whose baselines are the
detected rules. Every detected value can be overridden manually (calibration) for pages
where detection is unreliable (faint lines, coloured paper, photos).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from handwriter.config import DEFAULT_DPI, a4_size_px
from handwriter.templates.pages import PAPER_RGB, LineSlot, PageSpec

MAX_PAGES = 30
DEFAULT_VIRTUAL_SPACING_MM = 8.5


@dataclass
class Calibration:
    """Manual overrides (millimetres). None = use the detected / default value."""

    line_spacing_mm: float | None = None
    first_line_mm: float | None = None
    left_mm: float | None = None
    right_mm: float | None = None
    bottom_mm: float | None = None
    ruled: bool | None = None  # force "letters sit on visible rules" on/off


@dataclass
class DetectedLines:
    ys: list[float]
    spacing: float | None
    margin_x: float | None
    right_x: float | None
    angle_deg: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_background_pages(path: str | Path, dpi: int = DEFAULT_DPI) -> list[np.ndarray]:
    """Load a PDF (all pages) or image file as A4-sized RGB arrays."""
    path = Path(path)
    pages: list[np.ndarray] = []
    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            for page in list(doc)[:MAX_PAGES]:
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
                pages.append(fit_a4(arr.copy(), dpi))
    else:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        pages.append(fit_a4(np.asarray(img), dpi))
    if not pages:
        raise ValueError(f"No pages found in {path}")
    return pages


def fit_a4(rgb: np.ndarray, dpi: int = DEFAULT_DPI) -> np.ndarray:
    """Scale to A4 width; crop or pad (with paper colour) to A4 height.

    Landscape inputs are rotated to portrait first.
    """
    h, w = rgb.shape[:2]
    if w > h * 1.15:
        rgb = np.ascontiguousarray(np.rot90(rgb, -1))
        h, w = rgb.shape[:2]
    tw, th = a4_size_px(dpi)
    scale = tw / w
    nh = max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (tw, nh), interpolation=interp)
    if nh >= th:
        top = (nh - th) // 2
        return resized[top : top + th].copy()
    out = np.empty((th, tw, 3), np.uint8)
    out[:] = _paper_colour(resized)
    top = (th - nh) // 2
    out[top : top + nh] = resized
    return out


def _paper_colour(rgb: np.ndarray) -> tuple[int, int, int]:
    sample = rgb[:: max(1, rgb.shape[0] // 64), :: max(1, rgb.shape[1] // 64)].reshape(-1, 3)
    bright = sample[sample.sum(1) >= np.percentile(sample.sum(1), 60)]
    return tuple(int(v) for v in np.median(bright if len(bright) else sample, 0)) or PAPER_RGB


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _darkness(rgb: np.ndarray) -> np.ndarray:
    """How much darker than the local paper each pixel is (0..255, uint8)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Use the colour channel with most contrast too: blue rules are faint in grey.
    r = rgb[..., 0]
    gray = np.minimum(gray, r)
    bg = cv2.medianBlur(gray, 41)
    return cv2.subtract(bg, gray)


def detect_lines(rgb: np.ndarray, dpi: int = DEFAULT_DPI) -> DetectedLines:
    h, w = rgb.shape[:2]
    dark = _darkness(rgb)
    thr = max(10, int(np.percentile(dark, 99.0) * 0.35))
    binary = (dark > thr).astype(np.uint8)

    angle = _estimate_skew(binary)
    if abs(angle) > 0.08:
        binary = _rotate(binary, angle, border=0)

    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 10), 1)))
    coverage = horiz.sum(1) / w
    rows = np.nonzero(coverage > 0.30)[0]
    ys = _group_rows(rows)
    min_y = 6 / 25.4 * dpi  # ignore the page edge
    ys = [y for y in ys if min_y < y < h - min_y]
    spacing = None
    if len(ys) >= 3:
        diffs = np.diff(ys)
        spacing = float(np.median(diffs))
        ys = _regularise(ys, spacing)

    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 8))))
    vcov = vert.sum(0) / h
    margin_x = _strongest(vcov, 0, int(w * 0.35), 0.4)
    right_x = _strongest(vcov, int(w * 0.75), w, 0.4)
    return DetectedLines(ys, spacing, margin_x, right_x, angle)


def _group_rows(rows: np.ndarray) -> list[float]:
    if len(rows) == 0:
        return []
    groups, start, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        if r > prev + 2:
            groups.append((start + prev) / 2)
            start = r
        prev = r
    groups.append((start + prev) / 2)
    return [float(g) for g in groups]


def _regularise(ys: list[float], spacing: float) -> list[float]:
    """Drop stray detections off the ruling grid and fill single missing rules."""
    out = [ys[0]]
    for y in ys[1:]:
        gap = y - out[-1]
        k = round(gap / spacing)
        if k == 0 or abs(gap - k * spacing) > 0.25 * spacing:
            continue  # not on the grid (text, smudge, double header line)
        for _ in range(1, k):  # missing rules in between
            out.append(out[-1] + spacing)
        out.append(y)
    return out


def _strongest(cov: np.ndarray, lo: int, hi: int, min_cov: float) -> float | None:
    seg = cov[lo:hi]
    if len(seg) == 0 or seg.max() < min_cov:
        return None
    return float(lo + np.argmax(seg))


def _estimate_skew(binary: np.ndarray) -> float:
    small = cv2.resize(binary * 255, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    best, best_score = 0.0, -1.0
    for a in np.arange(-3.0, 3.01, 0.1):
        rot = _rotate(small, a, border=0) if abs(a) > 1e-6 else small
        score = float(np.var(rot.sum(1, dtype=np.float64)))
        if score > best_score:
            best, best_score = float(a), score
    return best


def _rotate(img: np.ndarray, angle: float, border) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


# ---------------------------------------------------------------------------
# PageSpec construction
# ---------------------------------------------------------------------------


def page_spec_from_background(rgb: np.ndarray, dpi: int = DEFAULT_DPI,
                              cal: Calibration | None = None, name: str = "custom") -> PageSpec:
    cal = cal or Calibration()
    det = detect_lines(rgb, dpi)
    if abs(det.angle_deg) > 0.08:
        rgb = _rotate(rgb, det.angle_deg, border=_paper_colour(rgb))
    h, w = rgb.shape[:2]
    mm = dpi / 25.4

    detected_ruled = len(det.ys) >= 5
    ruled = cal.ruled if cal.ruled is not None else detected_ruled

    if cal.line_spacing_mm or cal.first_line_mm is not None or not detected_ruled:
        spacing = (cal.line_spacing_mm * mm if cal.line_spacing_mm
                   else det.spacing if detected_ruled else DEFAULT_VIRTUAL_SPACING_MM * mm)
        first = (cal.first_line_mm * mm if cal.first_line_mm is not None
                 else det.ys[0] if detected_ruled else 25 * mm)
        bottom = h - (cal.bottom_mm if cal.bottom_mm is not None else 12) * mm
        ys = list(np.arange(first, bottom + 1e-6, spacing))
    else:
        spacing = det.spacing
        bottom = h - (cal.bottom_mm if cal.bottom_mm is not None else 6) * mm
        ys = [y for y in det.ys if y <= bottom]

    if cal.left_mm is not None:
        x_left = cal.left_mm * mm
    elif det.margin_x is not None:
        x_left = det.margin_x + 3 * mm
    else:
        x_left = 15 * mm
    if cal.right_mm is not None:
        x_right = w - cal.right_mm * mm
    elif det.right_x is not None:
        x_right = det.right_x - 2 * mm
    else:
        x_right = w - 12 * mm

    lines = [LineSlot(float(y), x_left, x_right) for y in ys]
    meta = {"detected_lines": len(det.ys), "skew_deg": det.angle_deg,
            "margin_detected": det.margin_x is not None}
    return PageSpec(rgb, lines, float(spacing), ruled, dpi, name, meta)


def scanned_page_factory(path: str | Path, settings=None, cal: Calibration | None = None):
    """Page factory cycling through the pages of an uploaded background file."""
    dpi = getattr(settings, "dpi", DEFAULT_DPI)
    if cal is None and settings is not None:
        cal = Calibration(line_spacing_mm=settings.line_spacing_mm,
                          first_line_mm=settings.top_mm, left_mm=settings.left_mm,
                          right_mm=settings.right_mm)
    specs = [page_spec_from_background(p, dpi, cal, Path(path).name)
             for p in load_background_pages(path, dpi)]
    return lambda i: specs[i % len(specs)]


def detection_overlay(spec: PageSpec) -> Image.Image:
    """Visualise the writing lines and margins on top of the page (for calibration)."""
    img = spec.background.copy()
    t = max(1, round(spec.dpi / 150))
    for s in spec.lines:
        y = int(round(s.y))
        cv2.line(img, (int(s.x_left), y), (int(s.x_right), y), (0, 170, 0), t * 2)
    if spec.lines:
        x0, x1 = int(spec.lines[0].x_left), int(spec.lines[0].x_right)
        cv2.line(img, (x0, 0), (x0, img.shape[0]), (220, 0, 0), t * 2)
        cv2.line(img, (x1, 0), (x1, img.shape[0]), (220, 0, 0), t * 2)
    return Image.fromarray(img)
