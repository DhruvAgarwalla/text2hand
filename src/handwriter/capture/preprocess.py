"""Clean up a photo/scan of a handwritten page and isolate the handwriting ink.

Steps: load (first PDF page or image) -> normalise size -> flatten photo perspective if a
page outline is visible -> separate ink from paper -> remove printed rules and margin
lines (repairing strokes that crossed them) -> straighten small rotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

TARGET_WIDTH = 2400


@dataclass
class InkPage:
    rgb: np.ndarray  # straightened page image
    ink: np.ndarray  # bool mask of handwriting ink (printed lines removed)
    rules_y: list[float]  # y positions of printed rules (may be empty)
    skew_deg: float
    flattened: bool


def load_page_image(path: str | Path, page: int = 0) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            pix = doc[min(page, len(doc) - 1)].get_pixmap(dpi=300, colorspace=pymupdf.csRGB,
                                                           alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    else:
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    h, w = rgb.shape[:2]
    scale = TARGET_WIDTH / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (TARGET_WIDTH, max(1, round(h * scale))), interpolation=interp)


def flatten_photo(rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """If the photo shows the page outline against a darker background, warp it flat."""
    h, w = rgb.shape[:2]
    small_scale = 800 / max(h, w)
    small = cv2.resize(rgb, None, fx=small_scale, fy=small_scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgb, False
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if not 0.35 < area < 0.97:  # no visible page edge (a scan) or nothing page-like
        return rgb, False
    quad = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    if len(quad) != 4:
        return rgb, False
    pts = quad.reshape(4, 2).astype(np.float32) / small_scale
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    tl, br, tr, bl = pts[np.argmin(s)], pts[np.argmax(s)], pts[np.argmin(d)], pts[np.argmax(d)]
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 200 or height < 200:
        return rgb, False
    m = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]),
                                    np.array([[0, 0], [width, 0], [width, height], [0, height]],
                                             np.float32))
    warped = cv2.warpPerspective(rgb, m, (width, height), flags=cv2.INTER_CUBIC)
    scale = TARGET_WIDTH / width
    return cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), True


def extract_ink(rgb: np.ndarray, flatten: bool = True) -> InkPage:
    flattened = False
    if flatten:
        rgb, flattened = flatten_photo(rgb)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, 51)
    dark = cv2.subtract(bg, gray)  # how much darker than local paper

    # Anything noticeably darker than paper: ink AND printed lines.
    faint = dark > 14
    # Printed rules: long thin horizontal runs; margin lines: long vertical runs.
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 12), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 10)))
    hlines = cv2.morphologyEx(faint.astype(np.uint8), cv2.MORPH_OPEN, hk)
    vlines = cv2.morphologyEx(faint.astype(np.uint8), cv2.MORPH_OPEN, vk)
    lines = cv2.dilate(np.maximum(hlines, vlines), np.ones((3, 3), np.uint8)).astype(bool)

    # Ink threshold from the non-line dark pixels (Otsu), never below a sane floor.
    cand = dark[faint & ~lines]
    if cand.size > 100:
        t, _ = cv2.threshold(cand.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(np.clip(t, 25, 120))
    else:
        t = 40.0
    ink = dark > t
    # Remove printed lines, but keep ink that is much darker than the line itself.
    line_level = np.median(dark[lines]) if lines.any() else 0
    ink &= ~(lines & (dark < max(t, line_level * 1.6)))
    # Repair handwriting strokes that were cut where they crossed a rule.
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
    ink |= closed.astype(bool) & lines
    ink = _remove_specks(ink, min_area=12)

    rules = _rule_positions(hlines, w)
    skew = _skew(ink, rules, hlines)
    if abs(skew) > 0.1:
        rgb = _rotate(rgb, skew, (255, 255, 255))
        ink = _rotate(ink.astype(np.uint8) * 255, skew, 0) > 127
        hl = _rotate(hlines * 255, skew, 0) > 127
        rules = _rule_positions(hl.astype(np.uint8), w)
    return InkPage(rgb, ink, rules, skew, flattened)


def _remove_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _rule_positions(hlines: np.ndarray, w: int) -> list[float]:
    cov = hlines.sum(1) / w
    rows = np.nonzero(cov > 0.3)[0]
    out, start, prev = [], None, None
    for r in rows:
        if start is None:
            start = prev = r
        elif r > prev + 2:
            out.append((start + prev) / 2)
            start = r
        prev = r
    if start is not None:
        out.append((start + prev) / 2)
    return [float(v) for v in out]


def _skew(ink: np.ndarray, rules: list[float], hlines: np.ndarray) -> float:
    src = hlines if len(rules) >= 3 else ink.astype(np.uint8)
    small = cv2.resize(src.astype(np.uint8) * 255, None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    best, score_best = 0.0, -1.0
    for a in np.arange(-4.0, 4.01, 0.1):
        rot = _rotate(small, a, 0) if abs(a) > 1e-6 else small
        score = float(np.var(rot.sum(1, dtype=np.float64)))
        if score > score_best:
            best, score_best = float(a), score
    return best


def _rotate(img: np.ndarray, angle: float, border) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)
