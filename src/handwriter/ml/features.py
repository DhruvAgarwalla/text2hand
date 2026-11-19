"""Shared input normalisation for the letter recogniser.

Both training samples and real cut-outs from a handwritten page are turned into the same
48x48 image: ink is deslanted, scaled so the x-height is ``XH_PX`` pixels, placed with the
baseline on row ``BASE_ROW`` and centred horizontally on the letter's band. Keeping the
baseline and scale fixed lets the network tell apart 'o'/'O', ','/''', 'p'/'P' etc.
"""

from __future__ import annotations

import string

import cv2
import numpy as np

INPUT = 48
XH_PX = 13.0
BASE_ROW = 31

JUNK = "\x00"  # class for "not a single letter" (fragments, two half letters...)
CHARSET = JUNK + "".join(c for c in string.printable if 33 <= ord(c) <= 126)
CLASS_OF = {c: i for i, c in enumerate(CHARSET)}


def raster(ds: np.ndarray, y_rel: np.ndarray, center: float, xheight: float,
           pixel_size: float = 1.0) -> np.ndarray:
    """Rasterise ink pixels into the normalised 48x48 input.

    ds: deslanted x of each ink pixel; y_rel: y relative to the baseline (down = +).
    center: deslanted x of the letter band centre. pixel_size: source pixel edge length.
    """
    if len(ds) == 0:
        return np.zeros((INPUT, INPUT), np.float32)
    s = XH_PX / max(xheight, 1e-3)
    fc = (ds - center) * s + INPUT / 2
    fr = y_rel * s + BASE_ROW
    c0, r0 = np.floor(fc).astype(np.int64), np.floor(fr).astype(np.int64)
    wc, wr = fc - c0, fr - r0
    area = (s * pixel_size) ** 2
    acc = np.zeros(INPUT * INPUT, np.float64)
    # bilinear splatting: each source pixel spreads its area over 4 output pixels
    for dr, dc, w in ((0, 0, (1 - wr) * (1 - wc)), (0, 1, (1 - wr) * wc),
                      (1, 0, wr * (1 - wc)), (1, 1, wr * wc)):
        rr, cc = r0 + dr, c0 + dc
        ok = (cc >= 0) & (cc < INPUT) & (rr >= 0) & (rr < INPUT)
        acc += np.bincount(rr[ok] * INPUT + cc[ok], weights=w[ok] * area,
                           minlength=INPUT * INPUT)
    img = np.clip(acc.reshape(INPUT, INPUT).astype(np.float32) * 1.3, 0, 1)
    img = cv2.GaussianBlur(img, (0, 0), 0.5)
    m = img.max()
    return img / m if m > 0 else img
