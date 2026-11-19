"""Smooth random functions used for natural, non-jittery variation."""

from __future__ import annotations

import numpy as np


class Smooth1D:
    """Band-limited random function f(x) in roughly [-1, 1] (sum of random sines)."""

    def __init__(self, rng: np.random.Generator, wavelength: float, n: int = 4):
        self.freq = 2 * np.pi / (wavelength * rng.uniform(0.5, 1.8, n))
        self.phase = rng.uniform(0, 2 * np.pi, n)
        amp = rng.uniform(0.5, 1.0, n)
        self.amp = amp / np.sqrt((amp**2).sum() / 2)  # unit RMS-ish

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)[..., None]
        return (self.amp * np.sin(self.freq * x + self.phase)).sum(-1) / np.sqrt(len(self.amp))


class Warp2D:
    """Smooth random 2-D displacement field d(x, y) = (dx, dy)."""

    def __init__(self, rng: np.random.Generator, amplitude: float, wavelength: float = 1.2,
                 n: int = 3):
        self.amplitude = amplitude
        k = 2 * np.pi / wavelength
        ang = rng.uniform(0, 2 * np.pi, (2, n))
        mag = k * rng.uniform(0.6, 1.5, (2, n))
        self.kx, self.ky = mag * np.cos(ang), mag * np.sin(ang)
        self.phase = rng.uniform(0, 2 * np.pi, (2, n))
        self.norm = 1 / np.sqrt(n)

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        if self.amplitude == 0 or len(pts) == 0:
            return pts
        x, y = pts[:, 0:1], pts[:, 1:2]
        dx = np.sin(x * self.kx[0] + y * self.ky[0] + self.phase[0]).sum(1)
        dy = np.sin(x * self.kx[1] + y * self.ky[1] + self.phase[1]).sum(1)
        return pts + self.amplitude * self.norm * np.column_stack([dx, dy])
