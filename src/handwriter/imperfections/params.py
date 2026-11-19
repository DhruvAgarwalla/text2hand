"""Parameters controlling how 'human' the handwriting looks.

All distances are in x-height units (1.0 = height of a lowercase 'x') unless noted.
``messiness`` scales every random-variation parameter at once; presets are just
messiness values.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PRESETS: dict[str, float] = {"neat": 0.5, "natural": 1.0, "hurried": 1.8}


class HandParams(BaseModel):
    # --- global shape ---
    slant_deg: float = Field(0.0, description="Extra slant; positive leans right")
    letter_spacing: float = Field(0.0, description="Extra space between letters")
    word_spacing: float = Field(1.0, description="Multiplier on the style's space width")

    # --- per-letter variation (standard deviations) ---
    rotation_jitter_deg: float = 1.4
    scale_jitter: float = 0.045
    slant_jitter_deg: float = 1.8
    baseline_jitter: float = 0.035
    letter_spacing_jitter: float = 0.035
    warp: float = 0.03  # smooth elastic distortion amplitude

    # --- per-word / per-line variation ---
    word_spacing_jitter: float = 0.18
    word_baseline_jitter: float = 0.03
    baseline_drift: float = 0.07  # slow up/down wander along a line
    line_slope_deg: float = 0.35  # tilt of each line on blank pages (std)
    margin_jitter: float = 0.25  # variation of where each line starts

    # --- pen & ink ---
    pressure_jitter: float = 0.12
    end_taper: float = 0.25  # stroke ends are thinner by this fraction
    ink_variation: float = 0.06
    thickness_contrast: float = 0.35  # 0 = monoline pen, 1 = keep source thickness

    # --- behaviour ---
    fatigue: float = 0.3  # how much messier writing gets towards the page bottom
    pen_lift_prob: float = 0.04  # chance a cursive join is skipped
    join_capitals: bool = True

    messiness: float = Field(1.0, ge=0.0, le=3.0)

    def jitter(self, name: str, extra: float = 1.0) -> float:
        """A variation parameter scaled by messiness (and any extra factor)."""
        return float(getattr(self, name)) * self.messiness * extra

    @classmethod
    def preset(cls, name: str, **overrides) -> HandParams:
        return cls(messiness=PRESETS[name], **overrides)
