"""
Occupancy Estimator — 3-Factor Fill Estimation
=================================================
Implements a robust occupancy algorithm using three complementary signals,
weighted by `Settings.area_weight` / `height_weight` / `overflow_weight`
(see config/settings.py — not repeated here so the two cannot drift):

1. **Pixel Area Ratio**:
   waste_pixels / bin_interior_pixels × 100.
   Directly measures what fraction of the bin interior is occupied.

2. **Vertical Fill Height**:
   Scan rows top→bottom within the bin bbox.  Find the highest row
   containing waste.  Express as percentage of bin height.

3. **Overflow Detection**:
   Check for waste extending ABOVE the rim top line.  If overflow is
   detected, apply a bonus that pushes the score towards 100%.

Final:  fill_pct = area_weight × area_ratio
                 + height_weight × height_ratio
                 + overflow_weight × overflow_score

Temporal smoothing via a per-bin rolling window reduces flickering in
video streams.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from config.settings import FillStatus, Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wall mask resolution
# ---------------------------------------------------------------------------

def wall_mask_for(bin_det, hsv_frame: np.ndarray) -> Optional[np.ndarray]:
    """The bin's wall region: its instance mask if the detector made one,
    else an expanded HSV colour-range match.

    Shared by ui/app.py, main.py, and verify_system.py so the wall-mask
    choice (and the colour-range expansion below) cannot drift between them.
    """
    surface = getattr(bin_det, "surface_mask", None)
    if surface is not None:
        return surface.astype(np.uint8)

    hsv_range = getattr(bin_det, "bin_hsv_range", None)
    if hsv_range is None:
        return None
    lower, upper = hsv_range
    lower_np = np.array([
        max(0, lower[0] - 10), max(0, lower[1] - 20), max(0, lower[2] - 30)
    ], dtype=np.uint8)
    upper_np = np.array([
        min(179, upper[0] + 10), min(255, upper[1] + 20), min(255, upper[2] + 40)
    ], dtype=np.uint8)
    return (cv2.inRange(hsv_frame, lower_np, upper_np) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class OccupancyResult:
    """Computed occupancy for a single bin."""
    bin_id: int
    fill_pct: float                 # Smoothed fill percentage (0–100)
    status: FillStatus              # Discrete fill state
    raw_fill: float                 # Un-smoothed fill for this frame
    area_ratio: float               # Pixel area ratio (0–100)
    height_ratio: float             # Vertical fill height (0–100)
    overflow_score: float           # Overflow score (0–100)
    waste_pixel_count: int          # Total waste pixels
    bin_pixel_count: int            # Total bin interior pixels


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class OccupancyEstimator:
    """
    Estimate how full a bin is using the 3-factor model.
    Maintains per-bin rolling windows for temporal smoothing.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._history: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=settings.smoothing_window)
        )

    def estimate(
        self,
        bin_id: int,
        interior_mask: Optional[np.ndarray],
        waste_masks: List[Optional[np.ndarray]],
        rim_top_y: int,
        rim_bottom_y: int,
        bin_color_mask: Optional[np.ndarray] = None,
    ) -> OccupancyResult:
        """
        Compute the fill percentage for a single bin.

        Parameters
        ----------
        bin_id : int
            Persistent tracking ID.
        interior_mask : np.ndarray or None
            Binary mask (0/1) of the bin interior (full-frame).
        waste_masks : list[np.ndarray]
            List of binary masks for each detected waste item.
        rim_top_y : int
            Y-coordinate of the bin rim top edge.
        rim_bottom_y : int
            Y-coordinate of the bin bottom edge.
        bin_color_mask : np.ndarray or None
            Binary mask of bin-wall-coloured pixels within the interior
            (full-frame). Used for aperture-focused estimation.
        """
        # --- Edge case: no interior mask ---
        if interior_mask is None or interior_mask.sum() == 0:
            return self._finalize(bin_id, 0.0, 0.0, 0.0, 0.0, 0, 0)

        bin_pixel_count = int(interior_mask.sum())

        # --- Build waste union mask ---
        h, w = interior_mask.shape[:2]
        waste_union = np.zeros((h, w), dtype=np.uint8)

        for wm in waste_masks:
            if wm is not None:
                if wm.shape[:2] != (h, w):
                    wm = cv2.resize(wm, (w, h), interpolation=cv2.INTER_NEAREST)
                waste_union = np.logical_or(
                    waste_union.astype(bool), wm.astype(bool)
                ).astype(np.uint8)

        # --- Waste pixels inside the bin interior ---
        waste_inside = np.logical_and(
            waste_union.astype(bool), interior_mask.astype(bool)
        ).astype(np.uint8)
        waste_pixel_count = int(waste_inside.sum())

        # --- Factor 1: APERTURE-FOCUSED Area Ratio ---
        # Old approach: waste_pixels / total_bbox_pixels  → always low because
        #   the bin's solid front wall dominates 60-80% of the bbox.
        # New approach: waste_pixels / opening_pixels, where opening_pixels
        #   = interior pixels that are NOT part of the bin's wall colour.
        # This means: "Of the visible opening area, how much is waste?"

        area_ratio = 0.0
        if bin_color_mask is not None and bin_color_mask.sum() > 0:
            # Wall pixels within the interior
            wall_inside = np.logical_and(
                bin_color_mask.astype(bool), interior_mask.astype(bool)
            ).astype(np.uint8)
            wall_count = int(wall_inside.sum())
            opening_count = bin_pixel_count - wall_count

            # Sanity: opening must be at least 5% of interior, otherwise
            # the bin is probably viewed from the side with no opening visible.
            if opening_count > bin_pixel_count * 0.05:
                area_ratio = min(100.0, (waste_pixel_count / opening_count) * 100.0)
            else:
                # Fallback: conventional ratio but with 2.5× amplification
                area_ratio = min(100.0, (waste_pixel_count / bin_pixel_count) * 100.0 * 2.5)
        else:
            # No color mask available — use conventional ratio with amplification
            area_ratio = min(100.0, (waste_pixel_count / bin_pixel_count) * 100.0 * 2.5)

        # --- Factor 2: Vertical Fill Height ---
        height_ratio = self._compute_vertical_fill(
            waste_union, rim_top_y, rim_bottom_y
        )

        # --- Factor 3: Overflow Detection ---
        overflow_score = self._compute_overflow(
            waste_union, rim_top_y, w
        )

        # --- Combined fill percentage ---
        raw_fill = (
            self._settings.area_weight * area_ratio
            + self._settings.height_weight * height_ratio
            + self._settings.overflow_weight * overflow_score
        )

        # Overflow override: If overflow is explicitly detected, enforce a minimum fill level.
        # Tiered: heavy overflow (≥80) guarantees FULL; moderate (≥50) guarantees NEARLY FULL.
        if overflow_score >= 80.0:
            raw_fill = max(raw_fill, 85.0)   # Guarantees FULL classification
        elif overflow_score >= 50.0:
            raw_fill = max(raw_fill, 70.0)   # Guarantees NEARLY FULL classification

        raw_fill = max(0.0, min(100.0, raw_fill))

        return self._finalize(
            bin_id, raw_fill, area_ratio, height_ratio, overflow_score,
            waste_pixel_count, bin_pixel_count,
        )

    # ---- Factor 2: Vertical fill height -----------------------------------

    def _compute_vertical_fill(
        self,
        waste_mask: np.ndarray,
        rim_top_y: int,
        rim_bottom_y: int,
    ) -> float:
        """
        Scan rows within the bin region.  Find the highest row that
        contains waste pixels.  Express as percentage of bin height.
        """
        bin_height = rim_bottom_y - rim_top_y
        if bin_height <= 0:
            return 0.0

        # Extract the bin vertical strip
        strip = waste_mask[rim_top_y:rim_bottom_y, :]
        if strip.size == 0:
            return 0.0

        # Row-wise sum of waste pixels
        row_sums = np.sum(strip > 0, axis=1)

        # Find the highest row (smallest index from top) with waste
        threshold = strip.shape[1] * 0.02  # At least 2% of row width
        waste_rows = np.where(row_sums > threshold)[0]

        if len(waste_rows) == 0:
            return 0.0

        highest_waste_row = waste_rows[0]

        # Height ratio: closer to top = higher fill
        fill_height = 1.0 - (highest_waste_row / bin_height)
        return max(0.0, min(100.0, fill_height * 100.0))

    # ---- Factor 3: Overflow detection -------------------------------------

    def _compute_overflow(
        self,
        waste_mask: np.ndarray,
        rim_top_y: int,
        frame_width: int,
    ) -> float:
        """
        Check for waste pixels ABOVE the bin rim.  If significant waste
        extends above the rim, return a high overflow score.
        """
        if rim_top_y <= 0:
            return 0.0

        # Look at the region above the rim (up to 30% of rim_top_y)
        check_height = max(10, rim_top_y // 3)
        above_start = max(0, rim_top_y - check_height)

        above_region = waste_mask[above_start:rim_top_y, :]
        if above_region.size == 0:
            return 0.0

        above_pixels = int(np.sum(above_region > 0))
        total_above_pixels = above_region.shape[0] * above_region.shape[1]

        if total_above_pixels == 0:
            return 0.0

        overflow_ratio = above_pixels / total_above_pixels

        # Scale: any significant overflow → high score
        if overflow_ratio > 0.15:
            return 100.0
        elif overflow_ratio > 0.05:
            return 80.0
        elif overflow_ratio > 0.01:
            return 50.0
        return 0.0

    # ---- Internal helpers --------------------------------------------------

    def _finalize(
        self,
        bin_id: int,
        raw_fill: float,
        area_ratio: float,
        height_ratio: float,
        overflow_score: float,
        waste_pixel_count: int,
        bin_pixel_count: int,
    ) -> OccupancyResult:
        """Apply temporal smoothing and classify the fill state."""
        if self._settings.smoothing_enabled:
            self._history[bin_id].append(raw_fill)
            window = self._history[bin_id]
            smoothed = float(np.median(window))
        else:
            smoothed = raw_fill

        status = self._settings.classify_fill(smoothed)

        return OccupancyResult(
            bin_id=bin_id,
            fill_pct=smoothed,
            status=status,
            raw_fill=raw_fill,
            area_ratio=area_ratio,
            height_ratio=height_ratio,
            overflow_score=overflow_score,
            waste_pixel_count=waste_pixel_count,
            bin_pixel_count=bin_pixel_count,
        )

    def reset(self, bin_id: Optional[int] = None) -> None:
        """Clear smoothing history for a specific bin or all bins."""
        if bin_id is not None:
            self._history.pop(bin_id, None)
        else:
            self._history.clear()
