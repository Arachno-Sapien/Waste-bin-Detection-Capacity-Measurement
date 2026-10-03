"""
Shared per-frame pipeline: the one code path the app, the headless CLI and
evaluate.py all run, so measured accuracy is the accuracy the app delivers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings
from detectors.baseline import pick_reference
from detectors.bin_detector import BinDetection, BinDetector
from detectors.waste_detector import WasteDetection, WasteDetector
from services.occupancy import OccupancyEstimator, OccupancyResult, wall_mask_for


def analyse_frame(
    frame: np.ndarray,
    settings: Settings,
    bin_detector: BinDetector,
    waste_detector: WasteDetector,
    estimator: OccupancyEstimator,
    references: Optional[Dict[int, List[np.ndarray]]] = None,
) -> List[Tuple[BinDetection, OccupancyResult, List[WasteDetection]]]:
    """Detect bins, then waste and occupancy per bin.

    references: {roi_index: [empty-bin crops]} of a fixed camera (Task B4)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = []
    for b in bin_detector.detect(frame):
        x1, y1, x2, y2 = WasteDetector.search_region(
            b.bbox, frame.shape, settings.waste_overflow_band_ratio)
        refs = (references or {}).get(b.roi_index)
        ref = pick_reference(refs, frame[y1:y2, x1:x2]) if refs else None
        wastes = waste_detector.detect(
            frame, b.bbox, b.interior_mask,
            bin_color_name=b.bin_color_name, bin_hsv_range=b.bin_hsv_range,
            reference=ref,
        )
        occluded = waste_detector.last_person_frac > settings.occlusion_person_frac
        occ = estimator.estimate(
            b.bin_id, b.interior_mask, [w.mask for w in wastes],
            b.rim_top_y, b.rim_bottom_y,
            bin_color_mask=wall_mask_for(b, hsv), occluded=occluded,
        )
        results.append((b, occ, wastes))
    return results
