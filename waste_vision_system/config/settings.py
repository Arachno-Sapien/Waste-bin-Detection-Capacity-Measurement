"""
Waste Vision System — Central Configuration
=============================================
All tunable parameters, model paths, thresholds, color palettes, and
performance settings live here.  Swap model weights or adjust fill-state
boundaries without touching any pipeline logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = _PROJECT_ROOT / "models"
EXPORT_DIR = _PROJECT_ROOT / "exports"
SNAPSHOT_DIR = EXPORT_DIR / "snapshots"
CSV_DIR = EXPORT_DIR / "csv"

# Ensure export directories exist
for _d in (MODELS_DIR, EXPORT_DIR, SNAPSHOT_DIR, CSV_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fill status enumeration
# ---------------------------------------------------------------------------
class FillStatus(str, Enum):
    """Discrete fullness classification states."""
    EMPTY = "EMPTY"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    NEARLY_FULL = "NEARLY FULL"
    FULL = "FULL"


# ---------------------------------------------------------------------------
# Color palette  (BGR for OpenCV)
# ---------------------------------------------------------------------------
STATUS_COLORS: Dict[FillStatus, Tuple[int, int, int]] = {
    FillStatus.EMPTY:       (0, 200, 0),       # Green
    FillStatus.LOW:         (0, 220, 100),      # Light green
    FillStatus.MEDIUM:      (0, 220, 220),      # Yellow
    FillStatus.NEARLY_FULL: (0, 140, 255),      # Orange
    FillStatus.FULL:        (0, 0, 240),        # Red
}

# Hex equivalents for Streamlit UI
STATUS_COLORS_HEX: Dict[FillStatus, str] = {
    FillStatus.EMPTY:       "#00C800",
    FillStatus.LOW:         "#64DC00",
    FillStatus.MEDIUM:      "#DCDC00",
    FillStatus.NEARLY_FULL: "#FF8C00",
    FillStatus.FULL:        "#F00000",
}

# Mask overlay colors (BGR, semi-transparent)
BIN_INTERIOR_COLOR = (200, 150, 50)     # Blue tint
WASTE_MASK_COLOR   = (50, 100, 220)     # Orange-red tint


# ---------------------------------------------------------------------------
# COCO class mappings  (EXPANDED — 39 waste-relevant classes)
# ---------------------------------------------------------------------------
# COCO-80 class names mapped to waste categories.
# This mapping is comprehensive: every COCO object that could plausibly
# appear as discarded litter or waste inside a bin is included.
WASTE_COCO_MAPPING: Dict[str, str] = {
    # ---- Containers & Packaging ----
    "bottle":       "Plastic Bottle",
    "cup":          "Cup",
    "wine glass":   "Glass",
    "bowl":         "Container",
    "vase":         "Container",
    # ---- Bags & Carriers ----
    "handbag":      "Bag",
    "backpack":     "Bag",
    "suitcase":     "Bag",
    # ---- Food Waste ----
    "apple":        "Food Waste",
    "banana":       "Food Waste",
    "sandwich":     "Food Waste",
    "orange":       "Food Waste",
    "cake":         "Food Waste",
    "pizza":        "Food Waste",
    "donut":        "Food Waste",
    "hot dog":      "Food Waste",
    "carrot":       "Food Waste",
    "broccoli":     "Food Waste",
    # ---- Paper & Fabric ----
    "book":         "Paper",
    "tie":          "Fabric",
    "umbrella":     "Misc Waste",
    "kite":         "Paper",
    "teddy bear":   "Fabric",
    # ---- E-Waste & Small Items ----
    "cell phone":   "E-Waste",
    "remote":       "E-Waste",
    "keyboard":     "E-Waste",
    "mouse":        "E-Waste",
    "laptop":       "E-Waste",
    "tv":           "E-Waste",
    # ---- Miscellaneous Litter ----
    "scissors":     "Sharp Object",
    "clock":        "Misc Waste",
    "toothbrush":   "Plastic Waste",
    "hair drier":   "Misc Waste",
    "sports ball":  "Misc Waste",
    "frisbee":      "Plastic Waste",
    "tennis racket": "Misc Waste",
    "skateboard":   "Misc Waste",
    "surfboard":    "Misc Waste",
    # ---- Kitchen / Dining ----
    "knife":        "Sharp Object",
    "spoon":        "Utensil",
    "fork":         "Utensil",
}


# ---------------------------------------------------------------------------
# HSV colour ranges for common bin colours (for heuristic bin detection)
# ---------------------------------------------------------------------------
# Each entry is (name, lower_hsv, upper_hsv) in OpenCV H(0-179) S(0-255) V(0-255)
BIN_HSV_RANGES: List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]] = [
    # Green wheelie bins — wide range covering yellow-green to blue-green
    # Lower S threshold catches faded/dirty green bins
    ("green",       (35, 40, 30),   (90, 255, 220)),
    # Blue recycling bins — wide range covering royal blue to navy
    ("blue",        (90, 50, 30),   (130, 255, 255)),
    # Gray / charcoal bins — tightened to avoid matching pavement
    # Pavement is typically S<15, V=80-160; bins are slightly more saturated
    ("gray",        (0, 5, 55),     (180, 35, 120)),
    # Brown / tan / beige bins — wider range for weathered bins
    ("brown",       (8, 40, 30),    (25, 200, 180)),
    # Black bins — very low value, catches dark charcoal
    ("black",       (0, 0, 10),     (180, 80, 50)),
    # Dark teal / dark cyan (some municipal bins)
    ("teal",        (85, 40, 30),   (100, 200, 180)),
]


# ---------------------------------------------------------------------------
# Main settings dataclass
# ---------------------------------------------------------------------------
@dataclass
class Settings:
    """Centralised runtime configuration — instantiate once, pass everywhere."""

    # ---- Model Paths ----
    bin_model_path: str = ""  # Empty → use open-vocab / HSV strategies below
    waste_model_path: str = "yolo11n-seg.pt"  # Segmentation model for waste

    # ---- Open-Vocabulary Bin Detection (YOLOE) ----
    # Localises bins from text prompts with no bin-specific training. Replaces
    # the HSV colour heuristic, which cannot separate a bin from a background
    # of the same colour (hedges, grass, walls) and fails outright at night.
    openvocab_enabled: bool = True
    openvocab_model_path: str = "yoloe-11l-seg.pt"
    openvocab_prompts: Tuple[str, ...] = (
        "trash can", "garbage bin", "waste bin", "rubbish bin",
        "recycling bin", "dumpster", "wheeled garbage bin",
    )
    openvocab_conf: float = 0.15
    openvocab_nms_iou: float = 0.5
    # Two inference scales, one chosen per frame. Optimal scale depends on how
    # much of the frame a bin fills: a close-up needs a small input, a row of
    # distant bins needs a large one. See _pick_scale in bin_detector.py.
    openvocab_scales: Tuple[int, ...] = (640, 1280)
    openvocab_contain_frac: float = 0.85  # Overlap above which boxes are duplicates

    # ---- Detection Thresholds ----
    confidence_threshold: float = 0.25  # Lower default for better recall
    iou_threshold: float = 0.45
    input_size: int = 640

    # ---- Fill-State Boundaries (inclusive %) ----
    fill_thresholds: Dict[FillStatus, Tuple[int, int]] = field(
        default_factory=lambda: {
            FillStatus.EMPTY:       (0, 20),
            FillStatus.LOW:         (21, 40),
            FillStatus.MEDIUM:      (41, 60),
            FillStatus.NEARLY_FULL: (61, 80),
            FillStatus.FULL:        (81, 100),
        }
    )

    # ---- Occupancy Algorithm Weights (aperture-focused model) ----
    area_weight: float = 0.55       # Weight for aperture-focused area ratio
    height_weight: float = 0.25     # Weight for vertical fill height
    overflow_weight: float = 0.20   # Weight for overflow detection
    smoothing_window: int = 5       # Frames for rolling average
    smoothing_enabled: bool = True  # False for single-image analysis (no temporal dimension)

    # ---- Bin Detection Heuristic ----
    bin_min_height_ratio: float = 0.15     # Min bin height as fraction of frame
    bin_interior_erode_kernel: int = 15

    # ---- Waste Detection Tuning ----
    waste_detection_min_overlap: float = 0.05
    non_bin_color_enabled: bool = True  # Use non-bin-color pixel analysis
    # Fraction of bin height searched ABOVE the rim for overflowing waste.
    # Without this the waste search is confined to the bin box, and waste
    # heaped on top of a full bin is invisible to the occupancy estimator.
    waste_overflow_band_ratio: float = 0.45
    waste_overflow_edge_gate: bool = True  # Require texture in the band
    # Stricter band gate: require overflow blobs to be continuous with waste
    # visible inside the bin. Off by default — it is correct only when the bin
    # interior is visible. On a front-on view the interior is hidden, so a pile
    # resting on the rim connects to nothing and real overflow is discarded.
    waste_overflow_require_contact: bool = False

    # ---- Performance / Streaming ----
    frame_skip: int = 3            # Run full pipeline every N frames
    target_fps: int = 15
    rtsp_reconnect_interval: float = 2.0
    rtsp_reconnect_max: float = 30.0
    rtsp_buffer_size: int = 1

    # ---- Export ----
    csv_flush_interval: int = 100

    # ---- Device ----
    device: str = ""  # Empty = auto-detect (cuda > mps > cpu)

    def classify_fill(self, fill_pct: float) -> FillStatus:
        """Map a continuous fill percentage to a discrete status."""
        pct = max(0.0, min(100.0, fill_pct))
        for status, (lo, hi) in self.fill_thresholds.items():
            if lo <= pct <= hi:
                return status
        return FillStatus.FULL  # Safety fallback

    def resolve_device(self) -> str:
        """Auto-detect the best available compute device."""
        if self.device:
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
