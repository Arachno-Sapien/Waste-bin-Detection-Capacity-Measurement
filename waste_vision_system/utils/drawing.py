"""
Drawing Utilities — HUD overlays, masks, and bounding boxes
=============================================================
All rendering functions operate on BGR numpy arrays (OpenCV convention)
and mutate the frame *in place* for zero-copy performance.  Callers
should pass a copy if the original frame must be preserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    from detectors.bin_detector import BinDetection
    from detectors.waste_detector import WasteDetection
    from services.occupancy import OccupancyResult

from config.settings import (
    BIN_INTERIOR_COLOR,
    STATUS_COLORS,
    WASTE_MASK_COLOR,
    FillStatus,
)


# ---- Private helpers -------------------------------------------------------

def _color_for_status(status: FillStatus) -> Tuple[int, int, int]:
    """Return the BGR color associated with a fill status."""
    return STATUS_COLORS.get(status, (200, 200, 200))


def _draw_rounded_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = -1,
    radius: int = 12,
    alpha: float = 0.70,
) -> None:
    """Draw a rounded rectangle with optional transparency."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)

    # Four corner circles + two rectangles to approximate rounded rect
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_fill_bar(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    fill_pct: float,
    status: FillStatus,
    height: int = 14,
) -> None:
    """Draw a horizontal fill-bar indicator."""
    # Background track
    cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)

    # Filled portion
    fill_w = int(width * max(0.0, min(1.0, fill_pct / 100.0)))
    if fill_w > 0:
        color = _color_for_status(status)
        cv2.rectangle(img, (x, y + 1), (x + fill_w, y + height - 1), color, -1)


# ---- Public API -----------------------------------------------------------

def draw_bin_overlay(
    frame: np.ndarray,
    bin_det: "BinDetection",
    occ: "OccupancyResult",
    waste_dets: Optional[List["WasteDetection"]] = None,
    mask_alpha: float = 0.35,
) -> None:
    """
    Composite overlay renderer for a single bin.

    Draws:
      • Color-coded bounding box
      • Semi-transparent interior mask (blue tint)
      • Waste segmentation masks (orange tint)
      • HUD text block with fill %, status, confidence
      • Horizontal fill bar
    """
    h, w = frame.shape[:2]
    color = _color_for_status(occ.status)

    # 1. Bounding box
    x1, y1, x2, y2 = bin_det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # 2. Interior mask overlay (blue tint)
    if bin_det.interior_mask is not None:
        mask_overlay = frame.copy()
        mask_bool = bin_det.interior_mask.astype(bool)
        if mask_bool.shape[:2] == frame.shape[:2]:
            mask_overlay[mask_bool] = BIN_INTERIOR_COLOR
            cv2.addWeighted(mask_overlay, mask_alpha * 0.5, frame, 1 - mask_alpha * 0.5, 0, frame)

    # 3. Waste masks overlay
    if waste_dets:
        waste_overlay = frame.copy()
        for wd in waste_dets:
            if wd.mask is not None:
                wd_mask = wd.mask.astype(bool)
                if wd_mask.shape[:2] == frame.shape[:2]:
                    waste_overlay[wd_mask] = WASTE_MASK_COLOR
        cv2.addWeighted(waste_overlay, mask_alpha, frame, 1 - mask_alpha, 0, frame)

    # 4. HUD text block with background panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness_text = 1
    line_gap = 24

    waste_count = len(waste_dets) if waste_dets else 0
    lines = [
        f"Bin #{occ.bin_id}",
        f"Fill: {occ.fill_pct:.0f}%",
        f"Status: {occ.status.value}",
        f"Waste Items: {waste_count}",
        f"Area: {occ.area_ratio:.0f}% | Height: {occ.height_ratio:.0f}%",
    ]

    # Position: just above the bounding box
    text_x = x1
    text_y_start = max(y1 - (len(lines) * line_gap) - 30, 10)

    # Compute panel dimensions
    max_text_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness_text)
        max_text_w = max(max_text_w, tw)

    panel_w = max_text_w + 20
    panel_h = len(lines) * line_gap + 24  # extra for fill bar
    panel_x1 = text_x
    panel_y1 = text_y_start
    panel_x2 = min(text_x + panel_w, w - 2)
    panel_y2 = min(panel_y1 + panel_h, h - 2)

    _draw_rounded_rect(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (30, 30, 30), -1, 8, 0.75)

    # Render text lines
    for i, line in enumerate(lines):
        ty = panel_y1 + 20 + i * line_gap
        text_color = color if i == 2 else (255, 255, 255)  # Status line in status color
        cv2.putText(frame, line, (panel_x1 + 10, ty), font, font_scale, text_color, thickness_text, cv2.LINE_AA)

    # 5. Fill bar below text
    bar_y = panel_y2 - 18
    _draw_fill_bar(frame, panel_x1 + 8, bar_y, panel_w - 16, occ.fill_pct, occ.status)


def draw_hud_header(
    frame: np.ndarray,
    fps: float,
    active_bins: int,
    latency_ms: float = 0.0,
) -> None:
    """
    Render a system-info bar at the top-left corner.

    Shows FPS, latency, and active bin count.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    info = f"FPS: {fps:.1f}  |  Latency: {latency_ms:.0f}ms  |  Bins: {active_bins}"
    (tw, th), _ = cv2.getTextSize(info, font, 0.5, 1)

    # Semi-transparent background strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (tw + 20, th + 16), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Colour-code FPS: green ≥ 15, yellow ≥ 10, red < 10
    if fps >= 15:
        fps_color = (0, 220, 0)
    elif fps >= 10:
        fps_color = (0, 220, 220)
    else:
        fps_color = (0, 0, 240)

    cv2.putText(frame, info, (10, th + 8), font, 0.5, fps_color, 1, cv2.LINE_AA)


def draw_no_detection(frame: np.ndarray) -> None:
    """Render a 'No bins detected' message on the frame."""
    h, w = frame.shape[:2]
    text = "No bins detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
    cx = (w - tw) // 2
    cy = (h + th) // 2
    cv2.putText(frame, text, (cx, cy), font, 0.8, (100, 100, 255), 2, cv2.LINE_AA)
