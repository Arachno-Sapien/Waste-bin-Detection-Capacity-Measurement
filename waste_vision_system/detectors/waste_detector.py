"""
Waste Detector — Detect visible waste inside bin interiors
============================================================
Uses a dual approach:

1. **YOLO11 Segmentation**: Runs yolo11n-seg on cropped bin regions to get
   per-pixel segmentation masks for known waste types (bottles, cups, food,
   bags, etc.).  39 COCO classes are mapped to waste categories.

2. **Non-bin-colour pixel analysis**: Within each bin bounding box, identifies
   pixels that are NOT the bin's own colour (e.g. not green for green bins).
   These non-bin pixels inside the bin region are treated as waste.  This
   catches ALL types of waste that YOLO cannot detect: wrappers, tissue,
   mixed garbage, overflowing trash, cardboard, plastic bags, etc.

The two masks are combined (union) for comprehensive coverage.

The search region also extends above the rim by ``waste_overflow_band_ratio``
(see Settings) so waste heaped on top of a full bin — outside the bin box
itself — is not invisible to detection. A texture gate keeps flat background
above a bin (wall, sky) from being mistaken for overflow; see
``_gate_band_by_texture`` / ``_keep_band_joined_to_bin``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import Settings, WASTE_COCO_MAPPING

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class WasteDetection:
    """Single detected waste item within a bin region."""
    class_name: str                         # Human-readable category
    bbox: Tuple[int, int, int, int]         # (x1, y1, x2, y2) in full-frame coords
    mask: Optional[np.ndarray]              # Full-frame binary mask (uint8 0/1)
    confidence: float


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class WasteDetector:
    """
    Detect and classify waste objects inside bin interior regions.
    Combines YOLO segmentation with non-bin-colour pixel analysis.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._model_loaded = False
        self._is_custom_model = False
        self._class_names: Dict[int, str] = {}
        self._has_seg_masks = False

    # ---- Lazy model loading ------------------------------------------------

    def load_model(self) -> None:
        """Load the waste detection YOLO model."""
        if self._model_loaded:
            return
        try:
            from ultralytics import YOLO

            model_path = self._settings.waste_model_path
            logger.info("Loading waste detection model: %s", model_path)
            self._model = YOLO(model_path)

            # Discover class names
            if hasattr(self._model, "names"):
                self._class_names = dict(self._model.names)
            else:
                self._class_names = {}

            # Check if segmentation model
            task = getattr(self._model, "task", None)
            self._has_seg_masks = (task == "segment" or "-seg" in str(model_path))
            if self._has_seg_masks:
                logger.info("Waste model supports instance segmentation")

            # Detect custom vs COCO
            coco_indicative = {"person", "car", "dog", "cat", "chair"}
            model_classes = set(self._class_names.values())
            if coco_indicative & model_classes:
                self._is_custom_model = False
                logger.info("Waste model is COCO-pretrained; using expanded mapping (%d classes)",
                            len(WASTE_COCO_MAPPING))
            else:
                self._is_custom_model = True
                logger.info("Waste model is custom-trained with classes: %s",
                            list(model_classes)[:10])

            self._model_loaded = True
        except Exception as e:
            logger.error("Failed to load waste model: %s", e)
            self._model = None
            self._model_loaded = True

    # ---- Non-bin-colour pixel analysis ------------------------------------

    def _non_bin_color_mask(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        bin_color_name: str,
        bin_hsv_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
        interior_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Within the bin bounding box, find pixels that are NOT the bin's own
        colour.  These non-bin pixels are treated as waste.

        This is the key innovation: it catches ALL types of waste (wrappers,
        tissue, mixed garbage, overflowing trash) without needing a model
        trained on those specific waste types.
        """
        h, w = frame.shape[:2]
        result_mask = np.zeros((h, w), dtype=np.uint8)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return result_mask

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Build the "bin colour" mask within the crop
        bin_color_mask = np.zeros(crop.shape[:2], dtype=np.uint8)

        if bin_hsv_range is not None:
            lower, upper = bin_hsv_range
            # Expand the range slightly for robustness
            lower_np = np.array([
                max(0, lower[0] - 10),
                max(0, lower[1] - 20),
                max(0, lower[2] - 30),
            ], dtype=np.uint8)
            upper_np = np.array([
                min(179, upper[0] + 10),
                min(255, upper[1] + 20),
                min(255, upper[2] + 40),
            ], dtype=np.uint8)
            bin_color_mask = cv2.inRange(hsv_crop, lower_np, upper_np)
        else:
            # Fallback: try all known bin colours
            from config.settings import BIN_HSV_RANGES
            for _, lower, upper in BIN_HSV_RANGES:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                m = cv2.inRange(hsv_crop, lower_np, upper_np)
                bin_color_mask = cv2.bitwise_or(bin_color_mask, m)

        # Also exclude very dark pixels (shadows) and very bright white
        # (reflections) to reduce noise
        v_channel = hsv_crop[:, :, 2]
        shadow_mask = v_channel < 20
        highlight_mask = (hsv_crop[:, :, 1] < 15) & (v_channel > 230)
        exclude_mask = shadow_mask | highlight_mask

        # Non-bin pixels = NOT bin colour AND NOT shadow/highlight
        non_bin = cv2.bitwise_not(bin_color_mask)
        non_bin[exclude_mask] = 0

        # Morphological cleanup to remove speckle noise
        k = max(3, min(crop.shape[:2]) // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        non_bin = cv2.morphologyEx(non_bin, cv2.MORPH_OPEN, kernel, iterations=1)
        non_bin = cv2.morphologyEx(non_bin, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Place into full-frame mask
        result_mask[y1:y2, x1:x2] = (non_bin > 0).astype(np.uint8)

        # Clip to interior mask if provided
        if interior_mask is not None:
            result_mask = np.logical_and(
                result_mask.astype(bool), interior_mask.astype(bool)
            ).astype(np.uint8)

        return result_mask

    # ---- Public API --------------------------------------------------------

    @staticmethod
    def _gate_band_by_texture(
        frame: np.ndarray,
        mask: np.ndarray,
        x1: int, by1: int, x2: int, by2: int,
    ) -> np.ndarray:
        """
        Keep only textured pixels within the overflow band.

        Heaped waste is visually busy and high-contrast. A wall, sky or stretch
        of pavement above a bin is smooth, but it is also "not the bin's
        colour", so without this gate the colour-inversion analysis would count
        all of it as waste and report permanent overflow.
        """
        if by2 <= by1:
            return mask
        crop = frame[by1:by2, x1:x2]
        if crop.size == 0:
            return mask
        edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 50, 150)
        k = max(3, min(crop.shape[:2]) // 25)
        edges = cv2.dilate(
            edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        )
        out = mask.copy()
        out[by1:by2, x1:x2] = np.logical_and(
            out[by1:by2, x1:x2].astype(bool), edges.astype(bool)
        ).astype(np.uint8)
        return out

    @staticmethod
    def _keep_band_joined_to_bin(
        mask: np.ndarray,
        x1: int, band_y1: int, x2: int, bin_top: int, y2: int,
    ) -> np.ndarray:
        """
        Drop overflow-band blobs that are not continuous with the bin's contents.

        Waste heaped on a full bin rests on the rim, so its blob reaches down
        into the bin box. A hedge, tree or wall behind the bin sits entirely
        within the band, separated from the contents by the bin's own rim.
        Texture alone cannot tell these apart — foliage is highly textured —
        but connectivity can.
        """
        if bin_top <= band_y1:
            return mask
        sub = mask[band_y1:y2, x1:x2].astype(np.uint8)
        if not sub.any():
            return mask
        n_labels, labels = cv2.connectedComponents(sub, connectivity=8)
        if n_labels <= 1:
            return mask
        split = bin_top - band_y1
        keep = np.unique(labels[split:, :])
        keep = keep[keep > 0]
        out = mask.copy()
        out[band_y1:y2, x1:x2] = np.isin(labels, keep).astype(np.uint8) * sub
        return out

    def detect(
        self,
        frame: np.ndarray,
        bin_bbox: Tuple[int, int, int, int],
        interior_mask: Optional[np.ndarray] = None,
        bin_color_name: str = "unknown",
        bin_hsv_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None,
    ) -> List[WasteDetection]:
        """
        Detect waste items within a bin region.

        Combines YOLO detections with non-bin-colour pixel analysis for
        comprehensive waste coverage.
        """
        self.load_model()

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bin_bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # ---- Overflow band ----
        # Waste heaped above the rim lies outside the bin box entirely, so a
        # search confined to that box can never see it — and the overflow
        # factor in OccupancyEstimator, which looks for waste above the rim,
        # can never fire. Extend the search region upward. Band pixels are
        # texture-gated below so flat background (wall, sky, pavement) above a
        # bin is not mistaken for waste.
        bin_top = y1
        band_h = int((y2 - y1) * self._settings.waste_overflow_band_ratio)
        band_y1 = max(0, y1 - band_h)
        if band_y1 < bin_top:
            if interior_mask is not None:
                interior_mask = interior_mask.copy().astype(np.uint8)
                interior_mask[band_y1:bin_top, x1:x2] = 1
            y1 = band_y1

        detections: List[WasteDetection] = []
        yolo_union_mask = np.zeros((h, w), dtype=np.uint8)

        # ---- Part 1: YOLO inference on bin crop ----
        if self._model is not None:
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_h, crop_w = crop.shape[:2]
                device = self._settings.resolve_device()

                results = self._model(
                    crop,
                    conf=self._settings.confidence_threshold,
                    iou=self._settings.iou_threshold,
                    imgsz=self._settings.input_size,
                    device=device,
                    verbose=False,
                )

                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes

                    if boxes is not None and len(boxes) > 0:
                        for i in range(len(boxes)):
                            cls_id = int(boxes.cls[i].cpu().numpy())
                            conf = float(boxes.conf[i].cpu().numpy())

                            raw_name = self._class_names.get(cls_id, f"class_{cls_id}")
                            if self._is_custom_model:
                                class_name = raw_name
                            else:
                                class_name = WASTE_COCO_MAPPING.get(raw_name, None)
                                if class_name is None:
                                    continue  # Skip non-waste COCO classes

                            # Map crop coords to full-frame
                            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                            fx1 = int(xyxy[0]) + x1
                            fy1 = int(xyxy[1]) + y1
                            fx2 = int(xyxy[2]) + x1
                            fy2 = int(xyxy[3]) + y1

                            # Build mask
                            full_mask = np.zeros((h, w), dtype=np.uint8)
                            if (self._has_seg_masks and result.masks is not None
                                    and i < len(result.masks)):
                                raw_mask = result.masks[i].data.cpu().numpy().squeeze()
                                crop_mask = cv2.resize(
                                    raw_mask, (crop_w, crop_h),
                                    interpolation=cv2.INTER_NEAREST
                                )
                                crop_mask = (crop_mask > 0.5).astype(np.uint8)
                                full_mask[y1:y2, x1:x2] = crop_mask
                            else:
                                full_mask[fy1:fy2, fx1:fx2] = 1

                            # Filter by interior overlap
                            if interior_mask is not None:
                                overlap = np.logical_and(
                                    full_mask.astype(bool),
                                    interior_mask.astype(bool)
                                )
                                overlap_ratio = overlap.sum() / max(full_mask.sum(), 1)
                                if overlap_ratio < self._settings.waste_detection_min_overlap:
                                    continue
                                full_mask = overlap.astype(np.uint8)

                            yolo_union_mask = np.logical_or(
                                yolo_union_mask.astype(bool),
                                full_mask.astype(bool)
                            ).astype(np.uint8)

                            detections.append(WasteDetection(
                                class_name=class_name,
                                bbox=(fx1, fy1, fx2, fy2),
                                mask=full_mask,
                                confidence=conf,
                            ))

        # ---- Part 2: Non-bin-colour pixel analysis ----
        if self._settings.non_bin_color_enabled:
            color_mask = self._non_bin_color_mask(
                frame, x1, y1, x2, y2,
                bin_color_name, bin_hsv_range, interior_mask,
            )

            if band_y1 < bin_top and self._settings.waste_overflow_edge_gate:
                color_mask = self._gate_band_by_texture(
                    frame, color_mask, x1, band_y1, x2, bin_top
                )
            if band_y1 < bin_top and self._settings.waste_overflow_require_contact:
                color_mask = self._keep_band_joined_to_bin(
                    color_mask, x1, band_y1, x2, bin_top, y2
                )

            # Remove pixels already covered by YOLO detections
            color_only = np.logical_and(
                color_mask.astype(bool),
                ~yolo_union_mask.astype(bool)
            ).astype(np.uint8)

            pixel_count = int(color_only.sum())
            if pixel_count > 50:  # At least some meaningful area
                detections.append(WasteDetection(
                    class_name="Mixed Waste",
                    bbox=(x1, y1, x2, y2),
                    mask=color_only,
                    confidence=0.6,
                ))
                logger.debug(
                    "Non-bin-colour analysis found %d waste pixels", pixel_count
                )

        return detections
