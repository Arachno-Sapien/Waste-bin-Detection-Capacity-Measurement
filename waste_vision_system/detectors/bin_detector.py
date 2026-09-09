"""
Bin Detector — Locate bins and segment their interior
=======================================================
Detection strategies, tried in this order in ``detect()``:

1. **Manual ROIs**: explicit operator-drawn boxes. Always wins when set —
   the deliberate override for a camera angle nothing else handles.

2. **Custom YOLO-Seg model**: used if ``bin_model_path`` points to a model
   fine-tuned on a "bin" class. Highest accuracy once one exists; see
   Settings.bin_model_path.

3. **Open-vocabulary (YOLOE)**: locates bins from text prompts
   (see Settings.openvocab_*) with no bin-specific training. Runs at two
   input scales and keeps whichever produces the cleaner box set — see
   ``_detect_openvocab`` / ``_messiness``. Default strategy today.

4. **HSV colour segmentation**: legacy heuristic, matches bins by body
   colour. Kept only as a fallback if the open-vocab pass finds nothing
   (or is disabled); it cannot separate a bin from same-coloured background
   (hedges, walls) and fails outright at night.

A lightweight IoU-based tracker assigns persistent ``Bin #N`` IDs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import BIN_HSV_RANGES, Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BinDetection:
    """Result for a single detected bin in a frame."""
    bin_id: int
    bbox: Tuple[int, int, int, int]       # (x1, y1, x2, y2)
    interior_mask: Optional[np.ndarray]    # Full-frame binary mask (uint8 0/1)
    confidence: float
    rim_top_y: int = 0                     # Y coordinate of the bin rim top
    rim_bottom_y: int = 0                  # Y coordinate of the bin bottom
    bin_color_name: str = "unknown"        # Detected colour (for waste detector)
    bin_hsv_range: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    surface_mask: Optional[np.ndarray] = None  # Instance mask of the bin's own
                                               # surface (walls/shell), when the
                                               # detector produces one. Occupancy
                                               # uses it as the wall mask.


# ---------------------------------------------------------------------------
# Simple IoU tracker
# ---------------------------------------------------------------------------

class _SimpleTracker:
    """Greedy IoU-based tracker for maintaining bin IDs across frames."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self._next_id: int = 1
        self._tracks: Dict[int, Tuple[int, int, int, int]] = {}
        self._iou_thresh = iou_threshold

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def update(self, bboxes: List[Tuple[int, int, int, int]]) -> List[int]:
        """Match new bboxes to existing tracks; return ordered IDs."""
        ids: List[int] = []
        used_tracks = set()

        for bbox in bboxes:
            best_id = -1
            best_iou = self._iou_thresh
            for tid, tbox in self._tracks.items():
                if tid in used_tracks:
                    continue
                iou_val = self._iou(bbox, tbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_id = tid

            if best_id > 0:
                ids.append(best_id)
                used_tracks.add(best_id)
                self._tracks[best_id] = bbox
            else:
                ids.append(self._next_id)
                self._tracks[self._next_id] = bbox
                self._next_id += 1

        active_ids = set(ids)
        self._tracks = {k: v for k, v in self._tracks.items() if k in active_ids}
        return ids

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class BinDetector:
    """
    Locate trash bins in a frame and segment their interior opening.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._tracker = _SimpleTracker()
        self._model_loaded = False
        self._ov_model = None
        self._ov_loaded = False
        self._use_segmentation = False
        self._manual_rois: List[Tuple[int, int, int, int]] = []

    # ---- Lazy model loading ------------------------------------------------

    def load_model(self) -> None:
        """Load the YOLO segmentation model for bin detection."""
        if self._model_loaded:
            return

        model_path = self._settings.bin_model_path
        if not model_path:
            logger.info("No bin model specified; using HSV colour segmentation")
            self._model_loaded = True
            return

        try:
            from ultralytics import YOLO

            logger.info("Loading bin detection model: %s", model_path)
            self._model = YOLO(model_path)

            task = getattr(self._model, "task", None)
            if task == "segment" or "-seg" in str(model_path):
                self._use_segmentation = True
                logger.info("Bin model supports instance segmentation")
            else:
                logger.info("Bin model is detection-only")

            self._model_loaded = True
        except Exception as e:
            logger.error("Failed to load bin model: %s", e)
            self._model = None
            self._model_loaded = True

    # ---- Public API --------------------------------------------------------

    def set_manual_rois(self, rois: List[Tuple[int, int, int, int]]) -> None:
        """Set manual ROI bounding boxes (x1, y1, x2, y2)."""
        self._manual_rois = [
            (max(0, x1), max(0, y1), max(x1 + 1, x2), max(y1 + 1, y2))
            for (x1, y1, x2, y2) in rois
        ]

    def clear_manual_rois(self) -> None:
        """Reset to automatic AI detection mode."""
        self._manual_rois = []

    def detect(self, frame: np.ndarray) -> List[BinDetection]:
        """Run bin detection on a frame."""
        self.load_model()
        h, w = frame.shape[:2]

        # Manual ROIs are an explicit operator override and outrank every
        # automatic strategy — including a configured bin model.
        if self._manual_rois:
            return self._detect_manual_rois(frame, h, w)
        if self._model is not None:
            return self._detect_yolo(frame, h, w)
        if self._settings.openvocab_enabled:
            dets = self._detect_openvocab(frame, h, w)
            if dets:
                return dets
            # Fall through to HSV only if the open-vocab pass found nothing.
        return self._detect_hsv_color(frame, h, w)

    # ---- Open-vocabulary strategy (YOLOE, text-prompted) --------------------

    @staticmethod
    def _frac_inside(a, b) -> float:
        """Fraction of box b's area lying inside box a."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / max((b[2] - b[0]) * (b[3] - b[1]), 1)

    @classmethod
    def _overlap(cls, a, b) -> float:
        """Max fraction of either box covered by the other."""
        return max(cls._frac_inside(a, b), cls._frac_inside(b, a))

    def _dedupe_boxes(self, items):
        """
        Greedy by confidence, dropping any box that heavily overlaps one already
        kept. Removes the two duplicate patterns YOLOE produces: the same bin
        matched by several prompt terms, and the same bin detected once with its
        open lid and once without.
        """
        frac = self._settings.openvocab_contain_frac
        kept = []
        for it in sorted(items, key=lambda t: -t[0]):
            if any(self._overlap(it[1], k[1]) >= frac for k in kept):
                continue
            kept.append(it)
        return kept

    def _messiness(self, kept) -> float:
        """
        Max mutual overlap among a scale's surviving boxes. A scale that cannot
        resolve the scene emits several partly-overlapping boxes for one object;
        a scale that resolves it emits cleanly separated ones. Low is better.
        """
        if len(kept) < 2:
            return 0.0
        return max(self._overlap(a[1], b[1])
                   for i, a in enumerate(kept) for b in kept[i + 1:])

    def _detect_openvocab(self, frame: np.ndarray, h: int, w: int) -> List[BinDetection]:
        """
        Detect bins by text prompt, with no bin-specific training.

        Runs each configured input scale, keeps the cleanest result (see
        _messiness), and turns the instance masks into BinDetection objects.
        The segmentation mask is the bin's own surface, which downstream
        occupancy uses directly as the wall mask — far more reliable than
        matching wall colour against a fixed HSV range.
        """
        model = self._load_openvocab()
        if model is None:
            return []

        candidates = []
        for imgsz in self._settings.openvocab_scales:
            try:
                res = model.predict(
                    frame,
                    conf=self._settings.openvocab_conf,
                    imgsz=imgsz,
                    iou=self._settings.openvocab_nms_iou,
                    agnostic_nms=True,
                    device=self._settings.resolve_device(),
                    verbose=False,
                )[0]
            except Exception as e:
                logger.error("Open-vocab inference failed at imgsz=%d: %s", imgsz, e)
                continue

            items = []
            n = 0 if res.boxes is None else len(res.boxes)
            for i in range(n):
                xyxy = res.boxes.xyxy[i].cpu().numpy().astype(int)
                box = (max(0, int(xyxy[0])), max(0, int(xyxy[1])),
                       min(w, int(xyxy[2])), min(h, int(xyxy[3])))
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    continue
                mask = None
                if res.masks is not None and i < len(res.masks):
                    raw = res.masks[i].data.cpu().numpy().squeeze()
                    mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.uint8)
                items.append((float(res.boxes.conf[i]), box, mask))

            kept = self._dedupe_boxes(items)
            candidates.append((self._messiness(kept), -len(kept), kept))

        if not candidates:
            return []

        # Cleanest scale wins; near-ties break toward the one finding more bins.
        candidates.sort(key=lambda c: (round(c[0], 2), c[1]))
        kept = candidates[0][2]
        if not kept:
            return []

        kept.sort(key=lambda t: t[1][0])  # left-to-right for stable IDs
        bboxes = [it[1] for it in kept]
        ids = self._tracker.update(bboxes)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_masks = []
        for name, lower, upper in BIN_HSV_RANGES:
            m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            color_masks.append((name, lower, upper, m))

        detections: List[BinDetection] = []
        for idx, (conf, (bx1, by1, bx2, by2), mask) in enumerate(kept):
            interior = np.zeros((h, w), dtype=np.uint8)
            interior[by1:by2, bx1:bx2] = 1

            # Rim: top of the detected bin surface, so overflow is measured
            # against the real bin edge rather than the top of the bbox.
            rim_top, rim_bot = by1, by2
            if mask is not None and mask.any():
                rows = np.nonzero(np.any(mask > 0, axis=1))[0]
                if len(rows):
                    rim_top, rim_bot = int(rows.min()), int(rows.max()) + 1

            color_name, hsv_range = self._dominant_bin_color(
                hsv, color_masks, bx1, by1, bx2, by2
            )

            detections.append(BinDetection(
                bin_id=ids[idx],
                bbox=(bx1, by1, bx2, by2),
                interior_mask=interior,
                confidence=round(conf, 2),
                rim_top_y=rim_top,
                rim_bottom_y=rim_bot,
                bin_color_name=color_name,
                bin_hsv_range=hsv_range,
                surface_mask=mask,
            ))
        return detections

    def _load_openvocab(self):
        """Lazily load and prompt the open-vocabulary model."""
        if self._ov_loaded:
            return self._ov_model
        self._ov_loaded = True
        try:
            from ultralytics import YOLOE

            prompts = list(self._settings.openvocab_prompts)
            logger.info("Loading open-vocab bin model: %s",
                        self._settings.openvocab_model_path)
            model = YOLOE(self._settings.openvocab_model_path)
            model.set_classes(prompts, model.get_text_pe(prompts))
            self._ov_model = model
        except Exception as e:
            logger.error("Open-vocab model unavailable (%s); falling back to HSV", e)
            self._ov_model = None
        return self._ov_model

    # ---- YOLO strategy (unchanged) ----------------------------------------

    def _detect_yolo(self, frame: np.ndarray, h: int, w: int) -> List[BinDetection]:
        """Use YOLO to detect bins and extract interior masks."""
        device = self._settings.resolve_device()
        results = self._model(
            frame,
            conf=self._settings.confidence_threshold,
            iou=self._settings.iou_threshold,
            imgsz=self._settings.input_size,
            device=device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        detections: List[BinDetection] = []
        bboxes: List[Tuple[int, int, int, int]] = []
        confidences: List[float] = []
        masks_list: List[Optional[np.ndarray]] = []

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            conf = float(boxes.conf[i].cpu().numpy())

            seg_mask = None
            if self._use_segmentation and result.masks is not None and i < len(result.masks):
                raw_mask = result.masks[i].data.cpu().numpy().squeeze()
                seg_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                seg_mask = (seg_mask > 0.5).astype(np.uint8)
                k = self._settings.bin_interior_erode_kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                seg_mask = cv2.erode(seg_mask, kernel, iterations=1)

            if seg_mask is None:
                seg_mask = self._make_rect_mask(h, w, x1, y1, x2, y2)

            bboxes.append((x1, y1, x2, y2))
            confidences.append(conf)
            masks_list.append(seg_mask)

        ids = self._tracker.update(bboxes)

        # The model localises the bin, but downstream stages still need its
        # wall colour: WasteDetector's non-bin-colour analysis and the
        # aperture-focused area ratio both key off bin_hsv_range. Without it
        # they silently drop to the amplified fallback ratio.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_masks = []
        for name, lower, upper in BIN_HSV_RANGES:
            m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            color_masks.append((name, lower, upper, m))

        for idx, (bbox, conf, mask) in enumerate(zip(bboxes, confidences, masks_list)):
            x1, y1, x2, y2 = bbox
            color_name, hsv_range = self._dominant_bin_color(
                hsv, color_masks, x1, y1, x2, y2
            )
            detections.append(BinDetection(
                bin_id=ids[idx], bbox=bbox, interior_mask=mask,
                confidence=conf, rim_top_y=y1, rim_bottom_y=y2,
                bin_color_name=color_name, bin_hsv_range=hsv_range,
            ))
        return detections

    # ---- Manual ROI strategy -----------------------------------------------

    def _detect_manual_rois(self, frame: np.ndarray, h: int, w: int) -> List[BinDetection]:
        """
        Generate BinDetection objects from user-defined manual ROIs.
        Extracts color and rim information from the actual frame within
        the manual boundaries so waste detection and occupancy work correctly.
        """
        if not self._manual_rois:
            return []

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_masks = []
        for name, lower, upper in BIN_HSV_RANGES:
            m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            color_masks.append((name, lower, upper, m))

        bboxes = [
            (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
            for (x1, y1, x2, y2) in self._manual_rois
        ]
        ids = self._tracker.update(bboxes)
        detections: List[BinDetection] = []

        for idx, (bx1, by1, bx2, by2) in enumerate(bboxes):
            # Full interior rectangle mask
            interior = np.zeros((h, w), dtype=np.uint8)
            interior[by1:by2, bx1:bx2] = 1

            # Dominant color within user ROI
            color_name, hsv_range = self._dominant_bin_color(
                hsv, color_masks, bx1, by1, bx2, by2
            )

            detections.append(BinDetection(
                bin_id=ids[idx],
                bbox=(bx1, by1, bx2, by2),
                interior_mask=interior,
                confidence=1.0,
                rim_top_y=by1,
                rim_bottom_y=by2,
                bin_color_name=color_name,
                bin_hsv_range=hsv_range,
            ))

        return detections

    # ---- HSV colour segmentation strategy (ROBUST) --------------------------

    def _detect_hsv_color(self, frame: np.ndarray, h: int, w: int) -> List[BinDetection]:
        """
        Detect bins using HSV colour segmentation with robust handling of:
        - Slatted/mesh bins (bridged via heavy morphological closing + convex hull)
        - Open lids (detected and excluded from bbox)
        - Adjacent same-colour bins (conservative splitting)
        - Overflowing trash (doesn't break bin detection)

        Algorithm:
        1. Multi-colour HSV masking.
        2. Heavy morphological closing + convex hull to bridge gaps (slats, dirt).
        3. Connected component analysis with area filtering.
        4. Conservative splitting for truly wide blobs (gap ≥15% of width).
        5. Lid detection: trim bbox if an open lid is detected above the body.
        6. Build per-bin BinDetection objects.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 1. Multi-colour union mask ---
        raw_mask = np.zeros((h, w), dtype=np.uint8)
        color_masks = []
        for name, lower, upper in BIN_HSV_RANGES:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            m = cv2.inRange(hsv, lower_np, upper_np)
            color_masks.append((name, lower, upper, m))
            raw_mask = cv2.bitwise_or(raw_mask, m)

        # --- 2. Robust morphological processing ---
        # Step A: Light open to remove speckle noise
        k_open = max(3, min(h, w) // 100)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # Step B: Prepare closing kernel for per-component use later
        k_close = max(5, min(h, w) // 40)
        if k_close % 2 == 0:
            k_close += 1
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))

        # Check if there's enough bin-coloured area
        total_bin_pixels = int(np.sum(clean_mask > 0))
        min_total = int(h * w * 0.02)  # At least 2% of frame
        if total_bin_pixels < min_total:
            return []

        # --- 3. Connected component analysis on CLEAN mask ---
        # Using the clean mask (pre-closing) preserves natural gaps between
        # adjacent same-colour bins. Closing is applied per-component later.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            clean_mask, connectivity=8
        )

        # Filter components by minimum area and aspect ratio
        min_area = int(h * w * 0.01)  # At least 1% of frame
        min_height_px = int(h * self._settings.bin_min_height_ratio)

        candidate_bins = []
        for label_id in range(1, num_labels):  # Skip background (0)
            cx1 = stats[label_id, cv2.CC_STAT_LEFT]
            cy1 = stats[label_id, cv2.CC_STAT_TOP]
            cw = stats[label_id, cv2.CC_STAT_WIDTH]
            ch = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]
            cx2 = cx1 + cw
            cy2 = cy1 + ch

            if area < min_area:
                continue
            if ch < min_height_px:
                continue

            # Create mask for this component and apply per-component closing
            comp_mask = (labels == label_id).astype(np.uint8) * 255
            comp_closed = cv2.morphologyEx(comp_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

            # Per-component convex hull for fragmented/slatted shapes
            comp_contours, _ = cv2.findContours(comp_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in comp_contours:
                cnt_area = cv2.contourArea(cnt)
                if cnt_area < 200:
                    continue
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0 and hull_area / max(cnt_area, 1) > 1.3:
                    cv2.fillConvexPoly(comp_closed, hull, 255)

            # Recompute bounds from the processed mask
            nz_rows, nz_cols = np.nonzero(comp_closed)
            if len(nz_rows) == 0:
                continue
            cx1_new = int(nz_cols.min())
            cy1_new = int(nz_rows.min())
            cx2_new = int(nz_cols.max()) + 1
            cy2_new = int(nz_rows.max()) + 1

            if (cy2_new - cy1_new) < min_height_px:
                continue

            candidate_bins.append((cx1_new, cy1_new, cx2_new, cy2_new, comp_closed))

        if not candidate_bins:
            return []

        # --- 4. Conservative splitting for wide blobs ---
        # Only split if a blob is wide enough to contain 2+ bins
        # AND there's a clear vertical gap in the RAW (unclosed) mask
        final_bins = []
        for bx1, by1, bx2, by2, comp_mask in candidate_bins:
            blob_w = bx2 - bx1
            blob_h = by2 - by1
            expected_single_bin_w = int(h * 0.35)  # Rough estimate

            if blob_w > expected_single_bin_w * 1.8:
                # Potentially multiple bins — check raw mask for gaps
                sub_bins = self._conservative_split(
                    clean_mask, comp_mask, bx1, by1, bx2, by2, h, w
                )
                final_bins.extend(sub_bins)
            else:
                final_bins.append((bx1, by1, bx2, by2, comp_mask))

        if not final_bins:
            return []

        # --- 5. Lid detection & bbox trimming ---
        trimmed_bins = []
        for bx1, by1, bx2, by2, mask in final_bins:
            by1_trimmed, by2_trimmed = self._trim_open_lid(
                clean_mask, bx1, by1, bx2, by2
            )
            trimmed_bins.append((bx1, by1_trimmed, bx2, by2_trimmed, mask))

        # --- 6. Merge adjacent thin segments (prevent over-splitting) ---
        trimmed_bins = self._merge_adjacent_bin_segments(trimmed_bins, h, w)

        # --- 7. Build BinDetection objects ---
        bboxes = [(b[0], b[1], b[2], b[3]) for b in trimmed_bins]
        sorted_indices = sorted(range(len(bboxes)), key=lambda i: bboxes[i][0])
        bboxes = [bboxes[i] for i in sorted_indices]
        trimmed_bins = [trimmed_bins[i] for i in sorted_indices]

        ids = self._tracker.update(bboxes)

        detections: List[BinDetection] = []
        for idx, (bx1, by1, bx2, by2, mask_full) in enumerate(trimmed_bins):
            # Interior mask = full bounding box rectangle
            interior = np.zeros((h, w), dtype=np.uint8)
            interior[by1:by2, bx1:bx2] = 1

            # Colour mask (for confidence and rim detection)
            color_region = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
            raw_crop = raw_mask[by1:by2, bx1:bx2]
            if raw_crop.shape == color_region.shape:
                color_region = (raw_crop > 0).astype(np.uint8)

            # Dominant colour
            color_name, hsv_range = self._dominant_bin_color(
                hsv, color_masks, bx1, by1, bx2, by2
            )

            # Confidence: based on how much of the bbox is bin-coloured
            bbox_area = max((bx2 - bx1) * (by2 - by1), 1)
            bin_pixel_count = int(color_region.sum())
            conf = min(1.0, bin_pixel_count / (bbox_area * 0.3))

            # Rim detection: use the colour mask to find where bin body starts/ends
            rim_top = by1
            rim_bot = by2
            rows_with_px = np.any(color_region > 0, axis=1)
            if rows_with_px.any():
                rim_top = by1 + int(np.argmax(rows_with_px))
                rim_bot = by1 + int(len(rows_with_px) - 1 - np.argmax(rows_with_px[::-1]))

            detections.append(BinDetection(
                bin_id=ids[idx],
                bbox=(bx1, by1, bx2, by2),
                interior_mask=interior,
                confidence=round(conf, 2),
                rim_top_y=rim_top,
                rim_bottom_y=rim_bot,
                bin_color_name=color_name,
                bin_hsv_range=hsv_range,
            ))

        return detections

    # ---- Conservative splitting (replaces fragile vertical profile) ---------

    def _conservative_split(
        self,
        raw_mask: np.ndarray,
        comp_mask: np.ndarray,
        bx1: int, by1: int, bx2: int, by2: int,
        frame_h: int, frame_w: int,
    ) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """
        Split a wide blob into individual bins ONLY if there's a clear,
        persistent vertical gap in the RAW (unclosed) mask.

        A gap must be:
        - At least 15% of blob width or 30px (whichever is larger)
        - The column density in the gap must drop below 20% of peak

        This prevents splitting when overflowing trash merely interrupts
        the color profile.
        """
        blob_w = bx2 - bx1
        crop_raw = raw_mask[by1:by2, bx1:bx2]
        if crop_raw.size == 0:
            return [(bx1, by1, bx2, by2, comp_mask)]

        # Column projection on the RAW mask
        col_sum = np.sum(crop_raw > 0, axis=0).astype(float)
        if col_sum.max() == 0:
            return [(bx1, by1, bx2, by2, comp_mask)]

        # Smooth
        ks = max(3, blob_w // 30)
        if ks % 2 == 0:
            ks += 1
        smoothed = cv2.GaussianBlur(col_sum.reshape(1, -1), (ks, 1), 0).flatten()

        peak = smoothed.max()
        gap_thresh = peak * 0.35  # Gap = columns below 35% of peak density

        # Find segments above threshold
        above = smoothed > gap_thresh
        segments = []
        in_seg = False
        seg_start = 0
        for i in range(len(above)):
            if above[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not above[i] and in_seg:
                segments.append((seg_start, i))
                in_seg = False
        if in_seg:
            segments.append((seg_start, len(above)))

        # Filter tiny segments
        min_seg_w = max(30, blob_w // 8)
        segments = [(s, e) for s, e in segments if (e - s) >= min_seg_w]

        if not segments:
            return [(bx1, by1, bx2, by2, comp_mask)]

        # Judge gap width against a single bin's width, not the whole blob:
        # the seam between two bins is roughly constant, while blob_w grows
        # with every extra bin in the row. Scaling by blob_w made a 4-bin row
        # demand a wider seam than any real one.
        seg_ref = int(np.median([e - s for s, e in segments]))
        min_gap_w = max(6, int(seg_ref * 0.05))

        # Merge segments separated by narrow gaps. Noise in the colour mask
        # (foliage, reflections) fragments a single bin into several segments;
        # only gaps wide enough to be real inter-bin space become split points.
        merged_segments = [segments[0]]
        for seg in segments[1:]:
            if (seg[0] - merged_segments[-1][1]) < min_gap_w:
                merged_segments[-1] = (merged_segments[-1][0], seg[1])
            else:
                merged_segments.append(seg)

        # Fewer than 2 groups means there's no real separation — keep the blob
        if len(merged_segments) < 2:
            return [(bx1, by1, bx2, by2, comp_mask)]

        # Build sub-bins from the merged groups
        sub_bins = []
        for seg_start_col, seg_end_col in merged_segments:
            sx1 = bx1 + seg_start_col
            sx2 = bx1 + seg_end_col

            # Find vertical extent
            sub_crop = comp_mask[by1:by2, sx1:sx2]
            rows_with_px = np.any(sub_crop > 0, axis=1)
            if not rows_with_px.any():
                continue

            first_row = int(np.argmax(rows_with_px))
            last_row = int(len(rows_with_px) - 1 - np.argmax(rows_with_px[::-1]))
            sy1 = by1 + first_row
            sy2 = by1 + last_row + 1

            sub_mask = np.zeros_like(comp_mask)
            sub_mask[sy1:sy2, sx1:sx2] = comp_mask[sy1:sy2, sx1:sx2]
            sub_bins.append((sx1, sy1, sx2, sy2, sub_mask))

        return sub_bins if sub_bins else [(bx1, by1, bx2, by2, comp_mask)]

    # ---- Lid detection & bbox trimming -------------------------------------

    def _trim_open_lid(
        self,
        raw_mask: np.ndarray,
        bx1: int, by1: int, bx2: int, by2: int,
    ) -> Tuple[int, int]:
        """
        Detect if a bin has an open lid flipped above the body.

        Open lids create a pattern in the vertical density profile:
          - HIGH density band (lid) at the top
          - LOW density band (gap/opening) in the middle
          - HIGH density band (body) at the bottom

        If this pattern is found, trim by1 to start at the body (below the gap),
        so the lid doesn't inflate the bounding box.

        Returns (new_by1, new_by2).
        """
        crop = raw_mask[by1:by2, bx1:bx2]
        if crop.size == 0:
            return by1, by2

        blob_h = by2 - by1
        if blob_h < 50:
            return by1, by2

        # Row-wise density
        row_density = np.sum(crop > 0, axis=1).astype(float)
        if row_density.max() == 0:
            return by1, by2

        # Smooth
        ks = max(3, blob_h // 20)
        if ks % 2 == 0:
            ks += 1
        smoothed = cv2.GaussianBlur(row_density.reshape(-1, 1), (1, ks), 0).flatten()

        peak = smoothed.max()
        low_thresh = peak * 0.25

        # Find the first significant "dip" (gap) from the top
        # Look in the top 60% of the blob for a gap
        search_limit = int(blob_h * 0.60)

        in_high = False
        gap_start = -1
        gap_end = -1
        for r in range(search_limit):
            if smoothed[r] > low_thresh:
                if not in_high:
                    in_high = True
            elif in_high:
                # Found transition from high to low — potential gap start
                gap_start = r
                break

        if gap_start > 0:
            # Find where density rises again (body starts)
            for r in range(gap_start, search_limit):
                if smoothed[r] > low_thresh:
                    gap_end = r
                    break

            if gap_end > 0:
                gap_height = gap_end - gap_start
                # Gap must be at least 5% of blob height to be a real opening
                if gap_height >= blob_h * 0.05:
                    # Trim: body starts at gap_end
                    new_by1 = by1 + gap_end
                    return new_by1, by2

        return by1, by2

    # ---- Merge adjacent bin segments ----------------------------------------

    def _merge_adjacent_bin_segments(
        self,
        individual_bins: List[Tuple[int, int, int, int, np.ndarray]],
        frame_h: int,
        frame_w: int,
    ) -> List[Tuple[int, int, int, int, np.ndarray]]:
        """
        Merge adjacent split segments if they belong to the same single bin.

        Criteria for merging two adjacent segments (A and B):
        1. Horizontal gap between A and B is very small (< 4% of frame width).
        2. Vertical overlap between A and B is high (> 76% IoU on Y-axis).
        3. One segment is a sliver next to the other (much narrower than its
           neighbour), or the two are all but touching.

        Note: "sliver" is judged relative to the neighbouring segment, not by
        absolute aspect ratio — a wheelie bin seen front-on is legitimately
        tall and narrow, and an aspect test would merge real neighbours.
        """
        if len(individual_bins) <= 1:
            return individual_bins

        # Sort segments left-to-right
        sorted_bins = sorted(individual_bins, key=lambda b: b[0])
        merged: List[Tuple[int, int, int, int, np.ndarray]] = []

        curr_x1, curr_y1, curr_x2, curr_y2, curr_mask = sorted_bins[0]

        for i in range(1, len(sorted_bins)):
            next_x1, next_y1, next_x2, next_y2, next_mask = sorted_bins[i]

            gap = next_x1 - curr_x2
            max_gap_allowed = int(frame_w * 0.04)

            # Check vertical alignment overlap
            y_overlap_top = max(curr_y1, next_y1)
            y_overlap_bot = min(curr_y2, next_y2)
            y_overlap_h = max(0, y_overlap_bot - y_overlap_top)

            curr_h = max(1, curr_y2 - curr_y1)
            next_h = max(1, next_y2 - next_y1)
            min_h = min(curr_h, next_h)
            y_overlap_ratio = y_overlap_h / min_h

            curr_w = curr_x2 - curr_x1
            next_w = next_x2 - next_x1

            # A spurious split leaves one piece much narrower than the other;
            # two real bins side by side have comparable widths.
            width_ratio = min(curr_w, next_w) / max(curr_w, next_w, 1)

            # Merge condition
            should_merge = (
                gap <= max_gap_allowed and
                y_overlap_ratio >= 0.70 and
                (width_ratio < 0.40 or gap <= 5)
            )

            if should_merge:
                # Combine bounding boxes
                curr_x1 = min(curr_x1, next_x1)
                curr_y1 = min(curr_y1, next_y1)
                curr_x2 = max(curr_x2, next_x2)
                curr_y2 = max(curr_y2, next_y2)
                curr_mask = cv2.bitwise_or(curr_mask, next_mask)
            else:
                merged.append((curr_x1, curr_y1, curr_x2, curr_y2, curr_mask))
                curr_x1, curr_y1, curr_x2, curr_y2, curr_mask = next_x1, next_y1, next_x2, next_y2, next_mask

        merged.append((curr_x1, curr_y1, curr_x2, curr_y2, curr_mask))
        return merged

    # ---- Helper: dominant bin colour for a region --------------------------

    def _dominant_bin_color(
        self,
        hsv_frame: np.ndarray,
        color_masks: List[Tuple[str, Tuple, Tuple, np.ndarray]],
        x1: int, y1: int, x2: int, y2: int,
    ) -> Tuple[str, Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]]:
        """Find which HSV colour range dominates in a given bounding box."""
        best_name = "unknown"
        best_count = 0
        best_range = None

        for name, lower, upper, mask in color_masks:
            roi = mask[y1:y2, x1:x2]
            count = int(np.sum(roi > 0))
            if count > best_count:
                best_count = count
                best_name = name
                best_range = (lower, upper)

        return best_name, best_range

    # ---- Helper: simple rectangular mask -----------------------------------

    @staticmethod
    def _make_rect_mask(
        h: int, w: int, x1: int, y1: int, x2: int, y2: int,
    ) -> np.ndarray:
        """Create a rectangular binary mask within a bounding box."""
        mask = np.zeros((h, w), dtype=np.uint8)
        # Inset slightly to approximate interior
        pad_x = max(1, (x2 - x1) // 10)
        pad_y = max(1, (y2 - y1) // 10)
        mask[y1 + pad_y:y2 - pad_y, x1 + pad_x:x2 - pad_x] = 1
        return mask

    def reset_tracking(self) -> None:
        """Reset all track IDs."""
        self._tracker.reset()
