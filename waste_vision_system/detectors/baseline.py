"""
Empty-bin reference differencing.

With a photo of each bin taken while it was empty, "waste" becomes "pixels
that changed since then" instead of "pixels that aren't the bin's colour".
Mud, stickers, the hedge behind the bin and the bin's own shadow appear in
both images and cancel out.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from config.settings import Settings


def pick_reference(refs: List[np.ndarray], crop: np.ndarray) -> np.ndarray:
    """Choose the reference shot under the most similar lighting (day, dusk,
    IR night) by comparing mean brightness and mean saturation."""
    def stats(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return np.array([hsv[..., 2].mean(), hsv[..., 1].mean()])
    target = stats(crop)
    return min(refs, key=lambda r: float(np.abs(stats(r) - target).sum()))


def _align(ref_gray: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Undo small camera shake (translation only). Returns cur warped onto ref."""
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            ref_gray.astype(np.float32), cur_gray.astype(np.float32), warp,
            cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4), None, 5)
    except cv2.error:
        return cur  # did not converge: compare unaligned rather than fail
    h, w = ref_gray.shape
    return cv2.warpAffine(cur, warp, (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def change_mask(ref: np.ndarray, cur: np.ndarray, settings: Settings) -> np.ndarray:
    """uint8 0/1 mask (cur's size) of pixels that differ from the empty reference."""
    h, w = cur.shape[:2]
    if ref.shape[:2] != (h, w):
        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
    cur = _align(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cur)

    ref_lab = cv2.cvtColor(cv2.GaussianBlur(ref, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    cur_lab = cv2.cvtColor(cv2.GaussianBlur(cur, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    # Global lighting change: scale cur's brightness to ref's. Medians, so a
    # large new object in the crop does not skew the correction.
    cur_lab[..., 0] *= np.median(ref_lab[..., 0]) / max(float(np.median(cur_lab[..., 0])), 1.0)

    d = cur_lab - ref_lab
    d[..., 0] *= 0.5  # brightness differs more than colour from shadows alone
    mask = (np.sqrt((d ** 2).sum(axis=2)) > settings.baseline_diff_thresh).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = settings.baseline_min_blob_frac * h * w
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(labels, keep).astype(np.uint8)
