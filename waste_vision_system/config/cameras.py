"""
Per-camera configuration for fixed CCTV, stored under config/cameras/<camera_id>/:

    rois.json             {"frame_size": [w, h], "rois": [[x1, y1, x2, y2], ...]}
    empty_<i>_<tag>.png   ROI i's search region photographed while empty;
                          tag = day / night / ir ... (one per lighting condition)

Camera ids and tags become path components, so both are limited to
[A-Za-z0-9_-]. Never store the RTSP URI here: it usually carries a password.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

CAMERAS_DIR = Path(__file__).resolve().parent / "cameras"
_SAFE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _checked(name: str) -> str:
    if not _SAFE.fullmatch(name or ""):
        raise ValueError(f"must match {_SAFE.pattern}: {name!r}")
    return name


def camera_dir(camera_id: str) -> Path:
    return CAMERAS_DIR / _checked(camera_id)


def save_rois(camera_id: str, rois: List[Tuple[int, int, int, int]],
              frame_size: Tuple[int, int]) -> Path:
    d = camera_dir(camera_id)
    d.mkdir(parents=True, exist_ok=True)
    out = d / "rois.json"
    out.write_text(json.dumps({"frame_size": list(frame_size),
                               "rois": [[int(v) for v in r] for r in rois]}, indent=1))
    return out


def load_rois(camera_id: str, frame_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    """Saved ROIs scaled to frame_size (w, h), in case the stream resolution changed."""
    f = camera_dir(camera_id) / "rois.json"
    if not f.exists():
        return []
    data = json.loads(f.read_text())
    sx = frame_size[0] / data["frame_size"][0]
    sy = frame_size[1] / data["frame_size"][1]
    return [(round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))
            for x1, y1, x2, y2 in data["rois"]]


def save_reference(camera_id: str, roi_index: int, crop: np.ndarray, tag: str) -> Path:
    d = camera_dir(camera_id)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"empty_{int(roi_index)}_{_checked(tag)}.png"
    cv2.imwrite(str(out), crop)
    return out


def load_references(camera_id: str) -> Dict[int, List[np.ndarray]]:
    refs: Dict[int, List[np.ndarray]] = {}
    for p in sorted(camera_dir(camera_id).glob("empty_*_*.png")):
        img = cv2.imread(str(p))
        if img is not None:
            refs.setdefault(int(p.stem.split("_")[1]), []).append(img)
    return refs
