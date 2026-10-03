"""
Accuracy harness. Runs the real pipeline over a labelled split and reports
bin-detection and fill metrics, overall and per condition tag.

    python evaluate.py --split val                 # tune against this
    python evaluate.py --split test --save         # final numbers -> exports/eval/*.json
    python evaluate.py --split val --detect-only   # bin detection only (fast)
    python evaluate.py --split val --cameras       # fixed-camera mode (saved ROIs + references)

Dataset layout (enhancements.md, Task A1):
    datasets/bins/images/*.jpg|png      datasets/bins/labels/<same stem>.txt
    datasets/bins/{train,val,test}.txt  "./images/<file>" per line (tools/make_splits.py)
    datasets/bins/meta.csv              image,camera_id,captured_at,tags
Label class id = fill tier: 0 EMPTY, 1 LOW, 2 MEDIUM, 3 NEARLY_FULL, 4 FULL,
5 UNKNOWN (bin present, true fill not established: counts for detection only).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import EXPORT_DIR, FillStatus, Settings  # noqa: E402

TIERS = [FillStatus.EMPTY, FillStatus.LOW, FillStatus.MEDIUM,
         FillStatus.NEARLY_FULL, FillStatus.FULL]
UNKNOWN = 5
MIDPOINTS = np.array([10, 30, 50, 70, 90])  # tier -> % for the MAE figure
DATASET = PROJECT_ROOT / "datasets" / "bins"


def read_labels(label_path: Path, w: int, h: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """YOLO label file -> [(class_id, (x1, y1, x2, y2) in pixels)].
    Accepts box lines (cls cx cy bw bh) and polygon lines (cls x1 y1 x2 y2 ...)."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        cls, nums = int(v[0]), np.array(v[1:], dtype=float)
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            xs, ys = np.array([cx - bw / 2, cx + bw / 2]), np.array([cy - bh / 2, cy + bh / 2])
        else:
            xs, ys = nums[0::2], nums[1::2]
        out.append((cls, (int(xs.min() * w), int(ys.min() * h),
                          int(xs.max() * w), int(ys.max() * h))))
    return out


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(pred: list, gt: list, thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy one-to-one matching, highest IoU first. Returns [(pred_i, gt_i)]."""
    pairs = sorted(((iou(p, g), i, j) for i, p in enumerate(pred)
                    for j, g in enumerate(gt)), reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            out.append((i, j))
    return out


def summarise(rows: list[dict]) -> dict:
    """rows: [{"n_pred", "n_gt", "matches": [(pred_tier or None, gt_class)]}].
    pred_tier None = the pipeline abstained (UNCERTAIN); counted as wrong."""
    tp = sum(len(r["matches"]) for r in rows)
    n_pred = sum(r["n_pred"] for r in rows)
    n_gt = sum(r["n_gt"] for r in rows)
    p = tp / n_pred if n_pred else 0.0
    rc = tp / n_gt if n_gt else 0.0
    pairs = [(pt, gt) for r in rows for pt, gt in r["matches"] if gt != UNKNOWN]
    cm = np.zeros((5, 6), int)  # rows = true tier, cols = predicted tier + "abstained"
    for pt, gt in pairs:
        cm[gt, 5 if pt is None else pt] += 1
    n = cm.sum()
    known = [(pt, gt) for pt, gt in pairs if pt is not None]
    full = 4

    def ratio(a, b):
        return round(float(a) / b, 3) if b else None

    return {
        "images": len(rows),
        "precision": round(p, 3),
        "recall": round(rc, 3),
        "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0,
        "count_accuracy": ratio(sum(r["n_pred"] == r["n_gt"] for r in rows), len(rows)),
        "fill_pairs": int(n),
        "tier_accuracy": ratio(np.trace(cm[:, :5]), n),
        "within_one_tier": ratio(sum(abs(pt - gt) <= 1 for pt, gt in known), n),
        "abstain_rate": ratio(cm[:, 5].sum(), n),
        "full_recall": ratio(cm[full, full], cm[full].sum()),
        "full_precision": ratio(cm[full, full], cm[:, full].sum()),
        "mae_pct": round(float(np.mean([abs(MIDPOINTS[pt] - MIDPOINTS[gt]) for pt, gt in known])), 1) if known else None,
        "confusion": cm.tolist(),
    }


def load_split(split: str) -> list[Path]:
    return [(DATASET / ln).resolve() for ln in (DATASET / f"{split}.txt").read_text().split()]


def load_meta() -> dict[str, dict]:
    meta = DATASET / "meta.csv"
    if not meta.exists():
        return {}
    with meta.open(newline="") as f:
        return {Path(r["image"]).name: r for r in csv.DictReader(f)}


def run(split: str, settings: Settings, detect_only: bool = False, cameras: bool = False) -> dict:
    """Evaluate the pipeline on one split; returns the report dict.

    detect_only: bin detection metrics only (skips waste and fill; for sweeps)
    cameras:     fixed-camera mode: use each image's saved ROIs and empty-bin
                 references (config/cameras/<camera_id>/) where they exist
    """
    from config.cameras import load_references, load_rois
    from detectors.bin_detector import BinDetector
    from detectors.waste_detector import WasteDetector
    from services.occupancy import OccupancyEstimator
    from services.pipeline import analyse_frame

    settings.smoothing_enabled = False  # every labelled image is its own scene
    bd, wd, oe = BinDetector(settings), WasteDetector(settings), OccupancyEstimator(settings)
    meta, rows = load_meta(), []
    for img_path in load_split(split):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  ! unreadable: {img_path}")
            continue
        h, w = frame.shape[:2]
        info = meta.get(img_path.name, {})
        bd.reset_tracking()
        bd.clear_manual_rois()
        refs = None
        if cameras and info.get("camera_id"):
            rois = load_rois(info["camera_id"], (w, h))
            if rois:
                bd.set_manual_rois(rois)
                refs = load_references(info["camera_id"])
        gt = read_labels(DATASET / "labels" / f"{img_path.stem}.txt", w, h)
        if detect_only:
            found = [(b.bbox, None) for b in bd.detect(frame)]
        else:
            found = [(b.bbox, occ.status) for b, occ, _ in
                     analyse_frame(frame, settings, bd, wd, oe, refs)]
        pairs = match([box for box, _ in found], [box for _, box in gt])
        rows.append({
            "image": img_path.name,
            "tags": sorted(t for t in info.get("tags", "").split(";") if t),
            "n_pred": len(found),
            "n_gt": len(gt),
            "matches": [(TIERS.index(found[i][1]) if found[i][1] in TIERS else None, gt[j][0])
                        for i, j in pairs],
        })
    report = {"split": split, "overall": summarise(rows), "by_tag": {}}
    for tag in sorted({t for r in rows for t in r["tags"]}):
        report["by_tag"][tag] = summarise([r for r in rows if tag in r["tags"]])
    report["per_image"] = rows
    return report


def print_report(rep: dict) -> None:
    def line(name, m):
        print(f"  {name:<12} n={m['images']:<4} P={m['precision']:.2f} R={m['recall']:.2f} "
              f"F1={m['f1']:.2f} count={m['count_accuracy']} | tier={m['tier_accuracy']} "
              f"+-1={m['within_one_tier']} FULL R/P={m['full_recall']}/{m['full_precision']} "
              f"MAE={m['mae_pct']} abstain={m['abstain_rate']}")
    print(f"\n=== {rep['split']} ===")
    line("overall", rep["overall"])
    for tag, m in rep["by_tag"].items():
        line(tag, m)
    print("  confusion (rows = true EMPTY..FULL, cols = predicted EMPTY..FULL, abstained):")
    for row in rep["overall"]["confusion"]:
        print("   ", row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--detect-only", action="store_true", help="bin detection metrics only")
    ap.add_argument("--cameras", action="store_true", help="use saved per-camera ROIs and references")
    ap.add_argument("--save", action="store_true", help="write exports/eval/<split>_<timestamp>.json")
    args = ap.parse_args()
    settings = Settings()
    rep = run(args.split, settings, detect_only=args.detect_only, cameras=args.cameras)
    print_report(rep)
    if args.save:
        out = EXPORT_DIR / "eval" / f"{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"settings": vars(settings), **rep}, indent=1, default=str))
        print(f"  saved {out}")
