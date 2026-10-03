# Accuracy Improvements: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise bin-detection and fill-estimation accuracy from "right on stock photos" to measured, dependable numbers on real CCTV footage.

**Architecture:** First build a labelled dataset and an evaluation harness, so every later change is judged by numbers. Then add improvements that need no training and use what fixed cameras offer: saved bin positions, empty-bin reference photos, and fill that only rises between collections. Once labelled data exists, train a bin segmentation model and a fill classifier. The current geometric estimator stays as the fallback until a learned model beats it on held-out data.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, pandas, Ultralytics (YOLO11, YOLOE), Streamlit. No new pip dependencies until Phase D.

**Spec:** This document. The current pipeline is described in `README.md` §3 to §5.

## Global Constraints

- Python 3.11+, `ultralytics>=8.4.0`, `streamlit>=1.50.0`. Everything must keep running on a CPU-only machine; training may use a GPU.
- No new pip dependencies in Phases A to C.
- Every tunable lives in `Settings` (`config/settings.py`). Each task lists the fields it adds.
- `BinDetection.interior_mask`, `surface_mask` and all waste masks are full-frame `uint8` masks with values 0/1.
- Manual / saved ROIs always outrank automatic bin detection.
- `verify_system.py` stays runnable with no arguments and keeps its exit codes (0 pass, 1 fail, 2 skipped). Tests that need no model must stay fast.
- Tune on the `val` split only. Run on `test` once per decision, never in a loop.
- `fill_mode = "geometric"` stays the default until the learned path beats it on `test`.
- Never write an RTSP URI to a config file; it usually contains a password.
- CCTV frames show people. Keep raw images out of git, label them with a self-hosted tool, and check local privacy law before sharing a dataset.

---

## 1. Where accuracy stands

Bin detection already works on the 5 stock fixtures (exact bin count on all 5). Fill estimation is the weak half: going by hand estimates, 3 of 6 readings are off by a tier or more. Diagnostics run during the October 2026 review traced that to these causes:

| Cause | Evidence |
|---|---|
| "Any pixel not the bin's colour is waste" | Mud, labels, shadows and background count as waste. On one fixture bin, 71% of the counted waste pixels sat on the bin's own wall. |
| Area-ratio numerator and denominator don't match | The denominator is the area around the bin silhouette; the numerator counts waste anywhere in the box. The ratio saturates at 100. |
| Overflow false alarms | Foliage behind a bin passes the texture gate (`timg (2).jpg` bin 3 reads 70%). |
| Geometry assumes an upright bin seen from the front | Tipped or top-down bins break the vertical-height factor. |
| Small COCO waste model | Bags, wrappers and mixed rubbish are not COCO classes, so the colour heuristic carries most of the load. |
| Front-on cameras | The camera can't see into the bin unless it overflows. |
| No ground truth | 5 stock photos, no labels, no metric. A change can't be shown to help. |

## 2. How the plan is organised

Phases are gated by data, not by code. Every task whose code needs no data can be built now and checked with the model-free tests; its benefit is confirmed once labelled data arrives.

| Task | What | Code now? | Needs labelled data to... | Status |
|---|---|---|---|---|
| A1 | Collect and label data | no code | (this produces it) | Step 1 done |
| A2 | Shared pipeline + `evaluate.py` | yes | produce real numbers | done |
| A3 | Leak-free splits | yes | run | done |
| A4 | Record the baseline | no code | run | to do |
| B1 | Camera placement guide | no code | compare views by tag | to do |
| B2 | Tracker grace period | yes | (none) | done |
| B3 | Fill only rises between collections | yes | tune two thresholds | done |
| B4 | Empty-bin reference differencing | yes | confirm the gain | done |
| B5 | Fixed-camera mode (saved ROIs + references) | yes | confirm the gain | done |
| B6 | Tune the zero-shot detector, visual prompts, YOLOE-26 | yes | pick settings | to do |
| B7 | Area-ratio numerator experiment | yes | decide keep or drop | to do |
| C1 | Detector training data (site + public) | small | train | to do |
| C2 | Fine-tune the bin segmentation model | wiring yes | train | to do |
| C3 | Learned fill classifier | wiring yes | train | to do |
| C4 | Review loop for hard cases | no code | run | to do |
| D1 | Monocular depth experiment | experiment | run | to do |
| D2 | Ultrasonic sensors as ground truth | small | run | to do |

Suggested order: A1 (start collecting immediately, it takes the longest) in parallel with A2, A3, B2, B3, B4, B5. Then A4, B6, B7. Then C1 to C4 as labels accumulate. D only if C3 plateaus.

**Progress (2026-10-03).** A2, A3, B2, B3, B4 and B5 are implemented in the repo, one task at a time, each test-first. `python verify_system.py` passes 13 of 13 after every task, and tests 1 to 8 print the same values throughout. Also checked by hand: the headless CLI on `tests/fixtures/001.jpg` (4 bins, 85% FULL), `setup_camera.py` then `main.py --camera` on a fixture, `evaluate.py --cameras` (a camera with 2 of its 4 bins saved reports 2 bins), and the dashboard's Camera ID box in webcam mode. Not yet run: the dashboard's Load / Save / Capture buttons (B5 Step 11b, needs a live camera) and every step that needs labelled data or recordings. Commit steps are left unticked for whoever commits. Next: A1, then A4, B6, B7.

**Verification record.** The end state of every code change in Phases A to C (48 edits to existing files, 11 new files) was applied to a copy of this codebase on 2026-10-02 and run there. All 13 tests passed (the 8 existing plus 5 new), and tests 1 to 8 printed the same values as before. On a 5-image seed set these ran end to end: `evaluate.py` in all three modes, `make_splits.py`, `export_crops.py` (including its refusal path), `setup_camera.py`, `main.py --camera`, a smoke-trained fill classifier, a visual-prompt model, and a sweep scoring call. Not executed: the intermediate states between tasks, the Streamlit UI changes (compile-checked only), training on real data (none exists), YOLOE-26 (needs a download), and the Phase D depth sketch.

## 3. File map

| File | Status | Responsibility |
|---|---|---|
| `waste_vision_system/services/pipeline.py` | done (A2) | `analyse_frame()`: the one per-frame code path used by app, CLI and evaluation |
| `waste_vision_system/evaluate.py` | done (A2) | Runs the pipeline over a labelled split; prints and saves metrics |
| `waste_vision_system/tools/make_splits.py` | done (A3) | Camera-day grouped train/val/test lists; single-class detector dataset |
| `waste_vision_system/services/temporal.py` | done (B3) | `FillTimeline`: monotonic fill with collection detection |
| `waste_vision_system/detectors/baseline.py` | done (B4) | `change_mask()` against an empty-bin reference; `pick_reference()` |
| `waste_vision_system/config/cameras.py` | done (B5) | Per-camera ROIs and references on disk |
| `waste_vision_system/tools/setup_camera.py` | done (B5) | CLI to save ROIs and capture references |
| `waste_vision_system/tools/sweep_openvocab.py` | to do (B6) | Staged parameter sweep for the zero-shot detector |
| `waste_vision_system/tools/make_vp_model.py` | to do (B6) | Builds a YOLOE model from visual prompts |
| `waste_vision_system/services/fill_model.py` | to do (C3) | `FillClassifier`: bin crop to fill % and confidence |
| `waste_vision_system/tools/export_crops.py` | to do (C3) | Bin crops into an ImageFolder tree for classifier training |
| `waste_vision_system/config/settings.py` | modify | New settings per task; `FillStatus.UNCERTAIN` (C3) |
| `waste_vision_system/detectors/bin_detector.py` | modify | Tracker grace (B2), `roi_index` (B5), visual-prompt loading (B6), custom-model masks (C2) |
| `waste_vision_system/detectors/waste_detector.py` | modify | Person coverage (B3), `search_region()` and reference input (B4) |
| `waste_vision_system/services/occupancy.py` | modify | Occlusion and monotonic smoothing (B3), area-ratio mode (B7), `from_classifier()` (C3) |
| `waste_vision_system/main.py`, `ui/app.py` | modify | Use `analyse_frame()` (A2); fixed-camera mode (B5); classifier (C3) |
| `waste_vision_system/verify_system.py` | modify | Tests 9 to 13 |
| `waste_vision_system/datasets/` | to do (A1) | Labelled data; images git-ignored |

Line numbers in A2 to B5 are as of commit `bb9c2a2`; from B6 on they match the code with A2 to B5 applied. Each edit names the text it replaces, so it still applies after earlier tasks shift the lines. Commands run from `waste_vision_system/`.

Single tests that need no model run like this:

```bash
python -c "import verify_system as v; assert v.test_eval_metrics() is True"
```

---

## Phase A: Measure

### Task A1: Collect and label data

No code. This is the long pole; start it first.

**Files:**
- Create: `waste_vision_system/datasets/bins/images/`, `labels/`, `meta.csv`, `LABELLING.md`
- Modify: `.gitignore`

- [x] **Step 1: Keep images out of git.** Append to the repo-root `.gitignore`:

```gitignore
# datasets: labels and meta are tracked, images are not (size, privacy)
waste_vision_system/datasets/**/images/
waste_vision_system/datasets/bins_det/
waste_vision_system/datasets/fill/
waste_vision_system/runs/
```

- [ ] **Step 2: Write `datasets/bins/LABELLING.md` before labelling anything.** Inconsistent labels cap accuracy and can only be fixed by relabelling. It must settle these rules (proposed content):

```markdown
# Labelling guide

## Boxes
- One box (or polygon) per bin, around the bin body from rim to base, wheels included.
- An open lid raised above the rim is NOT part of the box. Waste piled above the rim is NOT part of the box.
- Label a bin cut off by the frame edge if at least half of it is visible.
- Label a bin hidden by a person or vehicle if at least half is visible, and tag the image `occluded`.

## Class = true fill tier of that bin
0 EMPTY        0-20 %   bottom visible or a few items
1 LOW         21-40 %
2 MEDIUM      41-60 %
3 NEARLY_FULL 61-80 %
4 FULL        81-100 %  includes overflowing, or a lid that cannot close
5 UNKNOWN     bin present, true fill could not be established

## Where the fill tier comes from (best first)
1. Someone looked into the bin when the frame was captured (note the time).
2. A phone photo of the bin's opening taken within 2 minutes of the frame.
3. An ultrasonic sensor reading within 60 s (Task D2).
4. The frame itself, ONLY if the inside of the bin is visible in it.
Otherwise use 5 UNKNOWN. Never guess fill from the outside of a front-on bin:
the model would learn to guess the same way.

## Tags (meta.csv, separated by ;)
day, dusk, night, ir, rain, glare, occluded, front_on, angled, top_down, public

## Agreement check
A second person labels a random 10 % of images. If fewer than 80 % of bins get
the same tier, clarify this guide with examples and relabel those images.
```

- [ ] **Step 3: Collect frames.** Target 300+ frames (150 is the minimum for a usable val and test). Priority: the actual deployment cameras, then the 5 existing fixtures, then public data (C1, detector only). Per camera, sample one frame every 30 minutes over at least 7 days, so every time of day, weather and collection cycle appears. Keep frames from just before and just after collections; they give the EMPTY and FULL examples, the rarest and most important tiers. Get at least 30 frames of `night`/`ir` and 30 of the most common bad weather.

- [ ] **Step 4: Label.** Use CVAT or Label Studio, self-hosted on your machine. Export in YOLO format: one `labels/<image stem>.txt` per image, lines `class cx cy w h` (normalised) or `class x1 y1 x2 y2 ...` (polygon). Boxes are enough for Phases A and B; C2 needs polygons. To turn boxes into polygons later, use SAM through Ultralytics:

```python
from ultralytics.data.converter import yolo_bbox2segment
yolo_bbox2segment(im_dir="datasets/bins_det/images", save_dir="datasets/bins_det/labels-segment", sam_model="sam_b.pt")
```

- [ ] **Step 5: Write `datasets/bins/meta.csv`,** one row per image:

```csv
image,camera_id,captured_at,tags
gate_north_20261003_0930.jpg,gate_north,2026-10-03T09:30:00,day;front_on
gate_north_20261003_2130.jpg,gate_north,2026-10-03T21:30:00,night;ir;front_on
```

- [ ] **Step 6: Commit labels, meta and the guide** (images stay local).

```bash
git add .gitignore waste_vision_system/datasets/bins/labels waste_vision_system/datasets/bins/meta.csv waste_vision_system/datasets/bins/LABELLING.md
git commit -m "data: labelling guide and first labelled frames"
```

---

### Task A2: Shared pipeline and evaluation harness

Today the per-frame pipeline is copied in `ui/app.py`, `main.py` and `verify_system.py`. One copy drifting is exactly how an earlier bug made the tests measure a different path from the app. This task moves the app and CLI onto one function and builds `evaluate.py` on top of it.

**Files:**
- Create: `waste_vision_system/services/pipeline.py`, `waste_vision_system/evaluate.py`
- Modify: `waste_vision_system/ui/app.py:30` (import) and `:134-209` (`process_frame`), `waste_vision_system/main.py:33` (import) and `:95-126` (loop), `waste_vision_system/verify_system.py` (Test 9)

**Interfaces:**
- Produces: `analyse_frame(frame, settings, bin_detector, waste_detector, estimator) -> list[(BinDetection, OccupancyResult, list[WasteDetection])]`. `settings` is unused until B3; it is in the signature now so callers don't change again.
- Produces in `evaluate.py`: `TIERS`, `DATASET`, `read_labels(path, w, h)`, `iou(a, b)`, `match(pred, gt, thr=0.5) -> [(pred_i, gt_i)]`, `summarise(rows) -> dict`, `load_split(split) -> [Path]`, `load_meta() -> {name: row}`, `run(split, settings, detect_only=False) -> report dict`.

- [x] **Step 1: Write the failing test.** In `verify_system.py`, insert above `def test_additional_fixtures():`

```python
def test_eval_metrics():
    """
    Test 9: evaluate.py box matching and metrics (no model needed)
    """
    print("\n--- Running Test 9: Evaluation Metrics ---")
    from evaluate import match, summarise
    gt = [(0, 0, 10, 10), (20, 0, 30, 10)]
    pred = [(21, 0, 31, 10), (100, 100, 110, 110), (0, 0, 9, 10)]
    assert sorted(match(pred, gt)) == [(0, 1), (2, 0)], match(pred, gt)

    # (predicted tier or None = abstained, true class); 4 = FULL
    rows = [{"n_pred": 3, "n_gt": 2, "matches": [(4, 4), (2, 1)]},
            {"n_pred": 1, "n_gt": 1, "matches": [(None, 4)]}]
    m = summarise(rows)
    assert (m["precision"], m["recall"], m["count_accuracy"]) == (0.75, 1.0, 0.5), m
    assert (m["tier_accuracy"], m["within_one_tier"], m["abstain_rate"]) == (0.333, 0.667, 0.333), m
    assert (m["full_recall"], m["mae_pct"]) == (0.5, 10.0), m
    print("Test 9 PASSED")
    return True
```

and add `("Evaluation Metrics", test_eval_metrics),` as the last entry of `TESTS`.

- [x] **Step 2: Run it and confirm it fails.**

Run: `python -c "import verify_system as v; v.test_eval_metrics()"`
Expected: `ModuleNotFoundError: No module named 'evaluate'`

- [x] **Step 3: Create `services/pipeline.py`.**

```python
"""
Shared per-frame pipeline: the one code path the app, the headless CLI and
evaluate.py all run, so measured accuracy is the accuracy the app delivers.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from config.settings import Settings
from detectors.bin_detector import BinDetection, BinDetector
from detectors.waste_detector import WasteDetection, WasteDetector
from services.occupancy import OccupancyEstimator, OccupancyResult, wall_mask_for


def analyse_frame(
    frame: np.ndarray,
    settings: Settings,
    bin_detector: BinDetector,
    waste_detector: WasteDetector,
    estimator: OccupancyEstimator,
) -> List[Tuple[BinDetection, OccupancyResult, List[WasteDetection]]]:
    """Detect bins, then waste and occupancy per bin."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    results = []
    for b in bin_detector.detect(frame):
        wastes = waste_detector.detect(
            frame, b.bbox, b.interior_mask,
            bin_color_name=b.bin_color_name, bin_hsv_range=b.bin_hsv_range,
        )
        occ = estimator.estimate(
            b.bin_id, b.interior_mask, [w.mask for w in wastes],
            b.rim_top_y, b.rim_bottom_y, bin_color_mask=wall_mask_for(b, hsv),
        )
        results.append((b, occ, wastes))
    return results
```

- [x] **Step 4: Create `evaluate.py`.**

```python
"""
Accuracy harness. Runs the real pipeline over a labelled split and reports
bin-detection and fill metrics, overall and per condition tag.

    python evaluate.py --split val                 # tune against this
    python evaluate.py --split test --save         # final numbers -> exports/eval/*.json
    python evaluate.py --split val --detect-only   # bin detection only (fast)

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


def run(split: str, settings: Settings, detect_only: bool = False) -> dict:
    """Evaluate the pipeline on one split; returns the report dict.

    detect_only: bin detection metrics only (skips waste and fill; for sweeps)
    """
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
        gt = read_labels(DATASET / "labels" / f"{img_path.stem}.txt", w, h)
        if detect_only:
            found = [(b.bbox, None) for b in bd.detect(frame)]
        else:
            found = [(b.bbox, occ.status) for b, occ, _ in
                     analyse_frame(frame, settings, bd, wd, oe)]
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
    ap.add_argument("--save", action="store_true", help="write exports/eval/<split>_<timestamp>.json")
    args = ap.parse_args()
    settings = Settings()
    rep = run(args.split, settings, detect_only=args.detect_only)
    print_report(rep)
    if args.save:
        out = EXPORT_DIR / "eval" / f"{args.split}_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"settings": vars(settings), **rep}, indent=1, default=str))
        print(f"  saved {out}")
```

A predicted bin counts as correct when it overlaps a labelled bin with IoU of at least 0.5. This scores the whole pipeline (scale choice, de-duplication, ROIs), not just the raw model, which is why it doesn't use Ultralytics' `val()`. Tier MAE uses tier midpoints (10/30/50/70/90), so it is in percentage points but coarse.

- [x] **Step 5: Move `main.py` onto the shared pipeline.** In `_run_headless`, replace `from services.occupancy import OccupancyEstimator, wall_mask_for` with:

```python
    from services.occupancy import OccupancyEstimator
    from services.pipeline import analyse_frame
```

and replace everything from `# Pipeline` through `draw_hud_header(annotated, fps_cnt.fps, len(bin_dets), fps_cnt.latency_ms)` with:

```python
            # Pipeline
            results = analyse_frame(frame, settings, bin_detector, waste_detector, occ_estimator)
            if not results:
                draw_no_detection(annotated)
            for bd, occ, waste_dets in results:
                draw_bin_overlay(annotated, bd, occ, waste_dets)
                logger.log_entry(occ.bin_id, occ.fill_pct, occ.status, bd.confidence)
                print(f"  Frame {frame_count}: Bin #{occ.bin_id} → "
                      f"{occ.fill_pct:.0f}% ({occ.status.value})  "
                      f"[{fps_cnt.fps:.1f} FPS]", end="\r")

            draw_hud_header(annotated, fps_cnt.fps, len(results), fps_cnt.latency_ms)
```

- [x] **Step 6: Move `ui/app.py` onto the shared pipeline.** Replace `from services.occupancy import OccupancyEstimator, wall_mask_for` with:

```python
from services.occupancy import OccupancyEstimator
from services.pipeline import analyse_frame
```

and replace the body of `process_frame`, from `fps_counter.tick()` through `st.session_state["last_frame"] = annotated`, with:

```python
    fps_counter.tick()
    annotated = frame.copy()

    results_list = analyse_frame(
        frame, settings, bin_detector, waste_detector, occupancy_estimator,
    )
    if not results_list:
        draw_no_detection(annotated)
    for bin_det, occ, waste_dets in results_list:
        draw_bin_overlay(annotated, bin_det, occ, waste_dets)
        data_logger.log_entry(
            bin_id=occ.bin_id,
            fill_pct=occ.fill_pct,
            status=occ.status,
            confidence=bin_det.confidence,
        )

    draw_hud_header(
        annotated,
        fps=fps_counter.fps,
        active_bins=len(results_list),
        latency_ms=fps_counter.latency_ms,
    )

    st.session_state["last_frame"] = annotated
```

(The existing `st.session_state["last_results"] = results_list` and `return annotated` lines stay.)

- [x] **Step 7: Run Test 9 and the full suite.**

Run: `python -c "import verify_system as v; assert v.test_eval_metrics() is True"`, then `python verify_system.py`
Expected: `Test 9 PASSED`; the full suite reports `ALL 9 VERIFICATION TESTS PASSED`, with tests 1 to 8 printing the same fill values as before.

- [x] **Step 8: Check the app by hand.** `python main.py`, upload `tests/fixtures/001.jpg`: 4 bins, 85% FULL each, as before. (Checked through the headless CLI, which runs the same `analyse_frame()`.)

- [ ] **Step 9: Commit.**

```bash
git add waste_vision_system/services/pipeline.py waste_vision_system/evaluate.py waste_vision_system/main.py waste_vision_system/ui/app.py waste_vision_system/verify_system.py
git commit -m "feat: shared analyse_frame pipeline and evaluate.py accuracy harness"
```

---

### Task A3: Leak-free splits

Consecutive frames from one camera are near-duplicates. If they land in both train and test, every metric looks better than it is. Splits are therefore assigned per camera per day, deterministically.

**Files:**
- Create: `waste_vision_system/tools/make_splits.py`

**Interfaces:**
- Consumes: `datasets/bins/meta.csv`, `datasets/bins/labels/`
- Produces: `datasets/bins/{train,val,test}.txt`; `datasets/bins_det/` with `images/` (hard links), `labels/` (all classes rewritten to 0), its own split files and a `data.yaml` (`names: {0: trash can}`); `split_of(camera_id, captured_at) -> "train" | "val" | "test"`.

- [x] **Step 1: Create `tools/make_splits.py`.**

```python
"""
Assign labelled images to train / val / test by (camera, day), never by frame.
Frames from one camera on one day are near-duplicates; letting them straddle
train and test inflates every metric.

Writes datasets/bins/{train,val,test}.txt, plus a single-class copy for
detector training at datasets/bins_det/ (images hard-linked, labels rewritten
to class 0, data.yaml generated).

    python tools/make_splits.py
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINS = ROOT / "datasets" / "bins"
DET = ROOT / "datasets" / "bins_det"
SPLITS = (("train", 70), ("val", 85), ("test", 100))  # cumulative percent


def split_of(camera_id: str, captured_at: str) -> str:
    key = f"{camera_id}|{captured_at[:10]}"  # one group per camera per day
    bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return next(name for name, upto in SPLITS if bucket < upto)


def main() -> None:
    with (BINS / "meta.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    files = {name: [] for name, _ in SPLITS}
    groups = {name: set() for name, _ in SPLITS}
    for r in rows:
        name = split_of(r["camera_id"], r["captured_at"])
        files[name].append(Path(r["image"]).name)
        groups[name].add((r["camera_id"], r["captured_at"][:10]))

    for sub in ("images", "labels"):
        (DET / sub).mkdir(parents=True, exist_ok=True)
    for name, names in files.items():
        listing = "".join(f"./images/{n}\n" for n in sorted(names))
        (BINS / f"{name}.txt").write_text(listing)
        (DET / f"{name}.txt").write_text(listing)
        for n in names:
            dst = DET / "images" / n
            if not dst.exists():
                try:
                    os.link(BINS / "images" / n, dst)  # no second copy on disk
                except OSError:
                    shutil.copy2(BINS / "images" / n, dst)
            label = BINS / "labels" / f"{Path(n).stem}.txt"
            lines = label.read_text().splitlines() if label.exists() else []
            (DET / "labels" / label.name).write_text(
                "".join("0 " + ln.split(maxsplit=1)[1] + "\n" for ln in lines if ln.strip()))
        print(f"{name:5}: {len(names):4} images from {len(groups[name])} camera-days")

    (DET / "data.yaml").write_text(
        f'path: "{DET.as_posix()}"\ntrain: train.txt\nval: val.txt\ntest: test.txt\n'
        "names:\n  0: trash can\n")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Check the grouping rule.**

Run: `python -c "from tools.make_splits import split_of as s; assert s('cam','2026-10-03T09:00') == s('cam','2026-10-03T23:00'); print('same day, same split')"`
Expected: `same day, same split`

- [ ] **Step 3: Run it on the dataset and read the counts.**

Run: `python tools/make_splits.py`
Expected: three lines like `train:  212 images from 31 camera-days`. With fewer than about 20 camera-days the hash split can come out lopsided. If val or test gets under 10% of the images, collect more days rather than moving files by hand.

- [ ] **Step 4: Commit.**

```bash
git add waste_vision_system/tools/make_splits.py waste_vision_system/datasets/bins/*.txt
git commit -m "feat: camera-day grouped dataset splits"
```

---

### Task A4: Record the baseline

No code. Everything after this is judged against these numbers.

- [ ] **Step 1:** `python evaluate.py --split val --save` and `python evaluate.py --split test --save`.
- [ ] **Step 2:** Create `datasets/RESULTS.md` with one row per experiment, starting with the baseline:

```markdown
| Date | Change | Split | Det F1 | Count acc | Tier acc | ±1 tier | FULL R/P | MAE | Abstain | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| (date) | baseline (geometric, YOLOE zero-shot) | val | | | | | | | | |
| (date) | baseline | test | | | | | | | | |
```

- [ ] **Step 3:** Agree the targets in §5 with whoever operates the bins, then commit `RESULTS.md`.

---

## Phase B: Improvements that need no training

### Task B1: Camera placement

No code. For a front-on camera, physics caps fill accuracy no matter how good the model is.

- [ ] Mount cameras 3 to 5 m high, looking down at 30 to 60 degrees, so the bin openings are visible.
- [ ] Each bin at least 200 px tall in the frame (`bin_min_height_ratio` is 0.15 of frame height), with free space above every bin equal to half its height: that is where the overflow band is searched (`waste_overflow_band_ratio = 0.45`).
- [ ] No sun or bright lights directly behind the bins; an IR illuminator for night; fixed zoom and focus.
- [ ] Tag frames in `meta.csv` as `front_on`, `angled` or `top_down`. The per-tag lines of `evaluate.py` then show what each view costs. Expect `angled` and `top_down` to beat `front_on` on tier accuracy; if they don't, check the labels before moving cameras.

---

### Task B2: Tracker grace period

Today a bin that goes undetected for a single frame (someone walks past) loses its ID. It comes back with a new ID and an empty smoothing history, which causes flicker and breaks B3.

**Files:**
- Modify: `config/settings.py` (`Settings`), `detectors/bin_detector.py:68-118` (`_SimpleTracker`) and `:134` (`BinDetector.__init__`), `verify_system.py` (Test 10)

**Interfaces:**
- Produces: `_SimpleTracker(iou_threshold=0.3, max_age=0)`; `Settings.track_max_age: int = 15`

- [x] **Step 1: Write the failing test** (insert above `test_additional_fixtures`; add `("Tracker Grace Period", test_tracker_grace),` to `TESTS`):

```python
def test_tracker_grace():
    """
    Test 10: a bin hidden for a few frames keeps its ID; one gone longer
    than track_max_age gets a new ID.
    """
    print("\n--- Running Test 10: Tracker Grace Period ---")
    from detectors.bin_detector import _SimpleTracker
    t = _SimpleTracker(max_age=2)
    a, b = (0, 0, 100, 200), (300, 0, 400, 200)
    assert t.update([a, b]) == [1, 2]
    assert t.update([a]) == [1]          # b hidden for 1 frame
    assert t.update([a]) == [1]          # ... 2 frames
    assert t.update([a, b]) == [1, 2]    # back within max_age: same ID
    for _ in range(3):
        t.update([a])                    # hidden 3 frames > max_age
    assert t.update([a, b]) == [1, 3]
    print("Test 10 PASSED")
    return True
```

- [x] **Step 2: Run it.** `python -c "import verify_system as v; v.test_tracker_grace()"`
Expected: `TypeError: _SimpleTracker.__init__() got an unexpected keyword argument 'max_age'`

- [x] **Step 3: Implement.** In `Settings`, above `# ---- Export ----`:

```python
    # ---- Fixed-camera temporal model ----
    track_max_age: int = 15             # Processed frames a lost bin keeps its ID
```

Replace `_SimpleTracker.__init__`:

```python
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 0) -> None:
        self._next_id: int = 1
        self._tracks: Dict[int, Tuple[int, int, int, int]] = {}
        self._misses: Dict[int, int] = {}  # Consecutive frames each track went unmatched
        self._iou_thresh = iou_threshold
        self._max_age = max_age
```

At the end of `update()`, replace

```python
        active_ids = set(ids)
        self._tracks = {k: v for k, v in self._tracks.items() if k in active_ids}
        return ids
```

with

```python
        # Keep an unmatched track for max_age frames, so a bin hidden by a
        # passer-by gets its old ID (and smoothing history) back.
        for tid in list(self._tracks):
            self._misses[tid] = 0 if tid in ids else self._misses.get(tid, 0) + 1
            if self._misses[tid] > self._max_age:
                del self._tracks[tid], self._misses[tid]
        return ids
```

In `reset()`, add `self._misses.clear()` after `self._tracks.clear()`. In `BinDetector.__init__`, replace `self._tracker = _SimpleTracker()` with `self._tracker = _SimpleTracker(max_age=settings.track_max_age)`.

- [x] **Step 4: Run Test 10, then the full suite.** Expected: `Test 10 PASSED`; all tests pass, and tests 1 to 8 print unchanged values (single images reset tracking).

- [ ] **Step 5: Commit.** `git commit -am "feat: tracker keeps a lost bin's ID for track_max_age frames"`

---

### Task B3: Fill only rises between collections

Waste only accumulates; a bin gets emptier only when it is collected. So a reading far below the bin's current level is a passer-by, a lighting glitch, or a collection. A drop is accepted as a collection only after it lasts `collection_frames` readings in a row. Readings taken while a person covers the bin are left out.

**Files:**
- Create: `waste_vision_system/services/temporal.py`
- Modify: `config/settings.py`, `services/occupancy.py` (`__init__` :99, `estimate` :105-218, `_finalize` :301-331, `reset` :333), `detectors/waste_detector.py` (`__init__` :68, `detect` :271-276 and :324-326), `services/pipeline.py`, `verify_system.py` (Test 11)

**Interfaces:**
- Produces: `FillTimeline(settings).update(bin_id, raw_fill) -> (level, collected: bool)`, `.level(bin_id) -> float | None`, `.reset(bin_id=None)`; `OccupancyEstimator.estimate(..., occluded=False)`; `WasteDetector.last_person_frac: float` (set by every `detect()` call)
- Settings: `temporal_mode: str = "median"` (fixed-camera mode in B5 switches it to `"monotonic"`), `collection_drop_pct: float = 30.0`, `collection_frames: int = 5`, `occlusion_person_frac: float = 0.15`

- [x] **Step 1: Write the failing test** (add `("Fill Timeline", test_fill_timeline),` to `TESTS`):

```python
def test_fill_timeline():
    """
    Test 11: fill only rises; a short dip is ignored; a lasting drop is a collection.
    """
    print("\n--- Running Test 11: Fill Timeline ---")
    from services.temporal import FillTimeline
    s = Settings()
    s.smoothing_window, s.collection_frames, s.collection_drop_pct = 3, 3, 30.0
    tl = FillTimeline(s)
    levels = [tl.update(1, v)[0] for v in [20, 40, 35, 60, 5, 62, 60]]
    print(f"  Levels: {levels}")
    assert levels == sorted(levels), levels          # the 35 and the lone 5 never lower it
    collected = [tl.update(1, v)[1] for v in [5, 5, 5, 5, 5]]
    assert collected.count(True) == 1 and tl.level(1) < 10, (collected, tl.level(1))
    print("Test 11 PASSED")
    return True
```

- [x] **Step 2: Run it.** Expected: `ModuleNotFoundError: No module named 'services.temporal'`

- [x] **Step 3: Add the settings** under `track_max_age`:

```python
    temporal_mode: str = "median"       # "median" | "monotonic" (fixed cameras)
    collection_drop_pct: float = 30.0   # A drop this far below the level may be a collection...
    collection_frames: int = 5          # ...and is accepted after this many readings in a row
    occlusion_person_frac: float = 0.15 # Skip a reading when people cover this much of the bin
```

- [x] **Step 4: Create `services/temporal.py`.**

```python
"""
Fill that only rises between collections.

Waste accumulates; a bin only gets emptier when someone empties it. So a
reading far below the bin's current level is a passer-by, a lighting glitch,
or a collection. A drop is accepted as a collection only once it has lasted
`collection_frames` readings in a row; anything shorter is ignored.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from config.settings import Settings


@dataclass
class _Timeline:
    recent: Deque[float]
    level: Optional[float] = None
    low_streak: int = 0


class FillTimeline:
    """Per-bin monotonic fill level with collection detection."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._bins: Dict[int, _Timeline] = {}

    def update(self, bin_id: int, raw_fill: float) -> Tuple[float, bool]:
        """Feed one raw reading. Returns (level, collected_on_this_reading)."""
        t = self._bins.setdefault(
            bin_id, _Timeline(recent=deque(maxlen=self._s.smoothing_window)))
        t.recent.append(raw_fill)
        m = float(np.median(t.recent))  # one-frame spikes never reach the level
        if t.level is None:
            t.level = m
            return t.level, False
        if m < t.level - self._s.collection_drop_pct:
            t.low_streak += 1
            if t.low_streak < self._s.collection_frames:
                return t.level, False
            t.level, t.low_streak = m, 0
            t.recent = deque([m], maxlen=self._s.smoothing_window)
            return t.level, True
        t.low_streak = 0
        t.level = max(t.level, m)
        return t.level, False

    def level(self, bin_id: int) -> Optional[float]:
        """Current level without feeding a reading (used while occluded)."""
        t = self._bins.get(bin_id)
        return None if t is None else t.level

    def reset(self, bin_id: Optional[int] = None) -> None:
        if bin_id is None:
            self._bins.clear()
        else:
            self._bins.pop(bin_id, None)
```

- [x] **Step 5: Wire it into `OccupancyEstimator`.** Add `from services.temporal import FillTimeline` after the settings import, and `self._timeline = FillTimeline(settings)` at the end of `__init__`. In `estimate()`, add this parameter after `bin_color_mask`:

```python
        occluded: bool = False,
```

and document it under `bin_color_mask` in the docstring:

```
        occluded : bool
            A person covers the bin: report the current level but keep this
            reading out of the smoothing history.
```

Pass it on in the final call: inside the last `return self._finalize(`, replace `waste_pixel_count, bin_pixel_count,` with `waste_pixel_count, bin_pixel_count, occluded=occluded,`. Then replace the head of `_finalize`, from `bin_pixel_count: int,` through `smoothed = raw_fill`, with:

```python
        bin_pixel_count: int,
        occluded: bool = False,
    ) -> OccupancyResult:
        """Apply temporal smoothing and classify the fill state."""
        if self._settings.smoothing_enabled and self._settings.temporal_mode == "monotonic":
            level = self._timeline.level(bin_id)
            if occluded and level is not None:
                smoothed = level
            else:
                smoothed, _ = self._timeline.update(bin_id, raw_fill)
        elif self._settings.smoothing_enabled:
            if not (occluded and self._history[bin_id]):
                self._history[bin_id].append(raw_fill)
            smoothed = float(np.median(self._history[bin_id]))
        else:
            smoothed = raw_fill
```

In `reset()`, add `self._timeline.reset(bin_id)` as the first line.

- [x] **Step 6: Measure person coverage in `WasteDetector`.** The COCO waste model already detects people and currently throws them away. In `__init__` after `self._has_seg_masks = False`:

```python
        self.last_person_frac = 0.0  # Share of the last search region covered by people
```

In `detect()`, after `x2, y2 = min(w, x2), min(h, y2)`:

```python
        self.last_person_frac = 0.0
```

and replace

```python
                                if class_name is None:
                                    continue  # Skip non-waste COCO classes
```

with

```python
                                if class_name is None:
                                    if raw_name == "person":  # someone in front of the bin
                                        pw, ph = boxes.xywh[i, 2:].cpu().numpy()
                                        self.last_person_frac = min(1.0, self.last_person_frac
                                            + float(pw * ph) / (crop.shape[0] * crop.shape[1]))
                                    continue  # Skip non-waste COCO classes
```

- [x] **Step 7: Pass occlusion through `analyse_frame`.** In `services/pipeline.py`, replace the `occ = estimator.estimate(...)` call with:

```python
        occluded = waste_detector.last_person_frac > settings.occlusion_person_frac
        occ = estimator.estimate(
            b.bin_id, b.interior_mask, [w.mask for w in wastes],
            b.rim_top_y, b.rim_bottom_y,
            bin_color_mask=wall_mask_for(b, hsv), occluded=occluded,
        )
```

- [x] **Step 8: Run Test 11 and the full suite.** Expected: `Levels: [20.0, 30.0, 35.0, 40.0, 40.0, 60.0, 60.0]`, `Test 11 PASSED`; everything else unchanged (`temporal_mode` defaults to `"median"`).

- [ ] **Step 9: Commit.** `git commit -am "feat: monotonic fill timeline with collection detection and occlusion skip"`

**Tuning, once recordings exist:** record 2 to 3 days per camera with collection times written down. Count false collections (a drop accepted where none happened), missed collections, and tier changes per hour while the true fill was steady. Raise `collection_frames` if passers-by trigger false collections; lower `collection_drop_pct` if partial collections are missed.

---

### Task B4: Empty-bin reference differencing

With a photo of each bin taken while it was empty, "waste" becomes "what changed since then" instead of "anything that isn't the bin's colour". Mud, stickers, the hedge behind the bin and its shadow appear in both images and cancel out. This targets the first and third causes in §1.

**Files:**
- Create: `waste_vision_system/detectors/baseline.py`
- Modify: `config/settings.py`, `detectors/waste_detector.py` (import :34, new `search_region`, `detect` :255-292 and :372-390), `services/pipeline.py`, `verify_system.py` (Test 12)

**Interfaces:**
- Produces: `change_mask(ref_bgr, cur_bgr, settings) -> uint8 0/1 mask` (cur's size); `pick_reference(refs: list[ndarray], crop) -> ndarray`; `WasteDetector.search_region(bbox, frame_shape, band_ratio) -> (x1, y1, x2, y2)` (static); `WasteDetector.detect(..., reference=None)`; `analyse_frame(..., references: dict[int, list[ndarray]] | None = None)`, keyed by `BinDetection.roi_index` (added in B5)
- Settings: `baseline_diff_thresh: float = 18.0`, `baseline_min_blob_frac: float = 0.002`

- [x] **Step 1: Write the failing test** (add `("Empty-Reference Differencing", test_change_mask),` to `TESTS`):

```python
def test_change_mask():
    """
    Test 12: empty-reference differencing finds a new object despite camera
    shake and a 25% drop in light, and ignores everything else.
    """
    print("\n--- Running Test 12: Empty-Reference Differencing ---")
    from detectors.baseline import change_mask
    if not test_image_path().exists():
        print(f"Skipping Test 12: Image not found at {test_image_path()}")
        return SKIPPED
    ref = cv2.imread(str(test_image_path()))[100:340, 180:380].copy()  # real bin + mud texture
    cur = (ref.astype(np.float32) * 0.75).astype(np.uint8)   # dusk
    cur = np.roll(cur, (2, 3), axis=(0, 1))                  # camera shake
    cur[140:220, 40:150] = (40, 160, 230)                    # a new bag
    m = change_mask(ref, cur, Settings()).astype(bool)
    truth = np.zeros_like(m)
    truth[140:220, 40:150] = True
    iou = (m & truth).sum() / (m | truth).sum()
    print(f"  IoU of change mask with the inserted object: {iou:.2f}")
    assert iou > 0.85, iou
    print("Test 12 PASSED")
    return True
```

- [x] **Step 2: Run it.** Expected: `ModuleNotFoundError: No module named 'detectors.baseline'`

- [x] **Step 3: Add the settings** above `# ---- Export ----`:

```python
    # ---- Empty-bin reference differencing ----
    baseline_diff_thresh: float = 18.0  # LAB distance that counts as "changed"
    baseline_min_blob_frac: float = 0.002  # Drop change blobs smaller than this share of the crop
```

- [x] **Step 4: Create `detectors/baseline.py`.**

```python
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
```

Why medians: matching mean and standard deviation instead let a large new object shift the brightness correction for the whole crop. In testing that dropped the overlap with the true object from 0.92 to 0.69 at dusk light levels.

- [x] **Step 5: Teach `WasteDetector` to use a reference.** Add `from detectors.baseline import change_mask` under the settings import. Insert this static method directly above `def detect(`:

```python
    @staticmethod
    def search_region(
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, ...],
        band_ratio: float,
    ) -> Tuple[int, int, int, int]:
        """The bin box clipped to the frame and extended upward by the overflow
        band: the region waste is searched in. Reference capture and crop
        export use it too, so all three cover exactly the same pixels."""
        h, w = frame_shape[:2]
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w, bbox[2]), min(h, bbox[3])
        return x1, max(0, y1 - int((y2 - y1) * band_ratio)), x2, y2
```

Add `reference: Optional[np.ndarray] = None,` as the last parameter of `detect()`. Replace

```python
        bin_top = y1
        band_h = int((y2 - y1) * self._settings.waste_overflow_band_ratio)
        band_y1 = max(0, y1 - band_h)
```

with

```python
        bin_top = y1
        _, band_y1, _, _ = self.search_region(
            bin_bbox, frame.shape, self._settings.waste_overflow_band_ratio)
```

In Part 2, replace

```python
            color_mask = self._non_bin_color_mask(
                frame, x1, y1, x2, y2,
                bin_color_name, bin_hsv_range, interior_mask,
            )

            if band_y1 < bin_top and self._settings.waste_overflow_edge_gate:
```

with

```python
            if reference is not None:
                # Empty-bin reference: waste = what changed since the bin was
                # empty. Background cancels out, so the band gates are not needed.
                color_mask = np.zeros((h, w), dtype=np.uint8)
                color_mask[y1:y2, x1:x2] = change_mask(reference, frame[y1:y2, x1:x2], self._settings)
            else:
                color_mask = self._non_bin_color_mask(
                    frame, x1, y1, x2, y2,
                    bin_color_name, bin_hsv_range, interior_mask,
                )

            if reference is None and band_y1 < bin_top and self._settings.waste_overflow_edge_gate:
```

and change `if band_y1 < bin_top and self._settings.waste_overflow_require_contact:` to `if reference is None and band_y1 < bin_top and self._settings.waste_overflow_require_contact:`.

- [x] **Step 6: Pass references through `analyse_frame`.** In `services/pipeline.py`: change the typing import to `from typing import Dict, List, Optional, Tuple`, add `from detectors.baseline import pick_reference`, add the parameter `references: Optional[Dict[int, List[np.ndarray]]] = None,` after `estimator`, and replace the `wastes = waste_detector.detect(...)` call with:

```python
        x1, y1, x2, y2 = WasteDetector.search_region(
            b.bbox, frame.shape, settings.waste_overflow_band_ratio)
        refs = (references or {}).get(b.roi_index)
        ref = pick_reference(refs, frame[y1:y2, x1:x2]) if refs else None
        wastes = waste_detector.detect(
            frame, b.bbox, b.interior_mask,
            bin_color_name=b.bin_color_name, bin_hsv_range=b.bin_hsv_range,
            reference=ref,
        )
```

`b.roi_index` arrives in B5. Do B4 and B5 together, or add the `roi_index` field from B5 Step 4 now.

- [x] **Step 7: Run Test 12 and the full suite.** Expected: `IoU of change mask with the inserted object: 0.92`, `Test 12 PASSED`; tests 1 to 8 unchanged (no references are loaded yet).

- [ ] **Step 8: Commit.** `git commit -am "feat: empty-bin reference differencing for waste detection"`

---

### Task B5: Fixed-camera mode

Brings B2 to B4 together for CCTV: bin positions saved once per camera (so per-frame detection errors disappear), empty-bin references, and monotonic fill.

**Files:**
- Create: `waste_vision_system/config/cameras.py`, `waste_vision_system/tools/setup_camera.py`
- Modify: `detectors/bin_detector.py` (`BinDetection` :48-61, `_detect_manual_rois` :492), `main.py`, `ui/app.py`, `evaluate.py`, `verify_system.py` (Test 13), `.gitignore`, `README.md`

**Interfaces:**
- Produces: `camera_dir(id)`, `save_rois(id, rois, frame_size)`, `load_rois(id, frame_size) -> [(x1, y1, x2, y2)]` (rescaled if the stream resolution changed), `save_reference(id, roi_index, crop, tag)`, `load_references(id) -> {roi_index: [crops]}`; `BinDetection.roi_index: int | None`; `main.py --camera ID`; `evaluate.run(..., cameras=False)` and `--cameras`
- Storage: `config/cameras/<id>/rois.json` (`{"frame_size": [w, h], "rois": [[x1, y1, x2, y2], ...]}`), `config/cameras/<id>/empty_<roi_index>_<tag>.png`

- [x] **Step 1: Write the failing test** (add `("Camera Config", test_camera_config),` to `TESTS`):

```python
def test_camera_config():
    """
    Test 13: per-camera ROIs and references round-trip, ROIs rescale with the
    stream resolution, and unsafe camera ids are refused.
    """
    print("\n--- Running Test 13: Camera Config ---")
    import tempfile
    import config.cameras as cams
    cams.CAMERAS_DIR = Path(tempfile.mkdtemp())
    cams.save_rois("gate_north", [(10, 20, 110, 220)], (640, 480))
    assert cams.load_rois("gate_north", (1280, 960)) == [(20, 40, 220, 440)]
    crop = np.full((50, 40, 3), 128, np.uint8)
    cams.save_reference("gate_north", 0, crop, "day")
    refs = cams.load_references("gate_north")
    assert list(refs) == [0] and refs[0][0].shape == crop.shape
    for bad in ("../etc", "a/b", ""):
        try:
            cams.camera_dir(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted unsafe camera id {bad!r}")
    print("Test 13 PASSED")
    return True
```

- [x] **Step 2: Run it.** Expected: `ModuleNotFoundError: No module named 'config.cameras'`

- [x] **Step 3: Create `config/cameras.py`.** Camera ids become directory names and come from user input, so they are validated.

```python
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
```

- [x] **Step 4: Tag manual detections with their ROI index.** In `BinDetection`, after the `surface_mask` field and its comment lines:

```python
    roi_index: Optional[int] = None        # Position in the manual ROI list (fixed cameras)
```

In `_detect_manual_rois`, add `roi_index=idx,` after `confidence=1.0,`.

- [x] **Step 5: Run Test 13.** Expected: `Test 13 PASSED`.

- [x] **Step 6: Create `tools/setup_camera.py`.**

```python
"""
Fixed-camera setup from the command line (the dashboard can do the same).

    # 1. Save the bin boxes for a camera, in pixel coordinates of its stream:
    python tools/setup_camera.py --camera gate_north --source rtsp://... \\
        --roi 100 220 380 900 --roi 420 230 700 905
    # 2. While the bins are EMPTY, save references; repeat for each lighting condition:
    python tools/setup_camera.py --camera gate_north --source rtsp://... --tag day
    python tools/setup_camera.py --camera gate_north --source rtsp://... --tag ir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.cameras import load_rois, save_reference, save_rois  # noqa: E402
from config.settings import Settings  # noqa: E402
from detectors.waste_detector import WasteDetector  # noqa: E402
from services.stream_handler import StreamHandler  # noqa: E402


def grab_frame(source: str, settings: Settings):
    stream = StreamHandler(settings)
    if not stream.open(int(source) if source.isdigit() else source):
        sys.exit(f"cannot open {source}")
    frame = None
    for _ in range(10):  # live streams often start with a stale or grey frame
        ok, f = stream.read(skip_frames=False)
        frame = f if ok else frame
    stream.release()
    if frame is None:
        sys.exit("no frame received")
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--source", required=True, help="RTSP URI, video/image file, or webcam index")
    ap.add_argument("--roi", nargs=4, type=int, action="append", metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--tag", help="save empty-bin references under this lighting tag")
    args = ap.parse_args()

    s = Settings()
    frame = grab_frame(args.source, s)
    size = (frame.shape[1], frame.shape[0])
    if args.roi:
        print("saved", save_rois(args.camera, args.roi, size))
    if args.tag:
        rois = load_rois(args.camera, size)
        if not rois:
            sys.exit(f"no ROIs saved for {args.camera!r}; pass --roi first")
        for i, roi in enumerate(rois):
            x1, y1, x2, y2 = WasteDetector.search_region(roi, frame.shape, s.waste_overflow_band_ratio)
            print("saved", save_reference(args.camera, i, frame[y1:y2, x1:x2], args.tag))


if __name__ == "__main__":
    main()
```

- [x] **Step 7: Add `--camera` to `main.py`.** Add the argument after `--no-display`:

```python
    parser.add_argument(
        "--camera", type=str, default=None,
        help="Fixed-camera id: use its saved bin ROIs and empty references "
             "(config/cameras/<id>/, see tools/setup_camera.py).",
    )
```

In `_run_headless`, add `from config.cameras import load_references, load_rois` to the imports, and after the `if not stream.open(source): ... sys.exit(1)` block:

```python
    references = None
    if args.camera:
        # Fixed camera: its saved bin ROIs, its empty-bin references, and
        # fill that only rises between collections.
        rois = load_rois(args.camera, stream.frame_size)
        if rois:
            bin_detector.set_manual_rois(rois)
        references = load_references(args.camera)
        settings.temporal_mode = "monotonic"
        print(f"[Waste Vision] Camera {args.camera}: {len(rois)} ROIs, "
              f"empty references for {len(references)} of them")
```

and change the pipeline call to `analyse_frame(frame, settings, bin_detector, waste_detector, occ_estimator, references)`.

- [x] **Step 8: Add fixed-camera mode to the dashboard.** In `ui/app.py`, add the import `from config.cameras import load_references, load_rois, save_reference, save_rois`. In `_init_state`'s defaults, after `"_img_upload_id": None,`:

```python
        "raw_frame": None,        # last unannotated frame (for reference capture)
        "camera_refs": None,      # empty-bin references of the active fixed camera
```

Add above `def handle_image(`:

```python
def _use_fixed_camera(settings: Settings, enabled: bool) -> None:
    """Webcam/RTSP with a Camera ID: use that camera's empty-bin references
    and fill that only rises between collections. Uploads never do."""
    cam_id = st.session_state.get("camera_id") if enabled else None
    refs = None
    if cam_id:
        try:
            refs = load_references(cam_id)
        except ValueError as e:
            st.warning(f"Camera ID ignored: {e}")
            cam_id = None
    st.session_state["camera_refs"] = refs
    settings.temporal_mode = "monotonic" if cam_id else "median"
```

Call it on every new scene: in `handle_image`, after `settings.smoothing_enabled = False`, add `_use_fixed_camera(settings, enabled=False)`; in `handle_video`, after `settings.smoothing_enabled = True`, add `_use_fixed_camera(settings, enabled=False)`; in `handle_webcam` and `handle_rtsp`, after `settings.smoothing_enabled = True`, add `_use_fixed_camera(settings, enabled=True)`.

In `process_frame`, pass the references and keep the raw frame:

```python
    results_list = analyse_frame(
        frame, settings, bin_detector, waste_detector, occupancy_estimator,
        references=st.session_state["camera_refs"],
    )
```

and after `st.session_state["last_frame"] = annotated` add `st.session_state["raw_frame"] = frame`.

In `main()`'s webcam/RTSP sidebar, replace

```python
        st.sidebar.markdown("### Manual Bin Regions")
        manual_rois = st.session_state.get("manual_rois", [])
```

with

```python
        st.sidebar.markdown("### Manual Bin Regions")
        cam_id = st.sidebar.text_input(
            "Camera ID (fixed camera)", key="camera_id",
            help="Letters, digits, _ and -. ROIs and empty-bin references are "
                 "saved under config/cameras/<id>/.")
        raw = st.session_state.get("raw_frame")
        if cam_id and raw is not None:
            size = (raw.shape[1], raw.shape[0])
            try:
                c_load, c_save = st.sidebar.columns(2)
                if c_load.button("Load ROIs", key="cam_load"):
                    st.session_state["manual_rois"] = load_rois(cam_id, size)
                    st.rerun()
                if c_save.button("Save ROIs", key="cam_save") and st.session_state.get("manual_rois"):
                    save_rois(cam_id, st.session_state["manual_rois"], size)
                    st.sidebar.success("ROIs saved")
                tag = st.sidebar.selectbox("Lighting", ["day", "night", "ir"], key="ref_tag")
                if st.sidebar.button("Capture empty reference", key="cam_ref",
                                     help="Only while every bin is empty"):
                    band = st.session_state["settings"].waste_overflow_band_ratio
                    for i, roi in enumerate(st.session_state.get("manual_rois", [])):
                        x1, y1, x2, y2 = WasteDetector.search_region(roi, raw.shape, band)
                        save_reference(cam_id, i, raw[y1:y2, x1:x2], tag)
                    st.sidebar.success("Empty references saved")
            except ValueError as e:
                st.sidebar.error(str(e))
        manual_rois = st.session_state.get("manual_rois", [])
```

- [x] **Step 9: Add `--cameras` to `evaluate.py`.** Change the signature to `def run(split: str, settings: Settings, detect_only: bool = False, cameras: bool = False) -> dict:` and add to its docstring:

```
    cameras:     fixed-camera mode: use each image's saved ROIs and empty-bin
                 references (config/cameras/<camera_id>/) where they exist
```

Add `from config.cameras import load_references, load_rois` to its imports, and replace `bd.reset_tracking()` in the loop with:

```python
        bd.reset_tracking()
        bd.clear_manual_rois()
        refs = None
        if cameras and info.get("camera_id"):
            rois = load_rois(info["camera_id"], (w, h))
            if rois:
                bd.set_manual_rois(rois)
                refs = load_references(info["camera_id"])
```

Pass `refs` as the sixth argument to `analyse_frame`. Add `ap.add_argument("--cameras", action="store_true", help="use saved per-camera ROIs and references")` and call `run(args.split, settings, detect_only=args.detect_only, cameras=args.cameras)`.

- [x] **Step 10: Keep site images out of git.** Append `waste_vision_system/config/cameras/*/empty_*.png` to `.gitignore` (the `rois.json` files can be committed).

- [x] **Step 11: Run everything.** `python verify_system.py` (13 tests pass). Then a smoke run with a fixture standing in for a camera:

```bash
python tools/setup_camera.py --camera smoke --source tests/fixtures/001.jpg --roi 33 114 170 347 --roi 173 111 311 345
python tools/setup_camera.py --camera smoke --source tests/fixtures/001.jpg --tag day
python main.py --headless --camera smoke --source tests/fixtures/001.jpg --no-display
```

Expected: `Camera smoke: 2 ROIs, empty references for 2 of them`, and both bins read `0% (EMPTY)`. That is correct, because the "empty" reference was taken from this same image of full bins, so nothing changed. It is also this feature's main operational risk: references captured while bins aren't empty make full bins read empty. Delete `config/cameras/smoke/` afterwards.

- [ ] **Step 11b: Check the dashboard controls (needs a live camera).** Open a webcam, set a Camera ID, add ROIs, press Save ROIs, and check `config/cameras/<id>/rois.json`. Then Load ROIs after a restart, and Capture empty reference while the bins are empty.

- [x] **Step 12: Document it.** Add a "Fixed cameras" section to `README.md` §6 covering `tools/setup_camera.py`, the dashboard Camera ID controls, and this rule: capture references only when every bin is empty; then run the camera, confirm every bin reads EMPTY, drop one bag in, and confirm that bin rises. Recapture after a bin is replaced, moved or repainted, and once per season.

- [ ] **Step 13: Commit.** `git commit -am "feat: fixed-camera mode with saved ROIs and empty-bin references"`

**Confirm the gain (needs A1 data):** save ROIs and references for the labelled cameras, then compare `python evaluate.py --split val` with `python evaluate.py --split val --cameras`. Record both rows in `RESULTS.md`.

---

### Task B6: Tune the zero-shot detector

Needs the labelled val split. Three cheap levers, each judged with `evaluate.py --detect-only`.

**Files:**
- Create: `waste_vision_system/tools/sweep_openvocab.py`, `waste_vision_system/tools/make_vp_model.py`
- Modify: `config/settings.py`, `detectors/bin_detector.py` (`_load_openvocab` :363-395)

**Interfaces:**
- Consumes: `evaluate.run(split, settings, detect_only=True)`
- Settings: `openvocab_vp_model_path: str = ""`

- [ ] **Step 1: Create `tools/sweep_openvocab.py`.**

```python
"""
Tune the zero-shot bin detector against the labelled val split (needs Task A1 data).

Staged so it stays affordable on a CPU: scales x confidence first, then prompt
sets, then NMS, each stage keeping the winner of the stage before. Every new
prompt set re-bakes yoloe-11l-seg-bins.pt (the first one downloads mobileclip_blt.ts).

    python tools/sweep_openvocab.py

Copy the winning values into Settings, then confirm once with
`python evaluate.py --split test`.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from evaluate import run  # noqa: E402

PROMPT_SETS = [
    Settings().openvocab_prompts,
    ("trash can", "wheelie bin", "dumpster"),
    ("garbage bin", "recycling bin", "wheeled garbage bin", "litter bin"),
]


def score(s: Settings) -> tuple:
    m = run("val", s, detect_only=True)["overall"]
    print(f"  F1={m['f1']:.3f} P={m['precision']:.2f} R={m['recall']:.2f} "
          f"count={m['count_accuracy']}  scales={s.openvocab_scales} "
          f"conf={s.openvocab_conf} nms={s.openvocab_nms_iou} prompts={s.openvocab_prompts}")
    return m["f1"], m["recall"]


def main() -> None:
    s = Settings()
    s = max((replace(s, openvocab_scales=sc, openvocab_conf=c)
             for sc in ((640,), (1280,), (640, 1280))
             for c in (0.10, 0.15, 0.20, 0.25, 0.30)), key=score)
    s = max((replace(s, openvocab_prompts=p) for p in PROMPT_SETS), key=score)
    s = max((replace(s, openvocab_nms_iou=n) for n in (0.4, 0.5, 0.6)), key=score)
    print(f"\nBEST  openvocab_scales={s.openvocab_scales}  openvocab_conf={s.openvocab_conf}  "
          f"openvocab_nms_iou={s.openvocab_nms_iou}\n      openvocab_prompts={s.openvocab_prompts}")


if __name__ == "__main__":
    main()
```

Edit `PROMPT_SETS` to name the bin types actually on site. The sweep makes 21 passes over the val split; at roughly 1 to 4 s per image on CPU, a 50-image val split takes 30 to 60 minutes.

- [ ] **Step 2: Visual prompts.** YOLOE can take example boxes of your own bins instead of words. Add to `Settings`, next to the other model paths:

```python
    openvocab_vp_model_path: str = ""   # YOLOE built from visual prompts (tools/make_vp_model.py)
```

In `_load_openvocab`, replace

```python
            baked = model_path.with_name(f"{model_path.stem}-bins.pt")
            model = YOLOE(str(baked)) if baked.exists() else None
            if model is None or list(model.names.values()) != prompts:
```

with

```python
            # A visual-prompt model already has its classes fused in from
            # example images, so there is nothing to bake.
            vp = self._settings.openvocab_vp_model_path
            baked = Path(vp) if vp else model_path.with_name(f"{model_path.stem}-bins.pt")
            model = YOLOE(str(baked)) if baked.exists() else None
            if vp and model is None:
                raise FileNotFoundError(vp)
            if not vp and (model is None or list(model.names.values()) != prompts):
```

Create `tools/make_vp_model.py`:

```python
"""
Build a YOLOE bin detector from example boxes of YOUR bins (visual prompts).
No training: the appearance inside the boxes becomes the "bin" class.

    python tools/make_vp_model.py --image datasets/bins/images/cam1_0001.jpg \\
        --box 120 80 260 400 --box 300 90 430 410 --out yoloe-11l-seg-vp.pt

Pick one image that shows every bin model on site and box each of them. Then
set Settings.openvocab_vp_model_path to the output and compare with evaluate.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--box", nargs=4, type=float, action="append", required=True,
                    metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--out", default=str(ROOT / "yoloe-11l-seg-vp.pt"))
    args = ap.parse_args()

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"cannot read {args.image}")
    model = YOLOE(Settings().openvocab_model_path)
    prompts = {"bboxes": np.array(args.box), "cls": np.zeros(len(args.box), dtype=int)}
    model.predict(img, refer_image=img, visual_prompts=prompts,
                  predictor=YOLOEVPSegPredictor, verbose=False)
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
```

In testing, a model prompted with four green wheelie bins found all 4 on its reference image but only 2 of 3 dumpsters elsewhere. The examples have to cover every bin model on site. Keep the visual-prompt model only if `evaluate.py --detect-only` shows it beating the text prompts on val.

- [ ] **Step 3: Try the newer YOLOE-26 family.** Set `openvocab_model_path = str(_PROJECT_ROOT / "yoloe-26l-seg.pt")` and run `evaluate.py --split val --detect-only`. The first run downloads the model and its text encoder (`mobileclip2_b.ts`, about 254 MB) and bakes `yoloe-26l-seg-bins.pt`. Keep it only if F1 and speed both hold up.

- [ ] **Step 4: Record each result in `RESULTS.md`, put the winners into `Settings`, run `evaluate.py --split test` once, and commit.**

---

### Task B7: Area-ratio numerator experiment

The aperture area ratio divides "waste anywhere in the bin box" by "box minus bin wall". On one fixture bin, 71% of the counted waste sat on the wall, so the ratio saturated. This adds an opt-in mode that counts only waste inside the opening. Keep it or drop it on evidence.

**Files:**
- Modify: `config/settings.py`, `services/occupancy.py` (`estimate`, the aperture branch at :181-182)

**Interfaces:**
- Settings: `area_ratio_mode: str = "legacy"` (`"legacy"` | `"opening"`)

- [ ] **Step 1: Add the setting** next to the occupancy weights:

```python
    area_ratio_mode: str = "legacy"  # "legacy" | "opening" (Task B7 experiment)
```

- [ ] **Step 2: Implement.** In `estimate()`, replace

```python
            if opening_count > bin_pixel_count * 0.05:
                area_ratio = min(100.0, (waste_pixel_count / opening_count) * 100.0)
```

with

```python
            if opening_count > bin_pixel_count * 0.05:
                in_opening = waste_pixel_count
                if self._settings.area_ratio_mode == "opening":
                    # Count only waste inside the opening, matching the denominator.
                    in_opening = int(np.logical_and(waste_inside.astype(bool),
                                                    ~wall_inside.astype(bool)).sum())
                area_ratio = min(100.0, (in_opening / opening_count) * 100.0)
```

- [ ] **Step 3: Check it.** With a 10x10 interior, the bottom half as wall, and waste in rows 3 to 6, `legacy` must give an area ratio of 80 and `opening` 40:

```bash
python -c "import numpy as np; from dataclasses import replace; from config.settings import Settings; from services.occupancy import OccupancyEstimator; i=np.ones((10,10),np.uint8); w=np.zeros_like(i); w[5:]=1; x=np.zeros_like(i); x[3:7]=1; print([OccupancyEstimator(replace(Settings(),area_ratio_mode=m,smoothing_enabled=False)).estimate(1,i,[x],0,10,bin_color_mask=w).area_ratio for m in ('legacy','opening')])"
```

Expected: `[80.0, 40.0]`

- [ ] **Step 4: Decide with data.** Run `evaluate.py --split val` with each mode. Keep `opening` only if tier accuracy and MAE improve without FULL recall dropping; otherwise remove the code. Record the result in `RESULTS.md` and commit.

---

## Phase C: Training (once labelled data exists)

### Task C1: Detector training data

**Files:** data only, plus `datasets/README.md`.

- [ ] **Step 1:** Run `tools/make_splits.py` (A3). It builds `datasets/bins_det/` with every class rewritten to 0 and a `data.yaml` naming the class `trash can`.
- [ ] **Step 2 (optional): add public data,** for detection only. Roboflow Universe has bin and dumpster datasets; check each licence, as several are non-commercial. Export in YOLO format into `datasets/<name>/images` and `labels`. These have no fill tiers, so rewrite every class to 5 (UNKNOWN) before merging into `datasets/bins`:

```python
from pathlib import Path
for f in Path("datasets/<name>/labels").glob("*.txt"):
    f.write_text("".join("5 " + ln.split(maxsplit=1)[1] + "\n"
                         for ln in f.read_text().splitlines() if ln.strip()))
```

Then copy the images and labels into `datasets/bins/`, add rows to `meta.csv` with `camera_id` set to the dataset name and the tag `public`, and re-run `make_splits.py`. Public images must never end up in `test`: move those lines to `train.txt` by hand, so the test split measures your cameras only.
- [ ] **Step 3:** Record source, licence and date for each dataset in `datasets/README.md`.

---

### Task C2: Fine-tune the bin segmentation model

**Files:**
- Modify: `detectors/bin_detector.py` (`_detect_yolo` :397-464)

**Interfaces:**
- Produces: `_detect_yolo` returns `BinDetection` with `surface_mask` set and the rim taken from that mask, the same as the open-vocab path. Today it leaves `surface_mask` empty, so occupancy silently falls back to the HSV wall mask.

- [ ] **Step 1: Pass the instance mask through.** In `_detect_yolo`, after `masks_list: List[Optional[np.ndarray]] = []` add `surfaces: List[Optional[np.ndarray]] = []`. Replace

```python
            seg_mask = None
            if self._use_segmentation and result.masks is not None and i < len(result.masks):
                seg_mask = (result.masks[i].data.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                k = self._settings.bin_interior_erode_kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                seg_mask = cv2.erode(seg_mask, kernel, iterations=1)
```

with

```python
            seg_mask = surface = None
            if self._use_segmentation and result.masks is not None and i < len(result.masks):
                surface = (result.masks[i].data.cpu().numpy().squeeze() > 0.5).astype(np.uint8)
                k = self._settings.bin_interior_erode_kernel
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                seg_mask = cv2.erode(surface, kernel, iterations=1)
```

add `surfaces.append(surface)` after `masks_list.append(seg_mask)`, and replace the final `detections.append(BinDetection(...))` block with:

```python
            # Same as the open-vocab path: the instance mask is the wall mask
            # for occupancy, and its top row is the real rim.
            surface, rim_top, rim_bot = surfaces[idx], y1, y2
            if surface is not None and surface.any():
                rows = np.nonzero(surface.any(axis=1))[0]
                rim_top, rim_bot = int(rows[0]), int(rows[-1]) + 1
            detections.append(BinDetection(
                bin_id=ids[idx], bbox=bbox, interior_mask=mask,
                confidence=conf, rim_top_y=rim_top, rim_bottom_y=rim_bot,
                bin_color_name=color_name, bin_hsv_range=hsv_range,
                surface_mask=surface,
            ))
```

- [ ] **Step 2: Check the wiring before any training,** using the COCO waste model as a stand-in:

```bash
python -c "from dataclasses import replace; import cv2; from config.settings import Settings; from detectors.bin_detector import BinDetector; s=replace(Settings(), bin_model_path=Settings().waste_model_path, confidence_threshold=0.05); d=BinDetector(s).detect(cv2.imread('tests/fixtures/001.jpg')); print(len(d), all(x.surface_mask is not None and x.surface_mask[x.rim_top_y].any() for x in d))"
```

Expected: a detection count above 0, and `True`.

- [ ] **Step 3: Train.** Use a GPU if you can (Colab and Kaggle both work); CPU training of these models takes many hours. Pick the route by dataset size.

Under about 300 labelled images, linear-probe YOLOE so only the classification head learns:

```python
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

model = YOLOE("yoloe-11l-seg.pt")
head = len(model.model.model) - 1
freeze = [str(i) for i in range(head)]
for name, _ in model.model.model[-1].named_children():
    if "cv3" in name:
        freeze.extend(f"{head}.{name}.{i}.{j}" for i in range(3) for j in (0, 1))
    else:
        freeze.append(f"{head}.{name}")
model.train(data="datasets/bins_det/data.yaml", epochs=80, patience=10, imgsz=960,
            trainer=YOLOEPESegTrainer, freeze=freeze, project="runs", name="bins_yoloe_lp")
```

From about 300 images up, fine-tune a smaller, faster model end to end:

```python
from ultralytics import YOLO
YOLO("yolo11s-seg.pt").train(
    data="datasets/bins_det/data.yaml", epochs=100, patience=20, imgsz=960,
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.4, degrees=5, scale=0.5,  # lighting, angle, distance
    project="runs", name="bins_seg")
```

Both need polygon labels (A1 Step 4). There is no grayscale augmentation that stands in for night footage, so collect real IR frames.

- [ ] **Step 4: Point the app at the result.** Ultralytics only loads a checkpoint as YOLOE when "yoloe" appears in its file name, so copy a YOLOE result before using it:

```bash
copy runs\bins_yoloe_lp\weights\best.pt yoloe-bins-ft.pt
```

Then set `bin_model_path` in `Settings` to `str(_PROJECT_ROOT / "yoloe-bins-ft.pt")` (or to the `yolo11s-seg` result). The detection strategy order already prefers a configured model over open-vocab.

- [ ] **Step 5: Accept or reject.** `evaluate.py --split val --detect-only`. Accept if F1 beats the B6 zero-shot result by at least 0.05, recall is at least 0.95, and no tag (`night`, `ir`, `rain`) drops by more than 0.05. Confirm once on test, record in `RESULTS.md`, and commit the code change (weights stay out of git).

---

### Task C3: Learned fill classifier

The biggest expected gain. A small image classifier looks at each bin crop and predicts its tier, learning what the hand-tuned constants try to encode (weights, the 2.5x fallback, overflow tiers, texture thresholds). When it isn't confident it says so (UNCERTAIN) instead of guessing.

**Files:**
- Create: `waste_vision_system/services/fill_model.py`, `waste_vision_system/tools/export_crops.py`
- Modify: `config/settings.py` (`FillStatus` :34-41, `STATUS_COLORS` :46-52, `STATUS_COLORS_HEX` :55-61, `Settings`), `services/occupancy.py`, `services/pipeline.py`, `ui/app.py`, `main.py`, `evaluate.py`

**Interfaces:**
- Produces: `FillStatus.UNCERTAIN`; `FillClassifier(settings).predict(crop) -> (pct, confidence)`; `OccupancyEstimator.from_classifier(bin_id, pct, confidence, occluded=False) -> OccupancyResult`; `analyse_frame(..., fill_classifier=None)`
- Settings: `fill_mode: str = "geometric"`, `fill_model_path: str = ""`, `fill_min_conf: float = 0.6`, `fill_imgsz: int = 224`

- [ ] **Step 1: Add `UNCERTAIN` and the settings.** In `FillStatus`, after `FULL = "FULL"`: `UNCERTAIN = "UNCERTAIN"  # Learned fill model abstained (low confidence)`. In `STATUS_COLORS` add `FillStatus.UNCERTAIN:   (160, 160, 160),    # Grey`, and in `STATUS_COLORS_HEX` add `FillStatus.UNCERTAIN:   "#A0A0A0",`. `classify_fill()` needs no change, since it only walks `fill_thresholds`. In `Settings`:

```python
    # ---- Learned fill model (set once trained) ----
    fill_mode: str = "geometric"        # "geometric" | "model"
    fill_model_path: str = ""           # Fill classifier weights (YOLO11-cls)
    fill_min_conf: float = 0.6          # Below this the classifier abstains (UNCERTAIN)
    fill_imgsz: int = 224
```

- [ ] **Step 2: Create `tools/export_crops.py`.** It cuts each labelled bin's search region into a folder per tier, so the classifier sees exactly the region it will see at run time.

```python
"""
Cut one crop per labelled bin into the ImageFolder tree the fill classifier trains on:

    datasets/fill/{train,val,test}/{EMPTY,LOW,MEDIUM,NEARLY_FULL,FULL}/*.jpg

Crops cover the region WasteDetector searches (bin box + overflow band), so
the classifier sees what it will see at run time. Rare tiers in train are
oversampled to the size of the largest one. datasets/fill/ is generated:
it is rebuilt from the labels on every run.

    python tools/export_crops.py
"""

from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from detectors.waste_detector import WasteDetector  # noqa: E402
from evaluate import DATASET, TIERS, load_split, read_labels  # noqa: E402

OUT = ROOT / "datasets" / "fill"


def main() -> None:
    band = Settings().waste_overflow_band_ratio
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val", "test"):
        counts = Counter()
        for img_path in load_split(split):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            labels = read_labels(DATASET / "labels" / f"{img_path.stem}.txt", w, h)
            for k, (cls, box) in enumerate(labels):
                if cls >= len(TIERS):  # UNKNOWN: no fill label to learn from
                    continue
                x1, y1, x2, y2 = WasteDetector.search_region(box, frame.shape, band)
                d = OUT / split / TIERS[cls].name
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{img_path.stem}_{k}.jpg"), frame[y1:y2, x1:x2])
                counts[TIERS[cls].name] += 1
        print(f"{split:5}: {dict(counts)}")
        missing = [t.name for t in TIERS if counts[t.name] == 0]
        if missing:
            # The trainer numbers classes per split from the folders present;
            # a missing tier would silently shift every id after it.
            sys.exit(f"{split} has no crops for {missing}: label more, or rebalance the split")
        if split == "train":
            top = max(counts.values())
            for tier, n in counts.items():
                files = sorted((OUT / split / tier).glob("*.jpg"))
                for i in range(top - n):
                    src = files[i % len(files)]
                    shutil.copy2(src, src.with_name(f"{src.stem}_dup{i}.jpg"))


if __name__ == "__main__":
    main()
```

Run: `python tools/export_crops.py`
Expected: per-split counts, with train equalised. If a split lacks a tier, it stops with `... has no crops for [...]`. That refusal matters: the trainer numbers classes from the folders present in each split, so a missing tier would quietly mislabel every tier after it. Aim for at least 100 crops per tier in train and 15 per tier in val and test.

- [ ] **Step 3: Create `services/fill_model.py`.** The classifier stores its classes alphabetically (`{0: EMPTY, 1: FULL, 2: LOW, ...}` in testing), so tiers are mapped by name, never by index.

```python
"""
Learned fill classifier: bin crop -> tier probabilities -> fill %.

Trained with Ultralytics' classifier on datasets/fill/ (tools/export_crops.py).
The class folders are named after FillStatus members, so the model's own
names map back to tiers whatever order Ultralytics stores them in.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from config.settings import FillStatus, Settings

MIDPOINTS = {
    FillStatus.EMPTY: 10.0, FillStatus.LOW: 30.0, FillStatus.MEDIUM: 50.0,
    FillStatus.NEARLY_FULL: 70.0, FillStatus.FULL: 90.0,
}


class FillClassifier:
    def __init__(self, settings: Settings) -> None:
        from ultralytics import YOLO

        self._s = settings
        self._model = YOLO(settings.fill_model_path)
        self._mid = np.array([MIDPOINTS[FillStatus[name]]
                              for _, name in sorted(self._model.names.items())])

    def predict(self, crop: np.ndarray) -> Tuple[float, float]:
        """Returns (expected fill %, probability of the most likely tier)."""
        r = self._model(crop, imgsz=self._s.fill_imgsz,
                        device=self._s.resolve_device(), verbose=False)[0]
        p = r.probs.data.cpu().numpy()
        return float(p @ self._mid), float(p.max())
```

The reported fill % is the probability-weighted average of tier midpoints. It moves smoothly, which keeps the UI bar and the B3 timeline meaningful.

- [ ] **Step 4: Add `from_classifier` to `OccupancyEstimator`,** directly above `reset()`:

```python
    def from_classifier(
        self, bin_id: int, pct: float, confidence: float, occluded: bool = False,
    ) -> OccupancyResult:
        """Occupancy from the learned fill model (Settings.fill_mode == "model").
        Below fill_min_conf the model abstains: status UNCERTAIN, and the
        reading stays out of the smoothing history."""
        if confidence < self._settings.fill_min_conf:
            return OccupancyResult(bin_id, pct, FillStatus.UNCERTAIN, pct, 0.0, 0.0, 0.0, 0, 0)
        return self._finalize(bin_id, pct, 0.0, 0.0, 0.0, 0, 0, occluded=occluded)
```

- [ ] **Step 5: Branch in `analyse_frame`.** Add the parameter `fill_classifier=None,` after `references`, and replace the `occ = estimator.estimate(...)` call with:

```python
        if fill_classifier is not None:
            pct, conf = fill_classifier.predict(frame[y1:y2, x1:x2])
            occ = estimator.from_classifier(b.bin_id, pct, conf, occluded=occluded)
        else:
            occ = estimator.estimate(
                b.bin_id, b.interior_mask, [w.mask for w in wastes],
                b.rim_top_y, b.rim_bottom_y,
                bin_color_mask=wall_mask_for(b, hsv), occluded=occluded,
            )
```

- [ ] **Step 6: Load it where the pipeline runs.** In `ui/app.py`, add `"fill_classifier": None,` to the `_init_state` defaults, put this at the end of the creation block in `_get_pipeline`:

```python
    if settings.fill_mode == "model" and s["fill_classifier"] is None:
        from services.fill_model import FillClassifier
        s["fill_classifier"] = FillClassifier(settings)
```

and pass `fill_classifier=st.session_state["fill_classifier"],` to `analyse_frame` in `process_frame`. In `main.py`, after the `--camera` block:

```python
    fill_clf = None
    if settings.fill_mode == "model":
        from services.fill_model import FillClassifier
        fill_clf = FillClassifier(settings)
```

and pass `references, fill_clf` to `analyse_frame`. In `evaluate.run`, after creating `bd, wd, oe`:

```python
    clf = None
    if settings.fill_mode == "model" and not detect_only:
        from services.fill_model import FillClassifier
        clf = FillClassifier(settings)
```

and pass `refs, clf` to `analyse_frame`. Abstentions show up as `abstain=` in the report.

- [ ] **Step 7: Check the wiring before real training.** A one-epoch model trained from scratch on any exported crops is enough to prove the plumbing (it learns nothing useful):

```python
from ultralytics import YOLO
YOLO("yolo11n-cls.yaml").train(data="datasets/fill", epochs=1, imgsz=64, plots=False,
                               project="runs", name="fill_smoke")
```

Then evaluate with `fill_mode="model"` and `fill_model_path="runs/fill_smoke/weights/best.pt"`. Expected: `abstain=1.0` at the default `fill_min_conf` (an untrained model is never confident), and real predictions with `fill_min_conf=0`. Both were observed in testing. Delete `runs/fill_smoke` afterwards.

- [ ] **Step 8: Train for real.** Start from ImageNet weights. Below about 500 crops, train only the head (`freeze=10`; the classifier's backbone is layers 0 to 9):

```python
from ultralytics import YOLO
YOLO("yolo11n-cls.pt").train(data="datasets/fill", epochs=60, patience=15, imgsz=224,
                             fliplr=0.5, hsv_v=0.4, freeze=10, project="runs", name="fill_cls")
```

With more crops, drop `freeze=10`. Copy `runs/fill_cls/weights/best.pt` to `fill-cls.pt` and set `fill_model_path` to it.

- [ ] **Step 9: Pick `fill_min_conf` on val.** Run `evaluate.run("val", replace(Settings(), fill_mode="model", fill_model_path=..., fill_min_conf=c))` for `c` in 0.4, 0.5, 0.6, 0.7 and 0.8. Take the highest tier accuracy with an abstain rate of at most 10%.

- [ ] **Step 10: Accept or reject.** Switch the default to `fill_mode = "model"` only if, on test, tier accuracy beats the best geometric result (A4/B5/B7), FULL recall is at least 0.95, and the abstain rate is at most 10%. Otherwise keep collecting crops (C4) and retrain. Keep the geometric path either way; it is the fallback when the model file is missing. Record in `RESULTS.md` and commit.

---

### Task C4: Review loop for hard cases

No code. The model improves fastest on the frames it gets wrong.

- [ ] Weekly, open the most recent `evaluate.py --save` JSON. Its `per_image` list shows every image where count or tier was wrong. In production, rows with status `UNCERTAIN` in `exports/csv/session_*.csv` point to timestamps worth pulling from the recorder.
- [ ] Label those frames per A1, add them to `meta.csv`, re-run `make_splits.py` and `export_crops.py`, retrain C3 (and C2 if detection errors dominate), and compare on the same test split.
- [ ] Optional: FiftyOne (`pip install fiftyone`) shows predictions over ground truth and is the quickest way to find label mistakes. It is a tool for you, not a project dependency.

**Skipped on purpose: a bigger or waste-specific waste model** (for example YOLO11s-seg trained on TACO). It only matters if the product needs waste *types*; for fill level, C3 replaces the work the waste model does today. If waste types become a requirement, train `yolo11s-seg.pt` on TACO plus site labels (check TACO's licence) and set `Settings.waste_model_path`.

---

## Phase D: Past the limits of one camera image

Start only if C3 plateaus below the targets in §5.

### Task D1: Monocular depth experiment

A depth map can tell rubbish on the bin from rubbish behind it, which colour and texture can't. This is an experiment, not a feature.

- [ ] Run Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`, through the `transformers` depth-estimation pipeline) on 50 labelled crops. Not yet tested: the model needs a download.
- [ ] For each bin, compare the median depth inside the opening with the rim's depth; in a fuller bin the contents sit closer to the rim plane.
- [ ] Success means this single feature separates EMPTY/LOW from NEARLY_FULL/FULL better than chance on those 50 crops. If it does, add it as an extra input to C3; if not, stop.

### Task D2: Ultrasonic sensors as ground truth

A lid-mounted ultrasonic sensor measures fill directly. Even on 5 to 10 bins it produces a steady stream of correct labels for the camera model at no labelling cost. It changes the product, so it is a business decision.

- [ ] Log readings as CSV: `bin_uid,read_at,distance_cm`. Record each bin model's empty and full distances.
- [ ] Join readings to frames (nearest reading within 60 s) and convert them to tiers. Tested with synthetic data:

```python
import pandas as pd
frames = pd.read_csv("frames.csv", parse_dates=["captured_at"])   # image,bin_uid,captured_at
sensor = pd.read_csv("sensor.csv", parse_dates=["read_at"])       # bin_uid,read_at,distance_cm
j = pd.merge_asof(frames.sort_values("captured_at"), sensor.sort_values("read_at"),
                  left_on="captured_at", right_on="read_at", by="bin_uid",
                  direction="nearest", tolerance=pd.Timedelta("60s"))
EMPTY_CM, FULL_CM = 100.0, 10.0   # per bin model: lid-to-floor and lid-to-full distances
j["fill_pct"] = ((EMPTY_CM - j["distance_cm"]) / (EMPTY_CM - FULL_CM) * 100).clip(0, 100)
j["tier"] = pd.cut(j["fill_pct"], [-0.1, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4])
```

Frames without a reading within 60 s get no tier (NaN); label them by hand or leave them out.
- [ ] Feed the joined tiers into A1's labels (source 3 in the labelling guide), and evaluate the camera model against sensor truth per camera.

---

## 4. Data requirements

| Task | Data | Amount | Labelling effort |
|---|---|---|---|
| A2 to A4 | Frames with bin boxes and fill tiers | 150 minimum, 300+ target; 30+ per bad condition | About 10 s per box and 3 s per tier; physical checks for front-on cameras |
| B3 tuning | Continuous recordings with collection times | 2 to 3 days per camera | Write down collection times |
| B4/B5 | Empty-bin reference crops | 1 to 3 per bin (day, night/IR) | A few minutes per camera |
| B6 | Val split boxes | Same as A | None extra | to do |
| C2 | Polygons (or boxes converted with SAM) | 300 to 1000 frames | About 15 s per bin with SAM assist | to do |
| C3 | Bin crops per tier | 100+ per tier in train, 15+ in val and test | Comes from the A labels | to do |
| D2 | Sensor readings | 5 to 10 bins for 2 to 4 weeks | None (automatic) | to do |

## 5. Proposed targets

Agree these with operations during A4. They are starting points, not measurements.

| Metric | Target |
|---|---|
| Bin detection, automatic mode | precision and recall at least 0.95 (IoU 0.5) |
| Bin detection, fixed-camera mode | count accuracy 1.0 |
| Fill tier accuracy | at least 0.80 |
| Within one tier | at least 0.97 |
| FULL recall / precision | at least 0.95 / at least 0.85 |
| Fill MAE | at most 12 percentage points |
| Abstain rate (C3) | at most 0.10 |
| False collections (B3) | at most 1 per camera per week |

## 6. Risks

| Risk | Mitigation |
|---|---|
| Fill labels guessed from the outside of front-on bins | Ground-truth sources in the labelling guide; class 5 UNKNOWN |
| Near-duplicate frames leaking between splits | Camera-day grouping (A3); public data only in train |
| Stock photos unlike CCTV (compression, IR, angle) | Collect from the real cameras first |
| References captured while bins aren't empty make bins read EMPTY | Capture-then-verify rule (B5 Step 12); recapture after changes |
| Monotonic fill hides a partial collection | Tune `collection_drop_pct`; check against logged collections |
| Few FULL and EMPTY examples | Sample around collections; oversampling in `export_crops.py` |
| Public dataset licences | Record per dataset in `datasets/README.md` |
| People in CCTV frames | Images out of git, self-hosted labelling, check local law |
| CPU cost | The fill classifier adds a few ms per bin; a fine-tuned `yolo11s-seg` (C2) is far cheaper than YOLOE-L |
