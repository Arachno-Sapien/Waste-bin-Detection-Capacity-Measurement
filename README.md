# Waste Vision System

AI-powered trash bin occupancy monitor. Takes an image, video, webcam, or RTSP
feed, locates bins in the frame, segments the waste inside them, and reports a
fill percentage (0–100%) classified into **EMPTY / LOW / MEDIUM / NEARLY
FULL / FULL**.

Stack: Python 3.14, OpenCV, Ultralytics YOLO11 + YOLOE, Streamlit, NumPy.

---

## 1. What it does

| Capability | Status |
|---|---|
| Image / video / webcam / RTSP input | ✅ |
| Bin detection — open-vocabulary (YOLOE, text-prompted) | ✅ default |
| Bin detection — HSV colour heuristic | ✅ automatic fallback |
| Bin detection — fine-tuned custom model | plumbing ready, no model trained yet |
| Waste detection — YOLO segmentation + colour-inversion | ✅ |
| Overflow detection (waste above the rim) | ✅ |
| Manual bin ROI override (UI) | ✅ |
| Persistent bin tracking across frames (IoU) | ✅ |
| CSV export, snapshot export | ✅ |

## 2. Directory layout

```
waste_vision_system/
├── main.py                 # Entry point: no args → Streamlit UI; --headless --source X → CLI
├── run_app.bat              # Windows launcher (python main.py)
├── check_requirements.bat   # Verifies Python/pip before first run
├── requirements.txt
├── verify_system.py         # Test suite — see §6
├── yolo11n-seg.pt            # Waste detection model (COCO-pretrained)
├── yoloe-11l-seg.pt          # Bin detection model (open-vocabulary)
├── mobileclip_blt.ts         # Text encoder for YOLOE's prompts
│
├── config/settings.py        # All tunables — one dataclass, see §5
├── detectors/
│   ├── bin_detector.py       # BinDetector — 4 strategies, see §3
│   └── waste_detector.py     # WasteDetector — YOLO + colour-inversion + overflow band
├── services/
│   ├── occupancy.py          # OccupancyEstimator — 3-factor fill calculation
│   ├── stream_handler.py     # Webcam/RTSP lifecycle
│   └── logger.py             # CSV logging
├── ui/
│   ├── app.py                 # Streamlit dashboard + process_frame() pipeline
│   └── components.py          # Sidebar widgets, cards, buttons
├── utils/
│   ├── drawing.py             # HUD overlay rendering
│   └── fps_counter.py
├── models/                    # Extra/backup model weights
├── tests/fixtures/            # Images the test suite reads (see §6)
└── exports/{csv,snapshots}/   # Runtime output
```

## 3. Bin detection — 4 strategies, tried in order

`BinDetector.detect()` (`detectors/bin_detector.py`) tries each in turn and
uses the first that applies:

1. **Manual ROIs** — operator-drawn boxes from the UI. Always wins when set;
   the deliberate override for a camera angle nothing else handles.
2. **Custom YOLO-seg model** — used if `Settings.bin_model_path` points to a
   model fine-tuned on a `bin` class. Not trained yet; this is where a future
   fine-tuned model plugs in with no other code changes.
3. **Open-vocabulary (YOLOE)** — locates bins from text prompts
   (`Settings.openvocab_prompts`) with **no bin-specific training**. This is
   the default strategy today. Runs at two input scales (`openvocab_scales`)
   and keeps whichever produces the cleaner, less-overlapping set of boxes —
   see `_detect_openvocab` / `_messiness`. Handles lighting/background cases
   the HSV heuristic below cannot.
4. **HSV colour segmentation** — legacy heuristic, matches bins by body
   colour (`BIN_HSV_RANGES` in `config/settings.py`). Kept only as an
   automatic fallback if the open-vocab pass finds nothing. Cannot separate a
   bin from same-coloured background (hedges, walls) and fails outright at
   night — do not rely on it as the primary strategy.

A greedy IoU tracker (`_SimpleTracker`) assigns persistent `Bin #N` IDs
across frames regardless of which strategy ran.

## 4. Full pipeline (per frame)

`process_frame()` in `ui/app.py` (mirrored headlessly in `main.py`):

1. `BinDetector.detect(frame)` → list of bins (bbox, interior mask, colour,
   rim top/bottom, and — for YOLOE detections — a `surface_mask` giving the
   bin's actual wall pixels).
2. For each bin: `WasteDetector.detect(...)` — YOLO segmentation on the bin
   crop, unioned with non-bin-colour pixel analysis. The search region is
   extended above the rim by `waste_overflow_band_ratio` (default 45% of bin
   height) so waste piled on top of a full bin — physically outside the bin
   box — isn't invisible to detection. A texture gate
   (`_gate_band_by_texture`) stops a flat wall or sky above a bin from being
   mistaken for overflow.
3. Build the wall/colour mask used for the aperture-focused area ratio via
   `services.occupancy.wall_mask_for()` — the single shared helper for this:
   `surface_mask` when the detector produced one (YOLOE — the real bin wall
   pixels), else an expanded HSV colour-range match as a fallback. `ui/app.py`,
   `main.py`, and `verify_system.py` all call it, so the test suite exercises
   the same wall-mask logic as the app rather than a separate copy that could
   drift out of sync.
4. `OccupancyEstimator.estimate(...)` — combines area ratio, vertical fill
   height, and overflow score into a fill % (see §5), with temporal
   smoothing across frames.
5. `draw_bin_overlay()` renders the HUD, `DataLogger.log_entry()` writes the
   CSV row, `draw_hud_header()` draws the FPS/bin-count strip.

## 5. Fill calculation — 3-factor model

`OccupancyEstimator` (`services/occupancy.py`) combines three signals:

- **Area ratio** (`area_weight`, default 0.55) — `waste_pixels / opening_pixels`,
  where `opening_pixels` = bin interior minus wall-colour pixels. Aperture-
  focused: measures how full the *visible opening* is, not the whole bounding
  box (which is dominated by the bin's solid front wall).
- **Vertical fill height** (`height_weight`, 0.25) — how high up the bin the
  topmost waste pixel reaches.
- **Overflow score** (`overflow_weight`, 0.20) — waste detected above the rim,
  in the overflow band described in §4. Tiered override: overflow ≥80 forces
  fill ≥85% (FULL), ≥50 forces ≥70% (NEARLY FULL) — a bin visibly overflowing
  should never classify as anything but full.

All weights, thresholds, and the fill-tier boundaries live in
`config.settings.Settings` — nothing here is hardcoded outside that one
dataclass. A per-bin rolling window (`smoothing_window`, default 5 frames,
median-smoothed) reduces flicker on video. Smoothing is only meaningful
across frames of the *same* scene, so it's gated by `Settings.smoothing_enabled`:
video/webcam/RTSP turn it on at stream start, single-image analysis turns it
off (a still has no temporal dimension to smooth over). Every new
image upload or stream start also resets bin tracking IDs and the smoothing
history (`BinDetector.reset_tracking()`, `OccupancyEstimator.reset()`), so
fill numbers from one scene never bleed into the next.

**Known limitation:** fill accuracy is inherently capped by a single RGB
camera's inability to distinguish "trash resting on the bin" from "trash
behind the bin" — a depth question a 2D image can't fully answer. See the
project's accuracy-improvement plan for how this is being addressed
(dataset-driven fine-tuning, a learned fill classifier, optionally a depth
signal or ultrasonic sensor).

## 6. Development

### Install

```bash
pip install -r requirements.txt
```

`requirements.txt` includes everything except two model weight files that
Ultralytics downloads automatically on first run into this folder:
`yoloe-11l-seg.pt` (~70MB) and `mobileclip_blt.ts` (~600MB, the YOLOE text
encoder). First run will be slow while these download; after that they're
cached locally.

### Run

```bash
python main.py                              # Streamlit dashboard (default)
python main.py --headless --source img.jpg  # Headless: single image
python main.py --headless --source vid.mp4  # Headless: video file
python main.py --headless --source 0        # Headless: webcam
```

Or double-click `run_app.bat` on Windows.

### Test

```bash
python verify_system.py
```

Six tests, run independently (one failure doesn't block the rest):

| # | Test | Validates |
|---|---|---|
| 1 | Multi-bin separation | Adjacent bins correctly separated on `tests/fixtures/001.jpg` (ground truth: 4 bins) |
| 2 | Pipeline structural sanity | Synthetic image, no crashes |
| 3 | Headless app | End-to-end processing on a real image |
| 4 | Manual ROI path | Manual boxes → detection → clear → auto-detection resumes |
| 5 | Overflow override | Tiered fill-minimum logic |
| 6 | Additional fixtures | Every other image in `tests/fixtures/` processes without error |

A test that needs a fixture image reports **SKIP**, not a silent pass, if
that image is missing — exit code 2 means "incomplete", not "passed", so a
stripped-down checkout can't show an all-green board that verified nothing.

### Adding a fixture image

Drop it in `tests/fixtures/`. Test 6 picks it up automatically; no code
change needed unless you also want an exact-count assertion for it (add a
dedicated test like Test 1 for that).

### Configuration

Every tunable — model paths, detection thresholds, fill weights, HSV ranges,
performance settings — is a field on `Settings` in `config/settings.py`. To
try a fine-tuned bin model once one exists, set `bin_model_path`; strategy
priority (§3) picks it up with no other code changes.
