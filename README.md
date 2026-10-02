# Waste Vision System

AI-powered trash bin occupancy monitor. Takes an image, video, webcam, or RTSP
feed, locates bins in the frame, segments the waste inside them, and reports a
fill percentage (0–100%) classified into **EMPTY / LOW / MEDIUM / NEARLY
FULL / FULL**.

Stack: Python 3.11+, OpenCV, Ultralytics YOLO11 + YOLOE, Streamlit, NumPy.

---

## 0. Setup (clone from GitHub)

**Prerequisites:** Python 3.11+ (developed on 3.14), Git, and ~750MB free
disk + internet access for the first run (model download, see below).

```bash
git clone https://github.com/Arachno-Sapien/Waste-bin-Detection-Capacity-Measurement.git
cd Waste-bin-Detection-Capacity-Measurement/waste_vision_system

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt

python main.py                # Streamlit dashboard
```

Two model weight files are **not** in the repo (too large for GitHub) —
`.gitignore`d and fetched by `ultralytics` automatically the first time you
analyse an image or frame (models load lazily, not at launch):

- `yoloe-11l-seg.pt` (~68MB, bin detector)
- `mobileclip_blt.ts` (~572MB, YOLOE's text encoder)

The first run will pause while these download into `waste_vision_system/`
(needs internet access). It then encodes the bin prompts once and saves
`yoloe-11l-seg-bins.pt` (~68MB, also `.gitignore`d) — the YOLOE weights with
the prompts baked in. Every later run loads that file directly and never
touches the text encoder, so after the first run `mobileclip_blt.ts` can be
deleted. It is needed again only on a fresh clone's first run, or if you
change `Settings.openvocab_prompts`: the baked file's classes then no longer
match, so the next run re-bakes automatically (re-downloading the encoder if
you deleted it).
`yolo11n-seg.pt` (waste detector, ~6MB) is small enough to ship in the repo,
so no download is needed for that one.

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
├── check_requirements.bat   # Checks Python/pip, installs requirements.txt
├── requirements.txt
├── verify_system.py         # Test suite — see §6
├── yolo11n-seg.pt            # Waste detection model (COCO-pretrained)
├── yoloe-11l-seg.pt          # Bin detection model (open-vocabulary) — auto-downloaded
├── yoloe-11l-seg-bins.pt     # Same, prompts baked in — generated on first run
├── mobileclip_blt.ts         # Text encoder — auto-downloaded; only used to (re-)bake the prompts
│
├── config/settings.py        # All tunables — one dataclass, see §5
├── detectors/
│   ├── bin_detector.py       # BinDetector — 4 strategies, see §3
│   └── waste_detector.py     # WasteDetector — YOLO + colour-inversion + overflow band
├── services/
│   ├── occupancy.py          # OccupancyEstimator — 3-factor fill calculation
│   ├── stream_handler.py     # Webcam/RTSP lifecycle
│   └── logger.py             # CSV logging (appends to a per-session file)
├── ui/
│   ├── app.py                 # Streamlit dashboard + process_frame() pipeline
│   └── components.py          # Sidebar widgets, cards, buttons
├── utils/
│   ├── drawing.py             # HUD overlay rendering
│   └── fps_counter.py
├── models/                    # Spare weights (yolov8n-seg.pt) — nothing in the code loads from here
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
   the default strategy today. On a scene's first frame it runs at two input
   scales (`openvocab_scales`) and keeps whichever produces the cleaner,
   less-overlapping set of boxes — see `_detect_openvocab` / `_messiness`.
   Later frames reuse that scale (1.3–5× faster per frame) until the next
   upload or stream start resets it — or until it finds no bins, which
   triggers a fresh comparison. A single stream that pans/zooms or cuts
   between cameras keeps its first scene's scale. Handles lighting/background
   cases the HSV heuristic below cannot.
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
5. `draw_bin_overlay()` renders the HUD, `DataLogger.log_entry()` buffers the
   CSV row (appended to the session CSV every `csv_flush_interval` rows and on
   export), `draw_hud_header()` draws the FPS/bin-count strip.

## 5. Fill calculation — 3-factor model

`OccupancyEstimator` (`services/occupancy.py`) combines three signals:

- **Area ratio** (`area_weight`, default 0.55) — `waste_pixels / opening_pixels`,
  where `opening_pixels` = bin interior minus wall-colour pixels. Aperture-
  focused: measures how full the *visible opening* is, not the whole bounding
  box (which is dominated by the bin's solid front wall).
- **Vertical fill height** (`height_weight`, 0.25) — how high up the bin the
  topmost waste row reaches. Looks only at the bin's own columns; a row counts
  once waste covers `height_row_min_frac` (default 8%) of the bin's width.
- **Overflow score** (`overflow_weight`, 0.20) — waste detected above the rim,
  over the bin's own columns, in a band `waste_overflow_band_ratio` × bin
  height tall (the same band §4's waste search covers), so the score doesn't
  depend on frame width or where the bin sits in the frame. Tiered override:
  overflow ≥80 forces fill ≥85% (FULL), ≥50 forces ≥70% (NEARLY FULL) — a bin
  visibly overflowing should never classify as anything but full.

The weights, row-width fraction, overflow band, smoothing window, and fill-tier
boundaries live in `config.settings.Settings`. The model's own fixed constants
are in `services/occupancy.py`, not `Settings`: the area-ratio fallback
(waste ÷ interior × 2.5, used when there is no wall mask or the visible
opening is under 5% of the interior), the overflow-score cutoffs (waste over
1% / 5% / 15% of the band → 50 / 80 / 100), and the override tiers above.
`classify_fill` rounds the fill % to a whole number before matching the tiers
(20.4% → EMPTY, 20.6% → LOW), which also matches the `{:.0f}%` the UI shows.
A per-bin rolling window (`smoothing_window`, default 5 frames,
median-smoothed) reduces flicker on video. Smoothing is only meaningful
across frames of the *same* scene, so it's gated by `Settings.smoothing_enabled`:
video/webcam/RTSP turn it on at stream start, single-image analysis turns it
off (a still has no temporal dimension to smooth over). Every new
image upload or stream start also resets bin tracking IDs, the cached
open-vocab scale and the smoothing history (`BinDetector.reset_tracking()`,
`OccupancyEstimator.reset()`), so fill numbers from one scene never bleed
into the next.

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
`yoloe-11l-seg.pt` (~68MB) and `mobileclip_blt.ts` (~572MB, the YOLOE text
encoder). First run will be slow while these download and the prompts are
baked into `yoloe-11l-seg-bins.pt`; after that, model load skips the text
encoder entirely (see §0).

Python 3.11+ is required (the code uses `contextlib.chdir`). Ultralytics
already installs `opencv-python`; don't also install
`opencv-python-headless` — both provide `cv2`, and if the GUI-less build wins,
`cv2.imshow()` raises in headless mode (use `--no-display` to avoid the
window entirely).

### Run

```bash
python main.py                              # Streamlit dashboard (default)
python main.py --headless --source img.jpg  # Headless: single image
python main.py --headless --source vid.mp4  # Headless: video file
python main.py --headless --source 0        # Headless: webcam
```

Or double-click `run_app.bat` on Windows.

Headless runs and the dashboard's video upload process every `frame_skip`-th
frame (default 3: frames 1, 4, 7, …) — frame 1 is always included, so a single
image is still analysed. With `--output out.mp4`, the annotated video contains
only the processed frames, written at source fps ÷ `frame_skip` so playback
stays real-time. Readings are appended to `exports/csv/session_<timestamp>.csv`
every `csv_flush_interval` (100) rows and on export (dashboard Export button,
end of a headless run); that file holds the full session, while memory keeps
only the last 200 rows after each flush (the dashboard's Audit Log shows the
latest 50).

### Test

```bash
python verify_system.py
```

Eight tests, run independently (one failure doesn't block the rest):

| # | Test | Validates |
|---|---|---|
| 1 | Multi-bin separation | Adjacent bins correctly separated on `tests/fixtures/001.jpg` (ground truth: 4 bins) |
| 2 | Pipeline structural sanity | Synthetic image, no crashes |
| 3 | Headless app | End-to-end processing on a real image |
| 4 | Manual ROI path | Manual boxes → detection → clear → auto-detection resumes |
| 5 | Overflow override | Tiered fill-minimum logic |
| 6 | Additional fixtures | Every other image in `tests/fixtures/` processes without error |
| 7 | Occupancy invariants | Fill tiers have no gaps; overflow score is independent of frame size and bin position |
| 8 | Scale caching | Frame 2 of a scene runs only the cached YOLOE scale, with identical boxes; reset clears it |

A test that needs a fixture image reports **SKIP**, not a silent pass, if
that image is missing — exit code 2 means "incomplete", not "passed", so a
stripped-down checkout can't show an all-green board that verified nothing.

### Adding a fixture image

Drop it in `tests/fixtures/`. Test 6 picks it up automatically; no code
change needed unless you also want an exact-count assertion for it (add a
dedicated test like Test 1 for that).

### Configuration

Nearly every tunable — model paths, detection thresholds, fill weights, HSV
ranges, performance settings — lives in `config/settings.py` (a field on
`Settings`, or for the HSV ranges the module-level `BIN_HSV_RANGES`); the few
fixed constants in the occupancy model are listed in §5. Default model paths
are absolute (the project folder), so launching from another working
directory doesn't re-download weights. To try a fine-tuned bin model once one
exists, set `bin_model_path`; strategy priority (§3) picks it up with no
other code changes.
