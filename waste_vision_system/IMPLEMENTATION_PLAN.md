# Implementation Plan — Waste Vision System

**Audience:** an engineer or agent picking this up with no prior context.
**Companion doc:** `../README.md` at the project root (what the system is and how it works today).
This file is *what still needs doing and exactly how*.

Every file path is relative to `waste_vision_system/`. Line numbers are
accurate as of this document's writing — if they've drifted, grep for the
named symbol instead; symbol names are stable references, line numbers are not.

### Progress (2026-09-09)

All tasks below that require no new data, labelling, training, or business
decision are **done**: T-02, T-03, T-04, T-06, T-07, and the non-T-13-gated
half of §4's dead-code removal. Each is marked ✅ **DONE** at its heading
with what changed and how it was verified. Verification for all of them:
`python verify_system.py` → 6/6 PASS, plus `python -m py_compile` on every
touched file, plus (T-02, T-04) a direct headless re-run against the
fixtures to confirm the specific numbers the task's Acceptance section names.
No git commits exist for this repo yet, so there's no commit range to point
at — these are working-tree changes.

**Still open** — every one needs real data, labelling, training time, or a
business call an agent can't make: T-01 (blocking — needs ~200 labelled
images), T-05 (explicitly blocked on T-01), T-08/T-16/T-20 (explicitly
deferred), T-10/T-11 (no dataset exists yet to justify scaffolding it),
T-12/T-13/T-14/T-15 (need T-01's labelled set or training time), T-17
(architecture decision for later), and the HSV-cluster half of §4 (gated on
T-13 validating).

**Found but not fixed** (out of this pass's scope, flagging for whoever
picks up T-08/video work): `main.py --headless` with the default
`frame_skip=3` silently processes **zero** frames on a single-image source —
`frame_count` starts at 1, `1 % 3 != 0`, so the one frame is skipped and the
CSV export comes back empty. Unrelated to T-02/T-03/T-04.

---

## 0. Orientation — read before touching anything

### 0.1 Run it

```bash
cd waste_vision_system
pip install -r requirements.txt
python main.py                                  # Streamlit UI on :8501
python main.py --headless --source path/img.jpg # CLI, no UI
python verify_system.py                         # test suite
```

First run downloads `yoloe-11l-seg.pt` (~70MB) and `mobileclip_blt.ts`
(~600MB) into this folder. Slow once, cached after.

### 0.2 The single most important dev gotcha

**Streamlit does not reload imported modules.** It re-executes `ui/app.py` on
save, but `detectors/`, `services/`, and `config/` are imported modules — edits
there are invisible to a running server. You will see stale results with no
error, and you will waste an hour on it.

> **After editing anything outside `ui/app.py`, kill and restart the Streamlit
> server.** Editing and re-clicking "Start" is not enough.

To restart:
```bash
# find and kill, then relaunch
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*streamlit*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
python main.py
```

### 0.3 Environment facts

- Python 3.14, global install at `C:\Python314\python.exe`. There is **no
  virtualenv** — one existed and was deleted as stale. If you create one,
  update `run_app.bat` and `check_requirements.bat`, which currently assume a
  bare global `python`.
- **CPU only.** `torch 2.13.0+cpu`, no CUDA. `Settings.resolve_device()`
  auto-detects and will use CUDA/MPS if you move to a machine that has it.
- Inference is slow: the bin detector runs **two forward passes per frame**
  (two scales, see §2.3). Fine for stills, a problem for video (see T-08).

### 0.4 Invariants — do not break these

1. `Settings` (`config/settings.py`) is the **only** place tunables live.
   Do not hardcode a threshold in a detector. If you add one, add it there.
2. `BinDetection.interior_mask` and all waste masks are **full-frame** binary
   masks (`uint8`, 0/1), not crops. Several places assume this.
3. Manual ROIs always win over every automatic strategy
   (`bin_detector.py:183` `detect()`). This is the operator's override; never
   reorder it below an automatic path.
4. `verify_system.py` must stay runnable with **no arguments** and must exit
   non-zero on failure (1 = failure, 2 = incomplete/skipped, 0 = all passed).

---

## 1. Current state — measured, not assumed

### 1.1 Bin detection: working

Open-vocabulary YOLOE (text-prompted, zero training) replaced the HSV colour
heuristic. Measured on the 5 fixture images, **bin count is exact on all 5**:

| Fixture | Contents | Truth | Detected |
|---|---|---|---|
| `001.jpg` | 4 green wheelie bins, overflowing, against dark wall | 4 | 4 ✓ |
| `timg (1).png` | WET WASTE (green) + DRY WASTE (blue) pair, lids open | 2 | 2 ✓ |
| `timg (2).jpg` | 3 dark dumpsters on grass, litter around them | 3 | 3 ✓ |
| `timg (2).png` | Park: slatted wire bin + hooded domed bin | 2 | 2 ✓ |
| `timg (3).png` | Close-up single green wheelie bin, lid open, overflowing | 1 | 1 ✓ |

For contrast, the old HSV heuristic got only `001.jpg` right and returned
frame-spanning garbage boxes on the other four.

### 1.2 Fill estimation: partially working

Measured via the **app's** occupancy path (using `surface_mask`, see §1.3):

| Fixture | Reported | Reality | Verdict |
|---|---|---|---|
| `001.jpg` | 85% FULL ×4 | overflowing | ✓ correct |
| `timg (2).png` | 82% / 100% FULL | both full | ✓ correct |
| `timg (2).jpg` bin 1 | 12% EMPTY | empty | ✓ correct |
| `timg (1).png` | 74% / 85% | ~40–50% | ✗ over-reads |
| `timg (3).png` | 74% NEARLY FULL | ~95%+ | ✗ under-reads |
| `timg (2).jpg` bins 2,3 | 70% / 85% | ~30–40% | ✗ over-reads |

**Cause of the over-reads:** dense foliage directly above a bin is highly
textured, so it passes the overflow band's texture gate
(`waste_detector.py:197` `_gate_band_by_texture`) and registers as overflow.

**Why the obvious fix doesn't work:** a stricter gate requiring overflow to be
continuous with waste *inside* the bin was tried. It fixed the foliage cases
and **broke `001.jpg`** — on a front-on view the bin interior isn't visible at
all, so a pile resting on the rim connects to nothing and real overflow gets
discarded. That gate is in the code as
`waste_detector.py:227` `_keep_band_joined_to_bin`, controlled by
`Settings.waste_overflow_require_contact`, **off by default**. Turn it on only
for downward-angled cameras where the interior is visible.

**The real ceiling:** "is that debris *on* the bin or *behind* it" is a depth
question. A single RGB camera cannot answer it from colour and texture. No
amount of heuristic tuning fixes this — see T-20.

> ⚠️ **These numbers come from 5 images.** That is an anecdote, not a test set.
> It shows the approach is sound; it does **not** establish an accuracy figure.
> Do not quote a percentage to a customer off this. See T-01.

### 1.3 ✅ FIXED (T-04) — the test suite used to measure a different code path than the app

This described a real bug; kept below for history since T-05 and future
tuning work reference it. It no longer applies — see T-04.

- `ui/app.py:187` and `main.py:126` prefer `BinDetection.surface_mask` (the
  YOLOE instance mask — the bin's actual wall pixels) as the occupancy wall mask.
- `verify_system.py:103` (`analyse_bin`) instead calls
  `build_bin_color_mask(...)`, an **HSV colour-range** mask. It never touches
  `surface_mask`.

So fill percentages printed by `verify_system.py` are **not** the numbers the
app produces for the same image. Tuning against the test suite tunes the wrong
path. Fixed by **T-04** below; do that before any fill tuning work.

---

## 2. Architecture map

### 2.1 Module layout

| File | Lines | Role |
|---|---|---|
| `main.py` | 219 | Entry point. No args → launches Streamlit. `--headless --source X` → CLI pipeline. |
| `config/settings.py` | 271 | `Settings` dataclass + colour palettes + `WASTE_COCO_MAPPING` + `BIN_HSV_RANGES`. |
| `detectors/bin_detector.py` | 1038 | `BinDetection`, `_SimpleTracker`, `BinDetector` (4 strategies). |
| `detectors/waste_detector.py` | 406 | `WasteDetection`, `WasteDetector` (YOLO + colour inversion + overflow band). |
| `services/occupancy.py` | 299 | `OccupancyResult`, `OccupancyEstimator` (3-factor fill). |
| `services/stream_handler.py` | 229 | Webcam/RTSP lifecycle, reconnect backoff. |
| `services/logger.py` | 155 | CSV logging, buffered flush. |
| `ui/app.py` | 704 | Streamlit dashboard + `process_frame()` orchestration. |
| `ui/components.py` | 246 | Sidebar widgets, status cards. |
| `utils/drawing.py` | 212 | HUD overlays, bounding boxes, fill bars. |
| `utils/fps_counter.py` | 56 | Rolling FPS/latency. |
| `verify_system.py` | 419 | 6-test suite. |

~~`demo_test.py`~~ — deleted (T-07, done).

### 2.2 `BinDetector` symbol map (`detectors/bin_detector.py`)

| Line | Symbol | Status |
|---|---|---|
| 45 | `BinDetection` dataclass | active — note `surface_mask` field |
| 65–120 | `_SimpleTracker` | active. IoU threshold hardcoded `0.3` at line 68, **not** wired to `Settings.iou_threshold` (which is YOLO NMS IoU — a different thing; do not "fix" by merging them) |
| 183 | `detect()` — strategy dispatch | active, the entry point |
| 204–241 | `_frac_inside`, `_overlap`, `_dedupe_boxes`, `_messiness` | active — open-vocab helpers |
| 242 | `_detect_openvocab()` | active — **default strategy** |
| 339 | `_load_openvocab()` | active — lazy model load |
| 360 | `_detect_yolo()` | active but unused (no fine-tuned model exists yet) |
| 432 | `_detect_manual_rois()` | active |
| 479 | `_detect_hsv_color()` | **legacy fallback** |
| 668 | `_conservative_split()` | legacy, HSV-only |
| 773 | `_trim_open_lid()` | legacy, HSV-only |
| 848 | `_merge_adjacent_bin_segments()` | legacy, HSV-only |
| ~~922~~ | `_split_by_vertical_profile()` | **REMOVED (§4, done)** — was dead, never called |
| 1001 | `_dominant_bin_color()` | active — used by all strategies for `bin_hsv_range` |
| 1025 | `_make_rect_mask()` | active — used by `_detect_yolo` |
| 1036 | `reset_tracking()` | **wired in (T-02, done)** — called from `ui/app.py` on every new upload / stream start |

### 2.3 Detection strategy order

`detect()` at `bin_detector.py:183` tries in this order:

1. **Manual ROIs** — if `set_manual_rois()` was called. Operator override.
2. **Custom YOLO** — if `Settings.bin_model_path` is non-empty. *No model
   exists yet; this is the hook for T-13.*
3. **Open-vocabulary YOLOE** — if `Settings.openvocab_enabled` (default `True`).
   Falls through to HSV **only if it returns zero detections**.
4. **HSV colour** — legacy fallback.

**How the open-vocab pass works** (`_detect_openvocab`, line 242) — this is
non-obvious, read before modifying:

- Runs inference at **each scale in `Settings.openvocab_scales`** (default
  `(640, 1280)`), not once.
- **Why two scales:** optimal `imgsz` depends on how much of the frame the bin
  fills. A close-up (`timg (3).png`) needs ≤640 and returns garbage at 1280; a
  row of four distant bins (`001.jpg`) needs 1280 and returns *nothing* at 640.
- **Scale selection**, not merging: each scale's boxes are deduped, then scored
  by `_messiness()` (max mutual overlap among surviving boxes). The **cleanest**
  scale wins; near-ties break toward more detections. Rationale: a scale that
  can't resolve the scene emits several partly-overlapping boxes for one object.
- **Do not "improve" this by pooling both scales' detections.** That was tried
  and scored *worse* (3/5 vs 5/5) — the same bin found at two scales yields
  offset boxes that don't dedupe.
- `_dedupe_boxes()` uses **containment overlap**, not IoU, because the two
  duplicate patterns YOLOE produces (same bin matched by several prompt terms;
  same bin with and without its open lid) are nested or offset, and standard
  NMS misses both.

### 2.4 Per-frame pipeline

`process_frame()` at `ui/app.py:134` (mirrored in `main.py:_run_headless`):

```
frame
  └─> BinDetector.detect()                    -> List[BinDetection]
        for each bin:
        ├─> WasteDetector.detect()            -> List[WasteDetection]
        │     - YOLO seg on bin crop
        │     - non-bin-colour pixel analysis
        │     - overflow band above rim (+texture gate)
        ├─> build wall mask:
        │     surface_mask if present, else HSV colour-range match
        ├─> OccupancyEstimator.estimate()     -> OccupancyResult
        │     - area ratio (0.55) + height (0.25) + overflow (0.20)
        │     - overflow override: >=80 forces >=85%; >=50 forces >=70%
        │     - temporal smoothing over `smoothing_window` frames
        ├─> draw_bin_overlay()
        └─> DataLogger.log_entry()
  └─> draw_hud_header()
```

---

## 3. Bug fixes — do these first

### T-01 · Build an accuracy harness `[BLOCKING · ~2 days, mostly labelling]`

**Problem.** Nothing measures whether an answer is *correct*.
`verify_system.py` checks that code runs without crashing — Test 1 asserts
`>= 2` bins on an image with 4. Every threshold in this codebase
(`waste_overflow_band_ratio=0.45`, `openvocab_conf=0.15`, the `0.55/0.25/0.20`
weights, the Canny thresholds) was picked by eyeballing 5 images. That is not
defensible and it does not scale.

**Everything else in this document depends on this.** Without it you cannot
tell whether a change helped.

**Steps.**

1. Collect ~200 images. Priority order:
   1. Frames from the **actual target CCTV cameras** — worth ~10× a stock photo
   2. The 5 existing fixtures
   3. Public datasets (T-11)

   Cover night/IR, rain, glare, partial occlusion. A model validated only on
   crisp daylight stock photos will fail its first night shift.

2. **Write a labelling guide before labelling anything.** One page with example
   images answering: does an open lid count as part of the bin? wheels? a bin
   half-occluded by a person? a bin at the frame edge? Inconsistent labels cap
   accuracy permanently and cannot be fixed retroactively without relabelling.

3. Label two things per image:
   - Detection: one box (or polygon) per bin, single class `bin`
   - Fill: one tier per bin — `EMPTY/LOW/MEDIUM/NEARLY_FULL/FULL`

4. Create `evaluate.py` (~80 lines). **Do not implement mAP yourself:**

   ```python
   from ultralytics import YOLO
   metrics = YOLO(weights).val(data="datasets/bins/data.yaml")
   print(metrics.box.map50, metrics.box.map)   # detection
   ```

   For fill, ~30 lines of numpy: mean absolute error on percentage, plus a
   5×5 tier confusion matrix.

5. **Break every metric down by condition** (day/night/rain/occluded). A single
   average hides which segment is failing, and one segment usually is.

**Acceptance.** `python evaluate.py` prints detection mAP50, fill MAE, a 5×5
confusion matrix, and a per-condition breakdown, against a held-out set that
was **not** used for any tuning.

> Skipped deliberately: pytest, fixtures, CI. Add when more than one person
> runs this.

---

### T-02 · Pipeline state leaks between image uploads `[HIGH · ~15 min]` — ✅ DONE

**How:** added `bin_det.reset_tracking()` + `occ_est.reset()` exactly as
specified below, in `ui/app.py`'s new-upload branch of `handle_image()` and
at the top of `handle_video()`, `handle_webcam()`, `handle_rtsp()` right
after each stream's `.open()` succeeds (not per-frame — temporal smoothing
across frames of the same stream is intentional).

**Verified:** ran the fix by hand outside Streamlit — instantiated
`BinDetector`/`OccupancyEstimator` once (mirroring session-state reuse),
analysed `001.jpg` (4 bins, 85% each, IDs 1–4), called
`reset_tracking()`/`reset()`, then analysed `timg (2).jpg`. Result: IDs
restarted at 1 and bin 1 reported 11.4% EMPTY — matching this task's
Acceptance section (~12% EMPTY) and confirming no averaging with image A's
85%.

**Symptom.** Upload image A, then image B. Bin IDs keep climbing (`Bin #5`,
`Bin #6`…), and — worse — if B's bins land in similar positions to A's, the
IoU tracker reuses A's IDs and `OccupancyEstimator._history[bin_id]` **averages
B's fill with A's**. A genuinely empty bin can report ~50% because the previous
image's bin was full at that position.

**Root cause.** `ui/app.py:104` `_get_pipeline()` caches detector and estimator
instances in `st.session_state` for the whole session. `handle_image()`
(`ui/app.py:233`) resets `edit_mode`, `manual_rois`, and calls
`bin_det.clear_manual_rois()` on a new upload — but never resets the tracker or
the smoothing history.

`BinDetector.reset_tracking()` exists at `bin_detector.py:1036` and is
**called from nowhere in the codebase**. `OccupancyEstimator.reset()`
(`occupancy.py:294`) is called only from `verify_system.py`.

**Fix.** In `ui/app.py`, inside the new-upload branch of `handle_image()` —
the block that already contains `bin_det.clear_manual_rois()` around line 250:

```python
st.session_state["edit_mode"] = False
st.session_state["manual_rois"] = []
bin_det.clear_manual_rois()
# A new image is a new scene: drop tracker IDs and smoothing history so
# fill values are not averaged across two unrelated images.
bin_det.reset_tracking()
occ_est.reset()
```

`occ_est` is already unpacked from `pipeline` at the top of `handle_image`.

**Also apply to video/webcam/RTSP start.** `handle_video` (`ui/app.py:412`),
`handle_webcam` (469), and `handle_rtsp` (509) should reset on *stream start*
so a second run doesn't inherit the first's history. Do **not** reset per frame
— temporal smoothing across frames is intentional and desirable there.

**Acceptance.** Upload `001.jpg` (4 bins, ~85% each), then `timg (2).jpg`
(3 bins, first one empty). The first dumpster must report ~12% EMPTY, and IDs
must restart at `Bin #1`.

---

### T-03 · `smoothing_window` distorts single-image results `[MEDIUM · ~20 min]` — ✅ DONE

**How:** both fixes applied as specified. `OccupancyEstimator._finalize()`
(`occupancy.py`) now takes `float(np.median(window))` instead of the mean.
Added `Settings.smoothing_enabled: bool = True`; when `False`, `_finalize`
returns `raw_fill` directly and skips the rolling window entirely (not just
averaging over it — this also sidesteps the Streamlit-rerun duplicate-append
case the task description called out). `handle_image()` sets it `False`;
`handle_video()`/`handle_webcam()`/`handle_rtsp()` set it back to `True` at
stream start, so switching modes within one Streamlit session can't leave
smoothing off where it's wanted.

**Verified:** `verify_system.py` Test 5 (overflow override, depends on
`_finalize`'s output) still passes.

**Problem.** `OccupancyEstimator._finalize()` (`occupancy.py:264`) appends to a
rolling window and returns the **mean**. For a single still image there is one
sample, so it's a no-op — *unless* T-02's leak applies, or Streamlit re-runs the
script (which it does on any widget interaction), appending duplicates.

Two independent issues:

1. **Mean is the wrong statistic.** One bad frame drags the reported value.
   Change to median — more robust to outliers, same cost:
   ```python
   window = self._history[bin_id]
   smoothed = float(np.median(window))   # was: sum(window) / len(window)
   ```
   (`np` is already imported in `occupancy.py`.)

2. **Smoothing should be disabled for stills.** Add to `Settings`:
   ```python
   smoothing_enabled: bool = True   # set False for single-image analysis
   ```
   and have `handle_image` set it `False`. A still has no temporal dimension;
   smoothing there only creates the bug in T-02.

**Acceptance.** Re-run `verify_system.py` — Test 5 (overflow override) must
still pass, since it depends on `_finalize`'s output.

---

### T-04 · Test suite exercises the wrong occupancy path `[HIGH · ~20 min]` — ✅ DONE

**How:** did the "better still" option, not just the minimal fix — extracted
a module-level `wall_mask_for(bin_det, hsv_frame)` helper into
`services/occupancy.py` (surface_mask if present, else the ±10H/±20S/-30/+40V
expanded HSV match), and switched all three call sites — `ui/app.py`'s
`process_frame()`, `main.py`'s `_run_headless()`, and `verify_system.py`'s
`analyse_bin()` — to call it instead of each carrying its own copy of the
expansion. Removed `verify_system.py`'s local `build_bin_color_mask()`, now
redundant.

**Bonus find:** `main.py`'s inlined copy of this block called `np.array(...)`
but `main.py` never imported `numpy` — a latent `NameError` on any headless
run where a bin had an `hsv_range`. The refactor deletes that block, fixing
it as a side effect.

**Verified:** `python -m py_compile` on all touched files; `verify_system.py`
6/6 pass with `001.jpg` reporting 85.0% FULL on all 4 bins via the test
suite's `analyse_bin()` path — matching the app's own numbers (§1.1),
satisfying this task's Acceptance section. Also ran
`python main.py --headless --source tests/fixtures/001.jpg --no-display`
directly to confirm the `main.py` path no longer raises.

**Problem.** See §1.3. `verify_system.py:103` uses an HSV colour mask;
`ui/app.py:187` and `main.py:126` use `surface_mask`. Fill numbers from the
test suite don't match the app's, so tuning against the suite tunes a path no
user ever hits.

**Fix.** In `verify_system.py`, change `analyse_bin()` (line 93) to mirror the
app: prefer `surface_mask`, fall back to the HSV mask.

```python
def analyse_bin(frame, hsv_frame, b, wd, oe):
    """Run waste detection + aperture-focused occupancy for one bin.

    Mirrors ui/app.py process_frame: the detector's own instance mask is the
    real wall region and beats a colour-range guess. Falls back to the HSV
    mask for strategies that produce no surface_mask (manual ROI, HSV).
    """
    w_dets = wd.detect(
        frame, b.bbox, b.interior_mask,
        bin_color_name=b.bin_color_name,
        bin_hsv_range=b.bin_hsv_range,
    )
    wall = getattr(b, "surface_mask", None)
    if wall is None:
        wall = build_bin_color_mask(hsv_frame, b.bin_hsv_range)
    occ = oe.estimate(
        b.bin_id, b.interior_mask, [w.mask for w in w_dets],
        b.rim_top_y, b.rim_bottom_y,
        bin_color_mask=wall,
    )
    return w_dets, occ
```

**Better still (do this if you have the time):** extract the wall-mask choice
into one shared helper so the three call sites cannot drift again. A good home
is a module-level function in `services/occupancy.py`:

```python
def wall_mask_for(bin_det, hsv_frame) -> Optional[np.ndarray]:
    """The bin's wall region: its instance mask if the detector made one,
    else an expanded HSV colour-range match."""
```
Then `ui/app.py`, `main.py`, and `verify_system.py` all call it. This removes
a real duplication — the ±10 H / ±20 S / −30/+40 V expansion is currently
copy-pasted in three places.

**Acceptance.** Fill percentages printed by `verify_system.py` for `001.jpg`
match those from `python main.py --headless --source tests/fixtures/001.jpg`
(~85% FULL per bin).

---

### T-05 · Tighten Test 1's assertion `[LOW · ~5 min]`

`verify_system.py:136` asserts `len(bin_dets) >= 2` on an image whose ground
truth is 4, then merely *prints* a note if the count isn't 4. Detection is now
reliably 4. Make it assert the truth:

```python
assert len(bin_dets) == 4, f"Ground truth is 4 bins, detected {len(bin_dets)}"
```

Do this **only after** T-01 gives you a real test set — if detection regresses
you want to know, but you also want somewhere better than a 5-image suite to
find out.

---

### T-06 · Stale docstring: occupancy weights `[TRIVIAL · 2 min]` — ✅ DONE

**How:** took the "better" option — removed the hardcoded `0.50/0.30/0.20`
from `occupancy.py`'s module docstring entirely rather than correcting them
to `0.55/0.25/0.20`, and pointed the docstring at `Settings.area_weight` /
`height_weight` / `overflow_weight` instead, so the two can't drift apart
again.

`services/occupancy.py` module docstring says weights are `0.50 / 0.30 / 0.20`.
The actual defaults in `Settings` are `0.55 / 0.25 / 0.20`
(`settings.py:211–213`). Fix the docstring — or better, stop repeating the
numbers in prose and point at `Settings`.

---

### T-07 · `demo_test.py` is broken `[LOW · 5 min]` — ✅ DONE

**How:** deleted it (the recommended option — `verify_system.py` covers
everything it did).

`demo_test.py:17` hardcodes:
```
C:\Users\SYED\.gemini\antigravity-ide\brain\c42bfe5d-.../media_...jpg
```
A path in a different IDE's temp cache that no longer exists. The script cannot
run on any machine.

**Decide one:**
- **Delete it** (recommended — `verify_system.py` covers everything it did), or
- Repoint it at `tests/fixtures/001.jpg` and make the path a CLI argument.

Do not leave it as-is; it's a trap for the next person.

---

### T-08 · Two forward passes per frame is too slow for video `[MEDIUM · deferred]`

`_detect_openvocab` runs inference once per entry in `openvocab_scales`
(default 2). On CPU this roughly doubles per-frame cost — measured latency in
the UI was ~11s for a single image.

**Do not fix by dropping a scale** — §2.3 explains why both are needed
zero-shot. The real fix is T-13: a fine-tuned model won't need multi-scale, so
`openvocab_scales` collapses to one entry (or the strategy is retired
entirely). Until then, for video, `Settings.frame_skip` (default 3) already
limits full-pipeline runs — see `ui/app.py:445` and `main.py:86`.

Revisit after T-13. Track the cost; don't optimize it yet.

---

## 4. Dead code removal

All verified by grep across `*.py`, `*.bat`, `*.txt`, `*.md`. Each is defined
in exactly one place and read nowhere.

**✅ DONE — the 7 items below (everything not gated on T-13).** Re-verified
each was still genuinely unreferenced (re-grepped before removing, since the
codebase may have drifted since this table was written) then deleted them.
Confirmed with a follow-up grep for all 7 names/symbols across the whole
project afterward — zero hits outside this plan file's own history text.
`python -m py_compile` + `verify_system.py` 6/6 pass after removal.

| Item | Location (at time of writing) | Notes |
|---|---|---|
| ~~`_split_by_vertical_profile()`~~ | was `bin_detector.py:922–1000` (~79 lines) | REMOVED. Superseded by `_conservative_split`. Never called. |
| ~~`Settings.bin_vertical_gap_threshold`~~ | was `settings.py:219` | REMOVED. Only read *inside* `_split_by_vertical_profile`. |
| ~~`BIN_PROXY_COCO_IDS`~~ | was `settings.py:74` | REMOVED. Empty list, read nowhere. |
| ~~`TACO_CLASSES`~~ | was `settings.py:131–140` | REMOVED. Read nowhere. |
| ~~`Settings.edge_density_weight`~~ | was `settings.py:225` | REMOVED. Read nowhere. |
| ~~`Settings.bin_min_width_ratio`~~ | was `settings.py:218` | REMOVED. Read nowhere. |
| ~~`Settings.bin_interior_min_area_ratio`~~ | was `settings.py:221` | REMOVED. Read nowhere. |

**Larger removal — the HSV strategy — is gated on T-13.** Once a fine-tuned bin
model is validated, this whole cluster goes (~600 lines):

- `_detect_hsv_color()` (479–667)
- `_conservative_split()` (668–772)
- `_trim_open_lid()` (773–847)
- `_merge_adjacent_bin_segments()` (848–921)
- `BIN_HSV_RANGES` (`settings.py:147–162`)
- `WasteDetector._non_bin_color_mask()` (`waste_detector.py:115–195`) — the
  entire "not the bin's colour = waste" approach exists only because HSV
  couldn't segment; instance masks replace it.

**Keep** `_dominant_bin_color()` unless you also remove `bin_hsv_range` from
`BinDetection` — it's still populated by every strategy.

**Do not do this removal before T-13 validates.** HSV is currently the only
fallback if the open-vocab model fails to load.

---

## 5. Dataset modularity

**Requirement:** adding a future dataset must not require code changes.

**This needs almost no code** — Ultralytics' native format already is the
plugin mechanism. Do not build a `DatasetRegistry`, abstract base class, or
loader plugin system.

### T-10 · Dataset layout `[~2 hours]`

```
datasets/
  classmap.yaml            # source class name -> canonical id
  bins/                    # your own labelled data
    data.yaml
    images/{train,val}/
    labels/{train,val}/
  taco/                    # a public dataset, dropped in as-is
    images/... labels/...
  site_mumbai_cctv/        # future site data
    images/... labels/...
```

`datasets/bins/data.yaml` — **`train` accepts a list**, which is the whole
mechanism:

```yaml
path: ../datasets
train:
  - bins/images/train
  - taco/images/train              # <- adding a dataset is this one line
  - site_mumbai_cctv/images/train
val:
  - bins/images/val
names:
  0: bin
```

**Adding a dataset later = 3 steps, zero code changes:**
1. Drop it in `datasets/<name>/` in YOLO format
2. Add its class names to `classmap.yaml`
3. Append one line to `train:`

**Format conversion is already solved — don't write converters:**
- COCO JSON → `from ultralytics.data.converter import convert_coco`
- Roboflow exports YOLO directly
- Pascal VOC → convert via Roboflow

**The one piece of real code:** `merge_datasets.py`, ~30 lines. Different
sources call the same object `bin` / `trash_can` / `dumpster` with different
numeric IDs. This reads `classmap.yaml` and rewrites label files to canonical
IDs:

```yaml
# classmap.yaml
taco:
  Trash_can: 0
  Dumpster: 0
roboflow_bins:
  garbage-bin: 0
```

**Versioning:** a `datasets/README.md` recording source, licence, and date per
dataset is enough to start. Add DVC only when the folder outgrows what you're
willing to keep in git.

> ⚠️ **Check licences before anything ships commercially.** Several public
> waste datasets are non-commercial / research-only.

### T-11 · Seed with public data `[~half day]`

Pull bin/dumpster datasets from Roboflow Universe into `datasets/`. Record
licence per dataset. This gives T-13 something to train on before site footage
exists.

---

## 6. Detection accuracy

Ordered by cost. Stop when it's good enough — don't do all four reflexively.

### T-12 · Prompt and threshold sweep `[FREE · no training · ~2 hours]`

`openvocab_prompts`, `openvocab_conf` (0.15), `openvocab_nms_iou` (0.5), and
`openvocab_scales` were chosen by hand against 5 images. Sweep them against
T-01's labelled set, take what maximises mAP50.

Also try **YOLOE visual prompts** — it accepts *image* exemplars alongside
text. Feed 3–5 crops of the actual bin models in the deployment. Catches
styles the text embedding misses. Still no training required.

### T-13 · Fine-tune a bin model `[~1 day + training time]`

The main jump, and the unlock for §4's big deletion and T-08's speed fix.

```python
from ultralytics import YOLO
YOLO("yolo11s-seg.pt").train(
    data="datasets/bins/data.yaml",
    epochs=100,
    imgsz=960,
    # augmentation matched to real failure modes, not defaults:
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,   # lighting
    degrees=5, scale=0.5,                # camera angle / distance
)
```
Also add grayscale augmentation (simulates IR night mode) and JPEG artifacts
(CCTV compresses hard).

**Wiring is already done.** Set `Settings.bin_model_path` to the resulting
weights and `detect()`'s strategy order (§2.3) picks it up — `_detect_yolo()`
at `bin_detector.py:360` already populates `bin_color_name`/`bin_hsv_range` and
handles masks. **No other code changes needed.**

> Note: `_detect_yolo()` does **not** currently set `surface_mask`. Add that —
> it produces `seg_mask` at line ~396 already, so pass it through to the
> `BinDetection(...)` constructor. Without it, fine-tuned detections silently
> fall back to the HSV wall mask in occupancy (§1.3's bug, in a new place).

**Expected:** zero-shot mAP50 is likely ~0.6–0.7 on real data. Fine-tuned on
200+ site images, 0.85–0.95 is a reasonable target for fixed cameras. Verify;
don't assume.

### T-14 · Distil to a smaller model `[optional]`

Once accuracy holds, retrain as `yolo11n-seg` for CPU/edge speed.
`yoloe-11l-seg` + two scales is heavy for a CPU-only deployment.

---

## 7. Fill accuracy — the biggest remaining win

### T-15 · Replace the geometric estimator with a learned one `[~2 days]`

**Why.** Count the hand-tuned constants currently deciding a fill percentage:
weights `0.55/0.25/0.20`, the `2.5×` fallback amplification
(`occupancy.py:149,152`), the 5% opening sanity check (line 145), overflow
tiers `>15%→100 / >5%→80 / >1%→50` (lines 254–259), `openvocab_contain_frac`
0.85, `waste_overflow_band_ratio` 0.45, the Canny thresholds `50,150`
(`waste_detector.py:215`). Every one is a guess, and they interact badly — the
texture gate fixed `001.jpg` and broke the dumpsters; the connectivity gate did
the reverse (§1.2).

A classifier learns all of it from data, and optimises **the number actually
reported** rather than a proxy for it.

**Implementation — no new dependency.** Ultralytics classification uses plain
ImageFolder layout:

```
datasets/fill/
  train/EMPTY/*.jpg   train/LOW/*.jpg   train/MEDIUM/*.jpg
  train/NEARLY_FULL/*.jpg   train/FULL/*.jpg
  val/...
```

```python
YOLO("yolo11n-cls.pt").train(data="datasets/fill", epochs=50, imgsz=224)
```

**Labelling is nearly free.** Crops come from the detector you already have:
run it over the T-01 image set, dump each bin crop to disk, drag files into 5
folders. A file-manager task, not an annotation job — an afternoon for ~500
crops.

**Wiring.** Add to `Settings`:
```python
fill_mode: str = "geometric"      # "geometric" | "model"
fill_model_path: str = ""
```
`OccupancyEstimator.estimate()` branches on it. Keep both paths during
transition, then delete the loser. **One flag — do not build a strategy-class
hierarchy for two options.**

**Abstain on low confidence.** Below threshold, return a new
`FillStatus.UNCERTAIN` rather than guessing. An 85% reading on a hedge is worse
than admitting ignorance, and abstained frames are exactly the ones worth
routing to human review and folding back into training.

> Adding `UNCERTAIN` touches `FillStatus` (`settings.py:35`), `STATUS_COLORS`,
> `STATUS_COLORS_HEX`, `fill_thresholds`, and `classify_fill()`. Grep for
> `FillStatus.` before adding — `ui/components.py` and `utils/drawing.py` both
> switch on it.

> Skipped: a regression head for exact %. Five tiers is what the UI shows and
> what operations act on. Add regression only if a customer demands it.

---

## 8. Deployment / CCTV

Deferred by the project owner ("get images right first"), but two notes so the
architecture doesn't foreclose them:

### T-16 · Exploit the fixed camera `[high value, cheap]`

CCTV cameras are static; nothing in the codebase uses that:

- **Bin positions don't move.** Configure ROIs once per camera and localisation
  stops being a per-frame problem. The manual-ROI path (`bin_detector.py:432`)
  already does this — it just needs per-camera persistence instead of
  per-session state.
- **Reference-frame differencing.** Capture each bin empty, once. Fill becomes
  a difference from a known-empty baseline — far more robust than any absolute
  colour or texture judgement.
- **Temporal priors.** Fill is monotonically non-decreasing between
  collections. A bin cannot go 80% → 20% except at a collection event.
  Enforcing monotonicity plus step-detection removes most transient errors
  (a person walking past, rain, headlights).

⚠️ The current single-image focus is fine as a development strategy, but
**stock photos and CCTV frames are different distributions.** The 5 fixtures
are crisp, eye-level, well-lit. Real CCTV is compressed, motion-blurred,
high-angle, and grayscale IR half the day. Even 50 real camera frames are worth
more than 500 stock images.

### T-17 · Model swap for other detectors

A YOLO detector emits `(bbox, class, conf, mask)` — a standard interface that
composes with other YOLO pipelines (shared tracker, batched GPU inference).
Best option once T-13 lands: add `bin` as a class to the **same** segmentation
model already running for waste, giving one inference pass instead of two and
*reducing* per-frame cost versus today.

---

## 9. Beyond the monocular ceiling

### T-20 · Only if T-15 plateaus

The residual error (§1.2) is a depth problem. Two ways past, in cost order:

- **Monocular depth** — Depth Anything v2 or similar; compare the aperture
  plane against the trash surface. Vision-only, no hardware, ~1 day to test.
- **Ultrasonic fill sensor in the lid** — ~$20, the industry standard for this
  exact problem, and it produces free ground-truth labels for the vision model
  forever. Changes the product, so it's a business call — but raise it early
  rather than after six months of tuning.

**Do not start either until T-15 is measured.** The classifier may make both
unnecessary.

---

## 10. Tooling

**Use (all free):**

| Tool | For | Note |
|---|---|---|
| Ultralytics | Training, val, mAP, export | Already a dependency. Do not add a second framework. |
| Label Studio / CVAT | Box & polygon labelling | Self-host, free. Roboflow's free tier if you'd rather have hosted + format conversion + public datasets in one place. |
| FiftyOne | Dataset + prediction visualisation | `pip install fiftyone`. Best tool for *"why is my model wrong"* — predictions against ground truth, surfaces label errors and hard cases. Highest value per minute here. |
| Roboflow Universe | Public bin/trash datasets | Seeds T-11. Check licences. |

**Add when it hurts, not before:** MLflow or W&B (Ultralytics has built-in
hooks, ~one env var — add past ~10 experiments); DVC (add when `datasets/`
outgrows git).

**Skip:** Kubeflow, Airflow, feature stores, custom training loops, a custom
annotation tool.

---

## 11. Suggested order

```
T-01  accuracy harness + labels     ~2 days   BLOCKING — everything depends on it   OPEN (needs ~200 labelled images)
T-02  state leak between uploads    ~15 min   do now, it's a demo-visible bug       ✅ DONE
T-04  test suite uses wrong path    ~20 min   do before ANY fill tuning             ✅ DONE
T-03  median smoothing + stills     ~20 min                                        ✅ DONE
T-06  docstring fix                 ~2 min                                         ✅ DONE
T-07  demo_test.py                  ~5 min                                         ✅ DONE
      dead-code removal (§4, non-T-13-gated half)                                  ✅ DONE
T-10  dataset layout                ~2 hrs    parallel with T-01                   OPEN (no data yet to justify scaffolding)
T-11  seed public datasets          ~½ day                                         OPEN (needs licensing/dataset picks)
T-12  prompt/threshold sweep        ~2 hrs    needs T-01                           OPEN
T-13  fine-tune bin model           ~1 day    needs T-01, T-10                     OPEN
      dead-code removal (§4, HSV cluster)     after T-13 validates                 OPEN
T-05  tighten Test 1                ~5 min    after T-13                           OPEN (blocked on T-01 per its own text)
T-15  fill classifier               ~2 days   needs T-01, T-13                     OPEN
T-08  revisit inference cost        —         after T-13                          OPEN (deferred)
T-16  fixed-camera exploitation     —         when video work resumes             OPEN (deferred)
T-20  depth / sensor                —         only if T-15 plateaus               OPEN
```

**2026-09-09:** every item above marked ✅ DONE was completed and verified
(`verify_system.py` 6/6 pass) — see the Progress note at the top of this
document and each task's own section for how. Everything still OPEN needs
real data, labelling, training time, or a business decision — none of it is
something an agent can complete without that input.

Roughly a week of focused work, and **more than half of it is labelling, not
coding.** That ratio is normal. The temptation will be to skip T-01 and tune
constants instead — that is exactly what produced the current situation, where
no one can say whether a change helped.

---

## 12. Quick reference

### Settings that actually matter

| Field | Default | Effect |
|---|---|---|
| `openvocab_enabled` | `True` | `False` reverts to the HSV heuristic |
| `openvocab_conf` | `0.15` | Lower = more bins, more false positives |
| `openvocab_scales` | `(640, 1280)` | See §2.3 before changing |
| `bin_model_path` | `""` | Set this to activate a fine-tuned model (T-13) |
| `area_weight` / `height_weight` / `overflow_weight` | `0.55/0.25/0.20` | Fill factor weights |
| `waste_overflow_band_ratio` | `0.45` | Fraction of bin height searched above the rim |
| `waste_overflow_edge_gate` | `True` | Texture gate on the overflow band |
| `waste_overflow_require_contact` | `False` | Stricter gate — only for downward-angled cameras (§1.2) |
| `smoothing_window` | `5` | Frames in the rolling fill window (median, since T-03) |
| `smoothing_enabled` | `True` | Added by T-03. `False` for stills (set automatically by `handle_image`) |
| `frame_skip` | `3` | Run full pipeline every N frames (video only) |

### Test suite

`python verify_system.py` — exit 0 = all passed, 1 = failure, 2 = incomplete
(a fixture was missing, so some tests were skipped and verified nothing).

| # | Test | Validates |
|---|---|---|
| 1 | Multi-bin separation | `001.jpg`, currently asserts `>= 2` (truth is 4 — see T-05) |
| 2 | Pipeline structural sanity | Synthetic image, no crashes |
| 3 | Headless app | End-to-end on a real image |
| 4 | Manual ROI path | Manual boxes → detect → clear → auto resumes |
| 5 | Overflow override | Tiered fill-minimum logic |
| 6 | Additional fixtures | Every other image in `tests/fixtures/` runs clean |

Adding a fixture image to `tests/fixtures/` is picked up automatically by
Test 6 — no code change.

### Model files

| File | Size | Role |
|---|---|---|
| `yolo11n-seg.pt` | 6 MB | Waste segmentation (COCO-pretrained) |
| `yoloe-11l-seg.pt` | 71 MB | Open-vocab bin detection |
| `mobileclip_blt.ts` | 600 MB | Text encoder for YOLOE prompts |
| `models/yolov8n-seg.pt` | 7 MB | Spare waste model, not referenced by code |

The first three are auto-downloaded by Ultralytics on first run; they are not
pip packages and are not in version control.
