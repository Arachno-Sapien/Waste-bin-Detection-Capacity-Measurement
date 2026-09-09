"""
Automated Verification Suite – Waste Vision System
===================================================
Validates end-to-end detection, segmentation, and fill estimation logic.

Each test is run independently: a failure is reported and the suite carries
on, so one broken test never hides the state of the others.

A test that cannot run — because the fixture image it needs is absent —
reports SKIP rather than PASS, so a stripped-down copy of the project can
never produce an all-green board that verified almost nothing.

Exit codes:
    0  every test ran and passed
    1  at least one test failed
    2  no failures, but one or more tests were skipped (suite incomplete)
"""

import sys
import traceback
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from config.settings import Settings, FillStatus
from detectors.bin_detector import BinDetector
from detectors.waste_detector import WasteDetector
from services.occupancy import OccupancyEstimator, wall_mask_for

# Sentinel returned by a test that could not run (missing fixture image).
# Kept distinct from True/False so the summary reports SKIP instead of
# silently counting the test as a pass.
SKIPPED = "SKIP"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def test_image_path() -> Path:
    """
    Resolve the primary (ground-truth) multi-bin test image.

    The image ships with the project under tests/fixtures/, so this path is
    valid on any machine that has the repo. Callers check .exists() and skip
    their test if it is missing — no machine-specific fallback, which would
    only ever resolve on the original author's PC.
    """
    return PROJECT_ROOT / "tests" / "fixtures" / "001.jpg"


def additional_fixture_images() -> list:
    """
    Every image in tests/fixtures/ besides the ground-truth 001.jpg used by
    Tests 1, 3 and 4 above. These have no documented ground truth (bin
    count, colours, etc.), so Test 6 below runs them as structural smoke
    tests only — no crash, sane output — rather than exact-count assertions.
    """
    fixtures_dir = PROJECT_ROOT / "tests" / "fixtures"
    if not fixtures_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        (p for p in fixtures_dir.iterdir()
         if p.is_file() and p.suffix.lower() in exts and p.name != "001.jpg"),
        key=lambda p: p.name,
    )


def analyse_bin(frame, hsv_frame, b, wd, oe):
    """Run waste detection + aperture-focused occupancy for one bin.

    Mirrors ui/app.py process_frame: the detector's own instance mask is the
    real wall region and beats a colour-range guess. Falls back to the HSV
    mask (via the shared wall_mask_for helper) for strategies that produce
    no surface_mask (manual ROI, HSV).
    """
    w_dets = wd.detect(
        frame, b.bbox, b.interior_mask,
        bin_color_name=b.bin_color_name,
        bin_hsv_range=b.bin_hsv_range,
    )
    occ = oe.estimate(
        b.bin_id, b.interior_mask, [w.mask for w in w_dets],
        b.rim_top_y, b.rim_bottom_y,
        bin_color_mask=wall_mask_for(b, hsv_frame),
    )
    return w_dets, occ


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_multi_bin_detection():
    """
    Test 1: Multi-Bin Separation (001.jpg)
    Validates that multiple adjacent bins are correctly separated.

    Ground truth for 001.jpg: four green wheelie bins in a row, all
    overflowing, against a dark green wall.
    """
    print("\n--- Running Test 1: Multi-Bin Separation (001.jpg) ---")
    settings = Settings()
    bd = BinDetector(settings)
    wd = WasteDetector(settings)
    oe = OccupancyEstimator(settings)

    img_path = test_image_path()
    if not img_path.exists():
        print(f"Skipping Test 1: Image not found at {img_path}")
        return SKIPPED

    img = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bin_dets = bd.detect(img)
    print(f"Bins detected: {len(bin_dets)} (Expected: >= 2, ground truth: 4)")

    assert len(bin_dets) >= 2, f"Expected at least 2 bins, but detected {len(bin_dets)}"

    for b in bin_dets:
        _, occ = analyse_bin(img, hsv, b, wd, oe)
        print(f" Bin #{b.bin_id} ({b.bin_color_name}): BBox={b.bbox}, "
              f"Fill={occ.fill_pct:.1f}% ({occ.status.value})")

    if len(bin_dets) != 4:
        print(f"  NOTE: detected {len(bin_dets)} bins, ground truth is 4.")

    print("Test 1 PASSED")
    return True


def test_pipeline_synthetic():
    """
    Test 2: Pipeline Structural Sanity
    Validates that pipeline runs end-to-end on a synthetic image without errors.
    """
    print("\n--- Running Test 2: Pipeline Structural Sanity ---")
    settings = Settings()
    bd = BinDetector(settings)
    wd = WasteDetector(settings)
    oe = OccupancyEstimator(settings)

    # Synthetic image (640x480) with a dark green central rectangle (bin)
    synth = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw green bin
    cv2.rectangle(synth, (200, 108), (440, 440), (68, 140, 70), -1)
    hsv = cv2.cvtColor(synth, cv2.COLOR_BGR2HSV)

    bin_dets = bd.detect(synth)
    print(f"Synthetic Bins Detected: {len(bin_dets)} (Expected: >= 1)")

    assert len(bin_dets) >= 1, "Failed to detect synthetic green bin"

    for b in bin_dets:
        _, occ = analyse_bin(synth, hsv, b, wd, oe)
        print(f" Bin #{b.bin_id}: Fill={occ.fill_pct:.1f}% ({occ.status.value})")

    print("Test 2 PASSED")
    return True


def test_headless_app():
    """
    Test 3: Headless Application Test
    Validates that the headless pipeline can process a real image without GUI errors.
    """
    print("\n--- Running Test 3: Headless App Test ---")

    img_path = test_image_path()
    if not img_path.exists():
        print(f"Skipping Test 3: Image not found at {img_path}")
        return SKIPPED

    settings = Settings()
    bd = BinDetector(settings)
    wd = WasteDetector(settings)
    oe = OccupancyEstimator(settings)

    img = cv2.imread(str(img_path))
    assert img is not None, "Failed to load image"
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bin_dets = bd.detect(img)
    print(f"✓ BinDetector processed image successfully: {len(bin_dets)} bins")

    for b in bin_dets:
        w_dets, occ = analyse_bin(img, hsv, b, wd, oe)
        print(f"✓ WasteDetector processed Bin #{b.bin_id}: {len(w_dets)} waste items")
        print(f"✓ OccupancyEstimator computed: {occ.fill_pct:.1f}% ({occ.status.value})")

    print("Test 3 PASSED")
    return True


def test_manual_rois():
    """
    Test 4: Manual ROI Path
    Validates that manual bin boundaries produce correct detections with
    color info and proper interior masks.
    """
    print("\n--- Running Test 4: Manual ROI Path ---")
    settings = Settings()
    bd = BinDetector(settings)
    wd = WasteDetector(settings)
    oe = OccupancyEstimator(settings)

    img_path = test_image_path()
    if not img_path.exists():
        print(f"Skipping Test 4: Image not found at {img_path}")
        return SKIPPED

    img = cv2.imread(str(img_path))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # Set manual ROIs (approximate bin locations)
    rois = [(0, 0, w // 2, h), (w // 2, 0, w, h)]
    bd.set_manual_rois(rois)

    bin_dets = bd.detect(img)
    print(f"Manual ROI bins: {len(bin_dets)} (Expected: 2)")
    assert len(bin_dets) == 2, f"Expected 2 bins from manual ROIs, got {len(bin_dets)}"

    for b in bin_dets:
        assert b.bin_color_name != "unknown", f"Bin #{b.bin_id} has unknown color — color extraction failed"
        assert b.interior_mask is not None, f"Bin #{b.bin_id} has no interior mask"
        assert b.interior_mask.sum() > 0, f"Bin #{b.bin_id} interior mask is empty"
        print(f"  Bin #{b.bin_id}: color={b.bin_color_name}, conf={b.confidence}, "
              f"interior_px={b.interior_mask.sum()}")

        _, occ = analyse_bin(img, hsv, b, wd, oe)
        print(f"    Fill: {occ.fill_pct:.1f}% ({occ.status.value})")

    # Clear and verify auto-mode resumes
    bd.clear_manual_rois()
    auto_dets = bd.detect(img)
    print(f"After clear: AI detected {len(auto_dets)} bins (Expected: >= 2)")
    assert len(auto_dets) >= 2, f"After clearing manual ROIs, AI detection should work"

    print("Test 4 PASSED")
    return True


def test_overflow_override():
    """
    Test 5: Overflow Override Logic
    Validates the tiered overflow override computes correct minimum fills.
    """
    print("\n--- Running Test 5: Overflow Override Logic ---")
    settings = Settings()
    oe = OccupancyEstimator(settings)

    # Create dummy masks
    h, w = 100, 100
    interior = np.ones((h, w), dtype=np.uint8)

    # Test 1: overflow_score = 100 → fill >= 85%
    # Build a waste mask that triggers high overflow
    waste_full = np.ones((h, w), dtype=np.uint8)
    # We'll test the override directly by checking the OccupancyResult
    # with rim_top at 50 (middle), waste everywhere
    result = oe.estimate(bin_id=99, interior_mask=interior,
                         waste_masks=[waste_full], rim_top_y=50, rim_bottom_y=100)
    print(f"  Full waste: fill={result.fill_pct:.1f}%, overflow={result.overflow_score:.0f}")
    assert result.overflow_score >= 80.0, "Full-waste mask should register heavy overflow"
    assert result.fill_pct >= 85.0, f"Heavy overflow should force fill >= 85%, got {result.fill_pct}"

    # Test 2: No overflow → no override
    oe.reset(bin_id=98)
    no_waste = np.zeros((h, w), dtype=np.uint8)
    result2 = oe.estimate(bin_id=98, interior_mask=interior,
                          waste_masks=[no_waste], rim_top_y=0, rim_bottom_y=100)
    print(f"  No waste: fill={result2.fill_pct:.1f}%, overflow={result2.overflow_score:.0f}")
    assert result2.fill_pct == 0.0, f"Empty bin should be 0%, got {result2.fill_pct}"

    # Test 3: Moderate overflow (score=50) → fill >= 70%
    oe.reset(bin_id=97)
    # Small waste strip above rim only
    small_waste = np.zeros((h, w), dtype=np.uint8)
    small_waste[40:50, :] = 1  # Just above rim_top=50, 2% of total above area
    result3 = oe.estimate(bin_id=97, interior_mask=interior,
                          waste_masks=[small_waste], rim_top_y=50, rim_bottom_y=100)
    print(f"  Small overflow: fill={result3.fill_pct:.1f}%, overflow={result3.overflow_score:.0f}")
    assert result3.fill_pct >= 70.0, f"Moderate overflow should force fill >= 70%, got {result3.fill_pct}"

    print("Test 5 PASSED")
    return True


def test_additional_fixtures():
    """
    Test 6: Additional Fixture Images (Structural Sanity)
    Runs the full detection pipeline against every extra image dropped into
    tests/fixtures/ (besides 001.jpg). No ground truth is known for these,
    so this only checks that the pipeline runs end-to-end without raising
    and produces sane, in-range fill percentages — the real-image
    equivalent of Test 2's synthetic sanity check, one pass per image.
    """
    print("\n--- Running Test 6: Additional Fixture Images (Structural Sanity) ---")
    images = additional_fixture_images()
    if not images:
        print("Skipping Test 6: No additional images found in tests/fixtures/")
        return SKIPPED

    settings = Settings()
    all_ok = True

    for img_path in images:
        # Fresh detector/estimator instances per image so state never leaks
        # between fixtures (mirrors how each Streamlit session starts clean).
        bd = BinDetector(settings)
        wd = WasteDetector(settings)
        oe = OccupancyEstimator(settings)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ✗ {img_path.name}: failed to load (unsupported or corrupt file)")
            all_ok = False
            continue

        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            bin_dets = bd.detect(img)
            print(f"  {img_path.name}: {len(bin_dets)} bin(s) detected")

            for b in bin_dets:
                _, occ = analyse_bin(img, hsv, b, wd, oe)
                assert 0.0 <= occ.fill_pct <= 100.0, \
                    f"fill_pct out of range: {occ.fill_pct}"
                print(f"    Bin #{b.bin_id} ({b.bin_color_name}): "
                      f"Fill={occ.fill_pct:.1f}% ({occ.status.value})")
        except Exception as e:
            print(f"  ✗ {img_path.name}: pipeline raised {e}")
            all_ok = False

    if all_ok:
        print(f"Test 6 PASSED ({len(images)} additional image(s) processed cleanly)")
    else:
        print("Test 6 FAILED (see per-image errors above)")
    return all_ok


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Multi-Bin Separation", test_multi_bin_detection),
    ("Pipeline Structural Sanity", test_pipeline_synthetic),
    ("Headless App", test_headless_app),
    ("Manual ROI Path", test_manual_rois),
    ("Overflow Override", test_overflow_override),
    ("Additional Fixture Images", test_additional_fixtures),
]


MARKS = {"PASS": "✓ PASS", "FAIL": "✗ FAIL", "SKIP": "– SKIP"}


if __name__ == "__main__":
    results = []
    for name, fn in TESTS:
        try:
            outcome = fn()
            if outcome == SKIPPED:
                results.append((name, "SKIP"))
            else:
                results.append((name, "PASS" if outcome else "FAIL"))
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            traceback.print_exc()
            results.append((name, "FAIL"))

    print("\n" + "=" * 60)
    for name, status in results:
        print(f"  {MARKS[status]}  {name}")
    print("=" * 60)

    failed = [n for n, s in results if s == "FAIL"]
    skipped = [n for n, s in results if s == "SKIP"]
    passed = [n for n, s in results if s == "PASS"]

    if failed:
        print(f"✗ {len(failed)} of {len(results)} tests failed.")
        if skipped:
            print(f"  ({len(skipped)} also skipped: {', '.join(skipped)})")
        print("=" * 60)
        sys.exit(1)

    if skipped:
        print(f"⚠ INCOMPLETE: {len(passed)} passed, {len(skipped)} SKIPPED "
              f"of {len(results)} tests.")
        print(f"  Skipped: {', '.join(skipped)}")
        print("  These tests did NOT verify anything. Restore the images in")
        print("  tests/fixtures/ to run the full suite.")
        print("=" * 60)
        sys.exit(2)

    print(f"✓ ALL {len(passed)} VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    sys.exit(0)
