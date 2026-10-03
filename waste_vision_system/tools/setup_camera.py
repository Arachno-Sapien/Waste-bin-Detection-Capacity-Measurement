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
