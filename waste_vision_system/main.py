
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2


def _launch_streamlit() -> None:
    """Start the Streamlit UI as a subprocess."""
    app_path = Path(__file__).resolve().parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path),
           "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    print(f"[Waste Vision] Launching Streamlit UI …")
    print(f"  → {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[Waste Vision] Stopped by user.")


def _run_headless(args: argparse.Namespace) -> None:
    """Run the detection pipeline in headless CLI mode."""
    # Imports here to avoid heavy loading when only launching UI
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    from config.settings import Settings
    from detectors.bin_detector import BinDetector
    from detectors.waste_detector import WasteDetector
    from services.occupancy import OccupancyEstimator, wall_mask_for
    from services.stream_handler import StreamHandler
    from services.logger import DataLogger
    from utils.drawing import draw_bin_overlay, draw_hud_header, draw_no_detection
    from utils.fps_counter import FPSCounter

    settings = Settings()
    if args.conf:
        settings.confidence_threshold = args.conf
    if args.model_bin:
        settings.bin_model_path = args.model_bin
    if args.model_waste:
        settings.waste_model_path = args.model_waste

    bin_detector = BinDetector(settings)
    waste_detector = WasteDetector(settings)
    occ_estimator = OccupancyEstimator(settings)
    stream = StreamHandler(settings)
    logger = DataLogger(settings)
    fps_cnt = FPSCounter()

    # Determine source
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    print(f"[Waste Vision] Opening source: {source}")
    if not stream.open(source):
        print("[Waste Vision] ERROR: Failed to open source.")
        sys.exit(1)

    # Video writer setup
    writer = None
    if args.output:
        w, h = stream.frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, stream.fps, (w, h))
        print(f"[Waste Vision] Writing output to: {args.output}")

    frame_count = 0
    try:
        while True:
            ret, frame = stream.read(skip_frames=False)
            if not ret:
                break

            frame_count += 1

            # Skip frames for performance
            if frame_count % settings.frame_skip != 0:
                if writer:
                    writer.write(frame)
                continue

            fps_cnt.tick()
            annotated = frame.copy()

            # Pipeline
            bin_dets = bin_detector.detect(frame)

            if not bin_dets:
                draw_no_detection(annotated)
            else:
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for bd in bin_dets:
                    waste_dets = waste_detector.detect(
                        frame, bd.bbox, bd.interior_mask,
                        bin_color_name=getattr(bd, 'bin_color_name', 'unknown'),
                        bin_hsv_range=getattr(bd, 'bin_hsv_range', None),
                    )

                    # Build bin wall mask for aperture-focused occupancy: the
                    # detector's own surface mask if it made one, else an
                    # expanded HSV colour-range match.
                    bin_color_mask = wall_mask_for(bd, hsv_frame)

                    waste_masks = [wd.mask for wd in waste_dets]
                    occ = occ_estimator.estimate(
                        bd.bin_id, bd.interior_mask, waste_masks,
                        bd.rim_top_y, bd.rim_bottom_y,
                        bin_color_mask=bin_color_mask,
                    )
                    draw_bin_overlay(annotated, bd, occ, waste_dets)
                    logger.log_entry(occ.bin_id, occ.fill_pct, occ.status, bd.confidence)

                    print(f"  Frame {frame_count}: Bin #{occ.bin_id} → "
                          f"{occ.fill_pct:.0f}% ({occ.status.value})  "
                          f"[{fps_cnt.fps:.1f} FPS]", end="\r")

            draw_hud_header(annotated, fps_cnt.fps, len(bin_dets), fps_cnt.latency_ms)

            if writer:
                writer.write(annotated)

            # Show in window if not fully headless
            if not args.no_display:
                cv2.imshow("Waste Vision System", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[Waste Vision] Stopped by user.")

    finally:
        stream.release()
        if writer:
            writer.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        # Export CSV
        csv_path = logger.export_csv()
        print(f"\n[Waste Vision] Processed {frame_count} frames")
        print(f"[Waste Vision] CSV exported: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Waste Vision System — AI-powered trash bin occupancy monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--headless", action="store_true",
        help="Run in headless CLI mode (no Streamlit UI).",
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video file, webcam index (0, 1, ...), or RTSP URI.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for annotated output video (e.g., output.mp4).",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Detection confidence threshold (0.10–1.00).",
    )
    parser.add_argument(
        "--model-bin", type=str, default=None,
        help="Path to custom bin detection model weights.",
    )
    parser.add_argument(
        "--model-waste", type=str, default=None,
        help="Path to custom waste detection model weights.",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable cv2.imshow window in headless mode.",
    )

    args = parser.parse_args()

    if args.headless:
        if not args.source:
            parser.error("--source is required in headless mode")
        _run_headless(args)
    else:
        _launch_streamlit()


if __name__ == "__main__":
    main()
