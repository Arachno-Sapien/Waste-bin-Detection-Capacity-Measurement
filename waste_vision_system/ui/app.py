"""
Waste Vision System — Main Streamlit Dashboard
=================================================
Full-featured UI supporting four input modes (image, video, webcam, RTSP)
with real-time occupancy estimation, HUD overlays, and data export.

Launch:
    streamlit run waste_vision_system/ui/app.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ---- Fix import paths when run via `streamlit run` -----------------------
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from config.settings import Settings
from detectors.bin_detector import BinDetector
from detectors.waste_detector import WasteDetector
from services.occupancy import OccupancyEstimator, wall_mask_for
from services.stream_handler import StreamHandler
from services.logger import DataLogger
from utils.drawing import draw_bin_overlay, draw_hud_header, draw_no_detection
from utils.fps_counter import FPSCounter
from ui.components import (
    render_action_buttons,
    render_confidence_slider,
    render_export_buttons,
    render_fps_indicator,
    render_source_selector,
    render_status_card,
    render_system_header,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Waste Vision System — AI Bin Monitoring",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global dark-theme CSS
st.markdown(
    """
    <style>
    /* Hide Streamlit menu & footer */
    #MainMenu, footer {visibility: hidden;}
    /* Image/video containers */
    .stImage > img, .stVideo > video {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    """Initialise all session-state keys on first run."""
    defaults = {
        "settings": Settings(),
        "bin_detector": None,
        "waste_detector": None,
        "occupancy_estimator": None,
        "stream_handler": None,
        "data_logger": None,
        "fps_counter": FPSCounter(),
        "running": False,
        "last_frame": None,
        "last_results": [],    # list of (bin_det, occ_result, waste_dets)
        # Manual bin boundary selection
        "original_frame": None,
        "edit_mode": False,
        "manual_rois": [],
        "_img_upload_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


def _get_pipeline():
    """Lazy-init detector / estimator instances."""
    s = st.session_state
    settings: Settings = s["settings"]

    if s["bin_detector"] is None:
        s["bin_detector"] = BinDetector(settings)
    if s["waste_detector"] is None:
        s["waste_detector"] = WasteDetector(settings)
    if s["occupancy_estimator"] is None:
        s["occupancy_estimator"] = OccupancyEstimator(settings)
    if s["stream_handler"] is None:
        s["stream_handler"] = StreamHandler(settings)
    if s["data_logger"] is None:
        s["data_logger"] = DataLogger(settings)

    return (
        s["bin_detector"],
        s["waste_detector"],
        s["occupancy_estimator"],
        s["stream_handler"],
        s["data_logger"],
        s["fps_counter"],
    )


# ---------------------------------------------------------------------------
# Core pipeline runner (single frame)
# ---------------------------------------------------------------------------

def process_frame(
    frame: np.ndarray,
    bin_detector: BinDetector,
    waste_detector: WasteDetector,
    occupancy_estimator: OccupancyEstimator,
    data_logger: DataLogger,
    fps_counter: FPSCounter,
    settings: Settings,
) -> np.ndarray:
    """
    Run the full 8-step pipeline on a single frame and return the
    annotated frame.
    """
    fps_counter.tick()
    annotated = frame.copy()

    # Step 1 & 2: Detect bins + interior masks
    bin_detections = bin_detector.detect(frame)

    results_list = []

    if not bin_detections:
        draw_no_detection(annotated)
    else:
        # Pre-compute HSV for bin color mask generation
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for bin_det in bin_detections:
            # Step 3: Detect waste inside bin
            waste_dets = waste_detector.detect(
                frame, bin_det.bbox, bin_det.interior_mask,
                bin_color_name=getattr(bin_det, 'bin_color_name', 'unknown'),
                bin_hsv_range=getattr(bin_det, 'bin_hsv_range', None),
            )

            # Build bin wall mask for aperture-focused occupancy: the
            # detector's own surface mask if it made one, else an expanded
            # HSV colour-range match.
            bin_color_mask = wall_mask_for(bin_det, hsv_frame)

            # Steps 4-6: Estimate occupancy
            waste_masks = [wd.mask for wd in waste_dets]
            occ = occupancy_estimator.estimate(
                bin_id=bin_det.bin_id,
                interior_mask=bin_det.interior_mask,
                waste_masks=waste_masks,
                rim_top_y=bin_det.rim_top_y,
                rim_bottom_y=bin_det.rim_bottom_y,
                bin_color_mask=bin_color_mask,
            )

            # Step 7: Draw overlay
            draw_bin_overlay(annotated, bin_det, occ, waste_dets)

            # Log
            data_logger.log_entry(
                bin_id=occ.bin_id,
                fill_pct=occ.fill_pct,
                status=occ.status,
                confidence=bin_det.confidence,
            )

            results_list.append((bin_det, occ, waste_dets))

    # HUD header
    draw_hud_header(
        annotated,
        fps=fps_counter.fps,
        active_bins=len(bin_detections),
        latency_ms=fps_counter.latency_ms,
    )

    st.session_state["last_frame"] = annotated
    st.session_state["last_results"] = results_list

    return annotated


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def handle_image(source, pipeline):
    """Process a single uploaded image with optional manual bin boundary editing."""
    bin_det, waste_det, occ_est, _, data_log, fps_cnt = pipeline
    settings = st.session_state["settings"]
    # A still has no temporal dimension; smoothing across Streamlit re-runs
    # only creates the state-leak bug T-02 fixes in a new place.
    settings.smoothing_enabled = False

    # ---- Decode image (only once per upload) ----
    if st.session_state.get("_img_upload_id") != id(source):
        source.seek(0)
        file_bytes = np.frombuffer(source.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Failed to decode image. Please upload a valid image file.")
            return
        st.session_state["_img_upload_id"] = id(source)
        st.session_state["original_frame"] = frame.copy()
        # Reset edit mode on new upload
        st.session_state["edit_mode"] = False
        st.session_state["manual_rois"] = []
        bin_det.clear_manual_rois()
        # A new image is a new scene: drop tracker IDs and smoothing history so
        # fill values are not averaged across two unrelated images.
        bin_det.reset_tracking()
        occ_est.reset()
    else:
        source.seek(0)  # Keep file pointer valid for re-reads
        frame = st.session_state.get("original_frame")
        if frame is None:
            file_bytes = np.frombuffer(source.read(), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state["original_frame"] = frame.copy()

    if frame is None:
        st.error("Failed to decode image.")
        return

    img_h, img_w = frame.shape[:2]

    # ---- Apply manual ROIs if set ----
    manual_rois = st.session_state.get("manual_rois", [])
    if manual_rois:
        bin_det.set_manual_rois(manual_rois)
    else:
        bin_det.clear_manual_rois()

    # ---- Run detection pipeline ----
    annotated = process_frame(
        frame, bin_det, waste_det, occ_est, data_log, fps_cnt, settings
    )

    # ---- Display results ----
    col_img, col_status = st.columns([3, 1])
    with col_img:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_status:
        st.markdown("### Detection Results")
        for bin_d, occ, _ in st.session_state["last_results"]:
            render_status_card(occ.bin_id, occ.fill_pct, occ.status, bin_d.confidence)

        if not st.session_state["last_results"]:
            st.info("No bins detected in this image.")

    # ---- Manual Bin Selection Controls ----
    st.markdown("---")

    using_manual = bool(manual_rois)
    if using_manual:
        st.success(f"✏️ Using **manual** bin boundaries ({len(manual_rois)} bin{'s' if len(manual_rois) != 1 else ''})")
    else:
        st.info("🤖 Using **AI automatic** bin detection")

    edit_mode = st.session_state.get("edit_mode", False)

    # Toggle buttons
    btn_cols = st.columns(3)
    with btn_cols[0]:
        if st.button("✏️ Edit Bin Selection", key="btn_edit_roi", use_container_width=True):
            st.session_state["edit_mode"] = True
            st.rerun()
    with btn_cols[1]:
        if using_manual:
            if st.button("🤖 Reset to AI Detection", key="btn_reset_roi", use_container_width=True):
                st.session_state["manual_rois"] = []
                st.session_state["edit_mode"] = False
                bin_det.clear_manual_rois()
                st.rerun()

    # ---- Edit Mode Panel ----
    if st.session_state.get("edit_mode", False):
        st.markdown("### 📐 Manual Bin Boundary Editor")
        st.markdown(
            f"Draw rectangles around each bin. Image size: **{img_w} × {img_h}** pixels. "
            "Coordinates are `(x1, y1)` = top-left corner, `(x2, y2)` = bottom-right corner."
        )

        # Show the original image with a grid overlay for reference
        ref_img = frame.copy()
        # Draw light grid lines every 50px
        for gx in range(0, img_w, 50):
            cv2.line(ref_img, (gx, 0), (gx, img_h), (200, 200, 200), 1)
            if gx % 100 == 0:
                cv2.putText(ref_img, str(gx), (gx + 2, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        for gy in range(0, img_h, 50):
            cv2.line(ref_img, (0, gy), (img_w, gy), (200, 200, 200), 1)
            if gy % 100 == 0:
                cv2.putText(ref_img, str(gy), (2, gy + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Draw existing AI detections in blue for reference
        for bin_d, _, _ in st.session_state.get("last_results", []):
            bx1, by1, bx2, by2 = bin_d.bbox
            cv2.rectangle(ref_img, (bx1, by1), (bx2, by2), (255, 180, 0), 2)
            cv2.putText(ref_img, f"AI Bin #{bin_d.bin_id}", (bx1, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

        # Draw current manual ROIs in green
        edit_rois = st.session_state.get("manual_rois", [])
        for i, (rx1, ry1, rx2, ry2) in enumerate(edit_rois):
            cv2.rectangle(ref_img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
            cv2.putText(ref_img, f"Manual #{i + 1}", (rx1, ry1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        st.image(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB),
                 caption="Reference: Blue = AI detections, Green = your manual selections",
                 use_container_width=True)

        # ROI input form
        with st.form("roi_form", clear_on_submit=True):
            st.markdown("**Add a new bin boundary:**")
            roi_cols = st.columns(4)
            with roi_cols[0]:
                new_x1 = st.number_input("X1 (left)", 0, img_w - 1, 0, key="roi_x1")
            with roi_cols[1]:
                new_y1 = st.number_input("Y1 (top)", 0, img_h - 1, 0, key="roi_y1")
            with roi_cols[2]:
                new_x2 = st.number_input("X2 (right)", 1, img_w, img_w // 4, key="roi_x2")
            with roi_cols[3]:
                new_y2 = st.number_input("Y2 (bottom)", 1, img_h, img_h, key="roi_y2")

            form_cols = st.columns(3)
            with form_cols[0]:
                add_btn = st.form_submit_button("➕ Add Bin", use_container_width=True)

            if add_btn:
                if new_x2 > new_x1 and new_y2 > new_y1:
                    current_rois = st.session_state.get("manual_rois", [])
                    current_rois.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
                    st.session_state["manual_rois"] = current_rois
                    st.rerun()
                else:
                    st.error("Invalid coordinates: X2 must be > X1 and Y2 must be > Y1")

        # Show current manual ROIs with remove buttons
        current_rois = st.session_state.get("manual_rois", [])
        if current_rois:
            st.markdown("**Current manual bin boundaries:**")
            for i, (rx1, ry1, rx2, ry2) in enumerate(current_rois):
                rc1, rc2 = st.columns([4, 1])
                with rc1:
                    st.markdown(
                        f"**Bin {i + 1}**: ({rx1}, {ry1}) → ({rx2}, {ry2})  "
                        f"*{rx2 - rx1} × {ry2 - ry1} px*"
                    )
                with rc2:
                    if st.button("🗑️", key=f"rm_roi_{i}", help=f"Remove bin {i + 1}"):
                        current_rois.pop(i)
                        st.session_state["manual_rois"] = current_rois
                        st.rerun()

        # Apply / Cancel
        act_cols = st.columns(2)
        with act_cols[0]:
            if current_rois:
                if st.button("✅ Apply & Re-analyze", key="btn_apply_roi",
                             use_container_width=True, type="primary"):
                    st.session_state["edit_mode"] = False
                    st.rerun()
        with act_cols[1]:
            if st.button("❌ Cancel", key="btn_cancel_roi", use_container_width=True):
                st.session_state["edit_mode"] = False
                st.rerun()


def handle_video(source, pipeline):
    """Process an uploaded video file frame by frame."""
    bin_det, waste_det, occ_est, stream, data_log, fps_cnt = pipeline
    settings = st.session_state["settings"]

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(source.read())
        tmp_path = tmp.name

    if not stream.open(tmp_path):
        st.error("Failed to open video file.")
        return

    # A new stream start is a new scene: drop tracker IDs and smoothing
    # history so fill values are not averaged across two unrelated runs.
    bin_det.reset_tracking()
    occ_est.reset()
    settings.smoothing_enabled = True

    st.markdown("### Video Processing")

    col_vid, col_status = st.columns([3, 1])
    frame_placeholder = col_vid.empty()
    status_placeholder = col_status.empty()
    progress_bar = st.progress(0)
    stop_btn = st.button("Stop Processing", key="stop_video_proc")

    total = stream.total_frames if stream.total_frames > 0 else 1
    frame_idx = 0

    while not stop_btn:
        ret, frame = stream.read(skip_frames=False)
        if not ret:
            break

        frame_idx += 1

        # Process only every Nth frame for speed
        if frame_idx % settings.frame_skip != 0:
            progress_bar.progress(min(frame_idx / total, 1.0))
            continue

        annotated = process_frame(
            frame, bin_det, waste_det, occ_est, data_log, fps_cnt, settings
        )

        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        with status_placeholder.container():
            for bin_d, occ, _ in st.session_state["last_results"]:
                render_status_card(occ.bin_id, occ.fill_pct, occ.status, bin_d.confidence)

        progress_bar.progress(min(frame_idx / total, 1.0))

    stream.release()
    progress_bar.progress(1.0)
    st.success(f"Processed {frame_idx} frames")


def handle_webcam(cam_idx: int, pipeline):
    """Live webcam processing loop."""
    bin_det, waste_det, occ_est, stream, data_log, fps_cnt = pipeline
    settings = st.session_state["settings"]

    if not stream.open(cam_idx):
        st.error(f"Failed to open webcam {cam_idx}. Check your camera connection.")
        return

    # A new stream start is a new scene: drop tracker IDs and smoothing
    # history so fill values are not averaged across two unrelated runs.
    bin_det.reset_tracking()
    occ_est.reset()
    settings.smoothing_enabled = True

    st.markdown("### Live Webcam Feed")

    col_vid, col_status = st.columns([3, 1])
    frame_placeholder = col_vid.empty()
    status_placeholder = col_status.empty()

    stop_btn = st.button("Stop Webcam", key="stop_webcam")

    while not stop_btn and st.session_state.get("running", True):
        ret, frame = stream.read(skip_frames=True)
        if not ret:
            time.sleep(0.01)
            continue

        annotated = process_frame(
            frame, bin_det, waste_det, occ_est, data_log, fps_cnt, settings
        )

        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        with status_placeholder.container():
            render_fps_indicator(fps_cnt.fps, fps_cnt.latency_ms)
            for bin_d, occ, _ in st.session_state["last_results"]:
                render_status_card(occ.bin_id, occ.fill_pct, occ.status, bin_d.confidence)

    stream.release()


def handle_rtsp(uri: str, pipeline):
    """Live RTSP stream processing loop."""
    bin_det, waste_det, occ_est, stream, data_log, fps_cnt = pipeline
    settings = st.session_state["settings"]

    st.markdown("### RTSP Camera Feed")
    st.info(f"Connecting to: `{uri}`")

    if not stream.open(uri):
        st.error("Failed to connect to RTSP stream. Check the URI and network.")
        return

    # A new stream start is a new scene: drop tracker IDs and smoothing
    # history so fill values are not averaged across two unrelated runs.
    bin_det.reset_tracking()
    occ_est.reset()
    settings.smoothing_enabled = True

    st.success("Connected successfully!")

    col_vid, col_status = st.columns([3, 1])
    frame_placeholder = col_vid.empty()
    status_placeholder = col_status.empty()

    stop_btn = st.button("Disconnect", key="stop_rtsp")

    while not stop_btn and st.session_state.get("running", True):
        ret, frame = stream.read(skip_frames=True)
        if not ret:
            if stream.is_reconnecting:
                frame_placeholder.warning("Reconnecting to RTSP stream...")
            time.sleep(0.1)
            continue

        annotated = process_frame(
            frame, bin_det, waste_det, occ_est, data_log, fps_cnt, settings
        )

        frame_placeholder.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

        with status_placeholder.container():
            render_fps_indicator(fps_cnt.fps, fps_cnt.latency_ms)
            for bin_d, occ, _ in st.session_state["last_results"]:
                render_status_card(occ.bin_id, occ.fill_pct, occ.status, bin_d.confidence)

    stream.release()


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

def main():
    """Streamlit entry point."""
    render_system_header()

    # ---- Sidebar controls ----
    mode, source = render_source_selector()

    conf = render_confidence_slider()
    st.session_state["settings"].confidence_threshold = conf

    start_clicked, stop_clicked = render_action_buttons()

    if start_clicked:
        st.session_state["running"] = True
    if stop_clicked:
        st.session_state["running"] = False

    save_clicked, export_clicked = render_export_buttons()

    # ---- Pipeline init ----
    pipeline = _get_pipeline()
    _, _, _, _, data_log, fps_cnt = pipeline

    # ---- Sidebar: Live ROI Editor (for webcam/RTSP) ----
    if mode in ("webcam", "rtsp"):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Manual Bin Regions")
        manual_rois = st.session_state.get("manual_rois", [])
        if manual_rois:
            st.sidebar.success(f"Using {len(manual_rois)} manual ROI(s)")
        else:
            st.sidebar.info("AI auto-detection active")

        with st.sidebar.expander("✏️ Edit Bin ROIs", expanded=False):
            with st.form("live_roi_form", clear_on_submit=True):
                lr_x1 = st.number_input("X1", 0, 3840, 0, key="lr_x1")
                lr_y1 = st.number_input("Y1", 0, 2160, 0, key="lr_y1")
                lr_x2 = st.number_input("X2", 1, 3840, 320, key="lr_x2")
                lr_y2 = st.number_input("Y2", 1, 2160, 480, key="lr_y2")
                if st.form_submit_button("➕ Add Bin ROI"):
                    if lr_x2 > lr_x1 and lr_y2 > lr_y1:
                        rois = st.session_state.get("manual_rois", [])
                        rois.append((int(lr_x1), int(lr_y1), int(lr_x2), int(lr_y2)))
                        st.session_state["manual_rois"] = rois
                        st.rerun()

            for i, (rx1, ry1, rx2, ry2) in enumerate(manual_rois):
                c1, c2 = st.columns([3, 1])
                c1.caption(f"Bin {i+1}: ({rx1},{ry1})→({rx2},{ry2})")
                if c2.button("🗑️", key=f"lr_rm_{i}"):
                    manual_rois.pop(i)
                    st.session_state["manual_rois"] = manual_rois
                    st.rerun()

            if manual_rois and st.button("🤖 Reset to AI", key="lr_reset"):
                st.session_state["manual_rois"] = []
                st.rerun()

        # Apply live ROIs to bin detector
        bin_det_inst = pipeline[0]
        if manual_rois:
            bin_det_inst.set_manual_rois(manual_rois)
        else:
            bin_det_inst.clear_manual_rois()

    # ---- Sidebar info ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Info")
    device = st.session_state["settings"].resolve_device()
    st.sidebar.markdown(f"**Device:** `{device.upper()}`")
    _s = st.session_state['settings']
    if _s.bin_model_path:
        bin_model_label = _s.bin_model_path
    elif _s.openvocab_enabled:
        bin_model_label = f"{_s.openvocab_model_path} (open-vocab)"
    else:
        bin_model_label = 'HSV Colour Segmentation'
    st.sidebar.markdown(f"**Model (Bin):** `{bin_model_label}`")
    st.sidebar.markdown(f"**Model (Waste):** `{st.session_state['settings'].waste_model_path}`")
    render_fps_indicator(fps_cnt.fps, fps_cnt.latency_ms)

    # ---- Handle export buttons ----
    if save_clicked and st.session_state["last_frame"] is not None:
        out = data_log.save_snapshot(
            st.session_state["last_frame"],
            bin_id=0,
            fill_pct=0,
        )
        st.sidebar.success(f"Saved: {out.name}")

    if export_clicked:
        csv_path = data_log.export_csv()
        st.sidebar.success(f"Exported: {csv_path.name}")
        # Provide download link
        with open(csv_path, "rb") as f:
            st.sidebar.download_button(
                "Download CSV",
                data=f.read(),
                file_name=csv_path.name,
                mime="text/csv",
            )

    # ---- Route to mode handler ----
    if source is None and mode != "webcam":
        # Show instructions
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 80px 40px;
                color: #31333f;
            ">
                <h3 style="color: #31333f;">Select an Input Source</h3>
                <p style="color: #555555;">
                    Upload an image or video, connect a webcam, or enter an RTSP camera URI
                    using the sidebar controls to get started.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if mode == "image" and source is not None:
        handle_image(source, pipeline)
    elif mode == "video" and source is not None:
        handle_video(source, pipeline)
    elif mode == "webcam":
        if st.session_state["running"]:
            handle_webcam(source, pipeline)
        else:
            st.info("Press **Start** to begin webcam detection.")
    elif mode == "rtsp" and source:
        if st.session_state["running"]:
            handle_rtsp(source, pipeline)
        else:
            st.info("Press **Start** to connect to the RTSP stream.")

    # ---- Data log summary ----
    if data_log.entry_count > 0:
        with st.expander("Audit Log (Recent Entries)", expanded=False):
            df = data_log.get_dataframe()
            st.dataframe(df.tail(50), use_container_width=True)


if __name__ == "__main__":
    main()
