"""
Streamlit UI Components — Reusable widgets
============================================
Encapsulates sidebar controls, status cards, and display helpers
used by the main app.  All functions are pure Streamlit calls and
return the current widget state for the caller to act on.
"""

from __future__ import annotations

from typing import Optional, Tuple

import streamlit as st

from config.settings import FillStatus, STATUS_COLORS_HEX


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def render_source_selector() -> Tuple[str, Optional[object]]:
    """
    Render input source tabs in the sidebar.

    Returns
    -------
    tuple[str, object | None]
        (mode, source) where mode is one of
        ``"image"``, ``"video"``, ``"webcam"``, ``"rtsp"``
        and source is the uploaded file / URI / webcam index.
    """
    st.sidebar.markdown("### Input Source")
    mode = st.sidebar.radio(
        "Select source",
        ["Upload Image", "Upload Video", "Webcam", "RTSP Camera"],
        label_visibility="collapsed",
    )

    source = None

    if mode == "Upload Image":
        source = st.sidebar.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="img_uploader"
        )
        return "image", source

    elif mode == "Upload Video":
        source = st.sidebar.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"],
            key="vid_uploader"
        )
        return "video", source

    elif mode == "Webcam":
        cam_idx = st.sidebar.number_input("Camera index", 0, 10, 0, step=1)
        return "webcam", int(cam_idx)

    else:  # RTSP
        uri = st.sidebar.text_input(
            "RTSP URI",
            placeholder="rtsp://user:pass@192.168.1.100:554/stream",
            key="rtsp_uri",
        )
        return "rtsp", uri if uri else None


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def render_confidence_slider() -> float:
    """Render a confidence threshold slider. Returns the selected value."""
    return st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
        key="conf_slider",
    )


def render_action_buttons() -> Tuple[bool, bool]:
    """
    Render Start / Stop detection buttons.

    Returns
    -------
    tuple[bool, bool]
        (start_clicked, stop_clicked)
    """
    col1, col2 = st.sidebar.columns(2)
    start = col1.button("Start", use_container_width=True, key="btn_start")
    stop = col2.button("Stop", use_container_width=True, key="btn_stop")
    return start, stop


def render_export_buttons() -> Tuple[bool, bool]:
    """
    Render Save Output / Export CSV buttons.

    Returns
    -------
    tuple[bool, bool]
        (save_clicked, export_csv_clicked)
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Export")
    col1, col2 = st.sidebar.columns(2)
    save = col1.button("Save Output", use_container_width=True, key="btn_save")
    export = col2.button("Export CSV", use_container_width=True, key="btn_export")
    return save, export


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def render_status_card(
    bin_id: int,
    fill_pct: float,
    status: FillStatus,
    confidence: float,
) -> None:
    """Render a single colour-coded bin status card in the main area."""
    color = STATUS_COLORS_HEX.get(status, "#888888")

    st.markdown(
        f"""
        <div style="
            background: #ffffff;
            border-left: 5px solid {color};
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="color:#555; font-size:13px; text-transform:uppercase; letter-spacing:1px;">
                        Bin #{bin_id}
                    </span>
                    <h2 style="color:{color}; margin:4px 0 2px 0; font-size:28px;">
                        {fill_pct:.0f}%
                    </h2>
                    <span style="
                        background:{color}22;
                        color:{color};
                        padding:3px 12px;
                        border-radius:20px;
                        font-size:13px;
                        font-weight:600;
                    ">{status.value}</span>
                </div>
                <div style="text-align:right;">
                    <span style="color:#555; font-size:12px;">Confidence</span>
                    <p style="color:#222; font-size:20px; margin:2px 0; font-weight: 600;">{confidence*100:.0f}%</p>
                </div>
            </div>
            <div style="
                background:#e0e0e0;
                border-radius:6px;
                height:8px;
                margin-top:12px;
                overflow:hidden;
            ">
                <div style="
                    background:{color};
                    width:{min(fill_pct, 100):.0f}%;
                    height:100%;
                    border-radius:6px;
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fps_indicator(fps: float, latency_ms: float = 0.0) -> None:
    """Render a live FPS and latency badge in the sidebar."""
    if fps >= 15:
        color = "#00C800"
    elif fps >= 10:
        color = "#DCDC00"
    else:
        color = "#F00000"

    st.sidebar.markdown(
        f"""
        <div style="
            background:#f0f2f6;
            border-radius:8px;
            padding:8px 14px;
            text-align:center;
            margin-top:8px;
            border: 1px solid #e0e0e0;
        ">
            <span style="color:{color}; font-size:22px; font-weight:700;">
                {fps:.1f} FPS
            </span>
            <br/>
            <span style="color:#555; font-size:12px;">
                Latency: {latency_ms:.0f}ms
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_system_header() -> None:
    """Render the application title and branding bar."""
    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #4CAF50, #2E7D32);
            border-radius: 14px;
            padding: 24px 32px;
            margin-bottom: 24px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(76,175,80,0.2);
        ">
            <h1 style="
                color: #ffffff;
                font-size: 32px;
                margin: 0 0 6px 0;
                letter-spacing: 1px;
            ">
                Waste Vision System
            </h1>
            <p style="
                color: #e8f5e9;
                font-size: 14px;
                margin: 0;
            ">
                AI-Powered Trash Bin Occupancy Monitoring &amp; Classification
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
