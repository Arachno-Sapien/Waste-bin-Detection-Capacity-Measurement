"""
Stream Handler — Resilient video capture with RTSP reconnection
=================================================================
Wraps ``cv2.VideoCapture`` to support file, webcam, and RTSP sources
with automatic reconnection, frame-skipping, and buffer flushing.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from config.settings import Settings

logger = logging.getLogger(__name__)


class StreamHandler:
    """
    Unified video source handler supporting files, webcams, and RTSP streams.

    Features:
    - Automatic RTSP reconnection with exponential backoff.
    - Frame-skipping for performance optimisation.
    - Buffer flushing for RTSP to always get the latest frame.
    - Thread-safe open / read / release operations.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cap: Optional[cv2.VideoCapture] = None
        self._source: Union[str, int, None] = None
        self._is_rtsp: bool = False
        self._is_file: bool = False
        self._lock = threading.Lock()
        self._reconnect_thread: Optional[threading.Thread] = None
        self._reconnecting = threading.Event()
        self._stop_event = threading.Event()
        self._frame_count: int = 0
        self._total_frames: int = 0  # For video files

    # ---- Public API --------------------------------------------------------

    def open(self, source: Union[str, int]) -> bool:
        """
        Open a video source.

        Parameters
        ----------
        source : str or int
            File path, webcam index, or RTSP URI.

        Returns
        -------
        bool
            True if the source was opened successfully.
        """
        with self._lock:
            self.release_internal()
            self._source = source
            self._is_rtsp = isinstance(source, str) and (
                source.lower().startswith("rtsp://")
                or source.lower().startswith("rtsps://")
            )
            self._is_file = isinstance(source, str) and not self._is_rtsp
            self._frame_count = 0
            self._stop_event.clear()

            return self._open_capture()

    def read(self, skip_frames: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame from the source.

        Parameters
        ----------
        skip_frames : bool
            If True, implements frame-skipping strategy for non-file sources.

        Returns
        -------
        tuple[bool, np.ndarray | None]
            (success, frame) — frame is None on failure.
        """
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                if self._is_rtsp and not self._reconnecting.is_set():
                    self._start_reconnect()
                return False, None

            # Frame skipping for live sources
            if skip_frames and not self._is_file:
                skip_n = self._settings.frame_skip
                for _ in range(skip_n - 1):
                    self._cap.grab()

            ret, frame = self._cap.read()
            if not ret:
                if self._is_rtsp:
                    logger.warning("RTSP frame read failed — initiating reconnect")
                    self._start_reconnect()
                return False, None

            self._frame_count += 1
            return True, frame

    def release(self) -> None:
        """Release the video source and stop any reconnection threads."""
        with self._lock:
            self._stop_event.set()
            self.release_internal()

    @property
    def is_opened(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    @property
    def is_reconnecting(self) -> bool:
        return self._reconnecting.is_set()

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def total_frames(self) -> int:
        """Total frames in a video file; 0 for live sources."""
        return self._total_frames

    @property
    def fps(self) -> float:
        """Source FPS (from metadata), or 30.0 as default."""
        with self._lock:
            if self._cap is not None:
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                return fps if fps > 0 else 30.0
            return 30.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of the video source."""
        with self._lock:
            if self._cap is not None:
                w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                return (w, h)
            return (0, 0)

    # ---- Internal helpers --------------------------------------------------

    def _open_capture(self) -> bool:
        """Open the cv2.VideoCapture (must be called under lock)."""
        try:
            if isinstance(self._source, int):
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            else:
                self._cap = cv2.VideoCapture(self._source)

            if not self._cap.isOpened():
                logger.error("Failed to open source: %s", self._source)
                return False

            # Configure buffer for RTSP
            if self._is_rtsp:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._settings.rtsp_buffer_size)

            # Total frames for video files
            if self._is_file:
                self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self._total_frames = 0

            logger.info("Opened source: %s  (RTSP=%s, File=%s)",
                        self._source, self._is_rtsp, self._is_file)
            return True

        except Exception as e:
            logger.error("Exception opening source %s: %s", self._source, e)
            return False

    def release_internal(self) -> None:
        """Release capture (must be called under lock)."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _start_reconnect(self) -> None:
        """Spawn a background reconnection thread for RTSP."""
        if self._reconnecting.is_set():
            return  # Already reconnecting
        self._reconnecting.set()
        self.release_internal()

        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, daemon=True
        )
        self._reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        """Background thread: attempt RTSP reconnection with exponential backoff."""
        delay = self._settings.rtsp_reconnect_interval
        max_delay = self._settings.rtsp_reconnect_max

        while not self._stop_event.is_set():
            logger.info("Attempting RTSP reconnection in %.1fs ...", delay)
            time.sleep(delay)

            with self._lock:
                if self._stop_event.is_set():
                    break
                success = self._open_capture()
                if success:
                    logger.info("RTSP reconnection successful")
                    self._reconnecting.clear()
                    return

            # Exponential backoff
            delay = min(delay * 2, max_delay)

        self._reconnecting.clear()
