"""
FPS Counter — Latency & throughput tracking
============================================
Lightweight performance instrumentation for real-time video processing.
Maintains a sliding window of frame timestamps and computes smoothed FPS
and per-frame latency.
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Track frames-per-second and per-frame latency using a sliding window."""

    def __init__(self, window_size: int = 30) -> None:
        """
        Args:
            window_size: Number of recent frame timestamps to keep for
                         computing the rolling average FPS.
        """
        self._window_size = max(2, window_size)
        self._timestamps: deque[float] = deque(maxlen=self._window_size)
        self._last_tick: float = 0.0
        self._last_latency: float = 0.0

    def tick(self) -> None:
        """Record the timestamp of the current frame."""
        now = time.perf_counter()
        if self._last_tick > 0:
            self._last_latency = now - self._last_tick
        self._last_tick = now
        self._timestamps.append(now)

    @property
    def fps(self) -> float:
        """Return smoothed FPS over the sliding window."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    @property
    def latency_ms(self) -> float:
        """Return the processing time of the most recent frame in ms."""
        return self._last_latency * 1000.0

    def reset(self) -> None:
        """Clear all recorded timestamps."""
        self._timestamps.clear()
        self._last_tick = 0.0
        self._last_latency = 0.0
