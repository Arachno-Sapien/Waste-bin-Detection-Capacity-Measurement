"""
Fill that only rises between collections.

Waste accumulates; a bin only gets emptier when someone empties it. So a
reading far below the bin's current level is a passer-by, a lighting glitch,
or a collection. A drop is accepted as a collection only once it has lasted
`collection_frames` readings in a row; anything shorter is ignored.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from config.settings import Settings


@dataclass
class _Timeline:
    recent: Deque[float]
    level: Optional[float] = None
    low_streak: int = 0


class FillTimeline:
    """Per-bin monotonic fill level with collection detection."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._bins: Dict[int, _Timeline] = {}

    def update(self, bin_id: int, raw_fill: float) -> Tuple[float, bool]:
        """Feed one raw reading. Returns (level, collected_on_this_reading)."""
        t = self._bins.setdefault(
            bin_id, _Timeline(recent=deque(maxlen=self._s.smoothing_window)))
        t.recent.append(raw_fill)
        m = float(np.median(t.recent))  # one-frame spikes never reach the level
        if t.level is None:
            t.level = m
            return t.level, False
        if m < t.level - self._s.collection_drop_pct:
            t.low_streak += 1
            if t.low_streak < self._s.collection_frames:
                return t.level, False
            t.level, t.low_streak = m, 0
            t.recent = deque([m], maxlen=self._s.smoothing_window)
            return t.level, True
        t.low_streak = 0
        t.level = max(t.level, m)
        return t.level, False

    def level(self, bin_id: int) -> Optional[float]:
        """Current level without feeding a reading (used while occluded)."""
        t = self._bins.get(bin_id)
        return None if t is None else t.level

    def reset(self, bin_id: Optional[int] = None) -> None:
        if bin_id is None:
            self._bins.clear()
        else:
            self._bins.pop(bin_id, None)
