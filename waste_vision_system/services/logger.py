"""
Data Logger — CSV export and snapshot persistence
===================================================
Buffers per-frame occupancy readings in memory, appends them to the session
CSV periodically, and supports timestamped JPEG snapshots.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from config.settings import CSV_DIR, SNAPSHOT_DIR, FillStatus, Settings

logger = logging.getLogger(__name__)


class DataLogger:
    """
    Audit-trail logger for bin occupancy readings.

    Buffers records in memory and appends them to the session CSV every
    ``csv_flush_interval`` rows; only the recent rows stay in memory.
    Also supports saving annotated frame snapshots.
    """

    _COLUMNS = [
        "timestamp",
        "bin_id",
        "fill_pct",
        "status",
        "confidence",
    ]
    _KEEP_IN_MEMORY = 200  # Recent rows kept after a flush, for the UI audit view

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records: list[dict] = []
        self._pending: int = 0  # Newest records not yet written to disk
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = CSV_DIR / f"session_{self._session_id}.csv"

    # ---- Public API --------------------------------------------------------

    def log_entry(
        self,
        bin_id: int,
        fill_pct: float,
        status: FillStatus,
        confidence: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Record a single occupancy reading.

        Parameters
        ----------
        bin_id : int
            The bin tracking ID.
        fill_pct : float
            Smoothed fill percentage.
        status : FillStatus
            Discrete fullness state.
        confidence : float
            Detection confidence score.
        timestamp : str, optional
            ISO timestamp; defaults to ``now()``.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat(timespec="seconds")

        self._records.append({
            "timestamp": timestamp,
            "bin_id": bin_id,
            "fill_pct": round(fill_pct, 1),
            "status": status.value,
            "confidence": round(confidence, 2),
        })

        self._pending += 1
        if self._pending >= self._settings.csv_flush_interval:
            self._auto_flush()

    def export_csv(self) -> Path:
        """
        Append not-yet-written records to the session CSV inside exports/csv/
        and return its path. The file always holds the full session log.

        Appending (rather than rewriting everything each flush) keeps disk I/O
        constant on long webcam/RTSP sessions, and lets memory hold only the
        recent rows the UI shows.
        """
        out = self._csv_path
        out.parent.mkdir(parents=True, exist_ok=True)

        new = self._records[len(self._records) - self._pending:]
        pd.DataFrame(new, columns=self._COLUMNS).to_csv(
            out, mode="a", header=not out.exists(), index=False
        )
        self._pending = 0
        del self._records[:-self._KEEP_IN_MEMORY]
        logger.info("Appended %d entries to %s", len(new), out)
        return out

    def save_snapshot(
        self,
        frame: np.ndarray,
        bin_id: int,
        fill_pct: float,
        quality: int = 90,
    ) -> Path:
        """
        Save an annotated frame snapshot as a JPEG.

        Returns
        -------
        Path
            Absolute path to the saved snapshot.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"bin{bin_id}_fill{fill_pct:.0f}_{ts}.jpg"
        out_path = SNAPSHOT_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        logger.info("Saved snapshot: %s", out_path)
        return out_path

    def get_dataframe(self) -> pd.DataFrame:
        """Return the recent in-memory records as a DataFrame (full log is the CSV)."""
        return pd.DataFrame(self._records, columns=self._COLUMNS)

    @property
    def entry_count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        """Discard all in-memory records."""
        self._records.clear()
        self._pending = 0

    # ---- Internal ----------------------------------------------------------

    def _auto_flush(self) -> None:
        """Periodically write records to disk to prevent memory buildup."""
        try:
            self.export_csv()
        except Exception as e:
            logger.error("Auto-flush failed: %s", e)
