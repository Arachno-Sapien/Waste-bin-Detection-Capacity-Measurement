"""
Data Logger — CSV export and snapshot persistence
===================================================
Stores per-frame occupancy readings in an in-memory DataFrame with
periodic auto-flush to disk, and supports timestamped JPEG snapshots.
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

    Maintains an in-memory DataFrame and flushes to CSV at configurable
    intervals.  Also supports saving annotated frame snapshots.
    """

    _COLUMNS = [
        "timestamp",
        "bin_id",
        "fill_pct",
        "status",
        "confidence",
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records: list[dict] = []
        self._flush_counter: int = 0
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

        self._flush_counter += 1
        if self._flush_counter >= self._settings.csv_flush_interval:
            self._auto_flush()

    def export_csv(self, path: Optional[str] = None) -> Path:
        """
        Export the full audit log to a CSV file.

        Parameters
        ----------
        path : str, optional
            Custom output path.  Defaults to the session CSV inside exports/csv/.

        Returns
        -------
        Path
            Absolute path to the written CSV file.
        """
        out = Path(path) if path else self._csv_path
        out.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._records, columns=self._COLUMNS)
        df.to_csv(out, index=False)
        logger.info("Exported CSV with %d entries to %s", len(df), out)
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
        """Return the current records as a DataFrame."""
        return pd.DataFrame(self._records, columns=self._COLUMNS)

    @property
    def entry_count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        """Discard all in-memory records."""
        self._records.clear()
        self._flush_counter = 0

    # ---- Internal ----------------------------------------------------------

    def _auto_flush(self) -> None:
        """Periodically write records to disk to prevent memory buildup."""
        try:
            self.export_csv()
            self._flush_counter = 0
        except Exception as e:
            logger.error("Auto-flush failed: %s", e)
