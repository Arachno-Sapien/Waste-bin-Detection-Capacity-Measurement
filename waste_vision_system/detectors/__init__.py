# waste_vision_system/detectors
"""Detection module — bin localization and waste classification."""

from .bin_detector import BinDetector, BinDetection
from .waste_detector import WasteDetector, WasteDetection

__all__ = ["BinDetector", "BinDetection", "WasteDetector", "WasteDetection"]
