from .solver import BFSolver
from .core.aberrations import AberrationState
from .optimization.metrics import QualityMetrics
from .data.dataset4d import Dataset4D
from .data.geometry import DetectorGeometry, ScanGeometry, CoordinateTransform

__version__ = "0.2.0" # 2026.05.10

__all__ = [
    "BFSolver",
    "AberrationState",
    "QualityMetrics",
    "Dataset4D",
    "DetectorGeometry",
    "ScanGeometry",
    "CoordinateTransform",
]