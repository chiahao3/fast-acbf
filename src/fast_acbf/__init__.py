from .solver import BFSolver
from .core.aberrations import AberrationState
from .optimization.metrics import QualityMetrics
from .data.source import ArrayDatasetSource
from .data.geometry import DetectorGeometry, ScanGeometry, CoordinateTransform

__version__ = "0.2.0" # 2026.05.10

__all__ = [
    "BFSolver",
    "AberrationState",
    "QualityMetrics",
    "ArrayDatasetSource",
    "DetectorGeometry",
    "ScanGeometry",
    "CoordinateTransform",
]