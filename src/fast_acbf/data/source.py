"""Raw 4D dataset access. Minimal concrete class for now."""

from __future__ import annotations

import numpy as np


class ArrayDatasetSource:
    """Wraps a (Ry, Rx, Ky, Kx) numpy array as a dataset source."""

    def __init__(self, array: np.ndarray) -> None:
        if array.ndim != 4:
            raise ValueError(f"Dataset must be 4D (Ry, Rx, Ky, Kx), got shape {array.shape}.")
        self._array = array

    @property
    def scan_shape(self) -> tuple[int, int]:
        return (self._array.shape[0], self._array.shape[1])

    @property
    def detector_shape(self) -> tuple[int, int]:
        return (self._array.shape[2], self._array.shape[3])

    def get_array(self) -> np.ndarray:
        return self._array

    def crop_scan_roi(self, y0: int, y1: int, x0: int, x1: int) -> ArrayDatasetSource:
        """Return a new source backed by a scan-space crop (contiguous copy)."""
        cropped = np.ascontiguousarray(self._array[y0:y1, x0:x1])
        return ArrayDatasetSource(cropped)
