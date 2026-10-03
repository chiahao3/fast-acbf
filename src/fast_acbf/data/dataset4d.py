"""Dataset4D raw source abstraction for (Ry, Rx, Ky, Kx) data."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch


class Dataset4D:
    """Owns raw 4D-STEM data and backend lifecycle.

    ``Dataset4D`` deliberately has no BF-aperture or FFT policy.  It is either
    materialized as a contiguous float32 numpy array in host RAM, or it wraps a
    lazy HDF5/Zarr/memory-mapped raw backend and exposes generic raw reads for higher pipeline
    layers.

    Normalization invariant:
      - Materialized datasets store normalized data in ``_array``.
      - Lazy datasets leave disk data untouched and apply normalization on reads.
    """

    def __init__(self, array: np.ndarray | torch.Tensor, *, normalize: bool = False) -> None:
        if isinstance(array, torch.Tensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.detach().contiguous().numpy()
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(array)}.")
        if array.ndim != 4:
            raise ValueError(f"Dataset must be 4D (Ry, Rx, Ky, Kx), got shape {array.shape}.")

        self._array: np.ndarray | None = np.ascontiguousarray(array, dtype=np.float32)
        self._handle: Any | None = None
        self._h5_file: Any | None = None
        self._norm_factor: float | None = None

        if normalize:
            self._apply_normalization()

    @classmethod
    def from_hdf5(
        cls,
        path,
        key: str = 'data',
        *,
        materialize: bool = False,
        normalize: bool = False,
    ) -> Dataset4D:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for from_hdf5().") from exc

        obj = cls.__new__(cls)
        obj._array = None
        obj._h5_file = h5py.File(path, 'r')
        obj._handle = obj._h5_file[key]
        obj._norm_factor = None
        obj._validate_shape(obj._handle.shape, f"HDF5 dataset {key!r}")

        if materialize:
            obj.materialize()
        if normalize:
            try:
                obj._apply_normalization()
            except Exception:
                obj.close()
                raise
        return obj

    @classmethod
    def from_zarr(
        cls,
        path,
        key: str = 'data',
        *,
        materialize: bool = False,
        normalize: bool = False,
    ) -> Dataset4D:
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("zarr is required for from_zarr().") from exc

        obj = cls.__new__(cls)
        obj._array = None
        obj._h5_file = None
        store = zarr.open(path, mode='r')
        obj._handle = store[key]
        obj._norm_factor = None
        obj._validate_shape(obj._handle.shape, f"Zarr array {key!r}")

        if materialize:
            obj.materialize()
        if normalize:
            try:
                obj._apply_normalization()
            except Exception:
                obj.close()
                raise
        return obj

    @classmethod
    def from_raw(
        cls,
        path,
        scan_shape: tuple[int, int],
        detector_shape: tuple[int, int],
        *,
        dtype=np.float32,
        offset: int = 0,
        gap: int = 1024,
        materialize: bool = False,
        normalize: bool = False,
    ) -> Dataset4D:
        """Open a headerless binary file of frames, e.g. EMPAD ``.raw``.

        The file holds ``Ry * Rx`` frames of ``(Ky, Kx)`` pixels in scan order, with
        ``gap`` bytes after each frame (EMPAD stores 1024 bytes of metadata per frame;
        use ``gap=0`` for gapless data) and ``offset`` bytes before the first frame.
        The file is memory-mapped, so it stays lazy until ``materialize()``.
        """
        Ry, Rx = (int(v) for v in scan_shape)
        Ky, Kx = (int(v) for v in detector_shape)
        frame = np.dtype([('data', dtype, (Ky, Kx)), ('gap', np.uint8, (gap,))])
        expected = offset + Ry * Rx * frame.itemsize
        actual = os.path.getsize(path)
        if actual != expected:
            raise ValueError(
                f"Raw file {str(path)!r} is {actual} bytes, expected {expected} = offset "
                f"{offset} + {Ry}*{Rx} frames * ({Ky}*{Kx}*{np.dtype(dtype).itemsize} + gap "
                f"{gap}). Check scan_shape, detector_shape, dtype, offset, and gap."
            )

        obj = cls.__new__(cls)
        obj._array = None
        obj._h5_file = None
        frames = np.memmap(path, dtype=frame, mode='r', offset=offset, shape=(Ry * Rx,))
        obj._handle = frames['data'].reshape(Ry, Rx, Ky, Kx)
        obj._norm_factor = None

        if materialize:
            obj.materialize()
        if normalize:
            obj._apply_normalization()
        return obj

    @staticmethod
    def _validate_shape(shape: tuple[int, ...], label: str = "Dataset") -> None:
        if len(shape) != 4:
            raise ValueError(f"{label} must be 4D (Ry, Rx, Ky, Kx), got shape {shape}.")

    @property
    def is_lazy(self) -> bool:
        return self._handle is not None

    @property
    def norm_factor(self) -> float | None:
        return self._norm_factor

    @property
    def shape(self) -> tuple[int, int, int, int]:
        s = self._handle.shape if self.is_lazy else self._array.shape
        return tuple(int(v) for v in s)

    @property
    def scan_shape(self) -> tuple[int, int]:
        Ry, Rx, _, _ = self.shape
        return Ry, Rx

    @property
    def detector_shape(self) -> tuple[int, int]:
        _, _, Ky, Kx = self.shape
        return Ky, Kx

    @property
    def backend_chunks(self) -> tuple[int, int, int, int] | None:
        """Return backend chunk shape when available, otherwise ``None``."""
        handle = self._handle
        return None if handle is None else getattr(handle, 'chunks', None)

    @property
    def nbytes_float32(self) -> int:
        Ry, Rx, Ky, Kx = self.shape
        return int(Ry * Rx * Ky * Kx * 4)

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def raw_array(self) -> np.ndarray:
        """Return materialized raw 4D array; lazy datasets must materialize first."""
        if self.is_lazy:
            raise RuntimeError("Dataset4D is lazy. Call materialize() before raw_array().")
        return self._array

    def materialize(self) -> np.ndarray:
        """Load a lazy backend into normalized contiguous float32 host RAM."""
        if not self.is_lazy:
            return self._array

        shape = self._handle.shape
        size_bytes = self.nbytes_float32
        size_gib = size_bytes / 2**30
        try:
            import psutil
            available = psutil.virtual_memory().available
            if size_bytes > available:
                raise RuntimeError(
                    f"Cannot materialize disk-backed Dataset4D: shape {shape} requires "
                    f"{size_gib:.2f} GiB as float32, but only {available / 2**30:.2f} GiB "
                    "RAM is available."
                )
        except ImportError:
            pass

        try:
            arr = np.asarray(self._handle[:], dtype=np.float32)
        except MemoryError as exc:
            raise RuntimeError(
                f"Cannot materialize disk-backed Dataset4D: shape {shape} requires "
                f"{size_gib:.2f} GiB as float32."
            ) from exc

        self._array = np.ascontiguousarray(arr)
        if self._norm_factor is not None:
            self._array = self._array / np.float32(self._norm_factor)
        self.close()
        return self._array

    def read_raw(self, selection) -> np.ndarray:
        """Read an arbitrary raw backend selection as float32.

        For materialized datasets this returns from ``_array`` directly.  For
        lazy datasets normalization is applied after the read.
        """
        if self.is_lazy:
            out = np.asarray(self._handle[selection], dtype=np.float32)
            if self._norm_factor is not None:
                out = out / np.float32(self._norm_factor)
            return out
        return np.asarray(self._array[selection], dtype=np.float32)

    def iter_scan_rows(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(scan_y, row)`` where row has shape ``(Rx, Ky, Kx)``."""
        Ry, _ = self.scan_shape
        for scan_y in range(Ry):
            yield scan_y, self.read_raw((scan_y, slice(None), slice(None), slice(None)))

    def _compute_lazy_norm_factor(self) -> float:
        Ry, Rx, Ky, Kx = self._handle.shape
        if Ry == 0 or Rx == 0:
            raise ValueError("Cannot normalize: dataset has zero scan positions.")
        accum = np.zeros((Ky, Kx), dtype=np.float32)
        for _, row in self.iter_scan_rows():
            accum += row.sum(axis=0, dtype=np.float32)
        mean_dp = accum / np.float32(Ry * Rx)
        return float(mean_dp.max())

    def _apply_normalization(self) -> None:
        if self._norm_factor is not None:
            raise RuntimeError("Normalize only once at Dataset4D construction.")

        if self.is_lazy:
            factor = self._compute_lazy_norm_factor()
        else:
            Ry, Rx = self._array.shape[:2]
            if Ry == 0 or Rx == 0:
                raise ValueError("Cannot normalize: dataset has zero scan positions.")
            factor = float(self._array.mean(axis=(0, 1), dtype=np.float32).max())

        if not (factor > 0 and np.isfinite(factor)):
            raise ValueError(
                f"Cannot normalize: norm_factor={factor!r} is not finite and positive."
            )

        if not self.is_lazy:
            self._array = self._array / np.float32(factor)
        self._norm_factor = factor

    def crop_scan_roi(self, y0: int, y1: int, x0: int, x1: int) -> Dataset4D:
        """Return a materialized scan-space crop that preserves global scale."""
        cropped = self.read_raw((slice(y0, y1), slice(x0, x1), slice(None), slice(None)))
        new_ds = Dataset4D(cropped)
        new_ds._norm_factor = self._norm_factor
        return new_ds
