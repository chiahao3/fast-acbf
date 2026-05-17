"""Dataset4D — wraps a (Ry, Rx, Ky, Kx) 4D dataset with array or lazy backends."""

from __future__ import annotations

import numpy as np
import torch


class Dataset4D:
    """Wraps a (Ry, Rx, Ky, Kx) 4D dataset.

    Two backends:
      Array-backed: full numpy array in RAM. get_full_array() available.
      Lazy-backed:  h5py Dataset or zarr Array — data never fully loaded.
                    get_full_array() raises RuntimeError.
                    cache_mode='device' is incompatible with lazy backends.

    Note on HDF5/zarr access patterns: get_bf_chunk reads chunk_size detector-pixel
    slices in a per-pixel loop for lazy backends. For scan-major HDF5 chunking
    (1,1,Ky,Kx), each slice decompresses Ry×Rx disk chunks. Detector-major chunking
    (Ry,Rx,1,1) gives one disk read per pixel. Performance is inherent to the storage
    format; on_the_fly mode with HDF5/zarr may be slow for scan-major layouts.
    """

    # --- Array-backed constructor ---
    def __init__(self, array: np.ndarray | torch.Tensor) -> None:
        if isinstance(array, torch.Tensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.detach().contiguous().numpy()
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(array)}")
        if array.ndim != 4:
            raise ValueError(f"Dataset must be 4D (Ry, Rx, Ky, Kx), got shape {array.shape}")
        self._array = np.ascontiguousarray(array, dtype=np.float32)
        self._handle = None
        self._h5_file = None

    @property
    def is_lazy(self) -> bool:
        return self._handle is not None

    # --- Lazy constructors ---
    @classmethod
    def from_hdf5(cls, path, key: str = 'data', *, materialize: bool = False) -> Dataset4D:
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for from_hdf5(). Install it with: pip install h5py")
        obj = cls.__new__(cls)
        obj._array = None
        obj._h5_file = h5py.File(path, 'r')
        obj._handle = obj._h5_file[key]
        if obj._handle.ndim != 4:
            raise ValueError(f"HDF5 dataset '{key}' must be 4D, got shape {obj._handle.shape}")
        if materialize:
            obj._force_materialize()
        return obj

    @classmethod
    def from_zarr(cls, path, key: str = 'data', *, materialize: bool = False) -> Dataset4D:
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr is required for from_zarr(). Install it with: pip install zarr")
        obj = cls.__new__(cls)
        obj._array = None
        store = zarr.open(path, mode='r')
        obj._handle = store[key]
        if obj._handle.ndim != 4:
            raise ValueError(f"Zarr array '{key}' must be 4D, got shape {obj._handle.shape}")
        obj._h5_file = None
        if materialize:
            obj._force_materialize()
        return obj

    def close(self) -> None:
        """Release file handles for lazy backends. Safe to call multiple times."""
        if hasattr(self, '_h5_file') and self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # --- Properties ---
    @property
    def scan_shape(self) -> tuple[int, int]:
        s = self._handle.shape if self.is_lazy else self._array.shape
        return (int(s[0]), int(s[1]))

    @property
    def detector_shape(self) -> tuple[int, int]:
        s = self._handle.shape if self.is_lazy else self._array.shape
        return (int(s[2]), int(s[3]))

    # --- Data access ---
    def get_virtual_img(self, ky: int, kx: int) -> np.ndarray:
        """Return (Ry, Rx) float32. Single detector pixel. Disk read for lazy."""
        if self.is_lazy:
            return np.asarray(self._handle[:, :, ky, kx], dtype=np.float32)
        return self._array[:, :, ky, kx]

    def get_bf_chunk(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """Return (chunk_size, Ry, Rx) float32 for a batch of detector coords.

        In-memory path: numpy fancy indexing — fast, one pass.
        Lazy path: per-pixel loop — slow, but disk-backed I/O is inherently slow.
        """
        if self.is_lazy:
            return np.stack(
                [np.asarray(self._handle[:, :, int(i), int(j)], dtype=np.float32)
                 for i, j in zip(iy, ix)], axis=0
            )
        raw = self._array[:, :, iy, ix]            # (Ry, Rx, chunk_size) view
        return np.ascontiguousarray(raw.transpose(2, 0, 1), dtype=np.float32)

    def get_full_array(self) -> np.ndarray:
        """Return full (Ry, Rx, Ky, Kx) float32 numpy array. In-memory backends only."""
        if self.is_lazy:
            raise RuntimeError(
                "get_full_array() is not supported for disk-backed Dataset4D. "
                "Use cache_mode='host' or 'on_the_fly', or load data into RAM first."
            )
        return self._array

    def _force_materialize(self) -> np.ndarray:
        """Load a lazy (disk-backed) dataset fully into host RAM and convert to in-memory.

        On success: self._array is populated as a contiguous float32 ndarray in host RAM,
        self._handle is cleared, and is_lazy becomes False.
        On MemoryError: raises RuntimeError with dataset size and cache_mode suggestions.
        """
        if not self.is_lazy:
            return self._array
        shape = self._handle.shape  # (Ry, Rx, Ky, Kx)
        size_gb = (shape[0] * shape[1] * shape[2] * shape[3] * 4) / (1024 ** 3)
        try:
            arr = np.asarray(self._handle[:], dtype=np.float32)
        except MemoryError:
            raise RuntimeError(
                f"Cannot materialize disk-backed Dataset4D into RAM: "
                f"dataset shape {shape} requires {size_gb:.2f} GB as float32. "
                f"Use cache_mode='host' to cache FFTs incrementally in RAM, "
                f"or cache_mode='on_the_fly' to avoid caching entirely."
            )
        self._array = np.ascontiguousarray(arr)
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        self._handle = None
        return self._array

    def crop_scan_roi(self, y0: int, y1: int, x0: int, x1: int) -> Dataset4D:
        """Return a new in-memory Dataset4D cropped in scan space.

        For lazy backends, the cropped region is loaded to RAM. This is
        acceptable because ROI refinement targets small subregions.
        """
        if self.is_lazy:
            cropped = np.asarray(self._handle[y0:y1, x0:x1, :, :], dtype=np.float32)
        else:
            cropped = np.ascontiguousarray(self._array[y0:y1, x0:x1])
        return Dataset4D(cropped)
