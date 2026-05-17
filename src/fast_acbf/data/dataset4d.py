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

    Lazy read strategies (``lazy_read_mode``):
      ``'per_pixel'`` — original baseline: per-detector-pixel loop, each call
          reads ``handle[:,:,ky,kx]``. For contiguous C-order HDF5 storage
          (Ry,Rx,Ky,Kx) this produces a highly strided read (stride = Ky*Kx*4 B
          between consecutive Rx elements), which is slow on spinning disks.
          Optimal only when chunks are detector-major ``(Ry,Rx,1,1)``.

      ``'scan_row'`` — sequential scan-row gather: loop over Ry scan rows reading
          ``handle[iy,:,:,:]`` — a contiguous (Rx,Ky,Kx) block per row — then
          extract the requested BF pixels. Best for contiguous or scan-major
          ``(1,1,Ky,Kx)`` storage where per-pixel reads are catastrophically
          slow; reads the full dataset volume but sequentially.

      ``'slab'`` — ky bounding-box hyperslab: reads
          ``handle[:,:,ky_min:ky_max,:]`` in one HDF5 hyperslab call, covering
          the smallest ky-range that contains all BF pixels, then extracts with
          numpy fancy indexing. Reads less than scan_row for small BF disks but
          more than per_pixel; access is strided at the (ry,rx) level but
          contiguous within each (ky,kx) tile.

      ``'auto'`` — inspect ``_handle.chunks`` (HDF5) or chunk sizes (zarr) and
          pick: ``slab`` for contiguous C-order (chunks=None); ``per_pixel``
          for detector-major (Ry,Rx,1,1); ``scan_row`` for scan-major
          (1,1,Ky,Kx); ``per_pixel`` otherwise (safe default).

    Normalization invariant:
      In-memory (is_lazy=False): _array is already divided by _norm_factor.
        Getters return _array directly — _norm_factor is documentary metadata only.
      Lazy (is_lazy=True): disk data is untouched. Getters divide on the fly.
      _norm_factor=None means no normalization was requested.
    """

    LAZY_READ_MODES = ('auto', 'per_pixel', 'scan_row', 'slab')

    # --- Array-backed constructor ---
    def __init__(self, array: np.ndarray | torch.Tensor, *, normalize: bool = False) -> None:
        """
        Args:
            array: (Ry, Rx, Ky, Kx) array or tensor.
            normalize: If True, divide all data by the maximum of the mean diffraction
                pattern (PACBED). For a full dataset, PACBED max ≈ 1 after normalization.
                Disk data is never modified. The factor is accessible via norm_factor.
        """
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
        self._norm_factor: float | None = None
        self._lazy_read_mode: str = 'per_pixel'
        if normalize:
            self._apply_normalization()

    @property
    def is_lazy(self) -> bool:
        return self._handle is not None

    @property
    def norm_factor(self) -> float | None:
        """Normalization factor applied at construction, or None if normalize=False."""
        return self._norm_factor

    @property
    def lazy_read_mode(self) -> str:
        """Resolved lazy read strategy ('per_pixel', 'scan_row', or 'slab').

        For in-memory datasets this is always ``'per_pixel'`` (unused).
        For lazy backends this reflects the strategy selected at construction
        time (either the explicit value or the auto-detected one).
        """
        return self._lazy_read_mode

    # --- Lazy constructors ---
    @classmethod
    def from_hdf5(
        cls,
        path,
        key: str = 'data',
        *,
        materialize: bool = False,
        normalize: bool = False,
        lazy_read_mode: str = 'auto',
    ) -> Dataset4D:
        """
        Args:
            path: Path to HDF5 file.
            key: Dataset key inside the file.
            materialize: If True, load the full array into RAM immediately.
            normalize: If True, divide all data by the maximum of the mean diffraction
                pattern (PACBED). For a full dataset, PACBED max ≈ 1 after normalization.
                Disk data is never modified. The factor is accessible via norm_factor.
            lazy_read_mode: Strategy for lazy get_bf_chunk reads. One of
                ``'auto'`` (default), ``'per_pixel'``, ``'scan_row'``, ``'slab'``.
                See class docstring for trade-offs. ``'auto'`` inspects the
                HDF5 chunk layout to pick the best strategy.
        """
        if lazy_read_mode not in cls.LAZY_READ_MODES:
            raise ValueError(
                f"lazy_read_mode must be one of {cls.LAZY_READ_MODES}, got {lazy_read_mode!r}"
            )
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for from_hdf5(). Install it with: pip install h5py")
        obj = cls.__new__(cls)
        obj._array = None
        obj._h5_file = h5py.File(path, 'r')
        obj._handle = obj._h5_file[key]
        obj._norm_factor = None  # must be set before _force_materialize()
        if obj._handle.ndim != 4:
            raise ValueError(f"HDF5 dataset '{key}' must be 4D, got shape {obj._handle.shape}")
        obj._lazy_read_mode = (
            obj._detect_lazy_read_mode() if lazy_read_mode == 'auto' else lazy_read_mode
        )
        if materialize:
            obj._force_materialize()
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
        lazy_read_mode: str = 'auto',
    ) -> Dataset4D:
        """
        Args:
            path: Path to zarr store.
            key: Array key inside the store.
            materialize: If True, load the full array into RAM immediately.
            normalize: If True, divide all data by the maximum of the mean diffraction
                pattern (PACBED). For a full dataset, PACBED max ≈ 1 after normalization.
                Disk data is never modified. The factor is accessible via norm_factor.
            lazy_read_mode: Strategy for lazy get_bf_chunk reads. One of
                ``'auto'`` (default), ``'per_pixel'``, ``'scan_row'``, ``'slab'``.
                See class docstring for trade-offs.
        """
        if lazy_read_mode not in cls.LAZY_READ_MODES:
            raise ValueError(
                f"lazy_read_mode must be one of {cls.LAZY_READ_MODES}, got {lazy_read_mode!r}"
            )
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
        obj._norm_factor = None  # must be set before _force_materialize()
        obj._lazy_read_mode = (
            obj._detect_lazy_read_mode() if lazy_read_mode == 'auto' else lazy_read_mode
        )
        if materialize:
            obj._force_materialize()
        if normalize:
            try:
                obj._apply_normalization()
            except Exception:
                obj.close()
                raise
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
            result = np.asarray(self._handle[:, :, ky, kx], dtype=np.float32)
            if self._norm_factor is not None:
                result = result / np.float32(self._norm_factor)
            return result
        return self._array[:, :, ky, kx]

    def get_bf_chunk(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """Return (chunk_size, Ry, Rx) float32 for a batch of detector coords.

        In-memory path: numpy fancy indexing — fast, one pass.
        Lazy path: dispatch to strategy set by lazy_read_mode at construction
            (``'per_pixel'``, ``'scan_row'``, or ``'slab'``).
        """
        if self.is_lazy:
            if self._lazy_read_mode == 'scan_row':
                return self._get_bf_chunk_scan_row(iy, ix)
            if self._lazy_read_mode == 'slab':
                return self._get_bf_chunk_slab(iy, ix)
            return self._get_bf_chunk_per_pixel(iy, ix)
        raw = self._array[:, :, iy, ix]            # (Ry, Rx, chunk_size) view
        return np.ascontiguousarray(raw.transpose(2, 0, 1), dtype=np.float32)

    # --- Lazy read strategy implementations ---

    def _get_bf_chunk_per_pixel(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """Original strategy: one handle[:,:,ky,kx] read per BF pixel.

        Access pattern for C-order contiguous storage: stride = Ky*Kx*4 bytes
        between consecutive Rx elements — highly non-sequential.
        Best for detector-major HDF5 chunking (Ry,Rx,1,1) where each pixel is
        one decompression unit.
        """
        raw = np.stack(
            [np.asarray(self._handle[:, :, int(i), int(j)], dtype=np.float32)
             for i, j in zip(iy, ix)], axis=0
        )
        if self._norm_factor is not None:
            raw = raw / np.float32(self._norm_factor)
        return raw

    def _get_bf_chunk_scan_row(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """Scan-row gather: read one full (Rx,Ky,Kx) row at a time.

        Reads Ry sequential blocks of Rx*Ky*Kx*4 bytes each (= full dataset).
        Access is sequential → better for contiguous or scan-major storage.
        Trades extra I/O volume for sequential access patterns.
        """
        Ry, Rx = self.scan_shape
        chunk_size = len(iy)
        out = np.empty((chunk_size, Ry, Rx), dtype=np.float32)
        for scan_y in range(Ry):
            # (Rx, Ky, Kx) — one contiguous block per scan row
            row = np.asarray(self._handle[scan_y, :, :, :], dtype=np.float32)
            # row[:, iy, ix] → (Rx, chunk_size) via element-wise fancy index
            out[:, scan_y, :] = row[:, iy, ix].T
        if self._norm_factor is not None:
            out /= np.float32(self._norm_factor)
        return out

    def _get_bf_chunk_slab(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """Ky bounding-box hyperslab: read handle[:,:,ky_min:ky_max,:] once.

        Reads a slab bounded by the ky-extent of the requested pixels across
        all scan positions and all kx.  For a BF disk that spans only a fraction
        of the detector height, this reads less data than scan_row while using a
        single HDF5 hyperslab call instead of Ry separate calls.
        """
        ky_min = int(iy.min())
        ky_max = int(iy.max()) + 1
        # (Ry, Rx, slab_Ky, Kx) — one hyperslab read
        slab = np.asarray(
            self._handle[:, :, ky_min:ky_max, :], dtype=np.float32
        )
        iy_local = iy - ky_min
        # slab[:, :, iy_local, ix] → (Ry, Rx, chunk_size) via fancy indexing
        raw = slab[:, :, iy_local, ix]
        out = np.ascontiguousarray(raw.transpose(2, 0, 1))
        if self._norm_factor is not None:
            out = out / np.float32(self._norm_factor)
        return out

    def _detect_lazy_read_mode(self) -> str:
        """Inspect handle chunk layout and return the best lazy_read_mode.

        Rules:
          - h5py chunks=None (contiguous C-order): 'slab' — each chunk call
            reads a small ky bounding-box (~2 rows for sorted BF pixels),
            minimises h5py call count while keeping per-call I/O tiny.
            scan_row is 2× faster for single full-pass calls but requires
            all N_bf pixels in one call; slab is better for the chunked
            reconstruction loop.
          - Detector-major chunks (c2==1 and c3==1): 'per_pixel' — each
            (ky,kx) pixel is its own decompression unit.
          - Scan-major chunks (c0==1 and c1==1): 'scan_row' — each diffraction
            pattern is one chunk; reading by scan row amortises decompression.
          - Mixed/unknown chunking: 'per_pixel' (safe conservative default).
        """
        handle = self._handle
        # h5py.Dataset exposes chunks as None (contiguous) or a tuple.
        # zarr.Array exposes chunks as a tuple (always chunked).
        chunks = getattr(handle, 'chunks', None)
        if chunks is None:
            # h5py contiguous — adaptive ky-slab is best for chunked access
            return 'slab'
        c0, c1, c2, c3 = chunks
        if c2 == 1 and c3 == 1:
            return 'per_pixel'   # detector-major: one chunk per (ky,kx) image
        if c0 == 1 and c1 == 1:
            return 'scan_row'    # scan-major: one chunk per diffraction pattern
        return 'per_pixel'

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

        If _norm_factor is already set (lazy-normalized dataset), the loaded array is
        divided by _norm_factor so the in-memory invariant holds. The load reads
        self._handle[:] directly — never via getters — to avoid double-normalization.
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
        if self._norm_factor is not None:
            self._array = self._array / np.float32(self._norm_factor)
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        self._handle = None
        return self._array

    # --- Normalization ---
    def _compute_lazy_norm_factor(self) -> float:
        """Row-by-row PACBED mean without loading the full 4D array into RAM."""
        Ry, Rx, Ky, Kx = self._handle.shape
        if Ry == 0 or Rx == 0:
            raise ValueError("Cannot normalize: dataset has zero scan positions.")
        accum = np.zeros((Ky, Kx), dtype=np.float32)
        for iy in range(Ry):
            row = np.asarray(self._handle[iy, :, :, :], dtype=np.float32)  # (Rx, Ky, Kx)
            accum += row.sum(axis=0, dtype=np.float32)
        mean_dp = accum / np.float32(Ry * Rx)
        return float(mean_dp.max())

    def _apply_normalization(self) -> None:
        """Compute and apply PACBED-max normalization. Call once at construction only."""
        if self._norm_factor is not None:
            raise RuntimeError(
                "_apply_normalization() called on an already-normalized Dataset4D. "
                "Normalize only once at construction."
            )
        if self.is_lazy:
            factor = self._compute_lazy_norm_factor()
        else:
            Ry, Rx = self._array.shape[:2]
            if Ry == 0 or Rx == 0:
                raise ValueError("Cannot normalize: dataset has zero scan positions.")
            mean_dp = self._array.mean(axis=(0, 1), dtype=np.float32)  # (Ky, Kx)
            factor = float(mean_dp.max())
        if not (factor > 0 and np.isfinite(factor)):
            raise ValueError(
                f"Cannot normalize: norm_factor={factor!r} is not a finite positive number. "
                "Check that the dataset contains valid, non-zero intensities."
            )
        if not self.is_lazy:
            # Non-in-place division ensures we never mutate the caller's array
            # (np.ascontiguousarray may return the same object for contiguous float32 inputs).
            self._array = self._array / np.float32(factor)
        self._norm_factor = factor

    def crop_scan_roi(self, y0: int, y1: int, x0: int, x1: int) -> Dataset4D:
        """Return a new in-memory Dataset4D cropped in scan space.

        For lazy backends, the cropped region is loaded to RAM. This is
        acceptable because ROI refinement targets small subregions.

        The child dataset uses the parent's global normalization factor (same absolute
        scale), not a re-normalized PACBED. norm_factor is propagated as metadata.
        """
        if self.is_lazy:
            cropped = np.asarray(self._handle[y0:y1, x0:x1, :, :], dtype=np.float32)
            if self._norm_factor is not None:
                cropped = cropped / np.float32(self._norm_factor)
        else:
            cropped = np.ascontiguousarray(self._array[y0:y1, x0:x1])
            # _array already normalized; slice inherits the correct scale
        new_ds = Dataset4D(cropped)
        new_ds._norm_factor = self._norm_factor
        return new_ds
