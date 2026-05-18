"""ImageFFTProvider — serves (chunk_size, Ry, Rx) complex64 GPU chunks to the reconstruction loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from fast_acbf.data.dataset4d import Dataset4D
    from fast_acbf.data.geometry import DetectorGeometry


class ImageFFTProvider:
    """Serves (chunk_size, Ry, Rx) complex64 GPU chunks to the reconstruction loop.

    Attributes (read-only after construction):
        dataset: Dataset4D
        detector_geom: DetectorGeometry
        device: str
        cache_mode: str  — resolved 'auto' value
        scan_shape: tuple[int, int]
        nb: int  — number of BF pixels
    """

    def __init__(
        self,
        dataset: Dataset4D,
        detector_geom: DetectorGeometry,
        device: str,
        cache_mode: str = 'auto',
    ) -> None:
        self.dataset = dataset
        self.detector_geom = detector_geom
        self.device = device

        bf_iy, bf_ix = np.where(detector_geom.bf_mask_bool)
        if len(bf_iy) == 0:
            raise ValueError("BF aperture contains zero pixels. Check max_alpha and dk.")
        self._bf_iy = bf_iy
        self._bf_ix = bf_ix
        self.nb = len(bf_iy)
        self.scan_shape = dataset.scan_shape

        self.cache_mode = self._resolve_cache_mode(cache_mode)

        self._device_cache: torch.Tensor | None = None
        self._host_cache: np.ndarray | None = None
        self._host_filled: np.ndarray | None = None

        if self.cache_mode == 'device':
            self._init_device_cache()

    def _resolve_cache_mode(self, mode: str) -> str:
        mode = mode.strip().lower()
        if mode not in ('auto', 'device', 'host', 'on_the_fly'):
            raise ValueError(
                f"cache_mode must be one of ('auto','device','host','on_the_fly'), got {mode!r}"
            )
        if mode == 'auto':
            return self._resolve_auto()
        if mode == 'device' and self.dataset.is_lazy:
            self.dataset._force_materialize()
        if mode == 'host':
            self._check_host_cache_fits()
        return mode

    def _check_host_cache_fits(self) -> None:
        """Warn if the host cache allocation is likely to exceed available RAM."""
        Ry, Rx = self.scan_shape
        cache_bytes = self.nb * Ry * Rx * 8  # (Nb, Ry, Rx) complex64
        try:
            import psutil
            available = psutil.virtual_memory().available
            if cache_bytes > available:
                raise RuntimeError(
                    f"cache_mode='host' would allocate {cache_bytes / 2**30:.1f} GiB for the FFT "
                    f"cache ({self.nb} BF pixels × {Ry}×{Rx} scan × 8 B), but only "
                    f"{available / 2**30:.1f} GiB RAM is available. "
                    "Use cache_mode='on_the_fly' to stream without caching, or reduce the "
                    "scan area (scan_roi) or aperture (max_alpha) to shrink the cache."
                )
        except ImportError:
            pass  # psutil unavailable — let the allocation fail naturally

    def _resolve_auto(self) -> str:
        Ry, Rx = self.scan_shape
        Ky, Kx = self.dataset.detector_shape
        Nb = self.nb
        fft_bytes = Nb * Ry * Rx * 8
        array_bytes = Ry * Rx * Ky * Kx * 4

        dev = torch.device(self.device)
        if dev.type == 'cuda' and not self.dataset.is_lazy:
            peak_vram = array_bytes + 2 * fft_bytes
            free_vram, _ = torch.cuda.mem_get_info(dev)
            if peak_vram <= int(free_vram * 0.6):
                return 'device'

        try:
            import psutil
            available_ram = psutil.virtual_memory().available
            if fft_bytes <= int(available_ram * 0.8):
                return 'host'
        except ImportError:
            pass

        return 'on_the_fly'

    def _init_device_cache(self) -> None:
        arr = self.dataset.get_full_array()
        dev = torch.device(self.device)
        arr_gpu = torch.as_tensor(arr, device=dev)
        bf_mask_d = self.detector_geom.bf_mask.bool()
        vbf = arr_gpu[:, :, bf_mask_d].permute(2, 0, 1).contiguous()
        del arr_gpu
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        self._device_cache = torch.fft.fft2(vbf, dim=(-2, -1))
        del vbf

    def get_chunk(self, b_start: int, b_end: int) -> torch.Tensor:
        """Return (b_end-b_start, Ry, Rx) complex64 on device."""
        dev = torch.device(self.device)

        if self.cache_mode == 'device':
            if self._device_cache is None:
                self._init_device_cache()
            return self._device_cache[b_start:b_end]

        if self.cache_mode == 'on_the_fly':
            return self._compute_gpu_fft_chunk(b_start, b_end, dev)

        return self._get_chunk_host(b_start, b_end, dev)

    def _compute_gpu_fft_chunk(self, b_start: int, b_end: int, dev: torch.device) -> torch.Tensor:
        """CPU gather → H2D → GPU FFT. Returns (chunk_size, Ry, Rx) complex64 on device."""
        iy = self._bf_iy[b_start:b_end]
        ix = self._bf_ix[b_start:b_end]
        vbf_cpu = self.dataset.get_bf_chunk(iy, ix)
        vbf_gpu = torch.from_numpy(vbf_cpu).to(dev)
        return torch.fft.fft2(vbf_gpu, dim=(-2, -1))

    def _get_chunk_host(self, b_start: int, b_end: int, dev: torch.device) -> torch.Tensor:
        if self._host_cache is None:
            Ry, Rx = self.scan_shape
            self._host_cache = np.empty((self.nb, Ry, Rx), dtype=np.complex64)
            self._host_filled = np.zeros(self.nb, dtype=bool)
            if self.dataset.is_lazy:
                # For lazy backends, fill the entire cache in ONE sequential
                # scan-row pass instead of one call per chunk.  Chunk-by-chunk
                # filling with slab/per_pixel on a contiguous HDF5 file causes
                # h5py to issue one hyperslab read spanning the full file extent
                # per chunk — O(n_chunks) full-file reads instead of O(1).
                self._prefill_host_cache_sequential(dev)

        unfilled = ~self._host_filled[b_start:b_end]
        if unfilled.any():
            # Handles non-lazy datasets and any entries not covered by prefill.
            unfilled_local = np.where(unfilled)[0]
            iy_batch = self._bf_iy[b_start + unfilled_local]
            ix_batch = self._bf_ix[b_start + unfilled_local]
            vbf_cpu = self.dataset.get_bf_chunk(iy_batch, ix_batch)
            fft_gpu = torch.fft.fft2(torch.from_numpy(vbf_cpu).to(dev), dim=(-2, -1))
            fft_cpu = fft_gpu.cpu().numpy()
            for k, local_b in enumerate(unfilled_local):
                self._host_cache[b_start + local_b] = fft_cpu[k]
                self._host_filled[b_start + local_b] = True

        chunk_np = self._host_cache[b_start:b_end].copy()
        return torch.from_numpy(chunk_np).to(dev)

    def _prefill_host_cache_sequential(self, dev: torch.device) -> None:
        """Fill the entire host FFT cache in one pass, choosing the I/O strategy
        based on the dataset's chunk layout.

        Two strategies:

        scan_row / slab (contiguous HDF5):
            Reads the dataset one scan row at a time — Ry sequential h5py
            calls each reading Rx*Ky*Kx*4 bytes.  Scans the full file but
            avoids the O(Ry*Rx) individual pread() calls that h5py's hyperslab
            issues for per-pixel access on contiguous storage.

        per_pixel (detector-major chunked HDF5, chunks=(Ry,Rx,1,1)):
            Reads handle[:,:,ky,kx] for each BF pixel — Nb h5py calls each
            reading one 3–4 MB contiguous chunk.  Only the BF subset of
            the file is touched: I/O ∝ Nb*Ry*Rx*4 ≈ 2–3 GiB instead of
            the full 48–64 GiB file.

        Peak extra RAM: (Nb, Ry, Rx) float32 (2–5 GiB for typical scans).
        Freed before returning.
        """
        bf_iy, bf_ix = self._bf_iy, self._bf_ix
        if self.dataset.lazy_read_mode == 'per_pixel':
            raw_buf = self.dataset.stream_all_bf_images_per_pixel(bf_iy, bf_ix)
        else:
            raw_buf = self.dataset.stream_all_bf_images(bf_iy, bf_ix)

        chunk_size = 64
        for b in range(0, self.nb, chunk_size):
            b_end = min(b + chunk_size, self.nb)
            fft = torch.fft.fft2(
                torch.from_numpy(raw_buf[b:b_end]).to(dev), dim=(-2, -1)
            )
            self._host_cache[b:b_end] = fft.cpu().numpy()
            self._host_filled[b:b_end] = True
        del raw_buf

    def clear_cache(self) -> None:
        self._device_cache = None
        self._host_cache = None
        self._host_filled = None
        if torch.device(self.device).type == 'cuda':
            torch.cuda.empty_cache()
