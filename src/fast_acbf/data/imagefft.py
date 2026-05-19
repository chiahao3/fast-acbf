"""ImageFFT cache and chunk provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from fast_acbf.data.bf_extractor import BFExtractor


def _fft_zero_pad_2d(F_in: torch.Tensor, Ry_out: int, Rx_out: int) -> torch.Tensor:
    """
    Zero-pad a 2-D FFT (..., Ry_in, Rx_in) to (..., Ry_out, Rx_out).
    Preserves torch.fft.fft2 frequency layout (DC at corner).
    Scale factor (Ry_out*Rx_out)/(Ry_in*Rx_in) ensures correct IFFT amplitude.
    """
    Ry_in, Rx_in = F_in.shape[-2], F_in.shape[-1]
    F_c = torch.fft.fftshift(F_in, dim=(-2, -1))
    # Left pad = Nout//2 - Nin//2 so DC (at Nin//2 after fftshift) lands at Nout//2.
    # This is correct for both even and odd sizes; pad_total//2 only works for even Nin.
    pad_y_left = Ry_out // 2 - Ry_in // 2
    pad_x_left = Rx_out // 2 - Rx_in // 2
    pad_y_right = Ry_out - Ry_in - pad_y_left
    pad_x_right = Rx_out - Rx_in - pad_x_left
    # F.pad pads last dims first: (x_left, x_right, y_left, y_right)
    F_c_padded = F.pad(F_c, (pad_x_left, pad_x_right, pad_y_left, pad_y_right))
    out = torch.fft.ifftshift(F_c_padded, dim=(-2, -1))
    return out * ((Ry_out * Rx_out) / (Ry_in * Rx_in))


_VALID_STORAGE = ('auto', 'device', 'host', 'none')
_VALID_FILL = ('auto', 'precompute', 'lazy', 'on_the_fly')


class ImageFFT:
    """Serves virtual-BF FFT chunks to reconstruction code.

    ``ImageFFT`` is the only persistence object in the data pipeline.  It may
    store computed FFTs on device, store them in host RAM, or store nothing and
    recompute each requested chunk.
    """

    def __init__(
        self,
        extractor: BFExtractor,
        *,
        device: str,
        storage: str = 'none',
        fill: str = 'on_the_fly',
        batch_size: int = 64,
    ) -> None:
        self.extractor = extractor
        self.device = device
        self.storage = self._normalize_storage(storage)
        self.fill = self._normalize_fill(fill)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        if self.storage == 'none' and self.fill != 'on_the_fly':
            raise ValueError("imagefft_storage='none' requires imagefft_fill='on_the_fly'.")
        if self.fill == 'on_the_fly' and self.storage != 'none':
            raise ValueError("imagefft_fill='on_the_fly' requires imagefft_storage='none'.")
        if extractor.strategy == 'device_mask' and self.fill != 'precompute':
            raise ValueError(
                "extractor_strategy='device_mask' requires imagefft_fill='precompute'. "
                "Use a host or disk extraction strategy for lazy/on-the-fly ImageFFT."
            )

        self.nb = extractor.nb
        self.scan_shape = extractor.scan_shape
        self._cache: torch.Tensor | np.ndarray | None = None
        self._filled: np.ndarray | None = None

        if self.fill == 'precompute':
            self.precompute()

    @staticmethod
    def _normalize_storage(storage: str) -> str:
        storage = str(storage).strip().lower()
        if storage not in _VALID_STORAGE:
            raise ValueError(f"imagefft_storage must be one of {_VALID_STORAGE}, got {storage!r}.")
        if storage == 'auto':
            return 'none'  # safe fallback; use PipelineManager for hardware-aware resolution
        return storage

    @staticmethod
    def _normalize_fill(fill: str) -> str:
        fill = str(fill).strip().lower()
        if fill not in _VALID_FILL:
            raise ValueError(f"imagefft_fill must be one of {_VALID_FILL}, got {fill!r}.")
        if fill == 'auto':
            return 'on_the_fly'
        return fill

    @property
    def cache(self):
        return self._cache

    @property
    def filled(self) -> np.ndarray | None:
        return self._filled

    def get_chunk(self, b_start: int, b_end: int) -> torch.Tensor:
        """Return ``(b_end-b_start, Ry, Rx)`` complex64 on the compute device."""
        self._validate_range(b_start, b_end)
        if self.storage == 'none':
            return self._compute_fft_chunk(b_start, b_end)

        self._ensure_cache()
        if self.fill == 'lazy':
            self._fill_missing(b_start, b_end)

        if self.storage == 'device':
            return self._cache[b_start:b_end]

        chunk = np.asarray(self._cache[b_start:b_end]).copy()
        return torch.from_numpy(chunk).to(torch.device(self.device))

    def get_upscaled_chunk(self, b_start: int, b_end: int, upscale: float) -> torch.Tensor:
        """
        Return the FFT chunk for [b_start, b_end) zero-padded to the upscaled size.
        Shape: (b_end-b_start, Ry_out, Rx_out) complex64, on device.
        upscale=1.0 returns get_chunk() directly (no allocation).
        """
        chunk = self.get_chunk(b_start, b_end)
        if upscale == 1.0:
            return chunk
        Ry_in, Rx_in = self.scan_shape
        Ry_out = round(Ry_in * upscale)
        Rx_out = round(Rx_in * upscale)
        return _fft_zero_pad_2d(chunk, Ry_out, Rx_out)

    def precompute(self) -> None:
        if self.storage == 'none':
            return
        if self.extractor.strategy == 'device_mask':
            self._precompute_device_mask_all()
            return
        self._ensure_cache()
        for b_start in range(0, self.nb, self.batch_size):
            b_end = min(b_start + self.batch_size, self.nb)
            self._store_fft_chunk(b_start, b_end)

    def clear(self) -> None:
        self._cache = None
        self._filled = None
        if torch.device(self.device).type == 'cuda':
            torch.cuda.empty_cache()

    def _validate_range(self, b_start: int, b_end: int) -> None:
        if not (0 <= b_start <= b_end <= self.nb):
            raise ValueError(f"Invalid ImageFFT chunk range [{b_start}, {b_end}) for nb={self.nb}.")

    def _ensure_cache(self) -> None:
        if self._cache is not None:
            return
        Ry, Rx = self.scan_shape
        if self.storage == 'device':
            self._cache = torch.empty(
                (self.nb, Ry, Rx), dtype=torch.complex64, device=torch.device(self.device)
            )
        elif self.storage == 'host':
            self._cache = np.empty((self.nb, Ry, Rx), dtype=np.complex64)
        else:
            raise AssertionError(f"Unhandled storage {self.storage!r}.")
        self._filled = np.zeros(self.nb, dtype=bool)

    def _fill_missing(self, b_start: int, b_end: int) -> None:
        missing = ~self._filled[b_start:b_end]
        if not missing.any():
            return
        local = np.where(missing)[0]
        run_start = None
        prev = None
        for idx in local.tolist():
            absolute = b_start + idx
            if run_start is None:
                run_start = absolute
            elif prev is not None and absolute != prev + 1:
                self._store_fft_chunk(run_start, prev + 1)
                run_start = absolute
            prev = absolute
        if run_start is not None:
            self._store_fft_chunk(run_start, prev + 1)

    def _store_fft_chunk(self, b_start: int, b_end: int) -> None:
        fft = self._compute_fft_chunk(b_start, b_end)
        if self.storage == 'device':
            self._cache[b_start:b_end] = fft
        elif self.storage == 'host':
            self._cache[b_start:b_end] = fft.cpu().numpy()
        else:
            raise AssertionError(f"Cannot store FFT chunk for storage={self.storage!r}.")
        self._filled[b_start:b_end] = True

    def _precompute_device_mask_all(self) -> None:
        """Whole-pass precompute for the raw-on-device extraction path."""
        vbf = self.extractor.extract_all()
        dev = torch.device(self.device)
        if not isinstance(vbf, torch.Tensor):
            raise TypeError("device_mask extraction must return a torch.Tensor.")
        vbf_dev = vbf.to(dev)
        fft = torch.fft.fft2(vbf_dev, dim=(-2, -1))
        del vbf, vbf_dev

        if self.storage == 'device':
            self._cache = fft
        elif self.storage == 'host':
            self._cache = fft.cpu().numpy()
            del fft
        else:
            raise AssertionError(f"Cannot precompute FFT for storage={self.storage!r}.")
        self._filled = np.ones(self.nb, dtype=bool)
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

    def _compute_fft_chunk(self, b_start: int, b_end: int) -> torch.Tensor:
        vbf = self.extractor.extract_chunk(b_start, b_end)
        dev = torch.device(self.device)
        if isinstance(vbf, torch.Tensor):
            vbf_dev = vbf.to(dev)
        else:
            vbf_dev = torch.from_numpy(vbf).to(dev)
        return torch.fft.fft2(vbf_dev, dim=(-2, -1))


__all__ = ["ImageFFT", "_fft_zero_pad_2d"]
