"""BF aperture extraction from Dataset4D raw sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from fast_acbf.data.dataset4d import Dataset4D
    from fast_acbf.data.geometry import DetectorGeometry


_VALID_EXTRACTOR_STRATEGIES = (
    'auto',
    'device_mask',
    'host_mask',
    'disk_per_pixel',
    'disk_slab',
    'disk_scan_row',
)


class BFExtractor:
    """Extracts virtual BF images from a raw ``Dataset4D``.

    Returned arrays have shape ``(B, Ry, Rx)`` and dtype float32.  Most
    strategies return numpy arrays on host.  ``device_mask`` returns torch
    tensors on the compute device because the extraction itself happens there.
    """

    def __init__(
        self,
        dataset: Dataset4D,
        detector_geom: DetectorGeometry,
        *,
        device: str = 'cpu',
        strategy: str = 'auto',
    ) -> None:
        self.dataset = dataset
        self.detector_geom = detector_geom
        self.device = device

        strategy = str(strategy).strip().lower()
        if strategy not in _VALID_EXTRACTOR_STRATEGIES:
            raise ValueError(
                f"extractor_strategy must be one of {_VALID_EXTRACTOR_STRATEGIES}, "
                f"got {strategy!r}."
            )
        self.strategy = self.resolve_strategy(strategy)

        bf_iy, bf_ix = np.where(detector_geom.bf_mask_bool)
        if len(bf_iy) == 0:
            raise ValueError("BF aperture contains zero pixels. Check max_alpha and dk.")
        self.bf_iy = bf_iy.astype(np.int64, copy=False)
        self.bf_ix = bf_ix.astype(np.int64, copy=False)
        self.nb = int(len(bf_iy))
        self.scan_shape = dataset.scan_shape

    def resolve_strategy(self, strategy: str) -> str:
        if strategy != 'auto':
            return strategy
        if not self.dataset.is_lazy:
            return 'host_mask'
        chunks = self.dataset.backend_chunks
        if chunks is None:
            return 'disk_slab'
        c0, c1, c2, c3 = chunks
        if c2 == 1 and c3 == 1:
            return 'disk_per_pixel'
        if c0 == 1 and c1 == 1:
            return 'disk_scan_row'
        return 'disk_per_pixel'

    def extract_chunk(self, b_start: int, b_end: int):
        """Return BF images for detector-index range ``[b_start, b_end)``."""
        self._validate_range(b_start, b_end)
        iy = self.bf_iy[b_start:b_end]
        ix = self.bf_ix[b_start:b_end]

        if self.strategy == 'device_mask':
            return self._extract_device_mask(iy, ix)
        if self.strategy == 'host_mask':
            return self._extract_host_mask(iy, ix)
        if self.strategy == 'disk_per_pixel':
            return self._extract_disk_per_pixel(iy, ix)
        if self.strategy == 'disk_slab':
            return self._extract_disk_slab(iy, ix)
        if self.strategy == 'disk_scan_row':
            return self._extract_disk_scan_row(iy, ix)
        raise AssertionError(f"Unhandled extractor strategy {self.strategy!r}.")

    def extract_all(self):
        """Return all BF images selected by the detector geometry."""
        return self.extract_chunk(0, self.nb)

    def _validate_range(self, b_start: int, b_end: int) -> None:
        if not (0 <= b_start <= b_end <= self.nb):
            raise ValueError(
                f"Invalid BF chunk range [{b_start}, {b_end}) for nb={self.nb}."
            )

    def _extract_host_mask(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        arr = self.dataset.materialize() if self.dataset.is_lazy else self.dataset.raw_array()
        raw = arr[:, :, iy, ix]
        return np.ascontiguousarray(raw.transpose(2, 0, 1), dtype=np.float32)

    def _extract_device_mask(self, iy: np.ndarray, ix: np.ndarray) -> torch.Tensor:
        arr = self.dataset.materialize() if self.dataset.is_lazy else self.dataset.raw_array()
        dev = torch.device(self.device)
        arr_gpu = torch.as_tensor(arr, device=dev)
        iy_d = torch.as_tensor(iy, dtype=torch.long, device=dev)
        ix_d = torch.as_tensor(ix, dtype=torch.long, device=dev)
        vbf = arr_gpu[:, :, iy_d, ix_d].permute(2, 0, 1).contiguous()
        del arr_gpu
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
        return vbf

    def _extract_disk_per_pixel(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        raw = np.stack(
            [
                self.dataset.read_raw((slice(None), slice(None), int(ky), int(kx)))
                for ky, kx in zip(iy.tolist(), ix.tolist())
            ],
            axis=0,
        )
        return np.ascontiguousarray(raw, dtype=np.float32)

    def _extract_disk_slab(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        ky_min = int(iy.min())
        ky_max = int(iy.max()) + 1
        slab = self.dataset.read_raw(
            (slice(None), slice(None), slice(ky_min, ky_max), slice(None))
        )
        iy_local = iy - ky_min
        raw = slab[:, :, iy_local, ix]
        return np.ascontiguousarray(raw.transpose(2, 0, 1), dtype=np.float32)

    def _extract_disk_scan_row(self, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        Ry, Rx = self.scan_shape
        out = np.empty((len(iy), Ry, Rx), dtype=np.float32)
        for scan_y, row in self.dataset.iter_scan_rows():
            out[:, scan_y, :] = row[:, iy, ix].T
        return out


__all__ = ["BFExtractor"]
