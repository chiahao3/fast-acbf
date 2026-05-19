"""Pipeline policy resolution for Dataset4D -> BFExtractor -> ImageFFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from fast_acbf.data.bf_extractor import BFExtractor
from fast_acbf.data.imagefft import ImageFFT

if TYPE_CHECKING:
    from fast_acbf.data.dataset4d import Dataset4D
    from fast_acbf.data.geometry import DetectorGeometry


_VALID_PIPELINES = ('speed', 'balanced', 'memory')
_VALID_STORAGE = ('auto', 'device', 'host', 'none')
_VALID_FILL = ('auto', 'precompute', 'lazy', 'on_the_fly')
_VALID_EXTRACTOR = (
    'auto',
    'device_mask',
    'host_mask',
    'disk_per_pixel',
    'disk_slab',
    'disk_scan_row',
)


@dataclass(frozen=True)
class PipelineResolution:
    pipeline: str
    imagefft_storage: str
    imagefft_fill: str
    extractor_strategy: str
    raw_bytes: int
    vbf_bytes: int
    imagefft_bytes: int


class PipelineManager:
    """Resolve pipeline policy and build the ImageFFT object consumed by recon."""

    def __init__(
        self,
        dataset: Dataset4D,
        detector_geom: DetectorGeometry,
        *,
        device: str,
        pipeline: str = 'balanced',
        imagefft_storage: str = 'auto',
        imagefft_fill: str = 'auto',
        extractor_strategy: str = 'auto',
        fft_batch_size: int = 64,
        vram_margin: float = 0.60,
        ram_margin: float = 0.80,
    ) -> None:
        self.dataset = dataset
        self.detector_geom = detector_geom
        self.device = device
        self.pipeline = self._validate('pipeline', pipeline, _VALID_PIPELINES)
        self.imagefft_storage_request = self._validate(
            'imagefft_storage', imagefft_storage, _VALID_STORAGE
        )
        self.imagefft_fill_request = self._validate('imagefft_fill', imagefft_fill, _VALID_FILL)
        self.extractor_strategy_request = self._validate(
            'extractor_strategy', extractor_strategy, _VALID_EXTRACTOR
        )
        self.fft_batch_size = int(fft_batch_size)
        self.vram_margin = float(vram_margin)
        self.ram_margin = float(ram_margin)

        self.resolution = self._resolve()
        self.extractor: BFExtractor | None = None
        self.imagefft: ImageFFT | None = None

    @staticmethod
    def _validate(name: str, value: str, valid: tuple[str, ...]) -> str:
        value = str(value).strip().lower()
        if value not in valid:
            raise ValueError(f"{name} must be one of {valid}, got {value!r}.")
        return value

    @property
    def nb(self) -> int:
        return int(self.detector_geom.n_bf_pixels)

    @property
    def vbf_bytes(self) -> int:
        Ry, Rx = self.dataset.scan_shape
        return int(self.nb * Ry * Rx * 4)

    @property
    def imagefft_bytes(self) -> int:
        return int(self.vbf_bytes * 2)

    def build_imagefft(self) -> ImageFFT:
        self.extractor = BFExtractor(
            self.dataset,
            self.detector_geom,
            device=self.device,
            strategy=self.resolution.extractor_strategy,
        )
        self.imagefft = ImageFFT(
            self.extractor,
            device=self.device,
            storage=self.resolution.imagefft_storage,
            fill=self.resolution.imagefft_fill,
            batch_size=self.fft_batch_size,
        )
        return self.imagefft

    def _resolve(self) -> PipelineResolution:
        raw_bytes = self.dataset.nbytes_float32
        vbf_bytes = self.vbf_bytes
        imagefft_bytes = self.imagefft_bytes

        storage = self._resolve_storage(raw_bytes, vbf_bytes, imagefft_bytes)
        fill = self._resolve_fill(storage)
        extractor_strategy = self._resolve_extractor_strategy(
            storage, fill, raw_bytes, vbf_bytes, imagefft_bytes
        )

        return PipelineResolution(
            pipeline=self.pipeline,
            imagefft_storage=storage,
            imagefft_fill=fill,
            extractor_strategy=extractor_strategy,
            raw_bytes=raw_bytes,
            vbf_bytes=vbf_bytes,
            imagefft_bytes=imagefft_bytes,
        )

    def _resolve_storage(self, raw_bytes: int, vbf_bytes: int, imagefft_bytes: int) -> str:
        if self.imagefft_storage_request != 'auto':
            return self.imagefft_storage_request

        if self.pipeline == 'memory':
            return 'none'

        if self._device_cache_fits(imagefft_bytes):
            return 'device'
        if self._host_cache_fits(imagefft_bytes):
            return 'host'
        return 'none'

    def _resolve_fill(self, storage: str) -> str:
        if storage == 'none':
            if self.imagefft_fill_request not in ('auto', 'on_the_fly'):
                raise ValueError("imagefft_storage='none' requires imagefft_fill='on_the_fly'.")
            return 'on_the_fly'

        if self.imagefft_fill_request != 'auto':
            if self.imagefft_fill_request == 'on_the_fly':
                raise ValueError("imagefft_fill='on_the_fly' requires imagefft_storage='none'.")
            return self.imagefft_fill_request

        return 'precompute'

    def _resolve_extractor_strategy(
        self,
        storage: str,
        fill: str,
        raw_bytes: int,
        vbf_bytes: int,
        imagefft_bytes: int,
    ) -> str:
        if self.extractor_strategy_request != 'auto':
            return self.extractor_strategy_request

        if self._can_use_device_mask(raw_bytes, vbf_bytes, imagefft_bytes):
            return 'device_mask'

        if not self.dataset.is_lazy:
            return 'host_mask'

        chunks = self.dataset.backend_chunks
        if chunks is None:
            return 'disk_scan_row' if fill == 'precompute' else 'disk_slab'
        c0, c1, c2, c3 = chunks
        if c2 == 1 and c3 == 1:
            return 'disk_per_pixel'
        if c0 == 1 and c1 == 1:
            return 'disk_scan_row'
        return 'disk_per_pixel'

    def _can_use_device_mask(self, raw_bytes: int, vbf_bytes: int, imagefft_bytes: int) -> bool:
        if self.pipeline == 'memory':
            return False
        dev = torch.device(self.device)
        if dev.type != 'cuda':
            return False
        if self.dataset.is_lazy and self.pipeline != 'speed':
            return False
        free_vram = self._free_vram()
        if free_vram is None:
            return False
        needed = raw_bytes + vbf_bytes + imagefft_bytes
        return needed <= int(free_vram * self.vram_margin)

    def _device_cache_fits(self, imagefft_bytes: int) -> bool:
        dev = torch.device(self.device)
        if dev.type != 'cuda':
            return False
        free_vram = self._free_vram()
        return free_vram is not None and imagefft_bytes <= int(free_vram * self.vram_margin)

    def _host_cache_fits(self, imagefft_bytes: int) -> bool:
        available = self._available_ram()
        return available is not None and imagefft_bytes <= int(available * self.ram_margin)

    def _free_vram(self) -> int | None:
        dev = torch.device(self.device)
        if dev.type != 'cuda':
            return None
        free, _ = torch.cuda.mem_get_info(dev)
        return int(free)

    @staticmethod
    def _available_ram() -> int | None:
        try:
            import psutil
        except ImportError:
            return None
        return int(psutil.virtual_memory().available)


__all__ = ["PipelineManager", "PipelineResolution"]
