"""PreparedBFDataset — converts a DatasetSource into reconstruction-ready BF data."""

from __future__ import annotations

import numpy as np
import torch

from fast_acbf.data.geometry import DetectorGeometry
from fast_acbf.data.source import ArrayDatasetSource
from fast_acbf.recon.cache import ImageFFT, build_image_fft


class PreparedBFDataset:
    """Reconstruction-ready BF image stack on the target device.

    Interface contract: BFReconstructor must only access `vbf_images` and
    `build_image_fft()` — never the raw source array directly. This keeps
    future implementations (chunked HDF5, lazy zarr) swap-compatible without
    touching BFReconstructor's API.
    """

    def __init__(
        self,
        source: ArrayDatasetSource,
        detector_geom: DetectorGeometry,
        device: str,
        vbf_images: torch.Tensor,
    ) -> None:
        self.source = source
        self.detector_geom = detector_geom
        self.device = device
        self.vbf_images = vbf_images  # (Nb, Ry, Rx) float32 on device

    @classmethod
    def build(
        cls,
        source: ArrayDatasetSource,
        detector_geom: DetectorGeometry,
        device: str,
    ) -> PreparedBFDataset:
        """Eagerly extract the vBF image stack from the source on the target device."""
        array = source.get_array()
        bf_mask_bool = detector_geom.bf_mask_bool
        device_obj = torch.device(device)
        device_type = device_obj.type

        if device_type == 'cpu':
            masked = array[:, :, bf_mask_bool]                          # (Ry, Rx, Nb)
            masked = np.ascontiguousarray(np.moveaxis(masked, -1, 0))   # (Nb, Ry, Rx)
            vbf_images = torch.from_numpy(masked.astype(np.float32, copy=False))
        else:
            src = torch.as_tensor(
                np.ascontiguousarray(array, dtype=np.float32), device=device_obj,
            )
            bf_mask_d = detector_geom.bf_mask.bool()
            # src: (Ry, Rx, Ky, Kx) → gather BF pixels → (Nb, Ry, Rx)
            vbf_images = src[:, :, bf_mask_d].permute(2, 0, 1).contiguous()

        Nb = vbf_images.shape[0]
        print(
            f"Extracted {Nb} vBF images within max_alpha = "
            f"{detector_geom.max_alpha} mrad."
        )
        return cls(source=source, detector_geom=detector_geom, device=device, vbf_images=vbf_images)

    def build_image_fft(self) -> ImageFFT:
        """Compute the 2D FFT of the full vBF stack. Caller owns the result."""
        return build_image_fft(self.vbf_images)
