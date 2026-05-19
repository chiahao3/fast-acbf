"""BFReconstructor — minimal BF-specific optimizable model.

Owns: ImageFFT, DetectorGeometry, ScanGeometry, AberrationState, CoordinateTransform,
and the reconstruction cache workspace. Nothing else.

Strict boundaries:
  - no raw dataset/source access
  - no plotting, no refinement orchestration
  - no get_probe / get_chi_surface / get_yx_shifts_*
  - no metadata application
  - no post-construction mutation of DetectorGeometry or ScanGeometry
"""

from __future__ import annotations

import torch

from fast_acbf.core.aberrations import AberrationState
from fast_acbf.core.acbf import reconstruct_acbf, reconstruct_acbf_complex_inversion
from fast_acbf.core.tcbf import reconstruct_tcbf
from fast_acbf.data.geometry import CoordinateTransform, ScanGeometry
from fast_acbf.data.geometry import DetectorGeometry
from fast_acbf.data.imagefft import ImageFFT
from fast_acbf.recon.cache import (
    ACBFGeometryCache,
    ACBFOpticsCache,
    TCBFCache,
    build_acbf_geometry_cache,
    build_acbf_optics_cache,
    build_tcbf_cache,
)

_VALID_BASIS_MODES = ('on_the_fly', 'precompute')
_VALID_FOV = ('original', 'full')


def _crop_to_original(
    image: torch.Tensor,
    pad_offsets: tuple[int, int],
    orig_shape: tuple[int, int],
    padded_shape: tuple[int, int],
) -> torch.Tensor:
    """Crop a (possibly upscaled) padded image back to the original FOV.

    Derives the effective scale factor from the actual image dimensions rather
    than from a nominal upscale value, so fractional upscales with 5-smooth
    rounding produce the physically correct crop for any image size.
    """
    pad_y, pad_x = pad_offsets
    Ry, Rx = orig_shape
    Ry_padded, Rx_padded = padded_shape
    Ry_out, Rx_out = image.shape[-2], image.shape[-1]
    y0 = round(pad_y * Ry_out / Ry_padded)
    y1 = y0 + round(Ry * Ry_out / Ry_padded)
    x0 = round(pad_x * Rx_out / Rx_padded)
    x1 = x0 + round(Rx * Rx_out / Rx_padded)
    return image[y0:y1, x0:x1]


class BFReconstructor:
    """BF-specific optimizable model.

    Do not add user-facing conveniences here. Those live on BFSolver.
    """

    def __init__(
        self,
        imagefft: ImageFFT,
        detector_geom: DetectorGeometry,
        scan_geom: ScanGeometry,
        ab_state: AberrationState,
        coord_transform: CoordinateTransform,
        basis_mode: str = 'on_the_fly',
        eps: float = 1e-3,
        orig_scan_shape: tuple[int, int] | None = None,
        pad_offsets: tuple[int, int] | None = None,
        fov: str = 'original',
    ) -> None:
        self.imagefft = imagefft
        self.detector_geom = detector_geom
        self.scan_geom = scan_geom
        self.ab_state = ab_state
        self.coord_transform = coord_transform
        self.eps = eps
        self._orig_scan_shape = orig_scan_shape
        self._pad_offsets = pad_offsets
        fov_str = str(fov)
        if fov_str not in _VALID_FOV:
            raise ValueError(f"fov must be one of {_VALID_FOV}, got {fov_str!r}.")
        self._fov = fov_str
        self._tcbf_cache: dict[tuple, TCBFCache] = {}
        self._acbf_cache: dict[tuple, tuple[ACBFGeometryCache, ACBFOpticsCache | None]] = {}
        self.basis_mode = basis_mode  # validated via property setter (accesses _acbf_cache)

    @property
    def basis_mode(self) -> str:
        return self._basis_mode

    @basis_mode.setter
    def basis_mode(self, value: str) -> None:
        value = str(value).strip().lower()
        if value not in _VALID_BASIS_MODES:
            raise ValueError(f"basis_mode must be one of {_VALID_BASIS_MODES}, got {value!r}.")
        self._basis_mode = value
        # Invalidate precomputed optics when switching modes so they rebuild correctly.
        for key in self._acbf_cache:
            geometry, _ = self._acbf_cache[key]
            self._acbf_cache[key] = (geometry, None)

    @property
    def device(self) -> str:
        return self.imagefft.device

    @property
    def rotation_deg(self) -> float:
        return self.coord_transform.rotation_deg

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Full reset — clears both basis caches and the ImageFFT cache."""
        self._tcbf_cache = {}
        self._acbf_cache = {}
        self.imagefft.clear()

    def clear_basis_cache(self) -> None:
        """Clear orientation-dependent basis caches; ImageFFT cache is preserved."""
        self._tcbf_cache = {}
        self._acbf_cache = {}

    def set_coord_transform(self, ct: CoordinateTransform, clear_basis: bool = True) -> None:
        """Replace the coordinate transform. Clears basis cache by default."""
        self.coord_transform = ct
        if clear_basis:
            self.clear_basis_cache()

    # ------------------------------------------------------------------
    # Internal frame helpers — delegate to CoordinateTransform
    # ------------------------------------------------------------------

    def _frame_cache_key(self) -> tuple:
        ct = self.coord_transform
        return (ct.rotation_deg, ct.flipud, ct.fliplr, ct.transpose)

    def _get_scan_frame_coeffs(self) -> torch.Tensor:
        return self.ab_state.to_scan_frame(self.coord_transform.rotation_deg)

    def _get_transformed_centers(self, in_scan_frame: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        det = self.detector_geom
        return self.coord_transform.apply_to_centers(det.kY_centers, det.kX_centers, in_scan_frame)

    def _get_transformed_grids(self, in_scan_frame: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        det = self.detector_geom
        return self.coord_transform.apply_to_grids(det.kY_grid, det.kX_grid, in_scan_frame)

    # ------------------------------------------------------------------
    # Lazy-build cache accessors
    # ------------------------------------------------------------------

    def _get_tcbf_cache(self, chunk_size: int = 64) -> TCBFCache:
        key = (chunk_size, *self._frame_cache_key())
        if key not in self._tcbf_cache:
            kX_full, kY_full = self._get_transformed_centers()
            self._tcbf_cache[key] = build_tcbf_cache(
                kX_full, kY_full,
                self.ab_state.order_keys, self.detector_geom.wavelength, chunk_size,
            )
        return self._tcbf_cache[key]

    def _get_acbf_cache(
        self, rolloff: float = 0, chunk_size: int = 64,
    ) -> tuple[ACBFGeometryCache, ACBFOpticsCache | None]:
        """Return (geometry, optics_or_None).

        Geometry is mode-independent. Optics is built on demand only in
        'precompute' mode; the consumer-facing return is gated on basis_mode.
        """
        key = (rolloff, chunk_size, *self._frame_cache_key())
        if key not in self._acbf_cache:
            kX_full, kY_full = self._get_transformed_centers()
            det = self.detector_geom
            geometry = build_acbf_geometry_cache(
                kX_full, kY_full,
                self.ab_state.order_keys, det.max_alpha, det.wavelength,
                self.ab_state.max_order, rolloff, chunk_size,
            )
            self._acbf_cache[key] = (geometry, None)

        geometry, optics = self._acbf_cache[key]
        if self.basis_mode == 'precompute':
            if optics is None:
                sg = self.scan_geom
                optics = build_acbf_optics_cache(geometry, sg.qx_grid, sg.qy_grid)
                self._acbf_cache[key] = (geometry, optics)
            return geometry, optics
        return geometry, None

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_acbf_algorithm(acbf_algorithm) -> str:
        if acbf_algorithm is None:
            return 'phase_only'
        return str(acbf_algorithm).strip().lower().replace('-', '_')

    def _get_recon_grids(self, upscale: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (qx_grid, qy_grid) for the given upscale factor."""
        sg = self.scan_geom
        if upscale == 1.0:
            return sg.qx_grid, sg.qy_grid
        Ry_p, Rx_p = sg.scan_shape
        step = sg.scan_step_size
        Ry_out = round(Ry_p * upscale)
        Rx_out = round(Rx_p * upscale)
        dev = torch.device(self.device)
        # d = step * Rx_p / Rx_out ensures same low-frequency bin spacing as padded grid
        qx = torch.fft.fftfreq(Rx_out, d=step * Rx_p / Rx_out, device=dev).view(1, 1, Rx_out)
        qy = torch.fft.fftfreq(Ry_out, d=step * Ry_p / Ry_out, device=dev).view(1, Ry_out, 1)
        return qx, qy

    def _reconstruct_impl(self, mode: str = 'tcBF', **kwargs) -> torch.Tensor:
        upscale = float(kwargs.get('upscale', 1.0))
        fov = str(kwargs.get('fov', self._fov))
        if fov not in _VALID_FOV:
            raise ValueError(f"fov must be one of {_VALID_FOV}, got {fov!r}.")
        if upscale < 1.0:
            raise ValueError(f"upscale must be >= 1.0, got {upscale}.")

        qx_grid, qy_grid = self._get_recon_grids(upscale)
        coeffs = self._get_scan_frame_coeffs()
        mode_key = mode.lower()

        if mode_key == 'tcbf':
            cache = self._get_tcbf_cache(chunk_size=kwargs.get('chunk_size', 64))
            result = reconstruct_tcbf(
                self.imagefft, qx_grid, qy_grid, cache, coeffs, self.device,
                upscale=upscale,
            )

        elif mode_key == 'acbf':
            rolloff = kwargs.get('rolloff', 0)
            chunk_size = kwargs.get('chunk_size', 64)
            acbf_algorithm = self._normalize_acbf_algorithm(kwargs.get('acbf_algorithm'))
            geometry, optics = self._get_acbf_cache(rolloff=rolloff, chunk_size=chunk_size)
            if upscale != 1.0:
                optics = None  # precomputed optics are wrong size; use lazy path

            if acbf_algorithm == 'phase_only':
                result = reconstruct_acbf(
                    self.imagefft, qx_grid, qy_grid, geometry, optics,
                    coeffs, self.eps, self.device, upscale=upscale,
                )
            elif acbf_algorithm == 'complex_inversion':
                result = reconstruct_acbf_complex_inversion(
                    self.imagefft, qx_grid, qy_grid, geometry, optics,
                    coeffs, self.device,
                    regularization=kwargs.get('regularization', 1e-3),
                    support_threshold=kwargs.get('support_threshold', 1e-6),
                    upscale=upscale,
                )
            else:
                raise ValueError(
                    f"Unsupported acBF algorithm {acbf_algorithm!r}. "
                    "Choose 'phase_only' or 'complex_inversion'."
                )

        else:
            raise ValueError(f"Unsupported mode {mode!r}. Choose 'tcBF' or 'acBF'.")

        # fov crop — only when padding was used
        if self._pad_offsets is not None and fov == 'original':
            result = _crop_to_original(result, self._pad_offsets, self._orig_scan_shape, self.scan_geom.scan_shape)

        return result

    def reconstruct(self, mode: str = 'tcBF', requires_grad: bool = False, **kwargs) -> torch.Tensor:
        """Run reconstruction. No-grad by default; opt-in for AD optimization paths."""
        if requires_grad:
            return self._reconstruct_impl(mode=mode, **kwargs)
        with torch.no_grad():
            return self._reconstruct_impl(mode=mode, **kwargs)
