"""BFReconstructor — minimal BF-specific optimizable model.

Owns: PreparedBFDataset, ScanGeometry, AberrationState, CoordinateTransform,
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
from fast_acbf.data.prepared import PreparedBFDataset
from fast_acbf.recon.cache import (
    ACBFGeometryCache,
    ACBFOpticsCache,
    ImageFFT,
    TCBFCache,
    build_acbf_geometry_cache,
    build_acbf_optics_cache,
    build_image_fft,
    build_tcbf_cache,
)

_VALID_CACHE_MODES = ('lazy', 'full')


class BFReconstructor:
    """BF-specific optimizable model.

    Do not add user-facing conveniences here. Those live on BFSolver.
    """

    def __init__(
        self,
        prepared: PreparedBFDataset,
        scan_geom: ScanGeometry,
        ab_state: AberrationState,
        coord_transform: CoordinateTransform,
        cache_mode: str = 'lazy',
        eps: float = 1e-3,
    ) -> None:
        self.prepared = prepared
        self.scan_geom = scan_geom
        self.ab_state = ab_state
        self.coord_transform = coord_transform
        self.eps = eps
        self.cache_mode = cache_mode  # validated via property setter

        # Three typed cache fields instead of one untyped dict.
        self._image_fft: ImageFFT | None = None
        self._tcbf_cache: dict[tuple, TCBFCache] = {}
        self._acbf_cache: dict[tuple, tuple[ACBFGeometryCache, ACBFOpticsCache | None]] = {}

    @property
    def cache_mode(self) -> str:
        return self._cache_mode

    @cache_mode.setter
    def cache_mode(self, value: str) -> None:
        value = str(value).strip().lower()
        if value not in _VALID_CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {_VALID_CACHE_MODES}, got {value!r}.")
        self._cache_mode = value

    @property
    def device(self) -> str:
        return self.prepared.device

    @property
    def rotation_deg(self) -> float:
        return self.coord_transform.rotation_deg

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Full reset — clears image FFT and both basis caches."""
        self._image_fft = None
        self._tcbf_cache = {}
        self._acbf_cache = {}

    def clear_basis_cache(self) -> None:
        """Clear orientation-dependent basis caches; image FFT is preserved."""
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
        det = self.prepared.detector_geom
        return self.coord_transform.apply_to_centers(det.kY_centers, det.kX_centers, in_scan_frame)

    def _get_transformed_grids(self, in_scan_frame: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        det = self.prepared.detector_geom
        return self.coord_transform.apply_to_grids(det.kY_grid, det.kX_grid, in_scan_frame)

    # ------------------------------------------------------------------
    # Lazy-build cache accessors
    # ------------------------------------------------------------------

    def _get_image_fft(self) -> ImageFFT:
        if self._image_fft is None:
            self._image_fft = self.prepared.build_image_fft()
        return self._image_fft

    def _get_tcbf_cache(self, chunk_size: int = 64) -> TCBFCache:
        key = (chunk_size, *self._frame_cache_key())
        if key not in self._tcbf_cache:
            kX_full, kY_full = self._get_transformed_centers()
            self._tcbf_cache[key] = build_tcbf_cache(
                kX_full, kY_full,
                self.ab_state.order_keys, self.prepared.detector_geom.wavelength, chunk_size,
            )
        return self._tcbf_cache[key]

    def _get_acbf_cache(
        self, rolloff: float = 0, chunk_size: int = 64,
    ) -> tuple[ACBFGeometryCache, ACBFOpticsCache | None]:
        """Return (geometry, optics_or_None).

        Geometry is mode-independent. Optics is built on demand only in 'full'
        mode; the consumer-facing return is gated on the current cache_mode.
        """
        key = (rolloff, chunk_size, *self._frame_cache_key())
        if key not in self._acbf_cache:
            kX_full, kY_full = self._get_transformed_centers()
            det = self.prepared.detector_geom
            geometry = build_acbf_geometry_cache(
                kX_full, kY_full,
                self.ab_state.order_keys, det.max_alpha, det.wavelength,
                self.ab_state.max_order, rolloff, chunk_size,
            )
            self._acbf_cache[key] = (geometry, None)

        geometry, optics = self._acbf_cache[key]
        if self.cache_mode == 'full':
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
    def _validate_upscale(upscale: int) -> None:
        if upscale != 1:
            raise NotImplementedError(
                "upscale is not supported in v2. Use upscale=1."
            )

    @staticmethod
    def _normalize_acbf_algorithm(acbf_algorithm) -> str:
        if acbf_algorithm is None:
            return 'phase_only'
        return str(acbf_algorithm).strip().lower().replace('-', '_')

    def _reconstruct_impl(self, mode: str = 'tcBF', **kwargs) -> torch.Tensor:
        mode_key = mode.lower()
        sg = self.scan_geom
        coeffs = self._get_scan_frame_coeffs()

        if mode_key == 'tcbf':
            self._validate_upscale(kwargs.get('upscale', 1))
            cache = self._get_tcbf_cache(chunk_size=kwargs.get('chunk_size', 64))
            image_fft = self._get_image_fft()
            return reconstruct_tcbf(
                image_fft, sg.qx_grid, sg.qy_grid, cache, coeffs, self.device,
            )

        if mode_key == 'acbf':
            self._validate_upscale(kwargs.get('upscale', 1))
            rolloff = kwargs.get('rolloff', 0)
            chunk_size = kwargs.get('chunk_size', 64)
            acbf_algorithm = self._normalize_acbf_algorithm(kwargs.get('acbf_algorithm'))
            geometry, optics = self._get_acbf_cache(rolloff=rolloff, chunk_size=chunk_size)
            image_fft = self._get_image_fft()

            if acbf_algorithm == 'phase_only':
                return reconstruct_acbf(
                    image_fft, sg.qx_grid, sg.qy_grid, geometry, optics,
                    coeffs, self.eps, self.device,
                )
            if acbf_algorithm == 'complex_inversion':
                return reconstruct_acbf_complex_inversion(
                    image_fft, sg.qx_grid, sg.qy_grid, geometry, optics,
                    coeffs, self.device,
                    regularization=kwargs.get('regularization', 1e-3),
                    support_threshold=kwargs.get('support_threshold', 1e-6),
                )
            raise ValueError(
                f"Unsupported acBF algorithm {acbf_algorithm!r}. "
                "Choose 'phase_only' or 'complex_inversion'."
            )

        raise ValueError(f"Unsupported mode {mode!r}. Choose 'tcBF' or 'acBF'.")

    def reconstruct(self, mode: str = 'tcBF', requires_grad: bool = False, **kwargs) -> torch.Tensor:
        """Run reconstruction. No-grad by default; opt-in for AD optimization paths."""
        if requires_grad:
            return self._reconstruct_impl(mode=mode, **kwargs)
        with torch.no_grad():
            return self._reconstruct_impl(mode=mode, **kwargs)
