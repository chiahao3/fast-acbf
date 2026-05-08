"""BFSolver — stateful facade orchestrating tcBF and acBF reconstruction."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torchvision.transforms.functional import rotate as tv_rotate
from torchvision.transforms import InterpolationMode

from ptyrad.optics.aberrations import Aberrations
from ptyrad.utils.image_proc import mfft2

from fast_acbf.core.aberrations import AberrationState
from fast_acbf.core.acbf import reconstruct_acbf, reconstruct_acbf_complex_inversion
from fast_acbf.core.functional import generate_aberration_basis, generate_shift_basis, make_probe_from_chi
from fast_acbf.core.tcbf import reconstruct_tcbf
from fast_acbf.pipeline import (
    ACBFCache,
    BFImageCache,
    TCBFCache,
    build_acbf_cache,
    build_bf_image_cache,
    build_c10_axis,
    build_tcbf_cache,
    init_grid,
    init_vbf,
)

logger = logging.getLogger(__name__)


class BFSolver:
    def __init__(
        self,
        dataset: np.ndarray,
        max_alpha: float,
        scan_step_size: float,
        dk: float,
        wavelength: float,
        max_order: int,
        aberrations: dict,
        device='cuda',
        coord_transform=None,
        eps: float = 1e-3,
        cache_mode: str = 'full',
    ):
        """
        Initializes the solver. Dataset loading/parsing is assumed to be handled
        upstream (e.g., by PtyRAD's Initializer).

        Args:
            cache_mode: Controls how the static acBF/tcBF cache is stored.
                'full' — Precompute and cache all static acBF/tcBF tensors on device.
                'lazy' — Cache only FFTs and detector coordinates, regenerating heavy
                         acBF basis/aperture tensors during each reconstruction.
        """
        self.dataset = dataset
        self.max_alpha = max_alpha
        self.scan_step_size = scan_step_size
        self.dk = dk
        self.wavelength = wavelength
        self.max_order = max_order
        self.orig_aberrations = aberrations
        self.parsed_aberrations = Aberrations(aberrations).export(
            notation='krivanek', style='cartesian', layout='nested'
        )

        # Kirkland tolerance factors: T_n = (n+1)·lambda / (8·alpha_max^(n+1))
        alpha_rad = float(self.max_alpha) / 1e3
        self.tolerance_factors = {
            n: float((n + 1) * self.wavelength / (8 * alpha_rad ** (n + 1)))
            for n in range(1, max_order + 1)
        }
        # ab_state stores detector-frame coefficients. Reconstruction converts to scan
        # frame on demand via ab_state.to_scan_frame(rotation_deg).
        self.ab_state = AberrationState(
            self.parsed_aberrations, self.max_order, device=device,
            tolerance_factors=self.tolerance_factors,
        )
        self.eps = eps
        self.device = device

        _VALID_CACHE_MODES = ('full', 'lazy')
        cache_mode = str(cache_mode).strip().lower()
        if cache_mode not in _VALID_CACHE_MODES:
            raise ValueError(
                f"cache_mode must be one of {_VALID_CACHE_MODES}, got {cache_mode!r}."
            )
        self.cache_mode = cache_mode

        # Coordinate transform — maps acBF k-space orientation to the PtyRAD pipeline.
        #
        # flipud / fliplr / transpose:
        #   Correct discrete 90°-class detector orientation differences.
        #   Operations applied: flipud → fliplr → transpose (same order as PtyRAD _meas_flipT).
        #
        # rotation_deg:
        #   Scan rotation angle in detector frame (CCW positive as seen on screen).
        #   Maps to PtyRAD's pos_scan_affine rotation (same value, same sign).
        self.coord_transform = coord_transform or {
            'flipud': False,
            'fliplr': False,
            'transpose': False,
            'rotation_deg': 0.0,
        }
        self._rotation_deg = float(self.coord_transform.get('rotation_deg', 0.0))

        self.reconstructed_image = None
        self._basis_cache: dict = {}
        self._fft_cache: BFImageCache | None = None
        self.last_c10_stack_axis = None

        # One-time pipeline setup
        (self.vbf_images, self.kY_centers, self.kX_centers,
         self.kY_grid, self.kX_grid, self.bf_mask) = init_vbf(
            dataset, max_alpha, dk, wavelength, device
        )
        self.bf_coordinates = torch.stack([self.kY_centers, self.kX_centers], dim=-1)
        self.Ry, self.Rx = dataset.shape[0], dataset.shape[1]
        self.shift_grid = init_grid(self.Ry, self.Rx, device)

    # ------------------------------------------------------------------
    # Internal helpers — coordinate transforms & frame handling
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_acbf_algorithm(acbf_algorithm):
        if acbf_algorithm is None:
            return 'phase_only'
        return str(acbf_algorithm).strip().lower().replace('-', '_')

    @property
    def rotation_deg(self) -> float:
        """Scan rotation angle in degrees. Use set_rotation_deg() to update."""
        return self._rotation_deg

    def clear_basis_cache(self):
        """Clear orientation-dependent basis caches. FFT cache is preserved."""
        self._basis_cache = {}

    def clear_fft_cache(self):
        """Clear the FFT image cache. Also clears basis cache (entries reference FFT data)."""
        self._fft_cache = None
        self._basis_cache = {}

    def clear_cache(self):
        """Full reset — clears both basis and FFT caches."""
        self._basis_cache = {}
        self._fft_cache = None

    def _get_transform_flags(self):
        flipud = self.coord_transform.get('flipud', False)
        fliplr = self.coord_transform.get('fliplr', False)
        transpose = self.coord_transform.get('transpose', False)
        return flipud, fliplr, transpose, self.rotation_deg

    def _frame_cache_key(self):
        flipud, fliplr, transpose, _ = self._get_transform_flags()
        return (self.rotation_deg, flipud, fliplr, transpose)

    def _validate_frame(self, frame: str) -> str:
        frame = str(frame).lower()
        if frame not in ('detector', 'scan'):
            raise ValueError(f"frame must be 'detector' or 'scan', got {frame!r}.")
        return frame

    def _get_scan_frame_coeffs(self) -> torch.Tensor:
        return self.ab_state.to_scan_frame(self.rotation_deg)

    def _flat_to_cartesian_dict(self, flat: torch.Tensor) -> dict:
        out = {}
        idx = 0
        for (n, m) in self.ab_state.order_keys:
            if m == 0:
                out[(n, m)] = flat[idx].item()
                idx += 1
            else:
                out[(n, m)] = {'a': flat[idx].item(), 'b': flat[idx + 1].item()}
                idx += 2
        return out

    def set_rotation_deg(self, rotation_deg: float, clear_basis: bool = False):
        """
        Update scan rotation metadata.

        Basis caches include rotation in their keys so they do not need to be cleared
        for correctness. Set clear_basis=True to release the old rotation's basis cache.
        The FFT cache is orientation-independent and is never cleared here.
        """
        self._rotation_deg = float(rotation_deg)
        self.coord_transform['rotation_deg'] = self._rotation_deg
        if clear_basis:
            self.clear_basis_cache()
        return self

    def _get_transformed_bf_coordinates(self, in_scan_frame=True):
        """
        Return transformed reciprocal-space BF coordinates as (kX, kY).

        Operations applied: flipud → fliplr → transpose → rotation_deg.
        in_scan_frame=True  → all four steps; in_scan_frame=False → rotation skipped.
        """
        ky = self.kY_centers.clone()
        kx = self.kX_centers.clone()
        flipud, fliplr, transpose, rotation_deg = self._get_transform_flags()

        if flipud:
            ky = -ky
        if fliplr:
            kx = -kx
        if transpose:
            ky, kx = kx, ky
        if in_scan_frame and rotation_deg:
            theta = np.deg2rad(rotation_deg)
            kx_old = kx.clone()
            ky_old = ky.clone()
            kx = kx_old * np.cos(theta) - ky_old * np.sin(theta)
            ky = kx_old * np.sin(theta) + ky_old * np.cos(theta)

        return kx, ky

    def _get_transformed_k_grids(self, in_scan_frame=True):
        """
        Return transformed reciprocal-space full grids as (kX_grid, kY_grid).

        Same operation order as _get_transformed_bf_coordinates.
        """
        ky = self.kY_grid.clone()
        kx = self.kX_grid.clone()
        flipud, fliplr, transpose, rotation_deg = self._get_transform_flags()

        if flipud:
            ky = -ky
        if fliplr:
            kx = -kx
        if transpose:
            ky, kx = kx, ky
        if in_scan_frame and rotation_deg:
            theta = np.deg2rad(rotation_deg)
            kx_old = kx.clone()
            ky_old = ky.clone()
            kx = kx_old * np.cos(theta) - ky_old * np.sin(theta)
            ky = kx_old * np.sin(theta) + ky_old * np.cos(theta)

        return kx, ky

    # ------------------------------------------------------------------
    # Cache management — lazy-build wrappers
    # ------------------------------------------------------------------

    def _get_fft_cache(self) -> BFImageCache:
        """Return the orientation-independent FFT image cache, building it on first call."""
        if self._fft_cache is None:
            self._fft_cache = build_bf_image_cache(
                self.vbf_images, self.scan_step_size, self.device,
            )
        return self._fft_cache

    def _get_tcBF_cache(self, chunk_size=64) -> TCBFCache:
        key = ('tcBF', chunk_size, *self._frame_cache_key())
        if key not in self._basis_cache:
            kX_full, kY_full = self._get_transformed_bf_coordinates()
            ic = self._get_fft_cache()
            self._basis_cache[key] = build_tcbf_cache(
                kX_full, kY_full,
                self.ab_state.order_keys, self.wavelength, self.device, chunk_size,
                img_fft=ic.img_fft, qx_grid=ic.qx_grid, qy_grid=ic.qy_grid, out_shape=ic.out_shape,
            )
        return self._basis_cache[key]

    def _get_acBF_cache(self, rolloff=0, chunk_size=64) -> ACBFCache:
        key = ('acBF', rolloff, chunk_size, self.cache_mode, *self._frame_cache_key())
        if key not in self._basis_cache:
            kX_full, kY_full = self._get_transformed_bf_coordinates()
            ic = self._get_fft_cache()
            self._basis_cache[key] = build_acbf_cache(
                kX_full, kY_full,
                self.ab_state.order_keys, self.max_alpha,
                self.wavelength, self.max_order,
                self.device, self.cache_mode, rolloff, chunk_size,
                img_fft=ic.img_fft, qx_grid=ic.qx_grid, qy_grid=ic.qy_grid, out_shape=ic.out_shape,
            )
        return self._basis_cache[key]

    # ------------------------------------------------------------------
    # Reconstruction orchestration
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_native_upscale(upscale):
        if upscale != 1:
            raise NotImplementedError(
                "upscale is temporarily unsupported during the native-resolution "
                "cache cleanup. Use upscale=1."
            )

    def reconstruct(self, mode='tcBF', **kwargs):
        """
        Unified reconstruction entry point.

        Notes:
            - tcBF is the default mode.
            - acBF supports acbf_algorithm='phase_only' (default) and
              acbf_algorithm='complex_inversion'.
        """
        mode_key = mode.lower()

        if mode_key == 'tcbf':
            self._validate_native_upscale(kwargs.get('upscale', 1))
            cache = self._get_tcBF_cache(chunk_size=kwargs.get('chunk_size', 64))
            return reconstruct_tcbf(cache, self._get_scan_frame_coeffs(), self.device)

        elif mode_key == 'acbf':
            self._validate_native_upscale(kwargs.get('upscale', 1))
            rolloff = kwargs.get('rolloff', 0)
            chunk_size = kwargs.get('chunk_size', 64)
            acbf_algorithm = self._normalize_acbf_algorithm(kwargs.get('acbf_algorithm', 'phase_only'))
            cache = self._get_acBF_cache(rolloff=rolloff, chunk_size=chunk_size)
            coeffs = self._get_scan_frame_coeffs()

            if acbf_algorithm == 'phase_only':
                return reconstruct_acbf(cache, coeffs, self.eps, self.device)
            if acbf_algorithm == 'complex_inversion':
                return reconstruct_acbf_complex_inversion(
                    cache, coeffs, self.device,
                    regularization=kwargs.get('regularization', 1e-3),
                    support_threshold=kwargs.get('support_threshold', 1e-6),
                )
            raise ValueError(
                f"Unsupported acBF algorithm '{acbf_algorithm}'. "
                "Choose between 'phase_only' and 'complex_inversion'."
            )

        raise ValueError(f"Unsupported mode '{mode}'. Choose between 'tcBF' and 'acBF'.")

    def _build_c10_stack_axis(self, n_layers=None, z_top=None, z_bottom=None, slice_thickness=None):
        return build_c10_axis(
            c10_center=self.ab_state.get_physical('C_1_0'),
            device=self.device,
            n_layers=n_layers,
            z_top=z_top,
            z_bottom=z_bottom,
            slice_thickness=slice_thickness,
        )

    def _sweep_c10_stack(self, c10_axis, mode='tcBF', frame='scan', **kwargs):
        """
        Evaluate a read-only reconstruction stack over an absolute C10 axis.

        Updates self.last_c10_stack_axis. Does not modify self.reconstructed_image.
        """
        original_c10 = self.ab_state.coeffs['C_1_0'].detach().clone()
        stack_images = []

        try:
            with torch.no_grad():
                for c10 in c10_axis:
                    self.ab_state.set_physical('C_1_0', c10)
                    img = self.reconstruct(mode=mode, **kwargs)
                    if self._validate_frame(frame) == 'detector':
                        img = self.rotate_scan_image_to_detector(img)
                    stack_images.append(img)
        finally:
            with torch.no_grad():
                self.ab_state.coeffs['C_1_0'].copy_(original_c10)

        c10_axis = c10_axis.detach().clone()
        stack = torch.stack(stack_images, dim=0)
        self.last_c10_stack_axis = c10_axis
        return c10_axis, stack

    # ------------------------------------------------------------------
    # Public API — getters
    # ------------------------------------------------------------------

    def get_aberrations_dict(self, frame='detector', notation='krivanek', style='cartesian', layout='nested'):
        """Return aberration coefficients in the requested notation/style/layout."""
        frame = self._validate_frame(frame)
        if frame == 'detector':
            ab_dict = self.ab_state.get_cartesian_dict()
        else:
            ab_dict = self._flat_to_cartesian_dict(self._get_scan_frame_coeffs())
        return Aberrations(ab_dict).export(notation=notation, style=style, layout=layout)

    def print_aberrations(self, frame='detector'):
        """Print aberration coefficients."""
        frame = self._validate_frame(frame)
        if frame == 'scan' and self.rotation_deg:
            logger.warning(
                f"Printing scan-frame coefficients (rotation_deg={self.rotation_deg}). "
                "Asymmetric aberration orientations are relative to the scan fast-axis. "
                "Use frame='detector' for PtyRAD-compatible values."
            )
        print(Aberrations(self.get_aberrations_dict(frame=frame)))

    def get_chi_surface(self, frame='detector'):
        """
        Return aberration surface chi. Note: psi = exp(-1j*chi).

        The chi/probe raster is always the canonical flip/transpose-corrected
        detector raster. frame selects which coefficient frame is evaluated.
        """
        frame = self._validate_frame(frame)
        kX_grid, kY_grid = self._get_transformed_k_grids(in_scan_frame=False)
        chi_basis = generate_aberration_basis(
            self.max_order, self.ab_state.order_keys, kX_grid, kY_grid, self.wavelength,
        )
        coeffs = self.ab_state.get_flat_coeffs() if frame == 'detector' else self._get_scan_frame_coeffs()
        return torch.einsum('k,kij->ij', coeffs, chi_basis)

    def get_yx_shifts_ang(self, frame='detector'):
        """Return image shifts in Angstroms as (Nb, 2) tensor (shift_y, shift_x)."""
        frame = self._validate_frame(frame)
        in_scan = frame == 'scan'
        kX_centers, kY_centers = self._get_transformed_bf_coordinates(in_scan_frame=in_scan)
        b_dx, b_dy = generate_shift_basis(
            self.ab_state.order_keys, kX_centers, kY_centers, self.wavelength,
        )
        coeffs = self._get_scan_frame_coeffs() if in_scan else self.ab_state.get_flat_coeffs()
        shift_x_ang = torch.einsum('k,kb->b', coeffs, b_dx)
        shift_y_ang = torch.einsum('k,kb->b', coeffs, b_dy)
        return torch.stack([shift_y_ang, shift_x_ang], dim=-1)

    def get_yx_shifts_px(self, frame='detector'):
        """Return image shifts in real-space pixels as (Nb, 2) tensor."""
        return self.get_yx_shifts_ang(frame=frame) / self.scan_step_size

    def rotate_scan_image_to_detector(self, img: torch.Tensor) -> torch.Tensor:
        """Rotate a scan-frame image to detector frame via bilinear resampling (display path)."""
        if not self.rotation_deg:
            return img
        return tv_rotate(
            img.unsqueeze(0),
            angle=self.rotation_deg,
            interpolation=InterpolationMode.BILINEAR,
        ).squeeze(0)

    def get_reconstructed_image(self, mode='tcBF', frame='scan', **kwargs):
        """Return the reconstructed image, optionally rotated to detector frame."""
        mode = mode.lower()
        frame = self._validate_frame(frame)
        img = self.reconstruct(mode=mode, **kwargs)
        self.reconstructed_image = img

        if frame == 'detector':
            img = self.rotate_scan_image_to_detector(img)

        return img

    def get_tcBF(self, frame='scan', **kwargs):
        """Return tcBF image."""
        return self.get_reconstructed_image(mode='tcBF', frame=frame, **kwargs)

    def get_acBF(self, frame='scan', **kwargs):
        """Return acBF image."""
        return self.get_reconstructed_image(mode='acBF', frame=frame, **kwargs)

    def get_acBF_diagnostics(self, **kwargs):
        """Return transfer diagnostics for the complex-inversion acBF estimator."""
        self._validate_native_upscale(kwargs.get('upscale', 1))
        rolloff = kwargs.get('rolloff', 0)
        chunk_size = kwargs.get('chunk_size', 64)
        cache = self._get_acBF_cache(rolloff=rolloff, chunk_size=chunk_size)
        return reconstruct_acbf_complex_inversion(
            cache,
            self._get_scan_frame_coeffs(),
            self.device,
            regularization=kwargs.get('regularization', 1e-3),
            support_threshold=kwargs.get('support_threshold', 1e-6),
            return_diagnostics=True,
        )

    def get_defocus_stack(
        self,
        mode='tcBF',
        frame='scan',
        n_layers=None,
        z_top=None,
        z_bottom=None,
        slice_thickness=None,
        **kwargs,
    ):
        """
        Return a read-only defocus stack with shape (Nz, Ny, Nx).

        The stack axis is absolute C10 in Angstroms, stored in self.last_c10_stack_axis.
        Supported modes: n_layers + slice_thickness, or z_top + z_bottom + slice_thickness.
        """
        mode = mode.lower()
        c10_axis = self._build_c10_stack_axis(
            n_layers=n_layers, z_top=z_top, z_bottom=z_bottom, slice_thickness=slice_thickness,
        )
        _, stack = self._sweep_c10_stack(c10_axis, mode=mode, frame=frame, **kwargs)
        return stack

    def get_probe(self, frame='detector'):
        """Return the complex probe wavefield on the canonical detector raster."""
        return make_probe_from_chi(self.get_chi_surface(frame=frame), self.bf_mask)

    # ------------------------------------------------------------------
    # Refinement — thin pass-throughs to optimization.refinement
    # ------------------------------------------------------------------

    def refine_register(self, max_shifts=None):
        """Refines shifts using rigid registration (not yet implemented)."""
        print("Executed: Rigid Registration Refinement")
        return self

    def refine_defocus(self, search_range: tuple, **kwargs):
        """Line search for optimal C10 (defocus). See optimization.refinement.refine_defocus."""
        from fast_acbf.optimization import refinement
        refinement.refine_defocus(self, search_range, **kwargs)
        return self

    def refine_aberrations(self, **kwargs):
        """Gradient-based aberration refinement. See optimization.refinement.refine_aberrations."""
        from fast_acbf.optimization import refinement
        refinement.refine_aberrations(self, **kwargs)
        return self

    def refine_scan_rotation(self, search_range: tuple, **kwargs):
        """Line search for optimal scan rotation. See optimization.refinement.refine_scan_rotation."""
        from fast_acbf.optimization import refinement
        refinement.refine_scan_rotation(self, search_range, **kwargs)
        return self

    def refine_flips(self, **kwargs):
        """Exhaustive flip/transpose search. See optimization.refinement.refine_flips."""
        from fast_acbf.optimization import refinement
        return refinement.refine_flips(self, **kwargs)

    def refine_all_params(
        self,
        targets=('orientation_defocus', 'coarse_aberrations', 'fine_rotation', 'fine_aberrations'),
        metric: str = 'normalized_std',
        mode: str = 'tcBF',
        defocus_range=None,
        defocus_range_tolerance_factor: float = 24.0,
        rotation_num_points: int = 36,
        defocus_num_points: int = 11,
        fine_rotation_halfwidth: float = 5.0,
        fine_rotation_num_points: int = 11,
        aberration_lr: float = 1.0,
        aberration_iters: int = 50,
        **kwargs,
    ) -> 'BFSolver':
        """Coarse-to-fine parameter orchestration. See optimization.refinement.refine_all_params."""
        from fast_acbf.optimization import refinement
        refinement.refine_all_params(
            self,
            targets=targets,
            metric=metric,
            mode=mode,
            defocus_range=defocus_range,
            defocus_range_tolerance_factor=defocus_range_tolerance_factor,
            rotation_num_points=rotation_num_points,
            defocus_num_points=defocus_num_points,
            fine_rotation_halfwidth=fine_rotation_halfwidth,
            fine_rotation_num_points=fine_rotation_num_points,
            aberration_lr=aberration_lr,
            aberration_iters=aberration_iters,
            **kwargs,
        )
        return self

    # ------------------------------------------------------------------
    # Plotting — thin pass-throughs to vis.plotting
    # ------------------------------------------------------------------

    def plot_reconstruction(
        self,
        title_str=None,
        desc_str=None,
        save_path=None,
        mode='tcBF',
        frame='scan',
        vmin_img=None,
        vmax_img=None,
        vmin_fft=None,
        vmax_fft=None,
        **kwargs,
    ):
        from fast_acbf.vis import plotting

        mode = mode.lower()
        frame = self._validate_frame(frame)

        if title_str is None:
            title_str = f"Reconstructed {mode} and Probe amplitude"
        if desc_str is None:
            ab_dict = self.get_aberrations_dict(frame=frame, layout='flat')
            desc_str = ", ".join(f"{ab}: {val:.2f}" for ab, val in ab_dict.items())

        img = self.get_reconstructed_image(mode=mode, frame=frame, **kwargs).detach().cpu().numpy()
        fft = np.log(np.abs(np.fft.fftshift(mfft2(img)[0])))
        probe = self.get_probe(frame=frame).abs().detach().cpu().numpy()

        plotting.plot_reconstruction(
            img, fft, probe,
            title_str=title_str, desc_str=desc_str, save_path=save_path,
            vmin_img=vmin_img, vmax_img=vmax_img, vmin_fft=vmin_fft, vmax_fft=vmax_fft,
        )

    def plot_chi_surface(self, plot_probe_phase=False):
        """Plot the aberration (chi) surface."""
        from fast_acbf.vis import plotting

        chi = self.get_chi_surface()
        if plot_probe_phase:
            sign, title_str = -1, 'k-space probe phase (psi = exp(-1j*chi))'
        else:
            sign, title_str = 1, 'k-space aberration (chi) surface (psi = exp(-1j*chi))'

        surface = (self.bf_mask * sign * chi).detach().cpu().numpy()
        plotting.plot_chi_surface(surface, title_str=title_str)

    def plot_shift_quiver(self, subsample=None, scale=None, show=True, frame='detector'):
        """Quiver plot of image shifts over BF disk."""
        from fast_acbf.vis import plotting

        frame = self._validate_frame(frame)
        in_scan = frame == 'scan'
        with torch.no_grad():
            shift_yx_ang = self.get_yx_shifts_ang(frame=frame)
            kx, ky = self._get_transformed_bf_coordinates(in_scan_frame=in_scan)

        k_max = self.max_alpha / 1e3 / self.wavelength
        return plotting.plot_shift_quiver(
            kx=kx.cpu().numpy(),
            ky=ky.cpu().numpy(),
            sx=shift_yx_ang[:, 1].cpu().numpy(),
            sy=shift_yx_ang[:, 0].cpu().numpy(),
            k_max=k_max,
            subsample=subsample,
            scale=scale,
            show=show,
        )
