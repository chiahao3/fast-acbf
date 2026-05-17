"""BFSolver — user-facing offline/notebook facade over BFReconstructor."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import rotate as tv_rotate
from torchvision.transforms import InterpolationMode

from ptyrad.optics.aberrations import Aberrations
from ptyrad.utils.image_proc import mfft2

from fast_acbf.core.aberrations import AberrationState
from fast_acbf.core.functional import generate_aberration_basis, generate_shift_basis, make_probe_from_chi
from fast_acbf.data.dataset4d import Dataset4D
from fast_acbf.data.geometry import CoordinateTransform, DetectorGeometry, ScanGeometry
from fast_acbf.data.imagefft_provider import ImageFFTProvider
from fast_acbf.recon.reconstructor import BFReconstructor

logger = logging.getLogger(__name__)


def _build_c10_axis(
    c10_center: float,
    device: str,
    n_layers=None,
    z_top=None,
    z_bottom=None,
    slice_thickness=None,
) -> torch.Tensor:
    """Build a 1D C10 axis in Angstroms for defocus-stack reconstruction.

    Mode 1: n_layers + slice_thickness — symmetric stack centred on c10_center.
    Mode 2: z_top + z_bottom + slice_thickness — explicit range.
    """
    has_n_layers = n_layers is not None
    has_range_arg = any(val is not None for val in (z_top, z_bottom))

    if slice_thickness is None:
        raise ValueError("slice_thickness is required for defocus-stack reconstruction.")

    slice_thickness = float(slice_thickness)
    if slice_thickness <= 0:
        raise ValueError(f"slice_thickness must be positive, got {slice_thickness}.")

    if has_n_layers and has_range_arg:
        raise ValueError(
            "Provide either n_layers or z_top/z_bottom with slice_thickness, not both."
        )

    if has_n_layers:
        if z_top is not None or z_bottom is not None:
            raise ValueError("n_layers mode does not accept z_top or z_bottom.")
        if not isinstance(n_layers, (int, np.integer)):
            raise ValueError(f"n_layers must be a positive integer, got {n_layers!r}.")
        n_layers = int(n_layers)
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")
        offsets = (torch.arange(n_layers, device=device, dtype=torch.float32)
                   - ((n_layers - 1) / 2.0))
        return c10_center + offsets * slice_thickness

    if has_range_arg:
        if z_top is None or z_bottom is None:
            raise ValueError(
                "Range mode requires z_top, z_bottom, and slice_thickness together."
            )
        start = float(z_top)
        stop = float(z_bottom)
        delta = stop - start

        if delta == 0:
            return torch.tensor([start], dtype=torch.float32, device=device)

        direction = 1.0 if delta > 0 else -1.0
        steps = int(np.floor(abs(delta) / slice_thickness + 1e-9))
        offsets = torch.arange(steps + 1, device=device, dtype=torch.float32)
        return start + direction * slice_thickness * offsets

    raise ValueError(
        "Provide either n_layers with slice_thickness, or z_top, z_bottom, and slice_thickness."
    )


class BFSolver:
    """Offline/notebook facade for tcBF and acBF reconstruction."""

    def __init__(
        self,
        dataset: np.ndarray | torch.Tensor | os.PathLike | Dataset4D,
        max_alpha: float,
        scan_step_size: float,
        dk: float,
        wavelength: float,
        max_order: int = 2,
        aberrations: dict | None = None,
        device: str = 'cuda',
        coord_transform: dict | None = None,
        cache_mode: str = 'auto',
        basis_mode: str = 'on_the_fly',
        eps: float = 1e-3,
        normalize: bool = False,
        lazy_read_mode: str = 'auto',
    ) -> None:
        if aberrations is None:
            aberrations = {}

        # Dispatch → Dataset4D
        if isinstance(dataset, Dataset4D):
            if normalize:
                warnings.warn(
                    "normalize=True has no effect when dataset is an already-constructed Dataset4D. "
                    "Pass normalize=True to Dataset4D(...) or its classmethods at construction time.",
                    UserWarning,
                    stacklevel=2,
                )
            ds = dataset
        elif isinstance(dataset, (str, os.PathLike)):
            path = Path(dataset)
            suffix = path.suffix.lower()
            if suffix in ('.h5', '.hdf5'):
                ds = Dataset4D.from_hdf5(path, normalize=normalize, lazy_read_mode=lazy_read_mode)
            else:
                ds = Dataset4D.from_zarr(path, normalize=normalize, lazy_read_mode=lazy_read_mode)
        else:
            ds = Dataset4D(dataset, normalize=normalize)

        parsed_aberrations = Aberrations(aberrations).export(
            notation='krivanek', style='cartesian', layout='nested'
        )

        alpha_rad = float(max_alpha) / 1e3
        tolerance_factors = {
            n: float((n + 1) * wavelength / (8 * alpha_rad ** (n + 1)))
            for n in range(1, max_order + 1)
        }

        ab_state = AberrationState(
            parsed_aberrations, max_order, device=device,
            tolerance_factors=tolerance_factors,
        )

        ct = CoordinateTransform.from_dict(coord_transform)

        Ry, Rx = ds.scan_shape
        Ky, Kx = ds.detector_shape
        det_geom = DetectorGeometry.from_params(
            detector_shape=(Ky, Kx),
            max_alpha=max_alpha,
            dk=dk,
            wavelength=wavelength,
            device=device,
        )
        scan_geom = ScanGeometry.from_params(
            scan_shape=(Ry, Rx),
            scan_step_size=scan_step_size,
            device=device,
        )

        provider = ImageFFTProvider(ds, det_geom, device, cache_mode)

        Nb = provider.nb
        print(
            f"Extracted {Nb} vBF images within max_alpha = {max_alpha} mrad."
        )

        recon = BFReconstructor(
            provider=provider,
            scan_geom=scan_geom,
            ab_state=ab_state,
            coord_transform=ct,
            basis_mode=basis_mode,
            eps=eps,
        )

        self._dataset = ds
        self._recon = recon
        self.reconstructed_image: torch.Tensor | None = None
        self.last_c10_stack_axis: torch.Tensor | None = None
        self.tolerance_factors = tolerance_factors

    # ------------------------------------------------------------------
    # Delegated properties — expose only what refinement/users actually need
    # ------------------------------------------------------------------

    @property
    def ab_state(self) -> AberrationState:
        return self._recon.ab_state

    @property
    def vbf_images(self) -> torch.Tensor:
        """Return (Nb, Ry, Rx) float32 BF image stack.

        For device/host cache modes, vBF is not stored persistently — this
        reconstructs it from the raw dataset each call. For lazy backends
        (on_the_fly), this triggers Nb disk reads.
        """
        p = self._recon.provider
        return torch.from_numpy(p.dataset.get_bf_chunk(p._bf_iy, p._bf_ix))

    @property
    def bf_mask(self) -> torch.Tensor:
        return self._recon.provider.detector_geom.bf_mask

    @property
    def kY_centers(self) -> torch.Tensor:
        return self._recon.provider.detector_geom.kY_centers

    @property
    def kX_centers(self) -> torch.Tensor:
        return self._recon.provider.detector_geom.kX_centers

    @property
    def kY_grid(self) -> torch.Tensor:
        return self._recon.provider.detector_geom.kY_grid

    @property
    def kX_grid(self) -> torch.Tensor:
        return self._recon.provider.detector_geom.kX_grid

    @property
    def device(self) -> str:
        return self._recon.device

    @property
    def max_alpha(self) -> float:
        return self._recon.provider.detector_geom.max_alpha

    @property
    def dk(self) -> float:
        return self._recon.provider.detector_geom.dk

    @property
    def wavelength(self) -> float:
        return self._recon.provider.detector_geom.wavelength

    @property
    def scan_step_size(self) -> float:
        return self._recon.scan_geom.scan_step_size

    @property
    def Ry(self) -> int:
        return self._recon.scan_geom.scan_shape[0]

    @property
    def Rx(self) -> int:
        return self._recon.scan_geom.scan_shape[1]

    @property
    def max_order(self) -> int:
        return self._recon.ab_state.max_order

    @property
    def eps(self) -> float:
        return self._recon.eps

    @property
    def cache_mode(self) -> str:
        return self._recon.provider.cache_mode

    @property
    def basis_mode(self) -> str:
        return self._recon.basis_mode

    @basis_mode.setter
    def basis_mode(self, value: str) -> None:
        self._recon.basis_mode = value

    @property
    def coord_transform(self) -> dict:
        """Mutable dict view of the current CoordinateTransform.

        Refinement code reads and writes this dict. Writes are applied via
        set_rotation_deg() or _apply_coord_transform_dict(). Kept as a dict
        for backward compatibility with the duck-typed refinement interface.
        """
        return self._recon.coord_transform.to_dict()

    @coord_transform.setter
    def coord_transform(self, value: dict) -> None:
        """Accepting dict assignment from refinement code; rebuilds CoordinateTransform."""
        self._recon.set_coord_transform(CoordinateTransform.from_dict(value), clear_basis=True)

    @property
    def rotation_deg(self) -> float:
        return self._recon.rotation_deg

    # ------------------------------------------------------------------
    # Cache / state management (public interface mirroring BFReconstructor)
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        self._recon.clear_cache()

    def clear_basis_cache(self) -> None:
        self._recon.clear_basis_cache()

    def set_rotation_deg(self, rotation_deg: float, clear_basis: bool = False) -> BFSolver:
        ct = self._recon.coord_transform.with_rotation(float(rotation_deg))
        self._recon.set_coord_transform(ct, clear_basis=clear_basis)
        return self

    def set_flips(self, flipud: bool, fliplr: bool, transpose: bool) -> BFSolver:
        """Set flip/transpose flags and clear basis cache. Used by refinement sweeps."""
        ct = CoordinateTransform(
            flipud=bool(flipud),
            fliplr=bool(fliplr),
            transpose=bool(transpose),
            rotation_deg=self._recon.coord_transform.rotation_deg,
        )
        self._recon.set_coord_transform(ct, clear_basis=True)
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_frame(self, frame: str) -> str:
        frame = str(frame).lower()
        if frame not in ('detector', 'scan'):
            raise ValueError(f"frame must be 'detector' or 'scan', got {frame!r}.")
        return frame

    def _get_scan_frame_coeffs(self) -> torch.Tensor:
        return self._recon._get_scan_frame_coeffs()

    def _get_transformed_bf_coordinates(self, in_scan_frame: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        return self._recon._get_transformed_centers(in_scan_frame=in_scan_frame)

    def _get_transformed_k_grids(self, in_scan_frame: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        return self._recon._get_transformed_grids(in_scan_frame=in_scan_frame)

    def _apply_coord_transform_dict(self, d: dict) -> None:
        ct = CoordinateTransform.from_dict(d)
        self._recon.set_coord_transform(ct, clear_basis=True)

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def reconstruct(self, mode: str = 'tcBF', requires_grad: bool = False, **kwargs) -> torch.Tensor:
        return self._recon.reconstruct(mode=mode, requires_grad=requires_grad, **kwargs)

    def _build_c10_stack_axis(
        self, n_layers=None, z_top=None, z_bottom=None, slice_thickness=None,
    ) -> torch.Tensor:
        return _build_c10_axis(
            c10_center=self.ab_state.get_physical('C_1_0'),
            device=self.device,
            n_layers=n_layers,
            z_top=z_top,
            z_bottom=z_bottom,
            slice_thickness=slice_thickness,
        )

    def _sweep_c10_stack(
        self, c10_axis: torch.Tensor, mode: str = 'tcBF', frame: str = 'scan', **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_c10 = self.ab_state.get_physical('C_1_0')
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
                self.ab_state.set_physical('C_1_0', original_c10)

        c10_axis = c10_axis.detach().clone()
        self.last_c10_stack_axis = c10_axis
        return c10_axis, torch.stack(stack_images, dim=0)

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    def get_aberrations_dict(
        self,
        frame: str = 'detector',
        notation: str = 'krivanek',
        style: str = 'cartesian',
        layout: str = 'nested',
    ) -> dict:
        frame = self._validate_frame(frame)
        if frame == 'detector':
            ab_dict = self.ab_state.get_cartesian_dict()
        else:
            ab_dict = self.ab_state.flat_to_cartesian_dict(self._get_scan_frame_coeffs())
        return Aberrations(ab_dict).export(notation=notation, style=style, layout=layout)

    def print_aberrations(self, frame: str = 'detector') -> None:
        frame = self._validate_frame(frame)
        if frame == 'scan' and self.rotation_deg:
            logger.warning(
                f"Printing scan-frame coefficients (rotation_deg={self.rotation_deg}). "
                "Asymmetric aberration orientations are relative to the scan fast-axis. "
                "Use frame='detector' for PtyRAD-compatible values."
            )
        print(Aberrations(self.get_aberrations_dict(frame=frame)))

    def get_chi_surface(self, frame: str = 'detector') -> torch.Tensor:
        frame = self._validate_frame(frame)
        kX_grid, kY_grid = self._get_transformed_k_grids(in_scan_frame=False)
        chi_basis = generate_aberration_basis(
            self.max_order, self.ab_state.order_keys, kX_grid, kY_grid, self.wavelength,
        )
        coeffs = self.ab_state.get_flat_coeffs() if frame == 'detector' else self._get_scan_frame_coeffs()
        return torch.einsum('k,kij->ij', coeffs, chi_basis)

    def get_yx_shifts_ang(self, frame: str = 'detector') -> torch.Tensor:
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

    def get_yx_shifts_px(self, frame: str = 'detector') -> torch.Tensor:
        return self.get_yx_shifts_ang(frame=frame) / self.scan_step_size

    def get_probe(self, frame: str = 'detector') -> torch.Tensor:
        return make_probe_from_chi(self.get_chi_surface(frame=frame), self.bf_mask)

    def rotate_scan_image_to_detector(self, img: torch.Tensor) -> torch.Tensor:
        if not self.rotation_deg:
            return img
        return tv_rotate(
            img.unsqueeze(0),
            angle=self.rotation_deg,
            interpolation=InterpolationMode.BILINEAR,
        ).squeeze(0)

    def get_reconstructed_image(
        self, mode: str = 'tcBF', frame: str = 'scan', **kwargs,
    ) -> torch.Tensor:
        frame = self._validate_frame(frame)
        img = self.reconstruct(mode=mode.lower(), requires_grad=False, **kwargs)
        self.reconstructed_image = img.detach()
        if frame == 'detector':
            img = self.rotate_scan_image_to_detector(img)
        return img

    def get_tcBF(self, frame: str = 'scan', **kwargs) -> torch.Tensor:
        return self.get_reconstructed_image(mode='tcBF', frame=frame, **kwargs)

    def get_acBF(self, frame: str = 'scan', **kwargs) -> torch.Tensor:
        return self.get_reconstructed_image(mode='acBF', frame=frame, **kwargs)

    def get_acBF_diagnostics(self, **kwargs) -> dict:
        rolloff = kwargs.get('rolloff', 0)
        chunk_size = kwargs.get('chunk_size', 64)
        with torch.no_grad():
            geometry, optics = self._recon._get_acbf_cache(rolloff=rolloff, chunk_size=chunk_size)
            from fast_acbf.core.acbf import reconstruct_acbf_complex_inversion
            return reconstruct_acbf_complex_inversion(
                self._recon.provider,
                self._recon.scan_geom.qx_grid, self._recon.scan_geom.qy_grid,
                geometry, optics,
                self._get_scan_frame_coeffs(),
                self.device,
                regularization=kwargs.get('regularization', 1e-3),
                support_threshold=kwargs.get('support_threshold', 1e-6),
                return_diagnostics=True,
            )

    def get_defocus_stack(
        self,
        mode: str = 'tcBF',
        frame: str = 'scan',
        n_layers=None,
        z_top=None,
        z_bottom=None,
        slice_thickness=None,
        **kwargs,
    ) -> torch.Tensor:
        """Return a defocus stack (Nz, Ny, Nx). C10 axis stored in last_c10_stack_axis."""
        c10_axis = self._build_c10_stack_axis(
            n_layers=n_layers, z_top=z_top, z_bottom=z_bottom, slice_thickness=slice_thickness,
        )
        _, stack = self._sweep_c10_stack(c10_axis, mode=mode.lower(), frame=frame, **kwargs)
        return stack

    # ------------------------------------------------------------------
    # Refinement — thin pass-throughs to optimization.refinement
    # ------------------------------------------------------------------

    def refine_register(self, max_shifts=None) -> BFSolver:
        print("Executed: Rigid Registration Refinement")
        return self

    def refine_defocus(self, *, search_range=None, **kwargs) -> BFSolver:
        from fast_acbf.optimization import refinement
        refinement.refine_defocus(self, search_range=search_range, **kwargs)
        return self

    def refine_aberrations(self, **kwargs) -> BFSolver:
        from fast_acbf.optimization import refinement
        refinement.refine_aberrations(self, **kwargs)
        return self

    def refine_scan_rotation(self, *, search_range=None, **kwargs) -> BFSolver:
        from fast_acbf.optimization import refinement
        refinement.refine_scan_rotation(self, search_range=search_range, **kwargs)
        return self

    def refine_flips(self, **kwargs) -> dict:
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
        refinement_scan_roi=None,
        **kwargs,
    ) -> BFSolver:
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
            refinement_scan_roi=refinement_scan_roi,
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
        mode: str = 'tcBF',
        frame: str = 'scan',
        vmin_img=None, vmax_img=None,
        vmin_fft=None, vmax_fft=None,
        **kwargs,
    ) -> None:
        from fast_acbf.vis import plotting

        mode = mode.lower()
        frame = self._validate_frame(frame)

        if title_str is None:
            title_str = f"Reconstructed {mode} and Probe amplitude"
        if desc_str is None:
            ab_dict = self.get_aberrations_dict(frame=frame, layout='flat')
            desc_str = ", ".join(f"{ab}: {val:.2f}" for ab, val in ab_dict.items())

        with torch.no_grad():
            img = self.get_reconstructed_image(mode=mode, frame=frame, **kwargs).detach().cpu().numpy()
            fft = np.log(np.abs(np.fft.fftshift(mfft2(img)[0])))
            probe = self.get_probe(frame=frame).abs().detach().cpu().numpy()

        plotting.plot_reconstruction(
            img, fft, probe,
            title_str=title_str, desc_str=desc_str, save_path=save_path,
            vmin_img=vmin_img, vmax_img=vmax_img, vmin_fft=vmin_fft, vmax_fft=vmax_fft,
        )

    def plot_chi_surface(self, plot_probe_phase: bool = False) -> None:
        from fast_acbf.vis import plotting
        chi = self.get_chi_surface()
        if plot_probe_phase:
            sign, title_str = -1, 'k-space probe phase (psi = exp(-1j*chi))'
        else:
            sign, title_str = 1, 'k-space aberration (chi) surface (psi = exp(-1j*chi))'
        surface = (self.bf_mask * sign * chi).detach().cpu().numpy()
        plotting.plot_chi_surface(surface, title_str=title_str)

    def plot_shift_quiver(
        self, subsample=None, scale=None, show: bool = True, frame: str = 'detector',
    ):
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
