"""Detector geometry, scan geometry, and coordinate frame transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# ---------------------------------------------------------------------------
# DetectorGeometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorGeometry:
    """Immutable detector-space quantities derived from physical parameters.

    All k-space quantities are in Å⁻¹.
    """
    max_alpha: float       # convergence semi-angle (mrad)
    dk: float              # detector pixel size (Å⁻¹/pixel)
    wavelength: float      # electron wavelength (Å)
    rolloff: float         # aperture rolloff width (mrad); 0 = hard edge

    # Derived tensors — built by from_params(), stored on the target device.
    kY_grid: torch.Tensor   # (Ky, Kx)
    kX_grid: torch.Tensor   # (Ky, Kx)
    bf_mask: torch.Tensor   # (Ky, Kx) float32 binary BF aperture
    kY_centers: torch.Tensor  # (Nb,) k-coords of BF pixels
    kX_centers: torch.Tensor  # (Nb,)

    # CPU boolean mask retained for vBF extraction.
    bf_mask_bool: np.ndarray  # (Ky, Kx) bool

    @classmethod
    def from_params(
        cls,
        detector_shape: tuple[int, int],
        max_alpha: float,
        dk: float,
        wavelength: float,
        device: str,
        rolloff: float = 0.0,
    ) -> DetectorGeometry:
        Ky_dim, Kx_dim = detector_shape
        ky = np.fft.fftshift(np.fft.fftfreq(Ky_dim, d=(1 / dk / Ky_dim)))
        kx = np.fft.fftshift(np.fft.fftfreq(Kx_dim, d=(1 / dk / Kx_dim)))
        kX_grid_np, kY_grid_np = np.meshgrid(kx, ky, indexing='xy')
        kR_grid = np.sqrt(kX_grid_np**2 + kY_grid_np**2)
        bf_mask_bool = kR_grid <= (max_alpha / 1e3 / wavelength)

        dev = torch.device(device)
        kY_grid = torch.tensor(kY_grid_np, dtype=torch.float32, device=dev)
        kX_grid = torch.tensor(kX_grid_np, dtype=torch.float32, device=dev)
        bf_mask = torch.tensor(bf_mask_bool, dtype=torch.float32, device=dev)
        kY_centers = torch.tensor(kY_grid_np[bf_mask_bool], dtype=torch.float32, device=dev)
        kX_centers = torch.tensor(kX_grid_np[bf_mask_bool], dtype=torch.float32, device=dev)

        return cls(
            max_alpha=float(max_alpha),
            dk=float(dk),
            wavelength=float(wavelength),
            rolloff=float(rolloff),
            kY_grid=kY_grid,
            kX_grid=kX_grid,
            bf_mask=bf_mask,
            kY_centers=kY_centers,
            kX_centers=kX_centers,
            bf_mask_bool=bf_mask_bool,
        )

    @property
    def detector_shape(self) -> tuple[int, int]:
        return tuple(self.kY_grid.shape)

    @property
    def n_bf_pixels(self) -> int:
        return int(self.kY_centers.shape[0])


# ---------------------------------------------------------------------------
# ScanGeometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanGeometry:
    """Immutable scan-space frequency grids derived from scan parameters."""
    scan_shape: tuple[int, int]   # (Ry, Rx)
    scan_step_size: float         # Å/pixel

    qx_grid: torch.Tensor  # (1, 1, Rx)
    qy_grid: torch.Tensor  # (1, Ry, 1)

    @classmethod
    def from_params(
        cls,
        scan_shape: tuple[int, int],
        scan_step_size: float,
        device: str,
    ) -> ScanGeometry:
        Ry, Rx = scan_shape
        dev = torch.device(device)
        qx_grid = torch.fft.fftfreq(Rx, d=scan_step_size, device=dev).view(1, 1, Rx)
        qy_grid = torch.fft.fftfreq(Ry, d=scan_step_size, device=dev).view(1, Ry, 1)
        return cls(
            scan_shape=tuple(scan_shape),
            scan_step_size=float(scan_step_size),
            qx_grid=qx_grid,
            qy_grid=qy_grid,
        )


# ---------------------------------------------------------------------------
# CoordinateTransform
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoordinateTransform:
    """Frozen value object encoding the detector → scan frame mapping.

    Operations applied in order: flipud → fliplr → transpose → rotation_deg.
    rotation_deg is CCW-positive in screen coordinates, matching PtyRAD's
    pos_scan_affine convention.
    """
    flipud: bool = False
    fliplr: bool = False
    transpose: bool = False
    rotation_deg: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> CoordinateTransform:
        if d is None:
            return cls()
        return cls(
            flipud=bool(d.get('flipud', False)),
            fliplr=bool(d.get('fliplr', False)),
            transpose=bool(d.get('transpose', False)),
            rotation_deg=float(d.get('rotation_deg', 0.0)),
        )

    def to_dict(self) -> dict:
        return {
            'flipud': self.flipud,
            'fliplr': self.fliplr,
            'transpose': self.transpose,
            'rotation_deg': self.rotation_deg,
        }

    def with_rotation(self, rotation_deg: float) -> CoordinateTransform:
        return CoordinateTransform(
            flipud=self.flipud,
            fliplr=self.fliplr,
            transpose=self.transpose,
            rotation_deg=float(rotation_deg),
        )

    def apply_to_centers(
        self,
        kY: torch.Tensor,
        kX: torch.Tensor,
        in_scan_frame: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (kX, kY) after applying flipud → fliplr → transpose → rotation.

        in_scan_frame=False skips the rotation step (returns detector-frame coords).
        """
        ky = kY.clone()
        kx = kX.clone()
        if self.flipud:
            ky = -ky
        if self.fliplr:
            kx = -kx
        if self.transpose:
            ky, kx = kx, ky
        if in_scan_frame and self.rotation_deg:
            theta = np.deg2rad(self.rotation_deg)
            kx_old = kx.clone()
            ky_old = ky.clone()
            kx = kx_old * np.cos(theta) - ky_old * np.sin(theta)
            ky = kx_old * np.sin(theta) + ky_old * np.cos(theta)
        return kx, ky

    def apply_to_grids(
        self,
        kY_grid: torch.Tensor,
        kX_grid: torch.Tensor,
        in_scan_frame: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (kX_grid, kY_grid) after applying the same transform as apply_to_centers."""
        return self.apply_to_centers(kY_grid, kX_grid, in_scan_frame=in_scan_frame)
