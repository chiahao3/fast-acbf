"""One-time setup and cache management for tcBF and acBF reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fast_acbf.core.functional import (
    generate_aberration_basis,
    generate_shift_basis,
    make_soft_aperture_torch,
)


@dataclass
class BFImageCache:
    """Static, mode-independent, orientation-independent. Single allocation per session."""
    img_fft: torch.Tensor   # (Nb, Ry, Rx) — full BF stack FFT; dtype matches fft2(vbf_images)
    qx_grid: torch.Tensor   # (1, 1, Rx) — fftfreq(Rx, d=scan_step_size)
    qy_grid: torch.Tensor   # (1, Ry, 1) — fftfreq(Ry, d=scan_step_size)
    out_shape: tuple         # (Ry, Rx)


@dataclass
class TCBFCache:
    chunks: list          # each: {b_dx, b_dy, start, end}
    img_fft: torch.Tensor # (Nb, Ry, Rx) — shared reference from BFImageCache
    qx_grid: torch.Tensor # (1, 1, Rx) — shared reference
    qy_grid: torch.Tensor # (1, Ry, 1) — shared reference
    out_shape: tuple


@dataclass
class ACBFCache:
    chunks: list          # full: {ap_t,ap_mt,b_tr,b_t,b_mt,start,end}
                          # lazy: {kxt,kyt,start,end}
    img_fft: torch.Tensor # (Nb, Ry, Rx) — shared reference from BFImageCache
    qx_grid: torch.Tensor # (1, 1, Rx) — shared reference
    qy_grid: torch.Tensor # (1, Ry, 1) — shared reference
    out_shape: tuple
    rolloff: float
    cache_mode: str
    # Physics params stored for lazy-recompute path
    max_order: int
    order_keys: list
    max_alpha: float
    wavelength: float


def init_vbf(dataset: np.ndarray, max_alpha: float, dk: float, wavelength: float, device: str):
    """
    Build the BF mask, extract vBF image stack, and return k-space coordinate tensors.

    Returns:
        vbf_images (Tensor): (Nb, Ry, Rx) float32 on device
        kY_centers (Tensor): (Nb,) BF pixel k-coords in Å⁻¹
        kX_centers (Tensor): (Nb,)
        kY_grid (Tensor): (Ky, Kx) full k-grid
        kX_grid (Tensor): (Ky, Kx)
        bf_mask (Tensor): (Ky, Kx) float32 binary mask
    """
    Ry_dim, Rx_dim, Ky_dim, Kx_dim = dataset.shape

    ky = np.fft.fftshift(np.fft.fftfreq(Ky_dim, d=(1 / dk / Ky_dim)))
    kx = np.fft.fftshift(np.fft.fftfreq(Kx_dim, d=(1 / dk / Kx_dim)))
    kX_grid, kY_grid = np.meshgrid(kx, ky, indexing='xy')

    kR_grid = np.sqrt(kX_grid**2 + kY_grid**2)
    bf_mask = kR_grid <= (max_alpha / 1e3 / wavelength)

    kY_centers_np = kY_grid[bf_mask]
    kX_centers_np = kX_grid[bf_mask]

    vbf_np = dataset[:, :, bf_mask]
    vbf_np = np.ascontiguousarray(np.moveaxis(vbf_np, -1, 0))

    kY_centers = torch.tensor(kY_centers_np, dtype=torch.float32, device=device)
    kX_centers = torch.tensor(kX_centers_np, dtype=torch.float32, device=device)
    kX_grid_t = torch.tensor(kX_grid, dtype=torch.float32, device=device)
    kY_grid_t = torch.tensor(kY_grid, dtype=torch.float32, device=device)
    bf_mask_t = torch.tensor(bf_mask, dtype=torch.float32, device=device)
    vbf_images = torch.tensor(vbf_np, dtype=torch.float32, device=device)

    print(f"Extracted {vbf_images.shape[0]} vBF images within the max alpha angle = {max_alpha} mrad.")
    return vbf_images, kY_centers, kX_centers, kY_grid_t, kX_grid_t, bf_mask_t


def init_grid(Ry: int, Rx: int, device: str) -> torch.Tensor:
    """Build the FFT shift grid for real-space pixel shifts. Returns shape (2, Ry, Rx)."""
    kpy, kpx = torch.meshgrid(
        torch.fft.fftfreq(Ry, dtype=torch.float32, device=device),
        torch.fft.fftfreq(Rx, dtype=torch.float32, device=device),
        indexing='ij',
    )
    return torch.stack([kpy, kpx], dim=0)


def build_bf_image_cache(
    vbf_images: torch.Tensor,
    scan_step_size: float,
    device: str,
) -> BFImageCache:
    """Pre-compute the full-stack FFT and scan-frame frequency grids. Mode-independent."""
    _, Ry, Rx = vbf_images.shape
    img_fft = torch.fft.fft2(vbf_images, dim=(-2, -1))
    qx_grid = torch.fft.fftfreq(Rx, d=scan_step_size, device=device).view(1, 1, Rx)
    qy_grid = torch.fft.fftfreq(Ry, d=scan_step_size, device=device).view(1, Ry, 1)
    return BFImageCache(img_fft=img_fft, qx_grid=qx_grid, qy_grid=qy_grid, out_shape=(Ry, Rx))


def build_tcbf_cache(
    kX_full: torch.Tensor,
    kY_full: torch.Tensor,
    order_keys: list,
    wavelength: float,
    device: str,
    chunk_size: int = 64,
    *,
    img_fft: torch.Tensor,
    qx_grid: torch.Tensor,
    qy_grid: torch.Tensor,
    out_shape: tuple,
) -> TCBFCache:
    """Pre-compute the orientation-dependent shift basis chunks.

    img_fft, qx_grid, qy_grid, out_shape are shared references from BFImageCache.
    """
    Nb = img_fft.shape[0]
    chunks = []
    for i in range(0, Nb, chunk_size):
        end = min(i + chunk_size, Nb)
        b_dx, b_dy = generate_shift_basis(order_keys, kX_full[i:end], kY_full[i:end], wavelength)
        chunks.append({'b_dx': b_dx, 'b_dy': b_dy, 'start': i, 'end': end})

    return TCBFCache(chunks=chunks, img_fft=img_fft, qx_grid=qx_grid,
                    qy_grid=qy_grid, out_shape=out_shape)


def build_acbf_cache(
    kX_full: torch.Tensor,
    kY_full: torch.Tensor,
    order_keys: list,
    max_alpha: float,
    wavelength: float,
    max_order: int,
    device: str,
    cache_mode: str = 'lazy',
    rolloff: float = 0,
    chunk_size: int = 64,
    *,
    img_fft: torch.Tensor,
    qx_grid: torch.Tensor,
    qy_grid: torch.Tensor,
    out_shape: tuple,
) -> ACBFCache:
    """
    Pre-compute and chunk all static geometry and soft apertures.

    img_fft, qx_grid, qy_grid, out_shape are shared references from BFImageCache.
    qx_grid / qy_grid replace the locally computed kx_base / ky_base.

    What is stored per chunk depends on cache_mode:
        'full' — bases + apertures on device.
        'lazy' — only detector k-coords (kxt/kyt); bases/apertures recomputed each call.
    """
    Nb = img_fft.shape[0]

    kX_full = kX_full.view(Nb, 1, 1)
    kY_full = kY_full.view(Nb, 1, 1)

    chunks = []
    for i in range(0, Nb, chunk_size):
        end = min(i + chunk_size, Nb)
        kxt = kX_full[i:end]
        kyt = kY_full[i:end]

        if cache_mode == 'lazy':
            chunks.append({
                'kxt': kxt.detach().clone(),
                'kyt': kyt.detach().clone(),
                'start': i, 'end': end,
            })
            continue

        kx_t,  ky_t  = qx_grid + kxt, qy_grid + kyt
        kx_mt, ky_mt = qx_grid - kxt, qy_grid - kyt

        ap_t  = make_soft_aperture_torch(torch.sqrt(kx_t**2  + ky_t**2)  * wavelength, max_alpha, rolloff)
        ap_mt = make_soft_aperture_torch(torch.sqrt(kx_mt**2 + ky_mt**2) * wavelength, max_alpha, rolloff)

        b_tr = generate_aberration_basis(max_order, order_keys, kxt,    kyt,    wavelength)
        b_t  = generate_aberration_basis(max_order, order_keys, kx_t,   ky_t,   wavelength)
        b_mt = generate_aberration_basis(max_order, order_keys, -kx_mt, -ky_mt, wavelength)

        chunks.append({
            'ap_t': ap_t, 'ap_mt': ap_mt,
            'b_tr': b_tr, 'b_t': b_t, 'b_mt': b_mt,
            'start': i, 'end': end,
        })

    return ACBFCache(
        chunks=chunks,
        img_fft=img_fft,
        qx_grid=qx_grid,
        qy_grid=qy_grid,
        out_shape=out_shape,
        rolloff=rolloff,
        cache_mode=cache_mode,
        max_order=max_order,
        order_keys=order_keys,
        max_alpha=max_alpha,
        wavelength=wavelength,
    )


def build_c10_axis(
    c10_center: float,
    device: str,
    n_layers=None,
    z_top=None,
    z_bottom=None,
    slice_thickness=None,
) -> torch.Tensor:
    """
    Build a 1D absolute C10 axis in Angstroms for defocus-stack reconstruction.

    Supported mutually exclusive modes:
        1. n_layers + slice_thickness  — symmetric stack centered on c10_center
        2. z_top + z_bottom + slice_thickness  — explicit range
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
        steps = int(np.floor(abs(delta) / slice_thickness))
        offsets = torch.arange(steps + 1, device=device, dtype=torch.float32)
        return start + direction * slice_thickness * offsets

    raise ValueError(
        "Provide either n_layers with slice_thickness, or z_top, z_bottom, and slice_thickness."
    )
