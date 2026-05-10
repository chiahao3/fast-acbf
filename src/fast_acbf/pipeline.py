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
class ImageFFT:
    """Pre-computed FFT of the BF image stack. Mutable for live acquisition."""
    img_fft: torch.Tensor  # (Nb, Ry, Rx) complex64

    def fft_chunk(self, start: int, end: int) -> torch.Tensor:
        return self.img_fft[start:end]


@dataclass
class TCBFCache:
    """Per-chunk shift basis for tcBF. Orientation-dependent."""
    chunks: list  # each: {b_dx, b_dy, start, end}


@dataclass
class ACBFGeometryCache:
    """Per-chunk detector k-coords plus physics params needed to derive optics on demand."""
    chunks: list           # each: {kxt, kyt, start, end}
    max_order: int
    order_keys: list
    max_alpha: float
    wavelength: float
    rolloff: float


@dataclass
class ACBFOpticsCache:
    """Per-chunk apertures and aberration bases. Built only in 'full' mode."""
    chunks: list  # each: {ap_t, ap_mt, b_tr, b_t, b_mt, start, end}


def compute_bf_geometry(
    Ky_dim: int, Kx_dim: int, max_alpha: float, dk: float, wavelength: float,
):
    """Build the k-space coordinate grids and the boolean BF aperture mask.

    Returns numpy arrays only — caller decides where to put them.
    """
    ky = np.fft.fftshift(np.fft.fftfreq(Ky_dim, d=(1 / dk / Ky_dim)))
    kx = np.fft.fftshift(np.fft.fftfreq(Kx_dim, d=(1 / dk / Kx_dim)))
    kX_grid, kY_grid = np.meshgrid(kx, ky, indexing='xy')
    kR_grid = np.sqrt(kX_grid**2 + kY_grid**2)
    bf_mask_bool = kR_grid <= (max_alpha / 1e3 / wavelength)
    return kY_grid, kX_grid, bf_mask_bool


def extract_vbf_stack(
    dataset: np.ndarray,
    bf_mask_bool: np.ndarray,
    device: str,
    *,
    out: torch.Tensor | None = None,
    pinned_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the BF mask to a 4D dataset and stage onto `device`.

    Layout: dataset is (Ry, Rx, Ky, Kx); output is (Nb, Ry, Rx) float32.

    Live-acquisition fast path:
        - If `pinned_buffer` is provided and shape-matches, the masked stack is
          copied into it (host pinned memory) and then to `out` (device tensor)
          with non_blocking=True. Caller reuses both buffers across frames.
        - If `out` is provided, the result is copied in-place into it.
        - If neither is provided, both are freshly allocated.

    Returns (vbf_images, pinned_buffer) so the caller can stash the pinned
    buffer for reuse on the next frame. Pinned buffer is None on CPU device.
    """
    masked = dataset[:, :, bf_mask_bool]
    masked = np.ascontiguousarray(np.moveaxis(masked, -1, 0)).astype(np.float32, copy=False)
    expected_shape = masked.shape

    use_pinned = str(device).startswith('cuda')

    if use_pinned:
        if pinned_buffer is None or tuple(pinned_buffer.shape) != expected_shape:
            pinned_buffer = torch.empty(expected_shape, dtype=torch.float32, pin_memory=True)
        pinned_buffer.copy_(torch.from_numpy(masked))
        if out is None or tuple(out.shape) != expected_shape:
            out = torch.empty(expected_shape, dtype=torch.float32, device=device)
        out.copy_(pinned_buffer, non_blocking=True)
        return out, pinned_buffer

    # CPU path — no pinned-memory benefit; just allocate / copy.
    src = torch.from_numpy(masked)
    if out is None or tuple(out.shape) != expected_shape:
        out = src.to(device=device)
    else:
        out.copy_(src)
    return out, None


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
        bf_mask_bool (np.ndarray): (Ky, Kx) bool mask retained for live-acquisition reuse
        pinned_buffer (Tensor | None): pinned host buffer used for the initial H2D copy,
            returned so the solver can reuse it on subsequent dataset updates
    """
    _, _, Ky_dim, Kx_dim = dataset.shape

    kY_grid, kX_grid, bf_mask_bool = compute_bf_geometry(Ky_dim, Kx_dim, max_alpha, dk, wavelength)

    kY_centers_np = kY_grid[bf_mask_bool]
    kX_centers_np = kX_grid[bf_mask_bool]

    vbf_images, pinned_buffer = extract_vbf_stack(dataset, bf_mask_bool, device)

    kY_centers = torch.tensor(kY_centers_np, dtype=torch.float32, device=device)
    kX_centers = torch.tensor(kX_centers_np, dtype=torch.float32, device=device)
    kX_grid_t = torch.tensor(kX_grid, dtype=torch.float32, device=device)
    kY_grid_t = torch.tensor(kY_grid, dtype=torch.float32, device=device)
    bf_mask_t = torch.tensor(bf_mask_bool, dtype=torch.float32, device=device)

    print(f"Extracted {vbf_images.shape[0]} vBF images within the max alpha angle = {max_alpha} mrad.")
    return (
        vbf_images, kY_centers, kX_centers,
        kY_grid_t, kX_grid_t, bf_mask_t,
        bf_mask_bool, pinned_buffer,
    )


def init_grid(Ry: int, Rx: int, device: str) -> torch.Tensor:
    """Build the FFT shift grid for real-space pixel shifts. Returns shape (2, Ry, Rx)."""
    kpy, kpx = torch.meshgrid(
        torch.fft.fftfreq(Ry, dtype=torch.float32, device=device),
        torch.fft.fftfreq(Rx, dtype=torch.float32, device=device),
        indexing='ij',
    )
    return torch.stack([kpy, kpx], dim=0)


def build_image_fft(vbf_images: torch.Tensor) -> ImageFFT:
    """Pre-compute the full-stack FFT. Mode/orientation-independent."""
    img_fft = torch.fft.fft2(vbf_images, dim=(-2, -1))
    return ImageFFT(img_fft=img_fft)


def build_scan_freq_grids(
    Ry: int, Rx: int, scan_step_size: float, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build scan-frame frequency grids. Returns (qx_grid (1,1,Rx), qy_grid (1,Ry,1))."""
    qx_grid = torch.fft.fftfreq(Rx, d=scan_step_size, device=device).view(1, 1, Rx)
    qy_grid = torch.fft.fftfreq(Ry, d=scan_step_size, device=device).view(1, Ry, 1)
    return qx_grid, qy_grid


def build_tcbf_cache(
    kX_full: torch.Tensor,
    kY_full: torch.Tensor,
    order_keys: list,
    wavelength: float,
    chunk_size: int = 64,
) -> TCBFCache:
    """Pre-compute the orientation-dependent shift basis chunks.

    Holds basis only — no image or scan grid references.
    """
    Nb = kX_full.shape[0]
    chunks = []
    for i in range(0, Nb, chunk_size):
        end = min(i + chunk_size, Nb)
        b_dx, b_dy = generate_shift_basis(order_keys, kX_full[i:end], kY_full[i:end], wavelength)
        chunks.append({'b_dx': b_dx, 'b_dy': b_dy, 'start': i, 'end': end})

    return TCBFCache(chunks=chunks)


def build_acbf_geometry_cache(
    kX_full: torch.Tensor,
    kY_full: torch.Tensor,
    order_keys: list,
    max_alpha: float,
    wavelength: float,
    max_order: int,
    rolloff: float = 0,
    chunk_size: int = 64,
) -> ACBFGeometryCache:
    """Chunk detector k-coords and bundle the physics params needed to derive optics later.

    Always built (mode-independent). Optics are materialized separately via
    build_acbf_optics_cache when cache_mode='full'.
    """
    Nb = kX_full.shape[0]
    kX_full = kX_full.view(Nb, 1, 1)
    kY_full = kY_full.view(Nb, 1, 1)

    chunks = []
    for i in range(0, Nb, chunk_size):
        end = min(i + chunk_size, Nb)
        chunks.append({
            'kxt': kX_full[i:end].detach().clone(),
            'kyt': kY_full[i:end].detach().clone(),
            'start': i, 'end': end,
        })

    return ACBFGeometryCache(
        chunks=chunks,
        max_order=max_order,
        order_keys=order_keys,
        max_alpha=max_alpha,
        wavelength=wavelength,
        rolloff=rolloff,
    )


def build_acbf_optics_cache(
    geometry: ACBFGeometryCache,
    qx_grid: torch.Tensor,
    qy_grid: torch.Tensor,
) -> ACBFOpticsCache:
    """Materialize per-chunk apertures and aberration bases from geometry + scan grids.

    Heavy precompute used by 'full' cache mode. Reuses the physics params bundled
    on the geometry cache.
    """
    max_order = geometry.max_order
    order_keys = geometry.order_keys
    max_alpha = geometry.max_alpha
    wavelength = geometry.wavelength
    rolloff = geometry.rolloff

    chunks = []
    for chunk in geometry.chunks:
        kxt, kyt = chunk['kxt'], chunk['kyt']
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
            'start': chunk['start'], 'end': chunk['end'],
        })

    return ACBFOpticsCache(chunks=chunks)


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
        # Add a tiny epsilon before floor to absorb fp roundoff so a range that
        # mathematically divides evenly (e.g. exactly 11 slices) does not
        # silently lose its final slice when stored as 10.99999...
        steps = int(np.floor(abs(delta) / slice_thickness + 1e-9))
        offsets = torch.arange(steps + 1, device=device, dtype=torch.float32)
        return start + direction * slice_thickness * offsets

    raise ValueError(
        "Provide either n_layers with slice_thickness, or z_top, z_bottom, and slice_thickness."
    )
