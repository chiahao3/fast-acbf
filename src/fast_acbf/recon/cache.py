"""Reconstruction cache dataclasses and their builders for tcBF and acBF."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fast_acbf.core.functional import (
    generate_aberration_basis,
    generate_shift_basis,
    make_soft_aperture_torch,
)


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


def build_tcbf_cache(
    kX_full: torch.Tensor,
    kY_full: torch.Tensor,
    order_keys: list,
    wavelength: float,
    chunk_size: int = 64,
) -> TCBFCache:
    """Pre-compute the orientation-dependent shift basis chunks."""
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
    """Chunk detector k-coords and bundle physics params needed to derive optics later.

    Always built (mode-independent). Optics are materialized separately via
    build_acbf_optics_cache when basis_mode='precompute'.
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

    Heavy precompute used by 'full' cache mode.
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
