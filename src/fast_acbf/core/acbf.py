"""acBF reconstruction — pure math, no state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fast_acbf.core.functional import generate_aberration_basis, make_soft_aperture_torch
from fast_acbf.recon.cache import ACBFGeometryCache, ACBFOpticsCache

if TYPE_CHECKING:
    from fast_acbf.data.imagefft import ImageFFT


def compute_transfer(
    geom_chunk: dict,
    optics_chunk: dict | None,
    coeffs: torch.Tensor,
    qx_grid: torch.Tensor,
    qy_grid: torch.Tensor,
    geometry: ACBFGeometryCache,
    device: str,
) -> torch.Tensor:
    """
    Compute the detector-wise complex transfer for one chunk.

    Two paths driven by the presence of an optics chunk:
        Fast path: bases and apertures pre-cached on optics_chunk.
        Lazy path: optics_chunk is None — recompute from geom_chunk + scan grids + physics.

    Args:
        geom_chunk:   One ACBFGeometryCache chunk dict ({kxt, kyt, start, end}).
        optics_chunk: Matching ACBFOpticsCache chunk dict, or None for lazy path.
        coeffs:       Flat scan-frame aberration coefficients, shape (num_coeffs,).
        qx_grid:      Scan-frame frequency grid, shape (1, 1, Rx).
        qy_grid:      Scan-frame frequency grid, shape (1, Ry, 1).
        geometry:     ACBFGeometryCache — physics params used by the lazy path.
        device:       Target device string.

    Returns:
        Complex transfer T, shape (chunk_size, Ny, Nx).
    """
    j1 = torch.tensor(1.0j, dtype=torch.complex64, device=device)

    if optics_chunk is not None:
        b_tr  = optics_chunk['b_tr']
        b_t   = optics_chunk['b_t']
        b_mt  = optics_chunk['b_mt']
        ap_t  = optics_chunk['ap_t']
        ap_mt = optics_chunk['ap_mt']
    else:
        kxt = geom_chunk['kxt'].to(device)
        kyt = geom_chunk['kyt'].to(device)
        kx_t,  ky_t  = qx_grid + kxt, qy_grid + kyt
        kx_mt, ky_mt = qx_grid - kxt, qy_grid - kyt
        ap_t  = make_soft_aperture_torch(
            torch.sqrt(kx_t**2  + ky_t**2)  * geometry.wavelength,
            geometry.max_alpha, geometry.rolloff)
        ap_mt = make_soft_aperture_torch(
            torch.sqrt(kx_mt**2 + ky_mt**2) * geometry.wavelength,
            geometry.max_alpha, geometry.rolloff)
        b_tr = generate_aberration_basis(
            geometry.max_order, geometry.order_keys, kxt,    kyt,    geometry.wavelength)
        b_t  = generate_aberration_basis(
            geometry.max_order, geometry.order_keys, kx_t,   ky_t,   geometry.wavelength)
        b_mt = generate_aberration_basis(
            geometry.max_order, geometry.order_keys, -kx_mt, -ky_mt, geometry.wavelength)

    chi_tr_az = torch.einsum('k, kbxy -> bxy', coeffs, b_tr)
    chi_t     = torch.einsum('k, kbxy -> bxy', coeffs, b_t)
    chi_mt    = torch.einsum('k, kbxy -> bxy', coeffs, b_mt)

    term_mt = ap_mt * torch.exp(-j1 * (chi_tr_az - chi_mt))
    term_t  = ap_t  * torch.exp( j1 * (chi_tr_az - chi_t))
    D = term_mt - term_t

    # T = -i * D. Keeps the transfer aligned with the acBF phasor convention.
    return (-j1) * D


def _iter_chunks(geometry: ACBFGeometryCache, optics: ACBFOpticsCache | None):
    """Yield (geom_chunk, optics_chunk_or_None) pairs."""
    if optics is None:
        for g in geometry.chunks:
            yield g, None
        return
    if len(geometry.chunks) != len(optics.chunks):
        raise ValueError(
            f"geometry/optics chunk count mismatch: "
            f"{len(geometry.chunks)} vs {len(optics.chunks)}."
        )
    for g, o in zip(geometry.chunks, optics.chunks):
        if g['start'] != o['start'] or g['end'] != o['end']:
            raise ValueError(
                f"geometry/optics chunk bounds mismatch: "
                f"({g['start']},{g['end']}) vs ({o['start']},{o['end']})."
            )
        yield g, o


def reconstruct_acbf(
    provider: ImageFFT,
    qx_grid: torch.Tensor,
    qy_grid: torch.Tensor,
    geometry: ACBFGeometryCache,
    optics: ACBFOpticsCache | None,
    coeffs: torch.Tensor,
    eps: float,
    device: str,
) -> torch.Tensor:
    """
    Phase-only acBF reconstruction.

    Aligns detector contributions by their phase before summation.

    Args:
        provider:  ImageFFT serving (chunk_size, Ry, Rx) complex64 chunks.
        qx_grid:   Scan-frame frequency grid, shape (1, 1, Rx).
        qy_grid:   Scan-frame frequency grid, shape (1, Ry, 1).
        geometry:  ACBFGeometryCache built by pipeline.build_acbf_geometry_cache.
        optics:    ACBFOpticsCache (full mode) or None (lazy mode).
        coeffs:    Flat scan-frame aberration coefficients, shape (num_coeffs,).
        eps:       Small constant for phase normalization stability.
        device:    Target device string.

    Returns:
        Reconstructed acBF image, shape (Ry, Rx), float32.
    """
    Ry_out = qy_grid.shape[-2]
    Rx_out = qx_grid.shape[-1]
    acBF_total = torch.zeros((Ry_out, Rx_out), dtype=torch.float32, device=device)
    for geom_chunk, optics_chunk in _iter_chunks(geometry, optics):
        img_fft_chunk = provider.get_chunk(geom_chunk['start'], geom_chunk['end'])
        transfer = compute_transfer(geom_chunk, optics_chunk, coeffs, qx_grid, qy_grid, geometry, device)
        phasor = transfer / (transfer.abs() + eps)
        acBF_total += torch.sum(torch.fft.ifft2(img_fft_chunk * phasor, dim=(-2, -1)).real, dim=0)
    return acBF_total


def reconstruct_acbf_complex_inversion(
    provider: ImageFFT,
    qx_grid: torch.Tensor,
    qy_grid: torch.Tensor,
    geometry: ACBFGeometryCache,
    optics: ACBFOpticsCache | None,
    coeffs: torch.Tensor,
    device: str,
    regularization: float = 1e-3,
    support_threshold: float = 1e-6,
    return_diagnostics: bool = False,
):
    """
    Complex-inversion acBF reconstruction via regularized transfer inversion.

    Solves a regularized matched-filter inversion of the detector-wise complex transfer:

        V_hat(q) = M(q) / (S(q) + lambda * S_ref)

    where:
        M(q) = sum_b T_b(q) * I_b(q)
        S(q) = sum_b |T_b(q)|^2

    Args:
        provider:           ImageFFT serving (chunk_size, Ry, Rx) complex64 chunks.
        qx_grid:            Scan-frame frequency grid, shape (1, 1, Rx).
        qy_grid:            Scan-frame frequency grid, shape (1, Ry, 1).
        geometry:           ACBFGeometryCache.
        optics:             ACBFOpticsCache (full mode) or None (lazy mode).
        coeffs:             Flat scan-frame aberration coefficients.
        device:             Target device string.
        regularization:     Non-negative regularization weight lambda.
        support_threshold:  Fraction of median transfer power below which Fourier
                            components are zeroed.
        return_diagnostics: If True, return full diagnostic dict instead of just image.

    Returns:
        Reconstructed image (Ry, Rx) float32, or dict if return_diagnostics=True.
    """
    if regularization < 0:
        raise ValueError(f"regularization must be non-negative, got {regularization}.")
    if support_threshold < 0:
        raise ValueError(f"support_threshold must be non-negative, got {support_threshold}.")

    Ry_out = qy_grid.shape[-2]
    Rx_out = qx_grid.shape[-1]
    out_shape = (Ry_out, Rx_out)
    numerator = torch.zeros(out_shape, dtype=torch.complex64, device=device)
    transfer_power = torch.zeros(out_shape, dtype=torch.float32, device=device)

    for geom_chunk, optics_chunk in _iter_chunks(geometry, optics):
        img_fft_chunk = provider.get_chunk(geom_chunk['start'], geom_chunk['end'])
        transfer = compute_transfer(geom_chunk, optics_chunk, coeffs, qx_grid, qy_grid, geometry, device)
        numerator.add_(torch.sum(transfer * img_fft_chunk, dim=0))
        transfer_power.add_(torch.sum(transfer.abs().square(), dim=0))

    positive_power = transfer_power[transfer_power > 0]
    if positive_power.numel() == 0:
        transfer_reference = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        transfer_reference = positive_power.median()

    support = transfer_power > (support_threshold * transfer_reference)
    denom = transfer_power + (regularization * transfer_reference)
    fourier_estimate = torch.where(
        support,
        numerator / denom.to(torch.complex64),
        torch.zeros_like(numerator),
    )

    complex_image = torch.fft.ifft2(fourier_estimate, dim=(-2, -1))
    reconstructed = complex_image.real

    if not return_diagnostics:
        return reconstructed

    return {
        'image': reconstructed,
        'complex_image': complex_image,
        'real_channel': complex_image.real,
        'imag_channel': complex_image.imag,
        'fourier_estimate': fourier_estimate,
        'transfer_power': transfer_power,
        'support_mask': support,
        'support_reference': transfer_reference,
    }
