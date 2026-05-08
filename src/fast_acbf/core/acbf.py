"""acBF reconstruction — pure math, no state."""

from __future__ import annotations

import torch

from fast_acbf.core.functional import generate_aberration_basis, make_soft_aperture_torch
from fast_acbf.pipeline import ACBFCache


def compute_transfer(
    chunk: dict,
    coeffs: torch.Tensor,
    cache: ACBFCache,
    device: str,
) -> torch.Tensor:
    """
    Compute the detector-wise complex transfer for one cache chunk.

    Two paths depending on chunk content (self-describing structure):
        Fast path ('full'): bases are pre-cached in the chunk dict.
        Lazy path ('lazy'): bases are absent and recomputed from stored kxt/kyt,
                            using physics params carried on the ACBFCache.

    Args:
        chunk:   Cache chunk dict (one element of cache.chunks).
        coeffs:  Flat scan-frame aberration coefficients, shape (num_coeffs,).
        cache:   ACBFCache — carries rolloff + physics params for the lazy path.
        device:  Target device string.

    Returns:
        Complex transfer T, shape (chunk_size, Ny, Nx).
    """
    j1 = torch.tensor(1.0j, dtype=torch.complex64, device=device)

    if 'b_tr' in chunk:
        b_tr  = chunk['b_tr']
        b_t   = chunk['b_t']
        b_mt  = chunk['b_mt']
        ap_t  = chunk['ap_t']
        ap_mt = chunk['ap_mt']
    else:
        kxt = chunk['kxt'].to(device)
        kyt = chunk['kyt'].to(device)
        kx_base = cache.qx_grid
        ky_base = cache.qy_grid
        kx_t,  ky_t  = kx_base + kxt, ky_base + kyt
        kx_mt, ky_mt = kx_base - kxt, ky_base - kyt
        ap_t  = make_soft_aperture_torch(
            torch.sqrt(kx_t**2  + ky_t**2)  * cache.wavelength, cache.max_alpha, cache.rolloff)
        ap_mt = make_soft_aperture_torch(
            torch.sqrt(kx_mt**2 + ky_mt**2) * cache.wavelength, cache.max_alpha, cache.rolloff)
        b_tr = generate_aberration_basis(
            cache.max_order, cache.order_keys, kxt,    kyt,    cache.wavelength)
        b_t  = generate_aberration_basis(
            cache.max_order, cache.order_keys, kx_t,   ky_t,   cache.wavelength)
        b_mt = generate_aberration_basis(
            cache.max_order, cache.order_keys, -kx_mt, -ky_mt, cache.wavelength)

    chi_tr_az = torch.einsum('k, kbxy -> bxy', coeffs, b_tr)
    chi_t     = torch.einsum('k, kbxy -> bxy', coeffs, b_t)
    chi_mt    = torch.einsum('k, kbxy -> bxy', coeffs, b_mt)

    term_mt = ap_mt * torch.exp(-j1 * (chi_tr_az - chi_mt))
    term_t  = ap_t  * torch.exp( j1 * (chi_tr_az - chi_t))
    D = term_mt - term_t

    # T = -i * D. Keeps the transfer aligned with the acBF phasor convention.
    return (-j1) * D


def reconstruct_acbf(
    cache: ACBFCache,
    coeffs: torch.Tensor,
    eps: float,
    device: str,
) -> torch.Tensor:
    """
    Phase-only acBF reconstruction.

    Aligns detector contributions by their phase before summation.

    Args:
        cache:   ACBFCache built by pipeline.build_acbf_cache.
        coeffs:  Flat scan-frame aberration coefficients, shape (num_coeffs,).
        eps:     Small constant for phase normalization stability.
        device:  Target device string.

    Returns:
        Reconstructed acBF image, shape (Ry, Rx), float32.
    """
    acBF_total = torch.zeros(cache.out_shape, dtype=torch.float32, device=device)
    for chunk in cache.chunks:
        img_fft_chunk = cache.img_fft[chunk['start']:chunk['end']]
        transfer = compute_transfer(chunk, coeffs, cache, device)
        phasor = transfer / (transfer.abs() + eps)
        acBF_total += torch.sum(torch.fft.ifft2(img_fft_chunk * phasor, dim=(-2, -1)).real, dim=0)
    return acBF_total


def reconstruct_acbf_complex_inversion(
    cache: ACBFCache,
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
        cache:              ACBFCache.
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

    numerator = torch.zeros(cache.out_shape, dtype=torch.complex64, device=device)
    transfer_power = torch.zeros(cache.out_shape, dtype=torch.float32, device=device)

    for chunk in cache.chunks:
        img_fft_chunk = cache.img_fft[chunk['start']:chunk['end']]
        transfer = compute_transfer(chunk, coeffs, cache, device)
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
