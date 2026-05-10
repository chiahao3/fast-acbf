"""tcBF reconstruction — pure math, no state."""

from __future__ import annotations

import torch

from fast_acbf.pipeline import ImageFFT, TCBFCache


def reconstruct_tcbf(
    image_fft: ImageFFT,
    qx_grid: torch.Tensor,
    qy_grid: torch.Tensor,
    cache: TCBFCache,
    coeffs: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Ultra-lean AD forward pass for tcBF.

    Calculates exact analytical shifts via Einstein summation over the pre-built
    shift basis stored in the cache.

    Args:
        image_fft: ImageFFT carrying the pre-computed BF stack FFT.
        qx_grid:   Scan-frame frequency grid, shape (1, 1, Rx).
        qy_grid:   Scan-frame frequency grid, shape (1, Ry, 1).
        cache:     TCBFCache built by pipeline.build_tcbf_cache (basis only).
        coeffs:    Flat scan-frame aberration coefficients, shape (num_coeffs,).
        device:    Target device string.

    Returns:
        Reconstructed tcBF image, shape (Ry, Rx), float32.
    """
    neg_two_pi_j = torch.tensor(-2.0j * torch.pi, dtype=torch.complex64, device=device)
    out_shape = image_fft.img_fft.shape[-2:]

    tcBF_total = torch.zeros(out_shape, dtype=torch.float32, device=device)
    for chunk in cache.chunks:
        img_fft_chunk = image_fft.fft_chunk(chunk['start'], chunk['end'])
        shift_dx = torch.einsum('k, kb -> b', coeffs, chunk['b_dx']).view(-1, 1, 1)
        shift_dy = torch.einsum('k, kb -> b', coeffs, chunk['b_dy']).view(-1, 1, 1)
        ramp = shift_dx * qx_grid + shift_dy * qy_grid
        shift_op = torch.exp(neg_two_pi_j * ramp)
        tcBF_total += torch.sum(torch.fft.ifft2(img_fft_chunk * shift_op, dim=(-2, -1)).real, dim=0)

    return tcBF_total
