"""tcBF reconstruction — pure math, no state."""

from __future__ import annotations

import torch

from fast_acbf.pipeline import TCBFCache


def reconstruct_tcbf(cache: TCBFCache, coeffs: torch.Tensor, device: str) -> torch.Tensor:
    """
    Ultra-lean AD forward pass for tcBF.

    Calculates exact analytical shifts via Einstein summation over the pre-built
    shift basis stored in the cache.

    Args:
        cache:   TCBFCache built by pipeline.build_tcbf_cache.
        coeffs:  Flat scan-frame aberration coefficients, shape (num_coeffs,).
        device:  Target device string.

    Returns:
        Reconstructed tcBF image, shape (Ry, Rx), float32.
    """
    neg_two_pi_j = torch.tensor(-2.0j * torch.pi, dtype=torch.complex64, device=device)

    tcBF_total = torch.zeros(cache.out_shape, dtype=torch.float32, device=device)
    for chunk in cache.chunks:
        img_fft_chunk = cache.img_fft[chunk['start']:chunk['end']]
        shift_dx = torch.einsum('k, kb -> b', coeffs, chunk['b_dx']).view(-1, 1, 1)
        shift_dy = torch.einsum('k, kb -> b', coeffs, chunk['b_dy']).view(-1, 1, 1)
        ramp = shift_dx * cache.qx_grid + shift_dy * cache.qy_grid
        shift_op = torch.exp(neg_two_pi_j * ramp)
        tcBF_total += torch.sum(torch.fft.ifft2(img_fft_chunk * shift_op, dim=(-2, -1)).real, dim=0)

    return tcBF_total
