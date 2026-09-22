"""tcBF reconstruction — pure math, no state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from fast_acbf.recon.cache import TCBFCache

if TYPE_CHECKING:
    from fast_acbf.data.imagefft import ImageFFT


def reconstruct_tcbf(
    provider: ImageFFT,
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

    Accumulates in Fourier space and inverse-transforms once at the end, rather than
    inverse-transforming every shifted virtual image before summing.  Both ``ifft2``
    and ``Re(.)`` are linear in the detector index b, so

        sum_b Re(ifft2(F_b * S_b))  ==  Re(ifft2(sum_b F_b * S_b))

    which turns N_BF inverse FFTs into exactly one.  Verified equal to the per-image
    form to float32 roundoff in both image and d(loss)/d(coeffs).

    Args:
        provider:  ImageFFT serving (chunk_size, Ry, Rx) complex64 chunks.
        qx_grid:   Scan-frame frequency grid, shape (1, 1, Rx).
        qy_grid:   Scan-frame frequency grid, shape (1, Ry, 1).
        cache:     TCBFCache built by pipeline.build_tcbf_cache (basis only).
        coeffs:    Flat scan-frame aberration coefficients, shape (num_coeffs,).
        device:    Target device string.

    Returns:
        Reconstructed tcBF image, shape (Ry, Rx), float32.
    """
    neg_two_pi_j = torch.tensor(-2.0j * torch.pi, dtype=torch.complex64, device=device)
    Ry_out = qy_grid.shape[-2]
    Rx_out = qx_grid.shape[-1]

    # Complex accumulator in Fourier space — one (Ry, Rx) buffer for the whole scan,
    # never a (chunk_size, Ry, Rx) stack of inverse-transformed images.
    spectrum = torch.zeros((Ry_out, Rx_out), dtype=torch.complex64, device=device)
    for chunk in cache.chunks:
        img_fft_chunk = provider.get_chunk(chunk['start'], chunk['end'])
        shift_dx = torch.einsum('k, kb -> b', coeffs, chunk['b_dx']).view(-1, 1, 1)
        shift_dy = torch.einsum('k, kb -> b', coeffs, chunk['b_dy']).view(-1, 1, 1)

        # The tcBF shift operator is a pure translation, so its phase ramp is separable:
        #     exp(-2pi i (dx_b qx + dy_b qy)) == exp(-2pi i dx_b qx) * exp(-2pi i dy_b qy)
        # Building the two 1D factors costs b*(Ry + Rx) transcendentals instead of the
        # b*Ry*Rx needed for the full 2D grid.  (acBF has no analogue — its transfer
        # T_b(q) is genuinely 2D and cannot be factorized this way.)
        phase_x = torch.exp(neg_two_pi_j * (shift_dx * qx_grid)).reshape(-1, Rx_out)
        phase_y = torch.exp(neg_two_pi_j * (shift_dy * qy_grid)).reshape(-1, Ry_out)

        # Apply both phase factors and contract over the detector axis in a single pass.
        # Written as one einsum so no (chunk_size, Ry, Rx) intermediate is materialized;
        # doing it as f * phase_x * phase_y then .sum(0) costs ~2x the memory traffic.
        spectrum += torch.einsum('byx, bx, by -> yx', img_fft_chunk, phase_x, phase_y)

    return torch.fft.ifft2(spectrum, dim=(-2, -1)).real
