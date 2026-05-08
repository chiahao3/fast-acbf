"""Unit tests for fast_acbf.bf_solver module-level utility functions."""
from __future__ import annotations

import pytest
import torch

from fast_acbf.core.aberrations import AberrationState
from fast_acbf.core.functional import (
    generate_aberration_basis,
    generate_shift_basis,
    imshift_with_batch,
    make_soft_aperture_torch,
)


# ── make_soft_aperture_torch ──────────────────────────────────────────────────

class TestMakeSoftApertureTorch:

    def test_values_in_range(self):
        alpha = torch.linspace(0, 0.05, 100)
        ap = make_soft_aperture_torch(alpha, max_alpha_mrad=25.0, rolloff_mrad=2.0)
        assert (ap >= 0.0).all() and (ap <= 1.0).all()

    def test_center_is_one(self):
        ap = make_soft_aperture_torch(torch.tensor([[0.0]]), 25.0, 2.0)
        assert ap.item() == pytest.approx(1.0)

    def test_far_corner_is_zero(self):
        ap = make_soft_aperture_torch(torch.tensor([[0.5]]), 25.0, 2.0)
        assert ap.item() == pytest.approx(0.0)

    def test_hard_aperture_at_cutoff_is_one(self):
        ap = make_soft_aperture_torch(torch.tensor([[0.025]]), 25.0, rolloff_mrad=0.0)
        assert ap.item() == pytest.approx(1.0)

    def test_hard_aperture_above_cutoff_is_zero(self):
        ap = make_soft_aperture_torch(torch.tensor([[0.02501]]), 25.0, rolloff_mrad=0.0)
        assert ap.item() == pytest.approx(0.0)

    def test_output_shape_preserved_1d(self):
        alpha = torch.rand(16) * 0.02
        assert make_soft_aperture_torch(alpha, 25.0, 2.0).shape == alpha.shape

    def test_output_shape_preserved_2d(self):
        alpha = torch.rand(16, 16) * 0.02
        assert make_soft_aperture_torch(alpha, 25.0, 2.0).shape == alpha.shape

    def test_output_dtype_float32(self):
        ap = make_soft_aperture_torch(torch.zeros(1), 25.0, 2.0)
        assert ap.dtype == torch.float32

    def test_2d_grid_center_is_one(self):
        import numpy as np
        Npix = 32
        wavelength = 0.04176
        max_alpha_mrad = 25.0
        collection_angle_mrad = 50.0   # detector semi-angle, larger than convergence angle
        dk = (collection_angle_mrad / 1000.0) / (Npix / 2 * wavelength)
        ky_np = np.fft.fftshift(np.fft.fftfreq(Npix, d=(1.0 / dk / Npix)))
        kX, kY = np.meshgrid(ky_np, ky_np, indexing='xy')
        kR = np.sqrt(kX**2 + kY**2)
        alpha = torch.tensor(kR * wavelength, dtype=torch.float32)
        ap = make_soft_aperture_torch(alpha, max_alpha_mrad, 2.0)
        assert ap[Npix // 2, Npix // 2].item() == pytest.approx(1.0)


# ── imshift_with_batch ────────────────────────────────────────────────────────

@pytest.fixture
def shift_grid_16():
    Ny, Nx = 16, 16
    ky = torch.fft.fftfreq(Ny, dtype=torch.float32)
    kx = torch.fft.fftfreq(Nx, dtype=torch.float32)
    gy, gx = torch.meshgrid(ky, kx, indexing='ij')
    return torch.stack([gy, gx], dim=0)


class TestImshiftWithBatch:

    def test_zero_shift_real_identity(self, shift_grid_16):
        imgs = torch.randn(5, 16, 16)
        shifts = torch.zeros(5, 2)
        out = imshift_with_batch(imgs, shifts, shift_grid_16)
        torch.testing.assert_close(out.real, imgs, atol=1e-5, rtol=0)

    def test_zero_shift_complex_identity(self, shift_grid_16):
        imgs = torch.randn(3, 16, 16, dtype=torch.complex64)
        shifts = torch.zeros(3, 2)
        out = imshift_with_batch(imgs, shifts, shift_grid_16)
        torch.testing.assert_close(out, imgs, atol=1e-5, rtol=0)

    def test_output_shape_preserved(self, shift_grid_16):
        imgs = torch.randn(7, 16, 16)
        shifts = torch.zeros(7, 2)
        assert imshift_with_batch(imgs, shifts, shift_grid_16).shape == imgs.shape

    def test_batch_mismatch_raises(self, shift_grid_16):
        imgs = torch.randn(3, 16, 16)
        shifts = torch.zeros(5, 2)
        with pytest.raises(AssertionError):
            imshift_with_batch(imgs, shifts, shift_grid_16)

    def test_spatial_mismatch_raises(self, shift_grid_16):
        imgs = torch.randn(3, 8, 8)
        shifts = torch.zeros(3, 2)
        with pytest.raises(AssertionError):
            imshift_with_batch(imgs, shifts, shift_grid_16)

    def test_chunked_equals_full_batch(self, shift_grid_16):
        imgs = torch.randn(8, 16, 16)
        shifts = torch.randn(8, 2) * 0.5
        out_full = imshift_with_batch(imgs, shifts, shift_grid_16, batch_size=None)
        out_chunked = imshift_with_batch(imgs, shifts, shift_grid_16, batch_size=3)
        torch.testing.assert_close(out_full, out_chunked, atol=1e-5, rtol=0)


# ── generate_shift_basis ──────────────────────────────────────────────────────

@pytest.fixture
def order_keys_max2():
    return AberrationState({(1, 0): 0.0}, max_order=2, device='cpu').order_keys


class TestGenerateShiftBasis:

    def test_output_shape(self, order_keys_max2):
        Nb = 50
        kX = torch.rand(Nb) * 0.01
        kY = torch.rand(Nb) * 0.01
        b_dx, b_dy = generate_shift_basis(order_keys_max2, kX, kY, 0.04176)
        num_coeffs = len(AberrationState({(1, 0): 0.0}, max_order=2).get_flat_coeffs())
        assert b_dx.shape == (num_coeffs, Nb)
        assert b_dy.shape == (num_coeffs, Nb)

    def test_zero_k_gives_zero_basis(self, order_keys_max2):
        kX = torch.zeros(5)
        kY = torch.zeros(5)
        b_dx, b_dy = generate_shift_basis(order_keys_max2, kX, kY, 0.04176)
        torch.testing.assert_close(b_dx, torch.zeros_like(b_dx), atol=1e-6, rtol=0)
        torch.testing.assert_close(b_dy, torch.zeros_like(b_dy), atol=1e-6, rtol=0)

    def test_no_nan_inf(self, order_keys_max2):
        kX = torch.rand(100) * 0.05
        kY = torch.rand(100) * 0.05
        b_dx, b_dy = generate_shift_basis(order_keys_max2, kX, kY, 0.04176)
        assert torch.all(torch.isfinite(b_dx)) and torch.all(torch.isfinite(b_dy))

    def test_num_rows_matches_flat_coeffs(self, order_keys_max2):
        state = AberrationState({(1, 0): 0.0}, max_order=2)
        kX = torch.rand(10) * 0.01
        kY = torch.rand(10) * 0.01
        b_dx, _ = generate_shift_basis(order_keys_max2, kX, kY, 0.04176)
        assert b_dx.shape[0] == len(state.get_flat_coeffs())


# ── generate_aberration_basis ─────────────────────────────────────────────────

class TestGenerateAberrationBasis:

    def test_2d_output_shape(self, order_keys_max2):
        Ny, Nx = 8, 8
        kX = torch.rand(Ny, Nx) * 0.01
        kY = torch.rand(Ny, Nx) * 0.01
        basis = generate_aberration_basis(2, order_keys_max2, kX, kY, 0.04176)
        num_coeffs = len(AberrationState({(1, 0): 0.0}, max_order=2).get_flat_coeffs())
        assert basis.shape == (num_coeffs, Ny, Nx)

    def test_no_nan_inf(self, order_keys_max2):
        kX = torch.rand(16, 16) * 0.03
        kY = torch.rand(16, 16) * 0.03
        basis = generate_aberration_basis(2, order_keys_max2, kX, kY, 0.04176)
        assert torch.all(torch.isfinite(basis))

    def test_num_coeffs_matches_flat(self, order_keys_max2):
        state = AberrationState({(1, 0): 0.0}, max_order=2)
        kX = torch.rand(4, 4) * 0.01
        kY = torch.rand(4, 4) * 0.01
        basis = generate_aberration_basis(2, order_keys_max2, kX, kY, 0.04176)
        assert basis.shape[0] == len(state.get_flat_coeffs())
