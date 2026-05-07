"""Integration tests for BFSolver — init, shapes, dtypes, invariants, regression."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from fast_acbf.bf_solver import BFSolver, QualityMetrics


# ── Initialization ────────────────────────────────────────────────────────────

class TestBFSolverInit:

    def test_no_crash(self, synth_dataset, synth_params, device):
        p = synth_params
        BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 0.0},
            device=device,
        )

    def test_vbf_images_shape(self, solver_zero_ab, synth_params):
        p = synth_params
        Nb, Ry, Rx = solver_zero_ab.vbf_images.shape
        assert Nb > 0
        assert Ry == p["Ny"]
        assert Rx == p["Nx"]

    def test_bf_mask_shape(self, solver_zero_ab, synth_params):
        p = synth_params
        assert solver_zero_ab.bf_mask.shape == (p["Npix"], p["Npix"])

    def test_invalid_cache_mode_raises(self, synth_dataset, synth_params, device):
        p = synth_params
        with pytest.raises(ValueError, match="cache_mode"):
            BFSolver(
                dataset=synth_dataset,
                max_alpha=p["max_alpha"],
                scan_step_size=p["scan_step_size"],
                dk=p["dk"],
                wavelength=p["wavelength"],
                max_order=2,
                aberrations={"C10": 0.0},
                device=device,
                cache_mode="invalid_mode",
            )

    def test_invalid_upscale_method_raises(self, synth_dataset, synth_params, device):
        p = synth_params
        with pytest.raises(ValueError, match="upscale_method"):
            BFSolver(
                dataset=synth_dataset,
                max_alpha=p["max_alpha"],
                scan_step_size=p["scan_step_size"],
                dk=p["dk"],
                wavelength=p["wavelength"],
                max_order=2,
                aberrations={"C10": 0.0},
                device=device,
                upscale_method="bilinear",
            )


# ── get_chi_surface ───────────────────────────────────────────────────────────

class TestGetChiSurface:

    def test_shape(self, solver_zero_ab, synth_params):
        p = synth_params
        chi = solver_zero_ab.get_chi_surface()
        assert chi.shape == (p["Npix"], p["Npix"])

    def test_finite(self, solver_nonzero_ab):
        assert torch.all(torch.isfinite(solver_nonzero_ab.get_chi_surface()))

    def test_zero_aberrations_give_zero_chi(self, solver_zero_ab):
        chi = solver_zero_ab.get_chi_surface()
        torch.testing.assert_close(chi, torch.zeros_like(chi), atol=1e-6, rtol=0)


# ── get_probe ─────────────────────────────────────────────────────────────────

class TestGetProbe:

    def test_shape(self, solver_zero_ab, synth_params):
        p = synth_params
        assert solver_zero_ab.get_probe().shape == (p["Npix"], p["Npix"])

    def test_complex64_dtype(self, solver_zero_ab):
        probe = solver_zero_ab.get_probe()
        assert probe.is_complex() and probe.dtype == torch.complex64

    def test_normalized_zero_ab(self, solver_zero_ab):
        probe = solver_zero_ab.get_probe()
        norm = (probe.abs() ** 2).sum().item()
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_normalized_nonzero_ab(self, solver_nonzero_ab):
        probe = solver_nonzero_ab.get_probe()
        norm = (probe.abs() ** 2).sum().item()
        assert norm == pytest.approx(1.0, abs=1e-5)


# ── get_yx_shifts_px / get_yx_shifts_ang ─────────────────────────────────────

class TestGetYxShifts:

    def test_shifts_px_shape(self, solver_nonzero_ab):
        Nb = solver_nonzero_ab.vbf_images.shape[0]
        assert solver_nonzero_ab.get_yx_shifts_px().shape == (Nb, 2)

    def test_shifts_ang_shape(self, solver_nonzero_ab):
        Nb = solver_nonzero_ab.vbf_images.shape[0]
        assert solver_nonzero_ab.get_yx_shifts_ang().shape == (Nb, 2)

    def test_n_bf_pixels_positive(self, solver_zero_ab):
        assert solver_zero_ab.vbf_images.shape[0] > 0

    def test_zero_aberrations_zero_shifts(self, solver_zero_ab):
        shifts = solver_zero_ab.get_yx_shifts_px()
        torch.testing.assert_close(shifts, torch.zeros_like(shifts), atol=1e-5, rtol=0)

    def test_shifts_px_equals_ang_over_step(self, solver_nonzero_ab, synth_params):
        p = synth_params
        shifts_px = solver_nonzero_ab.get_yx_shifts_px()
        shifts_ang = solver_nonzero_ab.get_yx_shifts_ang()
        torch.testing.assert_close(shifts_px, shifts_ang / p["scan_step_size"], atol=1e-4, rtol=0)

    def test_shifts_finite(self, solver_nonzero_ab):
        assert torch.all(torch.isfinite(solver_nonzero_ab.get_yx_shifts_px()))


# ── get_tcBF ──────────────────────────────────────────────────────────────────

class TestGetTcBF:

    def test_shape(self, solver_zero_ab, synth_params):
        p = synth_params
        tcbf = solver_zero_ab.get_tcBF()
        assert tcbf.shape == (p["Ny"], p["Nx"])

    def test_float32_dtype(self, solver_zero_ab):
        assert solver_zero_ab.get_tcBF().dtype == torch.float32

    def test_all_finite(self, solver_nonzero_ab):
        assert torch.all(torch.isfinite(solver_nonzero_ab.get_tcBF()))

    def test_zero_aberrations_equals_vbf_sum(self, solver_zero_ab):
        tcbf = solver_zero_ab.get_tcBF()
        vbf_sum = solver_zero_ab.vbf_images.sum(dim=0).to(tcbf.device)
        torch.testing.assert_close(tcbf, vbf_sum, atol=1e-3, rtol=0)

    def test_idempotent(self, solver_zero_ab):
        r1 = solver_zero_ab.get_tcBF()
        r2 = solver_zero_ab.get_tcBF()
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)


# ── get_acBF ──────────────────────────────────────────────────────────────────

class TestGetAcBF:

    def test_shape(self, solver_zero_ab, synth_params):
        p = synth_params
        assert solver_zero_ab.get_acBF().shape == (p["Ny"], p["Nx"])

    def test_float32_dtype(self, solver_zero_ab):
        assert solver_zero_ab.get_acBF().dtype == torch.float32

    def test_all_finite(self, solver_nonzero_ab):
        assert torch.all(torch.isfinite(solver_nonzero_ab.get_acBF()))

    def test_idempotent(self, solver_zero_ab):
        r1 = solver_zero_ab.get_acBF()
        r2 = solver_zero_ab.get_acBF()
        torch.testing.assert_close(r1, r2, atol=0, rtol=0)

    def test_complex_inversion_shape(self, solver_nonzero_ab, synth_params):
        p = synth_params
        acbf_ci = solver_nonzero_ab.get_acBF(acbf_algorithm='complex_inversion')
        assert acbf_ci.shape == (p["Ny"], p["Nx"])

    def test_complex_inversion_finite(self, solver_nonzero_ab):
        assert torch.all(torch.isfinite(
            solver_nonzero_ab.get_acBF(acbf_algorithm='complex_inversion')
        ))


# ── get_defocus_stack ─────────────────────────────────────────────────────────

class TestGetDefocusStack:

    def test_n_layers_shape(self, solver_nonzero_ab, synth_params):
        p = synth_params
        stack = solver_nonzero_ab.get_defocus_stack(n_layers=5, slice_thickness=10.0)
        assert stack.shape == (5, p["Ny"], p["Nx"])

    def test_range_mode_shape(self, solver_nonzero_ab, synth_params):
        p = synth_params
        # floor(|40-0|/10) + 1 = 5 layers
        stack = solver_nonzero_ab.get_defocus_stack(
            z_top=0.0, z_bottom=40.0, slice_thickness=10.0
        )
        assert stack.shape == (5, p["Ny"], p["Nx"])

    def test_all_finite(self, solver_nonzero_ab):
        stack = solver_nonzero_ab.get_defocus_stack(n_layers=3, slice_thickness=10.0)
        assert torch.all(torch.isfinite(stack))

    def test_missing_slice_thickness_raises(self, solver_nonzero_ab):
        with pytest.raises(ValueError, match="slice_thickness"):
            solver_nonzero_ab.get_defocus_stack(n_layers=3)

    def test_c10_restored_after_stack(self, solver_nonzero_ab):
        original = solver_nonzero_ab.ab_state.get_physical('C_1_0')
        solver_nonzero_ab.get_defocus_stack(n_layers=3, slice_thickness=10.0)
        restored = solver_nonzero_ab.ab_state.get_physical('C_1_0')
        assert restored == pytest.approx(original, abs=1e-4)


# ── get_aberrations_dict ──────────────────────────────────────────────────────

class TestGetAberrationsDict:

    def test_flat_layout_contains_c10_key(self, solver_nonzero_ab):
        ab_dict = solver_nonzero_ab.get_aberrations_dict(layout='flat')
        assert 'C10' in ab_dict

    def test_c10_value_matches_constructor(self, synth_dataset, synth_params, device):
        p = synth_params
        solver = BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 123.0},
            device=device,
        )
        ab_dict = solver.get_aberrations_dict(layout='flat')
        assert ab_dict['C10'] == pytest.approx(123.0, abs=0.1)

    def test_nested_layout_has_tuple_keys(self, solver_nonzero_ab):
        ab_dict = solver_nonzero_ab.get_aberrations_dict()
        for key in ab_dict.keys():
            assert isinstance(key, tuple) and len(key) == 2


# ── QualityMetrics ────────────────────────────────────────────────────────────

class TestQualityMetrics:

    def test_2d_returns_scalar(self):
        score = QualityMetrics.evaluate(torch.randn(16, 16))
        assert isinstance(score, torch.Tensor) and score.dim() == 0

    def test_3d_returns_1d_tensor(self):
        scores = QualityMetrics.evaluate(torch.randn(5, 16, 16))
        assert scores.shape == (5,)

    def test_laplacian_nonnegative(self):
        assert QualityMetrics.evaluate(torch.randn(16, 16), metric='laplacian').item() >= 0.0

    def test_sobel_nonnegative(self):
        assert QualityMetrics.evaluate(torch.randn(16, 16), metric='sobel').item() >= 0.0

    def test_normalized_std_nonnegative(self):
        img = torch.randn(16, 16) + 5.0
        assert QualityMetrics.evaluate(img, metric='normalized_std').item() >= 0.0

    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            QualityMetrics.evaluate(torch.randn(8, 8), metric='nonexistent')

    def test_4d_input_raises(self):
        with pytest.raises(ValueError):
            QualityMetrics.evaluate(torch.randn(2, 2, 8, 8))

    def test_sharp_scores_higher_than_blurred(self):
        import torch.nn.functional as F
        sharp = torch.randn(64, 64)
        blurred = F.avg_pool2d(
            sharp.unsqueeze(0).unsqueeze(0), kernel_size=9, stride=1, padding=4
        ).squeeze()
        score_sharp = QualityMetrics.evaluate(sharp, metric='laplacian').item()
        score_blurred = QualityMetrics.evaluate(blurred, metric='laplacian').item()
        assert score_sharp > score_blurred


# ── Regression tests ──────────────────────────────────────────────────────────

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
DEFAULT_REAL_ZARR = (
    "/home/cl2696/scratch/"
    "test_1_static_cell_30_30_5_80kv_coherent_probe_ca_25_cl_139_C10_0_t_1_Nscans_64_64_dp_200/cbed.zarr"
)
REAL_ZARR = os.path.expanduser(
    os.environ.get("FAST_ACBF_REGRESSION_ZARR", DEFAULT_REAL_ZARR)
)
REAL_MAX_ALPHA = 25.0
REAL_COLLECTION_ANGLE = 139
REAL_WAVELENGTH = 0.04176
REAL_SCAN_STEP = 0.2
REAL_NPIX = 200
REAL_DK = (REAL_COLLECTION_ANGLE / 1000.0) / (REAL_NPIX / 2 * REAL_WAVELENGTH)
REAL_ABERRATIONS = {'C10': 2.50}
REAL_COORD_TRANSFORM = {}
REAL_MAX_ORDER=2


@pytest.fixture(scope="session")
def real_solver(device):
    pytest.importorskip("zarr")
    if not os.path.exists(REAL_ZARR):
        pytest.skip(f"Real zarr data not found: {REAL_ZARR}")
    import zarr
    z = zarr.open(REAL_ZARR, mode='r')
    dataset = np.array(z[0]).reshape(64, 64, REAL_NPIX, REAL_NPIX)
    return BFSolver(
        dataset=dataset,
        max_alpha=REAL_MAX_ALPHA,
        scan_step_size=REAL_SCAN_STEP,
        dk=REAL_DK,
        wavelength=REAL_WAVELENGTH,
        max_order=REAL_MAX_ORDER,
        aberrations=REAL_ABERRATIONS,
        coord_transform=REAL_COORD_TRANSFORM,
        device=device,
    )


@pytest.mark.regression
def test_regression_tcbf(real_solver):
    fixture = os.path.join(FIXTURE_DIR, "tcbf.npy")
    if not os.path.exists(fixture):
        pytest.skip(f"Fixture missing: {fixture}. Run tests/generate_fixtures.py first.")
    expected = np.load(fixture)
    actual = real_solver.get_tcBF().detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.regression
def test_regression_acbf(real_solver):
    fixture = os.path.join(FIXTURE_DIR, "acbf.npy")
    if not os.path.exists(fixture):
        pytest.skip(f"Fixture missing: {fixture}. Run tests/generate_fixtures.py first.")
    expected = np.load(fixture)
    actual = real_solver.get_acBF().detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.regression
def test_regression_defocus_stack(real_solver):
    fixture = os.path.join(FIXTURE_DIR, "defocus_stack.npy")
    if not os.path.exists(fixture):
        pytest.skip(f"Fixture missing: {fixture}. Run tests/generate_fixtures.py first.")
    expected = np.load(fixture)
    stack = real_solver.get_defocus_stack(n_layers=5, slice_thickness=10.0)
    actual = stack.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)
