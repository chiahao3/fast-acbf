"""Tests for BFSolver live-acquisition setters and apply_metadata dispatcher."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fast_acbf import BFSolver

from conftest import (  # type: ignore[import-not-found]
    SYNTH_DK,
    SYNTH_MAX_ALPHA,
    SYNTH_NX,
    SYNTH_NY,
    SYNTH_SCAN_STEP,
    SYNTH_WAVELENGTH,
    _make_synth_dataset,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_solver(
    dataset=None, *, max_alpha=None, scan_step_size=None, dk=None,
    wavelength=None, aberrations=None, device="cpu",
):
    return BFSolver(
        dataset=dataset if dataset is not None else _make_synth_dataset(seed=42),
        max_alpha=SYNTH_MAX_ALPHA if max_alpha is None else max_alpha,
        scan_step_size=SYNTH_SCAN_STEP if scan_step_size is None else scan_step_size,
        dk=SYNTH_DK if dk is None else dk,
        wavelength=SYNTH_WAVELENGTH if wavelength is None else wavelength,
        max_order=2,
        aberrations=aberrations or {"C10": 0.0},
        device=device,
    )


def _alt_dataset(seed=7):
    return _make_synth_dataset(seed=seed)


def _populate_caches(solver):
    """Force tcBF and acBF basis caches to materialize."""
    solver.get_tcBF()
    solver.get_acBF()


def _basis_keys(solver):
    return set(solver._basis_cache.keys())


def _reshape_dataset_to_scan_shape(dataset, new_Ry, new_Rx):
    """Build a synth dataset of a different scan shape with the same BF geometry."""
    rng = np.random.default_rng(123)
    Npix = dataset.shape[-1]
    base = dataset[0, 0]
    out = np.broadcast_to(
        base[np.newaxis, np.newaxis], (new_Ry, new_Rx, Npix, Npix)
    ).copy()
    out += rng.normal(0, 0.01, out.shape).astype(np.float32)
    return np.clip(out, 0, None).astype(np.float32)


# ── Tier 1: update_dataset ────────────────────────────────────────────────────

class TestUpdateDataset:

    def test_preserves_all_basis_caches(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        keys_before = _basis_keys(solver)
        assert keys_before, "expected basis cache to be populated"

        solver.update_dataset(_alt_dataset(seed=11))

        assert _basis_keys(solver) == keys_before
        assert solver._image_fft is not None  # FFT refreshed in-place

    def test_matches_fresh_solver(self, device):
        new_data = _alt_dataset(seed=11)
        solver = _build_solver(device=device)
        solver.get_tcBF()  # warm up
        solver.update_dataset(new_data)
        live = solver.get_tcBF()

        ref = _build_solver(dataset=new_data, device=device).get_tcBF()
        torch.testing.assert_close(live, ref, atol=1e-5, rtol=0)

    def test_shape_mismatch_raises(self, device):
        solver = _build_solver(device=device)
        bad = _make_synth_dataset()[: SYNTH_NY - 1, : SYNTH_NX - 1]
        with pytest.raises(ValueError, match="scan shape"):
            solver.update_dataset(bad)

    def test_self_dataset_tracks_latest(self, device):
        """`solver.dataset` must follow updates — refinement._build_roi_solver
        crops it for ROI sweeps, so a stale reference both leaks the original
        4D array and silently produces ROIs from the wrong frame."""
        solver = _build_solver(device=device)
        new_data = _alt_dataset(seed=11)
        solver.update_dataset(new_data)
        assert solver.dataset is new_data

    def test_cuda_path_b_reuses_4d_buffers(self, device):
        """On CUDA, update_dataset takes Path B (large H2D + device gather).

        After the first call the pinned host + device staging 4D buffers are
        allocated; subsequent calls must reuse the same buffer objects so the
        live loop never re-allocates ~2 GB per frame.
        """
        if device != "cuda":
            pytest.skip("Path B is CUDA-only")
        solver = _build_solver(device=device)
        # Before the first update_dataset call, the 4D buffers are unallocated.
        assert solver._dataset_pinned_buffer_4d is None
        assert solver._dataset_device_staging_4d is None

        solver.update_dataset(_alt_dataset(seed=11))
        pinned1 = solver._dataset_pinned_buffer_4d
        device1 = solver._dataset_device_staging_4d
        assert pinned1 is not None and device1 is not None
        assert pinned1.is_pinned()
        assert device1.device.type == "cuda"

        solver.update_dataset(_alt_dataset(seed=12))
        # Same Python objects on the second pass — proves buffer reuse.
        assert solver._dataset_pinned_buffer_4d is pinned1
        assert solver._dataset_device_staging_4d is device1

    def test_cuda_path_b_invalidates_buffers_on_bf_geometry_change(self, device):
        """Tier-3 setters that change the BF mask or scan shape must drop the
        stashed 4D buffers so the next update_dataset call reallocates them.
        """
        if device != "cuda":
            pytest.skip("Path B is CUDA-only")
        solver = _build_solver(device=device)
        solver.update_dataset(_alt_dataset(seed=11))
        assert solver._dataset_pinned_buffer_4d is not None

        new_data = _alt_dataset(seed=13)
        solver.update_convergence_angle(SYNTH_MAX_ALPHA * 1.2, new_data)
        # _bf_mask_bool_d must follow the new mask.
        np.testing.assert_array_equal(
            solver._bf_mask_bool_d.cpu().numpy(), solver._bf_mask_bool
        )
        # 4D buffers were released; next update_dataset will reallocate them.
        assert solver._dataset_pinned_buffer_4d is None
        assert solver._dataset_device_staging_4d is None

        solver.update_dataset(_alt_dataset(seed=14))
        assert solver._dataset_pinned_buffer_4d is not None


# ── Tier 2: update_scan_step ──────────────────────────────────────────────────

class TestUpdateScanStep:

    def test_invalidates_optics_only(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        keys_before = _basis_keys(solver)

        solver.update_scan_step(SYNTH_SCAN_STEP * 0.5)

        # Same keys (TCBFCache + ACBFGeometryCache survive).
        assert _basis_keys(solver) == keys_before
        # ACBFOpticsCache should be cleared (None) for every acBF entry.
        for k, v in solver._basis_cache.items():
            if k[0] == 'acBF':
                geometry, optics = v
                assert geometry is not None
                assert optics is None

    def test_no_op_when_unchanged(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        cache_ref = solver._basis_cache
        qx_before = solver.qx_grid

        solver.update_scan_step(SYNTH_SCAN_STEP)

        # The early-return path must not touch the cache or qx_grid.
        assert solver._basis_cache is cache_ref
        assert solver.qx_grid is qx_before

    def test_matches_fresh_solver(self, device):
        new_step = SYNTH_SCAN_STEP * 0.7
        solver = _build_solver(device=device)
        solver.update_scan_step(new_step)
        live = solver.get_tcBF()

        ref = _build_solver(scan_step_size=new_step, device=device).get_tcBF()
        torch.testing.assert_close(live, ref, atol=1e-5, rtol=0)


# ── Tier 3 partial: update_scan_shape ─────────────────────────────────────────

class TestUpdateScanShape:

    def test_preserves_geometry_caches(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        keys_before = _basis_keys(solver)

        new_data = _reshape_dataset_to_scan_shape(_make_synth_dataset(), 16, 12)
        solver.update_scan_shape(new_data)

        assert solver.Ry == 16 and solver.Rx == 12
        assert _basis_keys(solver) == keys_before  # geometry caches survive
        for k, v in solver._basis_cache.items():
            if k[0] == 'acBF':
                _, optics = v
                assert optics is None  # optics cleared (qx/qy changed)
        # FFT is rebuilt eagerly on the new shape (fast-path for live frames).
        assert solver._image_fft is not None
        assert solver._image_fft.img_fft.shape == solver.vbf_images.shape

    def test_self_dataset_tracks_latest(self, device):
        solver = _build_solver(device=device)
        new_data = _reshape_dataset_to_scan_shape(_make_synth_dataset(), 16, 12)
        solver.update_scan_shape(new_data)
        assert solver.dataset is new_data

    def test_falls_back_to_update_dataset_when_shape_unchanged(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        keys_before = _basis_keys(solver)

        same_shape = _alt_dataset(seed=11)
        solver.update_scan_shape(same_shape)

        # Behaves like update_dataset: caches preserved, FFT refreshed.
        assert _basis_keys(solver) == keys_before
        assert solver._image_fft is not None


# ── Tier 3 heavy: update_convergence_angle / dk / wavelength ──────────────────

class TestHeavyUpdates:

    def test_convergence_angle_clears_caches_and_rebuilds(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        assert solver._basis_cache

        new_alpha = SYNTH_MAX_ALPHA * 0.8
        solver.update_convergence_angle(new_alpha, _make_synth_dataset())

        assert solver._basis_cache == {}
        assert solver._image_fft is None
        assert solver.max_alpha == pytest.approx(new_alpha)
        # tolerance_factors must reflect the new alpha.
        alpha_rad = new_alpha / 1e3
        expected_T1 = 2 * SYNTH_WAVELENGTH / (8 * alpha_rad ** 2)
        assert solver.tolerance_factors[1] == pytest.approx(expected_T1, rel=1e-6)

    def test_convergence_angle_preserves_physical_aberration(self, device):
        solver = _build_solver(
            aberrations={"C10": 50.0, "C12": 10.0, "phi12": 30.0}, device=device,
        )
        c10_before = solver.ab_state.get_physical("C_1_0")

        solver.update_convergence_angle(SYNTH_MAX_ALPHA * 0.8, _make_synth_dataset())
        c10_after = solver.ab_state.get_physical("C_1_0")

        # Physical value must be preserved across re-normalization.
        assert c10_after == pytest.approx(c10_before, rel=1e-4)

    def test_convergence_angle_matches_fresh_solver(self, device):
        new_alpha = SYNTH_MAX_ALPHA * 0.8
        new_data = _make_synth_dataset()

        solver = _build_solver(device=device)
        solver.update_convergence_angle(new_alpha, new_data)
        live = solver.get_tcBF()

        ref = _build_solver(dataset=new_data, max_alpha=new_alpha, device=device).get_tcBF()
        torch.testing.assert_close(live, ref, atol=1e-5, rtol=0)

    def test_dk_clears_caches(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        T1_before = solver.tolerance_factors[1]

        solver.update_dk(SYNTH_DK * 1.05, _make_synth_dataset())

        assert solver._basis_cache == {}
        assert solver._image_fft is None
        # dk changes do NOT alter tolerance_factors (alpha and lambda unchanged).
        assert solver.tolerance_factors[1] == pytest.approx(T1_before, rel=1e-12)

    def test_heavy_setters_track_latest_dataset(self, device):
        solver = _build_solver(device=device)
        new_data = _make_synth_dataset(seed=33)
        solver.update_convergence_angle(SYNTH_MAX_ALPHA * 0.8, new_data)
        assert solver.dataset is new_data

    def test_wavelength_clears_caches_and_updates_tolerance(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)

        new_wl = SYNTH_WAVELENGTH * 1.1
        solver.update_wavelength(new_wl, _make_synth_dataset())

        assert solver._basis_cache == {}
        assert solver.wavelength == pytest.approx(new_wl)
        alpha_rad = SYNTH_MAX_ALPHA / 1e3
        expected_T1 = 2 * new_wl / (8 * alpha_rad ** 2)
        assert solver.tolerance_factors[1] == pytest.approx(expected_T1, rel=1e-6)


# ── apply_metadata dispatcher ─────────────────────────────────────────────────

class TestApplyMetadata:

    def test_warns_on_unknown_keys(self, device, caplog):
        solver = _build_solver(device=device)
        with caplog.at_level("WARNING"):
            solver.apply_metadata({"bogus_key": 1.0})
        assert any("bogus_key" in rec.message for rec in caplog.records)

    def test_heavy_change_requires_dataset(self, device):
        solver = _build_solver(device=device)
        with pytest.raises(ValueError, match="dataset"):
            solver.apply_metadata({"max_alpha": SYNTH_MAX_ALPHA * 0.9})

    def test_no_op_for_unchanged_metadata(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        cache_before = solver._basis_cache
        fft_before = solver._image_fft

        solver.apply_metadata({
            "max_alpha": SYNTH_MAX_ALPHA,
            "dk": SYNTH_DK,
            "wavelength": SYNTH_WAVELENGTH,
            "scan_step_size": SYNTH_SCAN_STEP,
            "rotation_deg": 0.0,
        })

        assert solver._basis_cache is cache_before
        assert solver._image_fft is fft_before

    def test_dataset_consumed_once_on_heavy_update(self, device):
        """Heavy setter must consume the dataset; trailing update_dataset must be skipped."""
        solver = _build_solver(device=device)

        calls = []
        original_update_dataset = solver.update_dataset

        def spy(ds):
            calls.append(ds.shape)
            return original_update_dataset(ds)

        solver.update_dataset = spy  # type: ignore[assignment]
        solver.apply_metadata(
            {"max_alpha": SYNTH_MAX_ALPHA * 0.9},
            dataset=_make_synth_dataset(),
        )
        assert calls == [], "update_dataset must not be called after heavy setter"

    def test_rotation_change_clears_basis(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        assert solver._basis_cache

        solver.apply_metadata({"rotation_deg": 5.0})

        # set_rotation_deg(clear_basis=True) must wipe the basis caches.
        assert solver._basis_cache == {}
        assert solver.rotation_deg == pytest.approx(5.0)

    def test_scan_step_only_invalidates_optics(self, device):
        solver = _build_solver(device=device)
        _populate_caches(solver)
        keys_before = _basis_keys(solver)

        solver.apply_metadata({"scan_step_size": SYNTH_SCAN_STEP * 0.7})

        assert _basis_keys(solver) == keys_before
        for k, v in solver._basis_cache.items():
            if k[0] == 'acBF':
                _, optics = v
                assert optics is None

    def test_combined_change_matches_fresh_solver(self, device):
        new_data = _alt_dataset(seed=99)
        new_step = SYNTH_SCAN_STEP * 0.6
        new_rot = 7.5

        solver = _build_solver(device=device)
        solver.apply_metadata(
            {"scan_step_size": new_step, "rotation_deg": new_rot},
            dataset=new_data,
        )
        live = solver.get_tcBF()

        ref = _build_solver(dataset=new_data, scan_step_size=new_step, device=device)
        ref.set_rotation_deg(new_rot)
        torch.testing.assert_close(live, ref.get_tcBF(), atol=1e-5, rtol=0)
