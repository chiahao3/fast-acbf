"""Integration tests for BFSolver — init, shapes, dtypes, invariants, regression."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from fast_acbf import BFSolver, QualityMetrics


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

    def test_removed_upscale_constructor_args_raise(self, synth_dataset, synth_params, device):
        p = synth_params
        with pytest.raises(TypeError, match="upscale_method"):
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

        with pytest.raises(TypeError, match="defer_upscale"):
            BFSolver(
                dataset=synth_dataset,
                max_alpha=p["max_alpha"],
                scan_step_size=p["scan_step_size"],
                dk=p["dk"],
                wavelength=p["wavelength"],
                max_order=2,
                aberrations={"C10": 0.0},
                device=device,
                defer_upscale=True,
            )

    def test_reconstruction_upscale_unsupported(self, solver_zero_ab):
        with pytest.raises(NotImplementedError, match="upscale"):
            solver_zero_ab.get_tcBF(upscale=2)

        with pytest.raises(NotImplementedError, match="upscale"):
            solver_zero_ab.get_acBF(upscale=2)

    def test_no_global_output_frame_state(self, solver_zero_ab):
        assert not hasattr(solver_zero_ab, "output_frame")

    def test_rotation_deg_is_updated_via_setter(self, solver_zero_ab):
        with pytest.raises(AttributeError):
            solver_zero_ab.rotation_deg = 12.0

        solver_zero_ab.set_rotation_deg(12.0)
        assert solver_zero_ab.rotation_deg == pytest.approx(12.0)
        assert solver_zero_ab.coord_transform["rotation_deg"] == pytest.approx(12.0)
        solver_zero_ab.set_rotation_deg(0.0)

    def test_invalid_reconstruction_frame_raises(self, solver_zero_ab):
        with pytest.raises(ValueError, match="frame"):
            solver_zero_ab.get_tcBF(frame="detetor")


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


# ── Autograd boundary behavior ───────────────────────────────────────────────

class TestAutogradBoundary:

    def _make_solver(self, synth_dataset, synth_params, device, **kwargs):
        p = synth_params
        defaults = dict(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 50.0, "C12": 10.0, "phi12": 30.0},
            device=device,
        )
        defaults.update(kwargs)
        return BFSolver(**defaults)

    def test_reconstruct_no_grad_by_default(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)
        img = solver.reconstruct(mode='tcBF', chunk_size=8)
        assert not img.requires_grad

    def test_reconstruct_requires_grad_opt_in(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)
        img = solver.reconstruct(mode='tcBF', requires_grad=True, chunk_size=8)
        assert img.requires_grad

        loss = QualityMetrics.evaluate(img, metric='normalized_std')
        loss.backward()
        grads = [p.grad for p in solver.ab_state.coeffs.values()]
        assert any(g is not None for g in grads)

    def test_public_getters_store_detached_images(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)
        assert not solver.get_tcBF(chunk_size=8).requires_grad
        assert solver.reconstructed_image is not None
        assert not solver.reconstructed_image.requires_grad

        assert not solver.get_acBF(chunk_size=8).requires_grad
        assert solver.reconstructed_image is not None
        assert not solver.reconstructed_image.requires_grad

    def test_plot_reconstruction_does_not_store_graph(self, synth_dataset, synth_params, device, monkeypatch):
        from fast_acbf.vis import plotting

        solver = self._make_solver(synth_dataset, synth_params, device)
        monkeypatch.setattr(plotting, "plot_reconstruction", lambda *args, **kwargs: None)

        solver.plot_reconstruction(mode='tcBF', chunk_size=8)

        assert solver.reconstructed_image is not None
        assert not solver.reconstructed_image.requires_grad

    def test_non_ad_refinements_do_not_store_graph(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)

        solver.refine_defocus(
            search_range=(40.0, 60.0),
            num_points=3,
            plot_search=False,
            chunk_size=8,
        )
        assert solver.reconstructed_image is not None
        assert not solver.reconstructed_image.requires_grad

        solver.refine_flips(chunk_size=8, plot_search=False)
        assert solver.reconstructed_image is not None
        assert not solver.reconstructed_image.requires_grad
        torch.testing.assert_close(
            solver.reconstructed_image,
            solver.reconstruct(mode='tcBF', chunk_size=8),
            atol=1e-6,
            rtol=1e-6,
        )

        solver.refine_scan_rotation(
            search_range=(-1.0, 1.0),
            num_points=3,
            plot_search=False,
            chunk_size=8,
        )
        assert solver.reconstructed_image is not None
        assert not solver.reconstructed_image.requires_grad
        torch.testing.assert_close(
            solver.reconstructed_image,
            solver.reconstruct(mode='tcBF', chunk_size=8),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_refine_defocus_defaults_to_tolerance_window(self, synth_dataset, synth_params, device, monkeypatch):
        from fast_acbf.vis import plotting

        solver = self._make_solver(synth_dataset, synth_params, device)
        captured = {}

        def fake_plot(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(plotting, "plot_defocus_line_search", fake_plot)

        solver.refine_defocus(
            num_points=3,
            method='max',
            chunk_size=8,
        )

        c10 = 50.0
        half = 24.0 * solver.tolerance_factors[1]
        assert captured["search_range"] == pytest.approx((c10 - half, c10 + half))
        assert captured["c10_axis_np"] == pytest.approx(np.linspace(c10 - half, c10 + half, 3))

    def test_refine_defocus_uses_literal_search_range(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)

        solver.refine_defocus(
            search_range=(40.0, 60.0),
            num_points=3,
            method='max',
            plot_search=False,
            chunk_size=8,
        )

        assert solver.last_c10_stack_axis.shape == (3,)
        torch.testing.assert_close(
            solver.last_c10_stack_axis.cpu(),
            torch.tensor([40.0, 50.0, 60.0]),
        )

    def test_refine_wrappers_reject_positional_options(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)

        with pytest.raises(TypeError):
            solver.refine_defocus((40.0, 60.0))

        with pytest.raises(TypeError):
            solver.refine_scan_rotation((-1.0, 1.0))

    def test_refine_defocus_halfwidth_centers_on_current_value(self, synth_dataset, synth_params, device, monkeypatch):
        from fast_acbf.vis import plotting

        solver = self._make_solver(synth_dataset, synth_params, device)
        captured = {}

        def fake_plot(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(plotting, "plot_defocus_line_search", fake_plot)

        solver.refine_defocus(
            search_halfwidth=5.0,
            num_points=3,
            method='max',
            chunk_size=8,
        )

        assert captured["search_range"] == pytest.approx((45.0, 55.0))

    def test_refine_scan_rotation_defaults_to_plus_minus_45_deg(self, synth_dataset, synth_params, device, monkeypatch):
        from fast_acbf.vis import plotting

        solver = self._make_solver(synth_dataset, synth_params, device)
        captured = {}

        def fake_plot(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(plotting, "plot_rotation_line_search", fake_plot)

        solver.refine_scan_rotation(
            num_points=3,
            chunk_size=8,
        )

        assert captured["angles_deg"] == pytest.approx(np.array([-45.0, 0.0, 45.0]))

    def test_refine_scan_rotation_halfwidth_centers_on_current_value(self, synth_dataset, synth_params, device, monkeypatch):
        from fast_acbf.vis import plotting

        solver = self._make_solver(synth_dataset, synth_params, device)
        solver.set_rotation_deg(20.0)
        captured = {}

        def fake_plot(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(plotting, "plot_rotation_line_search", fake_plot)

        solver.refine_scan_rotation(
            search_halfwidth=5.0,
            num_points=3,
            chunk_size=8,
        )

        assert captured["angles_deg"] == pytest.approx(np.array([15.0, 20.0, 25.0]))

    def test_refine_scan_rotation_plot_receives_scores(self, synth_dataset, synth_params, device, monkeypatch):
        from fast_acbf.vis import plotting

        solver = self._make_solver(synth_dataset, synth_params, device)
        captured = {}

        def fake_plot(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(plotting, "plot_rotation_line_search", fake_plot)

        solver.refine_scan_rotation(
            search_range=(-1.0, 1.0),
            num_points=3,
            chunk_size=8,
        )

        assert captured["angles_deg"].shape == (3,)
        assert captured["quality_scores"].shape == (3,)
        assert captured["optimal_rotation"] in captured["angles_deg"]
        assert captured["metric"] == "laplacian"
        assert captured["mode"] == "tcbf"

    def test_refine_flips_plot_receives_eight_panels(self, synth_dataset, synth_params, device, monkeypatch):
        from fast_acbf.vis import plotting

        solver = self._make_solver(synth_dataset, synth_params, device)
        captured = {}

        def fake_plot(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(plotting, "plot_flips_grid_search", fake_plot)

        results = solver.refine_flips(chunk_size=8)

        assert len(captured["images"]) == 8
        assert len(captured["scores"]) == 8
        assert captured["best_combo"] == results["best"]
        assert captured["best_combo"] in captured["scores"]
        assert captured["images"][0].shape == synth_dataset.shape[:2]

    def test_roi_refine_aberrations_updates_full_solver_no_grad(self, synth_dataset, synth_params, device):
        solver = self._make_solver(
            synth_dataset,
            synth_params,
            device,
            aberrations={"C10": 25.0, "C12": 5.0, "phi12": 30.0},
        )
        before = {
            key: solver.ab_state.get_physical(key)
            for key in solver.ab_state.coeffs
        }

        solver.refine_aberrations(
            iters=1,
            lr=0.1,
            scan_roi=(0, 4, 0, 4),
            chunk_size=8,
        )

        after = {
            key: solver.ab_state.get_physical(key)
            for key in solver.ab_state.coeffs
        }
        assert any(after[key] != pytest.approx(before[key]) for key in before)
        assert solver.reconstructed_image.shape == synth_dataset.shape[:2]
        assert not solver.reconstructed_image.requires_grad

    def test_refine_all_params_always_refreshes_final_image(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)

        solver.refine_all_params(
            targets=('orientation_defocus',),
            defocus_range=(40.0, 60.0),
            rotation_num_points=2,
            defocus_num_points=2,
            mode='tcBF',
            metric='laplacian',
            chunk_size=8,
        )

        assert solver.reconstructed_image is not None
        assert not solver.reconstructed_image.requires_grad
        torch.testing.assert_close(
            solver.reconstructed_image,
            solver.reconstruct(mode='tcBF', chunk_size=8),
            atol=1e-6,
            rtol=1e-6,
        )


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

    def test_detector_frame_export_ignores_scan_rotation(self, synth_dataset, synth_params, device):
        p = synth_params
        aberrations = {"C10": 50.0, "C12": 10.0, "phi12": 30.0}
        kwargs = dict(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations=aberrations,
            device=device,
        )
        reference = BFSolver(**kwargs).get_aberrations_dict(layout='flat')
        rotated = BFSolver(
            **kwargs,
            coord_transform={'rotation_deg': 37.0},
        ).get_aberrations_dict(frame='detector', layout='flat')

        assert rotated.keys() == reference.keys()
        for key in reference:
            assert rotated[key] == pytest.approx(reference[key], abs=1e-4)

    def test_scan_frame_export_matches_rotated_detector_coefficients(self, synth_dataset, synth_params, device):
        p = synth_params
        solver = BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 50.0, "C12": 10.0, "phi12": 30.0},
            coord_transform={'rotation_deg': 37.0},
            device=device,
        )

        det_flat = solver.ab_state.get_flat_coeffs()
        scan_flat = solver.ab_state.to_scan_frame(solver.rotation_deg)

        # Locate the C12 (n=1, m=2) pair in the flat layout.
        idx = 0
        for (n, m) in solver.ab_state.order_keys:
            if (n, m) == (1, 2):
                break
            idx += 1 if m == 0 else 2

        theta = 2 * solver.rotation_deg * np.pi / 180.0
        ca, cb = det_flat[idx].item(), det_flat[idx + 1].item()

        # C10 is symmetric; scan frame must equal detector frame.
        assert scan_flat[0].item() == pytest.approx(det_flat[0].item(), abs=1e-5)
        # C12a/b rotate by m*theta.
        assert scan_flat[idx].item()     == pytest.approx(ca * np.cos(theta) - cb * np.sin(theta), abs=1e-4)
        assert scan_flat[idx + 1].item() == pytest.approx(ca * np.sin(theta) + cb * np.cos(theta), abs=1e-4)

    def test_scan_frame_public_export_uses_rotated_coefficients(self, synth_dataset, synth_params, device):
        p = synth_params
        solver = BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 50.0, "C12": 10.0, "phi12": 30.0},
            coord_transform={'rotation_deg': 37.0},
            device=device,
        )

        exported = solver.get_aberrations_dict(frame='scan', layout='nested')
        expected = solver.ab_state.flat_to_cartesian_dict(solver.ab_state.to_scan_frame(solver.rotation_deg))

        assert exported[(1, 0)] == pytest.approx(expected[(1, 0)], abs=1e-4)
        assert exported[(1, 2)]['a'] == pytest.approx(expected[(1, 2)]['a'], abs=1e-3)
        assert exported[(1, 2)]['b'] == pytest.approx(expected[(1, 2)]['b'], abs=1e-3)

    def test_invalid_frame_raises(self, solver_nonzero_ab):
        with pytest.raises(ValueError, match="frame"):
            solver_nonzero_ab.get_aberrations_dict(frame='detetor')


# ── Frame cache behavior ─────────────────────────────────────────────────────

class TestFrameCacheBehavior:

    def test_no_final_image_cache(self, solver_zero_ab):
        r1 = solver_zero_ab.get_tcBF(chunk_size=8)
        r2 = solver_zero_ab.get_tcBF(chunk_size=8)
        assert r1 is not r2

    def test_aberration_change_recomputes_without_stale_final_image(self, synth_dataset, synth_params, device):
        p = synth_params
        solver = BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 0.0},
            device=device,
        )

        img0 = solver.get_tcBF(chunk_size=8).detach().clone()
        with torch.no_grad():
            solver.ab_state.set_physical('C_1_0', 50.0)
        img1 = solver.get_tcBF(chunk_size=8).detach()

        assert not torch.allclose(img0, img1)

    def test_rotation_change_misses_static_cache(self, synth_dataset, synth_params, device):
        p = synth_params
        solver = BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 0.0},
            device=device,
            cache_mode='host',
        )

        solver.get_tcBF(chunk_size=8)
        assert len(solver._recon._tcbf_cache) == 1
        assert solver._recon.provider._host_cache is not None
        host_cache_id = id(solver._recon.provider._host_cache)

        solver.set_rotation_deg(15.0)

        solver.get_tcBF(chunk_size=8)
        assert len(solver._recon._tcbf_cache) == 2
        # Provider cache is untouched by rotation — same numpy array object
        assert solver._recon.provider._host_cache is not None
        assert id(solver._recon.provider._host_cache) == host_cache_id

    def test_clear_cache_clears_all_caches(self, solver_zero_ab):
        img = solver_zero_ab.get_tcBF(chunk_size=8)
        torch.testing.assert_close(solver_zero_ab.reconstructed_image, img, atol=0, rtol=0)
        assert not solver_zero_ab.reconstructed_image.requires_grad
        assert len(solver_zero_ab._recon._tcbf_cache) + len(solver_zero_ab._recon._acbf_cache) >= 1
        # host mode: provider cache is populated after first reconstruction
        assert solver_zero_ab._recon.provider._host_cache is not None

        solver_zero_ab.clear_cache()

        torch.testing.assert_close(solver_zero_ab.reconstructed_image, img, atol=0, rtol=0)
        assert not solver_zero_ab.reconstructed_image.requires_grad
        assert not solver_zero_ab._recon._tcbf_cache and not solver_zero_ab._recon._acbf_cache
        assert solver_zero_ab._recon.provider._host_cache is None


# ── Cache mode parity ────────────────────────────────────────────────────────

class TestCacheModeParity:

    def _make_solver(self, basis_mode, synth_dataset, synth_params, device):
        p = synth_params
        return BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 50.0, "C12": 10.0, "phi12": 30.0},
            device=device,
            basis_mode=basis_mode,
        )

    def test_tcbf_precompute_on_the_fly_match(self, synth_dataset, synth_params, device):
        precompute = self._make_solver("precompute", synth_dataset, synth_params, device)
        on_the_fly = self._make_solver("on_the_fly", synth_dataset, synth_params, device)

        torch.testing.assert_close(
            precompute.get_tcBF(chunk_size=8),
            on_the_fly.get_tcBF(chunk_size=8),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_acbf_phase_only_precompute_on_the_fly_match(self, synth_dataset, synth_params, device):
        precompute = self._make_solver("precompute", synth_dataset, synth_params, device)
        on_the_fly = self._make_solver("on_the_fly", synth_dataset, synth_params, device)

        torch.testing.assert_close(
            precompute.get_acBF(chunk_size=8),
            on_the_fly.get_acBF(chunk_size=8),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_acbf_complex_inversion_precompute_on_the_fly_match(self, synth_dataset, synth_params, device):
        precompute = self._make_solver("precompute", synth_dataset, synth_params, device)
        on_the_fly = self._make_solver("on_the_fly", synth_dataset, synth_params, device)

        torch.testing.assert_close(
            precompute.get_acBF(chunk_size=8, acbf_algorithm='complex_inversion'),
            on_the_fly.get_acBF(chunk_size=8, acbf_algorithm='complex_inversion'),
            atol=1e-5,
            rtol=1e-5,
        )


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


# ── Image/basis split behavior ───────────────────────────────────────────────

class TestImageBasisSplit:

    def _make_solver(self, synth_dataset, synth_params, device, **ab_kwargs):
        p = synth_params
        return BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 0.0, **ab_kwargs},
            device=device,
            cache_mode='host',
        )

    def test_fft_cache_shared_across_rotations(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)
        solver.get_tcBF(chunk_size=8)
        host_cache_id = id(solver._recon.provider._host_cache)

        solver.set_rotation_deg(15.0, clear_basis=True)
        solver.get_tcBF(chunk_size=8)

        # Same underlying numpy array — provider cache not touched by rotation
        assert id(solver._recon.provider._host_cache) == host_cache_id

    def test_tcbf_and_acbf_share_same_provider_cache(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)
        # Trigger both modes; both must reuse the provider's single host cache.
        solver.get_tcBF(chunk_size=8)
        host_cache_after_tcbf = solver._recon.provider._host_cache
        solver.get_acBF(chunk_size=8)
        assert solver._recon.provider._host_cache is host_cache_after_tcbf

    def test_clear_basis_cache_preserves_provider_cache(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)
        solver.get_tcBF(chunk_size=8)
        host_cache = solver._recon.provider._host_cache

        solver.clear_basis_cache()

        assert not solver._recon._tcbf_cache and not solver._recon._acbf_cache
        assert solver._recon.provider._host_cache is host_cache

    def test_clear_cache_resets_both(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)
        solver.get_tcBF(chunk_size=8)
        assert solver._recon.provider._host_cache is not None
        assert len(solver._recon._tcbf_cache) >= 1

        solver.clear_cache()

        assert solver._recon.provider._host_cache is None
        assert not solver._recon._tcbf_cache and not solver._recon._acbf_cache

    def test_set_rotation_clear_basis_preserves_provider_cache(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device)
        solver.get_tcBF(chunk_size=8)
        host_cache = solver._recon.provider._host_cache

        solver.set_rotation_deg(20.0, clear_basis=True)

        assert solver._recon.provider._host_cache is host_cache
        assert not solver._recon._tcbf_cache and not solver._recon._acbf_cache

    def test_tcbf_output_unchanged_after_split(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device, C10=50.0, C12=10.0)
        r1 = solver.get_tcBF(chunk_size=8).detach().clone()
        solver.clear_cache()
        r2 = solver.get_tcBF(chunk_size=8)
        torch.testing.assert_close(r1, r2, atol=1e-5, rtol=1e-5)

    def test_acbf_output_unchanged_after_split(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device, C10=50.0, C12=10.0)
        r1 = solver.get_acBF(chunk_size=8).detach().clone()
        solver.clear_cache()
        r2 = solver.get_acBF(chunk_size=8)
        torch.testing.assert_close(r1, r2, atol=1e-5, rtol=1e-5)

    def test_refine_flips_preserves_provider_cache(self, synth_dataset, synth_params, device):
        from fast_acbf.optimization.refinement import refine_flips
        solver = self._make_solver(synth_dataset, synth_params, device)
        solver.get_tcBF(chunk_size=8)
        host_cache = solver._recon.provider._host_cache

        refine_flips(solver, mode='tcBF', metric='laplacian', plot_search=False)

        assert solver._recon.provider._host_cache is host_cache

    def test_refine_scan_rotation_preserves_provider_cache_and_clears_basis(self, synth_dataset, synth_params, device):
        from fast_acbf.optimization.refinement import refine_scan_rotation
        solver = self._make_solver(synth_dataset, synth_params, device)
        solver.get_tcBF(chunk_size=8)
        host_cache = solver._recon.provider._host_cache

        refine_scan_rotation(
            solver,
            search_range=(-5.0, 5.0),
            num_points=3,
            mode='tcBF',
            metric='laplacian',
            plot_search=False,
            chunk_size=8,
        )

        assert solver._recon.provider._host_cache is host_cache
        assert not solver._recon._tcbf_cache and not solver._recon._acbf_cache


# ── Geometry/optics split + live-update behavior ─────────────────────────────

class TestACBFGeometryOpticsSplit:

    def _make_solver(self, synth_dataset, synth_params, device, basis_mode='on_the_fly'):
        p = synth_params
        return BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 50.0, "C12": 10.0},
            device=device,
            basis_mode=basis_mode,
        )

    def test_on_the_fly_to_precompute_reuses_geometry(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device, basis_mode='on_the_fly')
        solver.get_acBF(chunk_size=8)
        geometry_otf, optics_otf = solver._recon._get_acbf_cache(chunk_size=8)
        assert optics_otf is None

        solver.basis_mode = 'precompute'
        solver.get_acBF(chunk_size=8)
        geometry_pre, optics_pre = solver._recon._get_acbf_cache(chunk_size=8)
        assert geometry_pre is geometry_otf
        assert optics_pre is not None

    def test_precompute_to_on_the_fly_gates_optics_to_none(self, synth_dataset, synth_params, device):
        solver = self._make_solver(synth_dataset, synth_params, device, basis_mode='precompute')
        solver.get_acBF(chunk_size=8)
        geometry_pre, optics_pre = solver._recon._get_acbf_cache(chunk_size=8)
        assert optics_pre is not None

        solver.basis_mode = 'on_the_fly'
        geometry_otf, optics_otf = solver._recon._get_acbf_cache(chunk_size=8)
        # Geometry preserved; optics gated to None even though storage retains it.
        assert geometry_otf is geometry_pre
        assert optics_otf is None
        # Reconstruction still works in on_the_fly mode after the toggle.
        solver.get_acBF(chunk_size=8)


# ── ImageFFT cache mode behavior ──────────────────────────────────────────────

class TestImageFFTCacheMode:
    """Verify that each cache_mode allocates/fills storage exactly as claimed.

    Tests check the internal fields _device_cache, _host_cache, _host_filled
    directly so that bugs in the cache logic surface immediately.
    """

    def _make_solver(self, cache_mode, synth_dataset, synth_params, device):
        p = synth_params
        return BFSolver(
            dataset=synth_dataset,
            max_alpha=p["max_alpha"],
            scan_step_size=p["scan_step_size"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            max_order=2,
            aberrations={"C10": 50.0},
            device=device,
            cache_mode=cache_mode,
        )

    # -- on_the_fly --------------------------------------------------------

    def test_on_the_fly_never_allocates_cache(self, synth_dataset, synth_params, device):
        solver = self._make_solver('on_the_fly', synth_dataset, synth_params, device)
        provider = solver._recon.provider
        assert provider._device_cache is None
        assert provider._host_cache is None
        solver.get_tcBF(chunk_size=8)
        assert provider._device_cache is None, "on_the_fly must not allocate _device_cache"
        assert provider._host_cache is None, "on_the_fly must not allocate _host_cache"

    # -- host --------------------------------------------------------------

    def test_host_cache_is_none_before_reconstruction(self, synth_dataset, synth_params, device):
        solver = self._make_solver('host', synth_dataset, synth_params, device)
        provider = solver._recon.provider
        assert provider._device_cache is None
        assert provider._host_cache is None
        assert provider._host_filled is None

    def test_host_cache_fills_lazily_after_reconstruction(self, synth_dataset, synth_params, device):
        solver = self._make_solver('host', synth_dataset, synth_params, device)
        provider = solver._recon.provider
        solver.get_tcBF(chunk_size=8)
        assert provider._device_cache is None, "host mode must not touch _device_cache"
        assert provider._host_cache is not None
        assert provider._host_filled is not None
        Ry, Rx = provider.scan_shape
        assert provider._host_cache.shape == (provider.nb, Ry, Rx)
        assert provider._host_cache.dtype == np.complex64
        assert provider._host_filled.all(), "all BF pixels must be filled after a full reconstruction"

    def test_host_cache_is_reused_on_second_call(self, synth_dataset, synth_params, device):
        solver = self._make_solver('host', synth_dataset, synth_params, device)
        provider = solver._recon.provider
        solver.get_tcBF(chunk_size=8)
        cache_id = id(provider._host_cache)
        solver.get_tcBF(chunk_size=8)
        assert id(provider._host_cache) == cache_id, "_host_cache must be the same object on second call"

    # -- device ------------------------------------------------------------

    def test_device_cache_populated_at_construction(self, synth_dataset, synth_params, device):
        solver = self._make_solver('device', synth_dataset, synth_params, device)
        provider = solver._recon.provider
        assert provider._device_cache is not None, "_device_cache must be allocated eagerly at construction"
        assert provider._host_cache is None

    def test_device_cache_shape_and_dtype(self, synth_dataset, synth_params, device):
        solver = self._make_solver('device', synth_dataset, synth_params, device)
        provider = solver._recon.provider
        Ry, Rx = provider.scan_shape
        assert provider._device_cache.shape == (provider.nb, Ry, Rx)
        assert provider._device_cache.dtype == torch.complex64

    def test_device_cache_unchanged_after_reconstruction(self, synth_dataset, synth_params, device):
        solver = self._make_solver('device', synth_dataset, synth_params, device)
        provider = solver._recon.provider
        cache_id = id(provider._device_cache)
        solver.get_tcBF(chunk_size=8)
        assert id(provider._device_cache) == cache_id, "_device_cache must not be reallocated during reconstruction"

    # -- guards / materialize ----------------------------------------------

    def _make_det_geom(self, synth_params, device='cpu'):
        from fast_acbf.data.geometry import DetectorGeometry
        p = synth_params
        return DetectorGeometry.from_params(
            detector_shape=(p["Npix"], p["Npix"]),
            max_alpha=p["max_alpha"],
            dk=p["dk"],
            wavelength=p["wavelength"],
            device=device,
        )

    def _make_fake_lazy(self, arr: np.ndarray):
        """Convert an in-memory Dataset4D to a fake-lazy one for testing."""
        from fast_acbf.data.dataset4d import Dataset4D
        ds = Dataset4D(arr.copy())
        ds._handle = ds._array
        ds._array = None
        return ds

    def test_force_materialize_converts_lazy_to_host_ram(self):
        original = np.random.rand(4, 4, 8, 8).astype(np.float32)
        ds = self._make_fake_lazy(original)
        assert ds.is_lazy

        result = ds._force_materialize()

        assert not ds.is_lazy
        assert isinstance(result, np.ndarray)        # host RAM, not GPU tensor
        assert result.dtype == np.float32
        assert result.data.contiguous
        assert ds._handle is None
        np.testing.assert_array_equal(result, original)

    def test_force_materialize_noop_if_already_in_memory(self):
        from fast_acbf.data.dataset4d import Dataset4D
        arr = np.zeros((4, 4, 8, 8), dtype=np.float32)
        ds = Dataset4D(arr)
        result = ds._force_materialize()
        assert not ds.is_lazy
        assert result is ds._array

    def test_device_mode_lazy_dataset_materializes_and_caches(self, synth_dataset, synth_params, device):
        from fast_acbf.data.imagefft_provider import ImageFFTProvider
        ds = self._make_fake_lazy(synth_dataset)
        assert ds.is_lazy

        provider = ImageFFTProvider(ds, self._make_det_geom(synth_params, device), device, cache_mode='device')

        assert not ds.is_lazy
        assert isinstance(ds._array, np.ndarray)        # raw data stays in host RAM
        assert provider._device_cache is not None        # FFT cache on device
        assert provider.cache_mode == 'device'

    def test_device_mode_lazy_oom_raises_informative_error(self, synth_params, monkeypatch):
        from fast_acbf.data.imagefft_provider import ImageFFTProvider
        import fast_acbf.data.dataset4d as ds_mod

        ds = self._make_fake_lazy(np.zeros((4, 4, 8, 8), dtype=np.float32))

        def _oom(*_, **__):
            raise MemoryError
        monkeypatch.setattr(ds_mod.np, 'asarray', _oom)

        with pytest.raises(RuntimeError, match="GB") as exc_info:
            ImageFFTProvider(ds, self._make_det_geom(synth_params), 'cpu', cache_mode='device')
        msg = str(exc_info.value)
        assert "host" in msg or "on_the_fly" in msg

    def test_from_hdf5_materialize_true_loads_to_host_ram(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        from fast_acbf.data.dataset4d import Dataset4D
        data = np.random.rand(4, 4, 8, 8).astype(np.float32)
        p = tmp_path / "test.h5"
        with h5py.File(p, 'w') as f:
            f.create_dataset('data', data=data)

        ds = Dataset4D.from_hdf5(p, materialize=True)

        assert not ds.is_lazy
        assert isinstance(ds._array, np.ndarray)        # host RAM, not GPU tensor
        assert ds._array.dtype == np.float32
        assert ds._array.data.contiguous
        np.testing.assert_array_equal(ds._array, data)

    def test_from_hdf5_materialize_downstream_cache_modes(self, tmp_path, synth_params, device):
        """After materialize=True, all three cache_modes must work correctly.

        cache_mode governs FFT caching, not where the 4D dataset lives.
        The raw dataset is always in host RAM; only FFT slices move to device.
        """
        h5py = pytest.importorskip("h5py")
        from fast_acbf.data.dataset4d import Dataset4D
        from fast_acbf.data.imagefft_provider import ImageFFTProvider
        p = synth_params
        arr = np.random.rand(p["Ny"], p["Nx"], p["Npix"], p["Npix"]).astype(np.float32)
        hf = tmp_path / "test.h5"
        with h5py.File(hf, 'w') as f:
            f.create_dataset('data', data=arr)
        det_geom = self._make_det_geom(synth_params, device)

        for mode in ('on_the_fly', 'host', 'device'):
            ds = Dataset4D.from_hdf5(hf, materialize=True)
            assert not ds.is_lazy                       # pre-condition: in host RAM
            provider = ImageFFTProvider(ds, det_geom, device, cache_mode=mode)
            assert provider.cache_mode == mode
            # Raw 4D data is still in host RAM regardless of cache_mode
            assert isinstance(ds._array, np.ndarray)
            # Smoke-test: chunk fetch must not raise
            provider.get_chunk(0, min(4, provider.nb))

    def test_cache_mode_property_reports_resolved_string(self, synth_dataset, synth_params, device):
        for mode in ('on_the_fly', 'host', 'device'):
            solver = self._make_solver(mode, synth_dataset, synth_params, device)
            assert solver.cache_mode == mode

    # -- numerical parity --------------------------------------------------

    def test_numerical_parity_tcbf_all_modes(self, synth_dataset, synth_params, device):
        ref  = self._make_solver('on_the_fly', synth_dataset, synth_params, device).get_tcBF(chunk_size=8)
        host = self._make_solver('host',       synth_dataset, synth_params, device).get_tcBF(chunk_size=8)
        dev  = self._make_solver('device',     synth_dataset, synth_params, device).get_tcBF(chunk_size=8)
        torch.testing.assert_close(host, ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(dev,  ref, atol=1e-5, rtol=1e-5)

    def test_numerical_parity_acbf_all_modes(self, synth_dataset, synth_params, device):
        ref  = self._make_solver('on_the_fly', synth_dataset, synth_params, device).get_acBF(chunk_size=8)
        host = self._make_solver('host',       synth_dataset, synth_params, device).get_acBF(chunk_size=8)
        dev  = self._make_solver('device',     synth_dataset, synth_params, device).get_acBF(chunk_size=8)
        torch.testing.assert_close(host, ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(dev,  ref, atol=1e-5, rtol=1e-5)


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
