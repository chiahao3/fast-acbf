"""Refinement routines — line search and AD-based optimization.

All functions accept a solver-like object via duck typing (no BFSolver import).
Required solver interface:
    .ab_state          — AberrationState with set_physical / get_physical
    .reconstruct(mode, **kwargs) -> Tensor
    .set_rotation_deg(deg)       — handles cache invalidation internally
    .coord_transform             — dict with flipud/fliplr/transpose flags
    .reconstructed_image         — writable attribute for caching last result
    .last_c10_stack_axis         — writable attribute

State safety: always mutate solver state through public setters, never direct
attribute writes, so that BFSolver's cache invalidation logic stays intact.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from fast_acbf.optimization.metrics import QualityMetrics


# ---------------------------------------------------------------------------
# Internal sweep helper (mirrors BFSolver._sweep_c10_stack logic)
# ---------------------------------------------------------------------------

def _sweep_c10(solver, c10_axis: torch.Tensor, mode: str, **kwargs):
    """Evaluate reconstruction over a C10 axis; restore original C10 on exit."""
    original_c10 = solver.ab_state.get_physical('C_1_0')
    stack_images = []
    try:
        with torch.no_grad():
            for c10 in c10_axis:
                solver.ab_state.set_physical('C_1_0', c10)
                img = solver.reconstruct(mode=mode, **kwargs)
                stack_images.append(img)
    finally:
        solver.ab_state.set_physical('C_1_0', original_c10)

    c10_axis = c10_axis.detach().clone()
    stack = torch.stack(stack_images, dim=0)
    solver.last_c10_stack_axis = c10_axis
    return c10_axis, stack


# ---------------------------------------------------------------------------
# Public refinement functions
# ---------------------------------------------------------------------------

def refine_defocus(
    solver,
    search_range: tuple,
    num_points: int = 5,
    metric: str = 'laplacian',
    method: str = 'fit_parabola',
    blur: bool = True,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1,
    plot_line_search: bool = True,
    mode: str = 'tcBF',
    **kwargs,
) -> None:
    """
    Brute-force line search for defocus (C10) with optional parabolic refinement.

    Sweeps C10 over search_range, scores each slice with QualityMetrics, then
    fits a parabola (or picks max) to find the optimal C10. Updates solver.ab_state
    in-place and caches the best reconstruction in solver.reconstructed_image.

    Args:
        solver:           Solver-like object (see module docstring).
        search_range:     (min_c10, max_c10) in Angstroms.
        num_points:       Number of C10 values to sample.
        metric:           Focus metric for QualityMetrics.evaluate.
        method:           'fit_parabola' or 'max'.
        blur:             Pre-blur images before scoring.
        blur_kernel_size: Kernel size for Gaussian blur.
        blur_sigma:       Sigma for Gaussian blur.
        plot_line_search: Show matplotlib line-search summary.
        mode:             Reconstruction mode ('tcBF' or 'acBF').
    """
    mode = mode.lower()
    min_def, max_def = min(search_range), max(search_range)
    search_range = (min_def, max_def)
    print(f"Starting defocus line search: {num_points} points between {search_range[0]} and {search_range[1]} Ang")

    device = next(iter(solver.ab_state.coeffs.values())).device
    c10_axis = torch.linspace(search_range[0], search_range[1], num_points,
                               dtype=torch.float32, device=device)

    c10_axis, scan_stack = _sweep_c10(solver, c10_axis, mode=mode, **kwargs)
    quality_scores = QualityMetrics.evaluate(
        scan_stack, metric=metric, blur=blur,
        blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma,
    ).detach().cpu().numpy()
    c10_axis_np = c10_axis.detach().cpu().numpy()

    method = method.lower()
    if method == 'fit_parabola':
        fit_coeffs = np.polyfit(c10_axis_np, quality_scores, 2)
        a, b, _ = fit_coeffs
        if a < 0:
            optimal_c10 = -b / (2 * a)
            optimal_c10 = np.clip(optimal_c10, search_range[0], search_range[1])
            fit_type = "Parabolic vertex"
            optimal_index = int(np.argmin(np.abs(c10_axis_np - optimal_c10)))
        else:
            optimal_index = int(np.argmax(quality_scores))
            optimal_c10 = c10_axis_np[optimal_index]
            fit_type = "Discrete max (fit inverted)"
    elif method == 'max':
        optimal_index = int(np.argmax(quality_scores))
        optimal_c10 = c10_axis_np[optimal_index]
        fit_coeffs = None
    else:
        raise ValueError(f"Unsupported method: {method!r}. Choose 'fit_parabola' or 'max'.")

    print(f"Optimal C10 found at {optimal_c10:.2f} Ang ({method})")

    with torch.no_grad():
        solver.ab_state.set_physical('C_1_0', float(optimal_c10))
    solver.reconstructed_image = scan_stack[optimal_index]

    if plot_line_search:
        from fast_acbf.vis.plotting import plot_defocus_line_search
        plot_defocus_line_search(
            c10_axis_np=c10_axis_np,
            quality_scores=quality_scores,
            optimal_c10=optimal_c10,
            search_range=search_range,
            metric=metric,
            method=method,
            mode=mode,
            fit_coeffs=fit_coeffs if method == 'fit_parabola' else None,
            fit_type=fit_type if method == 'fit_parabola' else None,
        )


def refine_aberrations(
    solver,
    lr: float = 1,
    lr_scales=None,
    iters: int = 50,
    metric: str = 'normalized_std',
    plot_recon_every_n_iter=None,
    save_dir=None,
    mode: str = 'tcBF',
    **kwargs,
) -> None:
    """
    Gradient-based refinement of all aberration coefficients via Adam.

    Minimizes -QualityMetrics(reconstruction) with per-order learning rate scaling.
    Updates solver.ab_state in-place.

    Args:
        solver:                  Solver-like object.
        lr:                      Base learning rate.
        lr_scales:               Per-order LR multipliers, length max_order.
        iters:                   Number of gradient steps.
        metric:                  Focus metric to maximize.
        plot_recon_every_n_iter: Show reconstruction every N iterations if set.
        save_dir:                Directory to save per-iteration figures if set.
        mode:                    Reconstruction mode.
    """
    mode = mode.lower()
    max_order = solver.ab_state.max_order

    if lr_scales is None:
        lr_scales = [1.0] * max_order
    lr_scales = list(lr_scales)
    if len(lr_scales) != max_order:
        raise ValueError(f"lr_scales must have length max_order={max_order}, got {len(lr_scales)}")

    param_groups = []
    for idx, n in enumerate(range(1, max_order + 1)):
        order_params = [
            solver.ab_state.coeffs[key]
            for key in solver.ab_state.coeffs
            if int(key.split('_')[1]) == n
        ]
        param_groups.append({'params': order_params, 'lr': lr * lr_scales[idx]})

    optimizer = torch.optim.Adam(param_groups, lr=lr)

    for i in range(iters):
        optimizer.zero_grad()
        summed_img = solver.reconstruct(mode=mode, **kwargs)
        solver.reconstructed_image = summed_img.detach()
        loss = -1 * QualityMetrics.evaluate(summed_img, metric=metric)
        loss.backward()
        optimizer.step()

        if plot_recon_every_n_iter is not None and i % plot_recon_every_n_iter == 0:
            with torch.no_grad():
                title_str = f'Iter {i}, Loss (-{metric}) : {loss.item():.4g}'
                ab_dict = solver.get_aberrations_dict(layout='flat')
                desc_str = ", ".join(f"{ab}: {val:.2f}" for ab, val in ab_dict.items())

                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = f'{save_dir}/figure_recon_iter_{str(i).zfill(3)}.png'
                else:
                    save_path = None

                solver.plot_reconstruction(
                    title_str=title_str, desc_str=desc_str,
                    save_path=save_path, mode=mode, **kwargs,
                )


def refine_scan_rotation(
    solver,
    search_range: tuple,
    num_points: int = 9,
    metric: str = 'laplacian',
    mode: str = 'tcBF',
    **kwargs,
) -> None:
    """
    Line search for optimal scan rotation angle.

    Sweeps rotation_deg over search_range, scores each reconstruction, and
    sets the optimal rotation via solver.set_rotation_deg() (which triggers
    cache invalidation automatically).

    Args:
        solver:       Solver-like object.
        search_range: (min_deg, max_deg) rotation range to search.
        num_points:   Number of angles to sample.
        metric:       Focus metric for QualityMetrics.evaluate.
        mode:         Reconstruction mode.
    """
    mode = mode.lower()
    min_rot, max_rot = min(search_range), max(search_range)
    angles = np.linspace(min_rot, max_rot, num_points)
    print(f"Starting rotation line search: {num_points} points between {min_rot:.1f} and {max_rot:.1f} deg")

    original_rotation = solver.rotation_deg
    scores = []

    try:
        with torch.no_grad():
            for angle in angles:
                # clear_static_cache=True: each angle produces a distinct cache key; without
                # clearing, all num_points entries accumulate in _cache_store simultaneously.
                # This sweep is sequential and never revisits angles, so only one entry is
                # needed at a time.
                solver.set_rotation_deg(float(angle), clear_static_cache=True)
                img = solver.reconstruct(mode=mode, **kwargs)
                scores.append(QualityMetrics.evaluate(img, metric=metric).item())
    finally:
        solver.set_rotation_deg(original_rotation)

    scores = np.array(scores)
    optimal_index = int(np.argmax(scores))
    optimal_rotation = float(angles[optimal_index])

    print(f"Optimal rotation found at {optimal_rotation:.2f} deg")
    solver.set_rotation_deg(optimal_rotation)
    solver.reconstructed_image = None # Clear stale image from last swept angle


def refine_flips(
    solver,
    metric: str = 'laplacian',
    mode: str = 'tcBF',
    **kwargs,
) -> dict:
    """
    Exhaustive search over all 4 flip/transpose combinations.

    Tests all combinations of (flipud, fliplr, transpose) using the current
    rotation_deg, then sets the best combination via solver.coord_transform
    and re-triggers cache invalidation.

    Args:
        solver: Solver-like object.
        metric: Focus metric for QualityMetrics.evaluate.
        mode:   Reconstruction mode.

    Returns:
        Dict mapping each (flipud, fliplr, transpose) combo to its score,
        with 'best' key indicating the winning combination.
    """
    mode = mode.lower()
    combos = [
        (flipud, fliplr, transpose)
        for flipud    in (False, True)
        for fliplr    in (False, True)
        for transpose in (False, True)
    ]

    original = {
        'flipud':    solver.coord_transform.get('flipud',    False),
        'fliplr':    solver.coord_transform.get('fliplr',    False),
        'transpose': solver.coord_transform.get('transpose', False),
    }

    results = {}
    try:
        with torch.no_grad():
            for (flipud, fliplr, transpose) in combos:
                solver.coord_transform['flipud']    = flipud
                solver.coord_transform['fliplr']    = fliplr
                solver.coord_transform['transpose'] = transpose
                solver.clear_cache()
                img = solver.reconstruct(mode=mode, **kwargs)
                score = QualityMetrics.evaluate(img, metric=metric).item()
                results[(flipud, fliplr, transpose)] = score
    finally:
        solver.coord_transform.update(original)
        solver.clear_cache()

    best_combo = max(results, key=results.__getitem__)
    results['best'] = best_combo

    print(f"Best flip combination: flipud={best_combo[0]}, fliplr={best_combo[1]}, "
          f"transpose={best_combo[2]}  (score={results[best_combo]:.4g})")

    solver.coord_transform['flipud']    = best_combo[0]
    solver.coord_transform['fliplr']    = best_combo[1]
    solver.coord_transform['transpose'] = best_combo[2]
    solver.clear_cache()

    return results


# ---------------------------------------------------------------------------
# O(2) orientation helpers — used by refine_all_params
# ---------------------------------------------------------------------------

# D4 lookup table: maps (is_flipped, q) -> (flipud, fliplr, transpose).
#
# Parameterization: the full scan→detector transform is decomposed as
#   "apply (flipud, fliplr, transpose) at rotation_deg = residual_angle"
# such that it equals the O(2) state
#   "(transpose=is_flipped only, flipud=fliplr=False) at rotation_deg = opt_angle_deg"
#
# is_flipped=False entries are the four pure CCW rotations (0°, 90°, 180°, 270°).
# is_flipped=True entries are the four reflections (transpose composed with CCW rotations).
# Derived from _get_transformed_bf_coordinates and verified numerically.
_D4_TABLE = {
    (False, 0): (False, False, False),  # identity
    (False, 1): (True,  False, True),   # CCW 90°
    (False, 2): (True,  True,  False),  # CCW 180°
    (False, 3): (False, True,  True),   # CCW 270°
    (True,  0): (False, False, True),   # transpose
    (True,  1): (False, True,  False),  # fliplr
    (True,  2): (True,  True,  True),   # flipud + fliplr + transpose
    (True,  3): (True,  False, False),  # flipud
}


def map_to_ptyrad_state(if_transposed: bool, opt_angle_deg: float) -> dict:
    """
    Maps an optimized O(2) state (chirality + continuous rotation) to
    PtyRAD-compatible D4 flags + residual rotation.

    O(2) parameterization uses only 2 chiralities rather than 8 flip combos:
      - if_transposed=False: (flipud=F, fliplr=F, transpose=F) + rotation opt_angle_deg
      - if_transposed=True:  (flipud=F, fliplr=F, transpose=T) + rotation opt_angle_deg

    Returns dict with keys: flipud, fliplr, transpose (bool), rotation_deg (float).
    Residual rotation is opt_angle_deg mod 90°, in roughly [-45°, +45°].
    """
    q = int(round(opt_angle_deg / 90.0)) % 4
    residual_angle = opt_angle_deg - q * 90.0
    flipud, fliplr, transpose = _D4_TABLE[(bool(if_transposed), q)]
    return {
        'flipud': flipud,
        'fliplr': fliplr,
        'transpose': transpose,
        'rotation_deg': residual_angle,
    }


def _orientation_grid_search(
    solver,
    defocus_range: tuple,
    rotation_num_points: int,
    defocus_num_points: int,
    metric: str,
    mode: str,
    **kwargs,
) -> None:
    """
    Joint grid search over 2 chiralities × rotation_num_points angles × defocus_num_points
    defocus values. Applies the best (chirality, angle, C_1_0) state to solver, decomposed
    to PtyRAD D4 flags via map_to_ptyrad_state.

    Cache discipline: chirality changes call clear_cache() (2 total); each rotation angle
    calls set_rotation_deg(..., clear_static_cache=True) so only one cache entry exists at
    a time. Defocus sweeps reuse the same cache entry since C_1_0 is not part of the key.
    """
    angles = np.linspace(0.0, 360.0, rotation_num_points, endpoint=False)
    c10_values = np.linspace(defocus_range[0], defocus_range[1], defocus_num_points)
    total = 2 * rotation_num_points * defocus_num_points
    print(f"Starting orientation+defocus grid search: {total} evaluations "
          f"(2 chiralities × {rotation_num_points} angles × {defocus_num_points} defocus points)")

    original_c10 = solver.ab_state.get_physical('C_1_0')
    # Non-rotationally-symmetric terms (m≠0) interact with rotation: C12 can compensate
    # for orientation errors, creating deep local minima. Zero them during the grid search.
    non_sym_keys = [k for k in solver.ab_state.coeffs if k.endswith('_a') or k.endswith('_b')]
    original_non_sym = {k: solver.ab_state.get_physical(k) for k in non_sym_keys}

    best_score = -np.inf
    best_if_transposed = False
    best_angle = 0.0
    best_c10 = float(c10_values[0])

    try:
        with torch.no_grad():
            for k in non_sym_keys:
                solver.ab_state.set_physical(k, 0.0)

            for if_transposed in (False, True):
                # One cache clear per chirality — the expensive operation
                solver.coord_transform['flipud'] = False
                solver.coord_transform['fliplr'] = False
                solver.coord_transform['transpose'] = if_transposed
                solver.set_rotation_deg(0.0)
                solver.clear_cache()

                for angle in angles:
                    # clear_static_cache=True: sequential search never revisits old angles,
                    # so accumulating per-angle cache entries only wastes memory.
                    solver.set_rotation_deg(float(angle), clear_static_cache=True)
                    for c10 in c10_values:
                        solver.ab_state.set_physical('C_1_0', float(c10))
                        img = solver.reconstruct(mode=mode, **kwargs)
                        score = QualityMetrics.evaluate(img, metric=metric).item()
                        if score > best_score:
                            best_score = score
                            best_if_transposed = if_transposed
                            best_angle = float(angle)
                            best_c10 = float(c10)
    finally:
        solver.ab_state.set_physical('C_1_0', original_c10)
        for k, v in original_non_sym.items():
            solver.ab_state.set_physical(k, v)
        solver.clear_cache()

    ptyrad_state = map_to_ptyrad_state(best_if_transposed, best_angle)
    solver.coord_transform['flipud']    = ptyrad_state['flipud']
    solver.coord_transform['fliplr']    = ptyrad_state['fliplr']
    solver.coord_transform['transpose'] = ptyrad_state['transpose']
    solver.set_rotation_deg(ptyrad_state['rotation_deg'])
    solver.ab_state.set_physical('C_1_0', best_c10)
    solver.clear_cache()

    print(f"Best: if_transposed={best_if_transposed}, angle={best_angle:.1f}°, C10={best_c10:.2f}Å "
          f"(score={best_score:.4g})")
    print(f"  → flipud={ptyrad_state['flipud']}, fliplr={ptyrad_state['fliplr']}, "
          f"transpose={ptyrad_state['transpose']}, rotation_deg={ptyrad_state['rotation_deg']:.1f}°")


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def refine_all_params(
    solver,
    targets=('orientation_defocus', 'coarse_aberrations', 'fine_rotation', 'fine_aberrations'),
    metric: str = 'normalized_std',
    mode: str = 'tcBF',
    defocus_range=None,
    defocus_range_tolerance_factor: float = 24.0,
    rotation_num_points: int = 36,
    defocus_num_points: int = 11,
    fine_rotation_halfwidth: float = 5.0,
    fine_rotation_num_points: int = 11,
    aberration_lr: float = 1.0,
    aberration_iters: int = 50,
    **kwargs,
) -> None:
    """
    Coarse-to-fine parameter orchestration.

    Executes a subset of the four pipeline steps, selected by `targets`:

      'orientation_defocus'         — Joint 2-chirality × rotation × defocus grid search (Step 1).
      'coarse_aberrations'  — Adam optimisation restricted to 1st + 2nd order (Step 2).
      'fine_rotation'       — Tight ±fine_rotation_halfwidth° line search (Step 3).
      'fine_aberrations'    — Full-order Adam optimisation (Step 4).

    Args:
        solver:                         Solver-like object.
        targets:                        Ordered sequence of step names to run.
        metric:                         Focus metric shared across all steps.
        mode:                           Reconstruction mode ('tcBF' or 'acBF').
        defocus_range:                  (min_c10, max_c10) in Å for Step 1. If None,
                                        auto-computed as current_C10 ± defocus_range_tolerance_factor × T₁.
        defocus_range_tolerance_factor: Multiplier on the 1st-order Kirkland tolerance T₁
                                        for the auto defocus range. Default 24 (= ±6π phase).
        rotation_num_points:            Rotation angles sampled in [0°, 360°) for Step 1.
        defocus_num_points:             Defocus samples in defocus_range for Step 1.
        fine_rotation_halfwidth:        Half-width in degrees for Step 3 search range.
        fine_rotation_num_points:       Number of angles for Step 3.
        aberration_lr:                  Base learning rate for Adam steps.
        aberration_iters:               Gradient steps per Adam call.
    """
    mode = mode.lower()
    targets = tuple(targets)

    if 'orientation_defocus' in targets and defocus_range is None:
        c10 = solver.ab_state.get_physical('C_1_0')
        T1 = solver.tolerance_factors[1]
        half = defocus_range_tolerance_factor * T1
        defocus_range = (c10 - half, c10 + half)
        print(f"Auto defocus_range: ({defocus_range[0]:.1f}, {defocus_range[1]:.1f}) Å "
              f"(C10={c10:.1f} ± {half:.1f} Å = ±{defocus_range_tolerance_factor:.0f}×T₁)")

    if 'orientation_defocus' in targets:
        _orientation_grid_search(
            solver,
            defocus_range=defocus_range,
            rotation_num_points=rotation_num_points,
            defocus_num_points=defocus_num_points,
            metric=metric,
            mode=mode,
            **kwargs,
        )

    if 'coarse_aberrations' in targets:
        max_order = solver.ab_state.max_order
        coarse_lr_scales = [1.0] * min(2, max_order) + [0.0] * max(0, max_order - 2)
        refine_aberrations(
            solver,
            lr=aberration_lr,
            lr_scales=coarse_lr_scales,
            iters=aberration_iters,
            metric=metric,
            mode=mode,
            **kwargs,
        )

    if 'fine_rotation' in targets:
        rot = solver.rotation_deg
        refine_scan_rotation(
            solver,
            search_range=(rot - fine_rotation_halfwidth, rot + fine_rotation_halfwidth),
            num_points=fine_rotation_num_points,
            metric=metric,
            mode=mode,
            **kwargs,
        )

    if 'fine_aberrations' in targets:
        refine_aberrations(
            solver,
            lr=aberration_lr,
            iters=aberration_iters,
            metric=metric,
            mode=mode,
            **kwargs,
        )
