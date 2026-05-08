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
        a, b, c = fit_coeffs
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
                solver.set_rotation_deg(float(angle))
                img = solver.reconstruct(mode=mode, **kwargs)
                scores.append(QualityMetrics.evaluate(img, metric=metric).item())
    finally:
        solver.set_rotation_deg(original_rotation)

    scores = np.array(scores)
    optimal_index = int(np.argmax(scores))
    optimal_rotation = float(angles[optimal_index])

    print(f"Optimal rotation found at {optimal_rotation:.2f} deg")
    solver.set_rotation_deg(optimal_rotation)
    solver.reconstructed_image = None  # invalidate stale cache


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
        (False, False, False),
        (True,  False, False),
        (False, True,  False),
        (False, False, True),
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
