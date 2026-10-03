"""Visualization functions — accept pre-computed arrays, no BFSolver dependency."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


# ---- Ported from PtyRAD (ptyrad.utils.image_proc.mfft2, v1.0.0); keep in sync ---------
# Identical to PtyRAD's function (tests/test_ptyrad_port.py checks when PtyRAD is installed).

def mfft2(im):
    # Periodic Artifact Reduction in Fourier Transforms of Full Field Atomic Resolution Images
    # https://doi.org/10.1017/S1431927614014639
    rows, cols = im.shape
    
    # Compute boundary conditions
    s = np.zeros_like(im)
    s[0, :] = im[0, :] - im[rows-1, :]
    s[rows-1, :] = -s[0, :]
    s[:, 0] += im[:, 0] - im[:, cols-1]
    s[:, cols-1] -= im[:, 0] - im[:, cols-1]

    # q[n] = 2π·n/N: DFT angular spatial frequency (rad/sample) used in discrete Laplacian eigenvalue
    q_y, q_x = np.meshgrid(2 * np.pi * np.arange(rows) / rows,
                            2 * np.pi * np.arange(cols) / cols, indexing='ij')

    # Generate smooth component from Poisson Eq with boundary condition
    D = 2 * (2 - np.cos(q_y) - np.cos(q_x))
    D[0, 0] = np.inf  # Enforce zero mean & handle division by zero
    S = np.fft.fft2(s) / D

    P = np.fft.fft2(im) - S  # FFT of periodic component
    return P, S

# ---- end of the PtyRAD port ------------------------------------------------------------


def plot_reconstruction(
    img: np.ndarray,
    fft: np.ndarray,
    probe: np.ndarray,
    title_str: str = None,
    desc_str: str = None,
    save_path: str = None,
    vmin_img=None,
    vmax_img=None,
    vmin_fft=None,
    vmax_fft=None,
):
    """3-panel figure: reconstructed image, log-FFT, probe amplitude."""
    vmin_img = np.percentile(img, vmin_img or 0.001)
    vmax_img = np.percentile(img, vmax_img or 99.99)
    vmin_fft = np.percentile(fft, vmin_fft or 1)
    vmax_fft = np.percentile(fft, vmax_fft or 99)

    fig, axs = plt.subplots(1, 3, figsize=(12, 5.5))
    if title_str:
        fig.suptitle(title_str, y=0.9)
    if desc_str:
        fig.text(x=0, y=0.8, s=desc_str)
    axs[0].imshow(img, vmin=vmin_img, vmax=vmax_img)
    axs[1].imshow(fft, vmin=vmin_fft, vmax=vmax_fft)
    axs[2].imshow(probe)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_chi_surface(surface: np.ndarray, title_str: str = None):
    """Plot a 2D aberration (chi) surface with colorbar."""
    plt.figure()
    if title_str:
        plt.title(title_str)
    plt.imshow(surface)
    plt.colorbar()
    plt.show()


def plot_shift_quiver(
    kx: np.ndarray,
    ky: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    k_max: float,
    subsample: int = None,
    scale=None,
    show: bool = True,
):
    """Quiver vector field of real-space image shifts over the BF disk."""
    Nb = len(kx)
    if subsample is None:
        step = max(1, Nb // 400)
    else:
        step = max(1, Nb // subsample)

    kx_sub, ky_sub = kx[::step], ky[::step]
    sx_sub, sy_sub = sx[::step], sy[::step]

    fig, ax = plt.subplots(figsize=(7, 7))
    disk_edge = plt.Circle((0, 0), k_max, color='red', fill=False,
                            linestyle='--', linewidth=1.5, alpha=0.6, label='BF Mask')
    ax.add_patch(disk_edge)
    ax.quiver(kx_sub, ky_sub, sx_sub, sy_sub,
              color='midnightblue', angles='xy', scale=scale, alpha=0.85)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x \ (\AA^{-1})$', fontsize=12)
    ax.set_ylabel(r'$k_y \ (\AA^{-1})$', fontsize=12)
    ax.set_title('Image Shifts over BF Disk', fontsize=14)
    limit = k_max * 1.15
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')

    if show:
        plt.show()

    return fig, ax


def plot_defocus_line_search(
    c10_axis_np: np.ndarray,
    quality_scores: np.ndarray,
    optimal_c10: float,
    search_range: tuple,
    metric: str,
    method: str,
    mode: str,
    fit_coeffs=None,
    fit_type: str = None,
):
    """Scatter + optional parabola fit plot for defocus line search results."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(c10_axis_np, quality_scores, color='dodgerblue', s=60,
               label='Tested Points', zorder=5)

    if method == 'fit_parabola' and fit_coeffs is not None:
        a, b, c = fit_coeffs
        c10_smooth = np.linspace(search_range[0], search_range[1], 100)
        fit_scores = a * c10_smooth**2 + b * c10_smooth + c
        ax.plot(c10_smooth, fit_scores, '--', color='gray',
                label=f'Parabolic Fit ({fit_type})', zorder=4)
        opt_score = (a * optimal_c10**2 + b * optimal_c10 + c
                     if a < 0 else max(quality_scores))
    else:
        opt_score = np.max(quality_scores)

    ax.scatter([optimal_c10], [opt_score], color='crimson', s=150, marker='*',
               label=f'Optimal ($C_{{10}}$ = {optimal_c10:.0f} Ang)', zorder=6)
    ax.set_xlabel('Overfocus $C_{10}$ (Ang)', fontsize=12)
    ax.set_ylabel(f'Focus Score ({metric.capitalize()})', fontsize=12)
    ax.set_title(f'Defocus Line Search ({mode}, {metric}, {method})', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_rotation_line_search(
    angles_deg: np.ndarray,
    quality_scores: np.ndarray,
    optimal_rotation: float,
    metric: str,
    mode: str,
):
    """Scatter plot for scan-rotation line search results."""
    optimal_index = int(np.argmax(quality_scores))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles_deg, quality_scores, color='dodgerblue', linewidth=1.5, zorder=4)
    ax.scatter(angles_deg, quality_scores, color='dodgerblue', s=55,
               label='Tested Points', zorder=5)
    ax.scatter(
        [optimal_rotation], [quality_scores[optimal_index]],
        color='crimson', s=150, marker='*',
        label=f'Optimal ({optimal_rotation:.2f} deg)', zorder=6,
    )
    ax.set_xlabel('Scan Rotation (deg)', fontsize=12)
    ax.set_ylabel(f'Focus Score ({metric.capitalize()})', fontsize=12)
    ax.set_title(f'Scan Rotation Line Search ({mode}, {metric})', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_flips_grid_search(
    images: list,
    scores: dict,
    best_combo: tuple,
    metric: str,
    mode: str,
):
    """2×4 reconstruction panel grid for flip/transpose exhaustive search."""
    stack = np.stack(images, axis=0)
    vmin = np.percentile(stack, 0.001)
    vmax = np.percentile(stack, 99.99)

    combos = list(scores.keys())
    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    for ax, combo, img in zip(axs.flat, combos, images):
        flipud, fliplr, transpose = combo
        ax.imshow(img, vmin=vmin, vmax=vmax)
        score = scores[combo]
        winner = "BEST | " if combo == best_combo else ""
        ax.set_title(
            f"{winner}ud={int(flipud)} lr={int(fliplr)} T={int(transpose)}\n"
            f"{metric}: {score:.4g}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Flip/Transpose Search ({mode}, {metric})', fontsize=14)
    plt.tight_layout()
    plt.show()
