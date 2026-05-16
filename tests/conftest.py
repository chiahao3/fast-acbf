"""Shared pytest fixtures for fast-acbf test suite."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


# ── pytest configuration ──────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device for BFSolver tests (default: cpu)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "regression: mark test as requiring real data fixtures"
    )


@pytest.fixture(scope="session")
def device(request):
    d = request.config.getoption("--device")
    if d == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requested but not available")
    return d


# ── Synthetic dataset parameters ──────────────────────────────────────────────

SYNTH_NY = 8
SYNTH_NX = 8
SYNTH_NPIX = 32
SYNTH_MAX_ALPHA = 25.0           # mrad  — probe convergence semi-angle
SYNTH_COLLECTION_ANGLE = 50.0   # mrad  — detector collection semi-angle (2× convergence)
SYNTH_WAVELENGTH = 0.04176       # Å  (80 kV)
SYNTH_SCAN_STEP = 0.2            # Å
SYNTH_DK = (SYNTH_COLLECTION_ANGLE / 1000.0) / (SYNTH_NPIX / 2 * SYNTH_WAVELENGTH)


@pytest.fixture(scope="session")
def synth_params():
    return dict(
        Ny=SYNTH_NY, Nx=SYNTH_NX, Npix=SYNTH_NPIX,
        max_alpha=SYNTH_MAX_ALPHA, wavelength=SYNTH_WAVELENGTH,
        scan_step_size=SYNTH_SCAN_STEP, dk=SYNTH_DK,
    )


def _make_synth_dataset(seed=42):
    """Synthetic 4D-STEM dataset with a realistic cosine-falloff BF disk."""
    rng = np.random.default_rng(seed)
    Ny, Nx, Npix = SYNTH_NY, SYNTH_NX, SYNTH_NPIX

    ky = np.fft.fftshift(np.fft.fftfreq(Npix, d=(1.0 / SYNTH_DK / Npix)))
    kx = np.fft.fftshift(np.fft.fftfreq(Npix, d=(1.0 / SYNTH_DK / Npix)))
    kX, kY = np.meshgrid(kx, ky, indexing='xy')
    kR = np.sqrt(kX**2 + kY**2)

    max_k = SYNTH_MAX_ALPHA / 1000.0 / SYNTH_WAVELENGTH
    bf_disk = (kR <= max_k).astype(np.float32)
    amplitude = bf_disk * np.cos(kR / max_k * math.pi / 2) ** 2

    dataset = np.broadcast_to(
        amplitude[np.newaxis, np.newaxis], (Ny, Nx, Npix, Npix)
    ).copy()
    dataset += rng.normal(0, 0.01, dataset.shape).astype(np.float32)
    dataset = np.clip(dataset, 0, None)
    return dataset.astype(np.float32)


@pytest.fixture(scope="session")
def synth_dataset():
    return _make_synth_dataset(seed=42)


@pytest.fixture(scope="session")
def solver_zero_ab(synth_dataset, synth_params, device):
    from fast_acbf import BFSolver
    p = synth_params
    return BFSolver.from_array(
        dataset=synth_dataset,
        max_alpha=p["max_alpha"],
        scan_step_size=p["scan_step_size"],
        dk=p["dk"],
        wavelength=p["wavelength"],
        max_order=2,
        aberrations={"C10": 0.0},
        device=device,
    )


@pytest.fixture(scope="session")
def solver_nonzero_ab(synth_dataset, synth_params, device):
    from fast_acbf import BFSolver
    p = synth_params
    return BFSolver.from_array(
        dataset=synth_dataset,
        max_alpha=p["max_alpha"],
        scan_step_size=p["scan_step_size"],
        dk=p["dk"],
        wavelength=p["wavelength"],
        max_order=2,
        aberrations={"C10": 50.0, "C12": 10.0, "phi12": 30.0},
        device=device,
    )
