#!/usr/bin/env python3
"""
generate_fixtures.py — Run ONCE before refactoring to save regression ground truth.

Usage:
    conda activate fast-acbf
    python tests/generate_fixtures.py

Set FAST_ACBF_REGRESSION_ZARR to override the default input zarr path.

Outputs (in tests/fixtures/):
    tcbf.npy            — tcBF reconstruction, shape (64, 64)
    acbf.npy            — acBF reconstruction (phase_only), shape (64, 64)
    defocus_stack.npy   — defocus stack (5 layers, 10 Å step), shape (5, 64, 64)
    metadata.json       — parameters used for reproducibility
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fast_acbf import BFSolver

DEFAULT_ZARR_PATH = (
    "/home/cl2696/scratch/"
    "test_1_static_cell_30_30_5_80kv_coherent_probe_ca_25_cl_139_C10_0_t_1_Nscans_64_64_dp_200/cbed.zarr"
)
ZARR_PATH = os.path.expanduser(
    os.environ.get("FAST_ACBF_REGRESSION_ZARR", DEFAULT_ZARR_PATH)
)
MAX_ALPHA = 25.0        # mrad
COLLECTION_ANGLE = 139  # mrad
WAVELENGTH = 0.04176    # Å (80 kV)
SCAN_STEP = 0.2         # Å
NPIX = 200
DK = (COLLECTION_ANGLE / 1000.0) / (NPIX / 2 * WAVELENGTH)
ABERRATIONS = {'C10': 2.5}
COORD_TRANSFORM = {}
MAX_ORDER = 2

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def main():
    import zarr

    if not os.path.exists(ZARR_PATH):
        print(f"ERROR: zarr data not found at:\n  {ZARR_PATH}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(FIXTURE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    z = zarr.open(ZARR_PATH, mode='r')
    dataset = np.array(z[0]).reshape(64, 64, NPIX, NPIX)
    print(f"Loaded dataset: shape={dataset.shape}, dtype={dataset.dtype}")

    solver = BFSolver(
        dataset=dataset,
        max_alpha=MAX_ALPHA,
        scan_step_size=SCAN_STEP,
        dk=DK,
        wavelength=WAVELENGTH,
        max_order=MAX_ORDER,
        aberrations=ABERRATIONS,
        coord_transform=COORD_TRANSFORM,
        device=device,
    )

    print("Computing tcBF…")
    tcbf = solver.get_tcBF().detach().cpu().numpy()
    np.save(os.path.join(FIXTURE_DIR, "tcbf.npy"), tcbf)
    print(f"  saved tcbf.npy  shape={tcbf.shape}  dtype={tcbf.dtype}")

    print("Computing acBF…")
    acbf = solver.get_acBF().detach().cpu().numpy()
    np.save(os.path.join(FIXTURE_DIR, "acbf.npy"), acbf)
    print(f"  saved acbf.npy  shape={acbf.shape}  dtype={acbf.dtype}")

    print("Computing defocus stack (5 layers, 10 Å step)…")
    stack = solver.get_defocus_stack(n_layers=5, slice_thickness=10.0)
    stack_np = stack.detach().cpu().numpy()
    np.save(os.path.join(FIXTURE_DIR, "defocus_stack.npy"), stack_np)
    print(f"  saved defocus_stack.npy  shape={stack_np.shape}  dtype={stack_np.dtype}")

    try:
        import fast_acbf
        version = fast_acbf.__version__
    except Exception:
        version = "unknown"

    meta = {
        "zarr_path": ZARR_PATH,
        "max_alpha_mrad": MAX_ALPHA,
        "wavelength_ang": WAVELENGTH,
        "scan_step_ang": SCAN_STEP,
        "npix": NPIX,
        "dk": float(DK),
        "aberrations": ABERRATIONS,
        "max_order": MAX_ORDER,
        "device_used": device,
        "fast_acbf_version": version,
    }
    with open(os.path.join(FIXTURE_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("  saved metadata.json")
    print(f"\nAll fixtures saved to: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
