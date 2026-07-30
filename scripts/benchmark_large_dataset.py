#!/usr/bin/env python
"""Benchmark the reconstruction pipeline on datasets larger than VRAM and RAM.

Covers:
  - 48 GiB dataset (768, 1024, 128, 128): attempt full materialization + lazy host cache
  - 64 GiB dataset (1024, 1024, 128, 128): lazy host cache and no persistent cache
  - Contiguous and detector-chunked HDF5 layouts for each

Times reported per step:
  1. Dataset4D init (file open / materialize)
  2. BFSolver init (geometry, BF extraction)
  3. First reconstruction (fills host cache from disk)
  4. Warm-up reconstruction (cache hot — pure compute, no disk I/O)

Physics parameters (80 kV, 25 mrad convergence):
  wavelength = 0.04176 Å, max_alpha = 25 mrad, dk = 0.04 Å⁻¹/px, scan_step = 0.43 Å

This is an internal dev/benchmarking script, not runnable out of the box:
it requires 48-64 GiB HDF5 files that are not distributed with this repo
and reads/writes under the maintainer's local `~/scratch/` directory.
Adjust SCRATCH below (or generate matching files yourself) before running.

Usage
-----
    # Generate the large files first (see tile_hdf5.py).
    # Then run this benchmark:
    conda run -n fast-acbf python scripts/benchmark_large_dataset.py
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH = Path.home() / 'scratch' / 'fast_acbf_large_test'
FILE_48GiB         = SCRATCH / 'scan_x1024_y768_48GiB.hdf5'
FILE_48GiB_DET     = SCRATCH / 'scan_x1024_y768_48GiB_detector_chunks.hdf5'
FILE_64GiB         = SCRATCH / 'scan_x1024_y1024_64GiB.hdf5'
FILE_64GiB_DET     = SCRATCH / 'scan_x1024_y1024_64GiB_detector_chunks.hdf5'
HDF5_KEY = 'array'

# ── Physics parameters (80 kV, 25 mrad convergence) ───────────────────────────
WAVELENGTH = 0.04176   # Å  (80 kV)
MAX_ALPHA  = 25.0      # mrad — BF convergence semi-angle
DK         = 0.04      # Å⁻¹/px
SCAN_STEP  = 0.43      # Å
DEVICE     = 'cuda'


# ── Helpers ────────────────────────────────────────────────────────────────────

def hdr(title: str) -> None:
    bar = '─' * 70
    print(f'\n{bar}')
    print(f'  {title}')
    print(bar)


OUTPUT_DIR = Path(__file__).parent.parent / 'output'

def make_solver(
    ds,
    imagefft_storage: str,
    extractor_strategy: str = 'auto',
    pipeline: str = 'balanced',
):
    from fast_acbf import BFSolver
    return BFSolver(
        dataset=ds,
        max_alpha=MAX_ALPHA,
        scan_step_size=SCAN_STEP,
        dk=DK,
        wavelength=WAVELENGTH,
        max_order=2,
        aberrations={'C10': 80.0},
        coord_transform={'flipud': True},
        device=DEVICE,
        pipeline=pipeline,
        imagefft_storage=imagefft_storage,
        extractor_strategy=extractor_strategy,
    )


def save_tiff(img: 'torch.Tensor', label: str) -> None:
    """Save a (Ry, Rx) reconstruction image as float32 TIFF under output/."""
    try:
        import tifffile
        import numpy as np
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = OUTPUT_DIR / f'benchmark_{label}.tiff'
        arr = img.detach().cpu().numpy().astype(np.float32)
        tifffile.imwrite(str(filename), arr)
        print(f'    Saved: {filename}')
    except ImportError:
        print('    tifffile not installed — skipping TIFF save')


def print_system_info() -> None:
    import psutil
    mem = psutil.virtual_memory()
    print(f'  RAM  total={mem.total/2**30:.1f} GiB  available={mem.available/2**30:.1f} GiB')
    if torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        print(f'  VRAM total={total_vram/2**30:.1f} GiB  free={free_vram/2**30:.1f} GiB')
    print(f'  Physics: wavelength={WAVELENGTH} Å  max_alpha={MAX_ALPHA} mrad  '
          f'dk={DK} Å⁻¹/px  scan_step={SCAN_STEP} Å')
    print(f'  Device: {DEVICE}')


# ── Scenario runners ───────────────────────────────────────────────────────────

def run_materialized_oom(results: dict) -> None:
    """48 GiB: expected-OOM attempt at full RAM materialization."""
    from fast_acbf.data.dataset4d import Dataset4D

    hdr('48 GiB contiguous — materialize=True (expected OOM)')
    label = '48GiB-contiguous-materialized'
    file_size_gib = FILE_48GiB.stat().st_size / 2**30
    print(f'  File: {FILE_48GiB}  ({file_size_gib:.1f} GiB)')

    t0 = time.perf_counter()
    try:
        ds = Dataset4D.from_hdf5(FILE_48GiB, key=HDF5_KEY, materialize=True)
        elapsed = time.perf_counter() - t0
        print(f'  Materialized in {elapsed:.2f} s  shape={ds.scan_shape + ds.detector_shape}')
        results[label] = {'note': 'materialized (unexpected)'}
    except RuntimeError as exc:
        elapsed = time.perf_counter() - t0
        print(f'  OOM after {elapsed:.2f} s (expected): {exc}')
        results[label] = {'error': 'OOM-materialize'}


def run_lazy_tcbf(
    path: Path,
    imagefft_storage: str,
    results: dict,
    extractor_strategy: str = 'auto',
    pipeline: str = 'balanced',
) -> None:
    """Lazy load + tcBF reconstruction: time init, cache fill, warm-up."""
    from fast_acbf.data.dataset4d import Dataset4D

    file_size_gib = path.stat().st_size / 2**30
    layout = 'detector-chunks' if 'detector_chunks' in path.name else 'contiguous'
    label = f'{file_size_gib:.0f}GiB-{layout}-{imagefft_storage}'

    hdr(
        f'{file_size_gib:.0f} GiB {layout} — lazy + pipeline={pipeline} '
        f'+ imagefft_storage={imagefft_storage}'
    )
    print(f'  File: {path}  ({file_size_gib:.1f} GiB)')
    print(f'  pipeline: {pipeline!r}')
    print(f'  extractor_strategy: {extractor_strategy!r}')

    # Step 1: Dataset4D init
    print('\n  Step 1: Dataset4D init')
    t0 = time.perf_counter()
    ds = Dataset4D.from_hdf5(path, key=HDF5_KEY, materialize=False)
    t_init = time.perf_counter() - t0
    print(f'    {t_init:.3f} s  shape={ds.scan_shape + ds.detector_shape}  '
          f'is_lazy={ds.is_lazy}  chunks={ds.backend_chunks!r}')

    # Step 2: BFSolver init
    print('\n  Step 2: BFSolver init')
    t0 = time.perf_counter()
    try:
        solver = make_solver(
            ds,
            imagefft_storage=imagefft_storage,
            extractor_strategy=extractor_strategy,
            pipeline=pipeline,
        )
        t_solver = time.perf_counter() - t0
        Nb = solver._recon.imagefft.nb
        Ry, Rx = ds.scan_shape
        fft_cache_gib = Nb * Ry * Rx * 8 / 2**30
        useful_gib = Nb * Ry * Rx * 4 / 2**30
        print(f'    {t_solver:.3f} s  Nb={Nb}  scan={Ry}×{Rx}  '
              f'FFT {fft_cache_gib:.2f} GiB  useful BF data={useful_gib:.2f} GiB')
        print(f'    resolved: storage={solver.imagefft_storage!r}  '
              f'fill={solver.imagefft_fill!r}  extractor={solver.extractor_strategy!r}')
    except RuntimeError as exc:
        t_solver = time.perf_counter() - t0
        print(f'    FAILED {t_solver:.2f} s: {exc}')
        ds.close()
        results[label] = {'error': str(exc)}
        return

    # Step 3: first tcBF reconstruction (fills host cache if applicable)
    print('\n  Step 3: tcBF first reconstruction (fills cache)')
    t0 = time.perf_counter()
    try:
        img = solver.reconstruct(mode='tcBF', requires_grad=False)
        t_first = time.perf_counter() - t0
        print(f'    Done in {t_first:.2f} s  shape={tuple(img.shape)}')
        save_tiff(img, label.replace('/', '_'))
    except Exception as exc:
        t_first = time.perf_counter() - t0
        print(f'    FAILED after {t_first:.2f} s: {exc}')
        ds.close()
        results[label] = {'error': str(exc), 'init_s': t_init, 'solver_s': t_solver}
        return

    # Step 4: warm-up reconstruction (cache hot, no disk I/O)
    t_warm = None
    if solver.imagefft_storage != 'none':
        print('\n  Step 4: tcBF warm-up reconstruction (cache hot)')
        t0 = time.perf_counter()
        solver.reconstruct(mode='tcBF', requires_grad=False)
        t_warm = time.perf_counter() - t0
        print(f'    Done in {t_warm:.2f} s')
    else:
        print('\n  Step 4: warm-up SKIPPED (none has no cache)')

    ds.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results[label] = {
        'extractor_strategy': solver.extractor_strategy,
        'pipeline': solver.pipeline,
        'imagefft_storage': solver.imagefft_storage,
        'imagefft_fill': solver.imagefft_fill,
        'init_s': t_init,
        'solver_s': t_solver,
        'tcBF_first_s': t_first,
        'tcBF_warm_s': t_warm,
        'Nb': Nb,
        'scan': f'{Ry}x{Rx}',
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    hdr('Large-Dataset Reconstruction Benchmark')
    print_system_info()

    for f in [FILE_48GiB, FILE_48GiB_DET, FILE_64GiB, FILE_64GiB_DET]:
        if not f.exists():
            print(f'\n  MISSING: {f.name} — run tile_hdf5.py to generate it')

    results: dict = {}

    # Scenario A: 48 GiB contiguous, expected OOM materialize
    if FILE_48GiB.exists():
        run_materialized_oom(results)
        gc.collect()

    # Scenario B: 48 GiB contiguous, lazy + host
    if FILE_48GiB.exists():
        run_lazy_tcbf(FILE_48GiB, 'host', results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scenario C: 48 GiB detector-chunked, lazy + host
    if FILE_48GiB_DET.exists():
        run_lazy_tcbf(FILE_48GiB_DET, 'host', results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scenario D: 64 GiB contiguous, lazy + host
    if FILE_64GiB.exists():
        run_lazy_tcbf(FILE_64GiB, 'host', results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scenario E: 64 GiB detector-chunked, lazy + host
    if FILE_64GiB_DET.exists():
        run_lazy_tcbf(FILE_64GiB_DET, 'host', results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scenario F: 64 GiB contiguous, lazy + none
    if FILE_64GiB.exists():
        run_lazy_tcbf(FILE_64GiB, 'none', results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    hdr('Timing Summary')
    col = 44
    print(f'  {"Scenario":<{col}}  {"tcBF-first":>12}  {"tcBF-warm":>12}  {"storage":>8}  {"fill":>12}  {"extractor_strategy":>16}')
    print(f'  {"-"*col}  {"-"*12}  {"-"*12}  {"-"*8}  {"-"*12}  {"-"*16}')
    for scenario, r in results.items():
        if r.get('error') == 'OOM-materialize':
            tf = tw = 'OOM'
            lm = '-'
            storage = fill = '-'
        else:
            tf = f'{r.get("tcBF_first_s", float("nan")):.2f} s'
            tw_val = r.get('tcBF_warm_s')
            tw = f'{tw_val:.2f} s' if tw_val is not None else 'skipped'
            lm = r.get('extractor_strategy', '-')
            storage = r.get('imagefft_storage', '-')
            fill = r.get('imagefft_fill', '-')
        print(f'  {scenario:<{col}}  {tf:>12}  {tw:>12}  {storage:>8}  {fill:>12}  {lm:>16}')
    print()


if __name__ == '__main__':
    main()
