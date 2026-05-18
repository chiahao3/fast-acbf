#!/usr/bin/env python
"""Stress test: reconstruction pipeline on datasets larger than VRAM and RAM.

Covers:
  - 48 GiB dataset (768, 1024, 128, 128): attempt full materialization + host cache
  - 64 GiB dataset (1024, 1024, 128, 128): lazy loading with host and on_the_fly cache

Times reported per step:
  1. Dataset4D init (file open / materialize)
  2. BFSolver init (geometry, BF extraction)
  3. First reconstruction (fills host cache if applicable)
  4. Warm-up reconstruction (cache hot, no disk I/O)
  5. AD refinement (2 iters, scan_roi=64×64)

Usage
-----
    # Generate the large files first:
    conda run -n fast-acbf python scripts/tile_hdf5.py \\
        --src "~/scratch/Figure 4/scan_x128_y128.hdf5" --src-key array \\
        --dst output/scan_x1024_y768_48GiB.hdf5 --rep-y 6 --rep-x 8

    conda run -n fast-acbf python scripts/tile_hdf5.py \\
        --src "~/scratch/Figure 4/scan_x128_y128.hdf5" --src-key array \\
        --dst output/scan_x1024_y1024_64GiB.hdf5 --rep-y 8 --rep-x 8

    # Then run this test:
    conda run -n fast-acbf python scripts/test_large_dataset_reconstruction.py

Notes
-----
- max_alpha=10 mrad is used to keep Nb ≈ 530 BF pixels and FFT host cache ≤ 5 GiB.
  Using 25 mrad (Nb ≈ 3294) would require 21-28 GiB for the host cache, which
  exceeds the ~23 GiB available RAM on this workstation.
- The 48 GiB materialization test is expected to FAIL (OOM) and is reported as such.
- AD refinement uses scan_roi=(0, 64, 0, 64) for speed; full-image refinement would
  be extremely slow on 768×1024 / 1024×1024 scans.
"""

from __future__ import annotations

import contextlib
import gc
import time
from pathlib import Path

import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
# Large test files live on local NVMe scratch (not NFS home) to avoid the
# catastrophic seek latency NFS imposes on h5py hyperslab reads.
SCRATCH = Path.home() / 'scratch' / 'fast_acbf_large_test'
FILE_48GiB = SCRATCH / 'scan_x1024_y768_48GiB.hdf5'
FILE_64GiB = SCRATCH / 'scan_x1024_y1024_64GiB.hdf5'
HDF5_KEY = 'array'

# ── Physics parameters ─────────────────────────────────────────────────────────
# dk is estimated for BF disk radius = 0.253 * Ky from the 25 mrad benchmark;
# using max_alpha=10 mrad with this dk gives Nb ≈ 530 BF pixels, keeping the
# host FFT cache ≤ 5 GiB for the large scan sizes tested here.
WAVELENGTH = 0.04176   # Å (300 kV)
MAX_ALPHA  = 10.0      # mrad  — kept small so Nb × Ry × Rx × 8 B < available RAM
DK         = (25e-3 / WAVELENGTH) / (0.253 * 128)   # Å⁻¹/pixel ≈ 0.01847
SCAN_STEP  = 0.3       # Å
DEVICE     = 'cuda'
AD_SCAN_ROI = (0, 64, 0, 64)   # 64×64 crop for fast AD refinement


# ── Helpers ────────────────────────────────────────────────────────────────────

def hdr(title: str) -> None:
    bar = '─' * 70
    print(f'\n{bar}')
    print(f'  {title}')
    print(bar)


def step(label: str) -> contextlib.AbstractContextManager:
    return _TimedStep(label)


class _TimedStep:
    def __init__(self, label: str) -> None:
        self._label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        print(f'  [{self._label}] ... ', end='', flush=True)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        self.elapsed = time.perf_counter() - self._t0
        if exc_type is None:
            print(f'{self.elapsed:.2f} s')
        else:
            print(f'FAILED after {self.elapsed:.2f} s')
        return False   # re-raise exceptions


def make_solver(ds, cache_mode: str):
    """Build BFSolver from an existing Dataset4D."""
    from fast_acbf import BFSolver
    return BFSolver(
        dataset=ds,
        max_alpha=MAX_ALPHA,
        scan_step_size=SCAN_STEP,
        dk=DK,
        wavelength=WAVELENGTH,
        max_order=2,
        aberrations={'C10': 0.0},
        device=DEVICE,
        cache_mode=cache_mode,
    )


def run_recon_pair(solver, label: str, mode: str, warmup: bool = True):
    """First reconstruction (fills cache), then warm-up reconstruction."""
    timings = {}
    t0 = time.perf_counter()
    img1 = solver.reconstruct(mode=mode, requires_grad=False)
    timings['first'] = time.perf_counter() - t0
    print(f'  [{label} {mode} first recon]  {timings["first"]:.2f} s  '
          f'image shape: {tuple(img1.shape)}  dtype: {img1.dtype}')

    if warmup:
        t0 = time.perf_counter()
        solver.reconstruct(mode=mode, requires_grad=False)
        timings['warmup'] = time.perf_counter() - t0
        print(f'  [{label} {mode} warm-up recon]  {timings["warmup"]:.2f} s')
    return timings


def run_ad_refinement(solver, label: str, mode: str, iters: int = 2):
    """Run AD-based aberration refinement on a small scan ROI."""
    t0 = time.perf_counter()
    try:
        solver.refine_aberrations(
            lr=0.1,
            iters=iters,
            metric='normalized_std',
            plot_recon_every_n_iter=None,
            save_dir=None,
            mode=mode,
            scan_roi=AD_SCAN_ROI,
        )
        elapsed = time.perf_counter() - t0
        print(f'  [{label} {mode} AD refinement ({iters} iters, roi={AD_SCAN_ROI})]  '
              f'{elapsed:.2f} s  ✓')
        return elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f'  [{label} {mode} AD refinement]  FAILED after {elapsed:.2f} s: {exc}')
        return None


def print_timing_summary(results: dict) -> None:
    hdr('Timing Summary')
    print(f'  {"Scenario":<50}  {"tcBF-first":>10}  {"tcBF-warm":>10}  '
          f'{"acBF-first":>10}  {"acBF-warm":>10}')
    print(f'  {"-"*50}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}')
    for scenario, r in results.items():
        tf = f'{r.get("tcBF_first", float("nan")):.2f}s'
        tw = f'{r.get("tcBF_warm", float("nan")):.2f}s'
        af = f'{r.get("acBF_first", float("nan")):.2f}s'
        aw = f'{r.get("acBF_warm", float("nan")):.2f}s'
        print(f'  {scenario:<50}  {tf:>10}  {tw:>10}  {af:>10}  {aw:>10}')


# ── Test scenarios ──────────────────────────────────────────────────────────────

def test_48gib_materialized_host(results: dict) -> None:
    """48 GiB: attempt full materialization into RAM, then host cache."""
    from fast_acbf.data.dataset4d import Dataset4D

    hdr('48 GiB dataset — materialize to RAM + cache_mode=host')
    label = '48GiB-materialized-host'
    file_size_gb = FILE_48GiB.stat().st_size / 1e9
    print(f'  File: {FILE_48GiB}  ({file_size_gb:.1f} GB on disk)')

    # Step 1: open and materialize
    print('\n  Step 1: Dataset4D init (materialize=True)')
    t0 = time.perf_counter()
    try:
        ds = Dataset4D.from_hdf5(FILE_48GiB, key=HDF5_KEY, materialize=True)
        elapsed_init = time.perf_counter() - t0
        print(f'    Materialized in {elapsed_init:.2f} s  '
              f'shape={ds.scan_shape + ds.detector_shape}  is_lazy={ds.is_lazy}')
    except RuntimeError as exc:
        elapsed_init = time.perf_counter() - t0
        print(f'    OOM after {elapsed_init:.2f} s (expected on 64 GiB workstation): {exc}')
        results[label] = {'error': 'OOM-materialize'}
        return

    # Step 2: BFSolver init
    print('\n  Step 2: BFSolver init (cache_mode=host)')
    t0 = time.perf_counter()
    try:
        solver = make_solver(ds, cache_mode='host')
        elapsed_solver = time.perf_counter() - t0
        print(f'    BFSolver ready in {elapsed_solver:.2f} s  Nb={solver._recon.provider.nb}')
    except RuntimeError as exc:
        elapsed_solver = time.perf_counter() - t0
        print(f'    Host cache OOM after {elapsed_solver:.2f} s: {exc}')
        results[label] = {'error': 'OOM-host-cache'}
        return

    # Steps 3–4: tcBF + acBF reconstruction
    print('\n  Step 3–4: Reconstruction')
    tcbf_t = run_recon_pair(solver, label, 'tcBF', warmup=True)
    acbf_t = run_recon_pair(solver, label, 'acBF', warmup=True)

    # Step 5: AD refinement
    print('\n  Step 5: AD refinement')
    run_ad_refinement(solver, label, 'tcBF', iters=2)
    run_ad_refinement(solver, label, 'acBF', iters=2)

    results[label] = {
        'init_s': elapsed_init,
        'solver_s': elapsed_solver,
        'tcBF_first': tcbf_t.get('first'),
        'tcBF_warm': tcbf_t.get('warmup'),
        'acBF_first': acbf_t.get('first'),
        'acBF_warm': acbf_t.get('warmup'),
    }


def test_lazy(
    path: Path,
    cache_mode: str,
    results: dict,
    lazy_read_mode: str = 'auto',
) -> None:
    """Generic lazy-loading test for any HDF5 path and cache_mode."""
    from fast_acbf.data.dataset4d import Dataset4D

    file_size_gib = path.stat().st_size / 2**30
    scan_str = 'x'.join(str(x) for x in ['?', '?'])  # filled below
    label = f'{file_size_gib:.0f}GiB-lazy-{cache_mode}'

    hdr(f'{file_size_gib:.0f} GiB dataset — lazy + cache_mode={cache_mode}')
    print(f'  File: {path}  ({file_size_gib:.1f} GiB on disk)')
    print(f'  lazy_read_mode: {lazy_read_mode!r}')

    # Step 1: Dataset4D init (lazy — no disk read yet)
    print('\n  Step 1: Dataset4D init (lazy)')
    t0 = time.perf_counter()
    ds = Dataset4D.from_hdf5(path, key=HDF5_KEY, materialize=False,
                             lazy_read_mode=lazy_read_mode)
    elapsed_init = time.perf_counter() - t0
    detected_mode = ds.lazy_read_mode
    print(f'    Ready in {elapsed_init:.3f} s  shape={ds.scan_shape + ds.detector_shape}  '
          f'is_lazy={ds.is_lazy}  detected lazy_read_mode={detected_mode!r}')

    scan_str = f'{ds.scan_shape[0]}x{ds.scan_shape[1]}'

    # Step 2: BFSolver init
    print('\n  Step 2: BFSolver init')
    t0 = time.perf_counter()
    try:
        solver = make_solver(ds, cache_mode=cache_mode)
        elapsed_solver = time.perf_counter() - t0
        Nb = solver._recon.provider.nb
        Ry, Rx = ds.scan_shape
        fft_cache_gib = Nb * Ry * Rx * 8 / 2**30
        print(f'    Ready in {elapsed_solver:.2f} s  Nb={Nb}  '
              f'scan={Ry}×{Rx}  '
              f'FFT host cache would be {fft_cache_gib:.2f} GiB')
    except RuntimeError as exc:
        elapsed_solver = time.perf_counter() - t0
        print(f'    Failed after {elapsed_solver:.2f} s: {exc}')
        ds.close()
        results[label] = {'error': str(exc)}
        return

    # on_the_fly has no cache: warm-up == first reconstruction (same I/O).
    # Skip warm-up and AD refinement for on_the_fly to keep the test finite.
    run_warmup = (cache_mode != 'on_the_fly')
    run_ad = (cache_mode != 'on_the_fly')

    # tcBF
    print('\n  Step 3: tcBF first reconstruction (cache fill if host mode)')
    t0 = time.perf_counter()
    try:
        tcbf_first = solver.reconstruct(mode='tcBF', requires_grad=False)
        t_tcbf_first = time.perf_counter() - t0
        print(f'    Done in {t_tcbf_first:.2f} s  shape={tuple(tcbf_first.shape)}')
    except Exception as exc:
        t_tcbf_first = time.perf_counter() - t0
        print(f'    FAILED after {t_tcbf_first:.2f} s: {exc}')
        ds.close()
        results[label] = {'error': str(exc)}
        return

    t_tcbf_warm = None
    if run_warmup:
        print('\n  Step 4: tcBF warm-up reconstruction (cache hot)')
        t0 = time.perf_counter()
        solver.reconstruct(mode='tcBF', requires_grad=False)
        t_tcbf_warm = time.perf_counter() - t0
        print(f'    Done in {t_tcbf_warm:.2f} s')
    else:
        print('\n  Step 4: tcBF warm-up SKIPPED (on_the_fly — no cache benefit)')

    # acBF
    print('\n  Step 5: acBF first reconstruction')
    t0 = time.perf_counter()
    try:
        acbf_first = solver.reconstruct(mode='acBF', requires_grad=False)
        t_acbf_first = time.perf_counter() - t0
        print(f'    Done in {t_acbf_first:.2f} s  shape={tuple(acbf_first.shape)}')
    except Exception as exc:
        t_acbf_first = time.perf_counter() - t0
        print(f'    FAILED after {t_acbf_first:.2f} s: {exc}')
        ds.close()
        results[label] = {
            'init_s': elapsed_init,
            'solver_s': elapsed_solver,
            'tcBF_first': t_tcbf_first,
            'tcBF_warm': t_tcbf_warm,
            'error_acbf': str(exc),
        }
        return

    t_acbf_warm = None
    if run_warmup:
        print('\n  Step 6: acBF warm-up reconstruction')
        t0 = time.perf_counter()
        solver.reconstruct(mode='acBF', requires_grad=False)
        t_acbf_warm = time.perf_counter() - t0
        print(f'    Done in {t_acbf_warm:.2f} s')
    else:
        print('\n  Step 6: acBF warm-up SKIPPED (on_the_fly — no cache benefit)')

    # AD refinement
    t_ad_tcbf = t_ad_acbf = None
    if run_ad:
        print('\n  Step 7: AD refinement (tcBF)')
        t_ad_tcbf = run_ad_refinement(solver, label, 'tcBF', iters=2)
        print('\n  Step 8: AD refinement (acBF)')
        t_ad_acbf = run_ad_refinement(solver, label, 'acBF', iters=2)
    else:
        print('\n  Step 7–8: AD refinement SKIPPED (on_the_fly — each iter re-reads disk)')

    ds.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results[label] = {
        'scan': scan_str,
        'Nb': Nb,
        'lazy_read_mode': detected_mode,
        'init_s': elapsed_init,
        'solver_s': elapsed_solver,
        'tcBF_first': t_tcbf_first,
        'tcBF_warm': t_tcbf_warm,
        'acBF_first': t_acbf_first,
        'acBF_warm': t_acbf_warm,
        'AD_tcBF_s': t_ad_tcbf,
        'AD_acBF_s': t_ad_acbf,
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    import psutil

    hdr('Large-Dataset Reconstruction Stress Test')
    mem = psutil.virtual_memory()
    print(f'  RAM total={mem.total/2**30:.1f} GiB  '
          f'available={mem.available/2**30:.1f} GiB  '
          f'used={mem.used/2**30:.1f} GiB')
    if torch.cuda.is_available():
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        print(f'  VRAM total={total_vram/2**30:.1f} GiB  free={free_vram/2**30:.1f} GiB')
    print(f'  max_alpha={MAX_ALPHA} mrad  dk={DK:.5f} Å⁻¹/px  '
          f'wavelength={WAVELENGTH} Å  scan_step={SCAN_STEP} Å')
    print(f'  AD scan_roi={AD_SCAN_ROI}  device={DEVICE}')

    for path, label in [(FILE_48GiB, '48 GiB'), (FILE_64GiB, '64 GiB')]:
        if not path.exists():
            print(f'\n  SKIP: {label} file not found: {path}')
            print(f'  Generate it with:')
            print(f'    python scripts/tile_hdf5.py \\')
            print(f'      --src "~/scratch/Figure 4/scan_x128_y128.hdf5" --src-key array \\')
            if '48' in label:
                print(f'      --dst {path} --rep-y 6 --rep-x 8')
            else:
                print(f'      --dst {path} --rep-y 8 --rep-x 8')

    results: dict = {}

    # ── Scenario A: 48 GiB, attempt full RAM load + host cache ────────────────
    if FILE_48GiB.exists():
        test_48gib_materialized_host(results)
    else:
        print(f'\nSKIP: 48 GiB file not found')

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Scenario B: 48 GiB lazy + host cache ──────────────────────────────────
    if FILE_48GiB.exists():
        test_lazy(FILE_48GiB, 'host', results)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Scenario C: 64 GiB lazy + host cache ──────────────────────────────────
    if FILE_64GiB.exists():
        test_lazy(FILE_64GiB, 'host', results)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Scenario D: 64 GiB lazy + on_the_fly ──────────────────────────────────
    if FILE_64GiB.exists():
        test_lazy(FILE_64GiB, 'on_the_fly', results)

    # ── Summary ────────────────────────────────────────────────────────────────
    hdr('Full Results')
    for label, r in results.items():
        print(f'\n  [{label}]')
        for k, v in r.items():
            if isinstance(v, float):
                print(f'    {k:<25} {v:.3f} s' if k.endswith('_s') or k.endswith('_first')
                      or k.endswith('_warm') else f'    {k:<25} {v:.3f}')
            else:
                print(f'    {k:<25} {v}')

    print_timing_summary(results)
    print()


if __name__ == '__main__':
    main()
