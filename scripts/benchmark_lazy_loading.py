#!/usr/bin/env python
"""Benchmark lazy loading strategies for Dataset4D on a real HDF5 file.

Tests three on-the-fly streaming approaches (per_pixel, scan_row, slab) against
the host (RAM-materialised) baseline.  Each approach loads all N_bf virtual-BF
images (one iteration of all chunks = full vBF pass).

Two timing scenarios
--------------------
1. **full_pass** – single ``get_bf_chunk(all_iy, all_ix)`` call with all N_bf
   detector pixels at once.  Fair apples-to-apples for total I/O time.

2. **chunked** – iterate through ceil(N_bf / chunk_size) calls of size
   ``chunk_size``.  Realistic pipeline scenario.  scan_row and slab are shown
   but expect to be much slower here (each call re-reads the whole
   dataset / slab).

Usage
-----
    conda run -n fast-acbf python scripts/benchmark_lazy_loading.py

Optional args::

    --file    path/to/file.hdf5 (default: ~/scratch/Figure 4/scan_x128_y128.hdf5)
    --key     HDF5 dataset key (default: array)
    --chunk-size  BF pixels per call for chunked scenario (default: 64)
    --runs    timing repetitions per scenario (default: 2)
    --max-alpha   BF aperture semi-angle in mrad (default: 25.0)
    --wavelength  electron wavelength in Å (default: 0.04176)
    --dk      detector pixel size in Å⁻¹/pixel (default: auto-estimate)
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bf_pixels(Ky: int, Kx: int, max_alpha: float, wavelength: float, dk: float):
    ky = np.fft.fftshift(np.fft.fftfreq(Ky, d=(1.0 / dk / Ky)))
    kx = np.fft.fftshift(np.fft.fftfreq(Kx, d=(1.0 / dk / Kx)))
    kX, kY = np.meshgrid(kx, ky, indexing='xy')
    bf_mask = np.sqrt(kX**2 + kY**2) <= (max_alpha / 1e3 / wavelength)
    iy, ix = np.where(bf_mask)
    return iy.astype(np.intp), ix.astype(np.intp), bf_mask


def timed(fn, label: str, runs: int):
    """Run fn() *runs* times, return list of elapsed seconds."""
    times = []
    for r in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  [{label}] run {r+1}/{runs}: {elapsed:.3f} s")
    return times, result


def fmt_row(label, times, n_bf, scan_shape, extra=""):
    mean = np.mean(times)
    best = np.min(times)
    Ry, Rx = scan_shape
    mb_out = n_bf * Ry * Rx * 4 / 1e6
    return (
        f"  {label:<28} best={best:7.3f}s  mean={mean:7.3f}s  "
        f"output={mb_out:6.0f} MB  {extra}"
    )


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

def bench_full_pass(path, key, iy_bf, ix_bf, modes, runs, per_pixel_sample: int = 0):
    """Single call with all N_bf BF pixels — pure I/O comparison.

    per_pixel_sample: if > 0, time per_pixel on that many pixels and
    extrapolate to N_bf (avoids waiting for the full catastrophically-slow pass).
    """
    from fast_acbf.data.dataset4d import Dataset4D

    print("\n=== Scenario 1: full_pass (single get_bf_chunk call, all N_bf pixels) ===")
    n_bf = len(iy_bf)
    results = {}
    for mode in modes:
        ds = Dataset4D.from_hdf5(path, key=key, lazy_read_mode=mode)
        scan_shape = ds.scan_shape

        if mode == 'per_pixel' and per_pixel_sample > 0 and per_pixel_sample < n_bf:
            sample = min(per_pixel_sample, n_bf)
            print(f"  [per_pixel] timing {sample}/{n_bf} pixels, extrapolating to full pass ...")
            times_sample, _ = timed(
                lambda: ds.get_bf_chunk(iy_bf[:sample], ix_bf[:sample]),
                f'per_pixel (sample={sample})', runs,
            )
            scale = n_bf / sample
            times = [t * scale for t in times_sample]
            print(f"  [per_pixel] extrapolated full-pass: best={min(times):.1f}s  "
                  f"(measured {min(times_sample):.3f}s for {sample} px, ×{scale:.1f})")
            is_extrapolated = True
        else:
            def _run():
                return ds.get_bf_chunk(iy_bf, ix_bf)
            times, _ = timed(_run, mode, runs)
            is_extrapolated = False

        ds.close()
        results[mode] = (times, scan_shape, is_extrapolated)
        extra = "  (extrapolated)" if is_extrapolated else ""
        print(fmt_row(mode, times, n_bf, scan_shape, extra))

    return results


def bench_host_full_pass(path, key, iy_bf, ix_bf, runs):
    """Materialize to RAM then fancy-index — baseline for RAM speed."""
    from fast_acbf.data.dataset4d import Dataset4D

    print("\n  [host] materialize + extract:")
    ds_lazy = Dataset4D.from_hdf5(path, key=key, lazy_read_mode='per_pixel')
    Ry, Rx = ds_lazy.scan_shape
    n_bf = len(iy_bf)

    times_mat = []
    times_idx = []
    for r in range(runs):
        import h5py
        gc.collect()
        # Re-open each time to avoid HDF5 read cache influencing results
        t0 = time.perf_counter()
        with h5py.File(path, 'r') as f:
            arr = np.asarray(f[key][:], dtype=np.float32)   # materialize
        t_mat = time.perf_counter() - t0

        t1 = time.perf_counter()
        raw = arr[:, :, iy_bf, ix_bf]                       # (Ry, Rx, N_bf)
        out = np.ascontiguousarray(raw.transpose(2, 0, 1))  # (N_bf, Ry, Rx)
        t_idx = time.perf_counter() - t1

        times_mat.append(t_mat)
        times_idx.append(t_idx)
        del arr, out; gc.collect()
        print(f"  [host] run {r+1}/{runs}: mat={t_mat:.3f}s  idx={t_idx:.3f}s  total={t_mat+t_idx:.3f}s")

    times_total = [m + i for m, i in zip(times_mat, times_idx)]
    ds_lazy.close()
    print(fmt_row('host (mat+idx)', times_total, n_bf, (Ry, Rx),
                  f"(best mat={min(times_mat):.3f}s idx={min(times_idx):.3f}s)"))
    return times_total


def bench_chunked(path, key, iy_bf, ix_bf, modes, chunk_size, runs):
    """Chunked iteration — realistic pipeline scenario."""
    from fast_acbf.data.dataset4d import Dataset4D

    print(f"\n=== Scenario 2: chunked (chunk_size={chunk_size}, N_bf={len(iy_bf)}) ===")
    n_bf = len(iy_bf)
    n_chunks = int(np.ceil(n_bf / chunk_size))
    print(f"  {n_chunks} chunks per full pass\n")
    results = {}

    for mode in modes:
        ds = Dataset4D.from_hdf5(path, key=key, lazy_read_mode=mode)
        scan_shape = ds.scan_shape
        out_buf = np.empty((n_bf,) + ds.scan_shape, dtype=np.float32)

        def _run():
            for b in range(n_chunks):
                b_start = b * chunk_size
                b_end = min(b_start + chunk_size, n_bf)
                out_buf[b_start:b_end] = ds.get_bf_chunk(iy_bf[b_start:b_end],
                                                          ix_bf[b_start:b_end])
            return out_buf

        times, _ = timed(_run, mode, runs)
        ds.close()
        results[mode] = (times, scan_shape)
        print(fmt_row(mode, times, n_bf, scan_shape, f"({n_chunks} calls × chunk_size={chunk_size})"))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--file',
        default=str(Path.home() / 'scratch' / 'Figure 4' / 'scan_x128_y128.hdf5'),
    )
    parser.add_argument('--key', default='array')
    parser.add_argument('--chunk-size', type=int, default=64)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--max-alpha', type=float, default=25.0,
                        help='BF aperture semi-angle in mrad')
    parser.add_argument('--wavelength', type=float, default=0.04176,
                        help='electron wavelength in Å')
    parser.add_argument('--dk', type=float, default=None,
                        help='detector pixel size in Å⁻¹/pixel (default: auto)')
    args = parser.parse_args(argv)

    import h5py
    with h5py.File(args.file, 'r') as f:
        ds = f[args.key]
        shape = ds.shape
        chunks = ds.chunks
        compression = ds.compression
    Ry, Rx, Ky, Kx = shape
    size_gb = Ry * Rx * Ky * Kx * 4 / 1e9

    # Auto-estimate dk so BF disk ≈ 20% of detector area (radius ≈ 0.253*Ky)
    if args.dk is None:
        r_pixels = 0.253 * Ky   # gives N_bf ≈ π*r² ≈ 20% of detector
        args.dk = (args.max_alpha / 1e3 / args.wavelength) / r_pixels

    iy_bf, ix_bf, _ = _bf_pixels(Ky, Kx, args.max_alpha, args.wavelength, args.dk)
    n_bf = len(iy_bf)

    print("=" * 60)
    print(f"File      : {args.file}")
    print(f"Key       : {args.key}")
    print(f"Shape     : {shape}  ({size_gb:.2f} GB)")
    print(f"Chunks    : {chunks}  compression: {compression}")
    print(f"max_alpha : {args.max_alpha} mrad   wavelength: {args.wavelength} Å")
    print(f"dk        : {args.dk:.5f} Å⁻¹/pixel")
    print(f"N_bf      : {n_bf} / {Ky * Kx} pixels  ({100*n_bf/(Ky*Kx):.1f}% of detector)")
    print(f"Output    : ({n_bf}, {Ry}, {Rx}) float32  = {n_bf*Ry*Rx*4/1e6:.0f} MB")
    print(f"chunk_size: {args.chunk_size}  runs: {args.runs}")
    print("=" * 60)

    # ---- Full-pass: all three lazy modes ----
    # per_pixel is ~N_bf × 0.1s per call; extrapolate from a small sample.
    lazy_modes = ['per_pixel', 'scan_row', 'slab']
    fp_results = bench_full_pass(
        args.file, args.key, iy_bf, ix_bf, lazy_modes, args.runs,
        per_pixel_sample=50,
    )
    host_times = bench_host_full_pass(args.file, args.key, iy_bf, ix_bf, args.runs)

    # ---- Chunked: per_pixel and slab (scan_row is impractical for small chunks) ----
    print("\n  Note: scan_row reads the full dataset (~1 GB) per get_bf_chunk call,")
    print("  so it is O(n_chunks)× worse than per_pixel for small chunk sizes.")
    chunked_modes = ['per_pixel', 'slab']
    bench_chunked(args.file, args.key, iy_bf, ix_bf, chunked_modes, args.chunk_size, args.runs)

    # ---- Summary ----
    ky_range = int(iy_bf.max()) - int(iy_bf.min()) + 1
    slab_mb = Ry * Rx * ky_range * Kx * 4 / 1e6
    data_reads = {
        'per_pixel': f"{n_bf * Ry * Rx * 4 / 1e6:.0f} MB (strided)",
        'scan_row':  f"{size_gb * 1e3:.0f} MB (sequential)",
        'slab':      f"{slab_mb:.0f} MB (1 hyperslab)",
        'host':      f"{size_gb * 1e3:.0f} MB (sequential)",
    }
    notes = {
        'per_pixel': f"~{n_bf} h5py calls, scatter-gather overhead",
        'scan_row':  f"{Ry} h5py calls, contiguous 8 MB blocks",
        'slab':      f"1 h5py call, ky-bbox slab ({ky_range}×{Kx} det)",
        'host':      "full materialize, then RAM fancy-index",
    }
    all_modes = ['per_pixel', 'scan_row', 'slab', 'host']
    all_times = {
        'per_pixel': fp_results['per_pixel'][0],
        'scan_row':  fp_results['scan_row'][0],
        'slab':      fp_results['slab'][0],
        'host':      host_times,
    }
    extrap_flag = {
        'per_pixel': fp_results['per_pixel'][2],
        'scan_row':  fp_results['scan_row'][2],
        'slab':      fp_results['slab'][2],
        'host':      False,
    }

    per_pix_best = min(all_times['per_pixel'])
    print("\n" + "=" * 60)
    print("SUMMARY — full_pass (single call, all N_bf pixels)")
    print(f"  (per_pixel marked * is extrapolated from 50-pixel sample)")
    print("=" * 60)
    print(f"  {'strategy':<20}  {'best (s)':>10}  {'mean (s)':>10}  {'data_read':>20}  speedup vs per_pixel")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*20}  {'-'*20}")
    for m in all_modes:
        t = all_times[m]
        speedup = per_pix_best / min(t)
        flag = "*" if extrap_flag[m] else " "
        print(
            f"  {m+flag:<21}  {min(t):>10.2f}  {np.mean(t):>10.2f}  "
            f"{data_reads[m]:>20}  ×{speedup:.1f}  {notes[m]}"
        )

    from fast_acbf.data.dataset4d import Dataset4D
    auto_ds = Dataset4D.from_hdf5(args.file, key=args.key, lazy_read_mode='auto')
    auto_mode = auto_ds.lazy_read_mode
    auto_ds.close()
    print(f"\nauto-detected mode for this file: {auto_mode!r}  (chunks={chunks})")
    print("=" * 60)


if __name__ == '__main__':
    main()
