#!/usr/bin/env python
"""Benchmark I/O throughput for loading BF images from 4D-STEM HDF5 files.

Measures the time to load all Nb BF detector pixels into a (Nb, Ry, Rx) float32
buffer — the bottleneck step before GPU FFT in the host-cache prefill path.

Two file layouts compared:
  contiguous      HDF5 contiguous storage (no chunking).
  detector-chunks HDF5 chunks = (Ry_out, Rx_out, 1, 1) — one chunk per detector
                  pixel spanning all scan positions.

Two loading strategies:
  sequential   stream_all_bf_images():          reads one scan row at a time.
               I/O = full file (48–64 GiB); optimal for contiguous files.
  per_pixel    stream_all_bf_images_per_pixel(): reads one h5py chunk per BF pixel.
               I/O = Nb/（Ky*Kx) × file (~4% for 25 mrad); optimal for detector chunks.

The benchmark also runs through all three legacy lazy_read_mode values
(per_pixel, scan_row, slab) via get_bf_chunk() on the 1 GiB file so the
single-call vs chunked-loop trade-offs remain visible.

Usage
-----
    conda run -n fast-acbf python scripts/benchmark_lazy_loading.py

    # Test specific lazy mode on a single file (legacy interface):
    conda run -n fast-acbf python scripts/benchmark_lazy_loading.py \\
        --file ~/scratch/fast_acbf_large_test/scan_x128_y128_detector_chunks.hdf5 \\
        --lazy-mode per_pixel
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH = Path.home() / 'scratch' / 'fast_acbf_large_test'
SRC_FILE = Path.home() / 'scratch' / 'Figure 4' / 'scan_x128_y128.hdf5'
FILES: dict[str, Path] = {
    '1GiB-contiguous':       SRC_FILE,
    '1GiB-detector-chunks':  SCRATCH / 'scan_x128_y128_detector_chunks.hdf5',
    '48GiB-contiguous':      SCRATCH / 'scan_x1024_y768_48GiB.hdf5',
    '48GiB-detector-chunks': SCRATCH / 'scan_x1024_y768_48GiB_detector_chunks.hdf5',
    '64GiB-contiguous':      SCRATCH / 'scan_x1024_y1024_64GiB.hdf5',
    '64GiB-detector-chunks': SCRATCH / 'scan_x1024_y1024_64GiB_detector_chunks.hdf5',
}
HDF5_KEY = 'array'

# ── Physics (80 kV, 25 mrad) ──────────────────────────────────────────────────
WAVELENGTH = 0.04176   # Å
MAX_ALPHA  = 25.0      # mrad
DK         = 0.04      # Å⁻¹/px


# ── Helpers ────────────────────────────────────────────────────────────────────

def hdr(title: str) -> None:
    bar = '─' * 70
    print(f'\n{bar}')
    print(f'  {title}')
    print(bar)


def get_bf_indices(ds) -> tuple[np.ndarray, np.ndarray]:
    from fast_acbf.data.geometry import DetectorGeometry
    Ky, Kx = ds.detector_shape
    geom = DetectorGeometry(Ky=Ky, Kx=Kx, max_alpha=MAX_ALPHA, dk=DK, wavelength=WAVELENGTH)
    bf_iy, bf_ix = np.where(geom.bf_mask_bool)
    return bf_iy, bf_ix


def measure_seq_read_speed(path: Path, max_bytes: int = 512 * 2**20) -> float:
    """Sequential read speed in GB/s, reading up to max_bytes from path."""
    n = min(max_bytes, path.stat().st_size)
    buf = bytearray(8 * 2**20)
    total, t0 = 0, time.perf_counter()
    with open(path, 'rb') as f:
        while total < n:
            read = f.readinto(buf)
            if not read:
                break
            total += read
    return total / 1e9 / (time.perf_counter() - t0)


# ── Prefill benchmarks (stream_all_bf_images* paths) ──────────────────────────

def bench_prefill(name: str, path: Path, results: list) -> None:
    """Benchmark stream_all_bf_images and stream_all_bf_images_per_pixel."""
    from fast_acbf.data.dataset4d import Dataset4D

    file_gib = path.stat().st_size / 2**30
    print(f'\n  {name}  ({file_gib:.1f} GiB)')

    ds = Dataset4D.from_hdf5(path, key=HDF5_KEY, materialize=False)
    lrm = ds.lazy_read_mode
    bf_iy, bf_ix = get_bf_indices(ds)
    Nb = len(bf_iy)
    Ry, Rx = ds.scan_shape
    Ky, Kx = ds.detector_shape
    bf_frac = Nb / (Ky * Kx)
    useful_gib = Nb * Ry * Rx * 4 / 2**30

    print(f'    lazy_read_mode={lrm!r}  Nb={Nb} ({bf_frac*100:.1f}% of detector)  '
          f'useful={useful_gib:.2f} GiB')

    # Determine which strategies make sense for this layout
    large_scan = (Ry * Rx > 65_536)
    strategies: list[tuple[str, str, float]] = []  # (strategy, label, io_gib)

    # sequential: always available; catastrophic for detector-chunked on large scans
    seq_io_gib = Ry * Rx * Ky * Kx * 4 / 2**30
    strategies.append(('sequential', 'stream_all_bf_images', seq_io_gib))

    # per_pixel: always available; catastrophic for contiguous on large scans
    pp_io_gib = useful_gib
    strategies.append(('per_pixel', 'stream_all_bf_images_per_pixel', pp_io_gib))

    for strat, fn_name, io_gib in strategies:
        is_bad = (
            (strat == 'per_pixel' and lrm != 'per_pixel' and large_scan) or
            (strat == 'sequential' and lrm == 'per_pixel' and large_scan)
        )
        tag = ' [SLOW — wrong layout]' if is_bad else ''
        print(f'    [{fn_name}]{tag}  '
              f'expected I/O={io_gib:.2f} GiB  ', end='', flush=True)

        if is_bad and large_scan:
            # Skip catastrophically slow combos on large files
            print('SKIPPED')
            continue

        t0 = time.perf_counter()
        if strat == 'sequential':
            raw = ds.stream_all_bf_images(bf_iy, bf_ix)
        else:
            raw = ds.stream_all_bf_images_per_pixel(bf_iy, bf_ix)
        elapsed = time.perf_counter() - t0
        throughput = io_gib / 1.024**3 / elapsed  # GB/s (not GiB/s)

        print(f'{elapsed:.2f} s  →  {throughput:.2f} GB/s')
        del raw
        gc.collect()

        results.append({
            'file': name, 'file_gib': file_gib, 'layout': lrm,
            'strategy': strat, 'fn': fn_name,
            'io_gib': io_gib, 'elapsed_s': elapsed, 'throughput_gbs': throughput,
            'Nb': Nb, 'scan': f'{Ry}x{Rx}',
        })

    ds.close()
    gc.collect()


# ── Legacy single-file mode benchmark (get_bf_chunk with lazy_read_mode) ──────

def bench_single_file_modes(path: Path, modes: list[str], runs: int) -> None:
    """Benchmark get_bf_chunk() across explicit lazy_read_mode values."""
    from fast_acbf.data.dataset4d import Dataset4D

    hdr(f'Single-file mode comparison: {path.name}')
    import h5py
    with h5py.File(path, 'r') as f:
        shape = f[HDF5_KEY].shape
        chunks = f[HDF5_KEY].chunks
    Ry, Rx, Ky, Kx = shape
    file_gib = Ry * Rx * Ky * Kx * 4 / 2**30
    print(f'  shape={shape}  chunks={chunks}  ({file_gib:.2f} GiB)')

    ds_tmp = Dataset4D.from_hdf5(path, key=HDF5_KEY, materialize=False)
    bf_iy, bf_ix = get_bf_indices(ds_tmp)
    Nb = len(bf_iy)
    ds_tmp.close()
    print(f'  Nb={Nb}  ({100*Nb/(Ky*Kx):.1f}% of detector)\n')

    for mode in modes:
        ds = Dataset4D.from_hdf5(path, key=HDF5_KEY, lazy_read_mode=mode, materialize=False)
        times = []
        for r in range(runs):
            gc.collect()
            t0 = time.perf_counter()
            raw = ds.get_bf_chunk(bf_iy, bf_ix)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            del raw
        ds.close()
        gc.collect()
        best = min(times)
        mean = sum(times) / len(times)
        print(f'  {mode:<12}  best={best:.3f}s  mean={mean:.3f}s  '
              f'({runs} run{"s" if runs>1 else ""})')


# ── Main ────────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--file', default=None,
                        help='Single file to benchmark (enables legacy mode-comparison)')
    parser.add_argument('--lazy-mode', nargs='+',
                        default=['per_pixel', 'scan_row', 'slab'],
                        help='lazy_read_mode values to test in single-file mode')
    parser.add_argument('--runs', type=int, default=2,
                        help='Timing repetitions per scenario')
    args = parser.parse_args(argv)

    import psutil
    hdr('Lazy Loading I/O Benchmark')
    mem = psutil.virtual_memory()
    print(f'  RAM  total={mem.total/2**30:.1f} GiB  available={mem.available/2**30:.1f} GiB')
    print(f'  Physics: wavelength={WAVELENGTH} Å  max_alpha={MAX_ALPHA} mrad  dk={DK} Å⁻¹/px')

    if args.file:
        # Legacy single-file mode comparison
        bench_single_file_modes(Path(args.file).expanduser(), args.lazy_mode, args.runs)
        return

    # ── Multi-file prefill benchmark ───────────────────────────────────────────
    available = {k: v for k, v in FILES.items() if v.exists()}
    missing   = {k: v for k, v in FILES.items() if not v.exists()}
    if missing:
        print(f'\n  Missing files (run tile_hdf5.py --chunk-layout detector to generate):')
        for k, v in missing.items():
            print(f'    {k}: {v.name}')

    # NVMe sequential read baseline
    hdr('NVMe Sequential Read Baseline (512 MiB sample)')
    for path in available.values():
        if path.stat().st_size > 512 * 2**20:
            gbs = measure_seq_read_speed(path)
            print(f'  {gbs:.2f} GB/s  (from {path.name})')
            break

    hdr('Prefill Benchmark: stream_all_bf_images vs stream_all_bf_images_per_pixel')
    results: list[dict] = []
    for name, path in available.items():
        bench_prefill(name, path, results)

    # ── Summary table ──────────────────────────────────────────────────────────
    hdr('Summary')
    col = 28
    print(f'  {"File":<{col}}  {"Strategy":>22}  {"I/O(GiB)":>9}  {"t(s)":>7}  {"GB/s":>7}')
    print(f'  {"-"*col}  {"-"*22}  {"-"*9}  {"-"*7}  {"-"*7}')
    for r in results:
        print(f'  {r["file"]:<{col}}  {r["fn"]:>22}  '
              f'{r["io_gib"]:>9.2f}  {r["elapsed_s"]:>7.2f}  {r["throughput_gbs"]:>7.2f}')

    # Speedup summary for matching file sizes
    print()
    seen = set()
    for r in results:
        if r['strategy'] != 'per_pixel' or 'detector' not in r['file']:
            continue
        base = r['file'].replace('-detector-chunks', '-contiguous')
        if base in seen:
            continue
        seq = next((x for x in results if x['file'] == base and x['strategy'] == 'sequential'), None)
        if not seq:
            continue
        seen.add(base)
        speedup = seq['elapsed_s'] / r['elapsed_s']
        io_saved = seq['io_gib'] / r['io_gib']
        print(f'  {base.split("-", 1)[0]}: detector-chunks+per_pixel  '
              f'{speedup:.1f}× faster,  {io_saved:.0f}× less I/O  '
              f'({r["throughput_gbs"]:.2f} vs {seq["throughput_gbs"]:.2f} GB/s)')
    print()


if __name__ == '__main__':
    main()
