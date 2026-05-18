#!/usr/bin/env python
"""Tile an existing HDF5 4D dataset along (Ry, Rx) to produce a larger file.

The source dataset must have shape (Ry, Rx, Ky, Kx). The output has shape
(Rep_y * Ry, Rep_x * Rx, Ky, Kx).  Data is written row-by-row (one tiled
Ry-block at a time) so peak RAM = Rep_x * source_Ry * Rx * Ky * Kx * 4 bytes.

Usage
-----
    python scripts/tile_hdf5.py \\
        --src "~/scratch/Figure 4/scan_x128_y128.hdf5" \\
        --src-key array \\
        --dst output/scan_x1024_y768_tiled.hdf5 \\
        --dst-key array \\
        --rep-y 6 --rep-x 8

    # 48 GiB file:  rep_y=6, rep_x=8  → (768, 1024, 128, 128)
    # 64 GiB file:  rep_y=8, rep_x=8  → (1024, 1024, 128, 128)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def tile_hdf5(
    src_path: str | Path,
    dst_path: str | Path,
    rep_y: int,
    rep_x: int,
    src_key: str = 'array',
    dst_key: str = 'array',
) -> None:
    src_path = Path(src_path).expanduser()
    dst_path = Path(dst_path).expanduser()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, 'r') as src_f:
        src_ds = src_f[src_key]
        src_shape = src_ds.shape
        if src_ds.ndim != 4:
            raise ValueError(f"Expected 4D source dataset, got shape {src_shape}")
        Ry, Rx, Ky, Kx = src_shape
        dtype = src_ds.dtype

        # Load source once; it is 1 GiB, well within RAM.
        print(f"Loading source: {src_path}  key={src_key!r}  shape={src_shape}  "
              f"dtype={dtype}  size={src_ds.nbytes / 1e9:.3f} GB")
        t_load = time.perf_counter()
        src_arr = src_ds[:]  # (Ry, Rx, Ky, Kx)
        load_time = time.perf_counter() - t_load
        print(f"  Loaded in {load_time:.2f} s")

    out_Ry = rep_y * Ry
    out_Rx = rep_x * Rx
    out_shape = (out_Ry, out_Rx, Ky, Kx)
    out_bytes = out_Ry * out_Rx * Ky * Kx * np.dtype(np.float32).itemsize
    print(f"\nOutput: {dst_path}")
    print(f"  shape={out_shape}  rep=({rep_y}, {rep_x})")
    print(f"  size={out_bytes / 1e9:.3f} GB  ({out_bytes / (1024**3):.3f} GiB)")

    # Build one tiled row-block (Rep_x * source_Ry rows) at a time to bound RAM.
    # Peak extra RAM per row-block = rep_x * Ry * Rx * Ky * Kx * 4 bytes.
    row_block = np.tile(src_arr, (1, rep_x, 1, 1)).astype(np.float32, copy=False)
    row_block_bytes = row_block.nbytes
    print(f"  row-block RAM: {row_block_bytes / 1e9:.2f} GB per write step  ({rep_y} steps)")

    t_write = time.perf_counter()
    with h5py.File(dst_path, 'w') as dst_f:
        ds = dst_f.create_dataset(
            dst_key,
            shape=out_shape,
            dtype=np.float32,
            chunks=None,          # contiguous layout — fastest sequential reads
        )
        for iy in range(rep_y):
            row_start = iy * Ry
            row_end = row_start + Ry
            ds[row_start:row_end, :, :, :] = row_block
            elapsed = time.perf_counter() - t_write
            pct = (iy + 1) / rep_y * 100
            print(f"  wrote rows [{row_start:>5}:{row_end:>5}]  "
                  f"{pct:5.1f}%  elapsed={elapsed:.1f}s", flush=True)

    total_time = time.perf_counter() - t_write
    write_speed = out_bytes / 1e9 / total_time
    print(f"\nDone: {dst_path}")
    print(f"  Total write time: {total_time:.1f} s  ({write_speed:.2f} GB/s)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--src', required=True, help='Source HDF5 file path')
    parser.add_argument('--src-key', default='array', help='HDF5 dataset key in source')
    parser.add_argument('--dst', required=True, help='Destination HDF5 file path')
    parser.add_argument('--dst-key', default='array', help='HDF5 dataset key in destination')
    parser.add_argument('--rep-y', type=int, required=True, help='Tiling repeat count along Ry')
    parser.add_argument('--rep-x', type=int, required=True, help='Tiling repeat count along Rx')
    args = parser.parse_args()

    tile_hdf5(
        src_path=args.src,
        dst_path=args.dst,
        rep_y=args.rep_y,
        rep_x=args.rep_x,
        src_key=args.src_key,
        dst_key=args.dst_key,
    )


if __name__ == '__main__':
    main()
