#!/usr/bin/env python
"""Tile an existing HDF5 4D dataset along (Ry, Rx) to produce a larger file.

The source dataset must have shape (Ry, Rx, Ky, Kx). The output has shape
(Rep_y * Ry, Rep_x * Rx, Ky, Kx).

Two HDF5 chunk layouts are supported (--chunk-layout):

  contiguous  (default)
    HDF5 contiguous storage (no chunking).  Sequential scan-row I/O is
    optimal for full-file passes.  File size on disk ≈ raw data size.

  detector
    HDF5 chunks of shape (Ry_out, Rx_out, 1, 1) — one chunk per detector
    pixel, spanning all scan positions.  h5py reads handle[:,:,ky,kx] as
    a single contiguous chunk, so loading only the BF pixels (Nb out of
    Ky*Kx) requires exactly Nb chunk reads instead of a full-file scan.
    Theoretical I/O savings: Nb / (Ky*Kx) × file_size (≈ 4–5% for 25 mrad).

Usage
-----
    python scripts/tile_hdf5.py \\
        --src "~/scratch/Figure 4/scan_x128_y128.hdf5" \\
        --src-key array \\
        --dst ~/scratch/fast_acbf_large_test/scan_x1024_y768_48GiB.hdf5 \\
        --rep-y 6 --rep-x 8 --chunk-layout contiguous

    python scripts/tile_hdf5.py \\
        --src "~/scratch/Figure 4/scan_x128_y128.hdf5" \\
        --src-key array \\
        --dst ~/scratch/fast_acbf_large_test/scan_x1024_y768_48GiB_detector_chunks.hdf5 \\
        --rep-y 6 --rep-x 8 --chunk-layout detector

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
    chunk_layout: str = 'contiguous',
) -> None:
    """Tile src HDF5 dataset and write to dst with the specified chunk layout.

    Parameters
    ----------
    chunk_layout : {'contiguous', 'detector'}
        'contiguous': no HDF5 chunking (fastest sequential scan-row reads).
        'detector': chunks=(Ry_out, Rx_out, 1, 1) — one chunk per detector
            pixel spanning all scan positions; optimal for reading individual
            BF pixels via handle[:,:,ky,kx] without scanning the full file.
    """
    if chunk_layout not in ('contiguous', 'detector'):
        raise ValueError(f"chunk_layout must be 'contiguous' or 'detector', got {chunk_layout!r}")

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

        print(f"Loading source: {src_path}  key={src_key!r}  shape={src_shape}  "
              f"dtype={dtype}  size={src_ds.nbytes / 1e9:.3f} GB")
        t_load = time.perf_counter()
        src_arr = src_ds[:]  # (Ry, Rx, Ky, Kx) — 1 GiB, fits in RAM
        load_time = time.perf_counter() - t_load
        print(f"  Loaded in {load_time:.2f} s")

    out_Ry = rep_y * Ry
    out_Rx = rep_x * Rx
    out_shape = (out_Ry, out_Rx, Ky, Kx)
    out_bytes = out_Ry * out_Rx * Ky * Kx * np.dtype(np.float32).itemsize
    print(f"\nOutput: {dst_path}")
    print(f"  shape={out_shape}  rep=({rep_y},{rep_x})  chunk_layout={chunk_layout!r}")
    print(f"  size={out_bytes / 1e9:.3f} GB  ({out_bytes / (1024**3):.3f} GiB)")

    t_write = time.perf_counter()

    if chunk_layout == 'contiguous':
        _write_contiguous(src_arr, dst_path, dst_key, out_shape, rep_y, Ry, t_write)
    else:
        _write_detector_chunks(src_arr, dst_path, dst_key, out_shape, rep_y, rep_x, Ky, Kx, t_write)

    total_time = time.perf_counter() - t_write
    write_speed = out_bytes / 1e9 / total_time
    print(f"\nDone: {dst_path}")
    print(f"  Total write time: {total_time:.1f} s  ({write_speed:.2f} GB/s)")


def _write_contiguous(
    src_arr: np.ndarray,
    dst_path: Path,
    dst_key: str,
    out_shape: tuple,
    rep_y: int,
    Ry: int,
    t0: float,
) -> None:
    """Write tiled data in contiguous layout, one scan-row block at a time."""
    rep_x = out_shape[1] // src_arr.shape[1]
    row_block = np.tile(src_arr, (1, rep_x, 1, 1)).astype(np.float32, copy=False)
    print(f"  row-block RAM: {row_block.nbytes / 1e9:.2f} GB  ({rep_y} steps)")

    with h5py.File(dst_path, 'w') as dst_f:
        ds = dst_f.create_dataset(dst_key, shape=out_shape, dtype=np.float32, chunks=None)
        for iy in range(rep_y):
            row_start = iy * Ry
            row_end = row_start + Ry
            ds[row_start:row_end, :, :, :] = row_block
            elapsed = time.perf_counter() - t0
            print(f"  wrote rows [{row_start:>5}:{row_end:>5}]  "
                  f"{(iy+1)/rep_y*100:5.1f}%  elapsed={elapsed:.1f}s", flush=True)


def _write_detector_chunks(
    src_arr: np.ndarray,
    dst_path: Path,
    dst_key: str,
    out_shape: tuple,
    rep_y: int,
    rep_x: int,
    Ky: int,
    Kx: int,
    t0: float,
) -> None:
    """Write tiled data with detector-major chunks (Ry_out, Rx_out, 1, 1).

    Iterates over all Ky*Kx detector pixels.  For each pixel (ky, kx):
      - tile src_arr[:,:,ky,kx] from (Ry_src, Rx_src) to (Ry_out, Rx_out)
      - write as one contiguous HDF5 chunk
    Peak RAM: one (Ry_out, Rx_out) float32 slice at a time.
    """
    out_Ry, out_Rx = out_shape[0], out_shape[1]
    chunk_shape = (out_Ry, out_Rx, 1, 1)
    chunk_bytes = out_Ry * out_Rx * 4
    n_total = Ky * Kx
    print(f"  chunk shape: {chunk_shape}  ({chunk_bytes/1e6:.1f} MB each)  "
          f"{n_total} chunks  ({Ky}×{Kx} detector pixels)")

    with h5py.File(dst_path, 'w') as dst_f:
        ds = dst_f.create_dataset(dst_key, shape=out_shape, dtype=np.float32,
                                   chunks=chunk_shape)
        report_every = max(1, n_total // 20)
        for ky in range(Ky):
            for kx in range(Kx):
                pixel_2d = src_arr[:, :, ky, kx]               # (Ry_src, Rx_src)
                tiled = np.tile(pixel_2d, (rep_y, rep_x)).astype(np.float32, copy=False)
                ds[:, :, ky, kx] = tiled
                n_done = ky * Kx + kx + 1
                if n_done % report_every == 0 or n_done == n_total:
                    elapsed = time.perf_counter() - t0
                    print(f"  wrote pixel ({ky:>3},{kx:>3})  "
                          f"{n_done}/{n_total}  {n_done/n_total*100:5.1f}%  "
                          f"elapsed={elapsed:.1f}s", flush=True)


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
    parser.add_argument(
        '--chunk-layout', default='contiguous', choices=['contiguous', 'detector'],
        help="HDF5 chunk layout: 'contiguous' (no chunking, default) or "
             "'detector' (one chunk per detector pixel spanning all scan positions; "
             "optimal for reading individual BF pixels without a full-file scan)",
    )
    args = parser.parse_args()

    tile_hdf5(
        src_path=args.src,
        dst_path=args.dst,
        rep_y=args.rep_y,
        rep_x=args.rep_x,
        src_key=args.src_key,
        dst_key=args.dst_key,
        chunk_layout=args.chunk_layout,
    )


if __name__ == '__main__':
    main()
