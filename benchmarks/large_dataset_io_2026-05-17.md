# Large-Dataset I/O Benchmark (2026-05-17)

**Hardware**: NVIDIA RTX 5000 Ada (31.5 GiB VRAM) · SK Hynix PC811 1 TB NVMe (`/dev/nvme0n1`, ext4)  
**System RAM**: 62 GiB total, ~22 GiB available at test time  
**Physics**: 80 kV (λ = 0.04176 Å), max_α = 25 mrad, dk = 0.04 Å⁻¹/px, scan step = 0.43 Å  
**Scripts**: `scripts/benchmark_large_dataset.py`, `scripts/benchmark_lazy_loading.py`

---

## 1. Motivation

Standard `Dataset4D` materialization loads the full 4D array into RAM before any
reconstruction.  For datasets larger than system RAM (or VRAM), this raises an OOM
error.  The lazy-loading pipeline streams data from HDF5 on demand and caches only
the BF-pixel FFTs needed for reconstruction (`imagefft_storage='host'`).

This benchmark characterises the end-to-end pipeline for datasets from 48 GiB to
64 GiB under different HDF5 chunk layouts and cache strategies.

---

## 2. Test Files

| Label | Shape (Ry, Rx, Ky, Kx) | Size | HDF5 layout | Chunk shape |
|---|---|---|---|---|
| 48 GiB contiguous | (768, 1024, 128, 128) | 48 GiB | contiguous | — |
| 48 GiB detector-chunks | (768, 1024, 128, 128) | 48 GiB | detector-major | (768, 1024, 1, 1) |
| 64 GiB contiguous | (1024, 1024, 128, 128) | 64 GiB | contiguous | — |
| 64 GiB detector-chunks | (1024, 1024, 128, 128) | 64 GiB | detector-major | (1024, 1024, 1, 1) |

Detector-chunked files are generated with `scripts/tile_hdf5.py --chunk-layout detector`.
Each HDF5 chunk contains the full scan image for one detector pixel `(ky, kx)`, so a
single h5py call `handle[:, :, ky, kx]` fetches exactly that chunk — no extra data
is read from disk.

---

## 3. Reconstruction Timing Results

All scenarios use `tcBF`, `max_order=2`, `aberrations={'C10': 80}`, `coord_transform={'flipud': True}`.

| Scenario | First recon | Warm recon | extractor_strategy | Notes |
|---|---|---|---|---|
| 48 GiB contiguous — **materialize** | OOM | OOM | — | Needs 48 GiB free RAM |
| 48 GiB contiguous — **host cache** | 41.2 s | 2.35 s | slab | Full file streamed once |
| 48 GiB detector-chunks — **host cache** | 8.35 s | 2.37 s | per_pixel | Only 2.04 GiB read |
| 64 GiB contiguous — **host cache** | 56.5 s | 3.20 s | slab | Full file streamed once |
| 64 GiB detector-chunks — **host cache** | 11.5 s | 3.34 s | per_pixel | Only 2.72 GiB read |
| 64 GiB contiguous — **none** | 273 s | — | slab | Re-reads disk every pass |

"First recon" fills the host FFT cache from disk.  "Warm recon" reads only from RAM
(the pre-filled cache) and reflects pure GPU compute cost.

---

## 4. I/O Speed Analysis

### 4.1 True NVMe speed

```
dd if=<64GiB file> of=/dev/null bs=128M iflag=direct   →  6.7 GB/s
```

`O_DIRECT` bypasses the OS page cache entirely (DMA straight to/from NVMe).
This is the hardware ceiling.

### 4.2 OS buffered I/O ceiling (file >> RAM)

```
dd if=<64GiB file> of=/dev/null bs=128M               →  2.6 GB/s
```

With 22 GiB free RAM and a 64 GiB file, the kernel continuously evicts pages as
new data arrives.  Every byte crosses the memory bus twice (NVMe→page cache, page
cache→user buffer), and the eviction bookkeeping adds overhead.  This sets the
practical ceiling for any buffered reader on this machine for large files.

### 4.3 h5py sequential reads

```
h5py handle[y, :, :, :]  for y in range(Ry)           →  1.55 GB/s
(batch size 1, 8, or 64 rows makes no measurable difference)
```

~40% slower than raw buffered dd.  The overhead is the HDF5 C-library (hyperslab
offset computation, filter pipeline, dataset object cache) plus the h5py Python
wrapper, all incurred once per `__getitem__` call.  Crucially, the per-call overhead
does **not** dominate — batching rows gives no speedup — so the bottleneck is the
buffered I/O path itself, not Python call frequency.

### 4.4 Summary

| Reader | Speed | Bottleneck |
|---|---|---|
| `dd` + `O_DIRECT` | 6.7 GB/s | Hardware limit |
| `dd` buffered | 2.6 GB/s | Page-cache eviction (file >> RAM) |
| BFExtractor `disk_scan_row` | ~1.5 GB/s | Buffered I/O + HDF5 library overhead |
| BFExtractor `disk_per_pixel` | ~1.0 GB/s | Same, but only reads BF-relevant data |

The h5py implementation is near the practical ceiling for buffered reads on this
system.  Reaching 6.7 GB/s would require `O_DIRECT` with 512-byte-aligned buffers,
which h5py does not support and would require a custom C reader.  The return on
investment is low given the detector-chunked format already reduces I/O to 2–3 GiB.

---

## 5. Cache Fill Breakdown

For `imagefft_storage='host'`, "first recon" = disk I/O + BF-image FFT + warm reconstruction overhead.
The fill cost alone is approximately `first − warm`:

| Scenario | Fill time | Data read | Effective throughput |
|---|---|---|---|
| 48 GiB contiguous | ~38.9 s | 48 GiB | ~1.2 GB/s |
| 48 GiB detector-chunks | ~5.98 s | 2.04 GiB | ~0.34 GiB read + FFT* |
| 64 GiB contiguous | ~53.3 s | 64 GiB | ~1.2 GB/s |
| 64 GiB detector-chunks | ~8.12 s | 2.72 GiB | ~0.33 GiB read + FFT* |

*For detector-chunks, disk I/O takes ~2–3 s at 1 GB/s; the remaining 3–5 s is GPU
FFT (float32→complex64) + H2D/D2H transfers for the 2–3 GiB of BF images.
At this data size the GPU FFT dominates, not the disk.

---

## 6. Best Practices

### 6.1 Prefer persistent ImageFFT storage when memory allows

`imagefft_storage='none'` re-reads from disk on every reconstruction pass and
avoids allocating the FFT cache.  For a single reconstruction this saves RAM but
costs the full I/O time per pass.  For any iterative use (multiple reconstructions,
parameter sweeps, AD refinement) the cached modes are far superior:

- `host`: cache fits in RAM → all subsequent reconstructions are ~2–3 s (pure GPU).
- `device`: cache fits in VRAM → fastest warm reconstructions (~0.03–0.07 s), but
  only the ImageFFT cache is persistent. For small datasets the `speed` pipeline
  may temporarily put raw 4D on device for a whole-pass precompute; for larger
  lazy datasets it can still stream from disk into a device ImageFFT cache.
- `none`: no cache → every pass reads ~2–3 GiB (detector-chunks) or the full
  file (contiguous).  Use only when RAM is genuinely exhausted.

The default `pipeline='balanced'` with `imagefft_storage='auto'` selects
`device` → `host` → `none` for the ImageFFT cache based on available VRAM and
RAM, so the default is already sensible. `pipeline='speed'` is more willing to
materialize lazy raw data when it fits; `pipeline='memory'` keeps
`imagefft_storage='none'`.

The FFT cache size is `Nb × Ry × Rx × 8` bytes (complex64).  For 697 BF pixels and
a 1024×1024 scan this is ~5.5 GiB — easily fits in the 22 GiB available here.

### 6.2 Re-writing data into detector-oriented chunks is worth it for repeated use

| | Contiguous | Detector-chunks |
|---|---|---|
| First reconstruction | 41–57 s | **8–12 s** |
| I/O per reconstruction | 48–64 GiB | **2–3 GiB** |
| Speedup | 1× | **5–7×** |
| File size | same | same |
| Write cost | — | ~53 s for 48 GiB |

The detector-chunked layout stores each `(ky, kx)` scan image as its own HDF5
chunk.  Reading `handle[:, :, ky, kx]` fetches exactly that chunk without touching
the rest of the file.  For a 25 mrad aperture, only 697 of 16 384 detector pixels
are inside the BF disk (~4%), so I/O drops by ~24×.

**When to convert**: if the dataset will be reconstructed more than once, or if
first-reconstruction latency matters, converting is worthwhile.  The one-time write
cost (~53 s for 48 GiB at ~1 GB/s write speed) is paid back on the very first
re-read.

**When not to convert**: one-shot reconstructions where none is acceptable,
or constrained scratch space (the converted file is the same byte size but requires
temporary space during conversion).

### 6.3 `extractor_strategy` is auto-detected and should rarely need manual override

`PipelineManager` resolves `extractor_strategy` automatically from the pipeline,
ImageFFT storage/fill policy, memory budget, and HDF5 chunk layout:

| HDF5 layout / memory state | Auto strategy |
|---|---|---|
| Raw 4D already materialized | `host_mask` |
| Small lazy data under `speed`, raw+vBF+ImageFFT fit VRAM and raw fits RAM | `device_mask` whole-pass precompute |
| Lazy raw fits RAM and ImageFFT storage is persistent | `host_mask` |
| Contiguous lazy file + precompute | `disk_scan_row` |
| Contiguous lazy file + on-the-fly | `disk_slab` |
| Detector-major (Ry,Rx,1,1) | `disk_per_pixel` |
| Scan-major (1,1,Ky,Kx) | `disk_scan_row` |

The `disk_slab` mode is correct for contiguous files in both `host` and `none`
contexts — it reads only the ky bounding box per `BFExtractor.extract_chunk()` call, avoiding a
full file scan per chunk in `none` mode.

---

## 7. Known Limitations and Future Work

- **`O_DIRECT` is not used**: h5py reads through the OS page cache.  For files
  larger than RAM, the effective ceiling is ~1.5 GB/s rather than the 6.7 GB/s
  hardware limit.  A custom reader using aligned `O_DIRECT` reads could approach
  hardware speed but the gain for detector-chunked files (already ~2 GiB I/O) is
  marginal.

- **HDF5 compression**: the test files are uncompressed float32.  Adding LZ4 or
  Blosc compression to detector-chunked files could reduce I/O further at the cost
  of CPU decompression time; not benchmarked here.

- **Zarr alternative**: Zarr natively supports detector-major chunking and can use
  `O_DIRECT`-like backends.  It could close the gap between h5py and hardware
  speed, at the cost of format compatibility.

- **Multi-threaded prefill**: `BFExtractor(strategy=`disk_per_pixel`)` loops over BF
  pixels sequentially.  Parallelising the h5py reads with a thread pool (h5py
  releases the GIL for reads) could improve prefill speed for detector-chunked
  files.
