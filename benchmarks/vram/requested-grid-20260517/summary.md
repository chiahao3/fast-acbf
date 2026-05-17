# fast-acBF VRAM Benchmark

Device: NVIDIA RTX 5000 Ada Generation

Rows marked `est` failed during measurement; their peak allocation is interpolated from successful cases in the same cache mode, with an analytic tensor-size fallback. `actual Nb` is the circular BF-mask pixel count used by the solver.

## Summary Report

For this implementation, VRAM is driven by `cache_mode` (controls ImageFFT storage) and `basis_mode` (controls aberration-basis precomputation). `cache_mode='device'` stores the full `(Nb, Ry, Rx)` complex64 FFT cache in VRAM. `cache_mode='host'` fills a RAM numpy cache lazily per chunk, copying only the active chunk to GPU. `cache_mode='on_the_fly'` recomputes FFTs every pass with no persistent cache. acBF with `basis_mode='precompute'` additionally stores aperture and basis tensors for all Nb pixels; `basis_mode='on_the_fly'` (default) regenerates them per chunk. tcBF only needs small shift-basis vectors, so its peak is nearly independent of cache settings.

Peak allocated VRAM at scan `256 x 256`:

| recon | cache | max_order | Nb~512 | Nb=1024 | Nb~2048 | Nb=4096 |
|---|---|---:|---:|---:|---:|---:|
| acbf | on_the_fly | 1 | 0.40 | 0.40 | 0.40 | 0.40 |
| acbf | on_the_fly | 2 | 0.62 | 0.62 | 0.62 | 0.62 |
| acbf | on_the_fly | 3 | 0.88 | 0.88 | 0.88 | 0.88 |
| acbf | on_the_fly | 4 | 1.20 | 1.20 | 1.20 | 1.20 |
| acbf | host | 1 | 0.40 | 0.40 | 0.40 | 0.40 |
| acbf | host | 2 | 0.62 | 0.62 | 0.62 | 0.62 |
| acbf | host | 3 | 0.88 | 0.88 | 0.88 | 0.88 |
| acbf | host | 4 | 1.20 | 1.20 | 1.20 | 1.20 |
| acbf | device | 1 | 0.63 | 1.25 | 2.50 | 5.00 |
| acbf | device | 2 | 0.84 | 1.25 | 2.50 | 5.00 |
| acbf | device | 3 | 1.10 | 1.35 | 2.50 | 5.00 |
| acbf | device | 4 | 1.41 | 1.66 | 2.50 | 5.00 |
| tcbf | on_the_fly | 1 | 0.16 | 0.16 | 0.16 | 0.16 |
| tcbf | on_the_fly | 2 | 0.16 | 0.16 | 0.16 | 0.16 |
| tcbf | on_the_fly | 3 | 0.16 | 0.16 | 0.16 | 0.16 |
| tcbf | on_the_fly | 4 | 0.16 | 0.16 | 0.16 | 0.17 |
| tcbf | host | 1 | 0.16 | 0.16 | 0.16 | 0.16 |
| tcbf | host | 2 | 0.16 | 0.16 | 0.16 | 0.16 |
| tcbf | host | 3 | 0.16 | 0.16 | 0.16 | 0.16 |
| tcbf | host | 4 | 0.16 | 0.16 | 0.16 | 0.17 |
| tcbf | device | 1 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | device | 2 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | device | 3 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | device | 4 | 0.63 | 1.25 | 2.50 | 5.00 |

## Estimation Equations

Let `B = actual Nb`, `S = Ry * Rx`, `M = max_order`, `K = M * (M + 5) / 2` flattened aberration coefficients, and `C = min(chunk_size, B)`. The benchmark used `chunk_size = 64`. The relevant dtypes are `float32 = 4 bytes` and `complex64 = 8 bytes`.

Persistent FFT cache footprint by cache_mode:

```text
device   : fft_bytes = 8*B*S       # full (Nb, Ry, Rx) complex64 in VRAM
host     : fft_bytes = 8*C*S       # only active chunk in VRAM; rest in RAM
on_the_fly: fft_bytes = 8*C*S      # recomputed per chunk; no persistent VRAM
```

tcBF peak estimate:

```text
tcBF_bytes ~= fft_bytes             # FFT cache (mode-dependent above)
             + 8*K*B                # b_dx and b_dy shift basis, float32
             + 28*C*S               # per-chunk ramp, phasor, multiply, ifft workspaces
```

acBF on_the_fly basis peak estimate (default basis_mode):

```text
acBF_otf_bytes ~= fft_bytes
                 + (64 + 12*K)*C*S  # regenerated aperture, basis, chi, transfer, FFT workspaces
```

acBF precompute basis peak estimate (basis_mode='precompute'):

```text
acBF_pre_bytes ~= fft_bytes
                 + (16 + 8*K)*B*S   # cached ap_t/ap_mt + b_tr/b_t/b_mt for all B
                 + (32 + 4*K)*C*S   # reconstruction-time chunk workspaces
```

Convert bytes to GiB by dividing by `1024**3`. These formulas track the main tensors in the code path; PyTorch allocator behavior, FFT work buffers, and temporary expression lifetimes add some overhead, so treat them as planning estimates rather than exact allocator readouts.

## Full Results

| recon | cache | max_order | scan | requested Nb | actual Nb | status | peak alloc GiB | peak reserved GiB | time s |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| acbf | device | 1 | 64 | 512 | 511 | ok | 0.05 | 0.07 | 0.17 |
| acbf | device | 1 | 64 | 1024 | 1024 | ok | 0.08 | 0.09 | 0.19 |
| acbf | device | 1 | 64 | 2048 | 2047 | ok | 0.16 | 0.17 | 0.21 |
| acbf | device | 1 | 64 | 4096 | 4096 | ok | 0.31 | 0.33 | 0.35 |
| acbf | device | 1 | 128 | 512 | 511 | ok | 0.16 | 0.17 | 0.21 |
| acbf | device | 1 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.30 |
| acbf | device | 1 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.36 |
| acbf | device | 1 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.80 |
| acbf | device | 1 | 256 | 512 | 511 | ok | 0.63 | 0.66 | 0.37 |
| acbf | device | 1 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.76 |
| acbf | device | 1 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.08 |
| acbf | device | 1 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.34 |
| acbf | device | 2 | 64 | 512 | 511 | ok | 0.06 | 0.07 | 0.17 |
| acbf | device | 2 | 64 | 1024 | 1024 | ok | 0.08 | 0.10 | 0.20 |
| acbf | device | 2 | 64 | 2048 | 2047 | ok | 0.16 | 0.18 | 0.22 |
| acbf | device | 2 | 64 | 4096 | 4096 | ok | 0.31 | 0.33 | 0.36 |
| acbf | device | 2 | 128 | 512 | 511 | ok | 0.22 | 0.24 | 0.22 |
| acbf | device | 2 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.30 |
| acbf | device | 2 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.38 |
| acbf | device | 2 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.87 |
| acbf | device | 2 | 256 | 512 | 511 | ok | 0.84 | 0.96 | 0.39 |
| acbf | device | 2 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.79 |
| acbf | device | 2 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.15 |
| acbf | device | 2 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.50 |
| acbf | device | 3 | 64 | 512 | 511 | ok | 0.08 | 0.10 | 0.17 |
| acbf | device | 3 | 64 | 1024 | 1024 | ok | 0.09 | 0.11 | 0.21 |
| acbf | device | 3 | 64 | 2048 | 2047 | ok | 0.16 | 0.18 | 0.23 |
| acbf | device | 3 | 64 | 4096 | 4096 | ok | 0.31 | 0.34 | 0.37 |
| acbf | device | 3 | 128 | 512 | 511 | ok | 0.28 | 0.32 | 0.21 |
| acbf | device | 3 | 128 | 1024 | 1024 | ok | 0.34 | 0.36 | 0.33 |
| acbf | device | 3 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.39 |
| acbf | device | 3 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.94 |
| acbf | device | 3 | 256 | 512 | 511 | ok | 1.10 | 1.25 | 0.40 |
| acbf | device | 3 | 256 | 1024 | 1024 | ok | 1.35 | 1.44 | 0.83 |
| acbf | device | 3 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.23 |
| acbf | device | 3 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.68 |
| acbf | device | 4 | 64 | 512 | 511 | ok | 0.10 | 0.12 | 0.17 |
| acbf | device | 4 | 64 | 1024 | 1024 | ok | 0.11 | 0.13 | 0.20 |
| acbf | device | 4 | 64 | 2048 | 2047 | ok | 0.16 | 0.19 | 0.24 |
| acbf | device | 4 | 64 | 4096 | 4096 | ok | 0.31 | 0.35 | 0.41 |
| acbf | device | 4 | 128 | 512 | 511 | ok | 0.36 | 0.43 | 0.21 |
| acbf | device | 4 | 128 | 1024 | 1024 | ok | 0.42 | 0.46 | 0.32 |
| acbf | device | 4 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.41 |
| acbf | device | 4 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.91 |
| acbf | device | 4 | 256 | 512 | 511 | ok | 1.41 | 1.66 | 0.41 |
| acbf | device | 4 | 256 | 1024 | 1024 | ok | 1.66 | 1.81 | 0.88 |
| acbf | device | 4 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.32 |
| acbf | device | 4 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.89 |
| acbf | host | 1 | 64 | 512 | 511 | ok | 0.03 | 0.05 | 0.17 |
| acbf | host | 1 | 64 | 1024 | 1024 | ok | 0.03 | 0.05 | 0.22 |
| acbf | host | 1 | 64 | 2048 | 2047 | ok | 0.03 | 0.05 | 0.28 |
| acbf | host | 1 | 64 | 4096 | 4096 | ok | 0.03 | 0.05 | 0.58 |
| acbf | host | 1 | 128 | 512 | 511 | ok | 0.11 | 0.13 | 0.27 |
| acbf | host | 1 | 128 | 1024 | 1024 | ok | 0.11 | 0.13 | 0.53 |
| acbf | host | 1 | 128 | 2048 | 2047 | ok | 0.11 | 0.13 | 0.80 |
| acbf | host | 1 | 128 | 4096 | 4096 | ok | 0.11 | 0.13 | 1.85 |
| acbf | host | 1 | 256 | 512 | 511 | ok | 0.40 | 0.50 | 0.81 |
| acbf | host | 1 | 256 | 1024 | 1024 | ok | 0.40 | 0.50 | 1.82 |
| acbf | host | 1 | 256 | 2048 | 2047 | ok | 0.40 | 0.50 | 2.92 |
| acbf | host | 1 | 256 | 4096 | 4096 | ok | 0.40 | 0.50 | 8.42 |
| acbf | host | 2 | 64 | 512 | 511 | ok | 0.05 | 0.06 | 0.18 |
| acbf | host | 2 | 64 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.23 |
| acbf | host | 2 | 64 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.29 |
| acbf | host | 2 | 64 | 4096 | 4096 | ok | 0.05 | 0.06 | 0.56 |
| acbf | host | 2 | 128 | 512 | 511 | ok | 0.16 | 0.18 | 0.28 |
| acbf | host | 2 | 128 | 1024 | 1024 | ok | 0.16 | 0.18 | 0.55 |
| acbf | host | 2 | 128 | 2048 | 2047 | ok | 0.16 | 0.18 | 0.72 |
| acbf | host | 2 | 128 | 4096 | 4096 | ok | 0.16 | 0.18 | 2.04 |
| acbf | host | 2 | 256 | 512 | 511 | ok | 0.62 | 0.72 | 0.82 |
| acbf | host | 2 | 256 | 1024 | 1024 | ok | 0.62 | 0.72 | 1.83 |
| acbf | host | 2 | 256 | 2048 | 2047 | ok | 0.62 | 0.72 | 2.90 |
| acbf | host | 2 | 256 | 4096 | 4096 | ok | 0.62 | 0.72 | 8.23 |
| acbf | host | 3 | 64 | 512 | 511 | ok | 0.06 | 0.08 | 0.18 |
| acbf | host | 3 | 64 | 1024 | 1024 | ok | 0.06 | 0.08 | 0.23 |
| acbf | host | 3 | 64 | 2048 | 2047 | ok | 0.06 | 0.08 | 0.30 |
| acbf | host | 3 | 64 | 4096 | 4096 | ok | 0.06 | 0.08 | 0.63 |
| acbf | host | 3 | 128 | 512 | 511 | ok | 0.23 | 0.26 | 0.28 |
| acbf | host | 3 | 128 | 1024 | 1024 | ok | 0.23 | 0.26 | 0.56 |
| acbf | host | 3 | 128 | 2048 | 2047 | ok | 0.23 | 0.26 | 0.82 |
| acbf | host | 3 | 128 | 4096 | 4096 | ok | 0.23 | 0.26 | 1.95 |
| acbf | host | 3 | 256 | 512 | 511 | ok | 0.88 | 0.99 | 0.81 |
| acbf | host | 3 | 256 | 1024 | 1024 | ok | 0.88 | 0.99 | 1.83 |
| acbf | host | 3 | 256 | 2048 | 2047 | ok | 0.88 | 0.99 | 2.95 |
| acbf | host | 3 | 256 | 4096 | 4096 | ok | 0.88 | 0.99 | 8.83 |
| acbf | host | 4 | 64 | 512 | 511 | ok | 0.08 | 0.11 | 0.18 |
| acbf | host | 4 | 64 | 1024 | 1024 | ok | 0.08 | 0.11 | 0.23 |
| acbf | host | 4 | 64 | 2048 | 2047 | ok | 0.08 | 0.11 | 0.30 |
| acbf | host | 4 | 64 | 4096 | 4096 | ok | 0.08 | 0.11 | 0.62 |
| acbf | host | 4 | 128 | 512 | 511 | ok | 0.31 | 0.37 | 0.27 |
| acbf | host | 4 | 128 | 1024 | 1024 | ok | 0.31 | 0.37 | 0.55 |
| acbf | host | 4 | 128 | 2048 | 2047 | ok | 0.31 | 0.37 | 0.75 |
| acbf | host | 4 | 128 | 4096 | 4096 | ok | 0.31 | 0.37 | 2.01 |
| acbf | host | 4 | 256 | 512 | 511 | ok | 1.20 | 1.30 | 0.84 |
| acbf | host | 4 | 256 | 1024 | 1024 | ok | 1.20 | 1.30 | 1.84 |
| acbf | host | 4 | 256 | 2048 | 2047 | ok | 1.20 | 1.30 | 2.91 |
| acbf | host | 4 | 256 | 4096 | 4096 | ok | 1.20 | 1.30 | 8.32 |
| acbf | on_the_fly | 1 | 64 | 512 | 511 | ok | 0.03 | 0.05 | 0.16 |
| acbf | on_the_fly | 1 | 64 | 1024 | 1024 | ok | 0.03 | 0.05 | 0.20 |
| acbf | on_the_fly | 1 | 64 | 2048 | 2047 | ok | 0.03 | 0.05 | 0.23 |
| acbf | on_the_fly | 1 | 64 | 4096 | 4096 | ok | 0.03 | 0.05 | 0.45 |
| acbf | on_the_fly | 1 | 128 | 512 | 511 | ok | 0.11 | 0.13 | 0.23 |
| acbf | on_the_fly | 1 | 128 | 1024 | 1024 | ok | 0.11 | 0.13 | 0.39 |
| acbf | on_the_fly | 1 | 128 | 2048 | 2047 | ok | 0.11 | 0.13 | 0.45 |
| acbf | on_the_fly | 1 | 128 | 4096 | 4096 | ok | 0.11 | 0.13 | 1.32 |
| acbf | on_the_fly | 1 | 256 | 512 | 511 | ok | 0.40 | 0.50 | 0.45 |
| acbf | on_the_fly | 1 | 256 | 1024 | 1024 | ok | 0.40 | 0.50 | 1.18 |
| acbf | on_the_fly | 1 | 256 | 2048 | 2047 | ok | 0.40 | 0.50 | 1.52 |
| acbf | on_the_fly | 1 | 256 | 4096 | 4096 | ok | 0.40 | 0.50 | 5.55 |
| acbf | on_the_fly | 2 | 64 | 512 | 511 | ok | 0.05 | 0.06 | 0.16 |
| acbf | on_the_fly | 2 | 64 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.20 |
| acbf | on_the_fly | 2 | 64 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.24 |
| acbf | on_the_fly | 2 | 64 | 4096 | 4096 | ok | 0.05 | 0.06 | 0.46 |
| acbf | on_the_fly | 2 | 128 | 512 | 511 | ok | 0.16 | 0.18 | 0.23 |
| acbf | on_the_fly | 2 | 128 | 1024 | 1024 | ok | 0.16 | 0.18 | 0.40 |
| acbf | on_the_fly | 2 | 128 | 2048 | 2047 | ok | 0.16 | 0.18 | 0.46 |
| acbf | on_the_fly | 2 | 128 | 4096 | 4096 | ok | 0.16 | 0.18 | 1.34 |
| acbf | on_the_fly | 2 | 256 | 512 | 511 | ok | 0.62 | 0.72 | 0.48 |
| acbf | on_the_fly | 2 | 256 | 1024 | 1024 | ok | 0.62 | 0.72 | 1.18 |
| acbf | on_the_fly | 2 | 256 | 2048 | 2047 | ok | 0.62 | 0.72 | 1.54 |
| acbf | on_the_fly | 2 | 256 | 4096 | 4096 | ok | 0.62 | 0.72 | 5.63 |
| acbf | on_the_fly | 3 | 64 | 512 | 511 | ok | 0.06 | 0.08 | 0.16 |
| acbf | on_the_fly | 3 | 64 | 1024 | 1024 | ok | 0.06 | 0.08 | 0.20 |
| acbf | on_the_fly | 3 | 64 | 2048 | 2047 | ok | 0.06 | 0.08 | 0.25 |
| acbf | on_the_fly | 3 | 64 | 4096 | 4096 | ok | 0.06 | 0.08 | 0.48 |
| acbf | on_the_fly | 3 | 128 | 512 | 511 | ok | 0.23 | 0.26 | 0.23 |
| acbf | on_the_fly | 3 | 128 | 1024 | 1024 | ok | 0.23 | 0.26 | 0.40 |
| acbf | on_the_fly | 3 | 128 | 2048 | 2047 | ok | 0.23 | 0.26 | 0.47 |
| acbf | on_the_fly | 3 | 128 | 4096 | 4096 | ok | 0.23 | 0.26 | 1.38 |
| acbf | on_the_fly | 3 | 256 | 512 | 511 | ok | 0.88 | 0.99 | 0.47 |
| acbf | on_the_fly | 3 | 256 | 1024 | 1024 | ok | 0.88 | 0.99 | 1.17 |
| acbf | on_the_fly | 3 | 256 | 2048 | 2047 | ok | 0.88 | 0.99 | 1.54 |
| acbf | on_the_fly | 3 | 256 | 4096 | 4096 | ok | 0.88 | 0.99 | 5.52 |
| acbf | on_the_fly | 4 | 64 | 512 | 511 | ok | 0.08 | 0.11 | 0.17 |
| acbf | on_the_fly | 4 | 64 | 1024 | 1024 | ok | 0.08 | 0.11 | 0.21 |
| acbf | on_the_fly | 4 | 64 | 2048 | 2047 | ok | 0.08 | 0.11 | 0.26 |
| acbf | on_the_fly | 4 | 64 | 4096 | 4096 | ok | 0.08 | 0.11 | 0.47 |
| acbf | on_the_fly | 4 | 128 | 512 | 511 | ok | 0.31 | 0.37 | 0.23 |
| acbf | on_the_fly | 4 | 128 | 1024 | 1024 | ok | 0.31 | 0.37 | 0.41 |
| acbf | on_the_fly | 4 | 128 | 2048 | 2047 | ok | 0.31 | 0.37 | 0.49 |
| acbf | on_the_fly | 4 | 128 | 4096 | 4096 | ok | 0.31 | 0.37 | 1.39 |
| acbf | on_the_fly | 4 | 256 | 512 | 511 | ok | 1.20 | 1.30 | 0.47 |
| acbf | on_the_fly | 4 | 256 | 1024 | 1024 | ok | 1.20 | 1.30 | 1.20 |
| acbf | on_the_fly | 4 | 256 | 2048 | 2047 | ok | 1.20 | 1.30 | 1.58 |
| acbf | on_the_fly | 4 | 256 | 4096 | 4096 | ok | 1.20 | 1.30 | 5.82 |
| tcbf | device | 1 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | device | 1 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.17 |
| tcbf | device | 1 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.20 |
| tcbf | device | 1 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.30 |
| tcbf | device | 1 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | device | 1 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.27 |
| tcbf | device | 1 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.32 |
| tcbf | device | 1 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.77 |
| tcbf | device | 1 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.31 |
| tcbf | device | 1 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.73 |
| tcbf | device | 1 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.96 |
| tcbf | device | 1 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.10 |
| tcbf | device | 2 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.15 |
| tcbf | device | 2 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.17 |
| tcbf | device | 2 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.20 |
| tcbf | device | 2 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.31 |
| tcbf | device | 2 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.20 |
| tcbf | device | 2 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.28 |
| tcbf | device | 2 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.33 |
| tcbf | device | 2 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.78 |
| tcbf | device | 2 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.34 |
| tcbf | device | 2 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.70 |
| tcbf | device | 2 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.95 |
| tcbf | device | 2 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.15 |
| tcbf | device | 3 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.15 |
| tcbf | device | 3 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.17 |
| tcbf | device | 3 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.20 |
| tcbf | device | 3 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.32 |
| tcbf | device | 3 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | device | 3 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.28 |
| tcbf | device | 3 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.33 |
| tcbf | device | 3 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.83 |
| tcbf | device | 3 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.34 |
| tcbf | device | 3 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.71 |
| tcbf | device | 3 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.96 |
| tcbf | device | 3 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.17 |
| tcbf | device | 4 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.15 |
| tcbf | device | 4 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.18 |
| tcbf | device | 4 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.21 |
| tcbf | device | 4 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.33 |
| tcbf | device | 4 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | device | 4 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.28 |
| tcbf | device | 4 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.34 |
| tcbf | device | 4 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.85 |
| tcbf | device | 4 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.33 |
| tcbf | device | 4 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.72 |
| tcbf | device | 4 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.98 |
| tcbf | device | 4 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.17 |
| tcbf | host | 1 | 64 | 512 | 511 | ok | 0.02 | 0.02 | 0.15 |
| tcbf | host | 1 | 64 | 1024 | 1024 | ok | 0.02 | 0.02 | 0.20 |
| tcbf | host | 1 | 64 | 2048 | 2047 | ok | 0.02 | 0.02 | 0.26 |
| tcbf | host | 1 | 64 | 4096 | 4096 | ok | 0.02 | 0.02 | 0.52 |
| tcbf | host | 1 | 128 | 512 | 511 | ok | 0.05 | 0.06 | 0.26 |
| tcbf | host | 1 | 128 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.48 |
| tcbf | host | 1 | 128 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.62 |
| tcbf | host | 1 | 128 | 4096 | 4096 | ok | 0.05 | 0.06 | 1.69 |
| tcbf | host | 1 | 256 | 512 | 511 | ok | 0.16 | 0.17 | 0.80 |
| tcbf | host | 1 | 256 | 1024 | 1024 | ok | 0.16 | 0.17 | 1.80 |
| tcbf | host | 1 | 256 | 2048 | 2047 | ok | 0.16 | 0.17 | 2.89 |
| tcbf | host | 1 | 256 | 4096 | 4096 | ok | 0.16 | 0.17 | 8.42 |
| tcbf | host | 2 | 64 | 512 | 511 | ok | 0.02 | 0.02 | 0.15 |
| tcbf | host | 2 | 64 | 1024 | 1024 | ok | 0.02 | 0.02 | 0.20 |
| tcbf | host | 2 | 64 | 2048 | 2047 | ok | 0.02 | 0.02 | 0.27 |
| tcbf | host | 2 | 64 | 4096 | 4096 | ok | 0.02 | 0.02 | 0.53 |
| tcbf | host | 2 | 128 | 512 | 511 | ok | 0.05 | 0.06 | 0.25 |
| tcbf | host | 2 | 128 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.48 |
| tcbf | host | 2 | 128 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.63 |
| tcbf | host | 2 | 128 | 4096 | 4096 | ok | 0.05 | 0.06 | 1.69 |
| tcbf | host | 2 | 256 | 512 | 511 | ok | 0.16 | 0.17 | 0.79 |
| tcbf | host | 2 | 256 | 1024 | 1024 | ok | 0.16 | 0.17 | 1.80 |
| tcbf | host | 2 | 256 | 2048 | 2047 | ok | 0.16 | 0.17 | 2.83 |
| tcbf | host | 2 | 256 | 4096 | 4096 | ok | 0.16 | 0.17 | 8.37 |
| tcbf | host | 3 | 64 | 512 | 511 | ok | 0.02 | 0.02 | 0.15 |
| tcbf | host | 3 | 64 | 1024 | 1024 | ok | 0.02 | 0.02 | 0.20 |
| tcbf | host | 3 | 64 | 2048 | 2047 | ok | 0.02 | 0.02 | 0.26 |
| tcbf | host | 3 | 64 | 4096 | 4096 | ok | 0.02 | 0.02 | 0.54 |
| tcbf | host | 3 | 128 | 512 | 511 | ok | 0.05 | 0.06 | 0.26 |
| tcbf | host | 3 | 128 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.48 |
| tcbf | host | 3 | 128 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.63 |
| tcbf | host | 3 | 128 | 4096 | 4096 | ok | 0.05 | 0.06 | 1.65 |
| tcbf | host | 3 | 256 | 512 | 511 | ok | 0.16 | 0.17 | 0.78 |
| tcbf | host | 3 | 256 | 1024 | 1024 | ok | 0.16 | 0.17 | 1.80 |
| tcbf | host | 3 | 256 | 2048 | 2047 | ok | 0.16 | 0.17 | 2.85 |
| tcbf | host | 3 | 256 | 4096 | 4096 | ok | 0.16 | 0.17 | 8.33 |
| tcbf | host | 4 | 64 | 512 | 511 | ok | 0.02 | 0.02 | 0.16 |
| tcbf | host | 4 | 64 | 1024 | 1024 | ok | 0.02 | 0.02 | 0.21 |
| tcbf | host | 4 | 64 | 2048 | 2047 | ok | 0.02 | 0.02 | 0.28 |
| tcbf | host | 4 | 64 | 4096 | 4096 | ok | 0.02 | 0.02 | 0.55 |
| tcbf | host | 4 | 128 | 512 | 511 | ok | 0.05 | 0.06 | 0.26 |
| tcbf | host | 4 | 128 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.49 |
| tcbf | host | 4 | 128 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.64 |
| tcbf | host | 4 | 128 | 4096 | 4096 | ok | 0.05 | 0.06 | 1.69 |
| tcbf | host | 4 | 256 | 512 | 511 | ok | 0.16 | 0.17 | 0.80 |
| tcbf | host | 4 | 256 | 1024 | 1024 | ok | 0.16 | 0.17 | 1.84 |
| tcbf | host | 4 | 256 | 2048 | 2047 | ok | 0.16 | 0.17 | 2.90 |
| tcbf | host | 4 | 256 | 4096 | 4096 | ok | 0.17 | 0.17 | 8.21 |
| tcbf | on_the_fly | 1 | 64 | 512 | 511 | ok | 0.02 | 0.02 | 0.14 |
| tcbf | on_the_fly | 1 | 64 | 1024 | 1024 | ok | 0.02 | 0.02 | 0.16 |
| tcbf | on_the_fly | 1 | 64 | 2048 | 2047 | ok | 0.02 | 0.02 | 0.20 |
| tcbf | on_the_fly | 1 | 64 | 4096 | 4096 | ok | 0.02 | 0.02 | 0.41 |
| tcbf | on_the_fly | 1 | 128 | 512 | 511 | ok | 0.05 | 0.06 | 0.20 |
| tcbf | on_the_fly | 1 | 128 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.36 |
| tcbf | on_the_fly | 1 | 128 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.42 |
| tcbf | on_the_fly | 1 | 128 | 4096 | 4096 | ok | 0.05 | 0.06 | 1.28 |
| tcbf | on_the_fly | 1 | 256 | 512 | 511 | ok | 0.16 | 0.17 | 0.43 |
| tcbf | on_the_fly | 1 | 256 | 1024 | 1024 | ok | 0.16 | 0.17 | 1.18 |
| tcbf | on_the_fly | 1 | 256 | 2048 | 2047 | ok | 0.16 | 0.17 | 1.51 |
| tcbf | on_the_fly | 1 | 256 | 4096 | 4096 | ok | 0.16 | 0.17 | 5.77 |
| tcbf | on_the_fly | 2 | 64 | 512 | 511 | ok | 0.02 | 0.02 | 0.14 |
| tcbf | on_the_fly | 2 | 64 | 1024 | 1024 | ok | 0.02 | 0.02 | 0.17 |
| tcbf | on_the_fly | 2 | 64 | 2048 | 2047 | ok | 0.02 | 0.02 | 0.21 |
| tcbf | on_the_fly | 2 | 64 | 4096 | 4096 | ok | 0.02 | 0.02 | 0.43 |
| tcbf | on_the_fly | 2 | 128 | 512 | 511 | ok | 0.05 | 0.06 | 0.20 |
| tcbf | on_the_fly | 2 | 128 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.37 |
| tcbf | on_the_fly | 2 | 128 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.43 |
| tcbf | on_the_fly | 2 | 128 | 4096 | 4096 | ok | 0.05 | 0.06 | 1.22 |
| tcbf | on_the_fly | 2 | 256 | 512 | 511 | ok | 0.16 | 0.17 | 0.45 |
| tcbf | on_the_fly | 2 | 256 | 1024 | 1024 | ok | 0.16 | 0.17 | 1.16 |
| tcbf | on_the_fly | 2 | 256 | 2048 | 2047 | ok | 0.16 | 0.17 | 1.52 |
| tcbf | on_the_fly | 2 | 256 | 4096 | 4096 | ok | 0.16 | 0.17 | 5.59 |
| tcbf | on_the_fly | 3 | 64 | 512 | 511 | ok | 0.02 | 0.02 | 0.14 |
| tcbf | on_the_fly | 3 | 64 | 1024 | 1024 | ok | 0.02 | 0.02 | 0.18 |
| tcbf | on_the_fly | 3 | 64 | 2048 | 2047 | ok | 0.02 | 0.02 | 0.22 |
| tcbf | on_the_fly | 3 | 64 | 4096 | 4096 | ok | 0.02 | 0.02 | 0.39 |
| tcbf | on_the_fly | 3 | 128 | 512 | 511 | ok | 0.05 | 0.06 | 0.20 |
| tcbf | on_the_fly | 3 | 128 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.38 |
| tcbf | on_the_fly | 3 | 128 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.43 |
| tcbf | on_the_fly | 3 | 128 | 4096 | 4096 | ok | 0.05 | 0.06 | 1.33 |
| tcbf | on_the_fly | 3 | 256 | 512 | 511 | ok | 0.16 | 0.17 | 0.44 |
| tcbf | on_the_fly | 3 | 256 | 1024 | 1024 | ok | 0.16 | 0.17 | 1.16 |
| tcbf | on_the_fly | 3 | 256 | 2048 | 2047 | ok | 0.16 | 0.17 | 1.52 |
| tcbf | on_the_fly | 3 | 256 | 4096 | 4096 | ok | 0.16 | 0.17 | 5.64 |
| tcbf | on_the_fly | 4 | 64 | 512 | 511 | ok | 0.02 | 0.02 | 0.14 |
| tcbf | on_the_fly | 4 | 64 | 1024 | 1024 | ok | 0.02 | 0.02 | 0.18 |
| tcbf | on_the_fly | 4 | 64 | 2048 | 2047 | ok | 0.02 | 0.02 | 0.22 |
| tcbf | on_the_fly | 4 | 64 | 4096 | 4096 | ok | 0.02 | 0.02 | 0.46 |
| tcbf | on_the_fly | 4 | 128 | 512 | 511 | ok | 0.05 | 0.06 | 0.20 |
| tcbf | on_the_fly | 4 | 128 | 1024 | 1024 | ok | 0.05 | 0.06 | 0.38 |
| tcbf | on_the_fly | 4 | 128 | 2048 | 2047 | ok | 0.05 | 0.06 | 0.44 |
| tcbf | on_the_fly | 4 | 128 | 4096 | 4096 | ok | 0.05 | 0.06 | 1.36 |
| tcbf | on_the_fly | 4 | 256 | 512 | 511 | ok | 0.16 | 0.17 | 0.44 |
| tcbf | on_the_fly | 4 | 256 | 1024 | 1024 | ok | 0.16 | 0.17 | 1.14 |
| tcbf | on_the_fly | 4 | 256 | 2048 | 2047 | ok | 0.16 | 0.17 | 1.54 |
| tcbf | on_the_fly | 4 | 256 | 4096 | 4096 | ok | 0.17 | 0.17 | 5.90 |
