# fast-acBF VRAM Benchmark

Device: NVIDIA RTX 5000 Ada Generation

Rows marked `est` failed during measurement; their peak allocation is interpolated from successful cases in the same cache mode, with an analytic tensor-size fallback. `actual Nb` is the circular BF-mask pixel count used by the solver.

## Summary Report

For this implementation, VRAM is driven by the extracted virtual-BF stack `vbf_images` as `float32` plus its `complex64` FFT cache. acBF `full` mode then stores detector-wide aperture and aberration-basis tensors, while acBF `lazy` mode regenerates those tensors per chunk. tcBF uses the same FFT cache but only stores small shift-basis vectors, so its peak is much closer to the common stack/FFT footprint and is effectively independent of `cache_mode`.

Peak allocated VRAM at scan `256 x 256`:

| recon | cache | max_order | Nb~512 | Nb=1024 | Nb~2048 | Nb=4096 |
|---|---|---:|---:|---:|---:|---:|
| acbf | lazy | 1 | 0.74 | 1.25 | 2.50 | 5.00 |
| acbf | lazy | 2 | 0.96 | 1.34 | 2.50 | 5.00 |
| acbf | lazy | 3 | 1.23 | 1.60 | 2.50 | 5.00 |
| acbf | lazy | 4 | 1.54 | 1.92 | 2.67 | 5.00 |
| acbf | full | 1 | 1.60 | 2.98 | 5.73 | 11.23 |
| acbf | full | 2 | 2.64 | 5.02 | 9.76 | 19.27 |
| acbf | full | 3 | 3.99 | 7.63 | 14.87 | 29.38 |
| acbf | full | 4 | 5.62 | 10.75 | 20.99 | 41.80 est |
| tcbf | lazy | 1 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | lazy | 2 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | lazy | 3 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | lazy | 4 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | full | 1 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | full | 2 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | full | 3 | 0.63 | 1.25 | 2.50 | 5.00 |
| tcbf | full | 4 | 0.63 | 1.25 | 2.50 | 5.00 |

## Estimation Equations

Let `B = actual Nb`, `S = Ry * Rx`, `M = max_order`, `K = M * (M + 5) / 2` flattened aberration coefficients, and `C = min(chunk_size, B)`. The benchmark used `chunk_size = 64`. The relevant dtypes are `float32 = 4 bytes` and `complex64 = 8 bytes`.

Common persistent stack/cache footprint:

```text
common_bytes ~= 4*B*S       # vBF image stack, float32
              + 8*B*S       # FFT cache, complex64
              = 12*B*S
```

tcBF peak estimate:

```text
tcBF_bytes ~= 12*B*S        # common vBF + FFT cache
             + 8*K*B        # b_dx and b_dy shift basis, float32
             + 28*C*S       # per-chunk ramp, phasor, multiply, ifft workspaces
```

acBF lazy peak estimate:

```text
acBF_lazy_bytes ~= 12*B*S
                  + (64 + 12*K)*C*S   # regenerated aperture, basis, chi, transfer, FFT workspaces
```

acBF full peak estimate:

```text
acBF_full_bytes ~= (20 + 8*K)*B*S      # common + cached ap_t/ap_mt + cached b_t/b_mt
                  + (32 + 4*K)*C*S     # reconstruction-time chunk workspaces
```

Convert bytes to GiB by dividing by `1024**3`. These formulas track the main tensors in the code path; PyTorch allocator behavior, FFT work buffers, and temporary expression lifetimes add some overhead, so treat them as planning estimates rather than exact allocator readouts.

## Full Results

| recon | cache | max_order | scan | requested Nb | actual Nb | status | peak alloc GiB | peak reserved GiB | time s |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| acbf | full | 1 | 64 | 512 | 511 | ok | 0.11 | 0.14 | 0.16 |
| acbf | full | 1 | 64 | 1024 | 1024 | ok | 0.19 | 0.22 | 0.17 |
| acbf | full | 1 | 64 | 2048 | 2047 | ok | 0.37 | 0.39 | 0.18 |
| acbf | full | 1 | 64 | 4096 | 4096 | ok | 0.71 | 0.74 | 0.59 |
| acbf | full | 1 | 128 | 512 | 511 | ok | 0.41 | 0.42 | 0.19 |
| acbf | full | 1 | 128 | 1024 | 1024 | ok | 0.75 | 0.78 | 0.33 |
| acbf | full | 1 | 128 | 2048 | 2047 | ok | 1.44 | 1.45 | 0.24 |
| acbf | full | 1 | 128 | 4096 | 4096 | ok | 2.81 | 2.84 | 1.93 |
| acbf | full | 1 | 256 | 512 | 511 | ok | 1.60 | 1.64 | 0.33 |
| acbf | full | 1 | 256 | 1024 | 1024 | ok | 2.98 | 3.02 | 0.92 |
| acbf | full | 1 | 256 | 2048 | 2047 | ok | 5.73 | 5.77 | 0.75 |
| acbf | full | 1 | 256 | 4096 | 4096 | ok | 11.23 | 11.27 | 7.45 |
| acbf | full | 2 | 64 | 512 | 511 | ok | 0.17 | 0.22 | 0.16 |
| acbf | full | 2 | 64 | 1024 | 1024 | ok | 0.32 | 0.42 | 0.18 |
| acbf | full | 2 | 64 | 2048 | 2047 | ok | 0.62 | 0.79 | 0.20 |
| acbf | full | 2 | 64 | 4096 | 4096 | ok | 1.21 | 1.55 | 0.58 |
| acbf | full | 2 | 128 | 512 | 511 | ok | 0.66 | 0.67 | 0.19 |
| acbf | full | 2 | 128 | 1024 | 1024 | ok | 1.25 | 1.26 | 0.33 |
| acbf | full | 2 | 128 | 2048 | 2047 | ok | 2.44 | 2.46 | 0.25 |
| acbf | full | 2 | 128 | 4096 | 4096 | ok | 4.82 | 4.83 | 1.96 |
| acbf | full | 2 | 256 | 512 | 511 | ok | 2.64 | 2.67 | 0.35 |
| acbf | full | 2 | 256 | 1024 | 1024 | ok | 5.02 | 5.05 | 0.95 |
| acbf | full | 2 | 256 | 2048 | 2047 | ok | 9.76 | 9.80 | 0.79 |
| acbf | full | 2 | 256 | 4096 | 4096 | ok | 19.27 | 19.30 | 7.57 |
| acbf | full | 3 | 64 | 512 | 511 | ok | 0.25 | 0.28 | 0.16 |
| acbf | full | 3 | 64 | 1024 | 1024 | ok | 0.48 | 0.51 | 0.18 |
| acbf | full | 3 | 64 | 2048 | 2047 | ok | 0.93 | 0.96 | 0.20 |
| acbf | full | 3 | 64 | 4096 | 4096 | ok | 1.84 | 1.87 | 0.60 |
| acbf | full | 3 | 128 | 512 | 511 | ok | 1.00 | 1.01 | 0.20 |
| acbf | full | 3 | 128 | 1024 | 1024 | ok | 1.91 | 1.91 | 0.34 |
| acbf | full | 3 | 128 | 2048 | 2047 | ok | 3.72 | 3.74 | 0.27 |
| acbf | full | 3 | 128 | 4096 | 4096 | ok | 7.34 | 7.37 | 1.99 |
| acbf | full | 3 | 256 | 512 | 511 | ok | 3.99 | 4.03 | 0.37 |
| acbf | full | 3 | 256 | 1024 | 1024 | ok | 7.63 | 7.66 | 0.99 |
| acbf | full | 3 | 256 | 2048 | 2047 | ok | 14.87 | 14.90 | 0.87 |
| acbf | full | 3 | 256 | 4096 | 4096 | ok | 29.38 | 29.41 | 7.72 |
| acbf | full | 4 | 64 | 512 | 511 | ok | 0.35 | 0.38 | 0.16 |
| acbf | full | 4 | 64 | 1024 | 1024 | ok | 0.67 | 0.71 | 0.19 |
| acbf | full | 4 | 64 | 2048 | 2047 | ok | 1.31 | 1.34 | 0.21 |
| acbf | full | 4 | 64 | 4096 | 4096 | ok | 2.59 | 2.62 | 0.62 |
| acbf | full | 4 | 128 | 512 | 511 | ok | 1.40 | 1.42 | 0.20 |
| acbf | full | 4 | 128 | 1024 | 1024 | ok | 2.69 | 2.70 | 0.35 |
| acbf | full | 4 | 128 | 2048 | 2047 | ok | 5.25 | 5.27 | 0.28 |
| acbf | full | 4 | 128 | 4096 | 4096 | ok | 10.38 | 10.39 | 2.03 |
| acbf | full | 4 | 256 | 512 | 511 | ok | 5.62 | 5.65 | 0.39 |
| acbf | full | 4 | 256 | 1024 | 1024 | ok | 10.75 | 10.78 | 1.04 |
| acbf | full | 4 | 256 | 2048 | 2047 | ok | 20.99 | 21.03 | 0.96 |
| acbf | full | 4 | 256 | 4096 | 4096 | oom_estimated est | 41.80 |  | 7.50 |
| acbf | lazy | 1 | 64 | 512 | 511 | ok | 0.05 | 0.09 | 0.20 |
| acbf | lazy | 1 | 64 | 1024 | 1024 | ok | 0.08 | 0.09 | 0.20 |
| acbf | lazy | 1 | 64 | 2048 | 2047 | ok | 0.16 | 0.17 | 0.18 |
| acbf | lazy | 1 | 64 | 4096 | 4096 | ok | 0.31 | 0.33 | 0.55 |
| acbf | lazy | 1 | 128 | 512 | 511 | ok | 0.19 | 0.20 | 0.19 |
| acbf | lazy | 1 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.33 |
| acbf | lazy | 1 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.23 |
| acbf | lazy | 1 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 1.92 |
| acbf | lazy | 1 | 256 | 512 | 511 | ok | 0.74 | 0.78 | 0.32 |
| acbf | lazy | 1 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.92 |
| acbf | lazy | 1 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.73 |
| acbf | lazy | 1 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.46 |
| acbf | lazy | 2 | 64 | 512 | 511 | ok | 0.07 | 0.09 | 0.15 |
| acbf | lazy | 2 | 64 | 1024 | 1024 | ok | 0.09 | 0.12 | 0.18 |
| acbf | lazy | 2 | 64 | 2048 | 2047 | ok | 0.16 | 0.18 | 0.19 |
| acbf | lazy | 2 | 64 | 4096 | 4096 | ok | 0.31 | 0.33 | 0.57 |
| acbf | lazy | 2 | 128 | 512 | 511 | ok | 0.25 | 0.26 | 0.19 |
| acbf | lazy | 2 | 128 | 1024 | 1024 | ok | 0.34 | 0.37 | 0.34 |
| acbf | lazy | 2 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.25 |
| acbf | lazy | 2 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 1.96 |
| acbf | lazy | 2 | 256 | 512 | 511 | ok | 0.96 | 1.03 | 0.36 |
| acbf | lazy | 2 | 256 | 1024 | 1024 | ok | 1.34 | 1.36 | 0.95 |
| acbf | lazy | 2 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.80 |
| acbf | lazy | 2 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.78 |
| acbf | lazy | 3 | 64 | 512 | 511 | ok | 0.08 | 0.10 | 0.16 |
| acbf | lazy | 3 | 64 | 1024 | 1024 | ok | 0.11 | 0.12 | 0.18 |
| acbf | lazy | 3 | 64 | 2048 | 2047 | ok | 0.16 | 0.18 | 0.20 |
| acbf | lazy | 3 | 64 | 4096 | 4096 | ok | 0.31 | 0.34 | 0.59 |
| acbf | lazy | 3 | 128 | 512 | 511 | ok | 0.31 | 0.36 | 0.20 |
| acbf | lazy | 3 | 128 | 1024 | 1024 | ok | 0.41 | 0.46 | 0.35 |
| acbf | lazy | 3 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.28 |
| acbf | lazy | 3 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 2.01 |
| acbf | lazy | 3 | 256 | 512 | 511 | ok | 1.23 | 1.38 | 0.37 |
| acbf | lazy | 3 | 256 | 1024 | 1024 | ok | 1.60 | 1.63 | 1.00 |
| acbf | lazy | 3 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.89 |
| acbf | lazy | 3 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.76 |
| acbf | lazy | 4 | 64 | 512 | 511 | ok | 0.10 | 0.12 | 0.16 |
| acbf | lazy | 4 | 64 | 1024 | 1024 | ok | 0.13 | 0.15 | 0.19 |
| acbf | lazy | 4 | 64 | 2048 | 2047 | ok | 0.17 | 0.19 | 0.21 |
| acbf | lazy | 4 | 64 | 4096 | 4096 | ok | 0.31 | 0.35 | 0.61 |
| acbf | lazy | 4 | 128 | 512 | 511 | ok | 0.39 | 0.45 | 0.20 |
| acbf | lazy | 4 | 128 | 1024 | 1024 | ok | 0.48 | 0.54 | 0.36 |
| acbf | lazy | 4 | 128 | 2048 | 2047 | ok | 0.67 | 0.70 | 0.30 |
| acbf | lazy | 4 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 2.05 |
| acbf | lazy | 4 | 256 | 512 | 511 | ok | 1.54 | 1.78 | 0.39 |
| acbf | lazy | 4 | 256 | 1024 | 1024 | ok | 1.92 | 2.16 | 1.04 |
| acbf | lazy | 4 | 256 | 2048 | 2047 | ok | 2.67 | 2.78 | 0.99 |
| acbf | lazy | 4 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.94 |
| tcbf | full | 1 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.13 |
| tcbf | full | 1 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.15 |
| tcbf | full | 1 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.15 |
| tcbf | full | 1 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.53 |
| tcbf | full | 1 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | full | 1 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.31 |
| tcbf | full | 1 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.21 |
| tcbf | full | 1 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 1.98 |
| tcbf | full | 1 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.28 |
| tcbf | full | 1 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.86 |
| tcbf | full | 1 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.63 |
| tcbf | full | 1 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.96 |
| tcbf | full | 2 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.13 |
| tcbf | full | 2 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.15 |
| tcbf | full | 2 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.16 |
| tcbf | full | 2 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.54 |
| tcbf | full | 2 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | full | 2 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.31 |
| tcbf | full | 2 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.21 |
| tcbf | full | 2 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 1.95 |
| tcbf | full | 2 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.29 |
| tcbf | full | 2 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.86 |
| tcbf | full | 2 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.61 |
| tcbf | full | 2 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.89 |
| tcbf | full | 3 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.13 |
| tcbf | full | 3 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.15 |
| tcbf | full | 3 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | full | 3 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.55 |
| tcbf | full | 3 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | full | 3 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.31 |
| tcbf | full | 3 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.22 |
| tcbf | full | 3 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 1.96 |
| tcbf | full | 3 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.29 |
| tcbf | full | 3 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.87 |
| tcbf | full | 3 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.62 |
| tcbf | full | 3 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.92 |
| tcbf | full | 4 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | full | 4 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.16 |
| tcbf | full | 4 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | full | 4 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.57 |
| tcbf | full | 4 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | full | 4 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.32 |
| tcbf | full | 4 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.23 |
| tcbf | full | 4 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 2.00 |
| tcbf | full | 4 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.30 |
| tcbf | full | 4 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.86 |
| tcbf | full | 4 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.66 |
| tcbf | full | 4 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.88 |
| tcbf | lazy | 1 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.13 |
| tcbf | lazy | 1 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.15 |
| tcbf | lazy | 1 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.15 |
| tcbf | lazy | 1 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.52 |
| tcbf | lazy | 1 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.16 |
| tcbf | lazy | 1 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.30 |
| tcbf | lazy | 1 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.20 |
| tcbf | lazy | 1 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 2.05 |
| tcbf | lazy | 1 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.28 |
| tcbf | lazy | 1 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.86 |
| tcbf | lazy | 1 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.63 |
| tcbf | lazy | 1 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 8.26 |
| tcbf | lazy | 2 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.13 |
| tcbf | lazy | 2 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.15 |
| tcbf | lazy | 2 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.16 |
| tcbf | lazy | 2 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.54 |
| tcbf | lazy | 2 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | lazy | 2 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.30 |
| tcbf | lazy | 2 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.21 |
| tcbf | lazy | 2 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 2.00 |
| tcbf | lazy | 2 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.29 |
| tcbf | lazy | 2 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.86 |
| tcbf | lazy | 2 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.61 |
| tcbf | lazy | 2 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 8.03 |
| tcbf | lazy | 3 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.15 |
| tcbf | lazy | 3 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.15 |
| tcbf | lazy | 3 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | lazy | 3 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.56 |
| tcbf | lazy | 3 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | lazy | 3 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.31 |
| tcbf | lazy | 3 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.23 |
| tcbf | lazy | 3 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 2.01 |
| tcbf | lazy | 3 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.29 |
| tcbf | lazy | 3 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.86 |
| tcbf | lazy | 3 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.63 |
| tcbf | lazy | 3 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 8.03 |
| tcbf | lazy | 4 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | lazy | 4 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.16 |
| tcbf | lazy | 4 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.18 |
| tcbf | lazy | 4 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.59 |
| tcbf | lazy | 4 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.17 |
| tcbf | lazy | 4 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.31 |
| tcbf | lazy | 4 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.23 |
| tcbf | lazy | 4 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 2.04 |
| tcbf | lazy | 4 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.30 |
| tcbf | lazy | 4 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.86 |
| tcbf | lazy | 4 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 0.63 |
| tcbf | lazy | 4 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 7.93 |
