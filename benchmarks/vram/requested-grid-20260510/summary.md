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
| acbf | full | 1 | 1.63 | 3.25 | 6.50 | 13.00 |
| acbf | full | 2 | 2.62 | 5.25 | 10.50 | 21.00 |
| acbf | full | 3 | 3.87 | 7.75 | 15.50 | 31.00 est |
| acbf | full | 4 | 5.37 | 10.75 | 21.49 | 43.00 est |
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
| acbf | full | 1 | 64 | 1024 | 1024 | ok | 0.20 | 0.24 | 0.19 |
| acbf | full | 1 | 64 | 2048 | 2047 | ok | 0.41 | 0.45 | 0.22 |
| acbf | full | 1 | 64 | 4096 | 4096 | ok | 0.81 | 0.88 | 0.35 |
| acbf | full | 1 | 128 | 512 | 511 | ok | 0.41 | 0.46 | 0.20 |
| acbf | full | 1 | 128 | 1024 | 1024 | ok | 0.81 | 0.87 | 0.31 |
| acbf | full | 1 | 128 | 2048 | 2047 | ok | 1.63 | 1.68 | 0.38 |
| acbf | full | 1 | 128 | 4096 | 4096 | ok | 3.25 | 3.30 | 0.88 |
| acbf | full | 1 | 256 | 512 | 511 | ok | 1.63 | 1.81 | 0.38 |
| acbf | full | 1 | 256 | 1024 | 1024 | ok | 3.25 | 3.44 | 0.82 |
| acbf | full | 1 | 256 | 2048 | 2047 | ok | 6.50 | 6.69 | 1.18 |
| acbf | full | 1 | 256 | 4096 | 4096 | ok | 13.00 | 13.19 | 3.64 |
| acbf | full | 2 | 64 | 512 | 511 | ok | 0.17 | 0.24 | 0.16 |
| acbf | full | 2 | 64 | 1024 | 1024 | ok | 0.33 | 0.44 | 0.19 |
| acbf | full | 2 | 64 | 2048 | 2047 | ok | 0.66 | 0.86 | 0.23 |
| acbf | full | 2 | 64 | 4096 | 4096 | ok | 1.31 | 1.71 | 0.37 |
| acbf | full | 2 | 128 | 512 | 511 | ok | 0.66 | 0.73 | 0.21 |
| acbf | full | 2 | 128 | 1024 | 1024 | ok | 1.31 | 1.38 | 0.31 |
| acbf | full | 2 | 128 | 2048 | 2047 | ok | 2.63 | 2.71 | 0.39 |
| acbf | full | 2 | 128 | 4096 | 4096 | ok | 5.25 | 5.34 | 0.89 |
| acbf | full | 2 | 256 | 512 | 511 | ok | 2.62 | 2.91 | 0.41 |
| acbf | full | 2 | 256 | 1024 | 1024 | ok | 5.25 | 5.53 | 0.85 |
| acbf | full | 2 | 256 | 2048 | 2047 | ok | 10.50 | 10.78 | 1.23 |
| acbf | full | 2 | 256 | 4096 | 4096 | ok | 21.00 | 21.28 | 3.67 |
| acbf | full | 3 | 64 | 512 | 511 | ok | 0.25 | 0.29 | 0.17 |
| acbf | full | 3 | 64 | 1024 | 1024 | ok | 0.48 | 0.51 | 0.20 |
| acbf | full | 3 | 64 | 2048 | 2047 | ok | 0.97 | 0.99 | 0.24 |
| acbf | full | 3 | 64 | 4096 | 4096 | ok | 1.94 | 1.96 | 0.38 |
| acbf | full | 3 | 128 | 512 | 511 | ok | 0.97 | 1.08 | 0.21 |
| acbf | full | 3 | 128 | 1024 | 1024 | ok | 1.94 | 2.05 | 0.32 |
| acbf | full | 3 | 128 | 2048 | 2047 | ok | 3.88 | 3.98 | 0.40 |
| acbf | full | 3 | 128 | 4096 | 4096 | ok | 7.75 | 7.86 | 0.90 |
| acbf | full | 3 | 256 | 512 | 511 | ok | 3.87 | 4.26 | 0.42 |
| acbf | full | 3 | 256 | 1024 | 1024 | ok | 7.75 | 8.14 | 0.89 |
| acbf | full | 3 | 256 | 2048 | 2047 | ok | 15.50 | 15.89 | 1.33 |
| acbf | full | 3 | 256 | 4096 | 4096 | oom_estimated est | 31.00 |  | 3.60 |
| acbf | full | 4 | 64 | 512 | 511 | ok | 0.34 | 0.38 | 0.17 |
| acbf | full | 4 | 64 | 1024 | 1024 | ok | 0.67 | 0.71 | 0.20 |
| acbf | full | 4 | 64 | 2048 | 2047 | ok | 1.34 | 1.38 | 0.25 |
| acbf | full | 4 | 64 | 4096 | 4096 | ok | 2.69 | 2.72 | 0.41 |
| acbf | full | 4 | 128 | 512 | 511 | ok | 1.34 | 1.48 | 0.22 |
| acbf | full | 4 | 128 | 1024 | 1024 | ok | 2.69 | 2.82 | 0.32 |
| acbf | full | 4 | 128 | 2048 | 2047 | ok | 5.37 | 5.52 | 0.43 |
| acbf | full | 4 | 128 | 4096 | 4096 | ok | 10.75 | 10.90 | 0.97 |
| acbf | full | 4 | 256 | 512 | 511 | ok | 5.37 | 5.88 | 0.44 |
| acbf | full | 4 | 256 | 1024 | 1024 | ok | 10.75 | 11.27 | 0.94 |
| acbf | full | 4 | 256 | 2048 | 2047 | ok | 21.49 | 22.01 | 1.42 |
| acbf | full | 4 | 256 | 4096 | 4096 | oom_estimated est | 43.00 |  | 3.68 |
| acbf | lazy | 1 | 64 | 512 | 511 | ok | 0.05 | 0.09 | 0.17 |
| acbf | lazy | 1 | 64 | 1024 | 1024 | ok | 0.08 | 0.09 | 0.19 |
| acbf | lazy | 1 | 64 | 2048 | 2047 | ok | 0.16 | 0.17 | 0.22 |
| acbf | lazy | 1 | 64 | 4096 | 4096 | ok | 0.31 | 0.33 | 0.35 |
| acbf | lazy | 1 | 128 | 512 | 511 | ok | 0.19 | 0.20 | 0.20 |
| acbf | lazy | 1 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.30 |
| acbf | lazy | 1 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.37 |
| acbf | lazy | 1 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.87 |
| acbf | lazy | 1 | 256 | 512 | 511 | ok | 0.74 | 0.78 | 0.38 |
| acbf | lazy | 1 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.82 |
| acbf | lazy | 1 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.17 |
| acbf | lazy | 1 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.67 |
| acbf | lazy | 2 | 64 | 512 | 511 | ok | 0.07 | 0.09 | 0.16 |
| acbf | lazy | 2 | 64 | 1024 | 1024 | ok | 0.09 | 0.12 | 0.19 |
| acbf | lazy | 2 | 64 | 2048 | 2047 | ok | 0.16 | 0.18 | 0.23 |
| acbf | lazy | 2 | 64 | 4096 | 4096 | ok | 0.31 | 0.33 | 0.36 |
| acbf | lazy | 2 | 128 | 512 | 511 | ok | 0.25 | 0.26 | 0.21 |
| acbf | lazy | 2 | 128 | 1024 | 1024 | ok | 0.34 | 0.37 | 0.31 |
| acbf | lazy | 2 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.38 |
| acbf | lazy | 2 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.90 |
| acbf | lazy | 2 | 256 | 512 | 511 | ok | 0.96 | 1.03 | 0.40 |
| acbf | lazy | 2 | 256 | 1024 | 1024 | ok | 1.34 | 1.36 | 0.84 |
| acbf | lazy | 2 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.24 |
| acbf | lazy | 2 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 4.05 |
| acbf | lazy | 3 | 64 | 512 | 511 | ok | 0.08 | 0.10 | 0.17 |
| acbf | lazy | 3 | 64 | 1024 | 1024 | ok | 0.11 | 0.12 | 0.19 |
| acbf | lazy | 3 | 64 | 2048 | 2047 | ok | 0.16 | 0.18 | 0.24 |
| acbf | lazy | 3 | 64 | 4096 | 4096 | ok | 0.31 | 0.34 | 0.38 |
| acbf | lazy | 3 | 128 | 512 | 511 | ok | 0.31 | 0.36 | 0.21 |
| acbf | lazy | 3 | 128 | 1024 | 1024 | ok | 0.41 | 0.46 | 0.32 |
| acbf | lazy | 3 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.41 |
| acbf | lazy | 3 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.92 |
| acbf | lazy | 3 | 256 | 512 | 511 | ok | 1.23 | 1.38 | 0.42 |
| acbf | lazy | 3 | 256 | 1024 | 1024 | ok | 1.60 | 1.63 | 0.88 |
| acbf | lazy | 3 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.34 |
| acbf | lazy | 3 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 4.13 |
| acbf | lazy | 4 | 64 | 512 | 511 | ok | 0.10 | 0.12 | 0.17 |
| acbf | lazy | 4 | 64 | 1024 | 1024 | ok | 0.13 | 0.15 | 0.20 |
| acbf | lazy | 4 | 64 | 2048 | 2047 | ok | 0.17 | 0.19 | 0.25 |
| acbf | lazy | 4 | 64 | 4096 | 4096 | ok | 0.31 | 0.35 | 0.38 |
| acbf | lazy | 4 | 128 | 512 | 511 | ok | 0.39 | 0.45 | 0.21 |
| acbf | lazy | 4 | 128 | 1024 | 1024 | ok | 0.48 | 0.54 | 0.34 |
| acbf | lazy | 4 | 128 | 2048 | 2047 | ok | 0.67 | 0.70 | 0.43 |
| acbf | lazy | 4 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 1.00 |
| acbf | lazy | 4 | 256 | 512 | 511 | ok | 1.54 | 1.78 | 0.43 |
| acbf | lazy | 4 | 256 | 1024 | 1024 | ok | 1.92 | 2.16 | 0.93 |
| acbf | lazy | 4 | 256 | 2048 | 2047 | ok | 2.67 | 2.78 | 1.43 |
| acbf | lazy | 4 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 4.05 |
| tcbf | full | 1 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | full | 1 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.16 |
| tcbf | full | 1 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | full | 1 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.30 |
| tcbf | full | 1 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.18 |
| tcbf | full | 1 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.28 |
| tcbf | full | 1 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.34 |
| tcbf | full | 1 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.83 |
| tcbf | full | 1 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.34 |
| tcbf | full | 1 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.74 |
| tcbf | full | 1 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.06 |
| tcbf | full | 1 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.27 |
| tcbf | full | 2 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | full | 2 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.16 |
| tcbf | full | 2 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.20 |
| tcbf | full | 2 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.32 |
| tcbf | full | 2 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | full | 2 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.28 |
| tcbf | full | 2 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.35 |
| tcbf | full | 2 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.84 |
| tcbf | full | 2 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.35 |
| tcbf | full | 2 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.74 |
| tcbf | full | 2 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.07 |
| tcbf | full | 2 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.45 |
| tcbf | full | 3 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | full | 3 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.17 |
| tcbf | full | 3 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.21 |
| tcbf | full | 3 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.34 |
| tcbf | full | 3 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.18 |
| tcbf | full | 3 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.28 |
| tcbf | full | 3 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.36 |
| tcbf | full | 3 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.85 |
| tcbf | full | 3 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.35 |
| tcbf | full | 3 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.74 |
| tcbf | full | 3 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.09 |
| tcbf | full | 3 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.50 |
| tcbf | full | 4 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.15 |
| tcbf | full | 4 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.17 |
| tcbf | full | 4 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.21 |
| tcbf | full | 4 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.35 |
| tcbf | full | 4 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | full | 4 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.29 |
| tcbf | full | 4 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.36 |
| tcbf | full | 4 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.89 |
| tcbf | full | 4 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.34 |
| tcbf | full | 4 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.75 |
| tcbf | full | 4 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.11 |
| tcbf | full | 4 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.47 |
| tcbf | lazy | 1 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | lazy | 1 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.16 |
| tcbf | lazy | 1 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | lazy | 1 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.30 |
| tcbf | lazy | 1 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.18 |
| tcbf | lazy | 1 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.27 |
| tcbf | lazy | 1 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.35 |
| tcbf | lazy | 1 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.83 |
| tcbf | lazy | 1 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.35 |
| tcbf | lazy | 1 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.75 |
| tcbf | lazy | 1 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.06 |
| tcbf | lazy | 1 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.38 |
| tcbf | lazy | 2 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | lazy | 2 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.16 |
| tcbf | lazy | 2 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | lazy | 2 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.32 |
| tcbf | lazy | 2 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.18 |
| tcbf | lazy | 2 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.28 |
| tcbf | lazy | 2 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.35 |
| tcbf | lazy | 2 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.84 |
| tcbf | lazy | 2 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.34 |
| tcbf | lazy | 2 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.76 |
| tcbf | lazy | 2 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.08 |
| tcbf | lazy | 2 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.33 |
| tcbf | lazy | 3 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.14 |
| tcbf | lazy | 3 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.17 |
| tcbf | lazy | 3 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.21 |
| tcbf | lazy | 3 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.34 |
| tcbf | lazy | 3 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | lazy | 3 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.28 |
| tcbf | lazy | 3 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.36 |
| tcbf | lazy | 3 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.86 |
| tcbf | lazy | 3 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.34 |
| tcbf | lazy | 3 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.76 |
| tcbf | lazy | 3 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.08 |
| tcbf | lazy | 3 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.65 |
| tcbf | lazy | 4 | 64 | 512 | 511 | ok | 0.04 | 0.05 | 0.15 |
| tcbf | lazy | 4 | 64 | 1024 | 1024 | ok | 0.08 | 0.08 | 0.17 |
| tcbf | lazy | 4 | 64 | 2048 | 2047 | ok | 0.16 | 0.16 | 0.21 |
| tcbf | lazy | 4 | 64 | 4096 | 4096 | ok | 0.31 | 0.32 | 0.35 |
| tcbf | lazy | 4 | 128 | 512 | 511 | ok | 0.16 | 0.16 | 0.19 |
| tcbf | lazy | 4 | 128 | 1024 | 1024 | ok | 0.31 | 0.31 | 0.29 |
| tcbf | lazy | 4 | 128 | 2048 | 2047 | ok | 0.63 | 0.63 | 0.38 |
| tcbf | lazy | 4 | 128 | 4096 | 4096 | ok | 1.25 | 1.25 | 0.88 |
| tcbf | lazy | 4 | 256 | 512 | 511 | ok | 0.63 | 0.63 | 0.35 |
| tcbf | lazy | 4 | 256 | 1024 | 1024 | ok | 1.25 | 1.25 | 0.75 |
| tcbf | lazy | 4 | 256 | 2048 | 2047 | ok | 2.50 | 2.50 | 1.07 |
| tcbf | lazy | 4 | 256 | 4096 | 4096 | ok | 5.00 | 5.00 | 3.37 |
