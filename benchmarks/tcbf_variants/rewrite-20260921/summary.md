# tcBF reconstruction variants

Device: NVIDIA RTX 5000 Ada Generation  
torch 2.7.1+cu126, Linux-6.8.0-138-generic-x86_64-with-glibc2.35  
Baseline ref: `e1e8ab4`, current tree: `1ff6031`  
N_BF = 797, measured copy bandwidth 450.7 GiB/s

Timings use a device-resident provider, so they isolate the reconstruction math. `transient` is peak allocated during the call minus the resident FFT store.

## Speed by scan size and upscale

`R = raw_scan * upscale`. Rows reaching the same R by different routes should agree.

| raw scan | upscale | R | chunk | baseline ms | fourier ms | +separable ms | current ms | total speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1x | 64 | 64 | 1.093 | 0.904 | 0.998 | 1.108 | 0.99x |
| 64 | 2x | 128 | 64 | 1.108 | 0.892 | 1.004 | 1.112 | 1.00x |
| 64 | 4x | 256 | 64 | 5.274 | 3.908 | 2.831 | 1.600 | 3.30x |
| 128 | 1x | 128 | 64 | 1.100 | 0.897 | 0.999 | 1.120 | 0.98x |
| 128 | 2x | 256 | 64 | 5.273 | 3.910 | 2.832 | 1.599 | 3.30x |
| 128 | 4x | 512 | 64 | 47.130 | 27.196 | 17.109 | 9.622 | 4.90x |
| 256 | 1x | 256 | 64 | 5.297 | 3.903 | 2.824 | 1.589 | 3.33x |
| 256 | 2x | 512 | 64 | 47.129 | 27.217 | 17.126 | 9.621 | 4.90x |
| 256 | 4x | 1024 | 64 | 190.406 | 108.823 | 69.691 | 40.737 | 4.67x |

## Transient VRAM (MiB)

| raw scan | upscale | R | chunk | baseline | fourier | +separable | current | reduction | resident |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1x | 64 | 64 | 7.0 | 7.0 | 4.1 | 2.1 | 3.30x | 25 |
| 64 | 2x | 128 | 64 | 28.1 | 28.1 | 16.3 | 8.4 | 3.35x | 100 |
| 64 | 4x | 256 | 64 | 112.3 | 112.5 | 64.8 | 33.3 | 3.38x | 398 |
| 128 | 1x | 128 | 64 | 28.1 | 28.1 | 16.3 | 8.4 | 3.35x | 100 |
| 128 | 2x | 256 | 64 | 112.3 | 112.5 | 64.8 | 33.3 | 3.38x | 398 |
| 128 | 4x | 512 | 64 | 449.0 | 450.0 | 258.5 | 132.5 | 3.39x | 1594 |
| 256 | 1x | 256 | 64 | 112.3 | 112.5 | 64.8 | 33.3 | 3.38x | 398 |
| 256 | 2x | 512 | 64 | 449.0 | 450.0 | 258.5 | 132.5 | 3.39x | 1594 |
| 256 | 4x | 1024 | 64 | 1796.0 | 1800.0 | 1033.0 | 529.0 | 3.40x | 6376 |

## Memory traffic

Passes over the `(N_BF, R, R)` store, from measured time and device bandwidth. A bandwidth-bound kernel is pinned by this number.

| raw scan | upscale | R | chunk | variant | ms | GiB/s | passes |
|---:|---:|---:|---:|---|---:|---:|---:|
| 64 | 1x | 64 | 64 | baseline | 1.093 | 22.2 | 20.3 |
| 64 | 1x | 64 | 64 | fourier | 0.904 | 26.9 | 16.7 |
| 64 | 1x | 64 | 64 | fourier_separable | 0.998 | 24.4 | 18.5 |
| 64 | 1x | 64 | 64 | current | 1.108 | 22.0 | 20.5 |
| 64 | 2x | 128 | 64 | baseline | 1.108 | 87.8 | 5.1 |
| 64 | 2x | 128 | 64 | fourier | 0.892 | 109.0 | 4.1 |
| 64 | 2x | 128 | 64 | fourier_separable | 1.004 | 96.9 | 4.7 |
| 64 | 2x | 128 | 64 | current | 1.112 | 87.5 | 5.1 |
| 64 | 4x | 256 | 64 | baseline | 5.274 | 73.8 | 6.1 |
| 64 | 4x | 256 | 64 | fourier | 3.908 | 99.6 | 4.5 |
| 64 | 4x | 256 | 64 | fourier_separable | 2.831 | 137.4 | 3.3 |
| 64 | 4x | 256 | 64 | current | 1.600 | 243.2 | 1.9 |
| 128 | 1x | 128 | 64 | baseline | 1.100 | 88.5 | 5.1 |
| 128 | 1x | 128 | 64 | fourier | 0.897 | 108.5 | 4.2 |
| 128 | 1x | 128 | 64 | fourier_separable | 0.999 | 97.4 | 4.6 |
| 128 | 1x | 128 | 64 | current | 1.120 | 86.9 | 5.2 |
| 128 | 2x | 256 | 64 | baseline | 5.273 | 73.8 | 6.1 |
| 128 | 2x | 256 | 64 | fourier | 3.910 | 99.5 | 4.5 |
| 128 | 2x | 256 | 64 | fourier_separable | 2.832 | 137.4 | 3.3 |
| 128 | 2x | 256 | 64 | current | 1.599 | 243.4 | 1.9 |
| 128 | 4x | 512 | 64 | baseline | 47.130 | 33.0 | 13.6 |
| 128 | 4x | 512 | 64 | fourier | 27.196 | 57.2 | 7.9 |
| 128 | 4x | 512 | 64 | fourier_separable | 17.109 | 91.0 | 5.0 |
| 128 | 4x | 512 | 64 | current | 9.622 | 161.8 | 2.8 |
| 256 | 1x | 256 | 64 | baseline | 5.297 | 73.5 | 6.1 |
| 256 | 1x | 256 | 64 | fourier | 3.903 | 99.7 | 4.5 |
| 256 | 1x | 256 | 64 | fourier_separable | 2.824 | 137.8 | 3.3 |
| 256 | 1x | 256 | 64 | current | 1.589 | 244.9 | 1.8 |
| 256 | 2x | 512 | 64 | baseline | 47.129 | 33.0 | 13.6 |
| 256 | 2x | 512 | 64 | fourier | 27.217 | 57.2 | 7.9 |
| 256 | 2x | 512 | 64 | fourier_separable | 17.126 | 90.9 | 5.0 |
| 256 | 2x | 512 | 64 | current | 9.621 | 161.8 | 2.8 |
| 256 | 4x | 1024 | 64 | baseline | 190.406 | 32.7 | 13.8 |
| 256 | 4x | 1024 | 64 | fourier | 108.823 | 57.2 | 7.9 |
| 256 | 4x | 1024 | 64 | fourier_separable | 69.691 | 89.3 | 5.0 |
| 256 | 4x | 1024 | 64 | current | 40.737 | 152.8 | 2.9 |

## Dispatch vs GPU work

CPU time to queue the call (no trailing sync) against full wall time. A fraction near 1.0 means the chunk loop is launch-bound and the device idles between kernels, so removing arithmetic cannot help.

| raw scan | upscale | R | chunk | variant | dispatch ms | wall ms | dispatch/wall |
|---:|---:|---:|---:|---|---:|---:|---:|
| 64 | 1x | 64 | 64 | baseline | 1.126 | 1.093 | 102.4% |
| 64 | 1x | 64 | 64 | fourier | 0.886 | 0.904 | 99.9% |
| 64 | 1x | 64 | 64 | fourier_separable | 0.996 | 0.998 | 100.0% |
| 64 | 1x | 64 | 64 | current | 1.104 | 1.108 | 100.1% |
| 64 | 2x | 128 | 64 | baseline | 1.091 | 1.108 | 97.7% |
| 64 | 2x | 128 | 64 | fourier | 0.885 | 0.892 | 100.0% |
| 64 | 2x | 128 | 64 | fourier_separable | 0.994 | 1.004 | 99.2% |
| 64 | 2x | 128 | 64 | current | 1.102 | 1.112 | 96.6% |
| 64 | 4x | 256 | 64 | baseline | 1.088 | 5.274 | 21.2% |
| 64 | 4x | 256 | 64 | fourier | 0.878 | 3.908 | 22.4% |
| 64 | 4x | 256 | 64 | fourier_separable | 0.994 | 2.831 | 35.4% |
| 64 | 4x | 256 | 64 | current | 1.106 | 1.600 | 69.0% |
| 128 | 1x | 128 | 64 | baseline | 1.094 | 1.100 | 99.2% |
| 128 | 1x | 128 | 64 | fourier | 0.883 | 0.897 | 98.8% |
| 128 | 1x | 128 | 64 | fourier_separable | 0.993 | 0.999 | 98.8% |
| 128 | 1x | 128 | 64 | current | 1.105 | 1.120 | 99.0% |
| 128 | 2x | 256 | 64 | baseline | 1.088 | 5.273 | 21.2% |
| 128 | 2x | 256 | 64 | fourier | 0.877 | 3.910 | 22.4% |
| 128 | 2x | 256 | 64 | fourier_separable | 0.987 | 2.832 | 35.2% |
| 128 | 2x | 256 | 64 | current | 1.105 | 1.599 | 68.9% |
| 128 | 4x | 512 | 64 | baseline | 0.967 | 47.130 | 2.0% |
| 128 | 4x | 512 | 64 | fourier | 0.807 | 27.196 | 3.0% |
| 128 | 4x | 512 | 64 | fourier_separable | 0.850 | 17.109 | 5.0% |
| 128 | 4x | 512 | 64 | current | 0.962 | 9.622 | 10.0% |
| 256 | 1x | 256 | 64 | baseline | 0.942 | 5.297 | 18.3% |
| 256 | 1x | 256 | 64 | fourier | 0.750 | 3.903 | 19.2% |
| 256 | 1x | 256 | 64 | fourier_separable | 0.852 | 2.824 | 30.5% |
| 256 | 1x | 256 | 64 | current | 0.972 | 1.589 | 61.0% |
| 256 | 2x | 512 | 64 | baseline | 1.019 | 47.129 | 2.2% |
| 256 | 2x | 512 | 64 | fourier | 0.753 | 27.217 | 2.8% |
| 256 | 2x | 512 | 64 | fourier_separable | 0.849 | 17.126 | 4.9% |
| 256 | 2x | 512 | 64 | current | 0.985 | 9.621 | 10.3% |
| 256 | 4x | 1024 | 64 | baseline | 0.988 | 190.406 | 0.5% |
| 256 | 4x | 1024 | 64 | fourier | 0.787 | 108.823 | 0.7% |
| 256 | 4x | 1024 | 64 | fourier_separable | 0.843 | 69.691 | 1.2% |
| 256 | 4x | 1024 | 64 | current | 0.957 | 40.737 | 2.4% |

## Correctness

Worst max-relative difference in the reconstructed image, against the baseline commit across all cases: **6.11e-07** (float32 roundoff).

Worst max-relative difference in `d(loss)/d(coeffs)`: **2.54e-07**, over R in 64, 128, 256, 512, 1024. tcBF is an AD forward pass, so gradient parity is the property that matters for fitting, not just image parity.
