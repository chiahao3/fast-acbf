# tcBF reconstruction variants

Device: NVIDIA RTX 5000 Ada Generation  
torch 2.7.1+cu126, Linux-6.8.0-138-generic-x86_64-with-glibc2.35  
Baseline ref: `e1e8ab4`, current tree: `1ff6031`  
N_BF = 797, measured copy bandwidth 449.7 GiB/s

Timings use a device-resident provider, so they isolate the reconstruction math. `transient` is peak allocated during the call minus the resident FFT store.

## Speed by scan size and upscale

`R = raw_scan * upscale`. Rows reaching the same R by different routes should agree.

| raw scan | upscale | R | chunk | baseline ms | fourier ms | +separable ms | current ms | total speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1x | 128 | 16 | 3.504 | 2.807 | 3.121 | 3.548 | 0.99x |
| 128 | 1x | 128 | 32 | 1.791 | 1.442 | 1.571 | 1.852 | 0.97x |
| 128 | 1x | 128 | 64 | 0.965 | 0.750 | 0.837 | 0.968 | 1.00x |
| 128 | 1x | 128 | 128 | 1.070 | 0.660 | 0.594 | 0.546 | 1.96x |
| 128 | 1x | 128 | 256 | 1.345 | 1.023 | 0.774 | 0.467 | 2.88x |
| 128 | 1x | 128 | 797 | 2.604 | 1.741 | 1.135 | 0.714 | 3.64x |

## Transient VRAM (MiB)

| raw scan | upscale | R | chunk | baseline | fourier | +separable | current | reduction | resident |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1x | 128 | 16 | 7.1 | 7.1 | 4.2 | 2.3 | 3.09x | 100 |
| 128 | 1x | 128 | 32 | 14.1 | 14.1 | 8.2 | 4.3 | 3.26x | 100 |
| 128 | 1x | 128 | 64 | 28.1 | 28.1 | 16.3 | 8.4 | 3.35x | 100 |
| 128 | 1x | 128 | 128 | 56.1 | 56.1 | 32.4 | 16.5 | 3.40x | 100 |
| 128 | 1x | 128 | 256 | 112.1 | 112.1 | 64.6 | 32.8 | 3.42x | 100 |
| 128 | 1x | 128 | 797 | 350.1 | 250.1 | 201.7 | 101.8 | 3.44x | 100 |

## Memory traffic

Passes over the `(N_BF, R, R)` store, from measured time and device bandwidth. A bandwidth-bound kernel is pinned by this number.

| raw scan | upscale | R | chunk | variant | ms | GiB/s | passes |
|---:|---:|---:|---:|---|---:|---:|---:|
| 128 | 1x | 128 | 16 | baseline | 3.504 | 27.8 | 16.2 |
| 128 | 1x | 128 | 16 | fourier | 2.807 | 34.7 | 13.0 |
| 128 | 1x | 128 | 16 | fourier_separable | 3.121 | 31.2 | 14.4 |
| 128 | 1x | 128 | 16 | current | 3.548 | 27.4 | 16.4 |
| 128 | 1x | 128 | 32 | baseline | 1.791 | 54.3 | 8.3 |
| 128 | 1x | 128 | 32 | fourier | 1.442 | 67.5 | 6.7 |
| 128 | 1x | 128 | 32 | fourier_separable | 1.571 | 61.9 | 7.3 |
| 128 | 1x | 128 | 32 | current | 1.852 | 52.5 | 8.6 |
| 128 | 1x | 128 | 64 | baseline | 0.965 | 100.8 | 4.5 |
| 128 | 1x | 128 | 64 | fourier | 0.750 | 129.7 | 3.5 |
| 128 | 1x | 128 | 64 | fourier_separable | 0.837 | 116.3 | 3.9 |
| 128 | 1x | 128 | 64 | current | 0.968 | 100.5 | 4.5 |
| 128 | 1x | 128 | 128 | baseline | 1.070 | 90.9 | 4.9 |
| 128 | 1x | 128 | 128 | fourier | 0.660 | 147.4 | 3.1 |
| 128 | 1x | 128 | 128 | fourier_separable | 0.594 | 163.9 | 2.7 |
| 128 | 1x | 128 | 128 | current | 0.546 | 178.3 | 2.5 |
| 128 | 1x | 128 | 256 | baseline | 1.345 | 72.4 | 6.2 |
| 128 | 1x | 128 | 256 | fourier | 1.023 | 95.1 | 4.7 |
| 128 | 1x | 128 | 256 | fourier_separable | 0.774 | 125.7 | 3.6 |
| 128 | 1x | 128 | 256 | current | 0.467 | 208.1 | 2.2 |
| 128 | 1x | 128 | 797 | baseline | 2.604 | 37.4 | 12.0 |
| 128 | 1x | 128 | 797 | fourier | 1.741 | 55.9 | 8.0 |
| 128 | 1x | 128 | 797 | fourier_separable | 1.135 | 85.7 | 5.2 |
| 128 | 1x | 128 | 797 | current | 0.714 | 136.2 | 3.3 |

## Dispatch vs GPU work

CPU time to queue the call (no trailing sync) against full wall time. A fraction near 1.0 means the chunk loop is launch-bound and the device idles between kernels, so removing arithmetic cannot help.

| raw scan | upscale | R | chunk | variant | dispatch ms | wall ms | dispatch/wall |
|---:|---:|---:|---:|---|---:|---:|---:|
| 128 | 1x | 128 | 16 | baseline | 3.586 | 3.504 | 101.5% |
| 128 | 1x | 128 | 16 | fourier | 2.797 | 2.807 | 101.5% |
| 128 | 1x | 128 | 16 | fourier_separable | 3.297 | 3.121 | 105.6% |
| 128 | 1x | 128 | 16 | current | 3.539 | 3.548 | 93.7% |
| 128 | 1x | 128 | 32 | baseline | 1.784 | 1.791 | 99.5% |
| 128 | 1x | 128 | 32 | fourier | 1.425 | 1.442 | 101.5% |
| 128 | 1x | 128 | 32 | fourier_separable | 1.603 | 1.571 | 98.8% |
| 128 | 1x | 128 | 32 | current | 1.845 | 1.852 | 100.4% |
| 128 | 1x | 128 | 64 | baseline | 0.987 | 0.965 | 101.5% |
| 128 | 1x | 128 | 64 | fourier | 0.743 | 0.750 | 99.8% |
| 128 | 1x | 128 | 64 | fourier_separable | 0.844 | 0.837 | 99.9% |
| 128 | 1x | 128 | 64 | current | 0.971 | 0.968 | 101.8% |
| 128 | 1x | 128 | 128 | baseline | 0.557 | 1.070 | 60.0% |
| 128 | 1x | 128 | 128 | fourier | 0.448 | 0.660 | 67.9% |
| 128 | 1x | 128 | 128 | fourier_separable | 0.500 | 0.594 | 84.9% |
| 128 | 1x | 128 | 128 | current | 0.548 | 0.546 | 96.4% |
| 128 | 1x | 128 | 256 | baseline | 0.321 | 1.345 | 23.6% |
| 128 | 1x | 128 | 256 | fourier | 0.260 | 1.023 | 26.0% |
| 128 | 1x | 128 | 256 | fourier_separable | 0.286 | 0.774 | 36.9% |
| 128 | 1x | 128 | 256 | current | 0.346 | 0.467 | 73.9% |
| 128 | 1x | 128 | 797 | baseline | 0.093 | 2.604 | 3.6% |
| 128 | 1x | 128 | 797 | fourier | 0.096 | 1.741 | 5.5% |
| 128 | 1x | 128 | 797 | fourier_separable | 0.105 | 1.135 | 9.3% |
| 128 | 1x | 128 | 797 | current | 0.120 | 0.714 | 16.7% |

## Correctness

Worst max-relative difference in the reconstructed image, against the baseline commit across all cases: **5.34e-07** (float32 roundoff).

Worst max-relative difference in `d(loss)/d(coeffs)`: **2.79e-07**, over R in 128. tcBF is an AD forward pass, so gradient parity is the property that matters for fitting, not just image parity.
