# tcBF Fourier-Space Rewrite (2026-09-21)

**Hardware**: NVIDIA RTX 5000 Ada Generation (32 GiB VRAM) · torch 2.7.1+cu126 · Linux 6.8.0  
**Commits**: baseline `e1e8ab4` ("Version to 0.7.0") → current `1ff6031` ("Accumulate tcBF/acBF in Fourier space; separable tcBF phase ramp")  
**Script**: `scripts/benchmark_tcbf_variants.py`  
**Artifacts**: `benchmarks/tcbf_variants/rewrite-20260921/` (R sweep), `benchmarks/tcbf_variants/chunk-floor-20260921/` (chunk sweep)  
**Measured device copy bandwidth**: 450 GiB/s

---

## 1. Motivation

`reconstruct_tcbf` is the AD forward pass. Every optimizer step over the aberration
coefficients runs it once forward and once backward, so its cost sets the cost of
fitting, and its transient VRAM sets the largest reconstruction grid that fits.

The 0.7.0 implementation inverse-transformed every shifted virtual bright-field image
before summing:

```python
for chunk in cache.chunks:
    img_fft_chunk = provider.get_chunk(...)            # (b, R, R) complex64, a view
    ramp     = shift_dx * qx_grid + shift_dy * qy_grid  # (b, R, R) float32
    shift_op = torch.exp(neg_two_pi_j * ramp)           # (b, R, R) complex64
    tcBF_total += torch.sum(torch.fft.ifft2(img_fft_chunk * shift_op).real, dim=0)
```

That is `N_BF` inverse FFTs of an `(R, R)` grid, a full 2D phase ramp materialized per
chunk, and a stack of `(b, R, R)` intermediates alive at once. All three are avoidable.
`1ff6031` removes all three. This benchmark peels them off one at a time, so the
contribution of each is measured rather than inferred.

---

## 2. Cost model and units

Two quantities explain everything below.

**`U`** — one chunk-sized complex64 buffer, `U = chunk_size · R² · 8` bytes. This is the
natural unit for transient VRAM: at `R=256, b=64`, `U = 32 MiB`; at `R=1024, b=64`,
`U = 512 MiB`.

**Passes over the data** — the wall time expressed in units of the time it would take to
stream the resident `(N_BF, R, R)` FFT store once at measured device bandwidth. For a
bandwidth-bound kernel this is the number that pins performance, and 1.0 is the floor.
It is an upper bound on true traffic, since compute time is folded in.

---

## 3. The three changes

### Change 1 — accumulate in Fourier space, inverse-transform once

`ifft2` and `Re(·)` are both linear in the detector index `b`, so

```
sum_b Re(ifft2(F_b · S_b))  ==  Re(ifft2(sum_b F_b · S_b))
```

Moving the sum inside turns `N_BF` inverse FFTs into exactly one. The accumulator
becomes a single complex `(R, R)` spectrum instead of a real `(R, R)` image.

**What it buys**: 13.6 → 7.9 passes at `R=512`; **1.73x**. The per-chunk `ifft2` alone
was costing ~5.7 passes over the store.

**What it does not buy**: any memory. Transient stays at 3.5 U (112.3 → 112.5 MiB at
`R=256`), because the peak is set by building the 2D phase ramp, which this change
does not touch. This is exactly why one change is not enough.

### Change 2 — build the phase ramp as two 1D factors

The tcBF shift operator is a pure translation, so its ramp factorizes:

```
exp(-2πi (dx_b·qx + dy_b·qy))  ==  exp(-2πi dx_b·qx) · exp(-2πi dy_b·qy)
```

Building the two 1D factors costs `b·(Ry + Rx)` transcendentals instead of the `b·Ry·Rx`
needed for the full 2D grid — a factor of `R/2` fewer, which at `R=1024` is 512x
(8.4 × 10⁸ complex exponentials per scan down to 1.6 × 10⁶).

**What it buys**: 7.9 → 5.0 passes; **1.59x**. Transient drops 3.5 U → 2.0 U, because
the `(b, R, R)` float ramp and the `(b, R, R)` complex `exp` output both disappear.

**Scope note**: this is specific to tcBF. acBF's transfer function `T_b(q)` is genuinely
2D and does not factorize, so it keeps the full-grid form.

### Change 3 — fuse both factors and the detector sum into one einsum

```python
spectrum += torch.einsum('byx, bx, by -> yx', img_fft_chunk, phase_x, phase_y)
```

Written as `f * phase_x * phase_y` then `.sum(0)`, change 2 still walks the chunk three
times — multiply by `phase_x` into a `(b,R,R)` temp, multiply by `phase_y` into a second
one, reduce. The einsum states the whole contraction at once, so only one intermediate
is ever live.

**What it buys**: 5.0 → 2.8 passes; **1.78x**. Transient drops 2.0 U → 1.0 U.

**Why it needs the other two.** This is the answer to "why three changes":

- It can only contract over `b` because change 1 made the sum-over-`b` happen in
  Fourier space. In the baseline each `b` goes through its own `ifft2`, so there is no
  contraction to fuse.
- It can only take `phase_x` and `phase_y` as separate rank-2 operands because change 2
  produced two 1D factors. A full 2D `shift_op` is just one more `(b,R,R)` operand, and
  fusing it changes nothing.

Changes 1 and 2 are independently valid and independently measured, but change 3 — the
one that gets transient down to a single buffer and traffic under 3 passes — is only
reachable on top of both.

The three multiply: **1.73 × 1.59 × 1.78 = 4.90x**, matching the measured 4.90x at
`R=512` exactly.

---

## 4. Method

The script builds a four-rung ladder, each rung adding exactly one change to the one
below it:

| Variant | Definition |
|---|---|
| `baseline` | `git show e1e8ab4:src/fast_acbf/core/tcbf.py`, exec'd verbatim |
| `fourier` | + change 1 |
| `fourier_separable` | + change 2 |
| `current` | + change 3 (the committed `src/`, imported directly) |

The baseline is loaded from git rather than copy-pasted, so the comparison cannot
silently drift from what that commit actually ran.

The provider is a device-resident stub mirroring `imagefft_storage='device'`. Keeping
the FFT store resident removes H2D transfer from the measurement, so the reconstruction
math is what varies. `host` and `on_the_fly` modes are transfer-bound and will not show
these differences.

Reported memory is the **transient** working set — peak allocated during the call minus
the resident store — which is what decides whether a given `chunk_size` fits.

`N_BF = 797`, 20 timed repeats after 5 warmups, minimum reported.

Each case additionally records the max-relative difference against the baseline in both
the image and `d(loss)/d(coeffs)` (§8), and, under `--probe-dispatch`, the CPU-only
launch time alongside the full wall time (§7).

---

## 5. Speed, and why the two sweep axes are one axis

`R = raw_scan × upscale`, and nothing downstream sees anything but `R`. The grid
deliberately reaches the same `R` by different routes so this is shown rather than
asserted:

| raw scan | upscale | R | baseline ms | fourier | +separable | current | total |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1x | 64 | 1.093 | 0.904 | 0.998 | 1.108 | 0.99x |
| 64 | 2x | **128** | 1.108 | 0.892 | 1.004 | 1.112 | 1.00x |
| 128 | 1x | **128** | 1.100 | 0.897 | 0.999 | 1.120 | 0.98x |
| 64 | 4x | **256** | 5.274 | 3.908 | 2.831 | 1.600 | 3.30x |
| 128 | 2x | **256** | 5.273 | 3.910 | 2.832 | 1.599 | 3.30x |
| 256 | 1x | **256** | 5.297 | 3.903 | 2.824 | 1.589 | 3.33x |
| 128 | 4x | **512** | 47.130 | 27.196 | 17.109 | 9.622 | 4.90x |
| 256 | 2x | **512** | 47.129 | 27.217 | 17.126 | 9.621 | 4.90x |
| 256 | 4x | 1024 | 190.406 | 108.823 | 69.691 | 40.737 | 4.67x |

The three routes to `R=256` agree to 0.7%, and the two routes to `R=512` to 0.01%.
**Upscale costs exactly what the `R` it produces costs**, and it enters quadratically:
`R` 512 → 1024 is 4.04x the baseline time and 4.23x the current time for 4x the pixels.
Doubling `upscale` quadruples the reconstruction cost regardless of the raw scan size it
was applied to.

Effective bandwidth, showing where the time actually goes:

| R | baseline | fourier | +separable | current | roofline |
|---:|---:|---:|---:|---:|---:|
| 512 | 33.0 GiB/s (13.6 passes) | 57.2 (7.9) | 91.0 (5.0) | 161.8 (**2.8**) | 450 GiB/s (1.0) |
| 1024 | 32.7 (13.8) | 57.2 (7.9) | 89.3 (5.0) | 152.8 (**2.9**) | 450 (1.0) |

Each change removes roughly one chunk-traversal's worth of traffic, and the speedups
track the pass counts rather than any change in arithmetic intensity — this is a
memory-traffic problem throughout.

---

## 6. Transient VRAM

Expressed in `U` (one chunk-sized complex64 buffer), the ratios are R-independent:

| R | U (MiB) | baseline | fourier | +separable | current |
|---:|---:|---:|---:|---:|---:|
| 256 | 32 | 112.3 (3.50 U) | 112.5 (3.52 U) | 64.8 (2.02 U) | 33.3 (**1.04 U**) |
| 512 | 128 | 449.0 (3.51 U) | 450.0 (3.52 U) | 258.5 (2.02 U) | 132.5 (**1.04 U**) |
| 1024 | 512 | 1796.0 (3.51 U) | 1800.0 (3.52 U) | 1033.0 (2.02 U) | 529.0 (**1.03 U**) |

The baseline's 3.5 U is the float32 ramp (0.5 U) plus three simultaneously-live complex
`(b, R, R)` buffers — the scaled ramp, `shift_op`, and the product/transform output.
Change 1 leaves this untouched. Change 2 eliminates the ramp and its `exp` output,
leaving the two chunk temporaries of the chained multiply. Change 3 leaves one.

**3.4x reduction overall**, and it is the change-2-plus-3 result, not change 1's.

---

## 7. Why nothing moves below R = 256

The `R ≤ 128` rows show no speedup at all — `current` lands within 2% of baseline, on
the slow side.
That is not a regression in the math; it is a floor that neither variant is above.
Sweeping `chunk_size` at fixed `R=128` isolates it:

| chunk | n_chunks | baseline ms | current ms | µs/chunk (base) | µs/chunk (cur) | dispatch/wall (cur) | cur transient |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 50 | 3.504 | 3.548 | 70.1 | 71.0 | ~100% | 2.3 MiB |
| 32 | 25 | 1.791 | 1.852 | 71.6 | 74.1 | ~100% | 4.3 MiB |
| 64 | 13 | 0.965 | 0.968 | 74.2 | 74.5 | ~100% | 8.4 MiB |
| 128 | 7 | 1.070 | **0.546** | 152.9 | 78.0 | 96.4% | 16.5 MiB |
| 256 | 4 | 1.345 | **0.467** | 336.2 | 116.9 | 73.9% | 32.8 MiB |
| 797 | 1 | 2.604 | 0.714 | 2604.2 | 714.5 | 16.7% | 101.8 MiB |

Below `chunk=128`, time is exactly linear in `n_chunks` at **~70-75 µs per chunk
iteration, identical for both variants**. That signature says the floor is per-iteration
overhead in the Python chunk loop rather than GPU work, and timing the call with and
without a trailing `cuda.synchronize()` confirms it directly — the first number is how
long the CPU takes just to queue the work:

| R | chunk | variant | dispatch (ms) | wall (ms) | dispatch / wall |
|---:|---:|---|---:|---:|---:|
| 64 | 64 | current | 1.104 | 1.108 | **~100%** |
| 128 | 64 | baseline | 0.987 | 0.965 | **~100%** |
| 128 | 64 | current | 0.971 | 0.968 | **~100%** |
| 128 | 256 | baseline | 0.321 | 1.345 | 23.6% |
| 128 | 256 | current | 0.346 | 0.467 | 73.9% |
| 256 | 64 | baseline | 1.088 | 5.274 | 21.2% |
| 256 | 64 | current | 1.106 | 1.600 | 69.0% |
| 512 | 64 | baseline | 0.967 | 47.130 | 2.0% |
| 512 | 64 | current | 0.962 | 9.622 | 10.0% |
| 1024 | 64 | current | 0.957 | 40.737 | 2.4% |

(Dispatch and wall are timed in separate loops and each reported as a minimum, so a
fully launch-bound case lands at 100% give or take a couple of points.)

At `R ≤ 128, chunk=64` the GPU finishes each chunk before the CPU can queue the next
one: essentially the whole call is dispatch, and the device is idle. No amount of
arithmetic removed from the loop body can help there, which is exactly what the `R ≤ 128`
rows show. The dispatch cost is ~1.0 ms regardless of `R` (it scales with `n_chunks`, not
grid size), so by `R=512` it is 2% of the call and the math dominates. `R=256` is the
crossover: the baseline is already GPU-bound there (21% dispatch) while `current`, having
removed enough work, has fallen back to 69%. Above `chunk=128` the GPU work per
iteration exceeds the floor, the baseline turns over into its traffic wall, and the
rewrite's advantage appears.

**This is the actionable finding.** The default `chunk_size=64`
(`recon/cache.py:44`, `recon/reconstructor.py:179`) was a reasonable choice when a chunk
cost 3.5 U. Now that it costs 1.0 U, the same VRAM budget buys a 3.4x larger chunk. At
`R=128`, moving to `chunk=256` takes the current implementation from 0.968 ms to
**0.467 ms (2.1x)** for 32.8 MiB transient — still only slightly above what the *old*
code needed at `chunk=64` (28.1 MiB). The memory saving converts directly into speed in
exactly the regime where the math changes alone did nothing.

Note that `chunk=797` (whole scan, one iteration) is *not* optimal — 0.714 ms, worse than
`chunk=256` — so this wants a tuned default, not an unbounded one. At `chunk=256` the
current implementation is already 74% dispatch-bound again, so `R=128` has roughly one
more doubling of headroom and no more.

---

## 8. Correctness

Across all 9 `(raw_scan, upscale)` cases and all three intermediate variants:

- **Image**: worst max-relative difference vs. the baseline commit **6.11 × 10⁻⁷**, i.e.
  float32 roundoff.
- **`d(loss)/d(coeffs)`**: worst max-relative difference **2.54 × 10⁻⁷**, checked at
  `R` = 64, 128, 256, 512, 1024 under a fixed ramp-weighted loss (a ramp rather than a
  plain mean, so the loss probes the whole image and not just its DC component).

Gradient parity is the property that matters here — tcBF is an AD forward pass, and
`coeffs` is what the optimizer moves. Matching images would not on its own guarantee
matching fits.

One incidental result worth recording: **change 1's gradient is bitwise identical to the
baseline's**, not merely close. The forward differs in rounding (sum-then-transform vs.
transform-then-sum), but the backward does not: the adjoint of `ifft2` applied to the
loss weight is computed once and broadcast over `b` in both forms, so the arithmetic
reaching `coeffs` is the same operations in the same order. Changes 2 and 3 alter the
phase construction and so do differ, at the 10⁻⁷ level.

---

## 9. Summary

| | change 1 | change 2 | change 3 | combined |
|---|---|---|---|---|
| | Fourier accumulation | separable ramp | fused einsum | |
| Time (R=512) | 1.73x | 1.59x | 1.78x | **4.90x** |
| Transient VRAM | 1.00x | 1.74x | 1.95x | **3.39x** |
| Passes over data | 13.6 → 7.9 | → 5.0 | → 2.8 | 13.6 → **2.8** |
| Needs | — | — | changes 1 + 2 | |

No one change would have done. The three time contributions are close in ratio
(1.73x / 1.59x / 1.78x) and compound rather than overlap, because each removes a
different traversal of the chunk: change 1 the per-image inverse FFT, change 2 the 2D
ramp construction, change 3 the chained multiply. On memory they are not close — change
1 contributes nothing at all, and the entire 3.39x comes from changes 2 and 3. And
change 3, the single largest time win, is only legal once the first two are in place.

**Where it applies**: `imagefft_storage='device'` with `R ≥ 256`. Below that the chunk
loop is dispatch-bound (§7), and in `host`/`on_the_fly` modes the H2D transfer dominates.

**Remaining headroom**: `current` runs at ~2.8-2.9 passes over the store against a
1.0-pass roofline, because `torch.einsum` still materializes one `(b, R, R)` intermediate
rather than contracting in registers. A fused kernel (a custom Triton kernel, or
`torch.compile` over the chunk body) could recover up to another ~2.8x. Worth attempting
only after the `chunk_size` default is retuned, which is cheaper and helps the small-`R`
regime that the rewrite did not.
