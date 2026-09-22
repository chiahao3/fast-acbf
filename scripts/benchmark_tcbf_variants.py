#!/usr/bin/env python
"""Benchmark the tcBF reconstruction rewrite: baseline commit vs current working tree.

The rewrite that landed in 1ff6031 stacks three independent changes on top of the
0.7.0 implementation.  This script measures them one at a time so the contribution of
each is visible rather than inferred:

    baseline           git <--baseline-ref>:src/fast_acbf/core/tcbf.py, loaded verbatim
    fourier            + accumulate in Fourier space, one ifft2 at the end
    fourier_separable  + build the shift ramp as two 1D phase factors
    current            + fuse both factors and the detector sum into one einsum
                         (this is the committed src, imported directly)

Sweep axes are the raw scan size and the upscale factor.  These are not independent --
the reconstruction grid is R = raw_scan * upscale and nothing downstream sees anything
but R -- so the grid deliberately includes several (raw_scan, upscale) routes to the
same R.  Those rows should agree, which is the point: upscale costs exactly what the R
it produces costs, and it enters quadratically.

Timings isolate the reconstruction math.  The provider is a device-resident stub that
mirrors imagefft_storage='device'; host and on_the_fly modes are H2D-bound and will not
show these differences.  Memory reported is the transient working set of the call (peak
allocated minus the resident FFT store), which is what decides whether a chunk_size fits.

Usage:
    python scripts/benchmark_tcbf_variants.py --label my-run
    python scripts/benchmark_tcbf_variants.py --raw-scans 128 --upscales 1,2,4
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from fast_acbf.recon.cache import build_tcbf_cache

DEFAULT_OUT_DIR = Path("benchmarks/tcbf_variants")
DEFAULT_BASELINE_REF = "e1e8ab4"  # 'Version to 0.7.0', the commit before the rewrite
TCBF_SRC_PATH = "src/fast_acbf/core/tcbf.py"

# Detector/optics constants. Only the BF-disk radius matters for the shift magnitudes,
# so these are representative rather than tied to any particular dataset.
WAVELENGTH = 0.0197
MAX_ALPHA = 25e-3
ORDER_KEYS = [(1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2), (3, 4)]

VARIANT_ORDER = ("baseline", "fourier", "fourier_separable", "current")


# ───────────────────────────── baseline loading ──────────────────────────────

def load_baseline_reconstruct(ref: str):
    """Exec the tcBF module as it existed at ``ref`` and return its reconstruct_tcbf.

    Loaded from git rather than copy-pasted so the comparison cannot silently drift
    away from what that commit actually ran.
    """
    try:
        source = subprocess.check_output(
            ["git", "show", f"{ref}:{TCBF_SRC_PATH}"],
            text=True,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Could not read {TCBF_SRC_PATH} at ref {ref!r}: {exc.stderr.strip()}"
        ) from exc

    namespace: dict = {"__name__": f"_tcbf_baseline_{ref}"}
    exec(compile(source, f"<git:{ref}:{TCBF_SRC_PATH}>", "exec"), namespace)
    if "reconstruct_tcbf" not in namespace:
        raise SystemExit(f"{TCBF_SRC_PATH} at {ref} defines no reconstruct_tcbf.")
    return namespace["reconstruct_tcbf"]


# ──────────────────────────── intermediate variants ──────────────────────────
# Each adds exactly one change to the one above it, so the ladder attributes the
# speedup to individual changes instead of to the rewrite as a whole.

def tcbf_fourier(provider, qx_grid, qy_grid, cache, coeffs, device):
    """Change 1 only: ifft2 and Re() are linear in the detector index, so accumulate
    the shifted spectra and inverse-transform once instead of N_BF times."""
    neg_two_pi_j = torch.tensor(-2.0j * torch.pi, dtype=torch.complex64, device=device)
    Ry_out, Rx_out = qy_grid.shape[-2], qx_grid.shape[-1]
    spectrum = torch.zeros((Ry_out, Rx_out), dtype=torch.complex64, device=device)
    for chunk in cache.chunks:
        img_fft_chunk = provider.get_chunk(chunk['start'], chunk['end'])
        shift_dx = torch.einsum('k, kb -> b', coeffs, chunk['b_dx']).view(-1, 1, 1)
        shift_dy = torch.einsum('k, kb -> b', coeffs, chunk['b_dy']).view(-1, 1, 1)
        ramp = shift_dx * qx_grid + shift_dy * qy_grid
        shift_op = torch.exp(neg_two_pi_j * ramp)
        spectrum += torch.sum(img_fft_chunk * shift_op, dim=0)
    return torch.fft.ifft2(spectrum, dim=(-2, -1)).real


def tcbf_fourier_separable(provider, qx_grid, qy_grid, cache, coeffs, device):
    """Changes 1+2: the translation ramp factorizes, so build two 1D phase factors
    (b*(Ry+Rx) transcendentals) instead of one full 2D grid (b*Ry*Rx)."""
    neg_two_pi_j = torch.tensor(-2.0j * torch.pi, dtype=torch.complex64, device=device)
    Ry_out, Rx_out = qy_grid.shape[-2], qx_grid.shape[-1]
    spectrum = torch.zeros((Ry_out, Rx_out), dtype=torch.complex64, device=device)
    for chunk in cache.chunks:
        img_fft_chunk = provider.get_chunk(chunk['start'], chunk['end'])
        shift_dx = torch.einsum('k, kb -> b', coeffs, chunk['b_dx']).view(-1, 1, 1)
        shift_dy = torch.einsum('k, kb -> b', coeffs, chunk['b_dy']).view(-1, 1, 1)
        phase_x = torch.exp(neg_two_pi_j * (shift_dx * qx_grid))  # (b, 1, Rx)
        phase_y = torch.exp(neg_two_pi_j * (shift_dy * qy_grid))  # (b, Ry, 1)
        spectrum += torch.sum(img_fft_chunk * phase_x * phase_y, dim=0)
    return torch.fft.ifft2(spectrum, dim=(-2, -1)).real


# ──────────────────────────────── harness ────────────────────────────────────

class DeviceResidentProvider:
    """Stub ImageFFT serving slices of an already-on-device FFT store.

    Mirrors imagefft_storage='device'. Keeping the store resident is deliberate: it
    removes H2D transfer from the measurement so the reconstruction math is what varies.
    """

    def __init__(self, fft_all: torch.Tensor) -> None:
        self._fft = fft_all

    def get_chunk(self, b_start: int, b_end: int) -> torch.Tensor:
        return self._fft[b_start:b_end]


@dataclass
class Row:
    raw_scan: int
    upscale: int
    ry: int
    rx: int
    nb: int
    chunk_size: int
    n_chunks: int
    variant: str
    time_min_ms: float
    time_mean_ms: float
    time_std_ms: float
    speedup_vs_baseline: float
    transient_mib: float
    transient_ratio_vs_baseline: float
    resident_mib: float
    peak_total_mib: float
    rel_err_vs_baseline: float
    grad_rel_err_vs_baseline: float
    dispatch_ms: float
    dispatch_fraction: float
    data_pass_gib: float
    effective_gib_s: float
    passes_over_data: float
    device_name: str
    torch_version: str


def build_case(nb: int, R: int, chunk_size: int, device: str, scan_step: float = 0.5):
    """Realistic-shaped inputs: all-positive vBFs with a large DC term, real shift basis."""
    torch.manual_seed(0)
    dev = torch.device(device)

    k_max = MAX_ALPHA / WAVELENGTH
    radius = torch.sqrt(torch.rand(nb, device=dev)) * k_max
    angle = torch.rand(nb, device=dev) * 2 * torch.pi
    kX_full, kY_full = radius * torch.cos(angle), radius * torch.sin(angle)

    qx_grid = torch.fft.fftfreq(R, d=scan_step, device=dev).view(1, 1, R)
    qy_grid = torch.fft.fftfreq(R, d=scan_step, device=dev).view(1, R, 1)

    vbf = 1000.0 + 50.0 * torch.rand(nb, R, R, device=dev)
    fft_all = torch.fft.fft2(vbf, dim=(-2, -1))
    del vbf

    cache = build_tcbf_cache(kX_full, kY_full, ORDER_KEYS, WAVELENGTH, chunk_size=chunk_size)
    # generate_shift_basis expands (n, m) keys into cos/sin pairs, so the coefficient
    # count comes from the built basis rather than len(ORDER_KEYS).
    num_coeffs = cache.chunks[0]['b_dx'].shape[0]
    coeffs = torch.randn(num_coeffs, device=dev)

    return DeviceResidentProvider(fft_all), qx_grid, qy_grid, coeffs, cache


def measure(fn, args, device: str, repeats: int, warmup: int):
    """Return (min_ms, mean_ms, std_ms, transient_mib, output)."""
    cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        out = fn(*args)
    if cuda:
        torch.cuda.synchronize()
        baseline_alloc = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args)
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    transient = (torch.cuda.max_memory_allocated() - baseline_alloc) / 2**20 if cuda else float("nan")
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    return min(times) * 1e3, mean * 1e3, var**0.5 * 1e3, transient, out


def measure_dispatch_fraction(fn, args, device: str, repeats: int, warmup: int):
    """Return (dispatch_ms, wall_ms): CPU-only launch cost vs. full call.

    Timing without a trailing synchronize returns once the last kernel is *queued*, so
    the ratio says whether the chunk loop is bound by GPU work or by the CPU's ability
    to feed it. Near 1.0 means the device is idling between launches and no amount of
    arithmetic removed from the loop body can help.
    """
    if torch.device(device).type != "cuda":
        return float("nan"), float("nan")
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    dispatch = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        dispatch.append(time.perf_counter() - t0)  # no sync: queue time only
    torch.cuda.synchronize()

    wall = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - t0)
    return min(dispatch) * 1e3, min(wall) * 1e3


def measure_copy_bandwidth(device: str, mib: int = 512) -> float:
    """Empirical device copy bandwidth in GiB/s, used to convert times into
    'passes over the data'. Measured rather than taken from a spec sheet."""
    if torch.device(device).type != "cuda":
        return float("nan")
    src = torch.empty(mib * 2**20 // 4, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    del src, dst
    torch.cuda.empty_cache()
    return (2 * mib / 1024) / min(times)  # read + write


def rel_err(reference: torch.Tensor, other: torch.Tensor) -> float:
    scale = reference.abs().max().clamp(min=1e-12)
    return ((reference - other).abs().max() / scale).item()


def coeff_grad(fn, provider, qx_grid, qy_grid, cache, coeffs, device):
    """d(loss)/d(coeffs) under a fixed deterministic loss.

    tcBF is an AD forward pass -- coeffs is what the optimizer moves -- so matching
    images is necessary but not sufficient. The weighting is a fixed ramp rather than
    a plain mean so the loss probes the whole image instead of only its DC component,
    which any of these variants would reproduce trivially.
    """
    c = coeffs.detach().clone().requires_grad_(True)
    out = fn(provider, qx_grid, qy_grid, cache, c, device)
    weight = torch.linspace(0.0, 1.0, out.numel(), device=out.device).view_as(out)
    (out * weight).sum().backward()
    return c.grad.detach().clone()


# ───────────────────────────────── writers ───────────────────────────────────

def write_jsonl(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[Row], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key = {}
    for r in rows:
        by_key.setdefault((r.raw_scan, r.upscale, r.chunk_size), {})[r.variant] = r

    with path.open("w") as f:
        f.write("# tcBF reconstruction variants\n\n")
        f.write(f"Device: {meta['device_name']}  \n")
        f.write(f"torch {meta['torch_version']}, {meta['platform']}  \n")
        f.write(f"Baseline ref: `{meta['baseline_ref']}`, current tree: `{meta['head_ref']}`  \n")
        f.write(f"N_BF = {meta['nb']}, measured copy bandwidth "
                f"{meta['copy_bandwidth_gib_s']:.1f} GiB/s\n\n")
        f.write("Timings use a device-resident provider, so they isolate the "
                "reconstruction math. `transient` is peak allocated during the call "
                "minus the resident FFT store.\n\n")

        f.write("## Speed by scan size and upscale\n\n")
        f.write("`R = raw_scan * upscale`. Rows reaching the same R by different routes "
                "should agree.\n\n")
        f.write("| raw scan | upscale | R | chunk | baseline ms | fourier ms | "
                "+separable ms | current ms | total speedup |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for (raw, up, cs), variants in sorted(by_key.items()):
            base = variants.get("baseline")
            if base is None:
                continue
            cells = [variants.get(v) for v in VARIANT_ORDER]
            times = " | ".join(f"{c.time_min_ms:.3f}" if c else "-" for c in cells)
            cur = variants.get("current")
            speed = f"{base.time_min_ms / cur.time_min_ms:.2f}x" if cur else "-"
            f.write(f"| {raw} | {up}x | {base.ry} | {cs} | {times} | {speed} |\n")

        f.write("\n## Transient VRAM (MiB)\n\n")
        f.write("| raw scan | upscale | R | chunk | baseline | fourier | +separable | "
                "current | reduction | resident |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for (raw, up, cs), variants in sorted(by_key.items()):
            base = variants.get("baseline")
            if base is None:
                continue
            cells = [variants.get(v) for v in VARIANT_ORDER]
            mems = " | ".join(f"{c.transient_mib:.1f}" if c else "-" for c in cells)
            cur = variants.get("current")
            red = f"{base.transient_mib / cur.transient_mib:.2f}x" if cur else "-"
            f.write(f"| {raw} | {up}x | {base.ry} | {cs} | {mems} | {red} | "
                    f"{base.resident_mib:.0f} |\n")

        f.write("\n## Memory traffic\n\n")
        f.write("Passes over the `(N_BF, R, R)` store, from measured time and "
                "device bandwidth. A bandwidth-bound kernel is pinned by this number.\n\n")
        f.write("| raw scan | upscale | R | chunk | variant | ms | GiB/s | passes |\n")
        f.write("|---:|---:|---:|---:|---|---:|---:|---:|\n")
        for (raw, up, cs), variants in sorted(by_key.items()):
            for v in VARIANT_ORDER:
                r = variants.get(v)
                if r is None:
                    continue
                f.write(f"| {raw} | {up}x | {r.ry} | {cs} | {v} | {r.time_min_ms:.3f} | "
                        f"{r.effective_gib_s:.1f} | {r.passes_over_data:.1f} |\n")

        has_dispatch = any(r.dispatch_fraction == r.dispatch_fraction for r in rows)
        if has_dispatch:
            f.write("\n## Dispatch vs GPU work\n\n")
            f.write("CPU time to queue the call (no trailing sync) against full wall "
                    "time. A fraction near 1.0 means the chunk loop is launch-bound and "
                    "the device idles between kernels, so removing arithmetic cannot "
                    "help.\n\n")
            f.write("| raw scan | upscale | R | chunk | variant | dispatch ms | "
                    "wall ms | dispatch/wall |\n")
            f.write("|---:|---:|---:|---:|---|---:|---:|---:|\n")
            for (raw, up, cs), variants in sorted(by_key.items()):
                for v in VARIANT_ORDER:
                    r = variants.get(v)
                    if r is None or r.dispatch_fraction != r.dispatch_fraction:
                        continue
                    f.write(f"| {raw} | {up}x | {r.ry} | {cs} | {v} | "
                            f"{r.dispatch_ms:.3f} | {r.time_min_ms:.3f} | "
                            f"{r.dispatch_fraction:.1%} |\n")

        f.write("\n## Correctness\n\n")
        worst = max((r.rel_err_vs_baseline for r in rows if r.variant != "baseline"),
                    default=0.0)
        f.write(f"Worst max-relative difference in the reconstructed image, against the "
                f"baseline commit across all cases: **{worst:.2e}** (float32 roundoff).\n\n")

        grads = [r.grad_rel_err_vs_baseline for r in rows
                 if r.variant != "baseline" and r.grad_rel_err_vs_baseline == r.grad_rel_err_vs_baseline]
        checked = sorted({r.ry for r in rows
                          if r.grad_rel_err_vs_baseline == r.grad_rel_err_vs_baseline})
        if grads:
            f.write(f"Worst max-relative difference in `d(loss)/d(coeffs)`: "
                    f"**{max(grads):.2e}**, over R in "
                    f"{', '.join(str(r) for r in checked)}. tcBF is an AD forward pass, "
                    f"so gradient parity is the property that matters for fitting, not "
                    f"just image parity.\n")


# ────────────────────────────────── main ─────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--nb", type=int, default=797,
                   help="Number of BF-disk detector pixels (default: 797)")
    p.add_argument("--raw-scans", default="64,128,256",
                   help="Comma-separated raw scan sizes before upscale")
    p.add_argument("--upscales", default="1,2,4", help="Comma-separated upscale factors")
    p.add_argument("--chunk-sizes", default="64",
                   help="Comma-separated chunk sizes to sweep")
    p.add_argument("--max-r", type=int, default=1024,
                   help="Skip cases whose R exceeds this (default: 1024)")
    p.add_argument("--max-grad-r", type=int, default=512,
                   help="Skip the gradient-parity check above this R. The baseline "
                        "keeps every chunk's intermediates alive for backward, so its "
                        "graph is ~n_chunks x the forward transient (default: 512)")
    p.add_argument("--probe-dispatch", action="store_true",
                   help="Also report CPU launch time vs wall time per case, which "
                        "distinguishes a dispatch-bound chunk loop from a GPU-bound one")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--label", default=time.strftime("%Y%m%d-%H%M%S"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    from fast_acbf.core.tcbf import reconstruct_tcbf as current_tcbf

    variants = {
        "baseline": load_baseline_reconstruct(args.baseline_ref),
        "fourier": tcbf_fourier,
        "fourier_separable": tcbf_fourier_separable,
        "current": current_tcbf,
    }

    raw_scans = [int(x) for x in args.raw_scans.split(",") if x]
    upscales = [int(x) for x in args.upscales.split(",") if x]
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",") if x]

    device_name = (torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor())
    bandwidth = measure_copy_bandwidth(args.device)
    head_ref = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or "unknown"

    print(f"Device: {device_name}")
    print(f"Baseline {args.baseline_ref} vs current tree ({head_ref})")
    print(f"N_BF={args.nb}, copy bandwidth {bandwidth:.1f} GiB/s\n")

    rows: list[Row] = []
    for raw in raw_scans:
        for up in upscales:
            R = raw * up
            if R > args.max_r:
                print(f"raw={raw} up={up}x -> R={R}: skipped (exceeds --max-r)")
                continue
            for cs in chunk_sizes:
                try:
                    case = build_case(args.nb, R, cs, args.device)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"raw={raw} up={up}x -> R={R} chunk={cs}: OOM building case")
                    continue

                provider, qx, qy, coeffs, cache = case
                resident_mib = provider._fft.numel() * 8 / 2**20
                data_pass_gib = args.nb * R * R * 8 / 2**30
                fn_args = (provider, qx, qy, cache, coeffs, args.device)

                check_grad = R <= args.max_grad_r
                reference = ref_grad = None
                base_time = base_mem = None
                for name in VARIANT_ORDER:
                    try:
                        tmin, tmean, tstd, mem, out = measure(
                            variants[name], fn_args, args.device, args.repeats, args.warmup)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"  {name}: OOM")
                        continue

                    if name == "baseline":
                        reference, base_time, base_mem = out.clone(), tmin, mem

                    grad_err = 0.0 if name == "baseline" else float("nan")
                    if check_grad:
                        try:
                            g = coeff_grad(variants[name], provider, qx, qy, cache,
                                           coeffs, args.device)
                            if name == "baseline":
                                ref_grad = g
                            elif ref_grad is not None:
                                grad_err = rel_err(ref_grad, g)
                            del g
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"  {name}: OOM during gradient check")
                            grad_err = float("nan")
                        torch.cuda.empty_cache()

                    dispatch_ms = dispatch_frac = float("nan")
                    if args.probe_dispatch:
                        d, w = measure_dispatch_fraction(
                            variants[name], fn_args, args.device,
                            repeats=max(5, args.repeats // 2), warmup=args.warmup)
                        dispatch_ms, dispatch_frac = d, d / w

                    eff_bw = data_pass_gib / (tmin / 1e3)
                    rows.append(Row(
                        raw_scan=raw, upscale=up, ry=R, rx=R, nb=args.nb,
                        chunk_size=cs, n_chunks=len(cache.chunks), variant=name,
                        time_min_ms=tmin, time_mean_ms=tmean, time_std_ms=tstd,
                        speedup_vs_baseline=(base_time / tmin) if base_time else float("nan"),
                        transient_mib=mem,
                        transient_ratio_vs_baseline=(base_mem / mem) if base_mem and mem else float("nan"),
                        resident_mib=resident_mib,
                        peak_total_mib=resident_mib + mem,
                        rel_err_vs_baseline=(0.0 if name == "baseline"
                                             else rel_err(reference, out)),
                        grad_rel_err_vs_baseline=grad_err,
                        dispatch_ms=dispatch_ms,
                        dispatch_fraction=dispatch_frac,
                        data_pass_gib=data_pass_gib,
                        effective_gib_s=eff_bw,
                        passes_over_data=(tmin / 1e3) * bandwidth / data_pass_gib,
                        device_name=device_name, torch_version=torch.__version__,
                    ))
                    del out

                speed = rows[-1].speedup_vs_baseline
                print(f"raw={raw:>4} up={up}x -> R={R:>5} chunk={cs:>4} | "
                      f"baseline {base_time:8.3f} ms -> current {rows[-1].time_min_ms:8.3f} ms "
                      f"({speed:.2f}x), VRAM {base_mem:7.1f} -> {rows[-1].transient_mib:7.1f} MiB")

                del provider, cache, reference, ref_grad
                torch.cuda.empty_cache()

    out_dir = args.out_dir / args.label
    meta = {
        "device_name": device_name, "torch_version": torch.__version__,
        "platform": platform.platform(), "baseline_ref": args.baseline_ref,
        "head_ref": head_ref, "nb": args.nb, "copy_bandwidth_gib_s": bandwidth,
    }
    write_jsonl(out_dir / "results.jsonl", rows)
    write_csv(out_dir / "results.csv", rows)
    write_summary(out_dir / "summary.md", rows, meta)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nWrote {len(rows)} rows to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
