#!/usr/bin/env python
"""Benchmark fast-acBF CUDA memory across synthetic dataset sizes.

The public sweep runs each case in a fresh child process. That keeps CUDA OOMs
contained and lets long sweeps resume from the JSONL results file.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_NB = (512, 1024, 2048, 4096)
DEFAULT_SCAN = (64, 128, 256)
DEFAULT_MAX_ORDER = (1, 2, 3, 4)
DEFAULT_CACHE_MODES = ("on_the_fly", "host", "device")
DEFAULT_RECON_MODES = ("acbf", "tcbf")
DEFAULT_WAVELENGTH = 0.04176
DEFAULT_DK = 0.01


@dataclass(frozen=True)
class DetectorGeometry:
    requested_nb: int
    actual_nb: int
    npix: int
    radius_px: float
    max_alpha_mrad: float
    dk: float = DEFAULT_DK
    wavelength: float = DEFAULT_WAVELENGTH


def detector_geometry_for_nb(
    requested_nb: int,
    *,
    dk: float = DEFAULT_DK,
    wavelength: float = DEFAULT_WAVELENGTH,
    max_npix: int = 256,
) -> DetectorGeometry:
    """Find a square detector and circular mask closest to requested_nb pixels."""
    requested_nb = int(requested_nb)
    if requested_nb <= 0:
        raise ValueError(f"requested_nb must be positive, got {requested_nb}.")

    best: tuple[int, int, int, float] | None = None
    for npix in range(8, max_npix + 1):
        coords = np.arange(npix, dtype=np.float32) - npix // 2
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        radii = np.sort(np.sqrt(xx * xx + yy * yy).ravel())
        unique_radii = np.unique(radii)
        counts = np.searchsorted(radii, unique_radii, side="right")
        for actual_nb, radius_px in zip(counts, unique_radii):
            delta = abs(int(actual_nb) - requested_nb)
            candidate = (delta, int(actual_nb), npix, float(radius_px))
            if best is None or candidate < best:
                best = candidate
                if delta == 0:
                    break
        if best is not None and best[0] == 0:
            break

    if best is None:
        raise RuntimeError("failed to build detector geometry")

    _, actual_nb, npix, radius_px = best
    # Nudge the radius upward so pixels exactly on the selected circular shell
    # survive float32/float64 roundoff inside pipeline.init_vbf.
    max_alpha_mrad = radius_px * (1.0 + 1e-6) * dk * wavelength * 1e3
    return DetectorGeometry(
        requested_nb=requested_nb,
        actual_nb=actual_nb,
        npix=npix,
        radius_px=radius_px,
        max_alpha_mrad=max_alpha_mrad,
        dk=dk,
        wavelength=wavelength,
    )


def make_synthetic_dataset(ry: int, rx: int, geom: DetectorGeometry) -> np.ndarray:
    """Create a deterministic 4D-STEM-like dataset for memory benchmarking."""
    coords = np.arange(geom.npix, dtype=np.float32) - geom.npix // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    rr = np.sqrt(xx * xx + yy * yy)
    disk = rr <= (geom.radius_px + 1e-6)
    detector = np.zeros((geom.npix, geom.npix), dtype=np.float32)
    if disk.any():
        taper = np.cos(0.5 * math.pi * rr[disk] / max(geom.radius_px, 1e-6)) ** 2
        detector[disk] = taper.astype(np.float32)

    y = np.linspace(0, 2 * math.pi, int(ry), endpoint=False, dtype=np.float32)
    x = np.linspace(0, 2 * math.pi, int(rx), endpoint=False, dtype=np.float32)
    scan = 1.0 + 0.05 * np.sin(y)[:, None] + 0.03 * np.cos(x)[None, :]
    dataset = scan[:, :, None, None].astype(np.float32) * detector[None, None, :, :]
    return np.ascontiguousarray(dataset, dtype=np.float32)


def coefficient_count(max_order: int) -> int:
    """Return the flattened aberration coefficient count used by AberrationState."""
    count = 0
    for n in range(1, int(max_order) + 1):
        start_m = (n + 1) % 2
        for m in range(start_m, n + 2, 2):
            count += 1 if m == 0 else 2
    return count


def normalize_recon_mode(value: str | None) -> str:
    """Normalize reconstruction mode names for storage and grouping."""
    if value is None:
        return "acbf"
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {"acbf": "acbf", "tcbf": "tcbf", "tc_bf": "tcbf", "ac_bf": "acbf"}
    if mode not in aliases:
        raise ValueError(f"Unsupported reconstruction mode: {value!r}.")
    return aliases[mode]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def benchmark_case(args: argparse.Namespace) -> dict:
    import torch
    from fast_acbf import BFSolver

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

    geom = detector_geometry_for_nb(
        args.nb,
        dk=args.dk,
        wavelength=args.wavelength,
        max_npix=args.max_npix,
    )
    result = {
        "requested_nb": int(args.nb),
        "actual_nb": int(geom.actual_nb),
        "ry": int(args.ry),
        "rx": int(args.rx),
        "max_order": int(args.max_order),
        "cache_mode": args.cache_mode,
        "recon_mode": normalize_recon_mode(args.recon_mode),
        "chunk_size": int(args.chunk_size),
        "npix": int(geom.npix),
        "radius_px": float(geom.radius_px),
        "max_alpha_mrad": float(geom.max_alpha_mrad),
        "dk": float(geom.dk),
        "wavelength": float(geom.wavelength),
        "device": device,
        "torch_version": torch.__version__,
        "status": "ok",
        "estimated": False,
    }

    if device == "cuda":
        result["device_name"] = torch.cuda.get_device_name()
        result["device_total_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

    dataset = make_synthetic_dataset(args.ry, args.rx, geom)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    try:
        solver = BFSolver(
            dataset=dataset,
            max_alpha=geom.max_alpha_mrad,
            scan_step_size=args.scan_step_size,
            dk=geom.dk,
            wavelength=geom.wavelength,
            max_order=args.max_order,
            aberrations={"C10": 0.0},
            device=device,
            cache_mode=args.cache_mode,
        )
        result["actual_nb"] = int(solver.vbf_images.shape[0])
        if result["recon_mode"] == "acbf":
            image = solver.get_acBF(chunk_size=args.chunk_size)
        elif result["recon_mode"] == "tcbf":
            image = solver.get_tcBF(chunk_size=args.chunk_size)
        else:
            raise ValueError(f"Unsupported reconstruction mode: {result['recon_mode']!r}.")
        if device == "cuda":
            torch.cuda.synchronize()
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            result["final_allocated_gib"] = torch.cuda.memory_allocated() / 1024**3
            result["final_reserved_gib"] = torch.cuda.memory_reserved() / 1024**3
        else:
            result["peak_allocated_gib"] = None
            result["peak_reserved_gib"] = None
            result["final_allocated_gib"] = None
            result["final_reserved_gib"] = None
        result["elapsed_s"] = time.perf_counter() - start
        result["image_mean"] = float(image.detach().float().mean().cpu())
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        msg = str(exc)
        if "out of memory" not in msg.lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        result["status"] = "oom"
        result["error"] = msg.splitlines()[0][:500]
        result["elapsed_s"] = time.perf_counter() - start
        if device == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            torch.cuda.empty_cache()

    return result


def run_child_case(script: Path, args: argparse.Namespace, case: dict) -> dict:
    cmd = [
        sys.executable,
        str(script),
        "--single-json",
        "--device",
        args.device,
        "--nb",
        str(case["nb"]),
        "--ry",
        str(case["scan"]),
        "--rx",
        str(case["scan"]),
        "--max-order",
        str(case["max_order"]),
        "--cache-mode",
        case["cache_mode"],
        "--recon-mode",
        case["recon_mode"],
        "--chunk-size",
        str(args.chunk_size),
        "--scan-step-size",
        str(args.scan_step_size),
        "--dk",
        str(args.dk),
        "--wavelength",
        str(args.wavelength),
        "--max-npix",
        str(args.max_npix),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "requested_nb": case["nb"],
            "actual_nb": None,
            "ry": case["scan"],
            "rx": case["scan"],
            "max_order": case["max_order"],
            "cache_mode": case["cache_mode"],
            "recon_mode": case["recon_mode"],
            "status": "timeout",
            "error": f"timed out after {exc.timeout} s",
            "estimated": False,
        }

    stdout = completed.stdout.strip()
    if completed.returncode == 0:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)

    return {
        "requested_nb": case["nb"],
        "actual_nb": None,
        "ry": case["scan"],
        "rx": case["scan"],
        "max_order": case["max_order"],
        "cache_mode": case["cache_mode"],
        "recon_mode": case["recon_mode"],
        "status": "error",
        "error": (completed.stderr or stdout)[-1000:],
        "returncode": completed.returncode,
        "estimated": False,
    }


def load_existing(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                normalize_recon_mode(row.get("recon_mode")),
                row.get("requested_nb"),
                row.get("ry"),
                row.get("max_order"),
                row.get("cache_mode"),
            )
            out[key] = row
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "recon_mode",
        "cache_mode",
        "requested_nb",
        "actual_nb",
        "ry",
        "rx",
        "max_order",
        "status",
        "estimated",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "measured_status",
        "elapsed_s",
        "chunk_size",
        "npix",
        "device_name",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["recon_mode"] = normalize_recon_mode(out.get("recon_mode"))
            writer.writerow(out)


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.copy() for row in rows]
    for row in rows:
        row["recon_mode"] = normalize_recon_mode(row.get("recon_mode"))

    ordered = sorted(rows, key=lambda r: (
        str(r.get("recon_mode")),
        str(r.get("cache_mode")),
        int(r.get("max_order", 0)),
        int(r.get("ry", 0)),
        int(r.get("requested_nb", 0)),
    ))
    with path.open("w", encoding="utf-8") as f:
        f.write("# fast-acBF VRAM Benchmark\n\n")
        if rows:
            device = next((r.get("device_name") for r in rows if r.get("device_name")), None)
            if device:
                f.write(f"Device: {device}\n\n")
        f.write(
            "Rows marked `est` failed during measurement; their peak allocation is "
            "interpolated from successful cases in the same cache mode, with an "
            "analytic tensor-size fallback. `actual Nb` is the circular BF-mask "
            "pixel count used by the solver.\n\n"
        )
        write_report_summary(f, rows)
        f.write("## Full Results\n\n")
        f.write("| recon | cache | max_order | scan | requested Nb | actual Nb | status | peak alloc GiB | peak reserved GiB | time s |\n")
        f.write("|---|---|---:|---:|---:|---:|---|---:|---:|---:|\n")
        for row in ordered:
            alloc = row.get("peak_allocated_gib")
            reserved = row.get("peak_reserved_gib")
            elapsed = row.get("elapsed_s")
            suffix = " est" if row.get("estimated") else ""
            alloc_text = f"{alloc:.2f}" if isinstance(alloc, (int, float)) else ""
            reserved_text = f"{reserved:.2f}" if isinstance(reserved, (int, float)) else ""
            elapsed_text = f"{elapsed:.2f}" if isinstance(elapsed, (int, float)) else ""
            f.write(
                f"| {row.get('recon_mode')} | {row.get('cache_mode')} | {row.get('max_order')} | "
                f"{row.get('ry')} | {row.get('requested_nb')} | {row.get('actual_nb')} | "
                f"{row.get('status')}{suffix} | "
                f"{alloc_text} | {reserved_text} | {elapsed_text} |\n"
            )


def write_report_summary(f, rows: list[dict]) -> None:
    """Write compact conclusions and sizing equations to the markdown report."""
    f.write("## Summary Report\n\n")
    f.write(
        "For this implementation, VRAM is driven by `cache_mode` (controls ImageFFT storage) "
        "and `basis_mode` (controls aberration-basis precomputation). `cache_mode='device'` "
        "stores the full `(Nb, Ry, Rx)` complex64 FFT cache in VRAM. `cache_mode='host'` "
        "fills a RAM numpy cache lazily per chunk, copying only the active chunk to GPU. "
        "`cache_mode='on_the_fly'` recomputes FFTs every pass with no persistent cache. "
        "acBF with `basis_mode='precompute'` additionally stores aperture and basis tensors "
        "for all Nb pixels; `basis_mode='on_the_fly'` (default) regenerates them per chunk. "
        "tcBF only needs small shift-basis vectors, so its peak is nearly independent of "
        "cache settings.\n\n"
    )

    f.write("Peak allocated VRAM at scan `256 x 256`:\n\n")
    f.write("| recon | cache | max_order | Nb~512 | Nb=1024 | Nb~2048 | Nb=4096 |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")
    for recon_mode in ("acbf", "tcbf"):
        for cache_mode in ("on_the_fly", "host", "device"):
            for max_order in (1, 2, 3, 4):
                vals = []
                for nb in (512, 1024, 2048, 4096):
                    match = next(
                        (
                            row for row in rows
                            if row.get("recon_mode") == recon_mode
                            and row.get("cache_mode") == cache_mode
                            and int(row.get("max_order")) == max_order
                            and int(row.get("ry")) == 256
                            and int(row.get("requested_nb")) == nb
                        ),
                        None,
                    )
                    if match is None:
                        vals.append("")
                        continue
                    peak = match.get("peak_allocated_gib")
                    text = f"{peak:.2f}" if isinstance(peak, (int, float)) else ""
                    if match.get("estimated"):
                        text += " est"
                    vals.append(text)
                f.write(
                    f"| {recon_mode} | {cache_mode} | {max_order} | "
                    f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |\n"
                )

    f.write("\n## Estimation Equations\n\n")
    f.write(
        "Let `B = actual Nb`, `S = Ry * Rx`, `M = max_order`, "
        "`K = M * (M + 5) / 2` flattened aberration coefficients, and "
        "`C = min(chunk_size, B)`. The benchmark used `chunk_size = 64`. "
        "The relevant dtypes are `float32 = 4 bytes` and `complex64 = 8 bytes`.\n\n"
    )
    f.write("Persistent FFT cache footprint by cache_mode:\n\n")
    f.write("```text\n")
    f.write("device   : fft_bytes = 8*B*S       # full (Nb, Ry, Rx) complex64 in VRAM\n")
    f.write("host     : fft_bytes = 8*C*S       # only active chunk in VRAM; rest in RAM\n")
    f.write("on_the_fly: fft_bytes = 8*C*S      # recomputed per chunk; no persistent VRAM\n")
    f.write("```\n\n")
    f.write("tcBF peak estimate:\n\n")
    f.write("```text\n")
    f.write("tcBF_bytes ~= fft_bytes             # FFT cache (mode-dependent above)\n")
    f.write("             + 8*K*B                # b_dx and b_dy shift basis, float32\n")
    f.write("             + 28*C*S               # per-chunk ramp, phasor, multiply, ifft workspaces\n")
    f.write("```\n\n")
    f.write("acBF on_the_fly basis peak estimate (default basis_mode):\n\n")
    f.write("```text\n")
    f.write("acBF_otf_bytes ~= fft_bytes\n")
    f.write("                 + (64 + 12*K)*C*S  # regenerated aperture, basis, chi, transfer, FFT workspaces\n")
    f.write("```\n\n")
    f.write("acBF precompute basis peak estimate (basis_mode='precompute'):\n\n")
    f.write("```text\n")
    f.write("acBF_pre_bytes ~= fft_bytes\n")
    f.write("                 + (16 + 8*K)*B*S   # cached ap_t/ap_mt + b_tr/b_t/b_mt for all B\n")
    f.write("                 + (32 + 4*K)*C*S   # reconstruction-time chunk workspaces\n")
    f.write("```\n\n")
    f.write(
        "Convert bytes to GiB by dividing by `1024**3`. These formulas track the "
        "main tensors in the code path; PyTorch allocator behavior, FFT work buffers, "
        "and temporary expression lifetimes add some overhead, so treat them as "
        "planning estimates rather than exact allocator readouts.\n\n"
    )


def _estimate_features(row: dict) -> tuple[float, float, float]:
    recon_mode = normalize_recon_mode(row.get("recon_mode"))
    actual_nb = row.get("actual_nb")
    if actual_nb is None:
        actual_nb = detector_geometry_for_nb(int(row["requested_nb"])).actual_nb
    ry = int(row["ry"])
    rx = int(row.get("rx", ry))
    coeffs = coefficient_count(int(row["max_order"]))
    cells = float(actual_nb) * ry * rx
    chunk = float(min(int(row.get("chunk_size") or 64), int(actual_nb)))
    chunk_cells = chunk * ry * rx
    if recon_mode == "tcbf":
        return (1.0, cells / 1e8, chunk_cells / 1e8)
    if row.get("cache_mode") == "device":
        return (1.0, cells / 1e8, chunk_cells * coeffs / 1e8)
    return (1.0, chunk_cells / 1e8, chunk_cells * coeffs / 1e8)


def _analytic_peak_estimate_gib(row: dict) -> float:
    recon_mode = normalize_recon_mode(row.get("recon_mode"))
    actual_nb = row.get("actual_nb")
    if actual_nb is None:
        actual_nb = detector_geometry_for_nb(int(row["requested_nb"])).actual_nb
    ry = int(row["ry"])
    rx = int(row.get("rx", ry))
    coeffs = coefficient_count(int(row["max_order"]))
    cells = float(actual_nb) * ry * rx
    chunk = float(min(int(row.get("chunk_size") or 64), int(actual_nb)))
    chunk_cells = chunk * ry * rx
    # Conservative tensor accounting for vBF, FFT cache, algorithm-specific caches,
    # and per-chunk temporary workspaces. This is a fallback when too few empirical
    # points exist for least-squares interpolation.
    if recon_mode == "tcbf":
        bytes_est = 8.0 * cells + 8.0 * coeffs * actual_nb + 28.0 * chunk_cells
    elif row.get("cache_mode") == "device":
        bytes_est = 8.0 * cells + (64.0 + 12.0 * coeffs) * chunk_cells
    else:
        bytes_est = 8.0 * chunk_cells + (64.0 + 12.0 * coeffs) * chunk_cells
    return bytes_est / 1024**3


def with_estimates(rows: list[dict]) -> list[dict]:
    """Return display rows with OOM/error/timeout peaks filled by interpolation."""
    output = [row.copy() for row in rows]
    for row in output:
        row["recon_mode"] = normalize_recon_mode(row.get("recon_mode"))
    groups = sorted({(row.get("recon_mode"), row.get("cache_mode")) for row in output})
    for recon_mode, cache_mode in groups:
        ok_rows = [
            row for row in output
            if row.get("recon_mode") == recon_mode
            and row.get("cache_mode") == cache_mode
            and row.get("status") == "ok"
            and isinstance(row.get("peak_allocated_gib"), (int, float))
        ]
        coef = None
        if len(ok_rows) >= 3:
            x = np.asarray([_estimate_features(row) for row in ok_rows], dtype=np.float64)
            y = np.asarray([row["peak_allocated_gib"] for row in ok_rows], dtype=np.float64)
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)

        for row in output:
            if (
                row.get("recon_mode") != recon_mode
                or row.get("cache_mode") != cache_mode
                or row.get("status") == "ok"
            ):
                continue
            row["measured_status"] = row.get("status")
            row["estimated"] = True
            row["actual_nb"] = row.get("actual_nb") or detector_geometry_for_nb(
                int(row["requested_nb"])
            ).actual_nb
            if coef is not None:
                estimate = float(np.dot(np.asarray(_estimate_features(row)), coef))
                estimate = max(estimate, _analytic_peak_estimate_gib(row) * 0.75)
            else:
                estimate = _analytic_peak_estimate_gib(row)
            row["peak_allocated_gib"] = estimate
            row["peak_reserved_gib"] = None
            row["status"] = f"{row.get('measured_status')}_estimated"
    return output


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-json", action="store_true", help="Run one case and print JSON.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--nb", type=int, default=512)
    parser.add_argument("--ry", type=int, default=64)
    parser.add_argument("--rx", type=int, default=64)
    parser.add_argument("--max-order", type=int, default=1)
    parser.add_argument("--cache-mode", choices=DEFAULT_CACHE_MODES, default="on_the_fly")
    parser.add_argument("--recon-mode", choices=DEFAULT_RECON_MODES, default="acbf")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--scan-step-size", type=float, default=0.2)
    parser.add_argument("--dk", type=float, default=DEFAULT_DK)
    parser.add_argument("--wavelength", type=float, default=DEFAULT_WAVELENGTH)
    parser.add_argument("--max-npix", type=int, default=256)
    parser.add_argument("--nbs", default=",".join(map(str, DEFAULT_NB)))
    parser.add_argument("--scans", default=",".join(map(str, DEFAULT_SCAN)))
    parser.add_argument("--max-orders", default=",".join(map(str, DEFAULT_MAX_ORDER)))
    parser.add_argument("--cache-modes", default=",".join(DEFAULT_CACHE_MODES))
    parser.add_argument("--recon-modes", default="acbf")
    parser.add_argument("--out-dir", type=Path, default=Path("benchmarks/vram"))
    parser.add_argument("--label", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.single_json:
        row = benchmark_case(args)
        print(json.dumps(row, sort_keys=True, default=_json_default))
        return 0

    out_dir = args.out_dir / args.label
    jsonl_path = out_dir / "results.jsonl"
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "summary.md"

    nbs = parse_int_list(args.nbs)
    scans = parse_int_list(args.scans)
    max_orders = parse_int_list(args.max_orders)
    cache_modes = tuple(part.strip() for part in args.cache_modes.split(",") if part.strip())
    recon_modes = tuple(
        normalize_recon_mode(part)
        for part in args.recon_modes.split(",")
        if part.strip()
    )
    cases = [
        {
            "nb": nb,
            "scan": scan,
            "max_order": max_order,
            "cache_mode": cache_mode,
            "recon_mode": recon_mode,
        }
        for recon_mode, cache_mode, max_order, scan, nb in itertools.product(
            recon_modes, cache_modes, max_orders, scans, nbs
        )
    ]

    existing = load_existing(jsonl_path) if args.resume else {}
    rows = list(existing.values())
    script = Path(__file__).resolve()
    for index, case in enumerate(cases, start=1):
        key = (
            case["recon_mode"],
            case["nb"],
            case["scan"],
            case["max_order"],
            case["cache_mode"],
        )
        if key in existing:
            print(f"[{index}/{len(cases)}] skip existing {case}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] run {case}", flush=True)
        row = run_child_case(script, args, case)
        rows.append(row)
        append_jsonl(jsonl_path, row)
        display_rows = with_estimates(rows)
        write_csv(csv_path, display_rows)
        write_markdown(md_path, display_rows)
        print(
            f"    -> {row.get('status')} peak={row.get('peak_allocated_gib')} GiB "
            f"time={row.get('elapsed_s')} s",
            flush=True,
        )

    display_rows = with_estimates(rows)
    write_csv(csv_path, display_rows)
    write_markdown(md_path, display_rows)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
