#!/usr/bin/env python
"""Benchmark fast-acBF timing on CUDA with synchronized section timers.

Each public sweep case runs in a fresh child process. That keeps CUDA OOMs
contained and avoids cross-case cache allocator state from contaminating timings.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import benchmark_vram as common


DEFAULT_OUT_DIR = Path("benchmarks/timing")


def sync_device(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def timed_call(fn, device: str):
    sync_device(device)
    start = time.perf_counter()
    value = fn()
    sync_device(device)
    return time.perf_counter() - start, value


def stat_block(values: list[float], prefix: str) -> dict:
    if not values:
        return {
            f"{prefix}_mean_s": None,
            f"{prefix}_std_s": None,
            f"{prefix}_min_s": None,
            f"{prefix}_max_s": None,
        }
    return {
        f"{prefix}_mean_s": float(statistics.mean(values)),
        f"{prefix}_std_s": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        f"{prefix}_min_s": float(min(values)),
        f"{prefix}_max_s": float(max(values)),
    }


def warm_up_device(device: str) -> None:
    if device != "cuda":
        return

    import torch

    torch.cuda.empty_cache()
    x = torch.randn((512, 512), device=device)
    for _ in range(3):
        y = x @ x.T
        torch.fft.fft2(y)
    torch.cuda.synchronize()
    del x, y
    torch.cuda.empty_cache()


def make_vbf_host(dataset: np.ndarray, geom: common.DetectorGeometry) -> np.ndarray:
    ky = np.fft.fftshift(np.fft.fftfreq(geom.npix, d=(1 / geom.dk / geom.npix)))
    kx = np.fft.fftshift(np.fft.fftfreq(geom.npix, d=(1 / geom.dk / geom.npix)))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    kr_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    bf_mask = kr_grid <= (geom.max_alpha_mrad / 1e3 / geom.wavelength)
    vbf_np = dataset[:, :, bf_mask]
    return np.ascontiguousarray(np.moveaxis(vbf_np, -1, 0), dtype=np.float32)


def measure_h2d(array: np.ndarray, device: str, repeats: int) -> dict:
    import torch

    times = []
    for _ in range(repeats):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        def copy_array():
            return torch.tensor(array, dtype=torch.float32, device=device)

        elapsed, tensor = timed_call(copy_array, device)
        times.append(elapsed)
        del tensor
        if device == "cuda":
            torch.cuda.empty_cache()
    return stat_block(times, "h2d")


def build_solver(dataset: np.ndarray, geom: common.DetectorGeometry, args: argparse.Namespace, solver_cls):
    return solver_cls(
        dataset=dataset,
        max_alpha=geom.max_alpha_mrad,
        scan_step_size=args.scan_step_size,
        dk=geom.dk,
        wavelength=geom.wavelength,
        max_order=args.max_order,
        aberrations={"C10": 0.0},
        device=args.device,
        cache_mode=args.cache_mode,
    )


def run_reconstruct(solver, recon_mode: str, chunk_size: int):
    if recon_mode == "acbf":
        return solver.get_acBF(chunk_size=chunk_size)
    if recon_mode == "tcbf":
        return solver.get_tcBF(chunk_size=chunk_size)
    raise ValueError(f"Unsupported reconstruction mode: {recon_mode!r}.")


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

    recon_mode = common.normalize_recon_mode(args.recon_mode)
    geom = common.detector_geometry_for_nb(
        args.nb,
        dk=args.dk,
        wavelength=args.wavelength,
        max_npix=args.max_npix,
    )
    dataset = common.make_synthetic_dataset(args.ry, args.rx, geom)
    vbf_host = make_vbf_host(dataset, geom)

    result = {
        "recon_mode": recon_mode,
        "cache_mode": args.cache_mode,
        "requested_nb": int(args.nb),
        "actual_nb": int(vbf_host.shape[0]),
        "ry": int(args.ry),
        "rx": int(args.rx),
        "max_order": int(args.max_order),
        "chunk_size": int(args.chunk_size),
        "warmup_repeats": int(args.warmup_repeats),
        "timing_repeats": int(args.timing_repeats),
        "transfer_repeats": int(args.transfer_repeats),
        "init_repeats": int(args.init_repeats),
        "npix": int(geom.npix),
        "raw_dataset_mib": float(dataset.nbytes / 1024**2),
        "vbf_stack_mib": float(vbf_host.nbytes / 1024**2),
        "device": device,
        "torch_version": torch.__version__,
        "status": "ok",
    }

    if device == "cuda":
        result["device_name"] = torch.cuda.get_device_name()
        result["device_total_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

    try:
        warm_up_device(device)

        raw_stats = measure_h2d(dataset, device, args.transfer_repeats)
        result.update({f"raw_dataset_{key}": value for key, value in raw_stats.items()})

        vbf_stats = measure_h2d(vbf_host, device, args.transfer_repeats)
        result.update({f"vbf_stack_{key}": value for key, value in vbf_stats.items()})

        init_times = []
        for _ in range(args.init_repeats):
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elapsed, solver = timed_call(lambda: build_solver(dataset, geom, args, BFSolver), device)
            result["actual_nb"] = int(solver.vbf_images.shape[0])
            init_times.append(elapsed)
            del solver
            if device == "cuda":
                torch.cuda.empty_cache()
        result.update(stat_block(init_times, "solver_init"))

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        _, solver = timed_call(lambda: build_solver(dataset, geom, args, BFSolver), device)
        result["actual_nb"] = int(solver.vbf_images.shape[0])

        cold_elapsed, image = timed_call(
            lambda: run_reconstruct(solver, recon_mode, args.chunk_size),
            device,
        )
        result["reconstruct_cold_s"] = float(cold_elapsed)
        result["image_mean"] = float(image.detach().float().mean().cpu())

        for _ in range(args.warmup_repeats):
            _, warm_img = timed_call(
                lambda: run_reconstruct(solver, recon_mode, args.chunk_size),
                device,
            )
            del warm_img

        warm_times = []
        for _ in range(args.timing_repeats):
            elapsed, timed_img = timed_call(
                lambda: run_reconstruct(solver, recon_mode, args.chunk_size),
                device,
            )
            warm_times.append(elapsed)
            result["image_mean"] = float(timed_img.detach().float().mean().cpu())
            del timed_img
        result.update(stat_block(warm_times, "reconstruct_warm"))

        if device == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3

        del solver, image

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        msg = str(exc)
        if "out of memory" not in msg.lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        result["status"] = "oom"
        result["error"] = msg.splitlines()[0][:500]
        if device == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            torch.cuda.empty_cache()

    return result


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def load_existing(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = result_key(row)
            rows[key] = row
    return rows


def result_key(row: dict) -> tuple:
    return (
        common.normalize_recon_mode(row.get("recon_mode")),
        row.get("cache_mode"),
        int(row.get("requested_nb")),
        int(row.get("ry")),
        int(row.get("max_order")),
    )


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def run_child_case(script: Path, args: argparse.Namespace, case: dict) -> dict:
    cmd = [
        sys.executable,
        str(script),
        "--single-json",
        "--device",
        args.device,
        "--recon-mode",
        case["recon_mode"],
        "--cache-mode",
        case["cache_mode"],
        "--nb",
        str(case["nb"]),
        "--ry",
        str(case["scan"]),
        "--rx",
        str(case["scan"]),
        "--max-order",
        str(case["max_order"]),
        "--chunk-size",
        str(args.chunk_size),
        "--warmup-repeats",
        str(args.warmup_repeats),
        "--timing-repeats",
        str(args.timing_repeats),
        "--transfer-repeats",
        str(args.transfer_repeats),
        "--init-repeats",
        str(args.init_repeats),
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
            "recon_mode": case["recon_mode"],
            "cache_mode": case["cache_mode"],
            "requested_nb": case["nb"],
            "ry": case["scan"],
            "rx": case["scan"],
            "max_order": case["max_order"],
            "status": "timeout",
            "error": f"timed out after {exc.timeout} s",
        }

    stdout = completed.stdout.strip()
    if completed.returncode == 0:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)

    return {
        "recon_mode": case["recon_mode"],
        "cache_mode": case["cache_mode"],
        "requested_nb": case["nb"],
        "ry": case["scan"],
        "rx": case["scan"],
        "max_order": case["max_order"],
        "status": "error",
        "error": (completed.stderr or stdout)[-1000:],
        "returncode": completed.returncode,
    }


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
        "raw_dataset_mib",
        "vbf_stack_mib",
        "raw_dataset_h2d_mean_s",
        "raw_dataset_h2d_std_s",
        "vbf_stack_h2d_mean_s",
        "vbf_stack_h2d_std_s",
        "solver_init_mean_s",
        "solver_init_std_s",
        "reconstruct_cold_s",
        "reconstruct_warm_mean_s",
        "reconstruct_warm_std_s",
        "reconstruct_warm_min_s",
        "reconstruct_warm_max_s",
        "warmup_repeats",
        "timing_repeats",
        "transfer_repeats",
        "init_repeats",
        "chunk_size",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "device_name",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_s(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return ""


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda r: (
            str(r.get("recon_mode")),
            str(r.get("cache_mode")),
            int(r.get("max_order", 0)),
            int(r.get("ry", 0)),
            int(r.get("requested_nb", 0)),
        ),
    )
    with path.open("w", encoding="utf-8") as f:
        f.write("# fast-acBF Timing Benchmark\n\n")
        device = next((row.get("device_name") for row in rows if row.get("device_name")), None)
        if device:
            f.write(f"Device: {device}\n\n")
        f.write(
            "All GPU timings use `torch.cuda.synchronize()` before and after the "
            "measured section. Plotting is not included. `reconstruct_cold_s` is the "
            "first reconstruction after `BFSolver` construction and includes cache "
            "building. `reconstruct_warm_mean_s` is averaged after cache warmup.\n\n"
        )

        write_summary_tables(f, rows)

        f.write("## Full Results\n\n")
        f.write(
            "| recon | cache | max_order | scan | Nb req | Nb actual | status | "
            "raw H2D s | vBF H2D s | init s | cold recon s | warm recon s |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row.get('recon_mode')} | {row.get('cache_mode')} | "
                f"{row.get('max_order')} | {row.get('ry')} | "
                f"{row.get('requested_nb')} | {row.get('actual_nb', '')} | "
                f"{row.get('status')} | "
                f"{fmt_s(row.get('raw_dataset_h2d_mean_s'))} | "
                f"{fmt_s(row.get('vbf_stack_h2d_mean_s'))} | "
                f"{fmt_s(row.get('solver_init_mean_s'))} | "
                f"{fmt_s(row.get('reconstruct_cold_s'))} | "
                f"{fmt_s(row.get('reconstruct_warm_mean_s'))} |\n"
            )


def write_summary_tables(f, rows: list[dict]) -> None:
    f.write("## Summary\n\n")
    f.write("Warm cached reconstruction mean at scan `256 x 256`:\n\n")
    f.write("| recon | cache | max_order | Nb~512 | Nb=1024 | Nb~2048 | Nb=4096 |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")
    for recon_mode in ("acbf", "tcbf"):
        for cache_mode in ("on_the_fly", "host", "device"):
            for max_order in (1, 2, 3, 4):
                vals = []
                for nb in (512, 1024, 2048, 4096):
                    row = find_row(rows, recon_mode, cache_mode, max_order, 256, nb)
                    vals.append(fmt_s(row.get("reconstruct_warm_mean_s")) if row else "")
                f.write(
                    f"| {recon_mode} | {cache_mode} | {max_order} | "
                    f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |\n"
                )

    f.write("\nCold first reconstruction time at scan `256 x 256` ")
    f.write("(includes FFT/cache construction):\n\n")
    f.write("| recon | cache | max_order | Nb~512 | Nb=1024 | Nb~2048 | Nb=4096 |\n")
    f.write("|---|---|---:|---:|---:|---:|---:|\n")
    for recon_mode in ("acbf", "tcbf"):
        for cache_mode in ("on_the_fly", "host", "device"):
            for max_order in (1, 2, 3, 4):
                vals = []
                for nb in (512, 1024, 2048, 4096):
                    row = find_row(rows, recon_mode, cache_mode, max_order, 256, nb)
                    vals.append(fmt_s(row.get("reconstruct_cold_s")) if row else "")
                f.write(
                    f"| {recon_mode} | {cache_mode} | {max_order} | "
                    f"{vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} |\n"
                )

    f.write("\nRepresentative transfer and initialization timings at scan `256 x 256` ")
    f.write("from `acbf/on_the_fly/max_order=1` rows:\n\n")
    f.write("| requested Nb | actual Nb | raw dataset MiB | vBF stack MiB | raw H2D s | vBF H2D s | BFSolver init s |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
    for nb in (512, 1024, 2048, 4096):
        row = find_row(rows, "acbf", "on_the_fly", 1, 256, nb)
        if not row:
            continue
        f.write(
            f"| {nb} | {row.get('actual_nb')} | "
            f"{row.get('raw_dataset_mib', 0):.1f} | {row.get('vbf_stack_mib', 0):.1f} | "
            f"{fmt_s(row.get('raw_dataset_h2d_mean_s'))} | "
            f"{fmt_s(row.get('vbf_stack_h2d_mean_s'))} | "
            f"{fmt_s(row.get('solver_init_mean_s'))} |\n"
        )

    f.write(
        "\nThe raw H2D column copies the complete synthetic 4D dataset. The vBF H2D "
        "column copies the extracted `(Nb, Ry, Rx)` virtual-BF stack, which is closer "
        "to the dominant device transfer performed by `BFSolver.__init__`.\n\n"
    )


def find_row(rows, recon_mode, cache_mode, max_order, scan, requested_nb):
    for row in rows:
        if (
            row.get("recon_mode") == recon_mode
            and row.get("cache_mode") == cache_mode
            and int(row.get("max_order", -1)) == max_order
            and int(row.get("ry", -1)) == scan
            and int(row.get("requested_nb", -1)) == requested_nb
        ):
            return row
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-json", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--recon-mode", choices=common.DEFAULT_RECON_MODES, default="acbf")
    parser.add_argument("--cache-mode", choices=common.DEFAULT_CACHE_MODES, default="on_the_fly")
    parser.add_argument("--nb", type=int, default=512)
    parser.add_argument("--ry", type=int, default=64)
    parser.add_argument("--rx", type=int, default=64)
    parser.add_argument("--max-order", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--transfer-repeats", type=int, default=3)
    parser.add_argument("--init-repeats", type=int, default=3)
    parser.add_argument("--scan-step-size", type=float, default=0.2)
    parser.add_argument("--dk", type=float, default=common.DEFAULT_DK)
    parser.add_argument("--wavelength", type=float, default=common.DEFAULT_WAVELENGTH)
    parser.add_argument("--max-npix", type=int, default=256)
    parser.add_argument("--nbs", default=",".join(map(str, common.DEFAULT_NB)))
    parser.add_argument("--scans", default=",".join(map(str, common.DEFAULT_SCAN)))
    parser.add_argument("--max-orders", default=",".join(map(str, common.DEFAULT_MAX_ORDER)))
    parser.add_argument("--cache-modes", default=",".join(common.DEFAULT_CACHE_MODES))
    parser.add_argument("--recon-modes", default=",".join(common.DEFAULT_RECON_MODES))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default=time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--timeout-s", type=float, default=1200)
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
        common.normalize_recon_mode(part)
        for part in args.recon_modes.split(",")
        if part.strip()
    )
    cases = [
        {
            "recon_mode": recon_mode,
            "cache_mode": cache_mode,
            "nb": nb,
            "scan": scan,
            "max_order": max_order,
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
            case["cache_mode"],
            case["nb"],
            case["scan"],
            case["max_order"],
        )
        if key in existing:
            print(f"[{index}/{len(cases)}] skip existing {case}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] run {case}", flush=True)
        row = run_child_case(script, args, case)
        rows.append(row)
        append_jsonl(jsonl_path, row)
        write_csv(csv_path, rows)
        write_markdown(md_path, rows)
        print(
            f"    -> {row.get('status')} init={row.get('solver_init_mean_s')} "
            f"cold={row.get('reconstruct_cold_s')} warm={row.get('reconstruct_warm_mean_s')}",
            flush=True,
        )

    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
