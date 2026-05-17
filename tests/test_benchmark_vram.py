"""Tests for the standalone benchmark helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_vram.py"
    spec = importlib.util.spec_from_file_location("benchmark_vram", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_timing_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    _load_benchmark_module()
    path = scripts_dir / "benchmark_timing.py"
    spec = importlib.util.spec_from_file_location("benchmark_timing", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_detector_geometry_matches_pipeline_mask_count():
    bench = _load_benchmark_module()
    geom = bench.detector_geometry_for_nb(1024)

    ky = np.fft.fftshift(np.fft.fftfreq(geom.npix, d=(1 / geom.dk / geom.npix)))
    kx = np.fft.fftshift(np.fft.fftfreq(geom.npix, d=(1 / geom.dk / geom.npix)))
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    kr_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    mask = kr_grid <= (geom.max_alpha_mrad / 1e3 / geom.wavelength)

    assert int(mask.sum()) == geom.actual_nb == 1024


def test_with_estimates_fills_failed_rows():
    bench = _load_benchmark_module()
    rows = [
        {
            "requested_nb": 512,
            "actual_nb": 511,
            "ry": 64,
            "rx": 64,
            "max_order": 1,
            "cache_mode": "device",
            "chunk_size": 64,
            "status": "ok",
            "peak_allocated_gib": 0.1,
        },
        {
            "requested_nb": 1024,
            "actual_nb": 1024,
            "ry": 64,
            "rx": 64,
            "max_order": 2,
            "cache_mode": "device",
            "chunk_size": 64,
            "status": "ok",
            "peak_allocated_gib": 0.3,
        },
        {
            "requested_nb": 2048,
            "actual_nb": 2047,
            "ry": 64,
            "rx": 64,
            "max_order": 3,
            "cache_mode": "device",
            "chunk_size": 64,
            "status": "ok",
            "peak_allocated_gib": 0.9,
        },
        {
            "requested_nb": 4096,
            "actual_nb": 4096,
            "ry": 256,
            "rx": 256,
            "max_order": 4,
            "cache_mode": "device",
            "chunk_size": 64,
            "status": "oom",
        },
    ]

    estimated = bench.with_estimates(rows)
    failed = estimated[-1]
    assert failed["estimated"] is True
    assert failed["measured_status"] == "oom"
    assert failed["status"] == "oom_estimated"
    assert failed["peak_allocated_gib"] > 0


def test_timing_stat_block_reports_basic_stats():
    timing = _load_timing_module()

    stats = timing.stat_block([1.0, 2.0, 3.0], "example")

    assert stats["example_mean_s"] == 2.0
    assert stats["example_min_s"] == 1.0
    assert stats["example_max_s"] == 3.0
    assert stats["example_std_s"] > 0


def test_timing_vbf_host_uses_requested_geometry():
    bench = _load_benchmark_module()
    timing = _load_timing_module()
    geom = bench.detector_geometry_for_nb(512)
    dataset = bench.make_synthetic_dataset(8, 8, geom)

    vbf = timing.make_vbf_host(dataset, geom)

    assert vbf.shape == (geom.actual_nb, 8, 8)
    assert vbf.dtype == np.float32
