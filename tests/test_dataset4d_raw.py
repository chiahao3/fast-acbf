"""Unit tests for Dataset4D.from_raw (headerless binary frames, e.g. EMPAD .raw)."""
from __future__ import annotations

import numpy as np
import pytest

from fast_acbf import BFSolver
from fast_acbf.data.dataset4d import Dataset4D


def _write_raw(path, arr: np.ndarray, gap: int, offset: int = 0) -> None:
    """Write (Ry, Rx, Ky, Kx) frames with ``gap`` filler bytes after each frame."""
    with open(path, 'wb') as f:
        f.write(b'\xab' * offset)
        for frame in arr.reshape(-1, *arr.shape[2:]):
            f.write(frame.tobytes())
            f.write(b'\xcd' * gap)


@pytest.fixture
def arr() -> np.ndarray:
    return np.random.default_rng(0).uniform(0.1, 1.0, (3, 4, 6, 5)).astype(np.float32)


@pytest.mark.parametrize("gap, offset", [(1024, 0), (0, 0), (16, 8)])
def test_from_raw_reads_frames(tmp_path, arr, gap, offset):
    path = tmp_path / "scan.raw"
    _write_raw(path, arr, gap, offset)
    ds = Dataset4D.from_raw(path, arr.shape[:2], arr.shape[2:], gap=gap, offset=offset)

    assert ds.is_lazy and ds.shape == arr.shape and ds.backend_chunks is None
    np.testing.assert_array_equal(ds.read_raw((1, slice(None), 2, 3)), arr[1, :, 2, 3])
    np.testing.assert_array_equal(ds.materialize(), arr)
    assert not ds.is_lazy


def test_from_raw_normalize_matches_in_memory(tmp_path, arr):
    path = tmp_path / "scan.raw"
    _write_raw(path, arr, 1024)
    lazy = Dataset4D.from_raw(path, (3, 4), (6, 5), normalize=True)
    eager = Dataset4D.from_raw(path, (3, 4), (6, 5), materialize=True, normalize=True)
    ref = Dataset4D(arr, normalize=True)

    assert lazy.norm_factor == pytest.approx(ref.norm_factor, rel=1e-6)
    np.testing.assert_allclose(lazy.materialize(), ref.raw_array(), rtol=1e-6)
    np.testing.assert_allclose(eager.raw_array(), ref.raw_array(), rtol=1e-6)


def test_from_raw_size_mismatch_raises(tmp_path, arr):
    path = tmp_path / "scan.raw"
    _write_raw(path, arr, 0)
    with pytest.raises(ValueError, match="expected"):
        Dataset4D.from_raw(path, (3, 4), (6, 5))  # default gap=1024 does not fit


def test_solver_rejects_raw_path(tmp_path, arr):
    path = tmp_path / "scan.raw"
    _write_raw(path, arr, 1024)
    with pytest.raises(ValueError, match="from_raw"):
        BFSolver(path, max_alpha=25, scan_step_size=0.5, dk=0.05, wavelength=0.04, device='cpu')
