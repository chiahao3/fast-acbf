"""Tests for BFExtractor strategy behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fast_acbf.data.bf_extractor import BFExtractor
from fast_acbf.data.dataset4d import Dataset4D
from fast_acbf.data.geometry import DetectorGeometry


def _make_data(shape=(4, 4, 16, 16), seed=11):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 1.0, shape).astype(np.float32)


def _make_geom(detector_shape=(16, 16), device='cpu'):
    return DetectorGeometry.from_params(
        detector_shape=detector_shape,
        max_alpha=25.0,
        dk=(25.0 / 1000.0) / (detector_shape[0] / 2 * 0.04176),
        wavelength=0.04176,
        device=device,
    )


def _as_numpy(vbf):
    if isinstance(vbf, torch.Tensor):
        return vbf.detach().cpu().numpy()
    return vbf


@pytest.mark.parametrize("strategy", ["device_mask", "host_mask"])
def test_materialized_strategies_match_host_mask(strategy):
    arr = _make_data()
    geom = _make_geom()
    ref = BFExtractor(Dataset4D(arr), geom, strategy='host_mask').extract_all()
    got = BFExtractor(Dataset4D(arr), geom, device='cpu', strategy=strategy).extract_all()
    np.testing.assert_allclose(_as_numpy(got), ref, atol=1e-6)


@pytest.mark.parametrize("strategy", ["disk_per_pixel", "disk_slab", "disk_scan_row"])
def test_hdf5_disk_strategies_match_host_mask(tmp_path, strategy):
    h5py = pytest.importorskip("h5py")
    arr = _make_data()
    path = tmp_path / "data.h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=arr)

    geom = _make_geom()
    ref = BFExtractor(Dataset4D(arr), geom, strategy='host_mask').extract_all()
    ds = Dataset4D.from_hdf5(path)
    got = BFExtractor(ds, geom, strategy=strategy).extract_all()
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_hdf5_disk_strategy_respects_normalization(tmp_path):
    h5py = pytest.importorskip("h5py")
    arr = _make_data()
    path = tmp_path / "data.h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=arr)

    geom = _make_geom()
    ref = BFExtractor(Dataset4D(arr, normalize=True), geom, strategy='host_mask').extract_all()
    got = BFExtractor(Dataset4D.from_hdf5(path, normalize=True), geom, strategy='disk_slab').extract_all()
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_auto_strategy_detects_contiguous_as_slab(tmp_path):
    h5py = pytest.importorskip("h5py")
    arr = _make_data()
    path = tmp_path / "contiguous.h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=arr)

    extractor = BFExtractor(Dataset4D.from_hdf5(path), _make_geom(), strategy='auto')
    assert extractor.strategy == 'disk_slab'


def test_auto_strategy_detects_detector_major_chunks(tmp_path):
    h5py = pytest.importorskip("h5py")
    arr = _make_data()
    path = tmp_path / "detector_chunks.h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=arr, chunks=(4, 4, 1, 1))

    extractor = BFExtractor(Dataset4D.from_hdf5(path), _make_geom(), strategy='auto')
    assert extractor.strategy == 'disk_per_pixel'


def test_auto_strategy_detects_scan_major_chunks(tmp_path):
    h5py = pytest.importorskip("h5py")
    arr = _make_data()
    path = tmp_path / "scan_chunks.h5"
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=arr, chunks=(1, 1, 16, 16))

    extractor = BFExtractor(Dataset4D.from_hdf5(path), _make_geom(), strategy='auto')
    assert extractor.strategy == 'disk_scan_row'
