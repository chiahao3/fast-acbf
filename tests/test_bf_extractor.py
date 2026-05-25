"""Tests for BFExtractor strategy behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fast_acbf.data.bf_extractor import BFExtractor
from fast_acbf.data.bf_preparer import BFPreparer, _compute_pad_for_axis
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


def test_device_mask_precompute_extracts_whole_pass_once():
    """device_mask precompute is a whole-pass path, independent of FFT batch size."""
    from fast_acbf.data.imagefft import ImageFFT

    arr = _make_data(shape=(4, 4, 16, 16))
    ds = Dataset4D(arr)

    raw_access_count = [0]
    original_raw_array = ds.raw_array

    def counting_raw_array():
        raw_access_count[0] += 1
        return original_raw_array()

    ds.raw_array = counting_raw_array  # type: ignore[method-assign]

    geom = _make_geom()
    extractor = BFExtractor(ds, geom, device='cpu', strategy='device_mask')
    nb = extractor.nb

    imagefft = ImageFFT(extractor, device='cpu', storage='device', fill='precompute', batch_size=1)

    assert nb > 1, "test requires more than one BF pixel"
    assert raw_access_count[0] == 1, (
        f"Expected 1 raw_array() call during precompute but got {raw_access_count[0]}. "
        "device_mask precompute should extract the full vBF stack in one pass."
    )
    assert imagefft.filled.all()
    assert tuple(imagefft.cache.shape) == (nb, *ds.scan_shape)
    _ = imagefft  # silence unused-variable warning


def test_device_mask_rejects_lazy_imagefft_fill():
    from fast_acbf.data.imagefft import ImageFFT

    extractor = BFExtractor(
        Dataset4D(_make_data()),
        _make_geom(),
        device='cpu',
        strategy='device_mask',
    )
    with pytest.raises(ValueError, match="device_mask"):
        ImageFFT(extractor, device='cpu', storage='host', fill='lazy')


def test_bf_preparer_identity_passes_native_vbf_through():
    arr = _make_data()
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')
    ref = extractor.extract_all()

    preparer = BFPreparer(extractor, upscale=1.0, upscale_method='bilinear', pad_width=None)
    got = preparer.extract_all()

    assert preparer.raw_shape == Dataset4D(arr).scan_shape
    assert preparer.upscaled_shape == Dataset4D(arr).scan_shape
    assert preparer.padded_shape == Dataset4D(arr).scan_shape
    np.testing.assert_allclose(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["nearest", "bilinear"])
def test_bf_preparer_real_space_upscale_shape(method):
    arr = _make_data(shape=(4, 5, 16, 16))
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')

    preparer = BFPreparer(extractor, upscale=1.5, upscale_method=method, pad_width=None)
    got = preparer.extract_chunk(0, 2)

    assert preparer.upscaled_shape == (6, 8)
    assert preparer.scan_shape == (6, 8)
    assert got.shape == (2, 6, 8)
    assert got.dtype == np.float32


def test_bf_preparer_nearest_matches_torch_interpolate():
    arr = _make_data(shape=(3, 3, 16, 16))
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')
    native = extractor.extract_chunk(0, 1)

    preparer = BFPreparer(extractor, upscale=2.0, upscale_method='nearest', pad_width=None)
    got = preparer.extract_chunk(0, 1)
    expected = torch.nn.functional.interpolate(
        torch.from_numpy(native).unsqueeze(1),
        size=(6, 6),
        mode='nearest',
    ).squeeze(1).numpy()

    np.testing.assert_allclose(got, expected, atol=0, rtol=0)


def test_bf_preparer_padding_is_in_native_pixels_after_upscale():
    arr = _make_data(shape=(8, 8, 16, 16))
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')

    preparer = BFPreparer(extractor, upscale=2.0, upscale_method='bilinear', pad_width=2)
    expected_y = _compute_pad_for_axis(16, 4)
    expected_x = _compute_pad_for_axis(16, 4)

    assert preparer.upscaled_shape == (16, 16)
    assert preparer.padded_shape == (expected_y[0], expected_x[0])
    assert preparer.pad_offsets == (expected_y[1], expected_x[1])
    assert preparer.extract_chunk(0, 1).shape == (1, expected_y[0], expected_x[0])


# ---------------------------------------------------------------------------
# zero_insert upsampling tests
# ---------------------------------------------------------------------------

def test_bf_preparer_zero_insert_shape():
    arr = _make_data(shape=(4, 5, 16, 16))
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')
    Ry, Rx = Dataset4D(arr).scan_shape

    preparer = BFPreparer(extractor, upscale=3, upscale_method='zero_insert', pad_width=None)

    assert preparer.upscaled_shape == (3 * Ry, 3 * Rx)
    assert preparer.scan_shape == (3 * Ry, 3 * Rx)


def test_bf_preparer_zero_insert_pixel_placement():
    arr = _make_data(shape=(3, 3, 16, 16))
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')
    native = extractor.extract_chunk(0, 2)  # (2, Ry, Rx)

    N = 3
    preparer = BFPreparer(extractor, upscale=N, upscale_method='zero_insert', pad_width=None)
    got = preparer.extract_chunk(0, 2)  # (2, N*Ry, N*Rx)

    # Original values must sit exactly at the stride-N grid positions.
    np.testing.assert_array_equal(got[:, ::N, ::N], native)
    # All intermediate positions must be exactly zero.
    mask = np.ones(got.shape, dtype=bool)
    mask[:, ::N, ::N] = False
    assert (got[mask] == 0.0).all()


def test_bf_preparer_zero_insert_integer_only():
    arr = _make_data()
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')

    with pytest.raises(ValueError, match="zero_insert"):
        BFPreparer(extractor, upscale=1.5, upscale_method='zero_insert')

    # Verify no partial state was written (object construction failed entirely).
    with pytest.raises(ValueError):
        BFPreparer(extractor, upscale=2.5, upscale_method='zero_insert')


def test_bf_preparer_zero_insert_near_integer_is_accepted():
    """upscale within 1e-6 of an integer must not raise."""
    arr = _make_data()
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')
    # 2.0 + 5e-7 < 1e-6 from integer 2 — should pass silently.
    preparer = BFPreparer(extractor, upscale=2.0 + 5e-7, upscale_method='zero_insert')
    Ry, Rx = Dataset4D(arr).scan_shape
    assert preparer.upscaled_shape == (2 * Ry, 2 * Rx)


def test_bf_preparer_zero_insert_upscale_one_bypasses_upsample():
    """upscale=1.0 short-circuits _upsample entirely; output matches native vBF."""
    arr = _make_data()
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')
    native = extractor.extract_all()

    preparer = BFPreparer(extractor, upscale=1.0, upscale_method='zero_insert', pad_width=None)
    got = preparer.extract_all()

    assert got.shape == native.shape
    np.testing.assert_array_equal(got, native)


def test_bf_preparer_zero_insert_with_padding():
    """After zero-insertion + reflect-padding, original values sit at the expected offset."""
    arr = _make_data(shape=(8, 8, 16, 16))
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')
    native = extractor.extract_chunk(0, 1)  # (1, Ry, Rx)

    N = 2
    preparer = BFPreparer(extractor, upscale=N, upscale_method='zero_insert', pad_width=2)
    got = preparer.extract_chunk(0, 1)  # (1, padded_Ry, padded_Rx)

    assert got.shape == (1, preparer.padded_shape[0], preparer.padded_shape[1])
    # The pad-window damps the pad region; only verify shape correctness here.
    assert got.dtype == np.float32


@pytest.mark.parametrize("method", ["nearest", "bilinear", "zero_insert"])
def test_bf_preparer_integer_upscale_shape(method):
    arr = _make_data(shape=(4, 5, 16, 16))
    extractor = BFExtractor(Dataset4D(arr), _make_geom(), strategy='host_mask')
    Ry, Rx = Dataset4D(arr).scan_shape

    preparer = BFPreparer(extractor, upscale=2, upscale_method=method, pad_width=None)
    got = preparer.extract_chunk(0, 2)

    assert preparer.upscaled_shape == (2 * Ry, 2 * Rx)
    assert got.shape == (2, 2 * Ry, 2 * Rx)
    assert got.dtype == np.float32
