"""Unit tests for Dataset4D normalize= flag and norm_factor property."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from fast_acbf.data.dataset4d import Dataset4D


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_arr(seed=0, shape=(4, 4, 8, 8)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(0.1, 1.0, shape).astype(np.float32)
    return arr


def _expected_norm_factor(arr: np.ndarray) -> float:
    mean_dp = arr.mean(axis=(0, 1), dtype=np.float32)
    return float(mean_dp.max())


# ── in-memory tests ───────────────────────────────────────────────────────────

def test_normalize_default_off():
    arr = _make_arr()
    ds = Dataset4D(arr)
    assert ds.norm_factor is None
    np.testing.assert_array_equal(ds._array, arr)


def test_normalize_in_memory():
    arr = _make_arr()
    expected_factor = _expected_norm_factor(arr)
    ds = Dataset4D(arr, normalize=True)

    assert ds.norm_factor == pytest.approx(expected_factor, rel=1e-5)
    np.testing.assert_allclose(ds._array, arr / np.float32(expected_factor), atol=1e-6)
    assert not ds.is_lazy


def test_normalize_in_memory_get_bf_chunk():
    arr = _make_arr()
    ds_norm = Dataset4D(arr, normalize=True)
    ds_raw = Dataset4D(arr)

    iy = np.array([0, 1, 2])
    ix = np.array([0, 1, 2])
    chunk_norm = ds_norm.get_bf_chunk(iy, ix)
    chunk_raw = ds_raw.get_bf_chunk(iy, ix)

    np.testing.assert_allclose(
        chunk_norm,
        chunk_raw / np.float32(ds_norm.norm_factor),
        atol=1e-6,
    )


def test_normalize_in_memory_get_virtual_img():
    arr = _make_arr()
    ds_norm = Dataset4D(arr, normalize=True)
    factor = ds_norm.norm_factor

    img = ds_norm.get_virtual_img(0, 0)
    expected = arr[:, :, 0, 0] / np.float32(factor)
    np.testing.assert_allclose(img, expected, atol=1e-6)


def test_normalize_invalid_data_zeros():
    arr = np.zeros((4, 4, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="norm_factor"):
        Dataset4D(arr, normalize=True)


def test_normalize_invalid_data_nan():
    arr = np.full((4, 4, 8, 8), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="norm_factor"):
        Dataset4D(arr, normalize=True)


def test_normalize_invalid_data_neg_all():
    arr = np.full((4, 4, 8, 8), -1.0, dtype=np.float32)
    with pytest.raises(ValueError, match="norm_factor"):
        Dataset4D(arr, normalize=True)


def test_normalize_does_not_mutate_caller_array():
    arr = _make_arr()
    arr_copy = arr.copy()
    Dataset4D(arr, normalize=True)
    np.testing.assert_array_equal(arr, arr_copy)


def test_normalize_idempotency_guard():
    arr = _make_arr()
    ds = Dataset4D(arr, normalize=True)
    with pytest.raises(RuntimeError, match="already-normalized"):
        ds._apply_normalization()


# ── crop tests ────────────────────────────────────────────────────────────────

def test_normalize_crop_roi_inmemory():
    arr = _make_arr(shape=(8, 8, 8, 8))
    ds = Dataset4D(arr, normalize=True)
    factor = ds.norm_factor

    cropped = ds.crop_scan_roi(1, 4, 2, 5)
    assert cropped.norm_factor == factor
    assert not cropped.is_lazy

    expected = arr[1:4, 2:5] / np.float32(factor)
    np.testing.assert_allclose(cropped._array, expected, atol=1e-6)


def test_normalize_crop_roi_unnormalized_inmemory():
    arr = _make_arr(shape=(8, 8, 8, 8))
    ds = Dataset4D(arr)
    cropped = ds.crop_scan_roi(1, 4, 2, 5)
    assert cropped.norm_factor is None
    np.testing.assert_array_equal(cropped._array, arr[1:4, 2:5])


# ── lazy HDF5 tests ───────────────────────────────────────────────────────────

class TestNormalizeLazyHDF5:
    @pytest.fixture(autouse=True)
    def _h5file(self, tmp_path):
        h5py = pytest.importorskip("h5py")
        self.arr = _make_arr()
        self.path = tmp_path / "test.h5"
        with h5py.File(self.path, 'w') as f:
            f.create_dataset('data', data=self.arr)

    def test_norm_factor_set(self):
        ds = Dataset4D.from_hdf5(self.path, normalize=True)
        expected = _expected_norm_factor(self.arr)
        assert ds.norm_factor == pytest.approx(expected, rel=1e-4)
        assert ds.is_lazy

    def test_get_bf_chunk_matches_reference(self):
        ds = Dataset4D.from_hdf5(self.path, normalize=True)
        ds_ref = Dataset4D(self.arr, normalize=True)

        iy = np.array([0, 1, 2])
        ix = np.array([0, 1, 2])
        np.testing.assert_allclose(
            ds.get_bf_chunk(iy, ix), ds_ref.get_bf_chunk(iy, ix), atol=1e-5,
        )

    def test_get_virtual_img_matches_reference(self):
        ds = Dataset4D.from_hdf5(self.path, normalize=True)
        ds_ref = Dataset4D(self.arr, normalize=True)

        img_lazy = ds.get_virtual_img(0, 0)
        img_ref = ds_ref.get_virtual_img(0, 0)
        np.testing.assert_allclose(img_lazy, img_ref, atol=1e-5)

    def test_disk_data_unchanged(self):
        """Normalization must never write to the HDF5 file."""
        h5py = pytest.importorskip("h5py")
        Dataset4D.from_hdf5(self.path, normalize=True)
        with h5py.File(self.path, 'r') as f:
            np.testing.assert_array_equal(f['data'][:], self.arr)

    def test_no_normalize_flag(self):
        ds = Dataset4D.from_hdf5(self.path)
        assert ds.norm_factor is None

    def test_force_materialize_applies_normalization(self):
        ds = Dataset4D.from_hdf5(self.path, normalize=True)
        factor = ds.norm_factor

        ds._force_materialize()
        assert not ds.is_lazy
        assert ds.norm_factor == factor

        expected = self.arr / np.float32(factor)
        np.testing.assert_allclose(ds._array, expected, atol=1e-5)

    def test_materialize_true_normalize_true(self):
        ds = Dataset4D.from_hdf5(self.path, materialize=True, normalize=True)
        assert not ds.is_lazy

        expected = self.arr / np.float32(_expected_norm_factor(self.arr))
        np.testing.assert_allclose(ds._array, expected, atol=1e-5)

    def test_crop_roi_lazy(self):
        ds = Dataset4D.from_hdf5(self.path, normalize=True)
        factor = ds.norm_factor

        cropped = ds.crop_scan_roi(0, 2, 0, 2)
        assert cropped.norm_factor == factor
        assert not cropped.is_lazy

        expected = self.arr[0:2, 0:2] / np.float32(factor)
        np.testing.assert_allclose(cropped._array, expected, atol=1e-5)


# ── lazy zarr tests ───────────────────────────────────────────────────────────

class TestNormalizeLazyZarr:
    @pytest.fixture(autouse=True)
    def _zarrstore(self, tmp_path):
        zarr = pytest.importorskip("zarr")
        self.arr = _make_arr()
        self.path = tmp_path / "test.zarr"
        store = zarr.open(str(self.path), mode='w')
        store['data'] = self.arr

    def test_norm_factor_set(self):
        ds = Dataset4D.from_zarr(str(self.path), normalize=True)
        expected = _expected_norm_factor(self.arr)
        assert ds.norm_factor == pytest.approx(expected, rel=1e-4)
        assert ds.is_lazy

    def test_get_bf_chunk_matches_reference(self):
        ds = Dataset4D.from_zarr(str(self.path), normalize=True)
        ds_ref = Dataset4D(self.arr, normalize=True)

        iy = np.array([0, 1, 2])
        ix = np.array([0, 1, 2])
        np.testing.assert_allclose(
            ds.get_bf_chunk(iy, ix), ds_ref.get_bf_chunk(iy, ix), atol=1e-5,
        )

    def test_get_virtual_img_matches_reference(self):
        ds = Dataset4D.from_zarr(str(self.path), normalize=True)
        ds_ref = Dataset4D(self.arr, normalize=True)

        np.testing.assert_allclose(
            ds.get_virtual_img(0, 0), ds_ref.get_virtual_img(0, 0), atol=1e-5,
        )

    def test_disk_data_unchanged(self):
        zarr = pytest.importorskip("zarr")
        Dataset4D.from_zarr(str(self.path), normalize=True)
        store = zarr.open(str(self.path), mode='r')
        np.testing.assert_array_equal(store['data'][:], self.arr)

    def test_force_materialize_applies_normalization(self):
        ds = Dataset4D.from_zarr(str(self.path), normalize=True)
        factor = ds.norm_factor

        ds._force_materialize()
        assert not ds.is_lazy
        assert ds.norm_factor == factor

        np.testing.assert_allclose(
            ds._array, self.arr / np.float32(factor), atol=1e-5,
        )


# ── BFSolver normalize= tests ─────────────────────────────────────────────────

def test_bfsolver_normalize_warning_on_prebuilt_dataset():
    from fast_acbf import BFSolver
    arr = _make_arr(shape=(8, 8, 32, 32))
    ds = Dataset4D(arr)

    dk = (25.0 / 1e3) / (32 / 2 * 0.04176)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BFSolver(
            dataset=ds,
            max_alpha=25.0,
            scan_step_size=0.2,
            dk=dk,
            wavelength=0.04176,
            device='cpu',
            normalize=True,
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "already-constructed Dataset4D" in str(user_warnings[0].message)


def test_bfsolver_normalize_array_input():
    from fast_acbf import BFSolver
    arr = _make_arr(shape=(8, 8, 32, 32))
    expected_factor = _expected_norm_factor(arr)
    dk = (25.0 / 1e3) / (32 / 2 * 0.04176)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = BFSolver(
            dataset=arr,
            max_alpha=25.0,
            scan_step_size=0.2,
            dk=dk,
            wavelength=0.04176,
            device='cpu',
            normalize=True,
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 0
    assert solver._dataset.norm_factor == pytest.approx(expected_factor, rel=1e-4)
