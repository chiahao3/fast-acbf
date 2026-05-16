"""Unit tests for CoordinateTransform — flip/transpose/rotation logic.

Convention: apply_to_centers(kY, kX) returns (kX_out, kY_out).
Transform order: flipud → fliplr → transpose → rotation (CCW-positive).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from fast_acbf.data.geometry import CoordinateTransform


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ct(**kwargs) -> CoordinateTransform:
    return CoordinateTransform(**kwargs)


def _centers(ct: CoordinateTransform, ky: float, kx: float, **kw):
    """Apply ct to scalar (ky, kx); return (kx_out, ky_out) as Python floats."""
    kX_out, kY_out = ct.apply_to_centers(
        torch.tensor([ky], dtype=torch.float32),
        torch.tensor([kx], dtype=torch.float32),
        **kw,
    )
    return kX_out.item(), kY_out.item()


# Reference implementation — order-independent from the production code.
def _ref(flipud, fliplr, transpose, rotation_deg, ky_in, kx_in, in_scan_frame=True):
    ky, kx = ky_in, kx_in
    if flipud:
        ky = -ky
    if fliplr:
        kx = -kx
    if transpose:
        ky, kx = kx, ky
    if in_scan_frame and rotation_deg:
        theta = np.deg2rad(rotation_deg)
        kx, ky = kx * np.cos(theta) - ky * np.sin(theta), kx * np.sin(theta) + ky * np.cos(theta)
    return kx, ky


# ── Construction helpers ──────────────────────────────────────────────────────

class TestFromDict:
    def test_none_returns_identity(self):
        assert CoordinateTransform.from_dict(None) == CoordinateTransform()

    def test_empty_dict_returns_identity(self):
        assert CoordinateTransform.from_dict({}) == CoordinateTransform()

    def test_partial_dict_fills_defaults(self):
        ct = CoordinateTransform.from_dict({'flipud': True, 'rotation_deg': 30.0})
        assert ct.flipud is True
        assert ct.fliplr is False
        assert ct.transpose is False
        assert ct.rotation_deg == 30.0

    def test_round_trip(self):
        ct = CoordinateTransform(flipud=True, fliplr=False, transpose=True, rotation_deg=45.0)
        assert CoordinateTransform.from_dict(ct.to_dict()) == ct


class TestWithRotation:
    def test_replaces_only_rotation(self):
        ct = CoordinateTransform(flipud=True, fliplr=True, transpose=True, rotation_deg=10.0)
        ct2 = ct.with_rotation(20.0)
        assert ct2.flipud is True
        assert ct2.fliplr is True
        assert ct2.transpose is True
        assert ct2.rotation_deg == 20.0

    def test_zero_rotation(self):
        ct = _ct(rotation_deg=45.0).with_rotation(0.0)
        assert ct.rotation_deg == 0.0


# ── apply_to_centers — 8 flip combos, no rotation ────────────────────────────
# Input: kY=1.0, kX=2.0. Expected (kX_out, kY_out) derived from the transform order.

_FLIP_CASES = [
    # (flipud, fliplr, transpose, kX_out, kY_out)
    (False, False, False,  2.0,  1.0),
    (True,  False, False,  2.0, -1.0),  # flipud: ky → -ky
    (False, True,  False, -2.0,  1.0),  # fliplr: kx → -kx
    (True,  True,  False, -2.0, -1.0),  # both flips
    (False, False, True,   1.0,  2.0),  # transpose: swap kx↔ky
    (True,  False, True,  -1.0,  2.0),  # flipud then transpose
    (False, True,  True,   1.0, -2.0),  # fliplr then transpose
    (True,  True,  True,  -1.0, -2.0),  # all three
]

@pytest.mark.parametrize("flipud,fliplr,transpose,exp_kx,exp_ky", _FLIP_CASES)
def test_centers_flip_combos_no_rotation(flipud, fliplr, transpose, exp_kx, exp_ky):
    ct = _ct(flipud=flipud, fliplr=fliplr, transpose=transpose)
    kx_out, ky_out = _centers(ct, ky=1.0, kx=2.0)
    assert kx_out == pytest.approx(exp_kx, abs=1e-5)
    assert ky_out == pytest.approx(exp_ky, abs=1e-5)


# ── apply_to_centers — 8 flip combos + 90° rotation ─────────────────────────
# CCW 90°: (kx, ky) → (-ky, kx). Input: kY=1.0, kX=2.0.
# Expected values derived from: apply flips first, then CCW-90 rotation.

_FLIP_ROT90_CASES = [
    # (flipud, fliplr, transpose, kX_out, kY_out)
    (False, False, False, -1.0,  2.0),  # (2,1) → (-1,2)
    (True,  False, False,  1.0,  2.0),  # (2,-1) → (1,2)
    (False, True,  False, -1.0, -2.0),  # (-2,1) → (-1,-2)
    (True,  True,  False,  1.0, -2.0),  # (-2,-1) → (1,-2)
    (False, False, True,  -2.0,  1.0),  # (1,2) → (-2,1)
    (True,  False, True,  -2.0, -1.0),  # (-1,2) → (-2,-1)
    (False, True,  True,   2.0,  1.0),  # (1,-2) → (2,1)
    (True,  True,  True,   2.0, -1.0),  # (-1,-2) → (2,-1)
]

@pytest.mark.parametrize("flipud,fliplr,transpose,exp_kx,exp_ky", _FLIP_ROT90_CASES)
def test_centers_flip_combos_rot90(flipud, fliplr, transpose, exp_kx, exp_ky):
    ct = _ct(flipud=flipud, fliplr=fliplr, transpose=transpose, rotation_deg=90.0)
    kx_out, ky_out = _centers(ct, ky=1.0, kx=2.0)
    assert kx_out == pytest.approx(exp_kx, abs=1e-5)
    assert ky_out == pytest.approx(exp_ky, abs=1e-5)


# ── apply_to_centers — all 8 combos at rotation_deg=15° vs. reference ────────

@pytest.mark.parametrize(
    "flipud,fliplr,transpose",
    list(itertools.product([False, True], repeat=3)),
)
def test_centers_vs_reference_rot15(flipud, fliplr, transpose):
    ct = _ct(flipud=flipud, fliplr=fliplr, transpose=transpose, rotation_deg=15.0)
    kx_out, ky_out = _centers(ct, ky=1.0, kx=2.0)
    exp_kx, exp_ky = _ref(flipud, fliplr, transpose, 15.0, 1.0, 2.0)
    assert kx_out == pytest.approx(exp_kx, abs=1e-5)
    assert ky_out == pytest.approx(exp_ky, abs=1e-5)


# ── in_scan_frame=False skips rotation ───────────────────────────────────────

def test_in_scan_frame_false_skips_rotation():
    ct = _ct(rotation_deg=45.0)
    kx_scan, ky_scan = _centers(ct, ky=1.0, kx=2.0, in_scan_frame=True)
    kx_det, ky_det = _centers(ct, ky=1.0, kx=2.0, in_scan_frame=False)
    # Detector frame: no rotation → same as identity
    assert kx_det == pytest.approx(2.0, abs=1e-5)
    assert ky_det == pytest.approx(1.0, abs=1e-5)
    # Scan frame: rotation applied → different
    assert kx_scan != pytest.approx(2.0, abs=1e-3)


def test_in_scan_frame_false_still_applies_flips():
    ct = _ct(flipud=True, fliplr=True, rotation_deg=45.0)
    kx_out, ky_out = _centers(ct, ky=1.0, kx=2.0, in_scan_frame=False)
    # flipud+fliplr, no rotation → kx=-2, ky=-1
    assert kx_out == pytest.approx(-2.0, abs=1e-5)
    assert ky_out == pytest.approx(-1.0, abs=1e-5)


# ── Algebraic invariants ──────────────────────────────────────────────────────

def test_identity_transform_is_noop():
    ct = CoordinateTransform()
    kx_out, ky_out = _centers(ct, ky=3.7, kx=-1.5)
    assert kx_out == pytest.approx(-1.5, abs=1e-6)
    assert ky_out == pytest.approx(3.7, abs=1e-6)


@pytest.mark.parametrize("field", ["flipud", "fliplr", "transpose"])
def test_double_flip_is_identity(field):
    ct = _ct(**{field: True})
    kY = torch.tensor([1.3, -0.5, 2.1], dtype=torch.float32)
    kX = torch.tensor([-0.7, 3.0, 0.4], dtype=torch.float32)
    kx1, ky1 = ct.apply_to_centers(kY, kX)
    kx2, ky2 = ct.apply_to_centers(ky1, kx1)  # second application
    torch.testing.assert_close(kx2, kX, atol=1e-5, rtol=0)
    torch.testing.assert_close(ky2, kY, atol=1e-5, rtol=0)


def test_rotation_inverse_is_identity():
    ct_fwd = _ct(rotation_deg=37.0)
    ct_inv = _ct(rotation_deg=-37.0)
    kY = torch.tensor([1.3, -0.5], dtype=torch.float32)
    kX = torch.tensor([-0.7, 3.0], dtype=torch.float32)
    kx1, ky1 = ct_fwd.apply_to_centers(kY, kX)
    kx2, ky2 = ct_inv.apply_to_centers(ky1, kx1)
    torch.testing.assert_close(kx2, kX, atol=1e-5, rtol=0)
    torch.testing.assert_close(ky2, kY, atol=1e-5, rtol=0)


def test_four_rot90_is_identity():
    ct = _ct(rotation_deg=90.0)
    kY = torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32)
    kX = torch.tensor([2.0, 0.5, -1.1], dtype=torch.float32)
    ky_cur, kx_cur = kY.clone(), kX.clone()
    for _ in range(4):
        kx_cur, ky_cur = ct.apply_to_centers(ky_cur, kx_cur)
    torch.testing.assert_close(kx_cur, kX, atol=1e-4, rtol=0)
    torch.testing.assert_close(ky_cur, kY, atol=1e-4, rtol=0)


def test_flipud_fliplr_equals_rot180():
    """flipud+fliplr is equivalent to a 180° rotation (negate both axes)."""
    ct_flips = _ct(flipud=True, fliplr=True)
    ct_rot = _ct(rotation_deg=180.0)
    kY = torch.tensor([1.0, -0.5, 0.3], dtype=torch.float32)
    kX = torch.tensor([2.0, 1.5, -0.7], dtype=torch.float32)
    kx_f, ky_f = ct_flips.apply_to_centers(kY, kX)
    kx_r, ky_r = ct_rot.apply_to_centers(kY, kX)
    torch.testing.assert_close(kx_f, kx_r, atol=1e-5, rtol=0)
    torch.testing.assert_close(ky_f, ky_r, atol=1e-5, rtol=0)


# ── apply_to_grids — 2D tensors behave identically to 1D centers ─────────────

@pytest.mark.parametrize("flipud,fliplr,transpose", [
    (False, False, False),
    (True,  False, True),
    (False, True,  True),
    (True,  True,  True),
])
def test_grids_match_centers_elementwise(flipud, fliplr, transpose):
    ct = _ct(flipud=flipud, fliplr=fliplr, transpose=transpose, rotation_deg=15.0)
    kY_grid = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    kX_grid = torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float32)

    kx_g, ky_g = ct.apply_to_grids(kY_grid, kX_grid)

    # Verify element-by-element against apply_to_centers
    for i in range(2):
        for j in range(2):
            kx_c, ky_c = ct.apply_to_centers(
                kY_grid[i, j].unsqueeze(0), kX_grid[i, j].unsqueeze(0)
            )
            assert kx_g[i, j].item() == pytest.approx(kx_c.item(), abs=1e-5)
            assert ky_g[i, j].item() == pytest.approx(ky_c.item(), abs=1e-5)


def test_grids_output_shape_preserved():
    ct = _ct(flipud=True, transpose=True, rotation_deg=30.0)
    kY_grid = torch.zeros(5, 7, dtype=torch.float32)
    kX_grid = torch.zeros(5, 7, dtype=torch.float32)
    kx_g, ky_g = ct.apply_to_grids(kY_grid, kX_grid)
    assert kx_g.shape == (5, 7)
    assert ky_g.shape == (5, 7)
