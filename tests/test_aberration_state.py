"""Unit tests for AberrationState — aberration coefficient bookkeeping."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fast_acbf.bf_solver import AberrationState


@pytest.fixture
def minimal_ab_dict():
    return {(1, 0): 50.0}


@pytest.fixture
def full_ab_dict():
    return {(1, 0): 50.0, (1, 2): {'a': 5.0, 'b': 8.66}}


# ── Initialization ────────────────────────────────────────────────────────────

class TestAberrationStateInit:

    def test_max_order_inferred_from_dict(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=None, device='cpu')
        assert state.max_order == 1

    def test_max_order_explicit_overrides(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=3, device='cpu')
        assert state.max_order == 3

    def test_order_keys_complete_for_max_order_2(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=2, device='cpu')
        assert (1, 0) in state.order_keys
        assert (1, 2) in state.order_keys
        assert (2, 1) in state.order_keys
        assert (2, 3) in state.order_keys

    def test_coeffs_on_cpu(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=1, device='cpu')
        for param in state.coeffs.values():
            assert param.device.type == 'cpu'

    def test_coeffs_are_nn_parameters(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=1, device='cpu')
        for param in state.coeffs.values():
            assert isinstance(param, nn.Parameter)


# ── get_physical / set_physical ───────────────────────────────────────────────

class TestGetSetPhysical:

    def test_get_physical_matches_init(self, full_ab_dict):
        state = AberrationState(full_ab_dict, max_order=2, device='cpu')
        assert state.get_physical('C_1_0') == pytest.approx(50.0, abs=1e-4)

    def test_set_then_get_roundtrip(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=1, device='cpu')
        state.set_physical('C_1_0', 99.9)
        assert state.get_physical('C_1_0') == pytest.approx(99.9, abs=1e-4)

    def test_set_then_get_roundtrip_with_tolerance_factors(self, minimal_ab_dict):
        state = AberrationState(
            minimal_ab_dict, max_order=1, device='cpu',
            tolerance_factors={1: 100.0}
        )
        state.set_physical('C_1_0', 75.0)
        assert state.get_physical('C_1_0') == pytest.approx(75.0, abs=1e-4)

    def test_set_physical_accepts_tensor(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=1, device='cpu')
        state.set_physical('C_1_0', torch.tensor(33.3))
        assert state.get_physical('C_1_0') == pytest.approx(33.3, abs=1e-4)


# ── get_flat_coeffs ───────────────────────────────────────────────────────────

class TestFlatCoeffs:

    def test_is_1d_tensor(self, full_ab_dict):
        state = AberrationState(full_ab_dict, max_order=2, device='cpu')
        assert state.get_flat_coeffs().ndim == 1

    def test_length_for_max_order_2(self, full_ab_dict):
        # max_order=2: C10, C12a, C12b, C21a, C21b, C23a, C23b → 7
        state = AberrationState(full_ab_dict, max_order=2, device='cpu')
        assert len(state.get_flat_coeffs()) == 7

    def test_values_match_cartesian_dict(self, full_ab_dict):
        state = AberrationState(full_ab_dict, max_order=2, device='cpu')
        flat = state.get_flat_coeffs()
        cart = state.get_cartesian_dict()
        assert flat[0].item() == pytest.approx(float(cart[(1, 0)]), abs=1e-4)
        assert flat[1].item() == pytest.approx(float(cart[(1, 2)]['a']), abs=1e-4)
        assert flat[2].item() == pytest.approx(float(cart[(1, 2)]['b']), abs=1e-4)


# ── get_cartesian_dict ────────────────────────────────────────────────────────

class TestCartesianDict:

    def test_round_term_is_scalar(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=1, device='cpu')
        val = state.get_cartesian_dict()[(1, 0)]
        assert not isinstance(val, dict)

    def test_asymmetric_term_is_dict_with_ab(self, full_ab_dict):
        state = AberrationState(full_ab_dict, max_order=2, device='cpu')
        val = state.get_cartesian_dict()[(1, 2)]
        assert isinstance(val, dict) and 'a' in val and 'b' in val

    def test_unset_terms_initialize_to_zero(self, minimal_ab_dict):
        state = AberrationState(minimal_ab_dict, max_order=2, device='cpu')
        cart = state.get_cartesian_dict()
        assert float(cart[(1, 2)]['a']) == pytest.approx(0.0, abs=1e-6)
        assert float(cart[(1, 2)]['b']) == pytest.approx(0.0, abs=1e-6)
