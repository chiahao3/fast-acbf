"""AberrationState — nn.Module holding detector-frame aberration coefficients."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AberrationState(torch.nn.Module):
    def __init__(self, ab_dict: dict, max_order: int = None, device='cpu', tolerance_factors: dict = None):
        super().__init__()
        self.orig_ab_dict = ab_dict.copy()
        self.device = torch.device(device)
        self.order_keys = []
        self.tolerance_factors = tolerance_factors
        # tolerance_factors: dict {n: Tn_in_ang} — Kirkland tolerance for each order.
        # https://doi.org/10.1016/j.ultramic.2017.12.002
        # When set, parameters are stored in normalized (dimensionless) form internally;
        # all external accessors (get_cartesian_dict, get_flat_coeffs, get_physical) return physical Å values.

        if max_order is None:
            max_order = max([n for (n, m) in ab_dict.keys()]) if ab_dict else 2

        self.max_order = max_order

        self.coeffs = nn.ParameterDict()

        for n in range(1, max_order + 1):
            start_m = (n + 1) % 2
            Tn = tolerance_factors[n] if tolerance_factors is not None else 1.0

            for m in range(start_m, n + 2, 2):
                self.order_keys.append((n, m))

                if m == 0:
                    val = ab_dict.get((n, m), 0.0)
                    val_clean = float(val) if isinstance(val, (int, float, np.number)) else 0.0

                    key = f"C_{n}_{m}"
                    self.coeffs[key] = nn.Parameter(torch.tensor(val_clean / Tn, device=self.device))

                else:
                    val = ab_dict.get((n, m), {'a': 0.0, 'b': 0.0})
                    if isinstance(val, (int, float, np.number)):
                        val_a, val_b = float(val), 0.0
                    else:
                        val_a = float(val.get('a', 0.0))
                        val_b = float(val.get('b', 0.0))

                    key_a = f"C_{n}_{m}_a"
                    key_b = f"C_{n}_{m}_b"
                    self.coeffs[key_a] = nn.Parameter(torch.tensor(val_a / Tn, device=self.device))
                    self.coeffs[key_b] = nn.Parameter(torch.tensor(val_b / Tn, device=self.device))

    def get_cartesian_dict(self):
        """Reconstructs the nested dictionary format in physical Å units."""
        out = {}
        for (n, m) in self.order_keys:
            Tn = self.tolerance_factors[n] if self.tolerance_factors is not None else 1.0
            if m == 0:
                out[(n, m)] = self.coeffs[f"C_{n}_{m}"].item() * Tn
            else:
                out[(n, m)] = {
                    'a': self.coeffs[f"C_{n}_{m}_a"].item() * Tn,
                    'b': self.coeffs[f"C_{n}_{m}_b"].item() * Tn
                }
        return out

    def get_flat_coeffs(self):
        """Returns all coefficients as a 1D tensor in physical Å units."""
        coeffs_list = []
        tn_list = []
        for (n, m) in self.order_keys:
            Tn = self.tolerance_factors[n] if self.tolerance_factors is not None else 1.0
            if m == 0:
                coeffs_list.append(self.coeffs[f"C_{n}_{m}"])
                tn_list.append(Tn)
            else:
                coeffs_list.append(self.coeffs[f"C_{n}_{m}_a"])
                tn_list.append(Tn)
                coeffs_list.append(self.coeffs[f"C_{n}_{m}_b"])
                tn_list.append(Tn)
        stacked = torch.stack(coeffs_list)
        if self.tolerance_factors is not None:
            tn_tensor = stacked.new_tensor(tn_list)
            stacked = stacked * tn_tensor
        return stacked

    def to_scan_frame(self, rotation_deg: float) -> torch.Tensor:
        """
        Return flat physical coefficients rotated from detector frame to scan frame.

        Differentiable with respect to the stored detector-frame parameters.
        Symmetric terms (m == 0) are rotation-invariant. Each asymmetric Cartesian
        pair (C_a, C_b) is rotated by m * rotation_deg.
        """
        if rotation_deg == 0.0:
            return self.get_flat_coeffs()

        flat = self.get_flat_coeffs()
        out = flat.clone()
        idx = 0
        for _, m in self.order_keys:
            if m == 0:
                idx += 1
                continue

            theta = flat.new_tensor(m * rotation_deg * np.pi / 180.0)
            c = torch.cos(theta)
            s = torch.sin(theta)
            ca = flat[idx]
            cb = flat[idx + 1]
            out[idx] = ca * c - cb * s
            out[idx + 1] = ca * s + cb * c
            idx += 2
        return out

    def get_physical(self, key: str) -> float:
        """Return the physical Å value for a single coefficient key."""
        n = int(key.split('_')[1])
        raw = self.coeffs[key].detach().item()
        return raw * self.tolerance_factors[n] if self.tolerance_factors is not None else raw

    def set_physical(self, key: str, phys_value):
        """Write a physical Å value to a coefficient, normalizing internally."""
        n = int(key.split('_')[1])
        val = phys_value.item() if isinstance(phys_value, torch.Tensor) else float(phys_value)
        norm_val = val / self.tolerance_factors[n] if self.tolerance_factors is not None else val
        with torch.no_grad():
            self.coeffs[key].copy_(torch.tensor(norm_val, dtype=torch.float32, device=self.device))

    def flat_to_cartesian_dict(self, flat: torch.Tensor) -> dict:
        """Convert a flat physical-unit coefficient tensor to the cartesian dict format."""
        out = {}
        idx = 0
        for (n, m) in self.order_keys:
            if m == 0:
                out[(n, m)] = flat[idx].item()
                idx += 1
            else:
                out[(n, m)] = {'a': flat[idx].item(), 'b': flat[idx + 1].item()}
                idx += 2
        return out

    def rebuild_normalization(self, new_tolerance_factors: dict):
        """Re-normalize all parameters under a new set of tolerance factors.

        Reads current physical values, updates self.tolerance_factors, then stores
        parameters normalized by the new Tₙ values.  Use this after overriding
        individual entries in tolerance_factors to keep physical ↔ internal mapping
        consistent.
        """
        phys = self.get_cartesian_dict()
        self.tolerance_factors = new_tolerance_factors
        with torch.no_grad():
            for (n, m), val in phys.items():
                Tn = new_tolerance_factors[n]
                if m == 0:
                    key = f"C_{n}_{m}"
                    self.coeffs[key].copy_(torch.tensor(val / Tn, dtype=torch.float32, device=self.device))
                else:
                    for suffix, v in [('_a', val['a']), ('_b', val['b'])]:
                        key = f"C_{n}_{m}{suffix}"
                        self.coeffs[key].copy_(torch.tensor(v / Tn, dtype=torch.float32, device=self.device))
