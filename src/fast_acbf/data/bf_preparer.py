"""BFPreparer — mirror-pad + Tukey window applied to vBF images before FFT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from fast_acbf.data.bf_extractor import BFExtractor


def _ceil_5smooth(n: int) -> int:
    """Smallest 5-smooth integer >= n. (Only prime factors 2, 3, 5.)"""
    candidate = n
    while True:
        x = candidate
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return candidate
        candidate += 1


def _compute_pad_for_axis(orig: int, min_pad: int) -> tuple[int, int, int]:
    """
    Find the smallest 5-smooth size >= orig + 2*min_pad.
    Returns (padded_size, pad_before, pad_after).
      pad_before = (padded_size - orig) // 2
      pad_after  = padded_size - orig - pad_before   <- at most pad_before + 1
    Both values are >= min_pad.
    """
    target = orig + 2 * min_pad
    padded = _ceil_5smooth(target)
    extra = padded - orig
    pad_before = extra // 2
    pad_after = extra - pad_before
    return padded, pad_before, pad_after


def _make_tukey_1d(total: int, pad_before: int, pad_after: int) -> np.ndarray:
    """
    Tukey window of length total.
      indices 0..pad_before-1         : cosine rise  0 -> 1
      indices pad_before..total-pad_after-1: flat 1.0
      indices total-pad_after..total-1: cosine fall 1 -> 0
    """
    w = np.ones(total, dtype=np.float32)
    if pad_before > 0:
        i = np.arange(pad_before, dtype=np.float32)
        w[:pad_before] = 0.5 * (1.0 - np.cos(np.pi * i / pad_before))
    if pad_after > 0:
        i = np.arange(pad_after, dtype=np.float32)
        w[total - pad_after:] = 0.5 * (1.0 - np.cos(np.pi * (pad_after - 1 - i) / pad_after))
    return w


def _make_tukey_2d(
    Ry: int, Rx: int,
    pad_top: int, pad_bottom: int,
    pad_left: int, pad_right: int,
) -> np.ndarray:
    """Outer product of per-axis Tukey windows. Shape (Ry, Rx), float32."""
    wy = _make_tukey_1d(Ry, pad_top, pad_bottom)
    wx = _make_tukey_1d(Rx, pad_left, pad_right)
    return np.outer(wy, wx).astype(np.float32)


class BFPreparer:
    """Wraps BFExtractor; applies mirror-padding + Tukey window before images reach ImageFFT."""

    def __init__(self, extractor: BFExtractor, pad_width: int) -> None:
        if pad_width <= 0:
            raise ValueError(f"pad_width must be positive, got {pad_width}.")
        Ry, Rx = extractor.scan_shape
        if pad_width >= min(Ry, Rx):
            raise ValueError(
                f"pad_width={pad_width} must be < min(Ry={Ry}, Rx={Rx}) for reflect padding."
            )
        Ry_p, pad_top, pad_bottom = _compute_pad_for_axis(Ry, pad_width)
        Rx_p, pad_left, pad_right = _compute_pad_for_axis(Rx, pad_width)

        self._extractor = extractor
        self.orig_shape = (Ry, Rx)
        self.padded_shape = (Ry_p, Rx_p)
        self.pad_offsets = (pad_top, pad_left)  # (pad_top, pad_left) for fov crop
        self.pad_top, self.pad_bottom = pad_top, pad_bottom
        self.pad_left, self.pad_right = pad_left, pad_right
        self._window_np = _make_tukey_2d(Ry_p, Rx_p, pad_top, pad_bottom, pad_left, pad_right)

    # Duck-typed ImageFFT extractor interface
    @property
    def nb(self) -> int:
        return self._extractor.nb

    @property
    def scan_shape(self) -> tuple[int, int]:
        return self.padded_shape  # override → ImageFFT allocates padded cache

    @property
    def strategy(self) -> str:
        return self._extractor.strategy

    @property
    def bf_iy(self):
        return self._extractor.bf_iy

    @property
    def bf_ix(self):
        return self._extractor.bf_ix

    def extract_chunk(self, b_start: int, b_end: int):
        return self._pad_and_window(self._extractor.extract_chunk(b_start, b_end))

    def extract_all(self):
        return self._pad_and_window(self._extractor.extract_all())

    def _pad_and_window(self, vbf):
        """vbf: (B, Ry, Rx) numpy or torch. Returns same type, padded and windowed."""
        if isinstance(vbf, torch.Tensor):
            padded = F.pad(
                vbf,
                (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
                mode='reflect',
            )
            w = torch.as_tensor(self._window_np, dtype=torch.float32, device=vbf.device)
            return padded * w
        else:
            padded = np.pad(
                vbf,
                ((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)),
                mode='reflect',
            ).astype(np.float32)
            return padded * self._window_np


__all__ = ["BFPreparer", "_compute_pad_for_axis"]
