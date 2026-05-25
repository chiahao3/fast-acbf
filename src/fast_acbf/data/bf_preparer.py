"""BFPreparer -- real-space vBF preparation before FFT caching."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from fast_acbf.data.bf_extractor import BFExtractor


_VALID_UPSCALE_METHODS = ("nearest", "bilinear", "zero_insert")


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
    """
    target = orig + 2 * min_pad
    padded = _ceil_5smooth(target)
    extra = padded - orig
    pad_before = extra // 2
    pad_after = extra - pad_before
    return padded, pad_before, pad_after


def _make_tukey_1d(total: int, pad_before: int, pad_after: int) -> np.ndarray:
    """Tukey window with flat unit central region and cosine pad rolloff."""
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


def _check_zero_insert_compat(method: str, upscale: float) -> None:
    """Raise before any state mutation if zero_insert is paired with a non-integer upscale."""
    if method == "zero_insert" and abs(upscale - round(upscale)) > 1e-6:
        raise ValueError(
            f"upscale_method='zero_insert' requires an integer upscale factor, "
            f"got upscale={upscale!r}. Use 'nearest' or 'bilinear' for fractional upscale."
        )


def _normalize_upscale_method(method: str) -> str:
    method = str(method).strip().lower()
    if method not in _VALID_UPSCALE_METHODS:
        raise ValueError(
            f"upscale_method must be one of {_VALID_UPSCALE_METHODS}, got {method!r}."
        )
    return method


def _prepared_shapes(
    raw_shape: tuple[int, int],
    upscale: float,
    pad_width: int | None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int, int], bool]:
    """Return upscaled/padded shapes and pad metadata for a preparation state."""
    if upscale < 1.0:
        raise ValueError(f"upscale must be >= 1.0, got {upscale}.")
    Ry, Rx = raw_shape
    Ry_up = round(Ry * upscale)
    Rx_up = round(Rx * upscale)
    if Ry_up <= 0 or Rx_up <= 0:
        raise ValueError(f"upscale produces invalid shape {(Ry_up, Rx_up)}.")

    if pad_width is None:
        return (Ry_up, Rx_up), (Ry_up, Rx_up), (0, 0), (0, 0, 0, 0), False

    pad_width = int(pad_width)
    if pad_width < 0:
        raise ValueError(f"pad_width must be non-negative, got {pad_width}.")
    if pad_width == 0:
        return (Ry_up, Rx_up), (Ry_up, Rx_up), (0, 0), (0, 0, 0, 0), False

    effective_pad = round(pad_width * upscale)
    if effective_pad <= 0:
        return (Ry_up, Rx_up), (Ry_up, Rx_up), (0, 0), (0, 0, 0, 0), False
    if effective_pad >= min(Ry_up, Rx_up):
        raise ValueError(
            f"effective pad_width={effective_pad} must be < min(Ry={Ry_up}, Rx={Rx_up}) "
            "after upscaling."
        )

    Ry_p, pad_top, pad_bottom = _compute_pad_for_axis(Ry_up, effective_pad)
    Rx_p, pad_left, pad_right = _compute_pad_for_axis(Rx_up, effective_pad)
    if max(pad_top, pad_bottom) >= Ry_up or max(pad_left, pad_right) >= Rx_up:
        raise ValueError(
            f"Effective padding after 5-smooth rounding "
            f"(top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}) "
            f"equals or exceeds upscaled input dimension (Ry={Ry_up}, Rx={Rx_up}). "
            "Reduce pad_width."
        )
    return (
        (Ry_up, Rx_up),
        (Ry_p, Rx_p),
        (pad_top, pad_left),
        (pad_top, pad_bottom, pad_left, pad_right),
        True,
    )


class BFPreparer:
    """Wraps BFExtractor; applies real-space upscale, padding, and windowing."""

    def __init__(
        self,
        extractor: BFExtractor,
        upscale: float = 1.0,
        upscale_method: str = "bilinear",
        pad_width: int | None = None,
    ) -> None:
        _check_zero_insert_compat(str(upscale_method).strip().lower(), float(upscale))
        self._extractor = extractor
        self.upscale = float(upscale)
        self.upscale_method = _normalize_upscale_method(upscale_method)
        if self.upscale_method == "zero_insert" and self.upscale != 1.0:
            warnings.warn(
                "upscale_method='zero_insert' is experimental: reconstruction intensity is not "
                "normalized for non-uniform coverage from sub-pixel shifts. Divide by a "
                "reweighting map before interpreting intensities.",
                UserWarning,
                stacklevel=2,
            )
        self.pad_width = None if pad_width is None or int(pad_width) == 0 else int(pad_width)

        self.raw_shape = tuple(extractor.scan_shape)
        (
            self.upscaled_shape,
            self.padded_shape,
            self.pad_offsets,
            pads,
            self._has_pad,
        ) = _prepared_shapes(self.raw_shape, self.upscale, self.pad_width)
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = pads
        self._window_np = (
            _make_tukey_2d(
                self.padded_shape[0],
                self.padded_shape[1],
                self.pad_top,
                self.pad_bottom,
                self.pad_left,
                self.pad_right,
            )
            if self._has_pad
            else None
        )

    # Duck-typed ImageFFT extractor interface
    @property
    def nb(self) -> int:
        return self._extractor.nb

    @property
    def scan_shape(self) -> tuple[int, int]:
        return self.padded_shape

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
        return self._prepare(self._extractor.extract_chunk(b_start, b_end))

    def extract_all(self):
        return self._prepare(self._extractor.extract_all())

    def _prepare(self, vbf):
        if self.upscaled_shape != self.raw_shape:
            vbf = self._upsample(vbf)
        if self._has_pad:
            vbf = self._pad_and_window(vbf)
        return vbf

    def _upsample(self, vbf):
        is_tensor = isinstance(vbf, torch.Tensor)
        if is_tensor:
            src = vbf.float()
            device = src.device
        else:
            src = torch.from_numpy(np.asarray(vbf, dtype=np.float32))
            device = torch.device("cpu")

        inp = src.to(device).unsqueeze(1)  # (Nb, 1, Ry, Rx)
        if self.upscale_method == "zero_insert":
            # Insert zeros between native pixels — no convolution kernel applied.
            # Values sit at [0, N, 2N, …, (Ry-1)*N]; trailing N-1 rows/cols are zero.
            N = round(self.upscale)
            Ry_up, Rx_up = self.upscaled_shape
            out = torch.zeros(inp.shape[0], 1, Ry_up, Rx_up, dtype=inp.dtype, device=inp.device)
            out[:, :, ::N, ::N] = inp
        elif self.upscale_method == "nearest":
            out = F.interpolate(inp, size=self.upscaled_shape, mode="nearest")
        else:
            out = F.interpolate(
                inp,
                size=self.upscaled_shape,
                mode="bilinear",
                align_corners=False,
            )
        out = out.squeeze(1).contiguous()
        if is_tensor:
            return out
        return out.cpu().numpy().astype(np.float32, copy=False)

    def make_ones_fft(self, device: str) -> torch.Tensor:
        """Return fft2 of a prepared ones vBF used to build the zero_insert reweighting map.

        Runs a single all-ones image through the same _prepare pipeline (zero-insertion,
        reflect-padding, Tukey windowing) so the resulting FFT accounts for all spatial
        modulations. Returns shape (1, padded_Ry, padded_Rx) complex64 on `device`.
        """
        ones = np.ones((1, *self.raw_shape), dtype=np.float32)
        prepared = self._prepare(ones)  # (1, padded_Ry, padded_Rx)
        t = torch.from_numpy(np.asarray(prepared, dtype=np.float32)).to(device)
        return torch.fft.fft2(t)

    def _pad_and_window(self, vbf):
        """vBF shape: (B, Ry, Rx). Returns same array/tensor type."""
        if isinstance(vbf, torch.Tensor):
            padded = F.pad(
                vbf,
                (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
                mode="reflect",
            )
            w = torch.as_tensor(self._window_np, dtype=torch.float32, device=vbf.device)
            return padded * w

        padded = np.pad(
            vbf,
            ((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)),
            mode="reflect",
        ).astype(np.float32)
        return padded * self._window_np


__all__ = [
    "BFPreparer",
    "_check_zero_insert_compat",
    "_compute_pad_for_axis",
    "_prepared_shapes",
]
