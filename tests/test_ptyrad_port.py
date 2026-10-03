"""The code ported from PtyRAD must stay in sync with PtyRAD.

fast-acbf does not depend on PtyRAD, but ``Aberrations`` (``fast_acbf.core.ptyrad_aberrations``),
``fftshift2`` / ``ifftshift2`` / ``torch_phasor`` (``fast_acbf.core.functional``), ``mfft2``
(``fast_acbf.vis.plotting``), and ``get_wavelength_ang`` / ``guess_radius_of_bright_field_disk``
with the physical constants (``fast_acbf.core.calibration``) are copies of PtyRAD's. These
tests compare them with the installed PtyRAD and are skipped when PtyRAD is not installed.
A failure means PtyRAD changed: copy its version over (see the module docstring of
``ptyrad_aberrations``).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

ptyrad_aberrations = pytest.importorskip("ptyrad.optics.aberrations")
ptyrad_constants = pytest.importorskip("ptyrad.optics.constants")
ptyrad_functional = pytest.importorskip("ptyrad.core.functional")
ptyrad_image_proc = pytest.importorskip("ptyrad.utils.image_proc")

from fast_acbf.core import calibration, functional, ptyrad_aberrations as port  # noqa: E402
from fast_acbf.vis import plotting  # noqa: E402

MARKER = "# ---- verbatim from PtyRAD ----\n"
NOTE = (
    "\n"
    "    Ported from PtyRAD (``ptyrad.optics.aberrations.Aberrations``); keep it in sync\n"
    "    with PtyRAD as much as possible. See the module docstring before changing anything.\n"
)


def _text(path) -> str:
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def test_aberrations_module_is_verbatim_copy():
    ours = _text(port.__file__)
    assert ours.count(MARKER) == 1 and ours.count(NOTE) == 1
    ours = ours.split(MARKER, 1)[1].replace(NOTE, "", 1)
    theirs = "".join(_text(ptyrad_aberrations.__file__).splitlines(keepends=True)[4:])
    assert ours == theirs, "ptyrad.optics.aberrations changed: re-port it (see its docstring)"


@pytest.mark.parametrize("ours, theirs", [
    (functional.fftshift2, ptyrad_functional.fftshift2),
    (functional.ifftshift2, ptyrad_functional.ifftshift2),
    (functional.torch_phasor, ptyrad_functional.torch_phasor),
    (plotting.mfft2, ptyrad_image_proc.mfft2),
    (calibration.get_wavelength_ang, ptyrad_constants.get_wavelength_ang),
    (calibration.guess_radius_of_bright_field_disk, ptyrad_image_proc.guess_radius_of_bright_field_disk),
], ids=["fftshift2", "ifftshift2", "torch_phasor", "mfft2", "get_wavelength_ang",
        "guess_radius_of_bright_field_disk"])
def test_functions_are_verbatim_copies(ours, theirs):
    assert inspect.getsource(ours) == inspect.getsource(theirs)


@pytest.mark.parametrize("name", ["PLANCKS", "REST_MASS_E", "CHARGE_E", "SPEED_OF_LIGHT", "HC",
                                  "REST_ENERGY_E"])
def test_constants_match(name):
    assert getattr(calibration, name) == getattr(ptyrad_constants, name)


AB = {"C10": -52.0, "C12": 7.5, "phi12": 30.0, "C21": 120.0, "phi21": -45.0, "C30": 1.2e4}


@pytest.mark.parametrize("notation", ["krivanek", "haider"])
@pytest.mark.parametrize("style", ["polar", "cartesian", "complex"])
@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_aberrations_export_matches(notation, style, layout):
    kw = dict(notation=notation, style=style, layout=layout)
    assert port.Aberrations(AB).export(**kw) == ptyrad_aberrations.Aberrations(AB).export(**kw)


def test_aberrations_aliases_and_str_match():
    data = {"defocus": 40.0, "Cs": 2.0e4}
    ours, theirs = port.Aberrations(data), ptyrad_aberrations.Aberrations(data)
    assert ours["C10"] == theirs["C10"] == -40.0
    assert str(ours) == str(theirs)


def test_functions_behave_the_same():
    x = torch.randn(3, 6, 7)
    assert torch.equal(functional.fftshift2(x), ptyrad_functional.fftshift2(x))
    assert torch.equal(functional.ifftshift2(x), ptyrad_functional.ifftshift2(x))
    assert torch.equal(functional.torch_phasor(x), ptyrad_functional.torch_phasor(x))
    im = np.random.default_rng(0).random((12, 10))
    for a, b in zip(plotting.mfft2(im), ptyrad_image_proc.mfft2(im), strict=True):
        np.testing.assert_array_equal(a, b)
    for kv in (60, 80, 200, 300):
        assert calibration.get_wavelength_ang(kv) == ptyrad_constants.get_wavelength_ang(kv)
    dp = np.random.default_rng(1).random((32, 32))
    assert (calibration.guess_radius_of_bright_field_disk(dp, thresh=0.3)
            == ptyrad_image_proc.guess_radius_of_bright_field_disk(dp, thresh=0.3))
