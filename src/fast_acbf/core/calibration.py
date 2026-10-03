"""Turn microscope metadata and a PACBED into solver inputs (wavelength, BF disk radius)."""

from __future__ import annotations

import numpy as np


# ---- Ported from PtyRAD (ptyrad.optics.constants, v1.0.0); keep in sync ---------------
# Identical to PtyRAD's code (tests/test_ptyrad_port.py checks when PtyRAD is installed).

# Physical Constants
PLANCKS = 6.62607015E-34 # m^2*kg / s
REST_MASS_E = 9.1093837015E-31 # kg
CHARGE_E = 1.602176634E-19 # coulomb
SPEED_OF_LIGHT = 299792458 # m/s

# Useful constants in EM unit
HC = PLANCKS * SPEED_OF_LIGHT / CHARGE_E*1E-3*1E10 # 12.398 keV-Ang, h*c
REST_ENERGY_E = REST_MASS_E*SPEED_OF_LIGHT**2/CHARGE_E*1E-3 # 511 keV, m0c^2

def get_wavelength_ang(kv):
    """Calculates the relativistic electron wavelength (Ang).

    The wavelength is calculated using the relativistic de Broglie relationship:
    lambda = h * c / sqrt((2 * m0 * c^2 + e * V) * e * V)

    Args:
        kv (float): The acceleration voltage in kilovolts (kV).

    Returns:
        float: The electron wavelength in Angstroms.
    """
    wavelength = HC/np.sqrt((2*REST_ENERGY_E + kv)*kv) # Angstrom, lambda = hc/sqrt((2*m0c^2 + e*V)*e*V))
    return wavelength

# ---- end of the PtyRAD port ------------------------------------------------------------


# ---- Ported from PtyRAD (ptyrad.utils.image_proc, v1.0.0); keep in sync ---------------
# Identical to PtyRAD's function (tests/test_ptyrad_port.py checks when PtyRAD is installed).

def guess_radius_of_bright_field_disk(image: np.ndarray, thresh: float=0.5):
    """ Utility function that returns an estimate of the radius of rbf from CBED """
    # meas: 2D array of (ky,kx)
    # thresh: 0.5 for FWHM, 0.1 for Full-width at 10th maximum
    max_val = np.max(image)
    binary_img = image > (max_val * thresh)
    area = np.sum(binary_img)
    rbf = np.sqrt(area / np.pi) # Assume the region is circular
    return rbf

# ---- end of the PtyRAD port ------------------------------------------------------------
