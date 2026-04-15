"""Torch-native bright-field solver with tcBF, and acBF reconstruction."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur, rotate as tv_rotate
from torchvision.transforms import InterpolationMode

from ptyrad.core.functional import fftshift2, ifftshift2, torch_phasor
from ptyrad.optics.aberrations import Aberrations

logger = logging.getLogger(__name__)

def imshift_with_batch(imgs, shifts, grid, batch_size=None):
    """
    Generates a batch of shifted images from an input stack of images.
    
    This function shifts complex/real-valued input images by applying phase shifts 
    in the Fourier domain, achieving subpixel shifts in both x and y directions.
    Optionally applies a CTF correction in the same operation.

    Args:
        imgs (torch.Tensor): The input images to be shifted. 
                             Shape must be (Nb, ..., Ny, Nx).
        shifts (torch.Tensor): The shifts to be applied. Shape is (Nb, 2), 
                               where each slice is (shift_y, shift_x) in pixels.
        grid (torch.Tensor): The k-space grid used for computing the shifts. 
                             Shape=(2, Ny, Nx), spanning [-0.5, 0.5).
        batch_size (int, optional): The number of images to process at once to avoid OOM.
                                    If None, processes the entire batch simultaneously.

    Returns:
        torch.Tensor: The shifted (and optionally CTF-corrected) images. 
                      Shape remains (Nb, ..., Ny, Nx).
    """
    assert imgs.shape[-2:] == grid.shape[-2:], \
        f"Incompatible dimensions: imgs {imgs.shape[-2:]} vs grid {grid.shape[-2:]}"
    assert imgs.shape[0] == shifts.shape[0], \
        f"Batch size mismatch: imgs has {imgs.shape[0]}, shifts has {shifts.shape[0]}"
    
    Nb = imgs.shape[0]
    ndim = imgs.ndim
    
    # Default to processing all at once if no batch_size is provided
    batch_size = batch_size or Nb

    # Pre-calculate broadcast shapes for .view() to cleanly expand dimensions
    # shifts go from (B,) -> (B, 1, 1, ...)
    shift_expand_shape = [-1] + [1] * (ndim - 1)
    # grid goes from (Ny, Nx) -> (1, 1, ..., Ny, Nx)
    grid_expand_shape = [1] * (ndim - 2) + list(grid.shape[-2:])
    
    ky = grid[0].view(*grid_expand_shape)
    kx = grid[1].view(*grid_expand_shape)

    shifted_imgs_list = []

    # Process in chunks to prevent OOM
    for i in range(0, Nb, batch_size):
        end = min(i + batch_size, Nb)
        
        imgs_b = imgs[i:end]
        shifts_b = shifts[i:end]
        
        shift_y = shifts_b[:, 0].view(*shift_expand_shape)
        shift_x = shifts_b[:, 1].view(*shift_expand_shape)
        
        # Calculate shifting phase kernel
        phase = -2 * torch.pi * (shift_x * kx + shift_y * ky)
        w = torch_phasor(phase)
            
        # Shift in Fourier domain
        img_fft = torch.fft.fft2(imgs_b)
        shifted_img_b = torch.fft.ifft2(img_fft * w)
        
        shifted_imgs_list.append(shifted_img_b)

    # Recombine the processed chunks
    return torch.cat(shifted_imgs_list, dim=0)

def make_soft_aperture_torch(alpha: torch.Tensor, max_alpha_mrad: float, rolloff_mrad: float) -> torch.Tensor:
    """
    Creates a differentiable circular aperture mask with a cosine edge taper.
    
    Args:
        alpha: Tensor of scattering angles in radians.
        max_alpha_mrad: The semiangle cutoff in milliradians.
        rolloff_mrad: The width of the soft edge transition in milliradians.
    """
    cutoff = max_alpha_mrad / 1000.0
    
    if rolloff_mrad <= 0.0:
        return (alpha <= cutoff).to(torch.float32)
        
    rolloff = rolloff_mrad / 1000.0
    
    # Calculate the smooth cosine transition curve
    transition = 0.5 * (1.0 + torch.cos(torch.pi * (alpha - cutoff + rolloff) / rolloff))
    
    # 1. Mask regions outside the cutoff to exactly 0.0
    mask = torch.where(alpha > cutoff, torch.zeros_like(alpha), transition)
    
    # 2. Mask regions completely inside the flat top to exactly 1.0
    mask = torch.where(alpha < (cutoff - rolloff), torch.ones_like(alpha), mask)
    
    return mask

def make_probe_from_chi(chi, mask):
    
    assert chi.shape[-2:] == mask.shape[-2:]
        
    # Make probe and normalize
    psi = torch_phasor(-1*chi)
    probe = mask*psi # It's now the masked wave function at the aperture plane
    probe = fftshift2(torch.fft.ifft2(ifftshift2(probe))) # Propagate the wave function from aperture to the sample plane. 
    probe = probe/torch.sqrt(torch.sum((torch.abs(probe))**2)) # Normalize the probe so sum(abs(probe)^2) = 1
    return probe

class QualityMetrics:
    """
    Output quality metrics (Laplacian, Sobel, Normalized Image Variance) in maximize direction
    """

    @classmethod
    def evaluate(cls, img: torch.Tensor, metric='laplacian', blur=False, blur_kernel_size=5, blur_sigma=1) -> float:
        """
        Calculates focus score(s). Higher is more focused.

        Accepts either:
            - 2D image of shape (Ny, Nx), returning a scalar tensor
            - 3D stack of shape (Nz, Ny, Nx), returning a 1D tensor of length Nz
        """
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim != 3:
            raise ValueError(f"img must be 2D or 3D, got shape {tuple(img.shape)}.")

        img = img.unsqueeze(1)  # Convert to (N, 1, H, W) for gaussian_blur and conv2d
        
        if blur:
            img = gaussian_blur(img, kernel_size=blur_kernel_size, sigma=blur_sigma)
            
        metric = metric.lower()
        if metric == 'laplacian':
            # Variance of Laplacian
            lap_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=img.device)
            lap = F.conv2d(img, lap_kernel, padding=1)
            scores = lap.flatten(start_dim=1).var(dim=1)
            
        elif metric == 'sobel':
            # Tenengrad (Sum of squared Sobel gradients)
            kx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], device=img.device)
            ky = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], device=img.device)
            gx = F.conv2d(img, kx, padding=1)
            gy = F.conv2d(img, ky, padding=1)
            scores = (gx**2 + gy**2).flatten(start_dim=1).mean(dim=1)
            
        elif metric == 'normalized_std':
            # Normalized Image Std
            flat = img.flatten(start_dim=1)
            scores = flat.std(dim=1) / (flat.mean(dim=1) + 1e-6)
            
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose 'laplacian', 'sobel', or 'normalized_std'.")

        if scores.shape[0] == 1:
            return scores.squeeze(0)
        return scores

class AberrationState(torch.nn.Module):
    def __init__(self, ab_dict: dict, max_order: int = None, device='cpu'):
        super().__init__()
        self.orig_ab_dict = ab_dict.copy()
        self.device = torch.device(device)
        self.order_keys = []
        
        # If max_order is not provided, infer it from the highest 'n' in the input dict
        if max_order is None:
            max_order = max([n for (n, m) in ab_dict.keys()]) if ab_dict else 2
            
        self.max_order = max_order
        
        # Flat ParameterDict for optimizer access
        self.coeffs = nn.ParameterDict()
        
        # Dynamically generate all valid (n, m) pairs up to max_order in ascending m
        for n in range(1, max_order + 1):
            # m starts at 0 if n is odd, 1 if n is even. Steps by 2 up to n+1.
            start_m = (n + 1) % 2
            
            for m in range(start_m, n + 2, 2):
                self.order_keys.append((n, m))
                
                # Fetch user value if provided, otherwise default to 0.0
                if m == 0:
                    val = ab_dict.get((n, m), 0.0)
                    val_clean = float(val) if isinstance(val, (int, float, np.number)) else 0.0
                    
                    key = f"C_{n}_{m}"
                    self.coeffs[key] = nn.Parameter(torch.tensor(val_clean, device=self.device))
                    
                else:
                    val = ab_dict.get((n, m), {'a': 0.0, 'b': 0.0})
                    if isinstance(val, (int, float, np.number)):
                        val_a, val_b = float(val), 0.0
                    else:
                        val_a = float(val.get('a', 0.0))
                        val_b = float(val.get('b', 0.0))
                        
                    key_a = f"C_{n}_{m}_a"
                    key_b = f"C_{n}_{m}_b"
                    self.coeffs[key_a] = nn.Parameter(torch.tensor(val_a, device=self.device))
                    self.coeffs[key_b] = nn.Parameter(torch.tensor(val_b, device=self.device))

    def get_cartesian_dict(self):
        """Reconstructs the nested dictionary format."""
        out = {}
        for (n, m) in self.order_keys:
            if m == 0:
                out[(n, m)] = self.coeffs[f"C_{n}_{m}"].item()
            else:
                out[(n, m)] = {
                    'a': self.coeffs[f"C_{n}_{m}_a"].item(),
                    'b': self.coeffs[f"C_{n}_{m}_b"].item()
                }
        return out
    
    def get_flat_coeffs(self):
        """Returns all coefficients as a 1D tensor in a deterministic order."""
        coeffs_list = []
        for (n, m) in self.order_keys:
            if m == 0:
                coeffs_list.append(self.coeffs[f"C_{n}_{m}"])
            else:
                coeffs_list.append(self.coeffs[f"C_{n}_{m}_a"])
                coeffs_list.append(self.coeffs[f"C_{n}_{m}_b"])
        return torch.stack(coeffs_list)

# tcBF

def generate_shift_basis(order_keys: list, kX: torch.Tensor, kY: torch.Tensor, wavelength: float):
    """Generates the unweighted (C=1) analytic shift basis vectors for dx and dy."""
    alphaX = kX * wavelength
    alphaY = kY * wavelength
    alpha_sq = alphaX**2 + alphaY**2
    
    # Tiny epsilon to prevent 0^(negative) resulting in NaN
    alpha_sq_safe = alpha_sq + 1e-12 
    
    max_m = max([m for (n, m) in order_keys]) if order_keys else 0
    
    # Precompute Angular Polynomials
    X, Y = {}, {}
    X[0] = torch.ones_like(alphaX)
    Y[0] = torch.zeros_like(alphaX)
    
    for m in range(max_m):
        X[m+1] = X[m] * alphaX - Y[m] * alphaY
        Y[m+1] = X[m] * alphaY + Y[m] * alphaX
        
    basis_dx_list = []
    basis_dy_list = []
    
    for (n, m) in order_keys:
        p = (n + 1 - m) / 2.0
        
        # Derivative of Radial Component
        if n + 1 - m > 0:
            rad_deriv_x = (n + 1 - m) * alphaX * (alpha_sq_safe ** (p - 1.0))
            rad_deriv_y = (n + 1 - m) * alphaY * (alpha_sq_safe ** (p - 1.0))
        else:
            rad_deriv_x = torch.zeros_like(alphaX)
            rad_deriv_y = torch.zeros_like(alphaX)
            
        rad_base = alpha_sq ** p
        
        if m == 0:
            # Round aberration (Only C_a)
            dx = (rad_deriv_x * X[m]) / (n + 1)
            dy = (rad_deriv_y * X[m]) / (n + 1)
            basis_dx_list.append(dx)
            basis_dy_list.append(dy)
            
        else:
            # Component A (C_a = 1, C_b = 0)
            part1_x_a = rad_deriv_x * X[m]
            part1_y_a = rad_deriv_y * X[m]
            part2_x_a = rad_base * (m * X[m-1])
            part2_y_a = rad_base * (m * -Y[m-1])
            
            basis_dx_list.append((part1_x_a + part2_x_a) / (n + 1))
            basis_dy_list.append((part1_y_a + part2_y_a) / (n + 1))
            
            # Component B (C_a = 0, C_b = 1)
            part1_x_b = rad_deriv_x * Y[m]
            part1_y_b = rad_deriv_y * Y[m]
            part2_x_b = rad_base * (m * Y[m-1])
            part2_y_b = rad_base * (m * X[m-1])
            
            basis_dx_list.append((part1_x_b + part2_x_b) / (n + 1))
            basis_dy_list.append((part1_y_b + part2_y_b) / (n + 1))
            
    # Stack into shape: [Num_Coeffs, Batch_Size]
    return torch.stack(basis_dx_list, dim=0), torch.stack(basis_dy_list, dim=0)

# acBF

def generate_aberration_basis(max_order: int, order_keys: list, kX: torch.Tensor, kY: torch.Tensor, wavelength: float):
    """Generates the unweighted (C=1) Cartesian polynomial basis tensors."""
    alphaX = kX * wavelength
    alphaY = kY * wavelength
    alpha_sq = alphaX**2 + alphaY**2
    
    X, Y = {}, {}
    X[0] = torch.ones_like(alpha_sq)
    Y[0] = torch.zeros_like(alpha_sq)
    
    for m in range(max_order+1):
        X[m+1] = X[m] * alphaX - Y[m] * alphaY
        Y[m+1] = X[m] * alphaY + Y[m] * alphaX
        
    multiplier = (2 * torch.pi / wavelength)
    basis_list = []
    
    # Iterate using the exact same order_keys from the AberrationState
    for (n, m) in order_keys:
        power_rad = (n + 1 - m) / 2.0
        term_radial = alpha_sq ** power_rad
        
        if m == 0:
            basis_list.append(term_radial * X[m] / (n + 1) * multiplier)
        else:
            basis_list.append(term_radial * X[m] / (n + 1) * multiplier) # a
            basis_list.append(term_radial * Y[m] / (n + 1) * multiplier) # b
            
    # Stack into a single tensor: Shape (Num_Coeffs, Nb, Nx, Ny)
    return torch.stack(basis_list, dim=0)

class BFSolver:
    def __init__(
        self,
        dataset: np.ndarray,
        max_alpha: float,
        scan_step_size: float,
        dk: float,
        wavelength: float,
        max_order: int,
        aberrations: dict,
        device='cuda',
        coord_transform=None,
        output_frame: str = 'scan',
        eps: float = 1e-3,
        cache_mode: str = 'full_gpu',
        upscale_method: str = 'real',
        defer_upscale: bool = False,
    ):
        """
        Initializes the solver. Dataset loading/parsing is assumed to be handled
        upstream (e.g., by PtyRAD's Initializer).

        Args:
            cache_mode: Controls how the static acBF/tcBF cache is stored.
                'full_gpu'  — All precomputed tensors (bases, apertures, img_fft) on GPU.
                              Maximum reconstruction speed; highest VRAM cost (~7 GB typical).
                'fft_gpu'   — Only img_fft cached on GPU; aberration bases recomputed on
                              every reconstruction call. VRAM drops ~6 GB; adds ~50–100 ms
                              per reconstruction.  Suitable for large datasets or high upscale.
                'full_cpu'  — Full cache stored in CPU pinned RAM; chunks transferred to GPU
                              one at a time during reconstruction (with async prefetch).
                              Minimal GPU VRAM; slower than 'fft_gpu' for repeated calls.
            upscale_method: Controls the upsampling strategy when upscale > 1.
                'real'    — repeat_interleave in real space, then fft2.  Integer upscale
                            factors only.  Same behaviour as before this parameter existed.
                'fourier' — fft2 at native resolution, then zero-pad in Fourier space to the
                            target size.  Supports arbitrary (non-integer) upscale factors;
                            avoids the large intermediate spatial tensor.
            defer_upscale: When True, img_fft and all cached grids/apertures are stored at
                native scan resolution (Ry, Rx).  Upscaling to (Ny_out, Nx_out) is deferred
                to a single Fourier zero-pad of the accumulated F_corr sum, immediately before
                the final IFFT.  This reduces cache memory by upscale^2 and replaces per-chunk
                IFFTs with one IFFT per reconstruction.  The result is mathematically identical
                to upscaling upfront because all operations in the reconstruction loop are linear
                and commute with zero-padding.  When upscale=1 this flag has no effect.
        """
        self.dataset = dataset
        self.max_alpha = max_alpha
        self.scan_step_size = scan_step_size
        self.dk = dk
        self.wavelength = wavelength
        self.max_order = max_order
        self.orig_aberrations = aberrations
        self.parsed_aberrations = Aberrations(aberrations).export(notation='krivanek', style='cartesian', layout='nested')
        self.ab_state = AberrationState(self.parsed_aberrations, self.max_order, device=device)
        self.eps = eps
        self.device = device

        _VALID_CACHE_MODES = ('full_gpu', 'fft_gpu', 'full_cpu')
        cache_mode = str(cache_mode).strip().lower()
        if cache_mode not in _VALID_CACHE_MODES:
            raise ValueError(
                f"cache_mode must be one of {_VALID_CACHE_MODES}, got {cache_mode!r}."
            )
        self.cache_mode = cache_mode

        _VALID_UPSCALE_METHODS = ('real', 'fourier')
        upscale_method = str(upscale_method).strip().lower()
        if upscale_method not in _VALID_UPSCALE_METHODS:
            raise ValueError(
                f"upscale_method must be one of {_VALID_UPSCALE_METHODS}, got {upscale_method!r}."
            )
        self.upscale_method = upscale_method
        self.defer_upscale = bool(defer_upscale)

        # Coordinate transform — maps acBF k-space orientation to the PtyRAD pipeline.
        #
        # flipud / fliplr / transpose:
        #   Correct discrete 90°-class detector orientation differences.
        #   These map 1-to-1 to PtyRAD's `meas_flipT = [flipud, fliplr, transpose]`.
        #   The same flag values should be used in both tools.
        #   Operations are applied in the same order as PtyRAD's `_meas_flipT`:
        #   flipud first, fliplr second, transpose last.
        #   Note that we flip the ky/kx coordinate, instead of the diffraction pattern
        #   for performance.
        #
        # rotation_deg:
        #   Corrects for a continuous scan rotation angle (the angle between the scan
        #   fast-axis and the detector kX axis). Positive = CCW rotation of k-vectors.
        #   Maps to PtyRAD's `pos_scan_affine = [1, 0, rotation_deg, 0]` (same value,
        #   same sign).
        self.coord_transform = coord_transform or {
            'flipud': False,
            'fliplr': False,
            'transpose': False,
            'rotation_deg': 0.0,
        }

        # Output frame selector — controls which k-space frame public outputs are
        # expressed in.  Two frames are defined:
        #
        #   'scan' (default):
        #     All coord_transform operations applied: flipud → fliplr → transpose →
        #     rotation_deg.  Outputs are expressed in the scan frame — the k-frame
        #     aligned with the scan fast-axis — which is the frame the aberrations
        #     were fitted in.
        #
        #   'detector':
        #     Only the discrete flips applied: flipud → fliplr → transpose.
        #     rotation_deg is excluded.  Outputs are expressed in the detector frame
        #     — the k-frame defined by the (flip-corrected) detector pixel grid.
        #     This frame is what PtyRAD uses internally: meas_flipT brings data into
        #     it, and probe_aberrations / chi are evaluated in it.  Use this frame
        #     when exporting aberration coefficients or chi/probe for PtyRAD.
        #
        # Individual methods also accept a per-call `output_frame` keyword that
        # overrides this global default (None → use global; 'scan'/'detector' →
        # override for that call only).
        self.output_frame = output_frame.lower()
        
        # Placeholders / caches
        self.reconstructed_image = None
        self._reconstructed_images = {}
        self._cache_store = {}
        self.last_c10_stack_axis = None

        # Initialize vBF dataset
        self._init_vBF()
        self._init_grid()

    # Internal Methods

    @staticmethod
    def _normalize_acbf_algorithm(acbf_algorithm):
        """Normalize the public acBF algorithm selector."""
        if acbf_algorithm is None:
            return 'phase_only'
        return str(acbf_algorithm).strip().lower().replace('-', '_')

    def clear_cache(self, clear_static_cache=True):
        """Clears reconstructed image cache and, optionally, the static tcBF/acBF caches."""
        self.reconstructed_image = None
        self._reconstructed_images = {}

        if clear_static_cache:
            self._cache_store = {}

    def _get_transform_flags(self):
        """
        Returns the ky/kx transformation flags.
        """
        flipud = self.coord_transform.get('flipud', False)
        fliplr = self.coord_transform.get('fliplr', False)
        transpose = self.coord_transform.get('transpose', False)
        rotation_deg = self.coord_transform.get('rotation_deg', 0.0)
               
        return flipud, fliplr, transpose, rotation_deg

    def _resolve_output_frame(self, output_frame):
        """
        Return the effective output frame string.
        None → defers to self.output_frame (global default).
        'scan' or 'detector' → explicit per-call override.
        """
        return self.output_frame if output_frame is None else output_frame.lower()

    def _in_scan_frame(self, output_frame):
        """Return True when output_frame resolves to 'scan'."""
        return self._resolve_output_frame(output_frame) == 'scan'

    def _get_detector_frame_cartesian_dict(self):
        """
        Return aberration coefficients converted from the scan frame back to the
        detector frame.

        The scan frame is the detector frame further rotated CCW by rotation_deg.
        Inverting that rotation (rotating by +rotation_deg) recovers the detector-frame
        representation.  For symmetric terms (m=0, e.g. C10, C30) the value is
        invariant.  For asymmetric terms (m>0) the Cartesian (a, b) pair transforms as:
            Ca_det = Ca_scan * cos(m·θ) - Cb_scan * sin(m·θ)
            Cb_det = Ca_scan * sin(m·θ) + Cb_scan * cos(m·θ)
        where θ = rotation_deg in radians.
        """
        theta = np.deg2rad(self.coord_transform.get('rotation_deg', 0.0))
        ab_dict = self.ab_state.get_cartesian_dict()
        if theta == 0:
            return ab_dict
        out = {}
        for (n, m), val in ab_dict.items():
            if m == 0:
                out[(n, m)] = val
            else:
                ca, cb = val['a'], val['b']
                c, s = np.cos(m * theta), np.sin(m * theta)
                out[(n, m)] = {'a': ca * c - cb * s, 'b': ca * s + cb * c}
        return out

    def _get_effective_ab_state(self, in_scan_frame):
        """
        Return the AberrationState appropriate for the requested output frame.

        in_scan_frame=True  → self.ab_state: coefficients as fitted in the scan frame.
        in_scan_frame=False → temporary AberrationState with coefficients converted to
                              the detector frame (rotation_deg un-applied).
        """
        if in_scan_frame or self.coord_transform.get('rotation_deg', 0.0) == 0:
            return self.ab_state
        return AberrationState(
            self._get_detector_frame_cartesian_dict(),
            max_order=self.ab_state.max_order,
            device=str(self.device),
        )

    def _get_transformed_bf_coordinates(self, in_scan_frame=True):
        """
        Return transformed reciprocal-space BF coordinates as (kX, kY).

        Operations are applied in the order: flipud → fliplr → transpose → rotation_deg.
        The first three bring raw detector coordinates into the detector frame (matching
        PtyRAD's `_meas_flipT` order).  The last step further rotates into the scan frame.

        in_scan_frame=True  → all four steps applied; coordinates in the scan frame.
        in_scan_frame=False → rotation_deg skipped; coordinates in the detector frame.
        """
        ky = self.kY_centers.clone()
        kx = self.kX_centers.clone()
        flipud, fliplr, transpose, rotation_deg = self._get_transform_flags()

        if flipud:
            ky = -ky
        if fliplr:
            kx = -kx
        if transpose:
            ky, kx = kx, ky
        if in_scan_frame and rotation_deg:
            theta = np.deg2rad(rotation_deg)
            kx_old = kx.clone()
            ky_old = ky.clone()
            kx = kx_old * np.cos(theta) - ky_old * np.sin(theta)
            ky = kx_old * np.sin(theta) + ky_old * np.cos(theta)

        return kx, ky

    def _get_transformed_k_grids(self, in_scan_frame=True):
        """
        Return transformed reciprocal-space full grids as (kX_grid, kY_grid).

        Same operation order as `_get_transformed_bf_coordinates`:
        flipud → fliplr → transpose → rotation_deg.

        in_scan_frame=True  → all four steps applied; grids in the scan frame.
        in_scan_frame=False → rotation_deg skipped; grids in the detector frame.
        """
        ky = self.kY_grid.clone()
        kx = self.kX_grid.clone()
        flipud, fliplr, transpose, rotation_deg = self._get_transform_flags()

        if flipud:
            ky = -ky
        if fliplr:
            kx = -kx
        if transpose:
            ky, kx = kx, ky
        if in_scan_frame and rotation_deg:
            theta = np.deg2rad(rotation_deg)
            kx_old = kx.clone()
            ky_old = ky.clone()
            kx = kx_old * np.cos(theta) - ky_old * np.sin(theta)
            ky = kx_old * np.sin(theta) + ky_old * np.cos(theta)

        return kx, ky

    def _init_vBF(self):
        """
        Initializes the Bright Field disk coordinates and extracts the vBF images
        directly from the NumPy dataset to minimize memory overhead.
        """

        Ry_dim, Rx_dim, Ky_dim, Kx_dim = self.dataset.shape

        # 1. Create centered k-space grid in NumPy
        ky = np.fft.fftshift(np.fft.fftfreq(Ky_dim, d=(1/self.dk/Ky_dim))) # k is now in unit of Ang-1
        kx = np.fft.fftshift(np.fft.fftfreq(Kx_dim, d=(1/self.dk/Kx_dim)))
        kX_grid, kY_grid = np.meshgrid(kx, ky, indexing='xy')
        
        # 2. Create BF mask
        kR_grid = np.sqrt(kX_grid**2 + kY_grid**2)
        bf_mask = kR_grid <= (self.max_alpha/1e3/self.wavelength)  # Shape: (Ky, Kx)

        # 3. Extract the original k-space coordinates for the masked pixels
        kY_centers_np = kY_grid[bf_mask]  # Shape: (Nb,)
        kX_centers_np = kX_grid[bf_mask]  # Shape: (Nb,)

        # 4. Apply mask via advanced indexing to avoid permuting the 4D array
        # dataset[:, :, bf_mask] naturally slices out the matching Ky, Kx pixels, 
        # resulting in a shape of (Ry, Rx, Nb)
        vbf_np = self.dataset[:, :, bf_mask]

        # 5. Permute to (Nb, Ry, Rx) and ensure memory contiguity for the small batch
        # Using moveaxis is safe and clean here
        vbf_np = np.ascontiguousarray(np.moveaxis(vbf_np, -1, 0))

        # 6. Transfer to PyTorch tensors and assign to device
        self.kY_centers = torch.tensor(kY_centers_np, dtype=torch.float32, device=self.device)
        self.kX_centers = torch.tensor(kX_centers_np, dtype=torch.float32, device=self.device)
        
        # Keep track of coordinates as requested
        self.bf_coordinates = torch.stack([self.kY_centers, self.kX_centers], dim=-1)
        
        self.kX_grid = torch.tensor(kX_grid, dtype=torch.float32, device=self.device)
        self.kY_grid = torch.tensor(kY_grid, dtype=torch.float32, device=self.device)
        self.bf_mask = torch.tensor(bf_mask, dtype=torch.float32, device=self.device)
        self.vbf_images = torch.tensor(vbf_np, dtype=torch.float32, device=self.device)
        self.Ry = Ry_dim
        self.Rx = Rx_dim

        print(f"Extracted {self.vbf_images.shape[0]} vBF images within the max alpha angle = {self.max_alpha} mrad.")

    def _init_grid(self):
        kpy, kpx = torch.meshgrid(torch.fft.fftfreq(self.Ry, dtype=torch.float32, device=self.device),
                                  torch.fft.fftfreq(self.Rx, dtype=torch.float32, device=self.device), indexing='ij')
        self.shift_grid = torch.stack([kpy, kpx], dim=0) 

    @staticmethod
    def _zero_pad_fft2(fft_native: torch.Tensor, Ny_out: int, Nx_out: int) -> torch.Tensor:
        """
        Zero-pad an fft2 output from its native resolution to (Ny_out, Nx_out).

        Equivalent to: ifft2 → repeat_interleave (integer factor) → fft2, but avoids
        allocating the large intermediate spatial-domain tensor.  Works for arbitrary
        (non-integer) scale factors since only the target dimensions matter.

        The output is scaled so that ifft2 of the result yields correctly normalised
        pixel values matching what you would get from the spatially-upsampled image.

        Args:
            fft_native: Complex tensor (..., Ny_in, Nx_in) in standard (un-shifted)
                        fft2 layout as returned by torch.fft.fft2.
            Ny_out, Nx_out: Target dimensions; must be >= Ny_in and Nx_in respectively.

        Returns:
            Complex tensor (..., Ny_out, Nx_out).
        """
        *_, Ny_in, Nx_in = fft_native.shape
        if Ny_out == Ny_in and Nx_out == Nx_in:
            return fft_native
        # Shift DC to centre for symmetric padding
        shifted = torch.fft.fftshift(fft_native, dim=(-2, -1))
        pad_y_top   = (Ny_out - Ny_in) // 2
        pad_y_bot   = Ny_out - Ny_in - pad_y_top
        pad_x_left  = (Nx_out - Nx_in) // 2
        pad_x_right = Nx_out - Nx_in - pad_x_left
        padded = F.pad(shifted, (pad_x_left, pad_x_right, pad_y_top, pad_y_bot))
        out = torch.fft.ifftshift(padded, dim=(-2, -1))
        # Compensate for the larger IFFT normalisation denominator
        return out * (Ny_out * Nx_out) / (Ny_in * Nx_in)

    def _compute_upscaled_img_fft(
        self,
        img_chunk_native: torch.Tensor,
        upscale: float,
        Ny_out: int,
        Nx_out: int,
    ) -> torch.Tensor:
        """
        Return img_fft at the upscaled output resolution.

        Delegates to the method selected by self.upscale_method:
            'real'    — repeat_interleave in real space → fft2  (integer upscale only)
            'fourier' — fft2 at native resolution → zero-pad to (Ny_out, Nx_out)
        """
        if self.upscale_method == 'real':
            up = int(upscale)
            if up != upscale:
                raise ValueError(
                    f"upscale_method='real' requires an integer upscale factor, got {upscale}. "
                    "Use upscale_method='fourier' for non-integer factors."
                )
            if up > 1:
                img_up = img_chunk_native.repeat_interleave(up, dim=-2).repeat_interleave(up, dim=-1)
            else:
                img_up = img_chunk_native
            return torch.fft.fft2(img_up, dim=(-2, -1))
        else:  # 'fourier'
            img_fft_native = torch.fft.fft2(img_chunk_native, dim=(-2, -1))
            return self._zero_pad_fft2(img_fft_native, Ny_out, Nx_out)

    def _build_tcBF_cache(self, upscale=1, chunk_size=64):
        """Pre-computes and chunks the analytical shift basis, spatial grids, and FFTs."""

        # 1. Determine output dimensions and per-pixel spacing
        raw_stack = self.vbf_images        # (Nb, Ry, Rx)
        Nb, Ry, Rx = raw_stack.shape
        Ny_out = round(Ry * upscale)
        Nx_out = round(Rx * upscale)

        # 2. Setup transformed k-shifts
        # Note: kX / kY are returned explicitly, but the transform itself is applied in y/x convention.
        kX_full, kY_full = self._get_transformed_bf_coordinates()

        # 3. Base Spatial Frequencies (For the sub-pixel Fourier phase ramp).
        # When deferring upscale, build the ramp grid at native resolution; the accumulated
        # F_shifted will be zero-padded to (Ny_out, Nx_out) in _get_tcBF_from_cache.
        # Otherwise, use the actual output pixel size so that non-integer upscale factors
        # are handled correctly.
        if self.defer_upscale:
            qx_grid = torch.fft.fftfreq(Rx, d=self.scan_step_size, device=self.device).view(1, 1, Rx)
            qy_grid = torch.fft.fftfreq(Ry, d=self.scan_step_size, device=self.device).view(1, Ry, 1)
        else:
            dx = (Rx * self.scan_step_size) / Nx_out
            dy = (Ry * self.scan_step_size) / Ny_out
            qx_grid = torch.fft.fftfreq(Nx_out, d=dx, device=self.device).view(1, 1, Nx_out)
            qy_grid = torch.fft.fftfreq(Ny_out, d=dy, device=self.device).view(1, Ny_out, 1)

        cache = []

        for i in range(0, Nb, chunk_size):
            end = min(i + chunk_size, Nb)

            kX_chunk = kX_full[i:end]
            kY_chunk = kY_full[i:end]

            # Generate unweighted analytic shift basis matrix [Num_Coeffs, ChunkSize]
            b_dx, b_dy = generate_shift_basis(self.ab_state.order_keys, kX_chunk, kY_chunk, self.wavelength)

            # Compute img_fft — at native resolution when deferring, upscaled otherwise.
            if self.defer_upscale:
                img_fft = torch.fft.fft2(raw_stack[i:end], dim=(-2, -1))
            else:
                img_fft = self._compute_upscaled_img_fft(raw_stack[i:end], upscale, Ny_out, Nx_out)

            cache.append({
                'b_dx': b_dx,
                'b_dy': b_dy,
                'img_fft': img_fft,
            })

        return cache, qx_grid, qy_grid, (Ny_out, Nx_out)

    def _build_acBF_cache(self, upscale=1, rolloff=0, chunk_size=64):
        """
        Pre-computes and chunks all static geometry, soft apertures, and FFTs.

        What is stored per chunk depends on self.cache_mode:
            'full_gpu' — bases + apertures + img_fft on GPU; vbf_images offloaded to CPU after.
            'fft_gpu'  — only img_fft (+ scalar kxt/kyt) on GPU; bases recomputed each call.
            'full_cpu' — full chunk dicts pinned to CPU RAM; transferred chunk-by-chunk at use.
        """

        # 1. Determine output dimensions and pixel spacing
        raw_stack = self.vbf_images        # (Nb, Ry, Rx)
        Nb, Ry, Rx = raw_stack.shape
        Ny_out = round(Ry * upscale)
        Nx_out = round(Rx * upscale)

        # 2. Setup transformed k-shifts.
        # kX/kY keep their explicit meaning even though the transform is applied in y/x convention.
        kX_full, kY_full = self._get_transformed_bf_coordinates()
        kX_full = kX_full.view(Nb, 1, 1)
        kY_full = kY_full.view(Nb, 1, 1)

        # 3. Base Spatial Frequencies.
        # When deferring upscale, work at native resolution so that cached img_fft, apertures,
        # and bases are all (chunk_size, Ry, Rx) — reducing VRAM by upscale^2.  The
        # accumulated F_corr will be zero-padded to (Ny_out, Nx_out) in the reconstruction
        # functions.  For fft_gpu mode the slow-path in _compute_acbf_transfer already reads
        # the grid size from chunk['img_fft'].shape[-2:], so storing native img_fft is enough.
        if self.defer_upscale:
            kx_base = torch.fft.fftfreq(Rx, d=self.scan_step_size, device=self.device).view(1, 1, Rx)
            ky_base = torch.fft.fftfreq(Ry, d=self.scan_step_size, device=self.device).view(1, Ry, 1)
        else:
            # Use actual output pixel size so non-integer upscale factors are handled correctly.
            dx = (Rx * self.scan_step_size) / Nx_out
            dy = (Ry * self.scan_step_size) / Ny_out
            kx_base = torch.fft.fftfreq(Nx_out, d=dx, device=self.device).view(1, 1, Nx_out)
            ky_base = torch.fft.fftfreq(Ny_out, d=dy, device=self.device).view(1, Ny_out, 1)

        cache = []

        for i in range(0, Nb, chunk_size):
            end = min(i + chunk_size, Nb)

            kxt = kX_full[i:end]   # (chunk, 1, 1)
            kyt = kY_full[i:end]

            # Compute img_fft — at native resolution when deferring, upscaled otherwise.
            if self.defer_upscale:
                img_fft = torch.fft.fft2(raw_stack[i:end], dim=(-2, -1))
            else:
                img_fft = self._compute_upscaled_img_fft(raw_stack[i:end], upscale, Ny_out, Nx_out)

            if self.cache_mode == 'fft_gpu':
                # Lightweight cache: only img_fft on GPU + scalar k-coords for lazy recompute.
                # Bases (~603 MB/chunk) are NOT stored; they are recomputed on every call.
                cache.append({
                    'kxt': kxt.detach().clone(),   # (chunk, 1, 1) float32 — tiny
                    'kyt': kyt.detach().clone(),
                    'img_fft': img_fft,
                })
                continue   # skip the expensive basis computation below

            # --- full_gpu / full_cpu: build complete chunk ---

            # Kinematic Grids (q-k and q+k)
            kx_t,  ky_t  = kx_base + kxt, ky_base + kyt
            kx_mt, ky_mt = kx_base - kxt, ky_base - kyt

            # Soft Apertures
            alpha_t  = torch.sqrt(kx_t**2  + ky_t**2)  * self.wavelength
            ap_t  = make_soft_aperture_torch(alpha_t,  self.max_alpha, rolloff)
            alpha_mt = torch.sqrt(kx_mt**2 + ky_mt**2) * self.wavelength
            ap_mt = make_soft_aperture_torch(alpha_mt, self.max_alpha, rolloff)

            # Polynomial Basis Tensors
            b_tr = generate_aberration_basis(self.max_order, self.ab_state.order_keys, kxt,    kyt,    self.wavelength)
            b_t  = generate_aberration_basis(self.max_order, self.ab_state.order_keys, kx_t,   ky_t,   self.wavelength)
            b_mt = generate_aberration_basis(self.max_order, self.ab_state.order_keys, -kx_mt, -ky_mt, self.wavelength)

            chunk = {
                'ap_t': ap_t, 'ap_mt': ap_mt,
                'b_tr': b_tr, 'b_t': b_t, 'b_mt': b_mt,
                'img_fft': img_fft,
            }

            if self.cache_mode == 'full_cpu':
                # Move entire chunk to CPU pinned RAM for async GPU transfer during reconstruction
                chunk = {k: v.cpu().pin_memory() for k, v in chunk.items()}

            cache.append(chunk)

        return cache, (Ny_out, Nx_out)

    def _get_tcBF_cache(self, upscale=1, chunk_size=64):
        key = ('tcBF', upscale, chunk_size, self.defer_upscale)

        if key not in self._cache_store:
            self._cache_store[key] = self._build_tcBF_cache(upscale=upscale, chunk_size=chunk_size)

        return self._cache_store[key]

    def _get_acBF_cache(self, upscale=1, rolloff=0, chunk_size=64):
        key = ('acBF', upscale, rolloff, chunk_size, self.defer_upscale)

        if key not in self._cache_store:
            cache, out_shape = self._build_acBF_cache(upscale=upscale, rolloff=rolloff, chunk_size=chunk_size)
            # Store rolloff alongside the cache so the lazy-recompute path in
            # _compute_acbf_transfer can recreate soft apertures with the same rolloff.
            self._cache_store[key] = (cache, out_shape, rolloff)

        return self._cache_store[key]

    def _get_tcBF_from_cache(self, cache, qx_grid, qy_grid, out_shape):
        """
        Ultra-lean AD forward pass for tcBF.
        Calculates exact analytical shifts instantly via Einstein summation.

        When self.defer_upscale is True, phase ramps and img_fft are at native resolution.
        F_shifted contributions are accumulated in native Fourier space, then zero-padded
        to out_shape and inverse-transformed in a single IFFT at the end.
        """
        neg_two_pi_j = torch.tensor(-2.0j * torch.pi, dtype=torch.complex64, device=self.device)

        # Extract the 1D coefficient tensor (Shape: Num_Coeffs)
        C = self.ab_state.get_flat_coeffs()

        if self.defer_upscale:
            # Accumulate in native Fourier space; one zero-pad + IFFT at the end.
            Ry_n, Rx_n = self.Ry, self.Rx
            F_sum = torch.zeros((Ry_n, Rx_n), dtype=torch.complex64, device=self.device)
            for chunk in cache:
                shift_dx = torch.einsum('k, kb -> b', C, chunk['b_dx']).view(-1, 1, 1)
                shift_dy = torch.einsum('k, kb -> b', C, chunk['b_dy']).view(-1, 1, 1)
                ramp = shift_dx * qx_grid + shift_dy * qy_grid
                shift_op = torch.exp(neg_two_pi_j * ramp)
                F_sum += torch.sum(chunk['img_fft'] * shift_op, dim=0)
            Ny_out, Nx_out = out_shape
            return torch.fft.ifft2(self._zero_pad_fft2(F_sum, Ny_out, Nx_out), dim=(-2, -1)).real
        else:
            tcBF_total = torch.zeros(out_shape, dtype=torch.float32, device=self.device)
            for chunk in cache:
                # 1. Instantly calculate physical shift vectors by dot-producing Coeffs and Basis
                # 'k' = Coeff index, 'b' = Batch size of chunk. Result shaped for broadcasting: [Batch, 1, 1]
                shift_dx = torch.einsum('k, kb -> b', C, chunk['b_dx']).view(-1, 1, 1)
                shift_dy = torch.einsum('k, kb -> b', C, chunk['b_dy']).view(-1, 1, 1)

                # 2. Generate Sub-pixel Phase Ramp
                ramp = shift_dx * qx_grid + shift_dy * qy_grid
                shift_op = torch.exp(neg_two_pi_j * ramp)

                # 3. Apply to pre-computed image FFT and Reconstruct
                F_shifted = chunk['img_fft'] * shift_op
                tcBF_total += torch.sum(torch.fft.ifft2(F_shifted, dim=(-2, -1)).real, dim=0)

            return tcBF_total

    def _compute_acbf_transfer(self, chunk, coeffs, rolloff=0):
        """
        Compute the detector-wise complex transfer for acBF.

        Two paths depending on chunk content (self-describing structure):
            Fast path ('full_gpu' / 'full_cpu'): bases are pre-cached in the chunk.
            Slow path ('fft_gpu'):  bases are absent; recomputed from stored kxt/kyt.
              The slow path adds ~50–100 ms per reconstruction call but reduces static
              VRAM by ~6 GB.  The recomputed tensors are local and freed after the einsum.

        Args:
            chunk:   Cache chunk dict.
            coeffs:  Flat aberration coefficient tensor (num_coeffs,).
            rolloff: Soft-aperture rolloff (used only in slow path to recreate apertures).

        Returns:
            torch.Tensor: Complex transfer T with shape (chunk_size, Ny, Nx).
        """
        j1 = torch.tensor(1.0j, dtype=torch.complex64, device=self.device)

        if 'b_tr' in chunk:
            # Fast path: all tensors are already on the target device
            b_tr  = chunk['b_tr']
            b_t   = chunk['b_t']
            b_mt  = chunk['b_mt']
            ap_t  = chunk['ap_t']
            ap_mt = chunk['ap_mt']
        else:
            # Slow path (fft_gpu mode): recompute bases from stored scalar k-coordinates.
            # kxt/kyt are (chunk, 1, 1); img_fft shape tells us the output grid size.
            kxt = chunk['kxt'].to(self.device)
            kyt = chunk['kyt'].to(self.device)
            Ny, Nx = chunk['img_fft'].shape[-2:]
            dx = (self.Rx * self.scan_step_size) / Nx
            dy = (self.Ry * self.scan_step_size) / Ny
            kx_base = torch.fft.fftfreq(Nx, d=dx, device=self.device).view(1, 1, Nx)
            ky_base = torch.fft.fftfreq(Ny, d=dy, device=self.device).view(1, Ny, 1)
            kx_t,  ky_t  = kx_base + kxt, ky_base + kyt
            kx_mt, ky_mt = kx_base - kxt, ky_base - kyt
            ap_t  = make_soft_aperture_torch(
                torch.sqrt(kx_t**2  + ky_t**2)  * self.wavelength, self.max_alpha, rolloff)
            ap_mt = make_soft_aperture_torch(
                torch.sqrt(kx_mt**2 + ky_mt**2) * self.wavelength, self.max_alpha, rolloff)
            b_tr = generate_aberration_basis(
                self.max_order, self.ab_state.order_keys, kxt,    kyt,    self.wavelength)
            b_t  = generate_aberration_basis(
                self.max_order, self.ab_state.order_keys, kx_t,   ky_t,   self.wavelength)
            b_mt = generate_aberration_basis(
                self.max_order, self.ab_state.order_keys, -kx_mt, -ky_mt, self.wavelength)
            # b_tr/b_t/b_mt are local tensors; freed after the einsums below

        chi_tr_az = torch.einsum('k, kbxy -> bxy', coeffs, b_tr)
        chi_t     = torch.einsum('k, kbxy -> bxy', coeffs, b_t)
        chi_mt    = torch.einsum('k, kbxy -> bxy', coeffs, b_mt)

        term_mt = ap_mt * torch.exp(-j1 * (chi_tr_az - chi_mt))
        term_t  = ap_t  * torch.exp( j1 * (chi_tr_az - chi_t))
        D = term_mt - term_t

        # T = -i * D. This keeps the transfer aligned with the current acBF phasor
        # convention while also exposing the full complex transfer for matched filtering.
        return (-j1) * D

    def _get_acBF_from_cache(self, cache, out_shape, rolloff=0):
        """
        Phase-only acBF reconstruction.

        This is the legacy acBF path that aligns detector contributions by phase
        before summation, preserving the original default behavior.

        For 'full_cpu' cache_mode, chunks are streamed from CPU pinned RAM with async
        prefetch so that the next chunk's H2D transfer overlaps with the current computation.

        When self.defer_upscale is True, F_corr is accumulated in native Fourier space across
        all chunks, then zero-padded to out_shape and inverse-transformed in a single IFFT.
        """
        coeffs = self.ab_state.get_flat_coeffs()
        Ny_out, Nx_out = out_shape

        if self.defer_upscale:
            # Accumulate in native Fourier space; one zero-pad + IFFT at the end.
            Ry_n, Rx_n = self.Ry, self.Rx
            F_sum = torch.zeros((Ry_n, Rx_n), dtype=torch.complex64, device=self.device)

            def _accumulate_deferred(chunk_gpu):
                transfer = self._compute_acbf_transfer(chunk_gpu, coeffs, rolloff=rolloff)
                phasor = transfer / (transfer.abs() + self.eps)
                F_sum.add_(torch.sum(chunk_gpu['img_fft'] * phasor, dim=0))

            if self.cache_mode == 'full_cpu':
                prefetch_stream = torch.cuda.Stream(device=self.device)

                def _to_gpu(cpu_chunk):
                    with torch.cuda.stream(prefetch_stream):
                        return {k: v.to(self.device, non_blocking=True) for k, v in cpu_chunk.items()}

                gpu_chunk = _to_gpu(cache[0])
                for idx in range(len(cache)):
                    next_gpu = _to_gpu(cache[idx + 1]) if idx + 1 < len(cache) else None
                    torch.cuda.current_stream(self.device).wait_stream(prefetch_stream)
                    _accumulate_deferred(gpu_chunk)
                    gpu_chunk = next_gpu
            else:
                for chunk in cache:
                    _accumulate_deferred(chunk)

            return torch.fft.ifft2(self._zero_pad_fft2(F_sum, Ny_out, Nx_out), dim=(-2, -1)).real

        else:
            acBF_total = torch.zeros(out_shape, dtype=torch.float32, device=self.device)

            if self.cache_mode == 'full_cpu':
                prefetch_stream = torch.cuda.Stream(device=self.device)

                def _to_gpu(cpu_chunk):
                    with torch.cuda.stream(prefetch_stream):
                        return {k: v.to(self.device, non_blocking=True) for k, v in cpu_chunk.items()}

                gpu_chunk = _to_gpu(cache[0])
                for idx in range(len(cache)):
                    next_gpu = _to_gpu(cache[idx + 1]) if idx + 1 < len(cache) else None
                    torch.cuda.current_stream(self.device).wait_stream(prefetch_stream)
                    transfer = self._compute_acbf_transfer(gpu_chunk, coeffs, rolloff=rolloff)
                    phasor = transfer / (transfer.abs() + self.eps)
                    F_corr = gpu_chunk['img_fft'] * phasor
                    acBF_total += torch.sum(torch.fft.ifft2(F_corr, dim=(-2, -1)).real, dim=0)
                    gpu_chunk = next_gpu
            else:
                for chunk in cache:
                    transfer = self._compute_acbf_transfer(chunk, coeffs, rolloff=rolloff)
                    phasor = transfer / (transfer.abs() + self.eps)
                    F_corr = chunk['img_fft'] * phasor
                    acBF_total += torch.sum(torch.fft.ifft2(F_corr, dim=(-2, -1)).real, dim=0)

            return acBF_total

    def _get_acBF_complex_inversion_from_cache(
        self,
        cache,
        out_shape,
        rolloff=0,
        regularization=1e-3,
        support_threshold=1e-6,
        return_diagnostics=False,
    ):
        """
        Complex-inversion acBF reconstruction via regularized transfer inversion.

        The estimator solves a regularized matched-filter inversion of the detector-wise
        complex transfer. The current phase-only acBF path implicitly uses the forward
        model I_b(q) = conj(T_b(q)) * V(q), so the matched-filter numerator must apply
        T_b (not conj(T_b)) to recover a phase-aligned estimate.

            V_hat(q) = M(q) / (S(q) + lambda * S_ref)

        where:
            M(q) = sum_b T_b(q) * I_b(q)
            S(q) = sum_b |T_b(q)|^2
        """
        if regularization < 0:
            raise ValueError(f"regularization must be non-negative, got {regularization}.")
        if support_threshold < 0:
            raise ValueError(f"support_threshold must be non-negative, got {support_threshold}.")

        coeffs = self.ab_state.get_flat_coeffs()
        Ny_out, Nx_out = out_shape

        # When deferring upscale, accumulate M(q) and S(q) at native resolution, compute
        # the Fourier estimate at native resolution, then zero-pad before the final IFFT.
        # Note: we zero-pad the *ratio* fourier_estimate, not numerator/denom separately,
        # because zero_pad(A/B) ≠ zero_pad(A) / zero_pad(B).
        acc_shape = (self.Ry, self.Rx) if self.defer_upscale else out_shape

        numerator = torch.zeros(acc_shape, dtype=torch.complex64, device=self.device)
        transfer_power = torch.zeros(acc_shape, dtype=torch.float32, device=self.device)

        def _accumulate(chunk_gpu):
            transfer = self._compute_acbf_transfer(chunk_gpu, coeffs, rolloff=rolloff)
            numerator.add_(torch.sum(transfer * chunk_gpu['img_fft'], dim=0))
            transfer_power.add_(torch.sum(transfer.abs().square(), dim=0))

        if self.cache_mode == 'full_cpu':
            prefetch_stream = torch.cuda.Stream(device=self.device)

            def _to_gpu(cpu_chunk):
                with torch.cuda.stream(prefetch_stream):
                    return {k: v.to(self.device, non_blocking=True) for k, v in cpu_chunk.items()}

            gpu_chunk = _to_gpu(cache[0])
            for idx in range(len(cache)):
                next_gpu = _to_gpu(cache[idx + 1]) if idx + 1 < len(cache) else None
                torch.cuda.current_stream(self.device).wait_stream(prefetch_stream)
                _accumulate(gpu_chunk)
                gpu_chunk = next_gpu
        else:
            for chunk in cache:
                _accumulate(chunk)

        positive_power = transfer_power[transfer_power > 0]
        if positive_power.numel() == 0:
            transfer_reference = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        else:
            transfer_reference = positive_power.median()

        support = transfer_power > (support_threshold * transfer_reference)
        denom = transfer_power + (regularization * transfer_reference)
        fourier_estimate = torch.where(
            support,
            numerator / denom.to(torch.complex64),
            torch.zeros_like(numerator),
        )

        if self.defer_upscale:
            fourier_estimate = self._zero_pad_fft2(fourier_estimate, Ny_out, Nx_out)
            transfer_power   = self._zero_pad_fft2(
                transfer_power.to(torch.complex64), Ny_out, Nx_out
            ).real

        complex_image = torch.fft.ifft2(fourier_estimate, dim=(-2, -1))
        reconstructed = complex_image.real

        if not return_diagnostics:
            return reconstructed

        return {
            'image': reconstructed,
            'complex_image': complex_image,
            'real_channel': complex_image.real,
            'imag_channel': complex_image.imag,
            'fourier_estimate': fourier_estimate,
            'transfer_power': transfer_power,
            'support_mask': support if not self.defer_upscale else transfer_power > (support_threshold * transfer_reference),
            'support_reference': transfer_reference,
        }

    def _build_c10_stack_axis(self, n_layers=None, z_top=None, z_bottom=None, slice_thickness=None):
        """
        Build a 1D absolute C10 axis in Angstroms for defocus-stack reconstruction.

        Supported mutually exclusive modes:
            1. n_layers + slice_thickness
            2. z_top + z_bottom + slice_thickness
        """
        has_n_layers = n_layers is not None
        has_range_arg = any(val is not None for val in (z_top, z_bottom))

        if slice_thickness is None:
            raise ValueError("slice_thickness is required for defocus-stack reconstruction.")

        slice_thickness = float(slice_thickness)
        if slice_thickness <= 0:
            raise ValueError(f"slice_thickness must be positive, got {slice_thickness}.")

        if has_n_layers and has_range_arg:
            raise ValueError(
                "Provide either n_layers or z_top/z_bottom with slice_thickness, not both."
            )

        if has_n_layers:
            if z_top is not None or z_bottom is not None:
                raise ValueError(
                    "n_layers mode does not accept z_top or z_bottom."
                )

            if not isinstance(n_layers, (int, np.integer)):
                raise ValueError(f"n_layers must be a positive integer, got {n_layers!r}.")

            n_layers = int(n_layers)
            if n_layers <= 0:
                raise ValueError(f"n_layers must be positive, got {n_layers}.")

            c10_center = float(self.ab_state.coeffs['C_1_0'].detach().item())
            offsets = (torch.arange(n_layers, device=self.device, dtype=torch.float32) - ((n_layers - 1) / 2.0))
            return c10_center + offsets * slice_thickness

        if has_range_arg:
            if z_top is None or z_bottom is None:
                raise ValueError(
                    "Range mode requires z_top, z_bottom, and slice_thickness together."
                )

            start = float(z_top)
            stop = float(z_bottom)
            delta = stop - start

            if delta == 0:
                return torch.tensor([start], dtype=torch.float32, device=self.device)

            direction = 1.0 if delta > 0 else -1.0
            steps = int(np.floor(abs(delta) / slice_thickness))
            offsets = torch.arange(steps + 1, device=self.device, dtype=torch.float32)
            return start + direction * slice_thickness * offsets

        raise ValueError(
            "Provide either n_layers with slice_thickness, or z_top, z_bottom, and slice_thickness."
        )

    def _sweep_c10_stack(self, c10_axis, mode='tcBF', output_frame=None, **kwargs):
        """
        Evaluate a read-only reconstruction stack over an absolute C10 axis.

        The returned stack is expressed in the requested output frame:
            - output_frame='scan' stores scan-frame slices
            - output_frame='detector' stores detector-frame slices with rotation applied eagerly per-slice

        Unlike get_reconstructed_image(), this helper does not populate the normal
        single-image reconstruction cache. It only updates self.last_c10_stack_axis
        to record the most recent sweep axis used by the solver.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (c10_axis, image_stack)
        """
        original_c10 = self.ab_state.coeffs['C_1_0'].detach().clone()
        original_reconstructed_image = self.reconstructed_image
        original_reconstructed_images = self._reconstructed_images
        stack_images = []

        try:
            with torch.no_grad():
                for c10 in c10_axis:
                    self.ab_state.coeffs['C_1_0'].copy_(c10)

                    # Rebuild only dynamic image caches so static tcBF/acBF caches can be reused.
                    self.reconstructed_image = None
                    self._reconstructed_images = {}

                    img = self.reconstruct(mode=mode, **kwargs)

                    rotation_deg = self.coord_transform.get('rotation_deg', 0.0)
                    if not self._in_scan_frame(output_frame) and rotation_deg:
                        img = tv_rotate(
                            img.unsqueeze(0),
                            angle=-rotation_deg,
                            interpolation=InterpolationMode.BILINEAR,
                        ).squeeze(0)

                    stack_images.append(img)
        finally:
            with torch.no_grad():
                self.ab_state.coeffs['C_1_0'].copy_(original_c10)

            self.reconstructed_image = original_reconstructed_image
            self._reconstructed_images = original_reconstructed_images

        c10_axis = c10_axis.detach().clone()
        stack = torch.stack(stack_images, dim=0)
        self.last_c10_stack_axis = c10_axis
        return c10_axis, stack

    # Public methods
    # Getter and Printing
    def get_aberrations_dict(self, output_frame=None, notation='krivanek', style='cartesian', layout='nested'):
        """
        Return aberration coefficients in the requested notation/style/layout.

        output_frame=None        → uses self.output_frame (global default).
        output_frame='scan'      → coefficients in the scan frame (as fitted, includes
                                   rotation_deg).  Non-symmetric terms encode orientation
                                   relative to the scan fast-axis.
        output_frame='detector'  → coefficients converted to the detector frame
                                   (rotation_deg un-applied).  Pass these directly to
                                   PtyRAD's `probe_aberrations`.
        """
        in_scan = self._in_scan_frame(output_frame)
        ab_state = self._get_effective_ab_state(in_scan)
        return Aberrations(ab_state.get_cartesian_dict()).export(notation=notation, style=style, layout=layout)

    def print_aberrations(self, output_frame=None):
        """
        Print aberration coefficients.

        output_frame=None        → uses self.output_frame (global default).
        output_frame='scan'      → prints coefficients in the scan frame (as fitted).
        output_frame='detector'  → prints coefficients in the detector frame, suitable
                                   for direct export to PtyRAD.

        A warning is emitted when output_frame='scan', rotation_deg != 0, and
        non-symmetric aberrations are nonzero — because those orientation angles are
        relative to the scan fast-axis, not the detector kX axis.  Use
        output_frame='detector' (or set output_frame='detector' globally) to get
        values that are directly compatible with PtyRAD's `probe_aberrations`.
        """
        in_scan = self._in_scan_frame(output_frame)
        rotation_deg = self.coord_transform.get('rotation_deg', 0.0)
        if in_scan and rotation_deg:
            ab_dict = self.ab_state.get_cartesian_dict()
            has_non_symmetric = any(
                v.get('a', 0.0) != 0.0 or v.get('b', 0.0) != 0.0
                for k, v in ab_dict.items()
                if k[-1] != 0
            )
            if has_non_symmetric:
                logger.warning(
                    f"rotation_deg = {rotation_deg} is set and non-symmetric aberrations "
                    "(C12, C23, ...) are nonzero. The displayed coefficients are in the "
                    "scan frame — orientation angles are relative to the scan fast-axis. "
                    "To get detector-frame values compatible with PtyRAD's "
                    "`probe_aberrations`, use print_aberrations(output_frame='detector') "
                    "or set output_frame='detector' on the solver."
                )
        print(Aberrations(self.get_aberrations_dict(output_frame=output_frame)))

    def get_chi_surface(self, output_frame=None):
        """
        Return aberration surface chi. Note: psi = exp(-1j*chi).

        The chi surface is computed exactly in the requested frame — k-grids and
        aberration coefficients are both expressed in the same frame, so no
        interpolation or approximation is involved.

        output_frame=None        → uses self.output_frame (global default).
        output_frame='scan'      → chi in the scan frame (k-grid rotated by rotation_deg,
                                   coefficients as fitted).
        output_frame='detector'  → chi in the detector frame (flip-corrected k-grid,
                                   coefficients converted back from the scan frame).
                                   Ready to use as a probe seed in PtyRAD.
        """
        in_scan = self._in_scan_frame(output_frame)
        kX_grid, kY_grid = self._get_transformed_k_grids(in_scan_frame=in_scan)
        ab_state = self._get_effective_ab_state(in_scan)

        chi_basis = generate_aberration_basis(
            self.max_order,
            ab_state.order_keys,
            kX_grid,
            kY_grid,
            self.wavelength,
        )

        coeffs = ab_state.get_flat_coeffs()
        chi = torch.einsum('k,kij->ij', coeffs, chi_basis)

        return chi

    def get_yx_shifts_ang(self, output_frame=None):
        """
        Return image shifts in Angstroms as (Nb, 2) tensor, each row is (shift_y, shift_x).

        Shifts are computed exactly in the requested frame — BF coordinates and
        aberration coefficients are both expressed in the same frame.

        output_frame=None        → uses self.output_frame (global default).
        output_frame='scan'      → shifts in the scan frame (as fitted).
        output_frame='detector'  → shifts in the detector frame (rotation_deg un-applied).
        """
        in_scan = self._in_scan_frame(output_frame)
        kX_centers, kY_centers = self._get_transformed_bf_coordinates(in_scan_frame=in_scan)
        ab_state = self._get_effective_ab_state(in_scan)

        b_dx, b_dy = generate_shift_basis(
            ab_state.order_keys,
            kX_centers,
            kY_centers,
            self.wavelength,
        )

        coeffs = ab_state.get_flat_coeffs()
        shift_x_ang = torch.einsum('k,kb->b', coeffs, b_dx)
        shift_y_ang = torch.einsum('k,kb->b', coeffs, b_dy)

        return torch.stack([shift_y_ang, shift_x_ang], dim=-1)  # shape (Nb, 2)

    def get_yx_shifts_px(self, output_frame=None):
        """Return image shifts in real-space pixels as (Nb, 2) tensor, each row is (shift_y, shift_x)."""
        return self.get_yx_shifts_ang(output_frame=output_frame) / self.scan_step_size

    def reconstruct(self, mode='tcBF', **kwargs):
        """
        Unified reconstruction entry point.
        
        Notes:
            - tcBF is the default mode.
            - acBF supports `acbf_algorithm='phase_only'` (default) and
              `acbf_algorithm='complex_inversion'`.
        """
        mode_key = mode.lower()
        
        if mode_key == 'tcbf':
            upscale = kwargs.get('upscale', 1)
            chunk_size = kwargs.get('chunk_size', 64)
            cache, qx_grid, qy_grid, out_shape = self._get_tcBF_cache(upscale=upscale, chunk_size=chunk_size)
            return self._get_tcBF_from_cache(cache, qx_grid, qy_grid, out_shape)
        
        elif mode_key == 'acbf':
            upscale = kwargs.get('upscale', 1)
            rolloff = kwargs.get('rolloff', 0)
            chunk_size = kwargs.get('chunk_size', 64)
            acbf_algorithm = self._normalize_acbf_algorithm(kwargs.get('acbf_algorithm', 'phase_only'))
            regularization = kwargs.get('regularization', 1e-3)
            support_threshold = kwargs.get('support_threshold', 1e-6)
            cache, out_shape, cached_rolloff = self._get_acBF_cache(upscale=upscale, rolloff=rolloff, chunk_size=chunk_size)

            if acbf_algorithm == 'phase_only':
                return self._get_acBF_from_cache(cache, out_shape, rolloff=cached_rolloff)
            if acbf_algorithm == 'complex_inversion':
                return self._get_acBF_complex_inversion_from_cache(
                    cache,
                    out_shape,
                    rolloff=cached_rolloff,
                    regularization=regularization,
                    support_threshold=support_threshold,
                )
            raise ValueError(
                f"Unsupported acBF algorithm '{acbf_algorithm}'. "
                "Choose between 'phase_only' and 'complex_inversion'."
            )
        
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Please choose between 'tcBF' and 'acBF'.")

    def get_reconstructed_image(self, mode='tcBF', output_frame=None, **kwargs):
        """
        Return (and cache) the reconstructed image.

        Note: the reconstruction itself is always computed in the scan frame (the frame
        the aberrations were fitted in).  The internal cache stores this scan-frame result.
        When output_frame='detector' and rotation_deg != 0, the cached image is
        post-processed with a 2D image rotation of -rotation_deg (bilinear, torchvision)
        to bring it into the detector frame for visualization.  This post-processed result
        is NOT stored in the cache.

        output_frame=None        → uses self.output_frame (global default).
        output_frame='scan'      → image as reconstructed, in the scan frame.
        output_frame='detector'  → image rotated back to the detector frame for display.
        """
        mode = mode.lower()
        cache_key = (mode, tuple(sorted(kwargs.items())))

        if cache_key not in self._reconstructed_images:
            self._reconstructed_images[cache_key] = self.reconstruct(mode=mode, **kwargs)

        self.reconstructed_image = self._reconstructed_images[cache_key]
        img = self.reconstructed_image

        rotation_deg = self.coord_transform.get('rotation_deg', 0.0)
        if not self._in_scan_frame(output_frame) and rotation_deg:
            # tv_rotate expects (..., H, W) and angle in degrees (CCW positive)
            img = tv_rotate(img.unsqueeze(0), angle=-rotation_deg,
                            interpolation=InterpolationMode.BILINEAR).squeeze(0)

        return img

    def get_tcBF(self, output_frame=None, **kwargs):
        return self.get_reconstructed_image(mode='tcBF', output_frame=output_frame, **kwargs)

    def get_acBF(self, output_frame=None, **kwargs):
        return self.get_reconstructed_image(mode='acBF', output_frame=output_frame, **kwargs)

    def get_acBF_diagnostics(self, **kwargs):
        """
        Return transfer diagnostics for the complex-inversion acBF estimator.

        Accepted kwargs mirror the acBF reconstruction path:
            - upscale
            - rolloff
            - chunk_size
            - regularization
            - support_threshold

        Returns a dictionary containing the complex estimate, real/imag channels,
        and the transfer-power support used by the regularized inversion.
        """
        upscale = kwargs.get('upscale', 1)
        rolloff = kwargs.get('rolloff', 0)
        chunk_size = kwargs.get('chunk_size', 64)
        regularization = kwargs.get('regularization', 1e-3)
        support_threshold = kwargs.get('support_threshold', 1e-6)
        cache, out_shape, cached_rolloff = self._get_acBF_cache(upscale=upscale, rolloff=rolloff, chunk_size=chunk_size)
        return self._get_acBF_complex_inversion_from_cache(
            cache,
            out_shape,
            rolloff=cached_rolloff,
            regularization=regularization,
            support_threshold=support_threshold,
            return_diagnostics=True,
        )

    def get_defocus_stack(
        self,
        mode='tcBF',
        output_frame=None,
        n_layers=None,
        z_top=None,
        z_bottom=None,
        slice_thickness=None,
        **kwargs,
    ):
        """
        Return a read-only defocus stack with shape (Nz, Ny, Nx).

        The stack axis is absolute C10 in Angstroms. The exact axis used for the
        returned stack is stored in self.last_c10_stack_axis.

        For acBF, callers may pass `acbf_algorithm='phase_only'` or
        `acbf_algorithm='complex_inversion'`. The latter performs a regularized
        complex transfer inversion intended to reduce depth-dependent contrast
        modulation within the weak-phase approximation.

        Note: self.last_c10_stack_axis records the most recent C10 sweep axis used
        by the solver, including internal sweeps such as refine_defocus().

        Supported mutually exclusive modes:
            1. n_layers + slice_thickness
            2. z_top + z_bottom + slice_thickness
        """
        mode = mode.lower()
        c10_axis = self._build_c10_stack_axis(
            n_layers=n_layers,
            z_top=z_top,
            z_bottom=z_bottom,
            slice_thickness=slice_thickness,
        )
        _, stack = self._sweep_c10_stack(c10_axis, mode=mode, output_frame=output_frame, **kwargs)
        return stack

    def get_probe(self, output_frame=None):
        """
        Return the complex probe wavefield.

        The probe is computed exactly in the requested frame (same as get_chi_surface).

        output_frame=None        → uses self.output_frame (global default).
        output_frame='scan'      → probe in the scan frame (as fitted).
        output_frame='detector'  → probe in the detector frame, ready for PtyRAD.
        """
        probe = make_probe_from_chi(self.get_chi_surface(output_frame=output_frame), self.bf_mask)
        return probe
    
    # Refinement
    # TODO, this might be helpful for challenging data without initial guess
    def refine_register(self, max_shifts=None):
        """Refines shifts using rigid registration, then back-fits aberrations."""
        # Rigid registration logic
        print("Executed: Rigid Registration Refinement")
        return self

    def refine_defocus(self, search_range: tuple, num_points=5, metric='laplacian', method='fit_parabola', blur=True, blur_kernel_size=5, blur_sigma=1, plot_line_search=True, mode='tcBF', **kwargs):
        """
        Performs a brute-force line search for defocus (-C10) and fits a parabola 
        to estimate the optimal defocus value.
        """
        mode = mode.lower()
        # 1. Setup the test array
        min_def, max_def = min(search_range), max(search_range)
        search_range = (min_def, max_def)
        print(f"Starting defocus line search: {num_points} points between {search_range[0]} and {search_range[1]} Ang")
        
        c10_axis = torch.linspace(search_range[0], search_range[1], num_points, dtype=torch.float32, device=self.device)

        # 2. Evaluate the full sweep first, then score each slice from the stack.
        # Always sweep in scan frame: quality metrics are frame-agnostic and the scan frame is
        # the canonical reconstruction frame. The optimal image is stored in the scan-frame cache;
        # get_reconstructed_image() applies output_frame rotation lazily on read.
        c10_axis, scan_stack = self._sweep_c10_stack(c10_axis, mode=mode, output_frame='scan', **kwargs)
        quality_scores = QualityMetrics.evaluate(
            scan_stack,
            metric=metric,
            blur=blur,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
        ).detach().cpu().numpy()
        c10_axis_np = c10_axis.detach().cpu().numpy()
        
        method = method.lower()
        if method == 'fit_parabola':
            coeffs = np.polyfit(c10_axis_np, quality_scores, 2)
            a, b, c = coeffs
            
            if a < 0:
                optimal_c10 = -b / (2 * a)
                optimal_c10 = np.clip(optimal_c10, search_range[0], search_range[1])
                fit_type = "Parabolic vertex"
                optimal_index = int(np.argmin(np.abs(c10_axis_np - optimal_c10)))
            else:
                optimal_index = int(np.argmax(quality_scores))
                optimal_c10 = c10_axis_np[optimal_index]
                fit_type = "Discrete max (fit inverted)"
                
        elif method == 'max':
            optimal_index = int(np.argmax(quality_scores))
            optimal_c10 = c10_axis_np[optimal_index]
        else:
            raise ValueError(f"Unsupported method: {method}, please choose between 'fit_parabola' or 'max'")
            
        print(f"Optimal C10 found at {optimal_c10:.2f} Ang ({method})")
        
        with torch.no_grad():
            self.ab_state.coeffs['C_1_0'].copy_(torch.tensor(float(optimal_c10), dtype=torch.float32, device=self.device))
        self.clear_cache(clear_static_cache=False)
        # Store the optimal scan-frame image; get_reconstructed_image applies output_frame rotation on read.
        self.reconstructed_image = scan_stack[optimal_index]
        self._reconstructed_images[(mode, tuple(sorted(kwargs.items())))] = scan_stack[optimal_index]
        
        if plot_line_search:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(c10_axis_np, quality_scores, color='dodgerblue', s=60, label='Tested Points', zorder=5)
            
            if method == 'fit_parabola':
                c10_smooth = np.linspace(search_range[0], search_range[1], 100)
                fit_scores = a * c10_smooth**2 + b * c10_smooth + c
                ax.plot(c10_smooth, fit_scores, '--', color='gray', label=f'Parabolic Fit ({fit_type})', zorder=4)
                opt_score = a * optimal_c10**2 + b * optimal_c10 + c if a < 0 else max(quality_scores)
            else:
                opt_score = np.max(quality_scores)
            
            ax.scatter([optimal_c10], [opt_score], color='crimson', s=150, marker='*', 
                       label=f'Optimal ($C_{{10}}$ = {optimal_c10:.0f} Ang)', zorder=6)
            
            ax.set_xlabel('Overfocus $C_{10}$ (Ang)', fontsize=12)
            ax.set_ylabel(f'Focus Score ({metric.capitalize()})', fontsize=12)
            ax.set_title(f'Defocus Line Search ({mode}, {metric}, {method})', fontsize=14)
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()
            plt.show()
        
        return self

    def refine_aberrations(self, lr=1, iters=50, metric='normalized_std', plot_recon_every_n_iter=None, save_dir=None, mode='tcBF', **kwargs):
        """ Refine aberration-induced image shifts by minimizing the quality metrics with a negative sign """
        mode = mode.lower()
        optimizer = torch.optim.Adam(self.ab_state.parameters(), lr=lr)

        for i in range(iters):
            optimizer.zero_grad()
            self.clear_cache(clear_static_cache=False)
            summed_img = self.reconstruct(mode=mode, **kwargs)
            self.reconstructed_image = summed_img
            self._reconstructed_images[(mode, tuple(sorted(kwargs.items())))] = summed_img
            loss = -1*QualityMetrics.evaluate(summed_img, metric=metric)
            loss.backward()
            optimizer.step()
            
            if plot_recon_every_n_iter is not None and i % plot_recon_every_n_iter == 0:
                with torch.no_grad():
                    title_str = f'Iter {i}, Loss (-{metric}) : {loss.item():.4g}'
                    ab_dict = self.get_aberrations_dict(layout='flat')
                    desc_str = ", ".join(f"{ab}: {val:.2f}" for ab, val in ab_dict.items())
                    
                    if save_dir is not None:
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = f'{save_dir}/figure_recon_iter_{str(i).zfill(3)}.png'
                    else:
                        save_path = None
                    
                    self.plot_reconstruction(title_str=title_str, desc_str=desc_str, save_path=save_path, mode=mode, **kwargs)

        self.clear_cache(clear_static_cache=False)
        return self

    # Plotting   
    def plot_reconstruction(self, title_str=None, desc_str=None, save_path=None, mode='tcBF', output_frame=None, **kwargs):
        mode = mode.lower()
        
        if title_str is None:
            title_str = f"Reconstructed {mode} and Probe amplitude"
        if desc_str is None:
            ab_dict = self.get_aberrations_dict(output_frame=output_frame, layout='flat')
            desc_str = ", ".join(f"{ab}: {val:.2f}" for ab, val in ab_dict.items())

        img = self.get_reconstructed_image(mode=mode, output_frame=output_frame, **kwargs).detach().cpu().numpy()
        probe = self.get_probe(output_frame=output_frame).abs().detach().cpu().numpy()
        
        fig, axs = plt.subplots(1,2)
        fig.suptitle(title_str, y=0.9)
        fig.text(x=0, y=0.8, s=desc_str)
        axs[0].imshow(img)
        axs[1].imshow(probe)
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_chi_surface(self, plot_probe_phase=False):
        """ Plot the aberration (chi) surface, set 'plot_probe_phase=True' to apply an additional negative sign """
        chi = self.get_chi_surface()
        
        if plot_probe_phase:
            sign = -1
            title_str = 'k-space probe phase (psi = exp(-1j*chi))'
        else:
            sign = 1
            title_str = 'k-space aberration (chi) surface (psi = exp(-1j*chi))'
        
        surface = (self.bf_mask * sign * chi).detach().cpu().numpy()
        
        plt.figure()
        plt.title(title_str)
        plt.imshow(surface)
        plt.colorbar()
        plt.show()
    
    def plot_shift_quiver(self, subsample=None, scale=None, show=True, output_frame=None):
        """
        Plot a quiver vector field of the calculated real-space image shifts
        over the reciprocal-space Bright Field disk.

        output_frame=None        → uses self.output_frame (global default).
        output_frame='scan'      → shifts and k-coords displayed in the scan frame.
        output_frame='detector'  → shifts and k-coords displayed in the detector frame.
        """
        if not hasattr(self, 'kX_centers') or self.kX_centers is None:
            raise RuntimeError("Coordinates not initialized. Run the initialization first.")

        in_scan = self._in_scan_frame(output_frame)
        with torch.no_grad():
            shift_yx_ang = self.get_yx_shifts_ang(output_frame=output_frame)
            kx, ky = self._get_transformed_bf_coordinates(in_scan_frame=in_scan)
        
        kx = kx.cpu().numpy()
        ky = ky.cpu().numpy()
        sx = shift_yx_ang[:,1].cpu().numpy()
        sy = shift_yx_ang[:,0].cpu().numpy()

        Nb = len(kx)
        if subsample is None:
            step = max(1, Nb // 400)
        else:
            step = max(1, Nb // subsample)

        kx_sub, ky_sub = kx[::step], ky[::step]
        sx_sub, sy_sub = sx[::step], sy[::step]

        fig, ax = plt.subplots(figsize=(7, 7))
        k_max = self.max_alpha / 1e3 / self.wavelength
        disk_edge = plt.Circle((0, 0), k_max, color='red', fill=False, 
                               linestyle='--', linewidth=1.5, alpha=0.6, label='BF Mask')
        ax.add_patch(disk_edge)
        
        ax.quiver(kx_sub, ky_sub, sx_sub, sy_sub, 
                  color='midnightblue', angles='xy', scale=scale, alpha=0.85)

        ax.set_aspect('equal')
        ax.set_xlabel(r'$k_x \ (\AA^{-1})$', fontsize=12)
        ax.set_ylabel(r'$k_y \ (\AA^{-1})$', fontsize=12)
        ax.set_title('Image Shifts over BF Disk', fontsize=14)
        
        limit = k_max * 1.15
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        
        if show:
            plt.show()
            
        return fig, ax
