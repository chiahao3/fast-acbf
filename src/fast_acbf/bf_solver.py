"""Torch-native bright-field solver with tcBF, and acBF reconstruction."""

from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

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
        Calculates the focus score of a 2D image. Higher is more focused.
        """
        
        img = img.unsqueeze(0).unsqueeze(0) # Convert img to (N,B,H,W) for gaussian_blur and con2d
        
        if blur:
            img = gaussian_blur(img, kernel_size=blur_kernel_size, sigma=blur_sigma)
            
        if metric == 'laplacian':
            # Variance of Laplacian
            lap_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=img.device)
            lap = F.conv2d(img, lap_kernel, padding=1)
            return lap.var()
            
        elif metric == 'sobel':
            # Tenengrad (Sum of squared Sobel gradients)
            kx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], device=img.device)
            ky = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], device=img.device)
            gx = F.conv2d(img, kx, padding=1)
            gy = F.conv2d(img, ky, padding=1)
            return (gx**2 + gy**2).mean()
            
        elif metric == 'normalized_std':
            # Normalized Image Std
            return (img.std() / (img.mean() + 1e-6))
            
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose 'laplacian', 'sobel', or 'normalized_std'.")

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
        coord_transform=None
    ):
        """
        Initializes the solver. Dataset loading/parsing is assumed to be handled 
        upstream (e.g., by PtyRAD's Initializer).
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
        self.device = device
        
        # Coordinate transform follows Python indexing convention: y first, then x.
        # We keep the transform intentionally simple and unambiguous for now.
        self.coord_transform = coord_transform or {
            'flipud': False,
            'fliplr': False,
            'transpose': False,
            'rotation_deg': 0.0,
        }
        
        # Placeholders / caches
        self.reconstructed_image = None
        self._reconstructed_images = {}
        self._cache_store = {}

        # Initialize vBF dataset
        self._init_vBF()
        self._init_grid()

    # Internal Methods

    def clear_cache(self, clear_static_cache=True):
        """Clears reconstructed image cache and, optionally, the static tcBF/acBF caches."""
        self.reconstructed_image = None
        self._reconstructed_images = {}
        
        if clear_static_cache:
            self._cache_store = {}

    def _get_transform_flags(self):
        """
        Returns the y/x-orientation flags.
        
        We support the new explicit names ('flipud', 'fliplr', 'transpose') and
        also accept the older aliases ('flip_y', 'flip_x') for compatibility.
        """
        flipud = self.coord_transform.get('flipud', self.coord_transform.get('flip_y', False))
        fliplr = self.coord_transform.get('fliplr', self.coord_transform.get('flip_x', False))
        transpose = self.coord_transform.get('transpose', False)
        rotation_deg = self.coord_transform.get('rotation_deg', 0.0)
        
        if abs(rotation_deg) > 1e-9:
            logger.warning("rotation_deg is currently ignored in BFSolver. Use flipud/fliplr/transpose for now.")
        
        return flipud, fliplr, transpose

    def _get_transformed_bf_coordinates(self):
        """
        Returns transformed reciprocal-space BF coordinates as (kX, kY).
        
        Internally we think in y/x order when applying transpose / flips, then
        convert back to the explicit (kX, kY) tensors expected by the analytical math.
        """
        # Start from (y, x) ordering because that matches the rest of the image code.
        ky = self.kY_centers.clone()
        kx = self.kX_centers.clone()
        flipud, fliplr, transpose = self._get_transform_flags()
        
        if transpose:
            ky, kx = kx, ky
        if flipud:
            ky = -ky
        if fliplr:
            kx = -kx
        
        return kx, ky

    def _get_transformed_k_grids(self):
        """Returns transformed reciprocal-space grids as (kX_grid, kY_grid)."""
        ky = self.kY_grid.clone()
        kx = self.kX_grid.clone()
        flipud, fliplr, transpose = self._get_transform_flags()
        
        if transpose:
            ky, kx = kx, ky
        if flipud:
            ky = -ky
        if fliplr:
            kx = -kx
        
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

    def _build_tcBF_cache(self, upscale=1, chunk_size=64):
        """Pre-computes and chunks the analytical shift basis, spatial grids, and FFTs."""
        
        # 1. Native GPU Upsampling
        raw_stack = self.vbf_images
        if upscale > 1:
            im_stack = raw_stack.repeat_interleave(upscale, dim=-2).repeat_interleave(upscale, dim=-1)
        else:
            im_stack = raw_stack
            
        Nb, Ny, Nx = im_stack.shape

        # 2. Setup transformed k-shifts
        # Note: kX / kY are returned explicitly, but the transform itself is applied in y/x convention.
        kX_full, kY_full = self._get_transformed_bf_coordinates()
        
        # 3. Base Spatial Frequencies (For the sub-pixel Fourier phase ramp)
        dx = self.scan_step_size / upscale
        dy = self.scan_step_size / upscale
        qx_grid = torch.fft.fftfreq(Nx, d=dx, device=self.device).view(1, 1, Nx)
        qy_grid = torch.fft.fftfreq(Ny, d=dy, device=self.device).view(1, Ny, 1)
        
        cache = []
        
        for i in range(0, Nb, chunk_size):
            end = min(i + chunk_size, Nb)
            
            kX_chunk = kX_full[i:end]
            kY_chunk = kY_full[i:end]
            img_chunk = im_stack[i:end]
            
            # Generate unweighted analytic shift basis matrix [Num_Coeffs, ChunkSize]
            b_dx, b_dy = generate_shift_basis(self.ab_state.order_keys, kX_chunk, kY_chunk, self.wavelength)
            
            # Pre-FFT the raw image slice
            img_fft = torch.fft.fft2(img_chunk, dim=(-2, -1))
            
            cache.append({
                'b_dx': b_dx,
                'b_dy': b_dy,
                'img_fft': img_fft
            })
            
        return cache, qx_grid, qy_grid, (Ny, Nx)

    def _build_acBF_cache(self, upscale=1, rolloff=0, chunk_size=64):
        """Pre-computes and chunks all static geometry, soft apertures, and FFTs."""
        
        # 1. Native GPU Upsampling
        raw_stack = self.vbf_images
        if upscale > 1:
            im_stack = raw_stack.repeat_interleave(upscale, dim=-2).repeat_interleave(upscale, dim=-1)
        else:
            im_stack = raw_stack
            
        Nb, Ny, Nx = im_stack.shape

        # 2. Setup transformed k-shifts
        # Here kX/kY keep their explicit meaning even though the transform is applied in y/x convention.
        kX_full, kY_full = self._get_transformed_bf_coordinates()
        kX_full = kX_full.view(Nb, 1, 1)
        kY_full = kY_full.view(Nb, 1, 1)
        
        # 3. Base Spatial Frequencies
        dx = self.scan_step_size / upscale
        dy = self.scan_step_size / upscale
        kx_base = torch.fft.fftfreq(Nx, d=dx, device=self.device).view(1, 1, Nx)
        ky_base = torch.fft.fftfreq(Ny, d=dy, device=self.device).view(1, Ny, 1)
        
        cache = []
        
        for i in range(0, Nb, chunk_size):
            end = min(i + chunk_size, Nb)
            
            kxt = kX_full[i:end]
            kyt = kY_full[i:end]
            img_chunk = im_stack[i:end]
            
            # Kinematic Grids (q-k and q+k mapped to NumPy logic)
            kx_t, ky_t = kx_base + kxt, ky_base + kyt
            kx_mt, ky_mt = kx_base - kxt, ky_base - kyt
            
            # Soft Apertures
            alpha_t = torch.sqrt(kx_t**2 + ky_t**2) * self.wavelength
            ap_t = make_soft_aperture_torch(alpha_t, self.max_alpha, rolloff)
            
            alpha_mt = torch.sqrt(kx_mt**2 + ky_mt**2) * self.wavelength
            ap_mt = make_soft_aperture_torch(alpha_mt, self.max_alpha, rolloff)
            
            # Polynomial Basis Tensors
            b_tr = generate_aberration_basis(self.max_order, self.ab_state.order_keys, kxt, kyt, self.wavelength)
            b_t  = generate_aberration_basis(self.max_order, self.ab_state.order_keys, kx_t, ky_t, self.wavelength)
            b_mt = generate_aberration_basis(self.max_order, self.ab_state.order_keys, -kx_mt, -ky_mt, self.wavelength)
            
            # Pre-FFT the raw image slice
            img_fft = torch.fft.fft2(img_chunk, dim=(-2, -1))
            
            cache.append({
                'ap_t': ap_t, 'ap_mt': ap_mt,
                'b_tr': b_tr, 'b_t': b_t, 'b_mt': b_mt,
                'img_fft': img_fft
            })
            
        return cache, (Ny, Nx)

    def _get_tcBF_cache(self, upscale=1, chunk_size=64):
        key = ('tcBF', upscale, chunk_size)
        
        if key not in self._cache_store:
            self._cache_store[key] = self._build_tcBF_cache(upscale=upscale, chunk_size=chunk_size)
            
        return self._cache_store[key]

    def _get_acBF_cache(self, upscale=1, rolloff=0, chunk_size=64):
        key = ('acBF', upscale, rolloff, chunk_size)
        
        if key not in self._cache_store:
            self._cache_store[key] = self._build_acBF_cache(upscale=upscale, rolloff=rolloff, chunk_size=chunk_size)
            
        return self._cache_store[key]

    def _get_tcBF_from_cache(self, cache, qx_grid, qy_grid, out_shape):
        """
        Ultra-lean AD forward pass for tcBF. 
        Calculates exact analytical shifts instantly via Einstein summation.
        """
        neg_two_pi_j = torch.tensor(-2.0j * torch.pi, dtype=torch.complex64, device=self.device)
        tcBF_total = torch.zeros(out_shape, dtype=torch.float32, device=self.device)
        
        # Extract the 1D coefficient tensor (Shape: Num_Coeffs)
        C = self.ab_state.get_flat_coeffs()
        
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

    def _get_acBF_from_cache(self, cache, out_shape):
        """
        Ultra-lean AD forward pass using Tensorized Einsum.
        """
        j1 = torch.tensor(1.0j, dtype=torch.complex64, device=self.device)
        acBF_total = torch.zeros(out_shape, dtype=torch.float32, device=self.device)
        
        # Extract the 1D coefficient tensor (Shape: Num_Coeffs)
        C = self.ab_state.get_flat_coeffs()
        
        for chunk in cache:
            # 1. Einsum: Matrix multiply the coefficients across the entire basis block instantly!
            # 'k' = Coeff index, 'b' = Batch, 'x' = Nx, 'y' = Ny
            chi_tr_az = torch.einsum('k, kbxy -> bxy', C, chunk['b_tr'])
            chi_t     = torch.einsum('k, kbxy -> bxy', C, chunk['b_t'])
            chi_mt    = torch.einsum('k, kbxy -> bxy', C, chunk['b_mt'])
            
            # 2. Rose CTF Interference Waves
            # Originally we were doing phasor = exp(-i*angle(ctf_t))
            # So phasor = conj(ctf_t) / |ctf_t|
            # Since ctf_t = -conj(0.5*i*D), we get phasor = -0.5*i*D / |0.5*i*D|
            # So phasor is simply just -i * D/|D|, and D/|D| = sgn(D)
            term_mt = chunk['ap_mt'] * torch.exp(-j1 * (chi_tr_az - chi_mt))
            term_t  = chunk['ap_t']  * torch.exp(j1 * (chi_tr_az - chi_t))
            D = term_mt - term_t
            phasor = (-j1) * torch.sgn(D) 
            
            # 3. Apply phase correction and IFFT
            F_corr = chunk['img_fft'] * phasor
            acBF_total += torch.sum(torch.fft.ifft2(F_corr, dim=(-2, -1)).real, dim=0)
            
        return acBF_total

    # Public methods
    # Getter and Printing
    def get_aberrations_dict(self, notation='krivanek', style='cartesian', layout='nested'):
        return Aberrations(self.ab_state.get_cartesian_dict()).export(notation=notation, style=style, layout=layout)

    def print_aberrations(self):
        print(Aberrations(self.get_aberrations_dict()))

    def get_chi_surface(self):
        """Return aberration surface chi, note that psi = exp(-1j*chi) so there's a negative sign between chi and k-space probe phase"""
        kX_grid, kY_grid = self._get_transformed_k_grids()
        
        # Build the same polynomial basis structure used in the cached acBF path,
        # then contract it with the flat aberration coefficients directly.
        chi_basis = generate_aberration_basis(
            self.max_order,
            self.ab_state.order_keys,
            kX_grid,
            kY_grid,
            self.wavelength,
        )
        
        # chi_basis has shape (Num_Coeffs, Ny, Nx) for the full reciprocal-space grid.
        coeffs = self.ab_state.get_flat_coeffs()
        chi = torch.einsum('k,kij->ij', coeffs, chi_basis)
        
        return chi

    def get_yx_shifts_ang(self):
        """ Return shifts in Ang as (Nb, 2) tensor, each row is (shift_y, shift_x) """
        kX_centers, kY_centers = self._get_transformed_bf_coordinates()
        
        # Build the same analytical shift basis used in the cached tcBF path,
        # then contract it with the flat aberration coefficients directly.
        b_dx, b_dy = generate_shift_basis(
            self.ab_state.order_keys,
            kX_centers,
            kY_centers,
            self.wavelength,
        )
        
        coeffs = self.ab_state.get_flat_coeffs()
        shift_x_ang = torch.einsum('k,kb->b', coeffs, b_dx)
        shift_y_ang = torch.einsum('k,kb->b', coeffs, b_dy)
        
        return torch.stack([shift_y_ang, shift_x_ang], dim=-1) # Return shape (Nb, 2)
    
    def get_yx_shifts_px(self):
        """ Return shifts in real-space px as (Nb, 2) tensor, each row is (shift_y, shift_x) """
        return self.get_yx_shifts_ang() / self.scan_step_size

    def reconstruct(self, mode='tcBF', **kwargs):
        """
        Unified reconstruction entry point.
        
        Notes:
            - tcBF is the default mode.
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
            cache, out_shape = self._get_acBF_cache(upscale=upscale, rolloff=rolloff, chunk_size=chunk_size)
            return self._get_acBF_from_cache(cache, out_shape)
        
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Please choose between 'tcBF' and 'acBF'.")

    def get_reconstructed_image(self, mode='tcBF', **kwargs):
        cache_key = (mode.lower(), tuple(sorted(kwargs.items())))
        
        if cache_key not in self._reconstructed_images:
            self._reconstructed_images[cache_key] = self.reconstruct(mode=mode, **kwargs)
            
        self.reconstructed_image = self._reconstructed_images[cache_key]
        return self.reconstructed_image

    def get_tcBF(self, **kwargs):
        return self.get_reconstructed_image(mode='tcBF', **kwargs)

    def get_acBF(self, **kwargs):
        return self.get_reconstructed_image(mode='acBF', **kwargs)
    
    def get_probe(self):
        probe = make_probe_from_chi(self.get_chi_surface(), self.bf_mask)
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
        # 1. Setup the test array
        min_def, max_def = min(search_range), max(search_range)
        search_range = (min_def, max_def)
        print(f"Starting defocus line search: {num_points} points between {search_range[0]} and {search_range[1]} Ang")
        
        c10_tests = np.linspace(search_range[0], search_range[1], num_points)
        quality_scores = []
        temp_images = []
        optimal_index = 0
        
        # 2. Evaluate each point
        for c10 in c10_tests:
            with torch.no_grad():
                self.ab_state.coeffs['C_1_0'].copy_(torch.tensor(float(c10), dtype=torch.float32, device=self.device))
            
            with torch.no_grad():
                self.clear_cache(clear_static_cache=False)
                summed_img = self.reconstruct(mode=mode, **kwargs)
                score = QualityMetrics.evaluate(summed_img, metric=metric, blur=blur, blur_kernel_size=blur_kernel_size, blur_sigma=blur_sigma).item()
                quality_scores.append(score)
                temp_images.append(summed_img)
        
        if method == 'fit_parabola':
            coeffs = np.polyfit(c10_tests, quality_scores, 2)
            a, b, c = coeffs
            
            if a < 0:
                optimal_c10 = -b / (2 * a)
                optimal_c10 = np.clip(optimal_c10, search_range[0], search_range[1])
                fit_type = "Parabolic vertex"
                optimal_index = int(np.argmin(np.abs(c10_tests - optimal_c10)))
            else:
                optimal_index = int(np.argmax(quality_scores))
                optimal_c10 = c10_tests[optimal_index]
                fit_type = "Discrete max (fit inverted)"
                
        elif method == 'max':
            optimal_index = int(np.argmax(quality_scores))
            optimal_c10 = c10_tests[optimal_index]
        else:
            raise ValueError(f"Unsupported method: {method}, please choose between 'fit_parabola' or 'max'")
            
        print(f"Optimal C10 found at {optimal_c10:.2f} Ang ({method})")
        
        with torch.no_grad():
            self.ab_state.coeffs['C_1_0'].copy_(torch.tensor(float(optimal_c10), dtype=torch.float32, device=self.device))
        self.clear_cache(clear_static_cache=False)
        self.reconstructed_image = temp_images[optimal_index]
        self._reconstructed_images[(mode.lower(), tuple(sorted(kwargs.items())))] = temp_images[optimal_index]
        
        if plot_line_search:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(c10_tests, quality_scores, color='dodgerblue', s=60, label='Tested Points', zorder=5)
            
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
        optimizer = torch.optim.Adam(self.ab_state.parameters(), lr=lr)

        for i in range(iters):
            optimizer.zero_grad()
            self.clear_cache(clear_static_cache=False)
            summed_img = self.reconstruct(mode=mode, **kwargs)
            self.reconstructed_image = summed_img
            self._reconstructed_images[(mode.lower(), tuple(sorted(kwargs.items())))] = summed_img
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
    def plot_reconstruction(self, title_str=None, desc_str=None, save_path=None, mode='tcBF', **kwargs):
        if title_str is None:
            title_str = f"Reconstructed {mode} and Probe amplitude"
        if desc_str is None:
            ab_dict = self.get_aberrations_dict(layout='flat')
            desc_str = ", ".join(f"{ab}: {val:.2f}" for ab, val in ab_dict.items())
            
        img = self.get_reconstructed_image(mode=mode, **kwargs).detach().cpu().numpy()
        probe = self.get_probe().abs().detach().cpu().numpy()
        
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
    
    def plot_shift_quiver(self, subsample=None, scale=None, show=True):
        """
        Plots a quiver vector field of the calculated real-space image shifts 
        over the reciprocal-space Bright Field disk.
        """
        if not hasattr(self, 'kX_centers') or self.kX_centers is None:
            raise RuntimeError("Coordinates not initialized. Run the initialization first.")

        with torch.no_grad():
            shift_yx_ang = self.get_yx_shifts_ang()
            kx, ky = self._get_transformed_bf_coordinates()
        
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
