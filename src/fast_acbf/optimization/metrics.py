"""Output quality metrics for focus scoring — higher score means more focused."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


class QualityMetrics:
    """Focus metrics (Laplacian, Sobel, Normalized Image Variance) in maximize direction."""

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

        img = img.unsqueeze(1)  # (N, 1, H, W) for gaussian_blur and conv2d

        if blur:
            img = gaussian_blur(img, kernel_size=blur_kernel_size, sigma=blur_sigma)

        metric = metric.lower()
        if metric == 'laplacian':
            lap_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=img.device)
            lap = F.conv2d(img, lap_kernel, padding=1)
            scores = lap.flatten(start_dim=1).var(dim=1)

        elif metric == 'sobel':
            kx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], device=img.device)
            ky = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], device=img.device)
            gx = F.conv2d(img, kx, padding=1)
            gy = F.conv2d(img, ky, padding=1)
            scores = (gx**2 + gy**2).flatten(start_dim=1).mean(dim=1)

        elif metric == 'normalized_std':
            flat = img.flatten(start_dim=1)
            scores = flat.std(dim=1) / (flat.mean(dim=1) + 1e-6)

        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose 'laplacian', 'sobel', or 'normalized_std'.")

        if scores.shape[0] == 1:
            return scores.squeeze(0)
        return scores
