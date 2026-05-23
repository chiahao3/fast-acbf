"""Output quality metrics for focus scoring — higher score means more focused."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


class QualityMetrics:
    """Focus metrics (Laplacian, Sobel, Normalized Image Variance) in maximize direction."""

    @staticmethod
    def _center_crop(img: torch.Tensor, crop_fraction: float) -> torch.Tensor:
        """Return the centered spatial crop, preserving leading dimensions."""
        crop_fraction = float(crop_fraction)
        if not (0.0 < crop_fraction <= 1.0):
            raise ValueError(f"crop_fraction must be in (0, 1], got {crop_fraction}.")

        h, w = img.shape[-2:]
        crop_h = max(1, round(h * crop_fraction))
        crop_w = max(1, round(w * crop_fraction))
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        return img[..., y0:y0 + crop_h, x0:x0 + crop_w]

    @classmethod
    def evaluate(
        cls,
        img: torch.Tensor,
        metric='sobel',
        blur=False,
        blur_kernel_size=5,
        blur_sigma=1,
        center_crop: bool = True,
        crop_fraction: float = 0.5,
    ) -> torch.Tensor:
        """
        Calculates focus score(s). Higher is more focused.

        Accepts either:
            - 2D image of shape (Ny, Nx), returning a scalar tensor
            - 3D stack of shape (Nz, Ny, Nx), returning a 1D tensor of length Nz

        By default, scores are computed on the central half-width/half-height
        region (one quarter of the image area), which avoids edge artifacts from
        padded FFT reconstruction guard bands.
        """
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim != 3:
            raise ValueError(f"img must be 2D or 3D, got shape {tuple(img.shape)}.")

        img = img.unsqueeze(1)  # (N, 1, H, W) for gaussian_blur and conv2d

        if center_crop:
            crop_fraction = float(crop_fraction)
            if not (0.0 < crop_fraction <= 1.0):
                raise ValueError(f"crop_fraction must be in (0, 1], got {crop_fraction}.")

        if blur:
            img = gaussian_blur(img, kernel_size=blur_kernel_size, sigma=blur_sigma)

        metric = metric.lower()
        if metric == 'laplacian':
            lap_kernel = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=img.device)
            lap = F.conv2d(img, lap_kernel, padding=1)
            if center_crop:
                lap = cls._center_crop(lap, crop_fraction)
            scores = lap.flatten(start_dim=1).var(dim=1)

        elif metric == 'sobel':
            kx = torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]], device=img.device)
            ky = torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]], device=img.device)
            gx = F.conv2d(img, kx, padding=1)
            gy = F.conv2d(img, ky, padding=1)
            if center_crop:
                gx = cls._center_crop(gx, crop_fraction)
                gy = cls._center_crop(gy, crop_fraction)
            scores = (gx**2 + gy**2).flatten(start_dim=1).mean(dim=1)

        elif metric == 'normalized_std':
            if center_crop:
                img = cls._center_crop(img, crop_fraction)
            flat = img.flatten(start_dim=1)
            scores = flat.std(dim=1) / (flat.mean(dim=1) + 1e-6)

        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose 'laplacian', 'sobel', or 'normalized_std'.")

        if scores.shape[0] == 1:
            return scores.squeeze(0)
        return scores
