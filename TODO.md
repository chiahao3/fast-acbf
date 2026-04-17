# TODO

## Tests
- Test the `complex_inversion` reweighting with more simulated datasets, check for the phase shift values
- Check if the cached reconstructed image is stale in any of the mode configurations
- Check if the order of upsampling actually makes a difference
- Check if the `refine_register` stub (bf_solver.py L1571) needs a real implementation


## Research / Experiments
- Experiment with Optuna for more comprehensive BO-based optimization for 1st and 2nd order aberrations estimation, see if it's fast enough for real-time pipeline
- Experiment the tcDF track and see if AD-based image optimization (via the `refine_register` stub) is worth implementing vs the current quality-metric approach in `refine_aberrations`


## Refactoring
- Clean up the upscale paths, ideally keeping only the k-space padding one
- `self.vbf_images` is never offloaded after cache build (the `full_gpu` docstring says it should be). Confirm whether it can be freed once `img_fft` is cached; check both acBF and tcBF paths.
- Improve the VRAM-friendly paths a bit more. acBF path seems to still contain a step that will materialize massive tensors simultaneously.


## Bug Fix
- Fix image rotation to avoid edge cropping when `output_frame = 'detector'`: `tv_rotate` uses `expand=False` by default (see `get_reconstructed_image` L1478, `_sweep_c10_stack` L1271); fix with `expand=True` or pre-padding. Affects PtyRAD export with `scan_rotation != 0`.


## Features
- Add scan affine transformation to the vBF images
- Explore whether we can combine this with Ning's routine for affine transformation


## Pipeline & Integration
- Add the scan rotation fitting / refinement routine with flipping / transpose determination, probably via Optuna as well
- Automate the data loading and calibration part (i.e., dk calculation)
- Wrap this as a py4DGUI plugin and push to Muller repo
- Consider add an export method to output necessary output for downstream PtyRAD (3D object, probe aberrations, meas_flipT, scan_rotation)


# Existing features
- tcBF and acBF reconstructions
- acBF can optionally use 'complex_inversion' algorithm to reweight the spatial frequencies
- upscale with arbitrary scaling factor, can upsample in real space (or equivalently pad in k-space), and can defer the upscale timing to right before iFFT
- Object, aberrations, and probe can be exported in either 'scan' or 'detector' frame. The 'detector' frame is the coordinate system used for PtyRAD.
- brute-force defocus line search (`refine_defocus`): C10 sweep + optional parabola fit; no AD
- aberration optimization with AD (`refine_aberrations`, controlled by `max_order`, `lr`, and `lr_scales`)
- 3 different cache modes (`full_gpu`, `fft_gpu`, `full_cpu`) to balance VRAM usage and speed; `full_cpu` uses async H2D prefetch (CUDA stream overlap)
- soft aperture with cosine rolloff (`rolloff` param in `reconstruct()`)
- multiple focus quality metrics: `laplacian` (variance of Laplacian), `sobel` (Tenengrad), `normalized_std`
- coordinate transform flags (flipud, fliplr, transpose, rotation_deg) matching PtyRAD's `meas_flipT`; used when computing scan-frame vs detector-frame outputs
- visualization: shift quiver over BF disk (`plot_shift_quiver`), chi surface (`plot_chi_surface`), reconstruction + probe side-by-side (`plot_reconstruction`)
- `get_acBF_diagnostics`: returns transfer power map, support mask, and complex image channels for complex-inversion debugging
- Export 3D defocus volume stack (`get_defocus_stack`)