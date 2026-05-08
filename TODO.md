# TODO

## Tests
- Test the `complex_inversion` reweighting with more simulated datasets, check for the phase shift values
- Revisit upscaling from a clean native-resolution baseline; add dedicated Fourier-padding tests before reintroducing it
- Check if the `refine_register` stub (bf_solver.py L1571) needs a real implementation


## Research / Experiments
- Experiment with Optuna for more comprehensive BO-based optimization for 1st and 2nd order aberrations estimation, see if it's fast enough for real-time pipeline
- Experiment the tcDF track and see if AD-based image optimization (via the `refine_register` stub) is worth implementing vs the current quality-metric approach in `refine_aberrations`


## Refactoring
- Split the native-resolution solver into smaller modules now that final-image caching and upscaling have been removed
- Profile whether `self.vbf_images` can be freed once static FFT caches are built
- Improve the lazy VRAM-friendly path. acBF may still materialize large temporary tensors per chunk.


## Bug Fix
- Fix image rotation to avoid edge cropping when `output_frame = 'detector'`: `tv_rotate` uses `expand=False` by default (see `get_reconstructed_image` L1478, `_sweep_c10_stack` L1271); fix with `expand=True` or pre-padding. Affects PtyRAD export with `scan_rotation != 0`.


## Features
- Add scan affine transformation to the vBF images
- Explore whether we can combine this with Ning's routine for affine transformation


## Pipeline & Integration
- Add the scan rotation fitting / refinement routine with flipping / transpose determination, probably via Optuna as well
- Automate the data loading part
- Wrap this as a py4DGUI plugin and push to Muller repo
- Consider add an export method to output necessary output for downstream PtyRAD (3D object, probe aberrations, meas_flipT, scan_rotation)


# Existing features
- tcBF and acBF reconstructions
- acBF can optionally use 'complex_inversion' algorithm to reweight the spatial frequencies
- Native-resolution reconstruction only; upscaling is temporarily unsupported pending a dedicated Fourier-padding implementation
- Object, aberrations, and probe can be exported in either 'scan' or 'detector' frame. The 'detector' frame is the coordinate system used for PtyRAD.
- brute-force defocus line search (`refine_defocus`): C10 sweep + optional parabola fit; no AD
- aberration optimization with AD (`refine_aberrations`, controlled by `max_order`, `lr`, and `lr_scales`)
- 2 static cache modes (`full`, `lazy`) to balance speed and VRAM usage
- soft aperture with cosine rolloff (`rolloff` param in `reconstruct()`)
- multiple focus quality metrics: `laplacian` (variance of Laplacian), `sobel` (Tenengrad), `normalized_std`
- coordinate transform flags (flipud, fliplr, transpose, rotation_deg) matching PtyRAD's `meas_flipT`; used when computing scan-frame vs detector-frame outputs
- visualization: shift quiver over BF disk (`plot_shift_quiver`), chi surface (`plot_chi_surface`), reconstruction + probe side-by-side (`plot_reconstruction`)
- `get_acBF_diagnostics`: returns transfer power map, support mask, and complex image channels for complex-inversion debugging
- Export 3D defocus volume stack (`get_defocus_stack`)
