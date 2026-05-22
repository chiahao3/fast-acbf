# TODO

## High Priority

- **Improve quality metrics** 
  - acBF images can have mean jittering around zero, making `normalized_std` useless
  - acBF can also introduce sharp fringes from incorrect aberrations

- **Export method** — `get_export()` returning a dict with `object` (reconstructed image),
  `probe` (complex wavefield), and `aberrations` (detector-frame dict ready for PtyRAD),
  optionally with `frame` selector. Consolidates the `get_reconstructed_image` + `get_probe`
  + `get_aberrations_dict` pattern used in notebooks.

- **Revisit scan-coordinate resampling approaches** — future affine/coordinate-transform
  correction still needs literature review: compare real-space interpolation (`grid_sample`
  / `tv_rotate`) with Fourier-space phase-ramp shifting for sub-pixel accuracy, aliasing
  behavior, and differentiability. Real-space vBF upscaling is already handled in
  `BFPreparer`; this item is about geometric correction quality, not enabling upscale.


## Tests
- Test the `complex_inversion` reweighting with more simulated datasets, check for the phase shift values


## Research / Experiments
- Experiment with Optuna for more comprehensive BO-based optimization for 1st and 2nd order aberrations estimation, see if it's fast enough for real-time pipeline
- Experiment the tcDF track 


## Features
- Add scan affine transformation to the vBF images
- Explore whether we can combine this with Ning's routine for affine transformation


## Pipeline & Integration
- Add asynchronous disk prefetch for the lazy disk strategies. `PipelineManager`
  now provides coherent speed/balanced/memory routes for data larger than VRAM
  or RAM, but reconstruction can still stall while waiting for each disk read.
- Decide whether benchmark result files should be versioned as historical
  artifacts or regenerated after the pipeline rewrite.


# Existing features
- tcBF and acBF reconstructions
- acBF can optionally use 'complex_inversion' algorithm to reweight the spatial frequencies
- Real-space vBF upscaling (`nearest`/`bilinear`) is handled in `BFPreparer` before ImageFFT caching
- Object, aberrations, and probe can be exported in either 'scan' or 'detector' frame. The 'detector' frame is the coordinate system used for PtyRAD.
- brute-force defocus line search (`refine_defocus`): C10 sweep + optional parabola fit; no AD
- scan rotation line search (`refine_scan_rotation`): sweep over rotation_deg with cache invalidation
- flip/transpose exhaustive search (`refine_flips`): scores all 8 combinations of flipud × fliplr × transpose
- aberration optimization with AD (`refine_aberrations`, controlled by `max_order`, `lr`, and `lr_scales`)
- Pipeline presets (`speed`, `balanced`, `memory`) resolve storage, fill timing,
  and BF extraction into coherent routes. Explicit `imagefft_storage`,
  `imagefft_fill`, and `extractor_strategy` overrides are still respected when
  feasible.
- 3 ImageFFT storage modes (`device`, `host`, `none`) to trade off speed vs. memory: `device` caches ImageFFT on the compute device, `host` caches ImageFFT in RAM, `none` streams with no persistent FFT cache
- ImageFFT fill policies (`precompute`, `lazy`, `on_the_fly`): presets choose
  `precompute` whenever persistent ImageFFT storage is selected, while
  `memory` defaults to `on_the_fly`.
- soft aperture with cosine rolloff (`rolloff` param in `reconstruct()`)
- multiple focus quality metrics: `laplacian` (variance of Laplacian), `sobel` (Tenengrad), `normalized_std`
- coordinate transform flags (flipud, fliplr, transpose, rotation_deg) matching PtyRAD's `meas_flipT`; used when computing scan-frame vs detector-frame outputs
- visualization: shift quiver over BF disk (`plot_shift_quiver`), chi surface (`plot_chi_surface`), reconstruction + probe side-by-side (`plot_reconstruction`), defocus/rotation line searches, flip/transpose search grid
- `get_acBF_diagnostics`: returns transfer power map, support mask, and complex image channels for complex-inversion debugging
- Export 3D defocus volume stack (`get_defocus_stack`)
- Smart BF extraction policy: `device_mask` is a whole-pass precompute route,
  `host_mask` materializes raw data in RAM when it fits, and disk strategies
  stream lazy sources based on HDF5/Zarr chunk layout. Can be overridden via
  `extractor_strategy` on `BFSolver`.
- Exhaustive pipeline matrix checker (`scripts/check_pipeline_matrix.py`) that
  verifies all-auto routes degrade instead of failing and catches wasteful
  strategy/storage/fill combinations.
- Modular package structure: `core/` (portable physics), `recon/pipeline.py` (pipeline policy), `optimization/` (metrics + refinement), `vis/` (plotting)
- py4D-browser-fast-acbf as a interactive GUI (currently only in Muller group Github repo @ Cornell)
