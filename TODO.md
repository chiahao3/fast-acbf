# TODO

## High Priority

- **`refine_params` orchestration method** — high-level entry point that sequentially runs
  `refine_flips`, `refine_defocus`, `refine_scan_rotation`, and `refine_aberrations` with
  boolean flags to enable/disable each step. Should accept a shared `mode` and `metric` and
  handle cache invalidation between steps automatically.

- **Improve `normalized_std` metric for acBF** — acBF images can have inverted contrast
  relative to tcBF; `normalized_std` is sign-agnostic but maximizing it can converge to the
  wrong polarity. Options: (a) detect and flip sign before scoring, (b) score on
  `abs(normalized_std)`, (c) add a `contrast_sign` param. Needs investigation with simulated
  data where ground truth polarity is known.

- **Export method** — `get_export()` returning a dict with `object` (reconstructed image),
  `probe` (complex wavefield), and `aberrations` (detector-frame dict ready for PtyRAD),
  optionally with `frame` selector. Consolidates the `get_reconstructed_image` + `get_probe`
  + `get_aberrations_dict` pattern used in notebooks.

- **Benchmark peak VRAM vs. dataset size and max_order** — systematic sweep over
  `(Nb, Ry, Rx, max_order)` for both `full` and `lazy` cache modes; measure peak GPU memory
  and wall-clock time. Intended to produce a reference table for users choosing `cache_mode`.

- **Revisit resampling approaches** — two separate questions that need literature before
  implementing:
  (1) *Method*: real-space bilinear (`tv_rotate`) vs. Fourier-space phase-ramp shifting for
  sub-pixel accuracy and aliasing behavior; read Desheng's and Yue's papers for guidance.
  (2) *Timing*: when to apply the resampling — directly to raw vBF images before caching
  (simpler, paid once) vs. around the FFT operations in the reconstruction hot path (more
  flexible, can defer). The right answer likely depends on whether the resampling is for
  coord-transform correction or for upscaling, and whether it needs to be differentiable.
  Required before re-enabling `upscale > 1`.


## Tests
- Test the `complex_inversion` reweighting with more simulated datasets, check for the phase shift values
- Revisit upscaling from a clean native-resolution baseline; add dedicated Fourier-padding tests before reintroducing it
- Check if the `refine_register` stub (bf_solver.py L1571) needs a real implementation


## Research / Experiments
- Experiment with Optuna for more comprehensive BO-based optimization for 1st and 2nd order aberrations estimation, see if it's fast enough for real-time pipeline
- Experiment the tcDF track and see if AD-based image optimization (via the `refine_register` stub) is worth implementing vs the current quality-metric approach in `refine_aberrations`


## Bug Fix
- Fix image rotation to avoid edge cropping when `output_frame = 'detector'`: `tv_rotate` uses `expand=False` by default (see `get_reconstructed_image` L1478, `_sweep_c10_stack` L1271); fix with `expand=True` or pre-padding. Affects PtyRAD export with `scan_rotation != 0`.


## Features
- Add scan affine transformation to the vBF images
- Explore whether we can combine this with Ning's routine for affine transformation


## Pipeline & Integration
- Automate the data loading part
- Wrap this as a py4DGUI plugin and push to Muller repo


# Existing features
- tcBF and acBF reconstructions
- acBF can optionally use 'complex_inversion' algorithm to reweight the spatial frequencies
- Native-resolution reconstruction only; upscaling is temporarily unsupported pending a dedicated Fourier-padding implementation
- Object, aberrations, and probe can be exported in either 'scan' or 'detector' frame. The 'detector' frame is the coordinate system used for PtyRAD.
- brute-force defocus line search (`refine_defocus`): C10 sweep + optional parabola fit; no AD
- scan rotation line search (`refine_scan_rotation`): sweep over rotation_deg with cache invalidation
- flip/transpose exhaustive search (`refine_flips`): scores all 8 combinations of flipud × fliplr × transpose
- aberration optimization with AD (`refine_aberrations`, controlled by `max_order`, `lr`, and `lr_scales`)
- 2 static cache modes (`full`, `lazy`) to balance speed and VRAM usage
- soft aperture with cosine rolloff (`rolloff` param in `reconstruct()`)
- multiple focus quality metrics: `laplacian` (variance of Laplacian), `sobel` (Tenengrad), `normalized_std`
- coordinate transform flags (flipud, fliplr, transpose, rotation_deg) matching PtyRAD's `meas_flipT`; used when computing scan-frame vs detector-frame outputs
- visualization: shift quiver over BF disk (`plot_shift_quiver`), chi surface (`plot_chi_surface`), reconstruction + probe side-by-side (`plot_reconstruction`), defocus/rotation line searches, flip/transpose search grid
- `get_acBF_diagnostics`: returns transfer power map, support mask, and complex image channels for complex-inversion debugging
- Export 3D defocus volume stack (`get_defocus_stack`)
- Modular package structure: `core/` (portable physics), `pipeline.py` (caching), `optimization/` (metrics + refinement), `vis/` (plotting)
