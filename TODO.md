# TODO

## High Priority

- **Improve quality metrics** 
  - acBF images can have mean jittering around zero, making `normalized_std` useless
  - acBF can also introduce sharp fringes from incorrect aberrations

- **Export method** — `get_export()` returning a dict with `object` (reconstructed image),
  `probe` (complex wavefield), and `aberrations` (detector-frame dict ready for PtyRAD),
  optionally with `frame` selector. Consolidates the `get_reconstructed_image` + `get_probe`
  + `get_aberrations_dict` pattern used in notebooks.

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


## Research / Experiments
- Experiment with Optuna for more comprehensive BO-based optimization for 1st and 2nd order aberrations estimation, see if it's fast enough for real-time pipeline
- Experiment the tcDF track 


## Features
- Add scan affine transformation to the vBF images
- Explore whether we can combine this with Ning's routine for affine transformation


## Pipeline & Integration
- Setup a dataloader path so we can run datasets larger than VRAM, or even RAM
  - `cache_mode='on_the_fly'` now streams directly from disk at practical speeds (see lazy
    disk-streaming in Existing features). Remaining gap: reconstruction still stalls while
    waiting for each disk read — prefetching would hide this latency.


# Existing features
- tcBF and acBF reconstructions
- acBF can optionally use 'complex_inversion' algorithm to reweight the spatial frequencies
- Native-resolution reconstruction only; upscaling is temporarily unsupported pending a dedicated Fourier-padding implementation
- Object, aberrations, and probe can be exported in either 'scan' or 'detector' frame. The 'detector' frame is the coordinate system used for PtyRAD.
- brute-force defocus line search (`refine_defocus`): C10 sweep + optional parabola fit; no AD
- scan rotation line search (`refine_scan_rotation`): sweep over rotation_deg with cache invalidation
- flip/transpose exhaustive search (`refine_flips`): scores all 8 combinations of flipud × fliplr × transpose
- aberration optimization with AD (`refine_aberrations`, controlled by `max_order`, `lr`, and `lr_scales`)
- 3 cache modes (`device`, `host`, `on_the_fly`) to trade off speed vs. memory: `device` caches everything on GPU, `host` caches in RAM, `on_the_fly` streams from disk with no caching
- soft aperture with cosine rolloff (`rolloff` param in `reconstruct()`)
- multiple focus quality metrics: `laplacian` (variance of Laplacian), `sobel` (Tenengrad), `normalized_std`
- coordinate transform flags (flipud, fliplr, transpose, rotation_deg) matching PtyRAD's `meas_flipT`; used when computing scan-frame vs detector-frame outputs
- visualization: shift quiver over BF disk (`plot_shift_quiver`), chi surface (`plot_chi_surface`), reconstruction + probe side-by-side (`plot_reconstruction`), defocus/rotation line searches, flip/transpose search grid
- `get_acBF_diagnostics`: returns transfer power map, support mask, and complex image channels for complex-inversion debugging
- Export 3D defocus volume stack (`get_defocus_stack`)
- Smart disk-streaming for `on_the_fly` mode: automatically selects the best read strategy based on how the file is stored on disk, up to ×1283 faster than the naive approach. Can be overridden via `lazy_read_mode` on `Dataset4D` and `BFSolver`.
- Modular package structure: `core/` (portable physics), `pipeline.py` (caching), `optimization/` (metrics + refinement), `vis/` (plotting)
- py4D-browser-fast-acbf as a interactive GUI (currently only in Muller group Github repo @ Cornell)
