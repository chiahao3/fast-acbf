# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- **No PtyRAD dependency.** fast-acbf is now a standalone library. The pieces it used from
  PtyRAD are ported from PtyRAD v1.0.0 and must be kept in sync with it: `Aberrations` in
  `fast_acbf.core.ptyrad_aberrations` (verbatim copy of `ptyrad.optics.aberrations`),
  `fftshift2` / `ifftshift2` / `torch_phasor` in `fast_acbf.core.functional` and `mfft2` in
  `fast_acbf.vis.plotting`, and `get_wavelength_ang` / `guess_radius_of_bright_field_disk`
  (with the physical constants) in `fast_acbf.core.calibration`. Aberration notation, units
  and output frames are unchanged
- `BFSolver` raises a clear error when given a `.raw` path (it needs the shape; use
  `Dataset4D.from_raw`) instead of failing inside the Zarr loader
### Added
- `Dataset4D.from_raw(path, scan_shape, detector_shape, *, dtype, offset, gap)` opens
  headerless frame files such as EMPAD `.raw` (1024-byte gap per frame by default) as a lazy
  memory map, with a file-size check; same `materialize` / `normalize` options as `from_hdf5`
- `get_wavelength_ang` and `guess_radius_of_bright_field_disk` are exported from `fast_acbf`
- `tests/test_ptyrad_port.py` compares the ported code with the installed PtyRAD (source and
  behaviour); skipped when PtyRAD is not installed

## [0.8.0] - 2026-09-21
### Added
- Add `scripts/benchmark_tcbf_variants.py`, which sweeps scan size and upscale over the baseline implementation and each change in isolation, plus artifacts and a report under `benchmarks/`
### Changed
- **tcBF reconstruction is up to 4.9x faster and uses 3.4x less transient VRAM.** Measured against 0.7.0 at N_BF=797, `chunk_size=64`, device-resident FFT store, RTX 5000 Ada: 3.3x at `R=256`, 4.9x at `R=512`, 4.7x at `R=1024`. Reconstructions at `R<=128` are unchanged, being kernel-launch bound rather than bandwidth bound
- Accumulate tcBF and acBF reconstructions in Fourier space and inverse-transform once, instead of inverse-transforming every shifted virtual image before summing over the detector axis (`ifft2` and `Re(.)` are linear in that axis), turning `N_BF` inverse FFTs into one
- Build the tcBF shift operator as two 1D phase factors rather than a full 2D grid, and apply both factors plus the detector sum as a single `einsum`, so no `(chunk_size, Ry, Rx)` intermediate is materialized
- acBF keeps only the Fourier accumulation, worth ~1.05x since `compute_transfer` dominates its runtime, but it drops the inverse-FFT buffer from peak VRAM
- Outputs match the previous implementation to float32 roundoff in both the image (6.1e-07 relative) and `d(loss)/d(coeffs)` (2.5e-07), the gradient mattering because `reconstruct_tcbf` is the AD forward pass behind `refine_aberrations`

## [0.7.0] - 2026-07-30
### Added
- Add LGPL-3.0 `LICENSE` and PyPI packaging metadata (license, missing dependencies, GitHub Actions release workflow) in preparation for the public PyPI release
- Add Brent's method as an adaptive line search option for `refine_defocus` and `refine_scan_rotation`
### Changed
- Normalize line endings to LF via `.gitattributes`/`.editorconfig`
- Declare `psutil` and `h5py` as explicit dependencies instead of relying on transitive installs via `ptyrad`
- Document the private-data requirements in regression tests and large-dataset benchmark scripts for external contributors
### Fixed
- `refine_register` now raises `NotImplementedError` instead of silently no-op'ing (it was an unfinished stub)

## [0.6.0] - 2026-05-25
### Added
- Add `zero_insert` upsampling method to `BFPreparer`, now the default upsampling mode for tcBF (`nearest` remains default for acBF), with a reweighting map to normalize sub-pixel coverage
### Changed
- Raise a clear `ValueError` when `zero_insert` is combined with acBF (physically incompatible); simplify the reweighting logic
- Update tests and notebook for the new `zero_insert` defaults

## [0.5.1] - 2026-05-22
### Added
- Add `metric_kwargs` to refinement methods for finer control of quality-metric evaluation (e.g. `center_crop`)
- Add center-crop support and validation to `QualityMetrics`
### Changed
- Switch the default image quality metric to `'sobel'`

## [0.5.0] - 2026-05-22
### Added
- Add `BFPreparer` (mirror-pad + Tukey window) with FFT zero-pad upscale and `pad_width` pipeline support
- Expose `pad_width`, `fov`, and `upscale` on `BFSolver`; update all output getters accordingly
- Support prepared-grid overrides in refinement; expose active and raw scan state; statefully rebuild the prepared `ImageFFT`
### Changed
- Move vBF preparation into `BFPreparer`; choose the `round(N*U)` crop semantic and add `get_pixel_size` metadata
- Fix correctness/safety issues surfaced across two rounds of code review
- Update the notebook to demonstrate the real-space upscale workflow
### Removed
- Remove the stale Fourier zero-pad upscale path and other upscale compatibility shims

## [0.4.0] - 2026-05-18
### Added
- Add `PipelineManager` with `speed`/`balanced`/`memory` presets to consistently resolve `pipeline`/`imagefft_storage`/`imagefft_fill`/`extractor_strategy` configuration, plus a pipeline policy matrix checker
- Add `BFExtractor` disk-read strategies; replace `ImageFFTProvider` with `ImageFFT`
### Changed
- Wire `BFSolver`/`BFReconstructor` to the new pipeline manager; refactor `Dataset4D`'s raw-source API
- Fix zarr materialize lifecycle; guard lazy device-mask auto-selection; enforce whole-pass device-mask precompute
- Update tests, benchmarks, notebook, and docs for the pipeline rewrite

## [0.3.0] - 2026-05-17
### Added
- Add `Dataset4D` and `ImageFFTProvider` as a dedicated data layer, decoupling data ingestion from reconstruction (v2 architecture rewrite)
- Add `lazy_read_mode` (`scan_row`/`slab`/`auto`) to `Dataset4D` for lazy HDF5 reads, plus detector-chunked HDF5 layout support and a per-pixel prefill path
- Support datasets larger than VRAM and RAM; add `_force_materialize()` for lazy datasets under `cache_mode='device'`
- Add `normalize=` flag to `Dataset4D`/`BFSolver`; add dedicated `CoordinateTransform` unit tests
### Changed
- Rewrite `BFReconstructor` to use `ImageFFTProvider`; new `BFSolver.__init__` with `basis_mode` replacing `cache_mode`
- Simplify vBF ingestion paths; move `flat_to_cartesian_dict` onto `AberrationState`
- Iterate on the auto-detection heuristic for contiguous-HDF5 lazy reads (`scan_row` vs `slab`)
### Removed
- Remove `solver.dataset` in favor of `crop_scan_roi()`; delete stale `source.py`/`prepared.py`

## [0.2.0] - 2026-05-10
### Added
- Add setter functions to update solver state, with a dispatcher to invalidate caches appropriately
- Add new benchmark results after the internal cache refactor
### Changed
- Fully split image data from optical/geometrical caches internally
- Switch live `update_dataset` to device-side BF mask gather

## [0.1.0] - 2026-05-08
### Added
- Add `refine_all_params` as a high-level automatic refinement method (originally `refine_params`)
- Add timing and VRAM benchmarks
- Split image caches into a static image cache and a scan-angle-dependent basis cache, avoiding recomputation of unchanged image FFTs during rotation search
### Changed
- Change the default `cache_mode='lazy'` for memory safety (`cache_mode='full'` remains available for speed)
- Unify `refine_*` methods to save the best image to `solver.reconstructed_image`; unify `search_range` semantics and add `search_halfwidth`
- Wrap most getter functions (e.g. `plot_reconstruction`) in a no-grad context to reduce memory consumption; add `scan_roi` support for sub-FOV AD refinement
- `refine_flips` now permutes all 8 flip configurations

## [0.0.2] - 2026-05-07
Released to unblock downstream tasks; API changes and major refactoring were expected.
### Added
- Add scan rotation support; normalize the flipping convention
- Add `output_frame` arg so results (object, probe, aberrations) can be fed directly into downstream packages like PtyRAD
- Add `get_defocus_stack` for quick 3D reconstruction
- Add `complex-inversion` as an acBF algorithm to suppress depth-dependent contrast changes
- Add the initial basic and regression test suite
### Changed
- Rename `output_frame` to `frame`; make string args (e.g. `acBF`) case-insensitive
- Fix opposite image rotation direction and aberration rotation sign in the detector frame; fix `output_frame` for `get_probe` so sign conventions are consistent between PtyRAD, fast-acbf, and quanpty
- Add a numerical-stability epsilon (default 1e-3), replacing `torch.sgn` with `D / (|D|+eps)`, particularly for CUDA

## [0.0.1] - 2026-04-06
Initial commit.
