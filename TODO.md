# TODO

## Bug Fix

- Change the image rotation mode so we don't crop the edges

## Core Algorithm

- Add the scan rotation fitting / refinement routine with flipping / transpose determination, probably via Optuna as well
- Add order-dependent scalings for learning rates for different aberration coefficients
- Implement the weighted-acBF as mentioned in the Ultramicroscopy paper to supress contrast-oscillation along depth

## Performance

- Add VRAM friendly alternative options, full caching with max_order=3 with upscale>1 tends to take tens of GB of VRAM and can easily OOM
- Experiment whether upsampling can be done as padding in k-space right before the final iFFT, this should be faster and more memory efficient

## Features

- Allow arbitrary output real-space pixel size specification if we were to use this for PtyRAD initialization, can be done in BF reconstruction, or during export
- Add scan affine transformation to the vBF images
- Consider add an export method to output necessary output for downstream PtyRAD (3D object, probe aberrations, meas_flipT, scan_rotation)
- Explore whether we can combine this with Ning's routine for affine transformation

## Pipeline & Integration

- Automate the data loading and calibration part (i.e., dk calculation)
- Wrap this as a py4DGUI plugin and push to Muller repo

## Research / Experiments

- Experiment with Optuna for more comprehensive BO-based optimization for 1st and 2nd order aberrations estimation, see if it's fast enough for real-time pipeline
- Experiment the tcDF track and see if AD-based image optimization works
