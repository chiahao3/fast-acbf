# TODO

- Check the scan rotation correction logic
- Add the scan rotation fitting routine with flipping / transpose determination
- Export 3D tcBF and acBF with defoci stack
- Add VRAM friendly alternative options, full caching with max_order=3 with upscale>1 tends to take tens of GB of VRAM and can easily OOM
- Test it on single C atom simulation
- Consider adding order-dependent learning rates for different aberration coefficients
- Add Optuna for more comprehensive BO-based optimization for 1st and 2nd order aberrations estimation 
- Add scan affine transformation to the vBF images
- Experiment whether upsampling can be done as padding in k-space right before the final iFFT
- Might be convenient to allow arbitrary output real-space pixel size if we were to use this for PtyRAD initialization
- Implement the weighted-acBF as mentioned in the Ultramicroscopy paper