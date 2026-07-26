# GPU-accelerated tcBF and acBF

This repository provides a fast GPU implementation of aberration-corrected bright-field 4D-STEM (tcBF/acBF) reconstruction.

It is designed to serve as the seed for:

1. Future native integration of tcBF and acBF into [PtyRAD](https://github.com/chiahao3/ptyrad) pipeline
2. Real-time visualization during 4D-STEM data acquisition

The underlying imaging theory and initial CPU implementation of **acBF** were developed by Dr. Desheng Ma and Dr. Steven Zeltmann [1, 2], while the **tcBF** method has a longer history; readers are encouraged to read this paper by Dr. Yue Yu [3].

This implementation (**fast-acBF**) was developed independently focusing on GPU acceleration and integration with the PtyRAD reconstruction framework. It is shared as a working research implementation; interfaces may change as development continues.

## Installation guide

**Major Dependencies:**
- python >=3.10
- pytorch >=2.4
- ptyrad

### 1. Get the fast-acBF code from GitHub
You can either download the repository as a .zip file and extract it, or use the following command if you have `git` installed.

```bash
git clone https://github.com/chiahao3/fast-acbf
```

### 2. Create and Activate the Python Environment
Assuming you're using conda, you can create an independent environment and install the packages with these commands:

```bash
# Enter the commands one by one
conda create -n fast-acbf python=3.12 -y
conda activate fast-acbf
cd fast-acbf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

If you prefer a legacy version of PyTorch, or a different version of CUDA runtime other than CUDA 12.6, see instruction [here](https://pytorch.org/get-started/previous-versions/).

## Get Started

1. Download the demo tBL-WSe2 data "Figure 4.zip" from the [Zenodo link](https://doi.org/10.5281/zenodo.15283331)
2. Run the `get_acBF.ipynb` Jupyter notebook to reconstruct tcBF / acBF images

## Data Pipeline Defaults

`BFSolver` is the main user entry point. By default it uses
`pipeline="balanced"` with automatic policy resolution:

```python
from fast_acbf import BFSolver, Dataset4D

dataset = Dataset4D.from_hdf5("scan.h5", key="array")  # lazy by default

solver = BFSolver(
    dataset=dataset,
    max_alpha=25.0,
    scan_step_size=0.43,
    dk=0.04,
    wavelength=0.04176,
    device="cuda",
)
```

The solver consumes `ImageFFT` chunks internally. The policy controls where that
FFT cache lives, when it is filled, and how virtual BF images are extracted from
the raw 4D data.

| Pipeline | Intended use | Default behavior |
| --- | --- | --- |
| `speed` | Small data that can afford temporary materialization | Prefer device ImageFFT and, when raw 4D can be materialized and fit in VRAM with vBF/ImageFFT, do a whole-pass `device_mask` precompute. |
| `balanced` | General default | Cache ImageFFT on device if it fits, otherwise host RAM, otherwise stream. Avoids raw-on-device extraction for lazy disk data unless explicitly requested. |
| `memory` | Largest data / lowest persistent memory | Keep `imagefft_storage="none"` and compute on the fly, using disk extraction for lazy data. |

Advanced users can override the resolved policy:

```python
solver = BFSolver(
    dataset=dataset,
    ...,
    pipeline="balanced",
    imagefft_storage="host",     # auto | device | host | none
    imagefft_fill="precompute",  # auto | precompute | lazy | on_the_fly
    extractor_strategy="auto",   # auto | device_mask | host_mask | disk_*
)
```

Impossible or wasteful combinations are rejected early with memory/path guidance.
For example, `extractor_strategy="device_mask"` requires persistent ImageFFT
storage and `imagefft_fill="precompute"` because it is a whole-pass route:
temporarily move raw 4D to the compute device, extract all vBF images, precompute
ImageFFT, then release raw/vBF intermediates.

## References

[1] Ma, Desheng, et al. "Information in 4D-STEM: Where it is, and How to Use it." Ultramicroscopy (2026). https://doi.org/10.1016/j.ultramic.2026.114351

[2] Ma, Desheng, David A. Muller, and Steven E. Zeltmann. "Using Aberrations to Improve Dose-Efficient Tilt-corrected 4D-STEM Imaging." Microscopy and Microanalysis (2026). https://doi.org/10.1093/mam/ozag008

[3] Yu, Yue, et al. "Dose-efficient cryo-electron microscopy for thick samples using tilt-corrected scanning transmission electron microscopy." Nature Methods (2025). https://doi.org/10.1038/s41592-025-02834-9

## Relevant Repositories

- [tcBF-STEM](https://github.com/yyu2017/tcBFSTEM)
- [acBF-STEM](https://github.com/dsmagiya/acBF-STEM)
- [PtyRAD](https://github.com/chiahao3/ptyrad)
- [py4DSTEM](https://github.com/py4dstem/py4DSTEM)
- [quantem](https://github.com/electronmicroscopy/quantem)

## Author 

Chia-Hao Lee (cl2696@cornell.edu)

Developed at the Muller Group, Cornell University.


