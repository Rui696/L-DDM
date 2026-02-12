# L-DDM: A Learning-based Domain Decomposition Method

[![Paper](https://img.shields.io/badge/CMAME-Paper-blue)](https://doi.org/10.1016/j.cma.2026.118799)
[![arXiv](https://img.shields.io/badge/arXiv-2507.17328-b31b1b)](https://arxiv.org/abs/2507.17328)

Official code for the paper *"A Learning-based Domain Decomposition Method"*, published in **Computer Methods in Applied Mechanics and Engineering (CMAME)**.

**[Rui Wu](https://github.com/Rui696), Nikola Kovachki, Burigede Liu.**

---

<!-- TODO: Replace with your method overview figure -->
<p align="center">
  <img src="assets/method_overview.png" alt="L-DDM Method Overview" width="90%"/>
</p>

## Overview

Can neural operators scale to arbitrary complex geometries — without retraining? **L-DDM** does this by pretraining a neural operator once on a simple canonical domain, then reusing it as a universal local solver inside a classical **(parallel) additive Schwarz** domain decomposition loop.

- **Geometry generalisation** — tackle large, nontrivial domains by stitching together many local solves
- **Parallel by design** — subdomain solves run concurrently (DDM-style scalability)
- **Robust** — strong resolution-invariance and generalisation to microstructures / boundary conditions beyond training
- **Theory + practice** — approximation theorem alongside extensive numerical verification of convergence

## Installation

**Prerequisites:** Python 3.8+ and a CUDA-capable GPU (recommended).

**Windows (one-click):**
```bash
setup.bat
```

**Manual:**
```bash
# 1. Install PyTorch (select your CUDA version from https://pytorch.org)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

All settings are controlled via `config.yaml`. Run with:

```bash
python main.py --config config.yaml
```

**Example `config.yaml`:**

```yaml
niter: 100                          # Number of Schwarz iterations
predict_ux: True                    # Also predict ∂u/∂x
saved_name_u: 'u-k3-res129-e500'   # Checkpoint ID for u model
saved_name_ux: 'ux-k3-res129-e500' # Checkpoint ID for ∂u/∂x model
base_kernel: 3                      # UNet base kernel size
trained_epoch: 500                  # Checkpoint epoch to load
data_dir: 'Data'                    # Path to input data
ckpt_dir: 'Checkpoints'            # Path to model checkpoints
Nx: 10                              # Subdomains in x-direction
Ny: 10                              # Subdomains in y-direction
boundary_type: 'rand'               # Boundary type: 'rand' or 'const'
resolution: 129                     # Grid resolution per subdomain
overlap_ratio: 0.3125               # Overlap ratio between subdomains
nsamples: 1                         # Number of test samples
eval_batch: 10                      # Batch size for evaluation
out_every: 1                        # Output frequency
```

Use `visualisation.ipynb` to plot convergence curves and animated predictions.


## Citation

```bibtex
@article{wu2026lddm,
  title={A learning-based domain decomposition method},
  author={Wu, Rui and Kovachki, Nikola and Liu, Burigede},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={453},
  pages={118799},
  year={2026},
  issn={0045-7825},
  doi={10.1016/j.cma.2026.118799},
  publisher={Elsevier}
}
```

## Contact

Questions or issues? Please [open an issue](https://github.com/Rui696/L-DDM/issues).
