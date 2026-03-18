# Molecular ML Pipeline

End-to-end MLOps pipeline for training MACE neural network potentials on DFT data, with MLflow experiment tracking and GPU-accelerated inference.

## Overview

Demonstrates a production ML pipeline for molecular property prediction using [MACE](https://github.com/ACEsuit/mace). Training data: 28,576 DFT structures from a pesticide bond dissociation energy study (wB97M-D3BJ/def2-TZVPPD).

### Pipeline

```
DFT (Psi4, wB97M-D3BJ/def2-TZVPPD) → Export (ASE extxyz) → MACE Training (Arc A750 GPU)
    → MLflow Tracking → Model Evaluation → Deployment (ASE calculator)
```

### Key Features
- **28,576 real DFT structures** with energies and forces
- **GPU training** on Intel Arc A750 via PyTorch XPU/SYCL
- **MLflow** experiment tracking, model comparison, artifact storage
- **Reproducible** conda environments and Docker containers

## Structure

```
├── notebooks/           # Jupyter notebooks for exploration and training
├── scripts/             # CLI scripts for training and evaluation
├── configs/             # MACE training configurations
└── README.md
```

## Quick Start

```bash
conda create -n mace-pipeline python=3.11 numpy scipy matplotlib
pip install mace-torch ase mlflow scikit-learn
jupyter lab notebooks/
```

## Related

- [adaptive-ml-potentials](https://github.com/albdprice/adaptive-ml-potentials) — Adaptive parameter learning
- [psi4_xdm](https://github.com/albdprice/psi4_xdm) — XDM dispersion in Psi4
