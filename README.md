# Molecular ML Pipeline

MLOps pipeline for training and evaluating MACE neural network potentials on DFT-computed molecular data, with experiment tracking via MLflow.

## Motivation

I run large-scale DFT calculations (wB97M-D3BJ/def2-TZVPPD) on pesticide molecules and their radical fragments as part of a bond dissociation energy study. The resulting dataset — 28,576 structures with energies and forces — serves as training data for MACE interatomic potentials. This repository structures that workflow into a reproducible pipeline with proper experiment tracking.

## Pipeline

```
DFT (Psi4, wB97M-D3BJ/def2-TZVPPD)
  → ASE extxyz export (energies in eV, forces in eV/Å)
  → Train/val/test split
  → MACE training (Intel Arc A750 via PyTorch XPU)
  → MLflow logging (hyperparameters, metrics, model artifacts)
  → Evaluation (energy MAE, force MAE, parity plots)
  → Deployment as ASE calculator
```

## Structure

- `notebooks/` — Data exploration, training, evaluation, deployment demos
- `scripts/` — CLI tools for training and batch evaluation
- `configs/` — MACE training configurations (YAML)

## Infrastructure

Training runs on a self-hosted Proxmox cluster:
- **GPU**: Intel Arc A750 (8GB VRAM, SYCL/Level-Zero via IPEX-LLM)
- **Tracking**: MLflow at mlflow.albdchem.org
- **Storage**: ZFS (tank/research for data, tank/ml-workspace for checkpoints)
- **Notebooks**: JupyterLab at jupyter.albdchem.org

## Data

The training data originates from ~1,445 pesticide parent molecules and ~25,640 radical fragments. Each structure includes atomic positions, DFT total energy, and analytical forces. Metadata: charge, spin multiplicity, SMILES, parent molecule ID.
