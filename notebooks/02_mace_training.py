#!/usr/bin/env python3
"""
MACE Training Pipeline on Intel Arc A750
=========================================

End-to-end pipeline for training MACE interatomic potentials on my
pesticide DFT data. Runs on an Intel Arc A750 GPU via IPEX-LLM/SYCL
with MLflow experiment tracking.

Architecture:
  dev-srv (this machine)
    - Data analysis, MLflow tracking, notebooks
    - /data/pesticides_dir_2/ (DFT results on ZFS)
    - /workspace/ (shared with GPU worker via virtiofs)

  media-srv (GPU host)
    - gpu-worker container with MACE + PyTorch XPU
    - Intel Arc A750 (8GB VRAM, SYCL/Level-Zero)
    - Ollama GPU also runs here (different GPU engine, no conflict)
"""
import json
import os
import subprocess
from pathlib import Path

# =====================================================================
# 1. DATA PREPARATION
# =====================================================================
# The raw training data from export_ase_training.py has Lattice=""
# entries that cause ASE parsing errors. These need to be stripped
# since our molecules are non-periodic.

TRAINING_RAW = Path("/data/pesticides_dir_2/mace_training.extxyz")
TRAINING_FIXED = Path("/workspace/mace_training_fixed.extxyz")

if not TRAINING_FIXED.exists():
    print("Fixing training data (removing empty Lattice entries)...")
    with open(TRAINING_RAW) as fin, open(TRAINING_FIXED, "w") as fout:
        for line in fin:
            fout.write(line.replace('Lattice="" ', ''))
    print(f"Fixed: {TRAINING_FIXED}")
else:
    print(f"Using existing fixed data: {TRAINING_FIXED}")

n = sum(1 for l in open(TRAINING_FIXED) if l.strip().isdigit())
print(f"Training structures: {n:,}")

# =====================================================================
# 2. TRAINING CONFIGURATION
# =====================================================================
# Key decisions and their rationale:
#
# r_max = 5.0 A
#   Standard cutoff for organic molecules. Captures first and second
#   coordination shells. Larger values increase memory usage quadratically.
#
# num_channels = 32 (test) / 128 (production)
#   The number of equivariant features per interaction layer. 32 fits
#   in 8GB VRAM with batch_size=4. Production runs need 128+ channels
#   which requires either larger VRAM or the BIOS ReBAR update to
#   expose the full 8GB to SYCL compute.
#
# max_L = 1
#   Maximum angular momentum for spherical harmonics. L=1 captures
#   dipole-like interactions. L=2 adds quadrupole terms and is needed
#   for transition metals but overkill for organic HCNOS chemistry.
#
# correlation = 3
#   Body order for the ACE (Atomic Cluster Expansion) correlation.
#   3-body captures angles between bonds. 4-body captures dihedrals
#   but is much more expensive.
#
# forces_weight = 100.0
#   Forces are the primary training signal for MACE. The energy is
#   a single scalar per structure; forces are 3N values. Higher
#   force weight ensures the model learns the PES shape, not just
#   relative energies.
#
# E0s = "average"
#   Isolated atom reference energies. MACE subtracts these from total
#   energies so the model learns formation energies (which are smaller
#   and easier to fit). "average" computes E0 per element from the
#   dataset mean.
#
# fp32 patch
#   Intel Arc consumer GPUs lack fp64 (double precision) compute.
#   MACE calls .double() for numerical precision in energy accumulation.
#   I patched this to .float() in the gpu-worker container, which is
#   sufficient for training accuracy.

config = {
    "name": "pesticide-test-v1",
    "train_file": str(TRAINING_FIXED),
    "r_max": 5.0,
    "num_channels": 32,
    "max_L": 1,
    "num_interactions": 2,
    "correlation": 3,
    "max_epochs": 50,
    "batch_size": 4,
    "lr": 0.01,
    "weight_decay": 5e-7,
    "energy_weight": 1.0,
    "forces_weight": 100.0,
    "valid_fraction": 0.05,
    "seed": 42,
    "device": "xpu",
    "energy_key": "energy",
    "forces_key": "forces",
    "E0s": "average",
    "scaling": "rms_forces_scaling",
}

print("\nTraining Configuration:")
for k, v in config.items():
    print(f"  {k:20s}: {v}")

# =====================================================================
# 3. LAUNCH TRAINING
# =====================================================================
# Training runs on the gpu-worker container on media-srv (10.10.49.9).
# The /workspace directory is shared between dev-srv and media-srv via
# ZFS virtiofs, so data and checkpoints are accessible from both.
#
# The training script (train_mace.py) wraps mace.cli.run_train and adds:
#   - MLflow logging of all hyperparameters and metrics
#   - Model artifact logging
#   - Support for --dry-run to log config without training
#
# For actual GPU training, use SSH to media-srv:
#
#   ssh albd@10.10.49.9 "docker exec gpu-worker python \
#       /workspace/train_mace.py \
#       --name pesticide-test-v1 \
#       --train_file /workspace/mace_training_fixed.extxyz \
#       --r_max 5.0 --num_channels 32 --max_epochs 50 \
#       --device xpu --energy_key energy --forces_key forces"
#
# Or via the gpu-submit helper:
#
#   ~/gpu-submit.sh train_mace.py --name pesticide-test-v1 ...

print("\nTo launch training on the GPU:")
print("  ssh albd@10.10.49.9 'docker exec gpu-worker python \\")
print("    /workspace/train_mace.py \\")
print(f"    --name {config['name']} \\")
print(f"    --train_file {config['train_file']} \\")
print(f"    --r_max {config['r_max']} --num_channels {config['num_channels']} \\")
print(f"    --max_epochs {config['max_epochs']} --batch_size {config['batch_size']} \\")
print(f"    --device {config['device']} --energy_key energy --forces_key forces'")

# =====================================================================
# 4. MONITORING TRAINING
# =====================================================================
# Training logs are at /tmp/mace-training.log on media-srv.
# Monitor with: ssh albd@10.10.49.9 'tail -f /tmp/mace-training.log'
#
# Key metrics per epoch:
#   loss: combined energy + force loss (force_weight * force_loss + energy_loss)
#   MAE_E_per_atom: energy mean absolute error in meV/atom
#   MAE_F: force mean absolute error in meV/Angstrom
#
# Expected training progression (32 channels, 28K structures):
#   Epoch 0: ~70 min, loss drops from ~54 to ~2.3
#   Epoch 1: ~70 min, loss ~2.1, force MAE ~160 meV/A
#   Epoch 10: force MAE should be <100 meV/A
#   Epoch 50: target force MAE <50 meV/A for useful predictions

print("\nMonitor training:")
print("  ssh albd@10.10.49.9 'grep Epoch /tmp/mace-training.log'")
print("  ssh albd@10.10.49.9 'grep Epoch /tmp/mace-continuation.log'")

# =====================================================================
# 5. CHECKPOINT MANAGEMENT
# =====================================================================
# Checkpoints are saved at /workspace/mace_runs/pesticide-test-v1/
#
# Structure:
#   checkpoints/
#     pesticide-test-v1_run-42_epoch-0.pt   (after epoch 0)
#     pesticide-test-v1_run-42_epoch-1.pt   (after epoch 1)
#     pesticide-test-v1_run-42.model        (best model)
#   results/
#     pesticide-test-v1_run-42.txt          (final error table)
#
# To resume from a checkpoint (e.g., after adding new data):
#   Add --restart_latest to the training command
#   MACE will load the latest checkpoint and continue training
#
# To use the trained model for inference:
#   from mace.calculators import MACECalculator
#   calc = MACECalculator(model_path, device="xpu")
#   atoms.calc = calc
#   energy = atoms.get_potential_energy()
#   forces = atoms.get_forces()

checkpoint_dir = Path("/workspace/mace_runs/pesticide-test-v1/checkpoints")
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    models = sorted(checkpoint_dir.glob("*.model"))
    print(f"\nCheckpoints: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  {cp.name} ({cp.stat().st_size/1e6:.1f} MB)")
    for m in models:
        print(f"  Model: {m.name} ({m.stat().st_size/1e6:.1f} MB)")
else:
    print("\nNo checkpoints yet (training not started or running remotely)")

# =====================================================================
# 6. PRODUCTION MODEL ROADMAP
# =====================================================================
# Current test model (running):
#   32 channels, max_L=1, batch_size=4, ~28K structures
#   Estimated: 2-3 days for 50 epochs
#
# Production model (after upgrades):
#   128 channels, max_L=2, batch_size=16+
#   Full dataset: ~55K+ structures
#   Requires: BIOS ReBAR update + 64GB RAM
#   Estimated: 5-7 days for 200 epochs
#
# Deployment:
#   Model as ASE calculator for property predictions
#   Log to MLflow for experiment comparison
#   Use for BDE prediction validation against DFT

print("\nProduction Roadmap:")
print("  1. Current: 32ch test model training (2-3 days)")
print("  2. Complete DFT pipeline (52% -> 100%, ~2 weeks)")
print("  3. Export full dataset (~55K structures)")
print("  4. BIOS ReBAR update for full GPU memory access")
print("  5. Production model: 128ch, max_L=2, 200 epochs")
print("  6. Validate against DFT BDE values")
print("  7. Deploy as ASE calculator for property screening")
