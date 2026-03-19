#!/usr/bin/env python3
"""
Pesticide DFT Dataset: Exploration and Analysis
================================================

This notebook explores the wB97M-D3BJ/def2-TZVPPD dataset from my pesticide
bond dissociation energy (BDE) pipeline. 28,576 structures with DFT energies
and forces for training MACE interatomic potentials.

Data Pipeline:
  Pesticide SMILES → RDKit fragmentation → Parent + fragment XYZ files
  → Psi4 DFT (wB97M-D3BJ/def2-TZVPPD) on Alliance Canada HPC
  → BDE snapshot (JSON) + MACE training data (extxyz)
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter

# === Configuration ===
SNAPSHOT = Path("/data/pesticides_dir_2/bde_snapshot_v3.json")
TRAINING = Path("/data/pesticides_dir_2/mace_training.extxyz")
# Use the fixed version if available (empty Lattice="" entries removed)
TRAINING_FIXED = Path("/workspace/mace_training_fixed.extxyz")
if TRAINING_FIXED.exists():
    TRAINING = TRAINING_FIXED

# =====================================================================
# 1. BDE SNAPSHOT OVERVIEW
# =====================================================================
# The snapshot tracks every calculation: energy, forces, geometry, charge,
# spin multiplicity, SMILES, and computation metadata.

with open(SNAPSHOT) as f:
    data = json.load(f)

summary = data["summary"]
parents = data["parents"]
fragments = data["fragments"]

print("=" * 60)
print("Pesticide BDE Dataset Summary")
print("=" * 60)
print(f"DFT Level: {summary['dft_level']}")
print(f"Reference: {summary['reference']}")
print(f"Total successful: {summary['total_successful']}")
print(f"Parents: {summary['parent_molecules']}")
print(f"Fragments: {summary['fragment_calculations']}")
print(f"Complete BDE pairs: {summary['complete_bde_pairs']}")
print(f"With forces: {summary['with_forces']}")

# =====================================================================
# 2. PARENT MOLECULE STATISTICS
# =====================================================================
# Size distribution matters because MACE cost scales with neighborhood
# size. Very large molecules (60+ atoms) need smaller batch sizes.

natoms_list = []
energy_list = []
time_list = []
for p in parents.values():
    if isinstance(p.get("natoms"), (int, float)):
        natoms_list.append(p["natoms"])
    if isinstance(p.get("energy_hartree"), (int, float)):
        energy_list.append(p["energy_hartree"])
    if isinstance(p.get("elapsed_seconds"), (int, float)):
        time_list.append(p["elapsed_seconds"])

print("\nParent Molecule Statistics")
print("-" * 40)
print(f"Count: {len(parents)}")
print(f"Size range: {min(natoms_list)}-{max(natoms_list)} atoms")
print(f"Mean: {np.mean(natoms_list):.1f}, Median: {np.median(natoms_list):.0f}")
print(f"Compute: mean {np.mean(time_list)/60:.1f} min, total {sum(time_list)/3600:.1f} hrs")

print("\nSize distribution:")
bins = [0, 10, 20, 30, 40, 50, 60, 80, 120]
hist, _ = np.histogram(natoms_list, bins=bins)
for i in range(len(bins)-1):
    bar = "#" * (hist[i] // 2)
    print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {hist[i]:4d} {bar}")

# =====================================================================
# 3. CONVERGENCE SUCCESS RATES
# =====================================================================
# The pipeline uses automatic convergence cascades:
#   SCF: DIIS -> SOSCF -> level shift -> damping -> "sledgehammer"
#   Geometry: internal tight -> relaxed -> Hessian recomp -> Cartesian

parent_statuses = Counter(p.get("status", "UNKNOWN") for p in parents.values())
frag_statuses = Counter(f.get("status", "UNKNOWN") for f in fragments.values())

print("\nConvergence Success Rates")
print("-" * 40)
print("Parents:")
for status, count in parent_statuses.most_common():
    print(f"  {status:20s}: {count:5d} ({count/len(parents)*100:.1f}%)")
print("Fragments:")
for status, count in frag_statuses.most_common():
    print(f"  {status:20s}: {count:5d} ({count/len(fragments)*100:.1f}%)")

# =====================================================================
# 4. TRAINING DATA ANALYSIS
# =====================================================================
# The extxyz file contains three structure types:
#   initial_maceoff_geom: MACE-OFF pre-optimized geometry (before DFT)
#   final_dft_optimized: DFT-optimized geometry
#   single_atom: isolated atomic energies
#
# Including both initial and final geometries teaches MACE the PES
# between them, not just the minimum.

n_structures = sum(1 for line in open(TRAINING) if line.strip().isdigit())
file_size_mb = TRAINING.stat().st_size / 1e6

print(f"\nMACE Training Data: {n_structures:,} structures ({file_size_mb:.1f} MB)")

# Count elements
element_counts = Counter()
with open(TRAINING) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4 and parts[0].isalpha() and len(parts[0]) <= 2:
            element_counts[parts[0]] += 1

print(f"Elements: {len(element_counts)}")
for elem, count in element_counts.most_common():
    print(f"  {elem:3s}: {count:>8,d}")

# =====================================================================
# 5. TRAINING CONFIGURATION
# =====================================================================
# Current: 32 channels on Arc A750 (8GB VRAM), ~70 min/epoch
# Production: 128 channels after RAM upgrade + BIOS ReBAR

print("\n" + "=" * 60)
print("MACE Training Configuration")
print("=" * 60)
config = {
    "r_max": "5.0 A (covers 2nd coordination shell)",
    "num_channels": "32 (test) / 128 (production)",
    "max_L": "1 (sufficient for organic chemistry)",
    "num_interactions": "2 (standard message passing depth)",
    "correlation": "3 (3-body interactions)",
    "batch_size": "4 (limited by 8GB VRAM)",
    "forces_weight": "100.0 (forces are primary training signal)",
    "E0s": "average (from dataset)",
    "device": "Intel Arc A750 via IPEX-LLM/SYCL",
    "fp32 patch": "Required (Arc lacks fp64 compute)",
}
for k, v in config.items():
    print(f"  {k:20s}: {v}")

print("\nData augmentation strategy:")
print("  - Fine-tune with --restart_latest when new data arrives")
print("  - Full retrain from scratch for production model")
print("  - Current pipeline: ~52% complete, ~28K structures")
print("  - Production target: ~55K+ structures")
