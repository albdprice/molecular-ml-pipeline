#!/usr/bin/env python3
"""
Pesticide DFT Dataset Exploration
==================================
Explore the wB97M-D3BJ/def2-TZVPPD dataset generated from my pesticide BDE pipeline.
28,576 structures with energies and forces from ~1,445 parent molecules and ~25,640 fragments.
"""
import json
import numpy as np
from pathlib import Path

# --- Load the BDE snapshot ---
SNAPSHOT = Path("/data/pesticides_dir_2/bde_snapshot_v3.json")
TRAINING = Path("/data/pesticides_dir_2/mace_training.extxyz")

print("=" * 60)
print("Pesticide DFT Dataset Exploration")
print("=" * 60)

with open(SNAPSHOT) as f:
    data = json.load(f)

summary = data["summary"]
parents = data["parents"]
fragments = data["fragments"]

print(f"\nDFT Level: {summary[dft_level]}")
print(f"Reference: {summary[reference]}")
print(f"\nTotal successful calculations: {summary[total_successful]}")
print(f"Parent molecules: {summary[parent_molecules]}")
print(f"Fragment calculations: {summary[fragment_calculations]}")
print(f"Complete BDE pairs: {summary[complete_bde_pairs]}")
print(f"Structures with forces: {summary[with_forces]}")

# --- Parent molecule statistics ---
print("\n" + "=" * 60)
print("Parent Molecule Statistics")
print("=" * 60)

natoms = [p["natoms"] for p in parents.values() if isinstance(p.get("natoms"), (int, float))]
energies = [p["energy_hartree"] for p in parents.values() if isinstance(p.get("energy_hartree"), (int, float))]
times = [p["elapsed_seconds"] for p in parents.values() if isinstance(p.get("elapsed_seconds"), (int, float))]

print(f"\nMolecule sizes: {min(natoms)}-{max(natoms)} atoms (mean: {np.mean(natoms):.1f})")
print(f"Energies: {min(energies):.2f} to {max(energies):.2f} Hartree")
print(f"Compute time: mean {np.mean(times)/60:.1f} min, max {max(times)/3600:.1f} hours")
print(f"Total compute: {sum(times)/3600:.1f} hours")

# --- Size distribution ---
print("\nAtom count distribution:")
bins = [0, 10, 20, 30, 40, 50, 60, 100]
hist, _ = np.histogram(natoms, bins=bins)
for i in range(len(bins)-1):
    bar = "#" * (hist[i] // 2)
    print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {hist[i]:4d} {bar}")

# --- Training data summary ---
print("\n" + "=" * 60)
print("MACE Training Data")
print("=" * 60)

n_structures = sum(1 for line in open(TRAINING) if line.strip().isdigit())
file_size_mb = TRAINING.stat().st_size / 1e6
print(f"\nFile: {TRAINING}")
print(f"Structures: {n_structures}")
print(f"File size: {file_size_mb:.1f} MB")
print(f"Format: Extended XYZ (ASE-compatible)")
print(f"Energy unit: eV (converted from Hartree)")
print(f"Force unit: eV/Angstrom")

# --- Success rate analysis ---
print("\n" + "=" * 60)
print("Calculation Success Rates")
print("=" * 60)

parent_statuses = {}
for p in parents.values():
    s = p.get("status", "UNKNOWN")
    parent_statuses[s] = parent_statuses.get(s, 0) + 1

frag_statuses = {}
for f in fragments.values():
    s = f.get("status", "UNKNOWN")
    frag_statuses[s] = frag_statuses.get(s, 0) + 1

print("\nParents:")
for status, count in sorted(parent_statuses.items(), key=lambda x: -x[1]):
    pct = count / len(parents) * 100
    print(f"  {status:20s}: {count:5d} ({pct:.1f}%)")

print("\nFragments:")
for status, count in sorted(frag_statuses.items(), key=lambda x: -x[1]):
    pct = count / len(fragments) * 100
    print(f"  {status:20s}: {count:5d} ({pct:.1f}%)")

print("\nDone.")
