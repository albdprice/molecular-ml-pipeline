#!/usr/bin/env python3
"""
MACE Training on Intel Arc A750
================================
Train MACE interatomic potentials on pesticide DFT data using the GPU worker
on media-srv, with MLflow experiment tracking.

The training data lives on shared ZFS storage (/workspace or /data) accessible
to both dev-srv (where I run analysis) and media-srv (where the GPU lives).

Usage (from dev-srv):
    # Direct GPU training via SSH to media-srv gpu-worker
    ssh albd@10.10.49.9 "docker exec gpu-worker python /workspace/train_mace.py \\
        --name pesticide-mace-v1 \\
        --train_file /workspace/mace_training.extxyz \\
        --r_max 6.0 --num_channels 128 --max_epochs 200"

    # Or via the gpu-submit helper
    ~/gpu-submit.sh train_mace.py --name test --train_file /workspace/mace_training.extxyz

Local dry-run (no GPU, just logs config to MLflow):
    python train_mace.py --name test --train_file /data/pesticides_dir_2/mace_training.extxyz --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train MACE on Intel Arc A750 with MLflow tracking")

    # Required
    p.add_argument("--name", required=True, help="Run name (used for MLflow and output dirs)")
    p.add_argument("--train_file", required=True, help="Training data (extxyz)")

    # Model architecture
    p.add_argument("--r_max", type=float, default=6.0, help="Cutoff radius (Angstrom)")
    p.add_argument("--num_channels", type=int, default=128, help="Hidden irreps channels")
    p.add_argument("--max_L", type=int, default=1, help="Max angular momentum")
    p.add_argument("--num_interactions", type=int, default=2, help="Number of message passing layers")
    p.add_argument("--correlation", type=int, default=3, help="Body order correlation")

    # Training
    p.add_argument("--max_epochs", type=int, default=200, help="Maximum epochs")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=5e-7, help="Weight decay")
    p.add_argument("--energy_weight", type=float, default=1.0, help="Energy loss weight")
    p.add_argument("--forces_weight", type=float, default=100.0, help="Forces loss weight")
    p.add_argument("--valid_fraction", type=float, default=0.1, help="Validation fraction")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Infrastructure
    p.add_argument("--device", default="xpu", choices=["cpu", "xpu", "cuda"], help="Training device")
    p.add_argument("--energy_key", default="energy", help="Energy key in extxyz")
    p.add_argument("--forces_key", default="forces", help="Forces key in extxyz")
    p.add_argument("--work_dir", default="/workspace/mace_runs", help="Working directory for outputs")
    p.add_argument("--mlflow_uri", default="http://10.10.49.104:5000", help="MLflow tracking URI")
    p.add_argument("--dry-run", action="store_true", help="Log config to MLflow without training")

    return p.parse_args()


def log_to_mlflow(args, metrics=None):
    """Log training configuration and results to MLflow on dev-srv."""
    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("mace-training-pipeline")

        with mlflow.start_run(run_name=args.name):
            # Architecture
            mlflow.log_param("model", "MACE")
            mlflow.log_param("r_max", args.r_max)
            mlflow.log_param("num_channels", args.num_channels)
            mlflow.log_param("max_L", args.max_L)
            mlflow.log_param("num_interactions", args.num_interactions)
            mlflow.log_param("correlation", args.correlation)

            # Training
            mlflow.log_param("max_epochs", args.max_epochs)
            mlflow.log_param("batch_size", args.batch_size)
            mlflow.log_param("lr", args.lr)
            mlflow.log_param("energy_weight", args.energy_weight)
            mlflow.log_param("forces_weight", args.forces_weight)
            mlflow.log_param("valid_fraction", args.valid_fraction)
            mlflow.log_param("device", args.device)
            mlflow.log_param("train_file", args.train_file)

            if metrics:
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)

            # Log model artifact if it exists
            model_path = Path(args.work_dir) / args.name / f"{args.name}.model"
            if model_path.exists():
                mlflow.log_artifact(str(model_path))

            mlflow.set_tag("gpu", "Intel Arc A750")
            mlflow.set_tag("backend", "IPEX-LLM/SYCL" if args.device == "xpu" else args.device)

        print(f"Logged to MLflow: {args.mlflow_uri}")
    except Exception as e:
        print(f"MLflow logging failed (non-fatal): {e}")


def run_training(args):
    """Execute MACE training via the mace.cli.run_train entry point."""
    os.makedirs(args.work_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", args.name,
        "--train_file", args.train_file,
        "--valid_fraction", str(args.valid_fraction),
        "--r_max", str(args.r_max),
        "--num_channels", str(args.num_channels),
        "--max_L", str(args.max_L),
        "--num_interactions", str(args.num_interactions),
        "--correlation", str(args.correlation),
        "--max_num_epochs", str(args.max_epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--energy_weight", str(args.energy_weight),
        "--forces_weight", str(args.forces_weight),
        "--seed", str(args.seed),
        "--device", args.device,
        "--work_dir", args.work_dir,
        "--model_dir", os.path.join(args.work_dir, args.name),
        "--checkpoints_dir", os.path.join(args.work_dir, args.name, "checkpoints"),
        "--results_dir", os.path.join(args.work_dir, args.name, "results"),
        "--model", "MACE",
        "--scaling", "rms_forces_scaling",
        "--error_table", "PerAtomMAE",
        "--default_dtype", "float32",
        "--E0s", "average",
        "--energy_key", args.energy_key,
        "--forces_key", args.forces_key,
    ]

    print(f"\nStarting MACE training: {args.name}")
    print(f"Device: {args.device}")
    print(f"Data: {args.train_file}")
    print(f"Output: {args.work_dir}/{args.name}/")
    print("Command: " + " ".join(cmd))

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\nTraining complete: {args.name}")
        # Parse results if available
        results_file = Path(args.work_dir) / args.name / "results" / f"{args.name}_run-{args.seed}.txt"
        metrics = {}
        if results_file.exists():
            for line in results_file.read_text().splitlines():
                if "=" in line:
                    key, val = line.split("=", 1)
                    try:
                        metrics[key.strip()] = float(val.strip())
                    except ValueError:
                        pass
        return metrics
    else:
        print(f"Training failed with exit code {result.returncode}")
        return None


def main():
    args = parse_args()

    if args.dry_run:
        print(f"DRY RUN: logging config for {args.name} to MLflow")
        n_structures = sum(1 for l in open(args.train_file) if l.strip().isdigit())
        print(f"Training data: {args.train_file} ({n_structures} structures)")
        log_to_mlflow(args, metrics={"n_structures": n_structures})
        return

    metrics = run_training(args)
    log_to_mlflow(args, metrics=metrics)


if __name__ == "__main__":
    main()
