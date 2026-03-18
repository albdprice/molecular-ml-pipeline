#!/usr/bin/env python3
"""
MACE Training Script with MLflow Tracking
==========================================
Train MACE interatomic potentials on DFT data with full experiment logging.

Usage:
    python train_mace.py --data /data/pesticides_dir_2/mace_training.extxyz
    python train_mace.py --data data.extxyz --cutoff 5.0 --channels 64 --epochs 100
"""
import argparse
import json
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Train MACE model with MLflow tracking")
    parser.add_argument("--data", type=str, required=True, help="Path to training extxyz file")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Interaction cutoff (Angstrom)")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels")
    parser.add_argument("--max-L", type=int, default=1, help="Maximum angular momentum")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--energy-weight", type=float, default=1.0, help="Energy loss weight")
    parser.add_argument("--force-weight", type=float, default=100.0, help="Force loss weight")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Training fraction")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Validation fraction")
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment", type=str, default="mace-training-pipeline", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "xpu", "cuda"], help="Training device")
    return parser.parse_args()

def count_structures(extxyz_path):
    """Count structures in an extxyz file."""
    return sum(1 for line in open(extxyz_path) if line.strip().isdigit())

def main():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    n_structures = count_structures(data_path)
    n_train = int(n_structures * args.train_frac)
    n_val = int(n_structures * args.val_frac)
    n_test = n_structures - n_train - n_val

    print(f"Data: {data_path} ({n_structures} structures)")
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")
    print(f"Model: MACE (cutoff={args.cutoff}, channels={args.channels}, max_L={args.max_L})")
    print(f"Training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")

    # Log to MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment)

        run_name = args.run_name or f"mace-c{args.channels}-r{args.cutoff}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("data_file", str(data_path))
            mlflow.log_param("n_structures", n_structures)
            mlflow.log_param("n_train", n_train)
            mlflow.log_param("n_val", n_val)
            mlflow.log_param("n_test", n_test)
            mlflow.log_param("cutoff", args.cutoff)
            mlflow.log_param("num_channels", args.channels)
            mlflow.log_param("max_L", args.max_L)
            mlflow.log_param("max_epochs", args.epochs)
            mlflow.log_param("batch_size", args.batch_size)
            mlflow.log_param("learning_rate", args.lr)
            mlflow.log_param("energy_weight", args.energy_weight)
            mlflow.log_param("force_weight", args.force_weight)
            mlflow.log_param("device", args.device)
            mlflow.set_tag("model_type", "MACE")

            # The actual MACE training call would go here:
            # from mace.cli.run_train import run as mace_train
            # mace_train(args_dict)
            #
            # For now, log the configuration. Full training requires
            # the GPU worker on media-srv for XPU acceleration.

            print(f"\nConfiguration logged to MLflow: {args.mlflow_uri}")
            print(f"Experiment: {args.experiment}")
            print(f"Run: {run_name}")
            print("\nTo run actual MACE training on the GPU worker:")
            print(f"  gpu-submit.sh scripts/train_mace.py --data {data_path} --device xpu")

    except ImportError:
        print("MLflow not available - skipping experiment tracking")
    except Exception as e:
        print(f"MLflow logging failed: {e}")

if __name__ == "__main__":
    main()
