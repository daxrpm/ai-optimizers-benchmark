#!/usr/bin/env python3
"""
Experiment runner — CLI entry point for running individual benchmark experiments.

Usage:
    uv run python scripts/run_experiment.py --config configs/cifar10_resnet18.yaml --optimizer adam --seed 42
    uv run python scripts/run_experiment.py --config configs/cifar10_resnet18.yaml --optimizer adam --seed 42 --max-epochs 1
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import set_seed, get_device, create_result_dir
from src.datasets import create_dataloaders
from src.models import create_model
from src.optimizers import create_optimizer
from src.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run a single optimizer benchmark experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--optimizer", type=str, required=True, choices=["adam", "adamuon", "kfac"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs from config")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index to use")
    parser.add_argument("--results-dir", type=str, default="results", help="Base results directory")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]

    # Override epochs if specified
    max_epochs = args.max_epochs or config["training"]["max_epochs"]

    # Set seed and device
    set_seed(args.seed)
    device = get_device(args.gpu)
    print(f"Using device: {device}")

    # Create data loaders
    data_kwargs = {
        "data_dir": config["dataset"].get("data_dir", "./data"),
        "batch_size": config["training"]["batch_size"],
        "num_workers": config["training"].get("num_workers", 4),
        "seed": args.seed,
    }
    data = create_dataloaders(dataset_name, **data_kwargs)

    # Create model
    model_kwargs = config["model"].get("kwargs", {})
    if "input_dim" in data:
        model_kwargs["input_dim"] = data["input_dim"]
    model = create_model(model_name, **model_kwargs)
    model = model.to(device)
    print(f"Model: {model_name} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    opt_config = config["optimizers"][args.optimizer]
    optimizer = create_optimizer(args.optimizer, model=model, **opt_config)
    print(f"Optimizer: {args.optimizer} | Config: {opt_config}")

    # Create result directory
    config_stem = Path(args.config).stem
    result_dir = create_result_dir(args.results_dir, config_stem, args.optimizer, args.seed)

    # Run training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        device=device,
        max_epochs=max_epochs,
        lr_schedule=config["training"].get("lr_schedule", "cosine"),
        result_dir=result_dir,
        seed=args.seed,
        num_train_samples=data["num_train_samples"],
    )

    results = trainer.train()

    # Print summary
    summary = results["metrics"]["summary"]
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Best Val Accuracy: {summary['best_val_accuracy']:.4f} (epoch {summary['best_val_epoch']})")
    print(f"Final Val Accuracy: {summary['final_val_accuracy']:.4f}")
    print(f"Total Time: {summary['total_time_sec']:.1f}s")
    print(f"Avg Epoch Time: {summary['avg_epoch_time_sec']:.2f}s")
    print(f"Peak Memory: {summary['peak_memory_mb']:.0f}MB")
    print(f"Avg Throughput: {summary['avg_throughput_samples_sec']:.0f} samples/sec")
    print(f"Convergence: {summary.get('convergence_milestones', {})}")


if __name__ == "__main__":
    main()
