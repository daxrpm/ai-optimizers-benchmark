"""
Unified training loop for optimizer benchmarking.

Handles all optimizer types (Adam, AdaMuon, K-FAC) uniformly,
collects comprehensive metrics, and saves results to JSON.
"""

import time
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.metrics import MetricLogger
from src.utils import set_seed, get_system_info, save_json


class Trainer:
    """
    Unified trainer for optimizer benchmarking experiments.

    Args:
        model: Neural network model
        optimizer: Optimizer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Torch device
        max_epochs: Maximum number of training epochs
        lr_schedule: LR schedule type ('cosine' or 'none')
        result_dir: Directory to save results
        seed: Random seed for this run
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        max_epochs: int = 100,
        lr_schedule: str = "cosine",
        result_dir: Optional[Path] = None,
        seed: int = 42,
        num_train_samples: int = 0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.result_dir = result_dir
        self.seed = seed
        self.num_train_samples = num_train_samples

        self.criterion = nn.CrossEntropyLoss()
        self.logger = MetricLogger()

        # LR scheduler
        if lr_schedule == "cosine":
            self.scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        else:
            self.scheduler = None

        self._global_step = 0

    def train_epoch(self, epoch: int) -> tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            step_start = time.perf_counter()

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            step_time = time.perf_counter() - step_start
            self._global_step += 1

            # Accumulate metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Log step metrics (every 10 steps to reduce overhead)
            if self._global_step % 10 == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_step(self._global_step, loss.item(), step_time, current_lr)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Evaluate on validation set. Returns (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self) -> dict:
        """Run the full training loop and return metrics."""
        system_info = get_system_info(self.device)
        print(f"\n{'='*60}")
        print(f"Training | Device: {self.device} | Epochs: {self.max_epochs}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}")

        # Reset GPU memory tracking
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.perf_counter()

            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.evaluate()

            # Timing and memory
            epoch_time = time.perf_counter() - epoch_start
            peak_mem_mb = 0.0
            if self.device.type == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / 1e6

            # LR step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                self.scheduler.step()

            # Log
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                epoch_time=epoch_time,
                peak_memory_mb=peak_mem_mb,
                num_train_samples=self.num_train_samples,
                lr=current_lr,
            )

            # Print progress
            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                f"Time: {epoch_time:.1f}s | "
                f"Mem: {peak_mem_mb:.0f}MB | "
                f"LR: {current_lr:.6f}"
            )

        # Finalize metrics
        results = {
            "system_info": system_info,
            "config": {
                "seed": self.seed,
                "max_epochs": self.max_epochs,
                "num_train_samples": self.num_train_samples,
            },
            "metrics": self.logger.to_dict(),
        }

        # Save results
        if self.result_dir is not None:
            self.result_dir.mkdir(parents=True, exist_ok=True)
            save_json(results, self.result_dir / "metrics.json")
            print(f"\nResults saved to {self.result_dir / 'metrics.json'}")

        return results
