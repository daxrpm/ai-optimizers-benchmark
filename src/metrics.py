"""Metric logging and aggregation for benchmarking experiments."""

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class StepMetrics:
    """Metrics collected at each training step."""
    step: int
    train_loss: float
    step_time_sec: float
    lr: float


@dataclass
class EpochMetrics:
    """Metrics collected at the end of each epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    epoch_time_sec: float
    cumulative_time_sec: float
    peak_memory_mb: float
    throughput_samples_sec: float
    lr: float


class MetricLogger:
    """Accumulates per-step and per-epoch metrics, computes convergence milestones."""

    def __init__(self):
        self.step_metrics: list[dict] = []
        self.epoch_metrics: list[dict] = []
        self.convergence_milestones: dict[str, Optional[dict]] = {}
        self._best_val_acc = 0.0
        self._start_time = time.perf_counter()

    def log_step(self, step: int, train_loss: float, step_time: float, lr: float):
        self.step_metrics.append({
            "step": step,
            "train_loss": train_loss,
            "step_time_sec": step_time,
            "lr": lr,
        })

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        epoch_time: float,
        peak_memory_mb: float,
        num_train_samples: int,
        lr: float,
    ):
        cumulative = time.perf_counter() - self._start_time
        throughput = num_train_samples / epoch_time if epoch_time > 0 else 0.0

        self.epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_time_sec": epoch_time,
            "cumulative_time_sec": cumulative,
            "peak_memory_mb": peak_memory_mb,
            "throughput_samples_sec": throughput,
            "lr": lr,
        })

        # Track convergence milestones
        self._best_val_acc = max(self._best_val_acc, val_accuracy)

    def compute_convergence_milestones(self) -> dict:
        """Compute epochs and wall-clock time to reach accuracy thresholds."""
        if not self.epoch_metrics:
            return {}

        best_acc = max(e["val_accuracy"] for e in self.epoch_metrics)
        thresholds = [0.50, 0.75, 0.90, 0.95]
        milestones = {}

        for threshold_frac in thresholds:
            target = best_acc * threshold_frac
            key = f"time_to_{int(threshold_frac * 100)}pct"
            milestones[key] = None
            for e in self.epoch_metrics:
                if e["val_accuracy"] >= target:
                    milestones[key] = {
                        "epoch": e["epoch"],
                        "wall_time_sec": e["cumulative_time_sec"],
                        "target_accuracy": target,
                        "achieved_accuracy": e["val_accuracy"],
                    }
                    break

        self.convergence_milestones = milestones
        return milestones

    def get_summary(self) -> dict:
        """Return a complete summary of the experiment."""
        if not self.epoch_metrics:
            return {}

        final = self.epoch_metrics[-1]
        best_val_acc = max(e["val_accuracy"] for e in self.epoch_metrics)
        best_val_epoch = next(
            e["epoch"] for e in self.epoch_metrics if e["val_accuracy"] == best_val_acc
        )
        avg_epoch_time = sum(e["epoch_time_sec"] for e in self.epoch_metrics) / len(self.epoch_metrics)
        avg_throughput = sum(e["throughput_samples_sec"] for e in self.epoch_metrics) / len(self.epoch_metrics)
        max_peak_memory = max(e["peak_memory_mb"] for e in self.epoch_metrics)

        milestones = self.compute_convergence_milestones()

        return {
            "final_train_loss": final["train_loss"],
            "final_train_accuracy": final["train_accuracy"],
            "final_val_loss": final["val_loss"],
            "final_val_accuracy": final["val_accuracy"],
            "best_val_accuracy": best_val_acc,
            "best_val_epoch": best_val_epoch,
            "total_time_sec": final["cumulative_time_sec"],
            "avg_epoch_time_sec": avg_epoch_time,
            "avg_throughput_samples_sec": avg_throughput,
            "peak_memory_mb": max_peak_memory,
            "num_epochs": len(self.epoch_metrics),
            "convergence_milestones": milestones,
        }

    def to_dict(self) -> dict:
        """Serialize all metrics."""
        return {
            "step_metrics": self.step_metrics,
            "epoch_metrics": self.epoch_metrics,
            "summary": self.get_summary(),
        }
