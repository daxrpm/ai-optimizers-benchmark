"""Utility functions for reproducibility, device setup, and system info logging."""

import os
import random
import json
import platform
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(gpu_id: int = 0) -> torch.device:
    """Get the appropriate device."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def get_system_info(device: torch.device) -> dict:
    """Log hardware and software versions for reproducibility."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        idx = device.index or 0
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": str(torch.backends.cudnn.version()),
            "gpu_name": torch.cuda.get_device_name(idx),
            "gpu_memory_total_mb": torch.cuda.get_device_properties(idx).total_mem / 1e6,
            "gpu_count": torch.cuda.device_count(),
        })
    return info


def create_result_dir(base_dir: str, dataset: str, optimizer: str, seed: int) -> Path:
    """Create and return the result directory for a specific experiment run."""
    result_dir = Path(base_dir) / dataset / optimizer / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as formatted JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    """Load JSON from a file."""
    with open(path, "r") as f:
        return json.load(f)
