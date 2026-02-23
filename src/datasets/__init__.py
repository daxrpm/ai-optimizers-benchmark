"""Dataset factory and registry."""

from torch.utils.data import DataLoader
from typing import Any

from src.datasets.image_datasets import (
    get_fashion_mnist,
    get_cifar10,
    get_cifar100,
    get_svhn,
)
from src.datasets.tabular_datasets import get_adult, get_covertype


def create_dataloaders(
    name: str, **kwargs: Any
) -> dict:
    """
    Create dataloaders by dataset name.

    Returns a dict with keys: train_loader, val_loader, test_loader,
    num_train_samples, and (for tabular) input_dim.
    """
    name = name.lower()

    if name == "fashion_mnist":
        train, val, test, n_train = get_fashion_mnist(**kwargs)
        return {"train_loader": train, "val_loader": val, "test_loader": test, "num_train_samples": n_train}

    elif name == "cifar10":
        train, val, test, n_train = get_cifar10(**kwargs)
        return {"train_loader": train, "val_loader": val, "test_loader": test, "num_train_samples": n_train}

    elif name == "cifar100":
        train, val, test, n_train = get_cifar100(**kwargs)
        return {"train_loader": train, "val_loader": val, "test_loader": test, "num_train_samples": n_train}

    elif name == "svhn":
        train, val, test, n_train = get_svhn(**kwargs)
        return {"train_loader": train, "val_loader": val, "test_loader": test, "num_train_samples": n_train}

    elif name == "adult":
        train, val, test, n_train, input_dim = get_adult(**kwargs)
        return {"train_loader": train, "val_loader": val, "test_loader": test, "num_train_samples": n_train, "input_dim": input_dim}

    elif name == "covertype":
        train, val, test, n_train, input_dim = get_covertype(**kwargs)
        return {"train_loader": train, "val_loader": val, "test_loader": test, "num_train_samples": n_train, "input_dim": input_dim}

    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: fashion_mnist, cifar10, cifar100, svhn, adult, covertype")
