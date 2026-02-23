"""Image dataset loaders with standard augmentations for benchmarking."""

from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T


# Standard normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
FMNIST_MEAN = (0.2860,)
FMNIST_STD = (0.3530,)


def get_fashion_mnist(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load Fashion-MNIST with standard transforms."""
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(FMNIST_MEAN, FMNIST_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(FMNIST_MEAN, FMNIST_STD),
    ])

    full_train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform_test)

    # Split train into train/val
    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    # Override val set transform to test transform
    val_set.dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    return train_loader, val_loader, test_loader, train_size


def get_cifar10(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load CIFAR-10 with standard augmentations."""
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    val_set.dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    return train_loader, val_loader, test_loader, train_size


def get_cifar100(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load CIFAR-100 with standard augmentations."""
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform_test)

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    val_set.dataset = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    return train_loader, val_loader, test_loader, train_size


def get_svhn(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load SVHN with standard transforms."""
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(SVHN_MEAN, SVHN_STD),
    ])

    full_train = torchvision.datasets.SVHN(data_dir, split="train", download=True, transform=transform_train)
    test_set = torchvision.datasets.SVHN(data_dir, split="test", download=True, transform=transform_test)

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    val_set.dataset = torchvision.datasets.SVHN(data_dir, split="train", download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    return train_loader, val_loader, test_loader, train_size
