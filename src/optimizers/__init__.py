"""Optimizer factory and registry."""

import torch
import torch.nn as nn
from typing import Any

from src.optimizers.adamuon import AdaMuon
from src.optimizers.kfac import KFAC


def create_optimizer(
    name: str,
    model: nn.Module,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Create an optimizer by name.

    Args:
        name: One of 'adam', 'adamuon', 'kfac'
        model: The neural network model
        **kwargs: Optimizer-specific keyword arguments

    Returns:
        Configured optimizer instance
    """
    name = name.lower()

    if name == "adam":
        lr = kwargs.get("lr", 1e-3)
        betas = tuple(kwargs.get("betas", (0.9, 0.999)))
        eps = kwargs.get("eps", 1e-8)
        weight_decay = kwargs.get("weight_decay", 0.0)
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    elif name == "adamuon":
        # AdaMuon: separate 2D params (AdaMuon) and 1D params (handled internally)
        lr = kwargs.get("lr", 0.02)
        weight_decay = kwargs.get("weight_decay", 0.01)
        momentum = kwargs.get("momentum", 0.95)
        nesterov = kwargs.get("nesterov", True)
        ns_steps = kwargs.get("ns_steps", 5)
        eps = kwargs.get("eps", 1e-8)
        return AdaMuon(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
        )

    elif name == "kfac":
        lr = kwargs.get("lr", 0.02)
        damping = kwargs.get("damping", 1e-3)
        cov_update_freq = kwargs.get("cov_update_freq", 10)
        inv_update_freq = kwargs.get("inv_update_freq", 100)
        momentum = kwargs.get("momentum", 0.9)
        weight_decay = kwargs.get("weight_decay", 0.0)
        cov_ema_decay = kwargs.get("cov_ema_decay", 0.95)
        return KFAC(
            model=model,
            lr=lr,
            damping=damping,
            cov_update_freq=cov_update_freq,
            inv_update_freq=inv_update_freq,
            momentum=momentum,
            weight_decay=weight_decay,
            cov_ema_decay=cov_ema_decay,
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}. Choose from: adam, adamuon, kfac")
