"""
K-FAC (Kronecker-Factored Approximate Curvature) optimizer.

Based on:
  Martens, J. & Grosse, R. (2015). Optimizing Neural Networks with
  Kronecker-factored Approximate Curvature. ICML 2015.

Implementation covers:
  - Linear and Conv2d layers
  - EMA covariance estimation for input activations (A) and output gradients (G)
  - Periodic factor updates and inversions
  - Factored Tikhonov damping
  - Cholesky-based inversion with eigendecomposition fallback
  - Modules not covered by K-FAC fall back to SGD-momentum updates
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from typing import Optional


class KFACFactor:
    """Stores and manages a single Kronecker factor (A or G) for one layer."""

    def __init__(self, shape: int, device: torch.device, ema_decay: float = 0.95):
        self.shape = shape
        self.ema_decay = ema_decay
        self.cov: Optional[Tensor] = None  # Running EMA of covariance
        self.inv: Optional[Tensor] = None  # Cached inverse
        self.device = device

    def update(self, data: Tensor):
        """Update covariance estimate with new data using EMA."""
        # data: [batch, dim] — compute outer product average
        batch_size = data.size(0)
        cov_new = (data.t() @ data) / batch_size

        if self.cov is None:
            self.cov = cov_new
        else:
            self.cov.mul_(self.ema_decay).add_(cov_new, alpha=1 - self.ema_decay)

    def compute_inverse(self, damping: float):
        """Compute damped inverse using Cholesky (with eigendecomposition fallback)."""
        if self.cov is None:
            return

        damped = self.cov + damping * torch.eye(self.shape, device=self.device)

        # 1. Try Cholesky (fastest, requires strict positive definiteness)
        try:
            L = torch.linalg.cholesky(damped)
            self.inv = torch.cholesky_inverse(L)
            return
        except torch.linalg.LinAlgError:
            pass

        # 2. Try symmetric eigendecomposition (handles ill-conditioned positive semi-definite)
        try:
            eigvals, eigvecs = torch.linalg.eigh(damped)
            # Clamp eigenvalues to strictly positive to force positive definiteness
            eigvals = eigvals.clamp(min=1e-8)
            self.inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.t()
            return
        except torch.linalg.LinAlgError:
            pass
            
        # 3. Try adding jitter and re-running eigendecomposition
        try:
            jitter = 1e-4 * torch.eye(self.shape, device=self.device)
            eigvals, eigvecs = torch.linalg.eigh(damped + jitter)
            eigvals = eigvals.clamp(min=1e-8)
            self.inv = eigvecs @ torch.diag(1.0 / eigvals) @ eigvecs.t()
            return
        except torch.linalg.LinAlgError:
            pass

        # 4. Final fallback: Pseudo-inverse (slowest but most robust)
        try:
            self.inv = torch.linalg.pinv(damped)
        except Exception:
            # If all else fails, just use scaled identity to prevent crashing
            self.inv = torch.eye(self.shape, device=self.device) / damping


class KFACLayer:
    """Manages K-FAC factors for a single Linear or Conv2d layer."""

    def __init__(
        self,
        module: nn.Module,
        ema_decay: float = 0.95,
    ):
        self.module = module
        self.ema_decay = ema_decay

        # Determine dimensions
        if isinstance(module, nn.Linear):
            d_in = module.in_features
            d_out = module.out_features
            self.has_bias = module.bias is not None
            if self.has_bias:
                d_in += 1  # Bias augmentation
        elif isinstance(module, nn.Conv2d):
            d_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            d_out = module.out_channels
            self.has_bias = module.bias is not None
            if self.has_bias:
                d_in += 1
        else:
            raise ValueError(f"Unsupported module type: {type(module)}")

        self.d_in = d_in
        self.d_out = d_out

        device = next(module.parameters()).device
        self.A_factor = KFACFactor(d_in, device, ema_decay)
        self.G_factor = KFACFactor(d_out, device, ema_decay)

        # Storage for forward/backward hook data
        self._input_data: Optional[Tensor] = None
        self._grad_output_data: Optional[Tensor] = None

    def save_input(self, input_data: Tensor):
        """Called by the forward hook."""
        if isinstance(self.module, nn.Conv2d):
            # Unfold input patches: [B, C_in, H, W] -> [B*H'*W', C_in*kH*kW]
            unfolded = torch.nn.functional.unfold(
                input_data,
                kernel_size=self.module.kernel_size,
                stride=self.module.stride,
                padding=self.module.padding,
            )
            # unfolded: [B, C_in*kH*kW, L] -> [B*L, C_in*kH*kW]
            B, D, L = unfolded.shape
            a = unfolded.permute(0, 2, 1).reshape(B * L, D)
        else:
            a = input_data.reshape(-1, input_data.size(-1))

        # Bias augmentation
        if self.has_bias:
            ones = torch.ones(a.size(0), 1, device=a.device, dtype=a.dtype)
            a = torch.cat([a, ones], dim=1)

        self._input_data = a.detach()

    def save_grad_output(self, grad_output: Tensor):
        """Called by the backward hook."""
        if isinstance(self.module, nn.Conv2d):
            # grad_output: [B, C_out, H', W'] -> [B*H'*W', C_out]
            B, C, H, W = grad_output.shape
            g = grad_output.permute(0, 2, 3, 1).reshape(B * H * W, C)
        else:
            g = grad_output.reshape(-1, grad_output.size(-1))

        self._grad_output_data = g.detach()

    def update_factors(self):
        """Update A and G covariance estimates."""
        if self._input_data is not None:
            self.A_factor.update(self._input_data)
        if self._grad_output_data is not None:
            self.G_factor.update(self._grad_output_data)

    def compute_inverses(self, damping: float):
        """Compute damped inverses using factored Tikhonov damping."""
        if self.A_factor.cov is None or self.G_factor.cov is None:
            return

        # Factored damping: π = sqrt(tr(A)/d_in / (tr(G)/d_out))
        tr_A = self.A_factor.cov.trace().item()
        tr_G = self.G_factor.cov.trace().item()

        if tr_G > 0 and tr_A > 0:
            pi = math.sqrt((tr_A / self.d_in) / (tr_G / self.d_out))
        else:
            pi = 1.0

        damping_A = pi * math.sqrt(damping)
        damping_G = math.sqrt(damping) / pi

        self.A_factor.compute_inverse(damping_A)
        self.G_factor.compute_inverse(damping_G)

    def precondition_gradient(self) -> Optional[Tensor]:
        """Compute preconditioned gradient: G^{-1} @ grad @ A^{-1}."""
        if self.A_factor.inv is None or self.G_factor.inv is None:
            return None

        weight = self.module.weight
        grad = weight.grad
        if grad is None:
            return None

        if isinstance(self.module, nn.Conv2d):
            # Reshape grad: [C_out, C_in, kH, kW] -> [C_out, C_in*kH*kW]
            grad_2d = grad.reshape(grad.size(0), -1)
        else:
            grad_2d = grad

        # Augment gradient with bias gradient if applicable
        if self.has_bias and self.module.bias is not None and self.module.bias.grad is not None:
            grad_2d = torch.cat([grad_2d, self.module.bias.grad.unsqueeze(1)], dim=1)

        # Preconditioned gradient: G^{-1} @ grad_2d @ A^{-1}
        precond = self.G_factor.inv @ grad_2d @ self.A_factor.inv

        if self.has_bias and self.module.bias is not None:
            # Split back weight and bias gradients
            weight_precond = precond[:, :-1]
            bias_precond = precond[:, -1]
        else:
            weight_precond = precond
            bias_precond = None

        if isinstance(self.module, nn.Conv2d):
            weight_precond = weight_precond.reshape_as(weight)

        return weight_precond, bias_precond


class KFAC(Optimizer):
    """
    K-FAC optimizer for PyTorch.

    Args:
        model: The neural network model
        lr: Learning rate (default: 0.02)
        damping: Tikhonov damping coefficient (default: 1e-3)
        cov_update_freq: Update covariance factors every N steps (default: 10)
        inv_update_freq: Recompute inverses every N steps (default: 100)
        momentum: SGD-like momentum for the update direction (default: 0.9)
        weight_decay: L2 weight decay (default: 0.0)
        cov_ema_decay: EMA decay for covariance estimation (default: 0.95)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.02,
        damping: float = 1e-3,
        cov_update_freq: int = 10,
        inv_update_freq: int = 100,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        cov_ema_decay: float = 0.95,
    ):
        # Collect all parameters
        params = list(model.parameters())
        defaults = dict(lr=lr, damping=damping, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.model = model
        self.damping = damping
        self.cov_update_freq = cov_update_freq
        self.inv_update_freq = inv_update_freq
        self.cov_ema_decay = cov_ema_decay
        self._step_count = 0

        # Register K-FAC layers and hooks
        self.kfac_layers: dict[nn.Module, KFACLayer] = {}
        self._hooks = []
        self._covered_params = set()

        self._disable_inplace(model)
        self._register_modules(model)

    @staticmethod
    def _disable_inplace(model: nn.Module):
        """Disable inplace operations in the model to avoid backward hook conflicts."""
        for module in model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

    def _register_modules(self, model: nn.Module):
        """Register forward/backward hooks for supported layers."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                kfac_layer = KFACLayer(module, self.cov_ema_decay)
                self.kfac_layers[module] = kfac_layer

                # Track which parameters are covered by K-FAC
                self._covered_params.add(id(module.weight))
                if module.bias is not None:
                    self._covered_params.add(id(module.bias))

                # Forward hook: save input activations
                def fwd_hook(mod, inp, out, kl=kfac_layer):
                    if mod.training:
                        kl.save_input(inp[0])
                handle = module.register_forward_hook(fwd_hook)
                self._hooks.append(handle)

                # Backward hook: save gradient of output
                def bwd_hook(mod, grad_input, grad_output, kl=kfac_layer):
                    if mod.training:
                        kl.save_grad_output(grad_output[0])
                handle = module.register_full_backward_hook(bwd_hook)
                self._hooks.append(handle)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # Update covariance factors periodically
        if self._step_count % self.cov_update_freq == 0:
            for kl in self.kfac_layers.values():
                kl.update_factors()

        # Recompute inverses periodically
        if self._step_count % self.inv_update_freq == 0:
            for kl in self.kfac_layers.values():
                kl.compute_inverses(self.damping)

        # Apply preconditioned gradients to K-FAC-covered modules
        for module, kl in self.kfac_layers.items():
            result = kl.precondition_gradient()

            weight = module.weight
            if weight.grad is None:
                continue

            for group in self.param_groups:
                lr = group["lr"]
                momentum = group["momentum"]
                wd = group["weight_decay"]

                if result is not None:
                    weight_precond, bias_precond = result

                    # Momentum buffer for weight
                    state_w = self.state[weight]
                    if "momentum_buffer" not in state_w:
                        state_w["momentum_buffer"] = torch.zeros_like(weight)
                    state_w["momentum_buffer"].mul_(momentum).add_(weight_precond)

                    # Weight decay
                    if wd > 0:
                        weight.mul_(1 - lr * wd)

                    weight.add_(state_w["momentum_buffer"], alpha=-lr)

                    # Bias update
                    if bias_precond is not None and module.bias is not None:
                        state_b = self.state[module.bias]
                        if "momentum_buffer" not in state_b:
                            state_b["momentum_buffer"] = torch.zeros_like(module.bias)
                        state_b["momentum_buffer"].mul_(momentum).add_(bias_precond)
                        module.bias.add_(state_b["momentum_buffer"], alpha=-lr)
                else:
                    # Fallback: standard SGD with momentum
                    grad = weight.grad
                    state_w = self.state[weight]
                    if "momentum_buffer" not in state_w:
                        state_w["momentum_buffer"] = torch.zeros_like(weight)
                    state_w["momentum_buffer"].mul_(momentum).add_(grad)
                    if wd > 0:
                        weight.mul_(1 - lr * wd)
                    weight.add_(state_w["momentum_buffer"], alpha=-lr)

                    if module.bias is not None and module.bias.grad is not None:
                        state_b = self.state[module.bias]
                        if "momentum_buffer" not in state_b:
                            state_b["momentum_buffer"] = torch.zeros_like(module.bias)
                        state_b["momentum_buffer"].mul_(momentum).add_(module.bias.grad)
                        module.bias.add_(state_b["momentum_buffer"], alpha=-lr)
                break  # Only use first param group

        # SGD-momentum fallback for uncovered parameters (BatchNorm, etc.)
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if id(p) in self._covered_params:
                    continue  # Already handled above

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                state["momentum_buffer"].mul_(momentum).add_(p.grad)
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(state["momentum_buffer"], alpha=-lr)

        return loss

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
