"""
AdaMuon optimizer — single-GPU implementation.

Faithfully reproduces the algorithm from:
  Si, C., Zhang, D., & Shen, W. (2025). AdaMuon: Adaptive Muon Optimizer.
  arXiv:2507.11005. https://github.com/Chongjie-Si/AdaMuon

Key components:
  1. Nesterov momentum accumulation
  2. Sign-stabilization of momentum (Theorem 1: f(x) = sign(x))
  3. Newton-Schulz polar decomposition (5 iterations)
  4. Element-wise second momentum on orthogonalized output
  5. RMS-aligned rescaling (γ = 0.2 * sqrt(max(m,n)))

Only applies to ≥2D parameters. 1D params (biases, norms) use standard Adam-like updates.
"""

import torch
from torch import Tensor
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """
    Approximate the polar decomposition U @ V^T of G using Newton-Schulz iteration.

    Coefficients (a, b, c) are from the polynomial f(x) = ax + bx^3 + cx^5
    tuned for rapid convergence of singular values toward 1.

    Reference: Bernstein & Newhouse (2024); Jordan et al. (2024).
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    # If tall matrix, work with transpose for efficiency
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Normalize by Frobenius norm
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


class AdaMuon(Optimizer):
    """
    AdaMuon: Adaptive Muon Optimizer (single-GPU variant).

    Args:
        params: Model parameters (iterable)
        lr: Learning rate (default: 0.02)
        weight_decay: Weight decay coefficient (default: 0.01)
        momentum: Momentum coefficient β (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-8,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # Apply weight decay (decoupled, like AdamW)
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # For 1D parameters (biases, norms), use simple SGD with momentum
                if p.ndim < 2:
                    state = self.state[p]
                    if "momentum_buffer_1d" not in state:
                        state["momentum_buffer_1d"] = torch.zeros_like(g)
                    buf = state["momentum_buffer_1d"]
                    buf.mul_(beta).add_(g)
                    if nesterov:
                        update = g.add(buf, alpha=beta)
                    else:
                        update = buf
                    p.add_(update, alpha=-lr)
                    continue

                # --- AdaMuon for ≥2D parameters ---
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                # Step 1: Momentum accumulation
                buf = state["momentum_buffer"]
                buf.mul_(beta).add_(g)

                # Step 2: Nesterov extrapolation
                if nesterov:
                    m = g.add(buf, alpha=beta)
                else:
                    m = buf

                # Step 3: Sign-stabilization (Theorem 1: f = sign)
                m_sign = torch.sign(m)

                # Step 4: Reshape for 2D polar decomposition (conv filters: [out, in, h, w] -> [out, in*h*w])
                if m_sign.ndim == 4:
                    orig_shape = m_sign.shape
                    m_sign = m_sign.view(m_sign.size(0), -1)
                else:
                    orig_shape = None

                # Step 5: Newton-Schulz orthogonalization
                O = zeropower_via_newtonschulz5(m_sign, steps=ns_steps)

                # Flatten for element-wise operations
                O_flat = O.flatten().to(p.dtype)

                # Step 6: Element-wise second momentum estimation on O_t
                if "v_buffer" not in state:
                    state["v_buffer"] = torch.zeros_like(O_flat)
                v = state["v_buffer"]
                v.mul_(beta).addcmul_(O_flat, O_flat, value=1 - beta)

                # Step 7: Variance-normalized update direction
                O_hat = O_flat / (v.sqrt() + eps)

                # Step 8: RMS-aligned rescaling
                # γ = 0.2 * sqrt(max(m, n)) — matches Adam's RMS ≈ 0.2
                scale = 0.2 * (min(p.shape) * max(p.shape)) ** 0.5 / (O_hat.norm() + eps)
                O_hat.mul_(scale)

                # Apply update
                p.add_(O_hat.view_as(p), alpha=-lr)

        return loss
