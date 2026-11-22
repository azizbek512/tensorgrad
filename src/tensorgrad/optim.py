from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Iterable, List, Sequence, Tuple, Union
import numpy as np

from tensorgrad.tensor import ArrayLike, Tensor, numerical_grad_check
from tensorgrad.nn import MLP


def mse_loss(pred: Tensor, target: Union[Tensor, ArrayLike]) -> Tensor:
    """
    Mean Squared Error (MSE):
        L_MSE = (1 / N) * sum((pred_i - target_i)^2)
    """
    target_t = pred._ensure_tensor(target)
    diff = pred - target_t
    return (diff ** 2).mean()


def softmax(logits: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Compute numerically stable Softmax probabilities via the Log-Sum-Exp shift."""
    z = logits.data if isinstance(logits, Tensor) else np.asarray(logits, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(1, -1)
    m = np.max(z, axis=1, keepdims=True)
    exp_shifted = np.exp(z - m)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def cross_entropy_loss(
    logits: Tensor,
    targets: Union[np.ndarray, Sequence[int], Tensor],
) -> Tensor:
    """
    Numerically stable Log-Sum-Exp Softmax + Cross-Entropy Loss:
        m_i = max_j z_{i,j}
        LSE(z_i) = m_i + ln(sum_j exp(z_{i,j} - m_i))
        L_CE(Z, y) = (1 / N) * sum_i (LSE(z_i) - z_{i, y_i})
    Analytic Backward VJP:
        dL_CE / dz_{i,j} = (1 / N) * (P_{i,j} - I(j == y_i)) * G
    """
    z = logits.data
    if z.ndim != 2:
        raise ValueError(f"`logits` must be 2D `(N, C)`, received shape: {z.shape}")

    n_samples, num_classes = z.shape
    raw_targets = targets.data if isinstance(targets, Tensor) else np.asarray(targets)

    if raw_targets.ndim == 1:
        if raw_targets.shape[0] != n_samples:
            raise ValueError(f"`targets` length {raw_targets.shape[0]} does not match N={n_samples}")
        class_indices = raw_targets.astype(np.int64)
        one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), class_indices] = 1.0
    elif raw_targets.shape == (n_samples, num_classes):
        one_hot = raw_targets.astype(np.float64)
    else:
        raise ValueError(f"Invalid `targets` shape: {raw_targets.shape}")

    # Log-Sum-Exp shift: subtract row-wise maximum m_i to prevent exp() overflow
    m = np.max(z, axis=1, keepdims=True)
    shifted = z - m
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)

    lse = m + np.log(sum_exp)
    probs = exp_shifted / sum_exp

    correct_logits = np.sum(z * one_hot, axis=1, keepdims=True)
    loss_val = float(np.mean(lse - correct_logits))

    out = Tensor(
        loss_val,
        requires_grad=logits.requires_grad,
        _children=(logits,),
        _op="cross_entropy",
    )

    def _backward() -> None:
        if logits.requires_grad:
            g_scalar = float(np.asarray(out.grad).squeeze())
            grad_z = ((probs - one_hot) / float(n_samples)) * g_scalar
            logits.grad = logits.grad + grad_z

    out._backward = _backward
    return out


class Optimizer:
    """Base class for gradient-based parameter optimizers."""

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self.params: List[Tensor] = [p for p in params if p.requires_grad]
        self.lr: float = float(lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with Momentum and L2 Weight Decay:
        v_t = mu * v_{t-1} + (g_t + weight_decay * theta_{t-1})
        theta_t = theta_{t-1} - lr * v_t
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.momentum: float = float(momentum)
        self.weight_decay: float = float(weight_decay)
        self.velocities: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0.0:
                self.velocities[i] = self.momentum * self.velocities[i] + g
                step_vec = self.velocities[i]
            else:
                step_vec = g
            p.data = p.data - self.lr * step_vec


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam):
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t,       m_hat = m_t / (1 - beta1^t)
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2,     v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.beta1: float = float(betas[0])
        self.beta2: float = float(betas[1])
        self.eps: float = float(eps)
        self.weight_decay: float = float(weight_decay)
        self.t: int = 0
        self.m: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]
        self.v: List[np.ndarray] = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        b1_corr = 1.0 - (self.beta1 ** self.t)
        b2_corr = 1.0 - (self.beta2 ** self.t)

        for i, p in enumerate(self.params):
            g = p.grad
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / b1_corr
            v_hat = self.v[i] / b2_corr

            p.data = p.data - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps))


if __name__ == "__main__":
    print("=" * 72)
    print("tensorgrad — Module 3: Log-Sum-Exp Cross-Entropy, SGD & Adam")
    print("=" * 72)

    extreme_logits = Tensor([[1000.0, 1001.0, 1000.0]])
    ce_extreme = cross_entropy_loss(extreme_logits, [1])
    ce_extreme.backward()
    print(f"[1] Extreme Logits       : {extreme_logits.data[0]}")
    print(f"    Cross-Entropy Loss   : {ce_extreme.data:.6f} (finite, no overflow)")
    print(f"    Analytic z.grad      : {np.round(extreme_logits.grad[0], 6)}")

    rng_demo = np.random.default_rng(42)
    X_syn = Tensor(rng_demo.normal(size=(32, 4)))
    y_syn = (X_syn.data[:, 0] ** 2 + X_syn.data[:, 1] ** 2 > 1.2).astype(np.int64)

    model = MLP(in_dim=4, hidden_dims=[16, 8], out_dim=2, rng=rng_demo)
    optimizer = Adam(model.parameters(), lr=0.05)

    loss_start = float(cross_entropy_loss(model(X_syn), y_syn).data)
    for _ in range(1, 41):
        optimizer.zero_grad()
        loss = cross_entropy_loss(model(X_syn), y_syn)
        loss.backward()
        optimizer.step()
    loss_end = float(cross_entropy_loss(model(X_syn), y_syn).data)

    print(f"\n[2] MLP + Adam Training  : Initial Loss = {loss_start:.4f} -> Step 40 Loss = {loss_end:.4f}")
    print("=" * 72)
