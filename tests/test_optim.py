from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from tensorgrad.tensor import Tensor, numerical_grad_check
from tensorgrad.nn import MLP
from tensorgrad.optim import Adam, SGD, cross_entropy_loss, mse_loss


def test_mse_loss_values_and_gradcheck() -> None:
    """Verify MSE loss evaluation and finite-difference gradient check."""
    pred = Tensor([10.0, 14.0])
    target = Tensor([12.0, 10.0], requires_grad=False)
    loss = mse_loss(pred, target)
    np.testing.assert_allclose(loss.data, 10.0)

    loss.backward()
    np.testing.assert_allclose(pred.grad, [-2.0, 4.0])

    rng = np.random.default_rng(11)
    x_mat = Tensor(rng.normal(size=(5, 3)))
    y_mat = rng.normal(size=(5, 3))
    chk = numerical_grad_check(lambda t: mse_loss(t, y_mat), x_mat)
    assert chk["passed"]


def test_cross_entropy_log_sum_exp_stability_and_batch_gradcheck() -> None:
    """Verify Log-Sum-Exp numerical stability at extreme logits and batch (N=6) VJP gradcheck."""
    extreme = Tensor([[1000.0, 1001.0, 1000.0]])
    loss_ext = cross_entropy_loss(extreme, [1])
    assert np.isfinite(loss_ext.data)
    loss_ext.backward()
    assert np.all(np.isfinite(extreme.grad))
    np.testing.assert_allclose(np.sum(extreme.grad), 0.0, atol=1e-12)

    rng = np.random.default_rng(23)
    logits = Tensor(rng.normal(size=(6, 4)))
    targets = np.array([0, 2, 1, 3, 2, 0], dtype=np.int64)

    chk = numerical_grad_check(lambda z: cross_entropy_loss(z, targets), logits)
    assert chk["passed"], f"Cross-Entropy batch gradcheck failed: rel_error={chk['rel_error']:.4f}"


def test_sgd_and_momentum_step() -> None:
    """Verify vanilla SGD and Momentum velocity updates."""
    theta = Tensor([10.0])
    theta.grad = np.array([4.0], dtype=np.float64)
    opt = SGD([theta], lr=0.5, momentum=0.0)
    opt.step()
    np.testing.assert_allclose(theta.data, [8.0])

    theta_m = Tensor([5.0])
    opt_m = SGD([theta_m], lr=0.1, momentum=0.9)
    opt_m.velocities[0] = np.array([2.0], dtype=np.float64)
    theta_m.grad = np.array([1.0], dtype=np.float64)
    opt_m.step()
    np.testing.assert_allclose(opt_m.velocities[0], [2.8])
    np.testing.assert_allclose(theta_m.data, [5.0 - 0.1 * 2.8])


def test_adam_bias_correction_and_mlp_convergence() -> None:
    """Verify Adam first-step bias correction and MLP training convergence."""
    theta = Tensor([1.0])
    theta.grad = np.array([8.0], dtype=np.float64)
    adam = Adam([theta], lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    adam.step()
    np.testing.assert_allclose(adam.m[0], [0.8])
    np.testing.assert_allclose(theta.data, [1.0 - 0.1 * 1.0], rtol=1e-6)

    rng = np.random.default_rng(99)
    X = Tensor(rng.normal(size=(40, 4)))
    y = (X.data[:, 0] * X.data[:, 1] > 0.0).astype(np.int64)

    mlp = MLP(in_dim=4, hidden_dims=[16, 8], out_dim=2, rng=rng)
    opt = Adam(mlp.parameters(), lr=0.05)

    initial_loss = float(cross_entropy_loss(mlp(X), y).data)
    for _ in range(35):
        opt.zero_grad()
        loss = cross_entropy_loss(mlp(X), y)
        loss.backward()
        opt.step()
    final_loss = float(cross_entropy_loss(mlp(X), y).data)
    assert final_loss < initial_loss * 0.4
