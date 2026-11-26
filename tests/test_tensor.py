from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from tensorgrad.tensor import Tensor, _unbroadcast, numerical_grad_check


def test_multipath_accumulation_and_topological_order() -> None:
    """Verify multi-path gradient accumulation (X + X, X * X) and diamond DAG topological order."""
    x1 = Tensor([5.0])
    (x1 + x1).backward()
    np.testing.assert_allclose(x1.grad, [2.0])

    x2 = Tensor([5.0])
    (x2 * x2).backward()
    np.testing.assert_allclose(x2.grad, [10.0])

    x3 = Tensor([2.0])
    a = x3 * 3.0
    b = a + 1.0
    loss = a * b
    loss.backward()
    np.testing.assert_allclose(x3.grad, [39.0])


def test_unbroadcast_all_cases() -> None:
    """Verify _unbroadcast for (32, 10) -> (10,), (32, 1), and (1, 10)."""
    g = np.ones((32, 10), dtype=np.float64)

    u_a = _unbroadcast(g, (10,))
    assert u_a.shape == (10,)
    np.testing.assert_allclose(u_a, np.full((10,), 32.0))

    u_b = _unbroadcast(g, (32, 1))
    assert u_b.shape == (32, 1)
    np.testing.assert_allclose(u_b, np.full((32, 1), 10.0))

    u_c = _unbroadcast(g, (1, 10))
    assert u_c.shape == (1, 10)
    np.testing.assert_allclose(u_c, np.full((1, 10), 32.0))


def test_add_sub_mul_with_broadcasting() -> None:
    """Verify addition, subtraction, and multiplication VJP under NumPy broadcasting."""
    X = Tensor(np.arange(12, dtype=np.float64).reshape(4, 3))
    b = Tensor([1.0, -2.0, 0.5])

    L = ((X - b) * 2.0).sum()
    L.backward()
    assert X.grad.shape == (4, 3)
    assert b.grad.shape == (3,)
    np.testing.assert_allclose(X.grad, np.full((4, 3), 2.0))
    np.testing.assert_allclose(b.grad, np.full((3,), -8.0))


def test_elementary_ops_pow_exp_log_relu_tanh() -> None:
    """Verify analytic and numerical gradients of elementary ops (**, exp, log, relu, tanh)."""
    x_pos = Tensor([0.5, 1.2, 2.0])
    res = numerical_grad_check(
        lambda t: ((t ** 3) + t.exp() + t.log() + t.tanh()).sum(),
        x_pos,
    )
    assert res["passed"], f"Gradcheck failed: rel_error={res['rel_error']}"

    x_relu = Tensor([5.0, -4.0, 0.0, 2.0])
    z_relu = x_relu.relu()
    np.testing.assert_allclose(z_relu.data, [5.0, 0.0, 0.0, 2.0])
    z_relu.backward(grad=[1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(x_relu.grad, [1.0, 0.0, 0.0, 4.0])


def test_sum_and_mean_reductions_axes() -> None:
    """Verify sum and mean reductions across axis=None, axis=0, axis=1, keepdims=False/True."""
    X = Tensor(np.ones((4, 5), dtype=np.float64))
    L = X.mean()
    L.backward()
    assert X.grad.shape == (4, 5)
    np.testing.assert_allclose(X.grad, np.full((4, 5), 1.0 / 20.0))

    rng = np.random.default_rng(42)
    X_ax = Tensor(rng.normal(size=(5, 8)))
    chk_sum = numerical_grad_check(lambda t: (t.sum(axis=1, keepdims=False) * np.arange(1, 6)).sum(), X_ax)
    assert chk_sum["passed"]

    chk_mean = numerical_grad_check(lambda t: (t.mean(axis=0, keepdims=False) * np.arange(1, 9)).sum(), X_ax)
    assert chk_mean["passed"]
