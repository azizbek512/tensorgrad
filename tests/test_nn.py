from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from tensorgrad.tensor import Tensor, numerical_grad_check
from tensorgrad.nn import Linear, MLP


def test_reshape_and_transpose_vjp() -> None:
    """Verify reshape and transpose (2D .T and 3D axes) VJP gradients."""
    rng = np.random.default_rng(7)
    x = Tensor(rng.normal(size=(2, 6)))

    chk_reshape = numerical_grad_check(
        lambda t: (t.reshape(3, 4) * np.arange(1, 13, dtype=np.float64).reshape(3, 4)).sum(),
        x,
    )
    assert chk_reshape["passed"], f"Reshape gradcheck failed: {chk_reshape['rel_error']}"

    chk_transpose = numerical_grad_check(
        lambda t: (t.T * np.arange(1, 13, dtype=np.float64).reshape(6, 2)).sum(),
        x,
    )
    assert chk_transpose["passed"], f"Transpose gradcheck failed: {chk_transpose['rel_error']}"

    x_3d = Tensor(rng.normal(size=(2, 3, 4)))
    w_3d = rng.normal(size=(3, 2, 4))
    chk_3d = numerical_grad_check(
        lambda t: (t.transpose((1, 0, 2)) * w_3d).sum(),
        x_3d,
    )
    assert chk_3d["passed"]


def test_matmul_vjp_shapes_and_gradcheck() -> None:
    """Verify matrix multiplication VJP shapes and finite-difference gradients."""
    rng = np.random.default_rng(13)
    X = Tensor(rng.normal(size=(16, 5)))
    Y = Tensor(rng.normal(size=(5, 8)))

    Z = X @ Y
    assert Z.shape == (16, 8)

    G = rng.normal(size=(16, 8))
    Z.backward(grad=G)

    assert X.grad.shape == (16, 5)
    assert Y.grad.shape == (5, 8)
    np.testing.assert_allclose(X.grad, G @ Y.data.T)
    np.testing.assert_allclose(Y.grad, X.data.T @ G)

    chk_x = numerical_grad_check(lambda t: ((t @ Y) * G).sum(), X)
    chk_y = numerical_grad_check(lambda t: ((X @ t) * G).sum(), Y)
    assert chk_x["passed"]
    assert chk_y["passed"]


def test_linear_he_init_and_zero_grad() -> None:
    """Verify Linear layer shapes, He initialization scale, and zero_grad()."""
    rng = np.random.default_rng(101)
    layer = Linear(in_features=4, out_features=10, bias=True, rng=rng)
    assert layer.weight.shape == (4, 10)
    assert layer.bias is not None and layer.bias.shape == (10,)

    big_layer = Linear(in_features=200, out_features=500, rng=rng)
    expected_std = np.sqrt(2.0 / 200.0)
    empirical_std = float(np.std(big_layer.weight.data))
    assert np.isclose(empirical_std, expected_std, rtol=0.05)

    layer.weight.grad = np.full_like(layer.weight.data, 5.0)
    layer.zero_grad()
    np.testing.assert_allclose(layer.weight.grad, 0.0)


def test_mlp_architecture_and_end_to_end_gradcheck() -> None:
    """Verify MLP parameter shapes, linear output head, and end-to-end backprop."""
    rng = np.random.default_rng(2026)
    mlp = MLP(in_dim=4, hidden_dims=[16, 8], out_dim=2, activation="relu", rng=rng)

    params = mlp.parameters()
    assert len(params) == 6
    assert [p.shape for p in params] == [(4, 16), (16,), (16, 8), (8,), (8, 2), (2,)]

    X = Tensor(rng.normal(size=(6, 4)))
    out = mlp(X)
    assert out.shape == (6, 2)

    head_weights = rng.normal(size=(6, 2))
    chk_input = numerical_grad_check(lambda t: (mlp(t) * head_weights).sum(), X)
    assert chk_input["passed"], f"MLP input gradcheck failed: {chk_input['rel_error']}"
