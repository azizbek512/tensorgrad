from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from tensorgrad.tensor import Tensor, numerical_grad_check
from tensorgrad.conv import (
    Conv2d,
    compute_conv_out_dims,
    conv2d,
    im2col,
    run_calorimeter_cnn_benchmark,
    run_collider_mlp_benchmark,
)


def test_conv_out_dims_and_im2col_shapes() -> None:
    """Verify spatial output dimensions and im2col matrix shapes."""
    assert compute_conv_out_dims(8, 8, 3, 3, stride=1, padding=0) == (6, 6)
    assert compute_conv_out_dims(8, 8, 3, 3, stride=1, padding=1) == (8, 8)

    x_v2 = np.ones((1, 1, 8, 8), dtype=np.float64)
    cols_v2, h_out, w_out = im2col(x_v2, 3, 3, stride=1, padding=0)
    assert (h_out, w_out) == (6, 6)
    assert cols_v2.shape == (36, 9)

    x_w1 = np.ones((1, 1, 7, 7), dtype=np.float64)
    cols_w1, _, _ = im2col(x_w1, 3, 3, stride=1, padding=0)
    assert cols_w1.shape == (25, 9)


def test_col2im_overlap_accumulation_and_conv2d_gradcheck() -> None:
    """Verify col2im overlapping patch accumulation (+=) and Conv2d VJP gradcheck."""
    rng = np.random.default_rng(31)
    X = Tensor(rng.normal(size=(2, 2, 5, 5)))
    conv = Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, rng=rng)
    head_w = rng.normal(size=(2, 3, 5, 5))

    chk_x = numerical_grad_check(lambda t: (conv(t) * head_w).sum(), X)
    assert chk_x["passed"], f"Conv2d input X gradcheck (`col2im`) failed: rel_error={chk_x['rel_error']:.4f}"

    chk_w = numerical_grad_check(
        lambda w: (conv2d(X, w, conv.bias, stride=1, padding=1) * head_w).sum(),
        conv.weight,
    )
    assert chk_w["passed"], f"Conv2d weight gradcheck failed: rel_error={chk_w['rel_error']:.4f}"


def test_in_silico_physics_benchmarks() -> None:
    """Verify Collider MLP (>95% accuracy) and 2D Calorimeter Conv2d benchmarks."""
    res_mlp = run_collider_mlp_benchmark(seed=42, steps=120)
    assert res_mlp["final_loss"] < res_mlp["initial_loss"] * 0.35
    assert res_mlp["accuracy"] >= 0.95, f"Collider MLP accuracy below 95%: {res_mlp['accuracy']}"

    res_cnn = run_calorimeter_cnn_benchmark(seed=7, steps=45)
    assert res_cnn["final_loss"] < res_cnn["initial_loss"] * 0.4
    assert res_cnn["accuracy"] >= 0.90
