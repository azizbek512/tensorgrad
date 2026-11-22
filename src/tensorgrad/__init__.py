"""tensorgrad — Pure-NumPy Reverse-Mode Automatic Differentiation & Deep Learning Engine."""

from tensorgrad.tensor import Tensor, _unbroadcast, numerical_grad_check
from tensorgrad.nn import Linear, MLP, Module, ReLU, Sequential, Tanh, matmul, reshape, transpose
from tensorgrad.optim import Adam, Optimizer, SGD, cross_entropy_loss, mse_loss, softmax
from tensorgrad.conv import (
    Conv2d,
    col2im,
    compute_conv_out_dims,
    conv2d,
    im2col,
    run_calorimeter_cnn_benchmark,
    run_collider_mlp_benchmark,
)

__all__ = [
    "Tensor",
    "_unbroadcast",
    "numerical_grad_check",
    "matmul",
    "reshape",
    "transpose",
    "Module",
    "Linear",
    "ReLU",
    "Tanh",
    "Sequential",
    "MLP",
    "mse_loss",
    "cross_entropy_loss",
    "softmax",
    "Optimizer",
    "SGD",
    "Adam",
    "compute_conv_out_dims",
    "im2col",
    "col2im",
    "conv2d",
    "Conv2d",
    "run_collider_mlp_benchmark",
    "run_calorimeter_cnn_benchmark",
]
