from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional, Tuple, Union
import numpy as np

from tensorgrad.tensor import Tensor
from tensorgrad.nn import Linear, MLP, Module
from tensorgrad.optim import Adam, cross_entropy_loss


def compute_conv_out_dims(
    h_in: int,
    w_in: int,
    k_h: int,
    k_w: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[int, int]:
    """
    Compute output spatial dimensions for 2D convolution:
        H_out = floor((H + 2*P - K_H) / S) + 1
        W_out = floor((W + 2*P - K_W) / S) + 1
    """
    h_out = (h_in + 2 * padding - k_h) // stride + 1
    w_out = (w_in + 2 * padding - k_w) // stride + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"Invalid convolution output dimensions: H_out={h_out}, W_out={w_out} "
            f"(H={h_in}, W={w_in}, K=({k_h},{k_w}), P={padding}, S={stride})"
        )
    return h_out, w_out


def im2col(
    x: np.ndarray,
    k_h: int,
    k_w: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[np.ndarray, int, int]:
    """
    Unroll all sliding `(K_H, K_W)` receptive fields from a 4D image batch `(N, C_in, H, W)`
    into a 2D matrix of shape `(N * H_out * W_out, C_in * K_H * K_W)`.
    """
    n_batch, c_in, h_in, w_in = x.shape
    h_out, w_out = compute_conv_out_dims(h_in, w_in, k_h, k_w, stride=stride, padding=padding)

    if padding > 0:
        x_padded = np.pad(
            x,
            pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0.0,
        )
    else:
        x_padded = x

    cols = np.zeros((n_batch, h_out, w_out, c_in, k_h, k_w), dtype=np.float64)
    for i in range(h_out):
        h_start = i * stride
        for j in range(w_out):
            w_start = j * stride
            cols[:, i, j, :, :, :] = x_padded[:, :, h_start : h_start + k_h, w_start : w_start + k_w]

    cols_2d = cols.reshape(n_batch * h_out * w_out, c_in * k_h * k_w)
    return cols_2d, h_out, w_out


def col2im(
    cols_2d: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    k_h: int,
    k_w: int,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """
    Adjoint VJP of `im2col`: accumulate column gradients `(N * H_out * W_out, C_in * K_H * K_W)`
    back onto the spatial image grid `(N, C_in, H, W)` using `+=` across overlapping patches.
    """
    n_batch, c_in, h_in, w_in = x_shape
    h_out, w_out = compute_conv_out_dims(h_in, w_in, k_h, k_w, stride=stride, padding=padding)

    cols = cols_2d.reshape(n_batch, h_out, w_out, c_in, k_h, k_w)
    dx_padded = np.zeros((n_batch, c_in, h_in + 2 * padding, w_in + 2 * padding), dtype=np.float64)

    for i in range(h_out):
        h_start = i * stride
        for j in range(w_out):
            w_start = j * stride
            dx_padded[:, :, h_start : h_start + k_h, w_start : w_start + k_w] += cols[:, i, j, :, :, :]

    if padding > 0:
        return dx_padded[:, :, padding : padding + h_in, padding : padding + w_in]
    return dx_padded


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    """
    2D convolution via `im2col` matrix multiplication and `col2im` gradient accumulation:
        x.shape      = (N, C_in, H, W)
        weight.shape = (C_out, C_in, K_H, K_W)
        bias.shape   = (C_out,)
        out.shape    = (N, C_out, H_out, W_out)
    """
    if x.ndim != 4 or weight.ndim != 4:
        raise ValueError(f"`x` and `weight` must be 4D tensors: x={x.shape}, w={weight.shape}")

    n_batch, c_in, h_in, w_in = x.shape
    c_out, w_c_in, k_h, k_w = weight.shape
    if c_in != w_c_in:
        raise ValueError(f"Input channels C_in={c_in} do not match filter C_in={w_c_in}")

    x_cols, h_out, w_out = im2col(x.data, k_h, k_w, stride=stride, padding=padding)
    w_cols = weight.data.reshape(c_out, c_in * k_h * k_w)

    out_cols = x_cols @ w_cols.T
    if bias is not None:
        out_cols = out_cols + bias.data.reshape(1, c_out)

    out_data = out_cols.reshape(n_batch, h_out, w_out, c_out).transpose(0, 3, 1, 2)

    children = (x, weight) if bias is None else (x, weight, bias)
    req_grad = x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    out = Tensor(out_data, requires_grad=req_grad, _children=children, _op="conv2d")

    def _backward() -> None:
        g_cols = out.grad.transpose(0, 2, 3, 1).reshape(n_batch * h_out * w_out, c_out)

        if bias is not None and bias.requires_grad:
            bias.grad = bias.grad + g_cols.sum(axis=0)

        if weight.requires_grad:
            dw_cols = g_cols.T @ x_cols
            weight.grad = weight.grad + dw_cols.reshape(weight.shape)

        if x.requires_grad:
            dx_cols = g_cols @ w_cols
            dx = col2im(dx_cols, x.shape, k_h, k_w, stride=stride, padding=padding)
            x.grad = x.grad + dx

    out._backward = _backward
    return out


class Conv2d(Module):
    """
    2D Convolutional layer with He (Kaiming) initialization:
        std = sqrt(2 / (in_channels * K_H * K_W))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if isinstance(kernel_size, int):
            self.k_h, self.k_w = kernel_size, kernel_size
        else:
            self.k_h, self.k_w = int(kernel_size[0]), int(kernel_size[1])
        self.stride = int(stride)
        self.padding = int(padding)

        gen = rng if rng is not None else np.random.default_rng()
        fan_in = float(self.in_channels * self.k_h * self.k_w)
        std = np.sqrt(2.0 / fan_in)

        w_data = gen.normal(
            loc=0.0,
            scale=std,
            size=(self.out_channels, self.in_channels, self.k_h, self.k_w),
        )
        self.weight = Tensor(w_data, requires_grad=True)
        self.bias: Optional[Tensor] = (
            Tensor(np.zeros((self.out_channels,), dtype=np.float64), requires_grad=True)
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)

    def __repr__(self) -> str:
        return (
            f"Conv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size=({self.k_h}, {self.k_w}), stride={self.stride}, padding={self.padding})"
        )


def run_collider_mlp_benchmark(seed: int = 42, steps: int = 120) -> dict:
    """
    In-Silico Benchmark 1:
    Non-linear 4D collider kinematics classification (resonant spherical shell Signal vs Background)
    trained with MLP and Adam.
    """
    rng = np.random.default_rng(seed)
    n_samples = 200
    x_raw = rng.normal(loc=0.0, scale=1.0, size=(n_samples, 4))
    radii_sq = np.sum(x_raw[:, :3] ** 2, axis=1)
    y_labels = ((radii_sq > 0.6) & (radii_sq < 2.4)).astype(np.int64)

    X = Tensor(x_raw, requires_grad=False)
    model = MLP(in_dim=4, hidden_dims=[32, 16], out_dim=2, activation="relu", rng=rng)
    optimizer = Adam(model.parameters(), lr=0.03)

    initial_loss = float(cross_entropy_loss(model(X), y_labels).data)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = cross_entropy_loss(logits, y_labels)
        loss.backward()
        optimizer.step()

    final_logits = model(X)
    final_loss = float(cross_entropy_loss(final_logits, y_labels).data)
    preds = np.argmax(final_logits.data, axis=1)
    accuracy = float(np.mean(preds == y_labels))

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "accuracy": accuracy,
    }


def run_calorimeter_cnn_benchmark(seed: int = 7, steps: int = 45) -> dict:
    """
    In-Silico Benchmark 2:
    8x8 2D Calorimeter shower classification (narrow photon/quark core vs wide gluon/hadron ring)
    trained with Conv2d -> ReLU -> Linear and Adam.
    """
    rng = np.random.default_rng(seed)
    n_samples = 48
    images = np.zeros((n_samples, 1, 8, 8), dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int64)

    grid_y, grid_x = np.meshgrid(np.linspace(-1.5, 1.5, 8), np.linspace(-1.5, 1.5, 8), indexing="ij")
    for idx in range(n_samples):
        cx = rng.uniform(-0.4, 0.4)
        cy = rng.uniform(-0.4, 0.4)
        r2 = (grid_x - cx) ** 2 + (grid_y - cy) ** 2
        if idx % 2 == 0:
            shower = 2.5 * np.exp(-r2 / 0.25)
            labels[idx] = 0
        else:
            shower = 1.4 * r2 * np.exp(-r2 / 0.9)
            labels[idx] = 1
        images[idx, 0] = shower + rng.normal(scale=0.05, size=(8, 8))

    X_img = Tensor(images, requires_grad=False)
    conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, rng=rng)
    head = Linear(in_features=4 * 8 * 8, out_features=2, rng=rng)
    optimizer = Adam(conv.parameters() + head.parameters(), lr=0.03)

    def forward_cnn(inp: Tensor) -> Tensor:
        feat = conv(inp).relu()
        flat = feat.reshape(inp.shape[0], -1)
        return head(flat)

    initial_loss = float(cross_entropy_loss(forward_cnn(X_img), labels).data)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = forward_cnn(X_img)
        loss = cross_entropy_loss(logits, labels)
        loss.backward()
        optimizer.step()

    final_logits = forward_cnn(X_img)
    final_loss = float(cross_entropy_loss(final_logits, labels).data)
    preds = np.argmax(final_logits.data, axis=1)
    accuracy = float(np.mean(preds == labels))

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("tensorgrad — Module 4: Conv2d (im2col/col2im) & Physics Benchmarks")
    print("=" * 72)

    res_mlp = run_collider_mlp_benchmark()
    print(
        f"[1] Collider MLP Benchmark  : Loss {res_mlp['initial_loss']:.4f} -> {res_mlp['final_loss']:.4f} | "
        f"Accuracy = {res_mlp['accuracy'] * 100:.1f}% (Target: >95%)"
    )

    res_cnn = run_calorimeter_cnn_benchmark()
    print(
        f"[2] Calorimeter Conv2d (8x8): Loss {res_cnn['initial_loss']:.4f} -> {res_cnn['final_loss']:.4f} | "
        f"Accuracy = {res_cnn['accuracy'] * 100:.1f}%"
    )
    print("=" * 72)
