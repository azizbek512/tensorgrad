from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

from tensorgrad.tensor import ArrayLike, Tensor, _unbroadcast, numerical_grad_check


def matmul(x: Tensor, y: Union[Tensor, ArrayLike]) -> Tensor:
    """
    Matrix multiplication: Z = X @ Y.
    Backward VJP:
        dL/dX = G @ Y^T
        dL/dY = X^T @ G
    """
    y_tensor = x._ensure_tensor(y)
    out = Tensor(
        x.data @ y_tensor.data,
        requires_grad=x.requires_grad or y_tensor.requires_grad,
        _children=(x, y_tensor),
        _op="@",
    )

    def _backward() -> None:
        if x.requires_grad:
            y_t = np.swapaxes(y_tensor.data, -1, -2)
            x.grad = x.grad + _unbroadcast(out.grad @ y_t, x.shape)
        if y_tensor.requires_grad:
            x_t = np.swapaxes(x.data, -1, -2)
            y_tensor.grad = y_tensor.grad + _unbroadcast(x_t @ out.grad, y_tensor.shape)

    out._backward = _backward
    return out


def rmatmul(x: Tensor, other: ArrayLike) -> Tensor:
    return Tensor._ensure_tensor(other) @ x


def reshape(x: Tensor, *new_shape: Union[int, Tuple[int, ...]]) -> Tensor:
    """Reshape tensor: Z = X.reshape(new_shape) -> Backward: G.reshape(X.shape)."""
    if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
        target_shape = tuple(new_shape[0])
    else:
        target_shape = tuple(int(s) for s in new_shape)

    out = Tensor(
        x.data.reshape(target_shape),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="reshape",
    )

    def _backward() -> None:
        if x.requires_grad:
            x.grad = x.grad + out.grad.reshape(x.shape)

    out._backward = _backward
    return out


def transpose(x: Tensor, axes: Optional[Sequence[int]] = None) -> Tensor:
    """
    Permute tensor axes: Z = X.transpose(axes).
    Backward applies the inverse permutation `np.argsort(axes)`.
    """
    axes_tuple = None if axes is None else tuple(axes)
    out = Tensor(
        x.data.transpose(axes_tuple),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op="transpose",
    )

    def _backward() -> None:
        if x.requires_grad:
            if axes_tuple is None:
                inv_grad = out.grad.transpose()
            else:
                inv_axes = tuple(int(i) for i in np.argsort(axes_tuple))
                inv_grad = out.grad.transpose(inv_axes)
            x.grad = x.grad + inv_grad

    out._backward = _backward
    return out


Tensor.__matmul__ = matmul  # type: ignore[attr-defined]
Tensor.__rmatmul__ = rmatmul  # type: ignore[attr-defined]
Tensor.matmul = matmul  # type: ignore[attr-defined]
Tensor.reshape = reshape  # type: ignore[attr-defined]
Tensor.transpose = transpose  # type: ignore[attr-defined]
Tensor.T = property(lambda self: transpose(self))  # type: ignore[attr-defined]


class Module:
    """Base class for all neural network layers and models."""

    def parameters(self) -> List[Tensor]:
        params: List[Tensor] = []
        for val in self.__dict__.values():
            if isinstance(val, Tensor) and val.requires_grad:
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Tensor) and item.requires_grad:
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self) -> None:
        """Reset gradients of all trainable parameters to zero."""
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)


class Linear(Module):
    """
    Fully connected linear layer: Y = X @ W + b.
    Initialized with He (Kaiming) normal scaling: std = sqrt(2 / in_features).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        gen = rng if rng is not None else np.random.default_rng()

        std = np.sqrt(2.0 / float(self.in_features))
        w_data = gen.normal(loc=0.0, scale=std, size=(self.in_features, self.out_features))
        self.weight = Tensor(w_data, requires_grad=True)
        self.bias: Optional[Tensor] = (
            Tensor(np.zeros((self.out_features,), dtype=np.float64), requires_grad=True)
            if bias
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )


class ReLU(Module):
    """Rectified Linear Unit activation layer: max(0, X)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    """Hyperbolic tangent activation layer: tanh(X)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class Sequential(Module):
    """Sequential container chaining an ordered list of modules."""

    def __init__(self, *layers: Module) -> None:
        self.layers: List[Module] = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def __repr__(self) -> str:
        inner = ", ".join(repr(layer) for layer in self.layers)
        return f"Sequential({inner})"


class MLP(Module):
    """
    Multi-Layer Perceptron:
        in_dim -> hidden_dims[0] -> ... -> out_dim
    Applies non-linear activations between hidden layers and leaves the final layer linear.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        activation: str = "relu",
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        gen = rng if rng is not None else np.random.default_rng()
        dims = [int(in_dim)] + [int(h) for h in hidden_dims] + [int(out_dim)]

        layers: List[Module] = []
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1], bias=True, rng=gen))
            if i < len(dims) - 2:
                if activation.lower() == "relu":
                    layers.append(ReLU())
                elif activation.lower() == "tanh":
                    layers.append(Tanh())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")

        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def __repr__(self) -> str:
        return f"MLP({self.net})"


if __name__ == "__main__":
    print("=" * 72)
    print("tensorgrad — Module 2: Matrix VJP (@), Linear Layer & MLP")
    print("=" * 72)

    rng_demo = np.random.default_rng(42)

    A = Tensor(rng_demo.normal(size=(4, 4)))
    B = Tensor(rng_demo.normal(size=(4, 4)))
    C = (A @ B).reshape(2, 8).T
    print(f"[1] A.shape={A.shape}, B.shape={B.shape} -> (A@B).reshape(2,8).T.shape={C.shape}")

    mlp = MLP(in_dim=4, hidden_dims=[16, 8], out_dim=2, rng=rng_demo)
    X_batch = Tensor(rng_demo.normal(size=(8, 4)))
    logits = mlp(X_batch)
    total_params = sum(p.size for p in mlp.parameters())
    print(f"\n[2] MLP Architecture     : {mlp}")
    print(f"    Input X.shape        : {X_batch.shape} -> Output logits.shape: {logits.shape}")
    print(f"    Total Parameters     : {total_params} (6 Tensors: 3 weights, 3 biases)")
    print("=" * 72)
