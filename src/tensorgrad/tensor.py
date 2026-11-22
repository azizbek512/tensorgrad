from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Callable, Optional, Tuple, Union
import numpy as np


ArrayLike = Union[int, float, list, tuple, np.ndarray]
AxisLike = Optional[Union[int, Tuple[int, ...]]]


def _unbroadcast(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Sum a broadcasted gradient tensor back to its original `target_shape`."""
    out = np.asarray(grad, dtype=np.float64)
    # Step 1: Sum out leading axes added by broadcasting (keepdims=False)
    while out.ndim > len(target_shape):
        out = out.sum(axis=0)
    # Step 2: Sum along axes that were length 1 before broadcasting (keepdims=True)
    for axis, dim in enumerate(target_shape):
        if dim == 1 and out.shape[axis] > 1:
            out = out.sum(axis=axis, keepdims=True)
    return out


def _restore_reduced_shape(
    grad: np.ndarray,
    input_shape: Tuple[int, ...],
    axis: AxisLike,
    keepdims: bool,
) -> np.ndarray:
    """Restore reduced axes as singleton dimensions `(..., 1, ...)` when `keepdims=False`."""
    if keepdims or axis is None:
        return grad
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    ndim = len(input_shape)
    norm_axes = sorted(a if a >= 0 else a + ndim for a in axes)
    out = grad
    for a in norm_axes:
        out = np.expand_dims(out, axis=a)
    return out


class Tensor:
    """Reverse-mode automatic differentiation (VJP) node over `float64` NumPy arrays."""

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = True,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: np.ndarray = np.zeros_like(self.data, dtype=np.float64)
        self._prev: set[Tensor] = set(_children)
        self._backward: Callable[[], None] = lambda: None
        self._op: str = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def zero_grad(self) -> None:
        """Reset accumulated gradient to zero."""
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    @staticmethod
    def _ensure_tensor(val: Union["Tensor", ArrayLike]) -> "Tensor":
        if isinstance(val, Tensor):
            return val
        return Tensor(val, requires_grad=False)

    def __add__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._ensure_tensor(other)
        out = Tensor(
            self.data + other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="+",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t.grad = other_t.grad + _unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __sub__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._ensure_tensor(other)
        out = Tensor(
            self.data - other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="-",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t.grad = other_t.grad - _unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._ensure_tensor(other) - self

    def __neg__(self) -> "Tensor":
        return self * (-1.0)

    def __mul__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._ensure_tensor(other)
        out = Tensor(
            self.data * other_t.data,
            requires_grad=self.requires_grad or other_t.requires_grad,
            _children=(self, other_t),
            _op="*",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + _unbroadcast(out.grad * other_t.data, self.shape)
            if other_t.requires_grad:
                other_t.grad = other_t.grad + _unbroadcast(out.grad * self.data, other_t.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __pow__(self, power: Union[int, float]) -> "Tensor":
        if not isinstance(power, (int, float)):
            raise TypeError("Exponent must be an `int` or `float` scalar.")
        p = float(power)
        out = Tensor(
            self.data ** p,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"**{p}",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad * (p * (self.data ** (p - 1.0)))

        out._backward = _backward
        return out

    def __truediv__(self, other: Union["Tensor", ArrayLike]) -> "Tensor":
        other_t = self._ensure_tensor(other)
        return self * (other_t ** -1.0)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._ensure_tensor(other) * (self ** -1.0)

    def exp(self) -> "Tensor":
        exp_x = np.exp(self.data)
        out = Tensor(
            exp_x,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad * out.data

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad / self.data

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward() -> None:
            if self.requires_grad:
                self.grad = self.grad + out.grad * (1.0 - out.data ** 2)

        out._backward = _backward
        return out

    def sum(self, axis: AxisLike = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        def _backward() -> None:
            if self.requires_grad:
                g = _restore_reduced_shape(out.grad, self.shape, axis, keepdims)
                self.grad = self.grad + np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis: AxisLike = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="mean",
        )
        n_reduced = float(self.size) / float(out.size)

        def _backward() -> None:
            if self.requires_grad:
                g = _restore_reduced_shape(out.grad / n_reduced, self.shape, axis, keepdims)
                self.grad = self.grad + np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        """Propagate gradients through the computation graph in reverse topological order."""
        if grad is None:
            if self.size != 1:
                raise ValueError(
                    "Initial `grad` must be provided when calling `backward()` on a non-scalar tensor."
                )
            self.grad = np.ones_like(self.data, dtype=np.float64)
        else:
            grad_arr = np.asarray(grad, dtype=np.float64)
            if grad_arr.shape != self.shape:
                raise ValueError(f"Gradient shape {grad_arr.shape} does not match tensor shape {self.shape}")
            self.grad = grad_arr

        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op!r}, data={np.array2string(self.data, precision=4)})"


def numerical_grad_check(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    eps: float = 1e-5,
    tol: float = 1e-5,
) -> dict:
    """
    Validate analytic backprop gradients against centered finite differences:
        df/dX_i ≈ (f(X + eps * e_i) - f(X - eps * e_i)) / (2 * eps)
    """
    x_clean = Tensor(x.data.copy(), requires_grad=True)
    loss = func(x_clean)
    if loss.size != 1:
        loss = loss.sum()
    loss.backward()
    analytic_grad = x_clean.grad.copy()

    numerical_grad = np.zeros_like(x.data, dtype=np.float64)
    x_work = x.data.copy()

    it = np.nditer(x_work, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = float(x_work[idx])

        x_work[idx] = orig + eps
        f_plus = float(func(Tensor(x_work, requires_grad=False)).sum().data)

        x_work[idx] = orig - eps
        f_minus = float(func(Tensor(x_work, requires_grad=False)).sum().data)

        x_work[idx] = orig
        numerical_grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()

    diff_norm = float(np.linalg.norm(analytic_grad - numerical_grad))
    denom = max(float(np.linalg.norm(analytic_grad)), float(np.linalg.norm(numerical_grad)), 1e-8)
    rel_error = diff_norm / denom
    passed = bool(rel_error < tol)

    return {
        "passed": passed,
        "rel_error": rel_error,
        "analytic_grad": analytic_grad,
        "numerical_grad": numerical_grad,
    }


if __name__ == "__main__":
    print("=" * 72)
    print("tensorgrad — Module 1: Tensor Autograd Engine & Numerical Gradcheck")
    print("=" * 72)

    x_demo = Tensor([1.0, 2.0, -1.5])
    loss_demo = (x_demo * x_demo + x_demo.tanh()).sum()
    loss_demo.backward()
    print(f"[1] Input X.data         : {x_demo.data}")
    print(f"    Loss = sum(X^2+tanh) : {loss_demo.data:.6f}")
    print(f"    Analytic X.grad      : {np.round(x_demo.grad, 6)}")

    X_mat = Tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]])
    b_vec = Tensor([10.0, 20.0])
    out_b = ((X_mat + b_vec).relu() * 2.0).mean()
    out_b.backward()
    print(f"\n[2] Broadcasting VJP     : X.shape={X_mat.shape}, b.shape={b_vec.shape}")
    print(f"    b.grad (shape={b_vec.grad.shape}) : {np.round(b_vec.grad, 6)}")

    chk = numerical_grad_check(
        lambda t: ((t ** 2) * t.exp() + t.relu() + t.tanh()).mean(),
        Tensor([[0.5, -0.7, 1.2], [0.3, 0.9, -0.4]]),
    )
    print(f"\n[3] Numerical Gradcheck  : passed={chk['passed']}, rel_error={chk['rel_error']:.3e}")
    print("=" * 72)
