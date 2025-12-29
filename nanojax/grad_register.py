from typing import Callable, Union
import numpy as np


_PRIMITIVE_GRAD: dict[Callable, Callable] = {}


def get_grad_func(func: Callable) -> Callable:
    return _PRIMITIVE_GRAD[func]


def _register_grad(func: Callable, grad_func: Callable) -> None:
    _PRIMITIVE_GRAD[func] = grad_func


def _unbroadcast(grad: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Sum gradient along broadcasted dimensions to match original shape.

    Args:
        grad: Gradient with potentially broadcasted shape
        original_shape: Original shape before broadcasting

    Returns:
        Gradient reduced to match original_shape
    """
    # Sum along prepended dimensions (when ndim increased)
    ndim_added = grad.ndim - len(original_shape)
    for _ in range(ndim_added):
        grad = grad.sum(axis=0)

    # Sum along dimensions that were size 1 in original
    for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
        if orig_dim == 1 and grad_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


def _add_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    grad_a = _unbroadcast(grad_out, a.shape)
    grad_b = _unbroadcast(grad_out, b.shape)
    return (grad_a, grad_b)


def _sub_grad(grad_out, a, b, **_kwargs) -> tuple[Union[np.ndarray, None], ...]:
    grad_a = _unbroadcast(grad_out, a.shape)

    if isinstance(b, np.ndarray):
        grad_b = _unbroadcast(-grad_out, b.shape)
    else:
        grad_b = None

    return (grad_a, grad_b)


def _mul_grad(grad_out, a, b, **_kwargs) -> tuple[Union[np.ndarray, None], ...]:
    grad_a = _unbroadcast(grad_out * b, a.shape)

    if isinstance(b, np.ndarray):
        grad_b = _unbroadcast(grad_out * a, b.shape)
    else:
        grad_b = None

    return (grad_a, grad_b)


def _div_grad(grad_out, a, b, **_kwargs) -> tuple[Union[np.ndarray, None], ...]:
    grad_a = _unbroadcast(grad_out / b, a.shape)

    if isinstance(b, np.ndarray):
        grad_b = _unbroadcast(-grad_out * a / (b**2), b.shape)
    else:
        # b is a scalar constant
        grad_b = None

    return (grad_a, grad_b)


def _neg_grad(grad_out, *_args, **_kwargs) -> tuple[np.ndarray, ...]:
    return (-grad_out,)


def _log_grad(grad_out, a, **_kwargs) -> tuple[np.ndarray, ...]:
    return (grad_out / a,)


def _exp_grad(grad_out, a, **_kwargs) -> tuple[np.ndarray, ...]:
    return (grad_out * np.exp(a),)


def _sin_grad(grad_out, a, **_kwargs) -> tuple[np.ndarray, ...]:
    return (grad_out * np.cos(a),)


def _cos_grad(grad_out, a, **_kwargs) -> tuple[np.ndarray, ...]:
    return (grad_out * -np.sin(a),)


def _sqrt_grad(grad_out, a, **_kwargs) -> tuple[np.ndarray, ...]:
    return (grad_out * 0.5 / np.sqrt(a),)


def _power_grad(grad_out, a, b, **_kwargs) -> tuple[Union[np.ndarray, None], ...]:
    # d/da (a^b) = b * a^(b-1)
    grad_a = grad_out * b * np.power(a, b - 1)

    # Only compute grad_b if b is an array
    if isinstance(b, np.ndarray):
        # d/db (a^b) = a^b * log(a)
        grad_b = _unbroadcast(grad_out * np.power(a, b) * np.log(a), b.shape)
    else:
        # b is a scalar constant, no gradient needed
        grad_b = None

    return (_unbroadcast(grad_a, a.shape), grad_b)


def _max_grad(grad_out, a, b, **_kwargs) -> tuple[Union[np.ndarray, None], ...]:
    # Following common standard where if a == b, gradient for both is equal
    a_mask = (a > b) + 0.5 * (a == b)
    grad_a = _unbroadcast(grad_out * a_mask, a.shape)

    if isinstance(b, np.ndarray):
        b_mask = (b > a) + 0.5 * (b == a)
        grad_b = _unbroadcast(grad_out * b_mask, b.shape)
    else:
        # b is a scalar constant
        grad_b = None

    return (grad_a, grad_b)


def _min_grad(grad_out, a, b, **_kwargs) -> tuple[Union[np.ndarray, None], ...]:
    # Following common standard where if a == b, gradient for both is equal
    a_mask = (a < b) + 0.5 * (a == b)
    grad_a = _unbroadcast(grad_out * a_mask, a.shape)

    if isinstance(b, np.ndarray):
        b_mask = (b < a) + 0.5 * (b == a)
        grad_b = _unbroadcast(grad_out * b_mask, b.shape)
    else:
        # b is a scalar constant
        grad_b = None

    return (grad_a, grad_b)


def _reshape_grad(
    grad_out, a, newshape, **_kwargs
) -> tuple[Union[np.ndarray, None], ...]:
    return (grad_out.reshape(a.shape), None)


def _transpose_grad(grad_out, a, *_args, **kwargs) -> tuple[np.ndarray, ...]:
    new_axes = kwargs.get("axes", None)
    if new_axes is None:
        return (grad_out.T,)
    inv_transpose = np.argsort(new_axes)
    return (np.transpose(grad_out, axes=inv_transpose),)


def _sum_grad(grad_out, a, **kwargs) -> tuple[np.ndarray, ...]:
    axis = kwargs.get("axis", None)
    if axis is None:
        # Sum over all elements
        return (np.full_like(a, grad_out),)
    # Sum along specific axis
    grad_expanded = np.expand_dims(grad_out, axis=axis)
    return (np.broadcast_to(grad_expanded, a.shape),)


def _mean_grad(grad_out, a, **kwargs) -> tuple[np.ndarray, ...]:
    axis = kwargs.get("axis", None)
    if axis is None:
        # Mean over all elements
        count = a.size
        return (np.full_like(a, grad_out / count),)
    # Mean along specific axis
    count = (
        a.shape[axis]
        if isinstance(axis, int)
        else np.prod([a.shape[ax] for ax in axis])
    )
    grad_expanded = np.expand_dims(grad_out, axis=axis)
    return (np.broadcast_to(grad_expanded, a.shape) / count,)


def _dot_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    return (np.dot(grad_out, b.T), np.dot(a.T, grad_out))


def _matmul_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    if b.ndim == 1:
        # Matrix @ vector case: grad_a = grad_out[:, None] @ b[None, :]
        grad_a = np.outer(grad_out, b)
        grad_b = a.T @ grad_out
    else:
        # Matrix @ matrix case
        grad_a = grad_out @ b.T
        grad_b = a.T @ grad_out

    return (_unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape))


# Element-wise arithmetic operations
_register_grad(np.add, _add_grad)
_register_grad(np.subtract, _sub_grad)
_register_grad(np.multiply, _mul_grad)
_register_grad(np.true_divide, _div_grad)
_register_grad(np.negative, _neg_grad)

# Element-wise functions
_register_grad(np.log, _log_grad)
_register_grad(np.exp, _exp_grad)
_register_grad(np.sin, _sin_grad)
_register_grad(np.cos, _cos_grad)
_register_grad(np.sqrt, _sqrt_grad)
_register_grad(np.power, _power_grad)
_register_grad(np.maximum, _max_grad)
_register_grad(np.minimum, _min_grad)

# Reshaping and axis permutations
_register_grad(np.reshape, _reshape_grad)
_register_grad(np.transpose, _transpose_grad)

# Reductions
_register_grad(np.sum, _sum_grad)
_register_grad(np.mean, _mean_grad)

# Matrix operations
_register_grad(np.dot, _dot_grad)
_register_grad(np.matmul, _matmul_grad)
