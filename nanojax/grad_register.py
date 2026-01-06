from typing import Callable, Union

import numpy as np

_PRIMITIVE_GRAD: dict[Callable, Callable] = {}


def get_grad_func(func: Callable) -> Callable:
    return _PRIMITIVE_GRAD[func]


def _register_grad(func: Callable, grad_func: Callable) -> None:
    _PRIMITIVE_GRAD[func] = grad_func


def _get_shape(x) -> tuple:
    """Get shape of x, handling scalars and arrays."""
    if hasattr(x, "shape"):
        return x.shape
    return ()  # Scalars have empty shape


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
        grad = np.sum(grad, axis=0)

    # Sum along dimensions that were size 1 in original
    for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
        if orig_dim == 1 and grad_dim > 1:
            grad = np.sum(grad, axis=i, keepdims=True)

    return grad


def _add_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    grad_a = _unbroadcast(grad_out, _get_shape(a))
    grad_b = _unbroadcast(grad_out, _get_shape(b))
    return (grad_a, grad_b)


def _sub_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    grad_a = _unbroadcast(grad_out, _get_shape(a))
    grad_b = _unbroadcast(-grad_out, _get_shape(b))

    return (grad_a, grad_b)


def _mul_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    grad_a = _unbroadcast(grad_out * b, _get_shape(a))
    grad_b = _unbroadcast(grad_out * a, _get_shape(b))

    return (grad_a, grad_b)


def _div_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    grad_a = _unbroadcast(grad_out / b, _get_shape(a))
    grad_b = _unbroadcast(-grad_out * a / (b**2), _get_shape(b))
    return (grad_a, grad_b)


def _pow_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    # d/da (a^b) = b * a^(b-1)
    # d/db (a^b) = a^b * ln(a)
    grad_a = _unbroadcast(grad_out * b * np.power(a, b - 1), _get_shape(a))
    # Only compute grad_b if b is not a scalar constant (avoids log(negative) warnings)
    b_shape = _get_shape(b)
    if b_shape == ():
        grad_b = np.array(0.0)
    else:
        grad_b = _unbroadcast(grad_out * np.power(a, b) * np.log(a), b_shape)
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


def _max_grad(grad_out, a, b, **_kwargs) -> tuple[Union[np.ndarray, None], ...]:
    # Following common standard where if a == b, gradient for both is equal
    a_mask = (a > b) + 0.5 * (a == b)
    b_mask = (b > a) + 0.5 * (b == a)
    grad_a = _unbroadcast(grad_out * a_mask, _get_shape(a))
    grad_b = _unbroadcast(grad_out * b_mask, _get_shape(b))
    return (grad_a, grad_b)


def _min_grad(grad_out, a, b, **_kwargs) -> tuple[Union[np.ndarray, None], ...]:
    # Following common standard where if a == b, gradient for both is equal
    a_mask = (a < b) + 0.5 * (a == b)
    b_mask = (b < a) + 0.5 * (b == a)
    grad_a = _unbroadcast(grad_out * a_mask, _get_shape(a))
    grad_b = _unbroadcast(grad_out * b_mask, _get_shape(b))
    return (grad_a, grad_b)


def _reshape_grad(
    grad_out, a, newshape, **_kwargs
) -> tuple[Union[np.ndarray, None], ...]:
    return (np.reshape(grad_out, _get_shape(a)), None)


def _transpose_grad(grad_out, a, *_args, **kwargs) -> tuple[np.ndarray, ...]:
    new_axes = kwargs.get("axes", None)
    if new_axes is None:
        return (np.transpose(grad_out),)
    inv_transpose = np.argsort(new_axes)
    return (np.transpose(grad_out, axes=inv_transpose),)


def _sum_grad(grad_out, a, **kwargs) -> tuple[np.ndarray, ...]:
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if axis is None:
        # Sum over all elements
        return (np.full_like(a, grad_out),)
    # Sum along specific axis
    if keepdims:
        grad_expanded = grad_out
    else:
        grad_expanded = np.expand_dims(grad_out, axis=axis)
    return (np.broadcast_to(grad_expanded, a.shape),)


def _mean_grad(grad_out, a, **kwargs) -> tuple[np.ndarray, ...]:
    axis = kwargs.get("axis", None)
    keepdims = kwargs.get("keepdims", False)
    if axis is None:
        # Mean over all elements
        count = np.prod(_get_shape(a))
        return (np.full_like(a, grad_out / count),)
    # Mean along specific axis
    count = (
        a.shape[axis]
        if isinstance(axis, int)
        else np.prod([a.shape[ax] for ax in axis])
    )
    if keepdims:
        grad_expanded = grad_out
    else:
        grad_expanded = np.expand_dims(grad_out, axis=axis)
    return (np.broadcast_to(grad_expanded, a.shape) / count,)


def _dot_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    if a.ndim == 1 and b.ndim == 1:
        # Vector @ vector case: result is scalar, grads are vectors
        grad_a = grad_out * b
        grad_b = grad_out * a
    elif a.ndim == 2 and b.ndim == 1:
        # Matrix @ vector case: grad_a is outer product, grad_b is matrix-vector product
        grad_a = np.outer(grad_out, b)
        grad_b = np.transpose(a) @ grad_out
    elif a.ndim == 1 and b.ndim == 2:
        # Vector @ matrix case: grad_a is matrix-vector product, grad_b is outer product
        grad_a = b @ grad_out
        grad_b = np.outer(a, grad_out)
    else:
        # Matrix @ matrix case
        grad_a = grad_out @ np.transpose(b)
        grad_b = np.transpose(a) @ grad_out

    return (_unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape))


def _matmul_grad(grad_out, a, b, **_kwargs) -> tuple[np.ndarray, ...]:
    if a.ndim == 1 and b.ndim == 1:
        # Vector @ vector case: result is scalar, grads are vectors
        grad_a = grad_out * b
        grad_b = grad_out * a
    elif a.ndim >= 1 and b.ndim == 1:
        # Matrix @ vector case: grad_a is outer product, grad_b is matrix-vector product
        grad_a = np.outer(grad_out, b)
        grad_b = np.transpose(a) @ grad_out
    elif a.ndim == 1 and b.ndim >= 2:
        # Vector @ matrix case: grad_a is matrix-vector product, grad_b is outer product
        grad_a = b @ grad_out
        grad_b = np.outer(a, grad_out)
    else:
        # Matrix @ matrix case
        grad_a = grad_out @ np.transpose(b)
        grad_b = np.transpose(a) @ grad_out

    return (_unbroadcast(grad_a, a.shape), _unbroadcast(grad_b, b.shape))


# Element-wise arithmetic operations
_register_grad(np.add, _add_grad)
_register_grad(np.subtract, _sub_grad)
_register_grad(np.multiply, _mul_grad)
_register_grad(np.true_divide, _div_grad)
_register_grad(np.power, _pow_grad)
_register_grad(np.negative, _neg_grad)

# Element-wise functions
_register_grad(np.log, _log_grad)
_register_grad(np.exp, _exp_grad)
_register_grad(np.sin, _sin_grad)
_register_grad(np.cos, _cos_grad)
_register_grad(np.sqrt, _sqrt_grad)
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
