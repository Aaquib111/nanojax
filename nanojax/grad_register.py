from typing import Callable
import numpy as np


_PRIMITIVE_GRAD: dict[Callable, Callable] = {}


def get_grad_func(func: Callable) -> Callable:
    return _PRIMITIVE_GRAD[func]


def _register_grad(func: Callable, grad_func: Callable):
    _PRIMITIVE_GRAD[func] = grad_func


def _add_grad(grad_out, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    return (grad_out, grad_out)


_register_grad(np.add, _add_grad)
