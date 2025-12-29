"""Tape-based reverse-mode auto-differentiation in Python"""

from __future__ import annotations
from contextvars import ContextVar
from collections.abc import Callable, Sequence
from collections import namedtuple
from . import grad_register

import numpy as np

TraceItem = namedtuple(
    "TraceItem", ["func", "args", "kwargs", "backward_func", "output"]
)

_TRACE_STACK_VAR: ContextVar[tuple[list, ...]] = ContextVar("trace_stack", default=())


class TraceTape:
    def __enter__(self):
        """Append new trace stack to current context."""
        self.trace = []
        current_stack = _TRACE_STACK_VAR.get()
        _TRACE_STACK_VAR.set(current_stack + (self.trace,))
        return self.trace

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove trace stack from current context"""
        current_stack = _TRACE_STACK_VAR.get()
        _TRACE_STACK_VAR.set(current_stack[:-1])


class FuncTracer:
    """Traces and records operations of a function."""

    def __init__(self, array: np.ndarray):
        self.array = array

    def __add__(self, other: FuncTracer | np.ndarray) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.add, self, other)

    def __sub__(self, other: FuncTracer | np.ndarray) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.subtract, self, other)

    def __mul__(self, other: FuncTracer | np.ndarray) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.multiply, self, other)

    def __truediv__(self, other: FuncTracer | np.ndarray) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.true_divide, self, other)

    def __neg__(self) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.negative, self)

    def __radd__(self, other: FuncTracer | np.ndarray) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.add, other, self)

    def __rsub__(self, other: FuncTracer | np.ndarray) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.subtract, other, self)

    def __rmul__(self, other: FuncTracer | np.ndarray) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.multiply, other, self)

    def __rtruediv__(self, other: FuncTracer | np.ndarray) -> FuncTracer | np.ndarray:
        return _run_with_trace(np.true_divide, other, self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Intercept numpy ufunc calls to trace operations."""
        if method == "__call__":
            return _run_with_trace(ufunc, *inputs, **kwargs)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        """Intercept numpy array function calls to trace operations."""
        return _run_with_trace(func, *args, **kwargs)


def _unwrap_if_tracer(
    tracer_or_array: FuncTracer | np.ndarray,
) -> np.ndarray:
    """Unwraps FuncTracer and returns underlying array.

    Returns the input if already an array."""
    return (
        tracer_or_array.array
        if isinstance(tracer_or_array, FuncTracer)
        else tracer_or_array
    )


def _get_current_trace():
    stack = _TRACE_STACK_VAR.get()
    if stack:
        return stack[-1]
    return None


def _run_with_trace(
    func: Callable, *func_args, **func_kwargs
) -> FuncTracer | np.ndarray:
    """Captures function run in the current trace stack."""
    trace_stack = _get_current_trace()
    arg_arrays = [_unwrap_if_tracer(arg) for arg in func_args]
    any_args_tracers = any(isinstance(arg, FuncTracer) for arg in func_args)

    func_result = func(*arg_arrays, **func_kwargs)
    if trace_stack is None or not any_args_tracers:
        # Don't bother tracing if all args are just constant arrays
        return func_result

    traced_result = FuncTracer(array=func_result)
    backward_func = grad_register.get_grad_func(func)

    result_trace_item = TraceItem(
        func=func,
        args=func_args,
        kwargs=func_kwargs,
        backward_func=backward_func,
        output=traced_result,
    )
    trace_stack.append(result_trace_item)
    return traced_result


def grad(
    func: Callable, argnums: Sequence[int], grad_direction: np.ndarray | None
) -> Callable:
    """Returns a gradient function.

    Returns a gradient function that returns the derivative of the function with
    respect to each input argument.

    Args:
        func: The function to differentiate.
        argnums: The indices of arguments to return gradients for.
        grad_direction: The gradient direction. Defaults to 1 if None.

    Returns:
        Function computing gradient w.r.t each positional input argument.
    """

    def wrapper(*func_args, **func_kwargs):
        """Returns gradient w.r.t each positional argument."""
        wrapped_func_args = [
            arg if isinstance(arg, FuncTracer) else FuncTracer(array=arg)
            for arg in func_args
        ]
        with TraceTape() as trace:
            func_output: FuncTracer = func(*wrapped_func_args, **func_kwargs)
            if not isinstance(func_output, FuncTracer):
                # Output is constant w.r.t. inputs, so grad is zero.
                return tuple(np.zeros_like(wrapped_func_args[i].array) for i in argnums)
            if func_output.array.size > 1 and grad_direction is None:
                raise ValueError(
                    f"Got vector output of shape {func_output.array.shape}"
                    " but grad_direction is None."
                )
            # Set initial gradient
            grad_out = np.array(1.0) if grad_direction is None else grad_direction
            gradient_by_arg = {func_output: grad_out}

            # Traverse tape backwards, updating gradients as we go
            for (
                trace_func,
                trace_args,
                trace_kwargs,
                trace_backward_func,
                trace_output,
            ) in trace[::-1]:
                trace_output_grad = gradient_by_arg[trace_output]
                unwrapped_args = [_unwrap_if_tracer(arg) for arg in trace_args]
                gradient_wrt_args = trace_backward_func(
                    trace_output_grad, *unwrapped_args, **trace_kwargs
                )
                for i, arg in enumerate(trace_args):
                    if isinstance(arg, FuncTracer):
                        if arg in gradient_by_arg:
                            gradient_by_arg[arg] += gradient_wrt_args[i]
                        else:
                            gradient_by_arg[arg] = gradient_wrt_args[i]
        return tuple(gradient_by_arg[wrapped_func_args[i]] for i in argnums)

    return wrapper
