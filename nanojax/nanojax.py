"""Tape-based reverse-mode auto-differentiation in Python"""

from contextvars import ContextVar
from collections import namedtuple
from . import grad_register

import numpy as np
from typing import Callable, Union

TraceItem = namedtuple(
    "TraceItem", ["func", "args", "kwargs", "backward_func", "output"]
)

_TRACE_STACK_VAR: ContextVar[tuple[list, ...]] = ContextVar("trace_stack", default=())


def get_current_trace():
    stack = _TRACE_STACK_VAR.get()
    if stack:
        return stack[-1]
    return None


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

    def __add__(
        self, other: Union["FuncTracer", np.ndarray]
    ) -> Union["FuncTracer", np.ndarray]:
        return _run_with_trace(np.add, self, other)


def _unwrap_if_tracer(
    tracer_or_array: Union[FuncTracer, np.ndarray],
) -> np.ndarray:
    """Unwraps FuncTracer and returns underlying array.

    Returns the input if already an array."""
    return (
        tracer_or_array.array
        if isinstance(tracer_or_array, FuncTracer)
        else tracer_or_array
    )


def _run_with_trace(
    func: Callable, *func_args, **func_kwargs
) -> Union[FuncTracer, np.ndarray]:
    """Captures function run in the current trace stack."""
    trace_stack = get_current_trace()
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
        args=arg_arrays,
        kwargs=func_kwargs,
        backward_func=backward_func,
        output=traced_result,
    )
    trace_stack.append(result_trace_item)
    return traced_result
