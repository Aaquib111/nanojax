import unittest
import numpy as np
from numpy.testing import assert_array_equal  # type: ignore[attr-defined]

from nanojax import TraceTape, FuncTracer, grad


class TestFuncTracer(unittest.TestCase):
    def test_tracer_wraps_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        tracer = FuncTracer(arr)
        assert_array_equal(tracer.array, arr)

    def test_add_traces_operation(self):
        with TraceTape() as trace:
            a = FuncTracer(np.array([1.0, 2.0]))
            b = FuncTracer(np.array([3.0, 4.0]))
            c = a + b
            assert isinstance(c, FuncTracer)

            self.assertEqual(len(trace), 1)
            self.assertEqual(trace[0].func, np.add)
            self.assertIs(trace[0].output, c)
            assert_array_equal(c.array, np.array([4.0, 6.0]))

    def test_no_trace_without_tape(self):
        a = FuncTracer(np.array([1.0, 2.0]))
        b = FuncTracer(np.array([3.0, 4.0]))
        c = a + b
        assert_array_equal(c, np.array([4.0, 6.0]))


class TestGrad(unittest.TestCase):
    def test_grad_addition_single_arg(self):
        def f(x):
            return x + x

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([1.0])
        (dx,) = grad_f(x)
        # d/dx (x + x) = 2
        assert_array_equal(dx, np.array([2.0]))

    def test_grad_addition_two_args(self):
        def f(x, y):
            return x + y

        grad_f = grad(f, argnums=(0, 1), grad_direction=None)
        x = np.array([1.0])
        y = np.array([3.0])
        dx, dy = grad_f(x, y)
        # d/dx (x + y) = 1, d/dy (x + y) = 1
        assert_array_equal(dx, np.ones_like(x))
        assert_array_equal(dy, np.ones_like(y))


if __name__ == "__main__":
    unittest.main()
