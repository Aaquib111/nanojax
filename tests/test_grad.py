import unittest

import numpy as np
from numpy.testing import assert_array_equal

from nanojax import grad


class TestGrad(unittest.TestCase):
    def test_grad_vector_output_raises_error(self):
        def f(x):
            return x + np.array([1.0, 2.0])

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([1.0])
        with self.assertRaises(ValueError):
            grad_f(x)

    def test_grad_addition_single_arg(self):
        def f(x):
            return x + x

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([1.0])
        dx = grad_f(x)
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

    def test_grad_with_broadcasting(self):
        def f(x, y):
            return x + y

        # Provide grad_direction for the (2, 2) output
        grad_direction = np.ones((2, 2))
        grad_f = grad(f, argnums=(0, 1), grad_direction=grad_direction)
        # x broadcasts along axis 0
        x = np.array([[1.0, 2.0]])  # shape (1, 2)
        y = np.array([[3.0], [4.0]])  # shape (2, 1)
        # Result has shape (2, 2), gradients should sum along broadcasted axes
        dx, dy = grad_f(x, y)

        # dx should sum along axis 0: shape (1, 2)
        assert_array_equal(dx, np.array([[2.0, 2.0]]))
        # dy should sum along axis 1: shape (2, 1)
        assert_array_equal(dy, np.array([[2.0], [2.0]]))


class TestArithmeticGrad(unittest.TestCase):
    def test_multiply_grad(self):
        def f(x, y):
            return x * y

        grad_f = grad(f, argnums=(0, 1), grad_direction=None)
        x = np.array([2.0])
        y = np.array([3.0])
        dx, dy = grad_f(x, y)
        # d/dx (x * y) = y, d/dy (x * y) = x
        assert_array_equal(dx, np.array([3.0]))
        assert_array_equal(dy, np.array([2.0]))

    def test_divide_grad(self):
        def f(x, y):
            return x / y

        grad_f = grad(f, argnums=(0, 1), grad_direction=None)
        x = np.array([6.0])
        y = np.array([2.0])
        dx, dy = grad_f(x, y)
        # d/dx (x / y) = 1/y, d/dy (x / y) = -x/y^2
        assert_array_equal(dx, np.array([0.5]))
        assert_array_equal(dy, np.array([-1.5]))

    def test_subtract_grad(self):
        def f(x, y):
            return x - y

        grad_f = grad(f, argnums=(0, 1), grad_direction=None)
        x = np.array([5.0])
        y = np.array([3.0])
        dx, dy = grad_f(x, y)
        # d/dx (x - y) = 1, d/dy (x - y) = -1
        assert_array_equal(dx, np.array([1.0]))
        assert_array_equal(dy, np.array([-1.0]))

    def test_negative_grad(self):
        def f(x):
            return -x

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([3.0])
        dx = grad_f(x)
        # d/dx (-x) = -1
        assert_array_equal(dx, np.array([-1.0]))


class TestElementwiseFunctionGrad(unittest.TestCase):
    def test_exp_grad(self):
        def f(x):
            return np.exp(x)

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([1.0])
        dx = grad_f(x)
        # d/dx exp(x) = exp(x)
        assert_array_equal(dx, np.exp(np.array([1.0])))

    def test_log_grad(self):
        def f(x):
            return np.log(x)

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([2.0])
        dx = grad_f(x)
        # d/dx log(x) = 1/x
        assert_array_equal(dx, np.array([0.5]))

    def test_sin_grad(self):
        def f(x):
            return np.sin(x)

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([0.0])
        dx = grad_f(x)
        # d/dx sin(x) = cos(x), cos(0) = 1
        assert_array_equal(dx, np.array([1.0]))

    def test_cos_grad(self):
        def f(x):
            return np.cos(x)

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([0.0])
        dx = grad_f(x)
        # d/dx cos(x) = -sin(x), -sin(0) = 0
        assert_array_equal(dx, np.array([0.0]))

    def test_sqrt_grad(self):
        def f(x):
            return np.sqrt(x)

        grad_f = grad(f, argnums=(0,), grad_direction=None)
        x = np.array([4.0])
        dx = grad_f(x)
        # d/dx sqrt(x) = 0.5/sqrt(x) = 0.5/2 = 0.25
        assert_array_equal(dx, np.array([0.25]))


class TestReshapeGrad(unittest.TestCase):
    def test_reshape_grad(self):
        def f(x):
            return np.reshape(x, (2, 2))

        grad_direction = np.ones((2, 2))
        grad_f = grad(f, argnums=(0,), grad_direction=grad_direction)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        dx = grad_f(x)
        # Gradient should flow back through reshape
        assert_array_equal(dx, np.array([1.0, 1.0, 1.0, 1.0]))
        assert_array_equal(dx.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
