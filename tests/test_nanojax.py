import math

import pytest

import nanojax as nj


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, 0.0),
        (1.0, 2.0),
        (-3.0, -6.0),
    ],
)
def test_grad_of_square_returns_linear_function(x, expected):
    f = lambda t: nj.pow(t, 2)
    df_dt = nj.grad(f)

    assert df_dt(x) == pytest.approx(expected)


@pytest.mark.parametrize(
    "x",
    [0.0, 0.5, 1.75],
)
def test_grad_matches_chain_rule_for_composed_functions(x):
    f = lambda t: nj.sin(nj.pow(t, 2))
    df_dt = nj.grad(f)

    expected = 2 * x * math.cos(x ** 2)
    assert df_dt(x) == pytest.approx(expected)


@pytest.mark.parametrize(
    "x",
    [-2.0, -0.5, 3.25],
)
def test_second_derivative_uses_grad_of_grad(x):
    f = lambda t: nj.pow(t, 3) + 2 * t
    d2f_dt2 = nj.grad(nj.grad(f))

    # d/dx (3x^2 + 2) = 6x
    expected = 6 * x
    assert d2f_dt2(x) == pytest.approx(expected)


def test_constants_produce_zero_gradient():
    f = lambda _t: 5.0
    df_dt = nj.grad(f)

    assert df_dt(1.23) == 0.0


def test_grad_handles_mixed_operations():
    def f(x, y):
        return nj.sin(x) * nj.exp(y) + nj.log(x + y)

    grad_f = nj.grad(f)

    x, y = 1.2, -0.3
    expected_x = math.cos(x) * math.exp(y) + 1 / (x + y)
    expected_y = math.sin(x) * math.exp(y) + 1 / (x + y)

    dx, dy = grad_f(x, y)

    assert dx == pytest.approx(expected_x)
    assert dy == pytest.approx(expected_y)
