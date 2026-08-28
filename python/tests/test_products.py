"""The seeded products: ``J v``, ``w' J`` and ``H v``.

Each is checked against the dense matrix it is named after, contracted by hand
with the same direction -- which is the comparison that catches a product built
from the wrong columns.
"""

from __future__ import annotations

import numpy as np
import pytest

import ddx


def test_jvp_is_the_jacobian_times_the_direction(system: ddx.Equation) -> None:
    x = np.array([1.3, 0.7])
    v = np.array([2.0, -0.5])

    _, jac = system.jacobian(x)
    value, product = system.jvp(v, x)

    np.testing.assert_allclose(product, jac @ v, rtol=1e-10)
    np.testing.assert_allclose(value, system.evaluate(x), rtol=1e-12)


def test_vjp_is_the_covector_times_the_jacobian(system: ddx.Equation) -> None:
    x = np.array([1.3, 0.7])
    w = np.array([2.0, -1.0])

    _, jac = system.jacobian(x)
    _, product = system.vjp(w, x)

    np.testing.assert_allclose(product, w @ jac, rtol=1e-10)


def test_hvp_is_the_hessian_times_the_direction(scalar: ddx.Equation) -> None:
    x = np.array([0.3, 0.7])
    v = np.array([1.0, -2.0])

    _, _, hess = scalar.hessian(x)
    value, grad, product = scalar.hvp(v, x)

    np.testing.assert_allclose(product, hess @ v, rtol=1e-10)
    # The gradient rides along in the same sweep rather than being recomputed.
    _, jac = scalar.jacobian(x)
    np.testing.assert_allclose(grad, jac, rtol=1e-12)
    np.testing.assert_allclose(value, scalar.evaluate(x), rtol=1e-12)


def test_a_unit_direction_picks_one_column(scalar: ddx.Equation) -> None:
    x = np.array([0.3, 0.7])
    _, _, hess = scalar.hessian(x)

    for j in range(2):
        v = np.zeros(2)
        v[j] = 1.0
        _, _, column = scalar.hvp(v, x)
        np.testing.assert_allclose(column, hess[:, j], rtol=1e-10)


def test_a_direction_may_be_a_dict(scalar: ddx.Equation) -> None:
    """A direction over symbols takes the same spellings a point does."""
    x = {"x": 0.3, "y": 0.7}
    _, _, byarray = scalar.hvp(np.array([1.0, -2.0]), x)
    _, _, bydict = scalar.hvp({"x": 1.0, "y": -2.0}, x)
    np.testing.assert_array_equal(byarray, bydict)


def test_batches_agree_with_the_points_in_them(scalar: ddx.Equation) -> None:
    xs = np.array([[0.2, 0.3, 0.4, 0.5], [0.9, 0.8, 0.7, 0.6]])
    vs = np.array([[1.0, 0.5, -1.0, 2.0], [0.0, 1.5, 1.0, -0.5]])

    _, _, batched = scalar.hvp(vs, xs)
    assert batched.shape == (2, 4)

    for i in range(xs.shape[1]):
        _, _, one = scalar.hvp(vs[:, i], xs[:, i])
        np.testing.assert_allclose(batched[:, i], one, rtol=1e-12)


def test_shapes_drop_the_axes_that_are_not_there(
    scalar: ddx.Equation, system: ddx.Equation
) -> None:
    point = np.array([0.3, 0.7])
    direction = np.array([1.0, -2.0])

    _, _, hv = scalar.hvp(direction, point)
    assert hv.shape == (2,)  # one value per symbol

    _, jv = system.jvp(direction, point)
    assert jv.shape == (2,)  # one value per function

    _, wj = system.vjp(np.array([1.0, 1.0]), point)
    assert wj.shape == (2,)  # one value per symbol


def test_hvp_refuses_a_system(system: ddx.Equation) -> None:
    with pytest.raises(ddx.Error):
        system.hvp(np.array([1.0, 1.0]), np.array([1.3, 0.7]))


def test_a_direction_of_the_wrong_length_is_refused(scalar: ddx.Equation) -> None:
    with pytest.raises(ddx.Error):
        scalar.hvp(np.array([1.0]), np.array([0.3, 0.7]))
