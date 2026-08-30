"""The last call kept, and the answers unchanged by keeping it.

``remember=True`` is the Python spelling of the C++ ``LastValue`` cache: a point
already asked comes back off the last call, and a point one symbol away sweeps
only what that symbol reaches.  Every assertion here is exact -- an answer that
came back nearly right would be a bug this file could not see.
"""

from __future__ import annotations

import numpy as np
import pytest

import ddx


def _model() -> ddx.Expression:
    """exp(x) * sin(y) + z * z * z."""
    x = ddx.var("x")
    y = ddx.var("y")
    z = ddx.var("z")
    return ddx.exp(x) * ddx.sin(y) + z * z * z


@pytest.fixture
def pair() -> tuple[ddx.Equation, ddx.Equation]:
    """Return the same model with and without a cache."""
    return ddx.equation(_model), ddx.equation(_model, remember=True)


@pytest.mark.parametrize(
    "point",
    [
        [2.0, 3.0, 1.0],
        [2.0, 3.0, 1.0],  # the same point again
        [2.0, 3.0, 4.0],  # one symbol away
        [2.0, 5.0, 4.0],  # another
        [1.0, 5.0, 4.0],
        [2.0, 3.0, 1.0],
    ],
)
def test_a_remembered_call_answers_in_the_same_bits(
    pair: tuple[ddx.Equation, ddx.Equation], point: list[float]
) -> None:
    cold, warm = pair
    x = np.array(point)
    assert cold(x) == warm(x)
    for a, b in zip(cold.jacobian(x), warm.jacobian(x), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(cold.gradient(x), warm.gradient(x))
    for a, b in zip(cold.hessian(x), warm.hessian(x), strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_sequence_of_points_agrees_call_for_call() -> None:
    cold = ddx.equation(_model)
    warm = ddx.equation(_model, remember=True)
    rng = np.random.default_rng(20260830)
    x = np.array([2.0, 3.0, 1.0])
    for call in range(60):
        if call % 3:  # one coordinate moves, then none
            x[call % 3] = rng.uniform(-2.0, 2.0)
        assert cold(x) == warm(x)
        np.testing.assert_array_equal(cold.gradient(x), warm.gradient(x))


def test_zero_and_minus_zero_are_different_points() -> None:
    eq = ddx.equation("1.0 / x", remember=True)
    assert eq(np.array([0.0])) == np.inf
    assert eq(np.array([-0.0])) == -np.inf


def test_a_batch_agrees_too() -> None:
    cold = ddx.equation(_model)
    warm = ddx.equation(_model, remember=True)
    xs = np.array([[2.0, 3.0, 1.0], [0.5, 1.0, 2.0], [2.0, 3.0, 1.0]])
    np.testing.assert_array_equal(cold(xs), warm(xs))
    np.testing.assert_array_equal(cold.gradient(xs), warm.gradient(xs))


def test_the_bound_call_is_remembered_as_well() -> None:
    cold = ddx.equation(_model)
    warm = ddx.equation(_model, remember=True)
    call = warm.buffer(np.array([2.0, 3.0, 1.0]))
    for z in (1.0, 1.0, 4.0, 4.0, 1.0):
        call.x[2] = z
        call()
        want = cold.jacobian(call.x)
        assert call.value == want[0]
        np.testing.assert_array_equal(call.jacobian, want[1])


def test_a_text_equation_and_a_loaded_one_take_the_flag(tmp_path) -> None:  # noqa: ANN001
    text = ddx.equation("x * y + x", remember=True)
    x = np.array([2.0, 3.0])
    assert text(x) == text(x)

    path = tmp_path / "f.ddx"
    text.save(path)
    back = ddx.load(path, remember=True)
    assert back(x) == text(x)
