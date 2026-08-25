from __future__ import annotations

import numpy as np
import pytest
from conftest import Derivatives

import ddx


def test_values_gradient_and_hessian(
    scalar: ddx.Equation, expected: Derivatives
) -> None:
    assert scalar.evaluate([2.0, 3.0]) == pytest.approx(expected.f)

    value, gradient = scalar.jacobian([2.0, 3.0])
    assert value == pytest.approx(expected.f)
    assert gradient == pytest.approx([expected.dx, expected.dy])

    value, gradient, hessian = scalar.hessian([2.0, 3.0])
    assert hessian == pytest.approx(
        np.array([[expected.dxx, expected.dxy], [expected.dxy, expected.dyy]])
    )


def test_call_is_evaluate(scalar: ddx.Equation, expected: Derivatives) -> None:
    assert scalar([2.0, 3.0]) == pytest.approx(expected.f)


def test_the_three_point_spellings_agree(scalar: ddx.Equation) -> None:
    """Every spelling of a point names the same point.

    A dict is the keyword form C++ spells named<"x">(v), and a column array is
    the batch form at one point.
    """
    positional = scalar.jacobian([2.0, 3.0])
    by_name = scalar.jacobian({"x": 2.0, "y": 3.0})
    as_columns = scalar.jacobian(np.array([[2.0], [3.0]]))

    assert by_name[0] == pytest.approx(positional[0])
    assert by_name[1] == pytest.approx(positional[1])
    # The column spelling carries a points axis, so its answer keeps one.
    assert as_columns[0].shape == (1,)
    assert as_columns[1].shape == (2, 1)
    assert as_columns[1][:, 0] == pytest.approx(positional[1])


def test_a_system_has_a_runtime_output_count(system: ddx.Equation) -> None:
    assert system.outputs == 2
    assert system.arity == 2

    values, jacobian = system.jacobian([1.0, 2.0])
    # x^2 + y^2 - 4 and xy - 1 at (1, 2)
    assert values == pytest.approx([1.0, 1.0])
    # [[2x, 2y], [y, x]]
    assert jacobian == pytest.approx(np.array([[2.0, 4.0], [2.0, 1.0]]))


def test_a_system_still_has_a_hessian_at_a_point(system: ddx.Equation) -> None:
    _, _, hessian = system.hessian([1.0, 2.0])
    assert hessian.shape == (2, 2, 2)
    assert hessian[0] == pytest.approx(np.array([[2.0, 0.0], [0.0, 2.0]]))
    assert hessian[1] == pytest.approx(np.array([[0.0, 1.0], [1.0, 0.0]]))


def test_a_batched_hessian_needs_one_function(system: ddx.Equation) -> None:
    """A batched Hessian needs one function.

    A frozen graph carries one colouring, so there is no lane that could hold m
    of them.
    """
    batch = np.array([[1.0, 1.5], [2.0, 2.5]])
    with pytest.raises(ddx.Error) as caught:
        system.hessian(batch)
    assert caught.value.code is ddx.errc.wrong_column_count


def test_shapes_drop_the_axes_that_are_not_there(scalar: ddx.Equation) -> None:
    batch = np.array([[2.0, 2.5, 3.0], [3.0, 3.5, 4.0]])

    assert isinstance(scalar.evaluate([2.0, 3.0]), float)
    assert scalar.jacobian([2.0, 3.0])[1].shape == (2,)
    assert scalar.hessian([2.0, 3.0])[2].shape == (2, 2)

    assert scalar.evaluate(batch).shape == (3,)
    assert scalar.jacobian(batch)[1].shape == (2, 3)
    assert scalar.hessian(batch)[2].shape == (2, 2, 3)


def test_a_batch_agrees_with_the_points_in_it(scalar: ddx.Equation) -> None:
    batch = np.array([[2.0, 2.5, 3.0], [3.0, 3.5, 4.0]])
    _, columns = scalar.jacobian(batch)
    for i in range(batch.shape[1]):
        _, at_point = scalar.jacobian(batch[:, i])
        assert columns[:, i] == pytest.approx(at_point)


def test_a_dict_batch_needs_no_single_block(scalar: ddx.Equation) -> None:
    """Each value is its own column, so the arrays need not be one array."""
    xs = np.linspace(0.5, 2.0, 4)
    ys = np.linspace(0.1, 1.0, 4)
    loose = scalar.jacobian({"x": xs, "y": ys})
    packed = scalar.jacobian(np.stack([xs, ys]))
    assert loose[1] == pytest.approx(packed[1])


def test_a_scalar_in_a_dict_batch_is_held_at_every_point(scalar: ddx.Equation) -> None:
    xs = np.linspace(0.5, 2.0, 4)
    held = scalar.evaluate({"x": xs, "y": 0.5})
    spread = scalar.evaluate({"x": xs, "y": np.full(4, 0.5)})
    assert held == pytest.approx(spread)


def test_a_stray_name_is_not_a_missing_one(scalar: ddx.Equation) -> None:
    with pytest.raises(ddx.Error) as stray:
        scalar.evaluate({"x": 2.0, "z": 3.0})
    assert stray.value.code is ddx.errc.unknown_symbol

    with pytest.raises(ddx.Error) as missing:
        scalar.evaluate({"x": 2.0})
    assert missing.value.code is ddx.errc.wrong_arity


def test_a_point_of_the_wrong_length_is_refused(scalar: ddx.Equation) -> None:
    with pytest.raises(ddx.Error) as caught:
        scalar.evaluate([2.0])
    assert caught.value.code is ddx.errc.wrong_arity


def test_the_model_name_and_docstring_survive(scalar: ddx.Equation) -> None:
    assert scalar.__name__ == "eq"
    assert scalar.__doc__ is not None
    assert "exp(x)" in scalar.__doc__


def test_symbols_are_in_the_order_a_point_is_read_in(scalar: ddx.Equation) -> None:
    assert scalar.symbols == ["x", "y"]


def test_to_dot_renders_the_frozen_graph(scalar: ddx.Equation) -> None:
    live = scalar.to_dot()
    assert live.startswith("digraph")
    assert len(scalar.to_dot(all=True)) >= len(live)
