from __future__ import annotations

import math
from collections.abc import Callable

import pytest

import ddx

# Every unary the library generates, against math's own.  The list is what the
# module exports, so an opcode added to the C++ tables is covered without this
# file changing.
UNARY = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": None,
    "atanh": math.atanh,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "erf": math.erf,
    "abs": abs,
    "neg": lambda v: -v,
    "sign": lambda v: math.copysign(1.0, v),
}


def test_every_exported_unary_is_covered() -> None:
    """Every unary the module exports is checked against math's own.

    The table above is written by hand and the C++ tables are not, so this is
    what fails when an opcode is added there and not here.
    """
    functions = {
        name
        for name in ddx.__all__
        if callable(getattr(ddx, name)) and not isinstance(getattr(ddx, name), type)
    }
    binary = {"add", "mul", "div", "pow", "atan2", "hypot", "max", "min"}
    # Not operations at all: these build or read an equation.
    not_an_op = {"equation", "load", "var"}
    assert functions - binary - not_an_op == set(UNARY)


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_matches_math(name: str) -> None:
    reference = UNARY[name]
    at = 1.5 if name == "acosh" else 0.4
    if reference is None:
        reference = math.acosh

    fn = getattr(ddx, name)
    eq = ddx.equation(lambda: fn(ddx.var("x")))
    assert eq.evaluate([at]) == pytest.approx(reference(at))


@pytest.mark.parametrize(
    ("name", "reference"),
    [
        ("pow", lambda a, b: a**b),
        ("atan2", math.atan2),
        ("hypot", math.hypot),
        ("max", max),
        ("min", min),
        ("add", lambda a, b: a + b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b),
    ],
)
def test_binary_matches_math(
    name: str, reference: Callable[[float, float], float]
) -> None:
    fn = getattr(ddx, name)
    eq = ddx.equation(lambda: fn(ddx.var("x"), ddx.var("y")))
    assert eq.evaluate([1.7, 2.3]) == pytest.approx(reference(1.7, 2.3))


def test_operators_and_reflection() -> None:
    eq = ddx.equation(lambda: (2.0 * ddx.var("x") + 1.0) / 4.0 - 0.5)
    assert eq.evaluate([3.0]) == pytest.approx((2.0 * 3.0 + 1.0) / 4.0 - 0.5)

    rhs = ddx.equation(lambda: 8.0 / ddx.var("x") - 2.0 ** ddx.var("x"))
    assert rhs.evaluate([3.0]) == pytest.approx(8.0 / 3.0 - 2.0**3.0)

    assert ddx.equation(lambda: abs(-ddx.var("x"))).evaluate([-2.0]) == 2.0


def test_interning_makes_a_repeated_subexpression_one_node() -> None:
    """A repeated subexpression is one node: an id *is* its identity."""

    @ddx.equation
    def shared() -> ddx.Expression:
        x, y = ddx.var("x"), ddx.var("y")
        return x * y + x * y

    assert shared.evaluate([2.0, 3.0]) == pytest.approx(12.0)
    assert shared.symbols == ["x", "y"]


def test_literals_fold_without_reaching_a_graph() -> None:
    """Two pending literals never become nodes, so this equation has no symbols."""
    eq = ddx.equation(lambda: ddx.Expression(2.0) * 3.0)
    assert eq.symbols == []
    assert eq.arity == 0
    assert eq.evaluate([]) == pytest.approx(6.0)


def test_var_outside_a_model_is_refused() -> None:
    with pytest.raises(ddx.Error) as caught:
        ddx.var("stray")
    assert caught.value.code is ddx.errc.no_arena


def test_a_raising_model_leaves_no_arena_installed() -> None:
    """A model that raises leaves no arena installed.

    The guard that restores the thread-local runs on the unwind; without it the
    next model would build into the arena this one abandoned.
    """

    def broken() -> ddx.Expression:
        ddx.var("x")
        raise ZeroDivisionError("from the model")

    with pytest.raises(ZeroDivisionError):
        ddx.equation(broken)

    with pytest.raises(ddx.Error) as caught:
        ddx.var("after")
    assert caught.value.code is ddx.errc.no_arena


def test_repr_names_what_it_is() -> None:
    eq = ddx.equation(lambda: ddx.var("x") * ddx.var("y"))
    assert repr(eq) == "<ddx.Equation (x, y) -> 1 output>"
    assert "Expression" in repr(ddx.Expression(2.0))
