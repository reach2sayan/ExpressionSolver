from __future__ import annotations

import pytest
from pydantic import ValidationError

import ddx


def test_defaults_come_from_the_library() -> None:
    """Defaults come from the library, not from Python.

    DDX_JIT_DEFAULT_OPT follows CMAKE_BUILD_TYPE, so restating it here would
    drift from the build.
    """
    from ddx import _ddx

    native = _ddx._Options()
    options = ddx.Options()
    for field in ddx.Options.model_fields:
        assert getattr(options, field) == getattr(native, field)


@pytest.mark.parametrize("field", ["opt_level", "codegen_level"])
def test_llvm_levels_are_zero_to_three(field: str) -> None:
    with pytest.raises(ValidationError):
        ddx.Options(**{field: 7})
    assert getattr(ddx.Options(**{field: 3}), field) == 3


def test_a_batch_has_at_least_one_point() -> None:
    with pytest.raises(ValidationError):
        ddx.Options(points=0)


def test_lanes_are_derived_or_at_least_one() -> None:
    assert ddx.Options().lanes is None
    with pytest.raises(ValidationError):
        ddx.Options(lanes=0)
    derived = ddx.Options(lanes=None)
    assert ddx.Options.model_validate_json(derived.model_dump_json()) == derived


def test_a_typo_is_an_error_and_not_a_dropped_keyword() -> None:
    with pytest.raises(ValidationError):
        ddx.Options(pionts=8)  # type: ignore[call-arg]  # deliberate typo


def test_options_are_frozen() -> None:
    options = ddx.Options()
    with pytest.raises(ValidationError):
        options.points = 8  # type: ignore[misc]  # deliberately frozen


def test_a_configuration_round_trips_through_json() -> None:
    options = ddx.Options(backend=ddx.Backend.COMPILE, points=64, lanes=4)
    assert ddx.Options.model_validate_json(options.model_dump_json()) == options


def test_assigning_options_reaches_the_equation(scalar: ddx.Equation) -> None:
    scalar.options = ddx.Options(points=32, opt_level=2)
    assert scalar.options.points == 32
    assert scalar.options.opt_level == 2
