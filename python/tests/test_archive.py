"""Saving and loading a built equation.

The file carries the arena and the sweeps, so a loaded equation must answer
exactly what the model that built it answers -- bit for bit, since it is the
same graph and not merely an equivalent one.  ``tests/rt/tests_rt_archive.cpp``
asserts the same round trip on the C++ side, over the same serialiser.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import ddx


def _coupled() -> ddx.Equation:
    """Return a model whose Hessian couples, so the colouring is real work."""

    @ddx.equation
    def eq() -> ddx.Expression:
        """exp(x y) + log(y z) sin(x + z)."""
        x = ddx.var("x")
        y = ddx.var("y")
        z = ddx.var("z")
        return ddx.exp(x * y) + ddx.log(y * z) * ddx.sin(x + z)

    return eq


POINT = {"x": 1.3, "y": 0.7, "z": 2.1}


def test_round_trips_every_derivative(tmp_path: Path) -> None:
    """A loaded equation answers what the built one answers, exactly."""
    built = _coupled()
    path = tmp_path / "coupled.ddx"
    built.save(path)

    loaded = ddx.load(path)
    assert loaded.loaded
    assert not built.loaded
    assert loaded.symbols == built.symbols
    assert loaded.arity == built.arity
    assert loaded.outputs == built.outputs

    np.testing.assert_array_equal(loaded(POINT), built(POINT))
    for got, want in zip(loaded.jacobian(POINT), built.jacobian(POINT), strict=True):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(loaded.hessian(POINT), built.hessian(POINT), strict=True):
        np.testing.assert_array_equal(got, want)


def test_round_trips_a_system(tmp_path: Path) -> None:
    """A system keeps its output count and its row-major Jacobian."""

    @ddx.equation
    def built() -> tuple[ddx.Expression, ddx.Expression]:
        """A circle and a hyperbola."""  # noqa: D401
        x = ddx.var("x")
        y = ddx.var("y")
        return x * x + y * y - 4.0, x * y - 1.0

    path = tmp_path / "system.ddx"
    built.save(path)

    loaded = ddx.load(path)
    assert loaded.outputs == 2
    at = {"x": 1.3, "y": 0.7}
    for got, want in zip(loaded.jacobian(at), built.jacobian(at), strict=True):
        np.testing.assert_array_equal(got, want)


def test_load_accepts_a_string_path(tmp_path: Path) -> None:
    """Paths arrive as ``str`` as readily as ``Path``."""
    path = tmp_path / "str.ddx"
    _coupled().save(str(path))
    assert ddx.load(str(path)).loaded


def test_verify_answers_whether_it_is_this_equation(tmp_path: Path) -> None:
    """``verify`` passes for the model that wrote the file and not another."""
    path = tmp_path / "verify.ddx"
    built = _coupled()
    built.save(path)
    built.verify(path)  # this equation: no refusal

    @ddx.equation
    def other() -> ddx.Expression:
        """Return the same symbols over a different model."""
        return ddx.var("x") + ddx.var("y") + ddx.var("z")

    with pytest.raises(ddx.Error) as refused:
        other.verify(path)
    assert refused.value.code == ddx.errc.archive_mismatch


def test_missing_file_is_not_corruption(tmp_path: Path) -> None:
    """An absent file is the first run of a cache, not a damaged one."""
    with pytest.raises(ddx.Error) as refused:
        ddx.load(tmp_path / "nothing-here.ddx")
    assert refused.value.code == ddx.errc.archive_io


def test_refuses_a_damaged_file(tmp_path: Path) -> None:
    """A flipped byte is caught by the checksum, and nothing aborts."""
    path = tmp_path / "damaged.ddx"
    _coupled().save(path)

    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    path.write_bytes(bytes(raw))

    with pytest.raises(ddx.Error) as refused:
        ddx.load(path)
    assert refused.value.code == ddx.errc.archive_corrupt


def test_refuses_a_foreign_file(tmp_path: Path) -> None:
    """Something that is not a ddx graph at all."""
    path = tmp_path / "foreign.ddx"
    path.write_bytes(b"not a ddx graph, just some bytes" * 8)
    with pytest.raises(ddx.Error) as refused:
        ddx.load(path)
    assert refused.value.code == ddx.errc.bad_archive


# --- the cache ---------------------------------------------------------------


def _cached(path: Path) -> ddx.Equation:
    """Build the same model twice over, through the cache."""

    def model() -> ddx.Expression:
        x = ddx.var("x")
        y = ddx.var("y")
        return ddx.exp(x * y) + ddx.sin(x) * ddx.log(y)

    return ddx.equation(model, cache=path)


def test_cache_builds_then_loads(tmp_path: Path) -> None:
    """The first run writes the file and the second reads it."""
    path = tmp_path / "cache.ddx"

    first = _cached(path)
    assert not first.loaded
    assert path.exists()

    second = _cached(path)
    assert second.loaded
    at = {"x": 1.3, "y": 0.7}
    np.testing.assert_array_equal(second(at), first(at))
    for got, want in zip(second.hessian(at), first.hessian(at), strict=True):
        np.testing.assert_array_equal(got, want)


def test_cache_rebuilds_when_the_model_changes(tmp_path: Path) -> None:
    """An edited model is not a cache hit."""
    path = tmp_path / "stale.ddx"

    def square() -> ddx.Expression:
        return ddx.var("x") * ddx.var("x")

    def cube() -> ddx.Expression:
        return ddx.var("x") * ddx.var("x") * ddx.var("x")

    assert not ddx.equation(square, cache=path).loaded
    assert not ddx.equation(cube, cache=path).loaded, "the model moved"

    again = ddx.equation(cube, cache=path)
    assert again.loaded, "the rebuild overwrote the stale file"
    assert again({"x": 2.0}) == pytest.approx(8.0)


def test_cache_survives_a_corrupt_file(tmp_path: Path) -> None:
    """A damaged cache rebuilds; it never raises and never aborts."""
    path = tmp_path / "corrupt-cache.ddx"
    assert not _cached(path).loaded

    raw = bytearray(path.read_bytes())
    raw[-3] ^= 0xFF
    path.write_bytes(bytes(raw))

    rebuilt = _cached(path)
    assert not rebuilt.loaded
    assert rebuilt({"x": 1.3, "y": 0.7}) == pytest.approx(
        math.exp(1.3 * 0.7) + math.sin(1.3) * math.log(0.7)
    )
    assert _cached(path).loaded, "and it left a good file behind"


# --- across builds -----------------------------------------------------------


def test_a_saved_graph_loads_without_the_jit(tmp_path: Path) -> None:
    """The file is a graph, not machine code, so it does not need a backend.

    Branching on ``has_jit`` rather than skipping: the no-JIT preset is the
    case worth proving, not the one worth excusing.
    """
    path = tmp_path / "portable.ddx"
    built = _coupled()
    built.save(path)

    loaded = ddx.load(path)
    np.testing.assert_array_equal(loaded(POINT), built(POINT))

    if ddx.has_jit:
        # And it still compiles from there, having never seen the model.
        assert loaded.compile().uses_kernel
        np.testing.assert_allclose(loaded(POINT), built(POINT), rtol=0, atol=0)
    else:
        assert not loaded.uses_kernel


def test_the_kernel_travels_with_the_graph(tmp_path: Path) -> None:
    """A saved equation carries its machine code, and loads answering through it.

    The graph half of a file saves milliseconds; this half saves the compile,
    which is the expensive thing an equation ever does.
    """
    if not ddx.has_jit:
        pytest.skip("no JIT in this build")

    path = tmp_path / "kernel.ddx"
    built = _coupled()
    # Off by default: a kernel does not hold a megabyte of machine code alive
    # on the chance that someone saves it.
    built.compile(retain_object=True)
    assert built.uses_kernel
    built.save(path)

    loaded = ddx.load(path)
    # No compile() call: the kernel was linked from the file, so it is already
    # answering.  That is the whole claim.
    assert loaded.uses_kernel
    np.testing.assert_array_equal(loaded(POINT), built(POINT))
    for got, want in zip(loaded.jacobian(POINT), built.jacobian(POINT), strict=True):
        np.testing.assert_array_equal(got, want)


def test_a_graph_saved_without_a_kernel_still_loads(tmp_path: Path) -> None:
    """No machine code is a shortfall in what a file is worth, not an error."""
    path = tmp_path / "graph-only.ddx"
    _coupled().save(path)
    loaded = ddx.load(path)
    assert not loaded.uses_kernel
    np.testing.assert_array_equal(loaded(POINT), _coupled()(POINT))
