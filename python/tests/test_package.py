from __future__ import annotations

import ddx


def test_all_covers_everything_the_extension_exports() -> None:
    """__all__ covers everything the extension exports.

    The re-export list is written by hand and the C++ tables are not, so this is
    what stops a new opcode from reaching _ddx and stopping there.
    """
    from ddx import _ddx

    exported = {name for name in dir(_ddx) if not name.startswith("_")}
    assert exported <= set(ddx.__all__)


def test_everything_named_in_all_is_importable() -> None:
    for name in ddx.__all__:
        assert hasattr(ddx, name), name


def test_the_package_is_typed() -> None:
    from importlib.resources import files

    assert (files("ddx") / "py.typed").is_file()


def test_has_jit_says_what_the_build_did() -> None:
    assert isinstance(ddx.has_jit, bool)
