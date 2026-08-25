"""JIT configuration, with the ranges the C++ struct cannot check for a caller."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from . import _ddx
from ._ddx import Backend, VecLib

# What the build settled on: DDX_JIT_DEFAULT_OPT follows CMAKE_BUILD_TYPE and
# DDX_JIT_DEFAULT_CONTRACT follows DDX_FP_FLAGS, so the defaults are read off the
# library rather than restated here, where they would drift.
_DEFAULTS: _ddx._Options = _ddx._Options()


class Options(BaseModel):
    """How an equation compiles, if it compiles at all.

    Frozen because ``Equation.options`` is the one call that mutates an
    equation: reconfiguring is assigning a new value, not editing one under a
    call already in flight.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: Backend = _DEFAULTS.backend
    """``COMPILE`` starts the compile there and then; calls before it lands are
    swept and switch over when it arrives."""

    points: int = Field(_DEFAULTS.points, ge=1)
    """The batch a caller intends to hand to one call. Decides the lane width
    and nothing else."""

    lanes: int = Field(_DEFAULTS.lanes, ge=0)
    """Points per loop iteration. 0 derives it from ``points``; 1 is scalar."""

    opt_level: int = Field(_DEFAULTS.opt_level, ge=0, le=3)
    """LLVM's optimisation level for the IR pipeline."""

    codegen_level: int = Field(_DEFAULTS.codegen_level, ge=0, le=3)
    """LLVM's codegen level, which is ~95% of a compile and so the knob that
    trades kernel speed for compile time."""

    slp: bool = _DEFAULTS.slp
    loop_vectorize: bool = _DEFAULTS.loop_vectorize
    veclib: VecLib = _DEFAULTS.veclib
    contract: bool = _DEFAULTS.contract
    """Whether a multiply feeding an add is one rounding. Read by the graph, so
    the kernel and the sweep agree whichever answers."""

    time_passes: bool = _DEFAULTS.time_passes

    def _native(self) -> _ddx._Options:
        native = _ddx._Options()
        for field in type(self).model_fields:
            setattr(native, field, getattr(self, field))
        return native

    @classmethod
    def _from_native(cls, native: _ddx._Options) -> Self:
        return cls(**{f: getattr(native, f) for f in cls.model_fields})
