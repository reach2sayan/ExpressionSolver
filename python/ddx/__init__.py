"""Reverse-mode automatic differentiation over a graph built at run time.

    >>> import ddx
    >>> @ddx.equation
    ... def f():
    ...     x = ddx.var("x")
    ...     y = ddx.var("y")
    ...     return ddx.exp(x) * ddx.sin(y)
    >>> values, gradient = f.jacobian({"x": 2.0, "y": 3.0})

``equation`` is a decorator or a call, and takes whatever the model returns: one
expression, or a tuple of them for a system. A point is a sequence, a
``(symbols, points)`` array, or a dict keyed by symbol name.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from ._ddx import (Backend, Expression, VecLib, abs,  # noqa: A004  -- the opcode is spelled `abs`, as it is in C++
                   acos, acosh, add, asin, asinh, atan, atan2, atanh,
                   cbrt, cos, cosh, div, erf, errc, exp, has_jit, hypot,
                   log, log10, max,  # noqa: A004
                   min,  # noqa: A004
                   mul, neg,
                   pow,  # noqa: A004
                   sign, sin, sinh, sqrt, tan, tanh, var, )
from ._options import Options

if TYPE_CHECKING:
	from collections.abc import Callable
	import ddx._ddx as _extension
	
	# What the runtime adds and a generated stub cannot see: the code the
	# exception translator attaches, the pydantic-facing members installed at
	# the bottom of this file, and the shape of the decorator -- which the
	# bindings hand over as py::object and so describe only as Any.
	class Error(_extension.Error):
		"""A refusal from ddx, carrying the code it refused with."""
		code: errc
	
	class Equation(_extension.Equation):
		"""A frozen model, with the conveniences this module installs."""
		__name__: str
		options: Options
		
		def compile(self, **kwargs: Any) -> Equation:  # noqa: ANN401, D102
			pass
	
	Model = Callable[[], Expression | tuple[Expression, ...]]
	
	def equation(model: Model) -> Equation:
		"""Build an equation from a model, as a call or a decorator."""

else:
	from ._ddx import Equation, Error, equation

try: __version__ = version("ddx")
except PackageNotFoundError:  # an in-tree build that was never installed
	__version__ = "0+unknown"

# The two places pydantic sits between a caller and the struct.
def _get_options(self: Equation) -> Options:
	return Options._from_native(self._options)

def _set_options(self: Equation, options: Options) -> None:
	self._options = options._native()

def _compile(self: Equation, **kwargs: Any) -> Equation:  # noqa: ANN401
	"""Compile this equation's graph, and answer once the kernel has landed.
	Keywords are ``Options`` fields; ``backend`` is implied. Returns self, so
	``eq.compile(points=n).jacobian(xs)`` reads in one line.
	"""
	self.options = Options(backend=Backend.COMPILE, **kwargs)
	self.wait_for_kernel()
	return self

Equation.options = property(_get_options, _set_options)  # type: ignore[assignment]
Equation.compile = _compile  # type: ignore[method-assign]

__all__ = ["Backend", "Equation", "Error", "Expression", "Options",
           "VecLib", "__version__",	"abs", "acos", "acosh",	"add",
           "asin", "asinh",	"atan", "atan2", "atanh", "cbrt", "cos",
           "cosh", "div", "equation", "erf", "errc", "exp", "has_jit",
           "hypot", "log", "log10", "max", "min", "mul", "neg", "pow",
           "sign", "sin", "sinh", "sqrt", "tan", "tanh", "var",]
