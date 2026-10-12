"""pyaicad: MDD compilation for constraint satisfaction and optimisation problems.

The three things you do with it:

* **Model** a problem: :class:`Model`, with the constraints :func:`all_different`,
  :func:`not_equal`, :func:`equal`, :func:`among`, :func:`at_least`, :func:`sum_equals`,
  :func:`gcc` and :func:`regular`.
* **Compile** it into a decision diagram, :func:`compile`, and query the :class:`MDD`.
* **Solve** it, :func:`solve`, by :class:`Compilation` or by neural local search, :class:`NLS`.

Networks for :class:`NLS` are trained with :mod:`pyaicad.learning`.

>>> import pyaicad as ac
>>> m = ac.Model()
>>> x = m.int_vars(3, domain=range(3))
>>> m.add(ac.all_different(x))
>>> ac.compile(m).count()
6
"""

from . import learning
from ._runtime import set_log_level, set_seed
from .errors import AicadError, CheckpointError, ModelError
from .mdd import MDD, compile
from .model import (
    Constraint,
    Model,
    Var,
    all_different,
    among,
    at_least,
    equal,
    gcc,
    not_equal,
    regular,
    sum_equals,
)
from .solve import NLS, Compilation, Limits, Result, Status, solve

__version__ = "0.0.1"

__all__ = [
    # modelling
    "Model",
    "Var",
    "Constraint",
    "all_different",
    "not_equal",
    "equal",
    "among",
    "at_least",
    "sum_equals",
    "gcc",
    "regular",
    # compiling
    "compile",
    "MDD",
    # solving
    "solve",
    "Compilation",
    "NLS",
    "Limits",
    "Result",
    "Status",
    # learning (the rest is in pyaicad.learning)
    "learning",
    # settings
    "set_seed",
    "set_log_level",
    # errors
    "AicadError",
    "ModelError",
    "CheckpointError",
]
