"""pyaicad: MDD compilation and local search for constraint satisfaction problems.

The three things you do with it:

* **Model** a problem: :class:`Model`, with the constraints :func:`all_different`,
  :func:`not_equal`, :func:`equal`, :func:`among`, :func:`at_least`, :func:`sum_equals`,
  :func:`gcc`, :func:`regular` and :func:`table`.
* **Compile** it into a decision diagram: a :class:`Compiler` made for the model and its
  heuristics hands out an :class:`MDD`, as exact or as relaxed as you ask, to count, sample and
  refine.
* **Search** for solutions with :func:`local_search`, repairing with a trained network
  (:class:`NeuralPriorSearch`, :class:`NeuralPriorBPSearch`) or with belief propagation alone
  (:class:`BeliefPropagationSearch`).

Networks for the neural searches are trained with :mod:`pyaicad.learning`.

>>> import pyaicad as ac
>>> m = ac.Model()
>>> x = m.int_vars(3, domain=range(3))
>>> m.add(ac.all_different(x))
>>> ac.Compiler(m).compile().count()
6
"""

from . import learning
from ._runtime import (
    HAS_LEARNING,
    BuildInfo,
    get_num_threads,
    set_log_level,
    set_num_threads,
    set_seed,
    version,
)
from .errors import AicadError, CheckpointError, FeatureUnavailableError, ModelError
from .mdd import MDD, Compiler
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
    table,
)
from .search import (
    BeliefPropagationSearch,
    Limits,
    NeuralPriorBPSearch,
    NeuralPriorSearch,
    Result,
    Status,
    local_search,
)

__version__ = version().version

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
    "table",
    # compiling
    "Compiler",
    "MDD",
    # searching
    "local_search",
    "NeuralPriorSearch",
    "NeuralPriorBPSearch",
    "BeliefPropagationSearch",
    "Limits",
    "Result",
    "Status",
    # learning (the rest is in pyaicad.learning)
    "learning",
    # settings
    "set_seed",
    "set_num_threads",
    "get_num_threads",
    "set_log_level",
    # build
    "version",
    "BuildInfo",
    # errors
    "AicadError",
    "ModelError",
    "CheckpointError",
    "FeatureUnavailableError",
    "HAS_LEARNING",
]
