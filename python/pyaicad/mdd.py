"""Compiling a model into a multi-valued decision diagram (MDD).

>>> import pyaicad as ac
>>> m = ac.Model()
>>> x = m.int_vars(3, domain=range(3))
>>> m.add(ac.all_different(x))
>>> mdd = ac.compile(m)
>>> mdd.count()
6
"""

from __future__ import annotations

import operator
from typing import Any, List, Optional, Sequence, Tuple, Union

from . import _core
from .errors import ModelError
from .model import Model, Var

__all__ = ["compile", "MDD"]

_USIZE_MAX = 2**64 - 1

_MERGE = {
    "less_relaxed": _core.PyMergeHeuristic.LessRelaxed,
    "most_likely": _core.PyMergeHeuristic.MostLikely,
    "state_similarity": _core.PyMergeHeuristic.StateSimilarity,
}


def _merge_heuristic(name: str) -> Any:
    try:
        return _MERGE[name]
    except KeyError:
        raise ValueError(f"merge must be one of {sorted(_MERGE)}, got {name!r}") from None


def _ordering(order: Union[str, Sequence[Union[Var, int]]], model: Model) -> Any:
    if isinstance(order, str):
        if order != "auto":
            raise ValueError(f"order must be 'auto' or a sequence of variables, got {order!r}")
        return _core.PyOrderingHeuristic.MinDomMaxLinked()
    indices = [v.index if isinstance(v, Var) else operator.index(v) for v in order]
    if sorted(indices) != sorted(set(indices)) or any(
        not 0 <= i < model.num_vars for i in indices
    ):
        raise ModelError("order must list distinct variables of the model")
    return _core.PyOrderingHeuristic.Custom(indices)


def compile(
    model: Model,
    *,
    width: Optional[int] = None,
    constraints: Optional[Sequence[int]] = None,
    order: Union[str, Sequence[Union[Var, int]]] = "auto",
    merge: str = "less_relaxed",
) -> "MDD":
    """Compiles ``model`` into an MDD.

    Without ``width`` the diagram is exact: its paths are exactly the solutions of the model.
    **An exact diagram can grow without bound and fill the memory**, so give a ``width`` for
    models whose size you do not know.

    With ``width``, the diagram is relaxed: no layer has more than ``width`` nodes, and every
    solution of the model is still a path, but some paths are not solutions.

    Args:
        model: The model to compile.
        width: The maximum number of nodes per layer. ``None`` compiles an exact diagram.
        constraints: Compile only these constraints, given as positions in ``model.constraints``.
            By default all of them. Only the variables in their scope get a layer.
        order: ``"auto"`` to let the compiler choose the order of the layers, or a sequence
            listing the variables from the first layer to the last.
        merge: How the nodes of a layer that is too wide are merged: ``"less_relaxed"``,
            ``"most_likely"`` or ``"state_similarity"``. Only used with ``width``.
    """
    if width is not None:
        width = operator.index(width)
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
    if constraints is not None:
        constraints = [operator.index(c) for c in constraints]
        if any(not 0 <= c < model.num_constraints for c in constraints):
            raise ModelError("constraints must be positions in model.constraints")
    native = _core.Compiler(model._native_problem()).compile(
        constraints=constraints,
        max_width=_USIZE_MAX if width is None else width,
        pyordering=_ordering(order, model),
        pymerge=_merge_heuristic(merge),
    )
    return MDD(model, native, width)


class MDD:
    """A compiled diagram. Returned by :func:`compile`.

    Each layer branches on one variable; a path from the root to the sink assigns every variable
    in the diagram. In an exact diagram the paths are the solutions, in a relaxed one they are a
    superset of them: use :meth:`Model.check` to know whether a path is a solution.
    """

    def __init__(self, model: Model, native: Any, width: Optional[int]) -> None:
        self._model = model
        self._native = native
        self._width = width

    @property
    def model(self) -> Model:
        """The model this diagram was compiled from, as it was when compiled."""
        return self._model

    @property
    def width(self) -> Optional[int]:
        """The width bound it was compiled with, ``None`` for an exact diagram."""
        return self._width

    @property
    def is_unsat(self) -> bool:
        """Whether the model is proved to have no solution. A relaxed diagram can prove it too."""
        return bool(self._native.is_unsat())

    def count(self) -> int:
        """The number of paths. For an exact diagram, the number of solutions of the model.

        For a relaxed one it is an upper bound on that number. 0 when the model is unsatisfiable.
        """
        return int(self._native.count_solutions())

    def solution(self) -> Optional[List[int]]:
        """A path of the diagram as an assignment, or ``None`` if there is none.

        It is a solution if the diagram is exact. If it is relaxed, check it with
        :meth:`Model.check`.
        """
        if self.is_unsat:
            return None
        sol = self._native.get_solution(False)
        return None if sol is None else list(sol)

    def sample(self, n: int = 1) -> List[List[int]]:
        """``n`` paths drawn at random, as assignments.

        Call :func:`pyaicad.set_seed` first to make them reproducible. Empty if the model is
        unsatisfiable.
        """
        if self.is_unsat:
            return []
        return [list(self._native.get_solution(True)) for _ in range(operator.index(n))]

    @property
    def widths(self) -> List[int]:
        """The number of nodes of each layer, from the root to the sink (which is a layer)."""
        return [self._native.number_nodes_in_layer(i) for i in range(self._native.number_layers())]

    @property
    def num_layers(self) -> int:
        """The number of layers, the sink's included."""
        return int(self._native.number_layers())

    def edges(self) -> List[Tuple[int, int, int, int]]:
        """The edges as ``(source, target, variable, value)``.

        Nodes are numbered layer by layer, from the root, and ``variable`` is the index of a
        variable of the model. Every edge goes from a layer to the next.
        """
        return [tuple(e) for e in self._native.topological_order()]  # type: ignore[misc]

    def to_dot(self) -> str:
        """The diagram in Graphviz's DOT format."""
        return str(self._native.as_graphviz())

    def __repr__(self) -> str:
        kind = "exact" if self._width is None else f"relaxed, width {self._width}"
        return f"MDD({kind}, {self.num_layers} layers)"
