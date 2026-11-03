"""Compiling a model into a multi-valued decision diagram (MDD).

>>> import pyaicad as ac
>>> m = ac.Model()
>>> x = m.int_vars(3, domain=range(3))
>>> m.add(ac.all_different(x))
>>> mdd = ac.Compiler(m).compile()
>>> mdd.count()
6
"""

from __future__ import annotations

import operator
from typing import Any, List, Optional, Sequence, Union

from . import _core
from .errors import ModelError
from .model import Model, Var

__all__ = ["Compiler", "MDD"]

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


class Compiler:
    """Compiles a model into decision diagrams.

    The compiler is made for a model and the heuristics that drive the compilation; it then hands
    out diagrams, as exact or as relaxed as asked: :meth:`compile`. It works on the model as it is
    when the compiler is created. Each diagram is independent of the others and can be refined
    further (:meth:`MDD.refine`) until it is frozen (:meth:`MDD.freeze`).

    >>> import pyaicad as ac
    >>> m = ac.Model()
    >>> x = m.int_vars(3, domain=range(3))
    >>> m.add(ac.all_different(x))
    >>> mdd = ac.Compiler(m).compile()
    >>> mdd.count()
    6

    Args:
        model: The model to compile.
        order: ``"auto"`` to let the compiler choose the order of the layers, or a sequence
            listing the variables from the first layer to the last.
        merge: How the nodes of a layer that is too wide are merged: ``"state_similarity"``
            (the default), ``"less_relaxed"`` or ``"most_likely"``. Only used by relaxed diagrams.
        constraints: Compile only these constraints, given as positions in ``model.constraints``.
            By default all of them. Only the variables in their scope get a layer.
    """

    def __init__(
        self,
        model: Model,
        *,
        order: Union[str, Sequence[Union[Var, int]]] = "auto",
        merge: str = "state_similarity",
        constraints: Optional[Sequence[int]] = None,
    ) -> None:
        if constraints is not None:
            constraints = [operator.index(c) for c in constraints]
            if any(not 0 <= c < model.num_constraints for c in constraints):
                raise ModelError("constraints must be positions in model.constraints")
        self._model = model.copy()
        self._ordering = _ordering(order, model)
        self._merge = _merge_heuristic(merge)
        self._constraints = constraints
        self._native = _core.Compiler(model._native_problem())

    @property
    def model(self) -> Model:
        """The model this compiler was made for, as it was then."""
        return self._model

    def compile(self, width: Optional[int] = None) -> "MDD":
        """A new diagram of the model.

        Without ``width`` the diagram is exact: its paths are exactly the solutions of the model.
        **An exact diagram can grow without bound and fill the memory**, so give a ``width`` for
        models whose size you do not know.

        With ``width``, the diagram is relaxed: no layer has more than ``width`` nodes, and every
        solution of the model is still a path, but some paths are not solutions.
        """
        if width is not None:
            width = operator.index(width)
            if width < 1:
                raise ValueError(f"width must be at least 1, got {width}")
        native = self._native.compile(
            constraints=self._constraints,
            max_width=_USIZE_MAX if width is None else width,
            pyordering=self._ordering,
            pymerge=self._merge,
        )
        return MDD(self._model, native, width)

    def __repr__(self) -> str:
        return f"Compiler({self._model!r})"


class MDD:
    """A compiled diagram. Returned by :meth:`Compiler.compile`.

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

    def refine(self, width: Optional[int] = None) -> None:
        """Tightens the diagram in place, up to ``width`` nodes per layer (``None``: exact).

        A diagram compiled with a small width can be refined step by step, each step keeping the
        solutions and removing paths that are not. Refining to a width below the current one
        merges nodes again, giving a looser diagram. Fails on a frozen diagram.
        """
        if self.is_frozen:
            raise RuntimeError("this diagram is frozen and cannot be refined")
        if width is not None:
            width = operator.index(width)
            if width < 1:
                raise ValueError(f"width must be at least 1, got {width}")
        self._native.refine(_USIZE_MAX if width is None else width)
        self._width = width

    def freeze(self) -> "MDD":
        """Drops what only refining needs and keeps the diagram. Returns the diagram itself.

        A diagram that can still be refined keeps its compilation alive: the constraints, their
        properties and the heuristics, which is often most of the memory. Freeze it once it is
        tight enough. Everything but :meth:`refine` keeps working, and freezing twice does nothing.
        """
        self._native.freeze()
        return self

    @property
    def is_frozen(self) -> bool:
        """Whether :meth:`freeze` was called: the diagram cannot be refined any more."""
        return bool(self._native.is_frozen())

    def count(self) -> int:
        """The number of paths. For an exact diagram, the number of solutions of the model.

        For a relaxed one it is an upper bound on that number. 0 when the model is unsatisfiable.
        """
        return int(self._native.count_solutions())

    @property
    def scope(self) -> List[Var]:
        """The variables the diagram branches on, in the order they were declared.

        All the variables of the model unless some are in no compiled constraint: they have no
        layer, and :meth:`solution` and :meth:`sample` give them the first value of their domain.
        """
        variables = self._model.variables
        return [variables[index] for index in self._native.scope()]

    def solution(self) -> Optional[List[int]]:
        """A solution read off the diagram, as one value per variable of the model, or ``None``.

        The solution is a path of exact nodes, so it satisfies the compiled constraints. An
        exact diagram has one whenever the model is satisfiable. A relaxed diagram may hold
        solutions that this does not find: ``None`` proves nothing unless :attr:`is_unsat`.
        """
        found = None if self.is_unsat else self._native.get_solution(False)
        return None if found is None else self._assignment(found)

    def sample(self, n: int = 1) -> List[List[int]]:
        """``n`` paths drawn at random, as one value per variable of the model.

        In an exact diagram every solution is equally likely to be drawn. A path of a relaxed
        diagram is not necessarily a solution: check it with :meth:`Model.check`. Call
        :func:`pyaicad.set_seed` first to make them reproducible. Empty if the model is
        unsatisfiable.
        """
        if self.is_unsat:
            return []
        return [
            self._assignment(self._native.get_solution(True)) for _ in range(operator.index(n))
        ]

    def _assignment(self, found: Any) -> List[int]:
        """The assignment of the model for a path given as ``(values, variable indices)``."""
        values, indices = found
        assignment = [self._model.domain(v)[0] for v in self._model.variables]
        for index, value in zip(indices, values):
            assignment[index] = value
        return assignment

    @property
    def widths(self) -> List[int]:
        """The number of nodes of each layer, from the root to the sink (which is a layer)."""
        return [self._native.number_nodes_in_layer(i) for i in range(self._native.number_layers())]

    @property
    def num_layers(self) -> int:
        """The number of layers, the sink's included."""
        return int(self._native.number_layers())

    def to_dot(self) -> str:
        """The diagram in Graphviz's DOT format."""
        return str(self._native.as_graphviz())

    def __repr__(self) -> str:
        kind = "exact" if self._width is None else f"relaxed, width {self._width}"
        return f"MDD({kind}, {self.num_layers} layers)"
