"""Modelling: variables, constraints and the :class:`Model` that holds them.

A model is plain Python data. It records the variables and constraints added to it and builds the
native problem only when something needs it (compiling, solving, training, checking an
assignment). The consequences are that a model can be extended after it has been compiled, copied,
and pickled, and that every mistake in it is reported as a :class:`~pyaicad.ModelError` when the
constraint is built, instead of as a failure somewhere in the native code.

Variables have no names: a :class:`Var` is a handle returned by the model, and an assignment is a
list of values in the order in which the variables were created.

>>> import pyaicad as ac
>>> m = ac.Model()
>>> x = m.int_vars(3, domain=range(1, 4))
>>> m.add(ac.all_different(x))
>>> m.check([1, 2, 3]), m.check([1, 1, 3])
(True, False)
"""

from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from . import _core
from .errors import ModelError

__all__ = [
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
]

_model_ids = itertools.count()


class Var:
    """A decision variable of a model. Created by :class:`Model`, never directly.

    ``var.index`` is its position in an assignment.
    """

    __slots__ = ("index", "_owner")

    def __init__(self, index: int, owner: int) -> None:
        self.index = index
        self._owner = owner

    def __repr__(self) -> str:
        return f"Var({self.index})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Var):
            return NotImplemented
        return self.index == other.index and self._owner == other._owner

    def __hash__(self) -> int:
        return hash((self.index, self._owner))

    def __getstate__(self) -> Tuple[int, int]:
        return (self.index, self._owner)

    def __setstate__(self, state: Tuple[int, int]) -> None:
        self.index, self._owner = state


# --------------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------------


class Constraint:
    """A constraint, built by one of the functions below and added with :meth:`Model.add`.

    ``scope`` is the tuple of variables it constrains.
    """

    scope: Tuple[Var, ...]

    def _apply(self, native: Any) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


def _variables(variables: Iterable[Var], what: str, *, distinct: bool = False) -> Tuple[Var, ...]:
    scope = tuple(variables)
    for v in scope:
        if not isinstance(v, Var):
            raise ModelError(f"{what}: expected variables created by a Model, got {v!r}")
    if len({v._owner for v in scope}) > 1:
        raise ModelError(f"{what}: the variables come from different models")
    if distinct and len(set(scope)) != len(scope):
        raise ModelError(f"{what}: a variable appears more than once in the scope")
    return scope


def _ints(values: Iterable[Any], what: str) -> Tuple[int, ...]:
    try:
        return tuple(operator.index(v) for v in values)
    except TypeError:
        raise ModelError(f"{what}: values must be integers") from None


def _count(value: Any, what: str) -> int:
    try:
        n = operator.index(value)
    except TypeError:
        raise ModelError(f"{what} must be an integer, got {value!r}") from None
    if n < 0:
        raise ModelError(f"{what} must be non-negative, got {n}")
    return n


def _idx(scope: Sequence[Var]) -> List[int]:
    return [v.index for v in scope]


@dataclass(frozen=True)
class _AllDifferent(Constraint):
    scope: Tuple[Var, ...]

    def _apply(self, native: Any) -> None:
        native.add_all_different(_idx(self.scope))


@dataclass(frozen=True)
class _NotEqual(Constraint):
    x: Var
    y: Var

    @property
    def scope(self) -> Tuple[Var, ...]:  # type: ignore[override]
        return (self.x, self.y)

    def _apply(self, native: Any) -> None:
        native.add_not_equals(self.x.index, self.y.index)


@dataclass(frozen=True)
class _Equal(Constraint):
    x: Var
    value: int

    @property
    def scope(self) -> Tuple[Var, ...]:  # type: ignore[override]
        return (self.x,)

    def _apply(self, native: Any) -> None:
        native.add_equal(self.x.index, self.value)


@dataclass(frozen=True)
class _Among(Constraint):
    scope: Tuple[Var, ...]
    values: Tuple[int, ...]
    lb: int
    ub: int

    def _apply(self, native: Any) -> None:
        native.add_among(_idx(self.scope), list(self.values), self.lb, self.ub)


@dataclass(frozen=True)
class _AtLeast(Constraint):
    scope: Tuple[Var, ...]
    values: Tuple[int, ...]
    lb: int

    def _apply(self, native: Any) -> None:
        native.add_at_least(_idx(self.scope), list(self.values), self.lb)


@dataclass(frozen=True)
class _SumEquals(Constraint):
    scope: Tuple[Var, ...]
    target: int

    def _apply(self, native: Any) -> None:
        native.add_sum(_idx(self.scope), self.target)


@dataclass(frozen=True)
class _Gcc(Constraint):
    scope: Tuple[Var, ...]
    bounds: Tuple[Tuple[int, int, int], ...]

    def _apply(self, native: Any) -> None:
        native.add_gcc(_idx(self.scope), list(self.bounds))


@dataclass(frozen=True)
class _Regular(Constraint):
    scope: Tuple[Var, ...]
    transitions: Tuple[Tuple[Optional[int], ...], ...]
    initial_state: int
    accepting_states: Tuple[int, ...]

    def _apply(self, native: Any) -> None:
        native.add_regular(
            _idx(self.scope),
            [list(row) for row in self.transitions],
            self.initial_state,
            list(self.accepting_states),
        )


def all_different(variables: Iterable[Var]) -> Constraint:
    """The variables take pairwise different values."""
    return _AllDifferent(_variables(variables, "all_different", distinct=True))


def not_equal(x: Var, y: Var) -> Constraint:
    """``x`` and ``y`` take different values."""
    scope = _variables([x, y], "not_equal")
    return _NotEqual(*scope)


def equal(x: Var, value: int) -> Constraint:
    """``x`` takes ``value``. Adding it to a model shrinks the domain of ``x`` to that value."""
    (x,) = _variables([x], "equal")
    return _Equal(x, operator.index(value))


def among(variables: Iterable[Var], values: Iterable[int], lb: int, ub: int) -> Constraint:
    """Between ``lb`` and ``ub`` of the variables take a value in ``values``."""
    scope = _variables(variables, "among")
    lb, ub = _count(lb, "among: lb"), _count(ub, "among: ub")
    if lb > ub:
        raise ModelError(f"among: lb ({lb}) is greater than ub ({ub})")
    return _Among(scope, _ints(values, "among"), lb, ub)


def at_least(variables: Iterable[Var], values: Iterable[int], lb: int) -> Constraint:
    """At least ``lb`` of the variables take a value in ``values``."""
    scope = _variables(variables, "at_least")
    return _AtLeast(scope, _ints(values, "at_least"), _count(lb, "at_least: lb"))


def sum_equals(variables: Iterable[Var], target: int) -> Constraint:
    """The variables add up to ``target``. Values may be negative."""
    scope = _variables(variables, "sum_equals", distinct=True)
    return _SumEquals(scope, operator.index(target))


def gcc(
    variables: Iterable[Var],
    bounds: Union[Mapping[int, Tuple[int, int]], Iterable[Tuple[int, int, int]]],
) -> Constraint:
    """Global cardinality: each value of ``bounds`` occurs between its ``lo`` and ``hi`` times.

    ``bounds`` is either a mapping ``{value: (lo, hi)}`` or an iterable of ``(value, lo, hi)``.
    A value that is not in ``bounds`` is unconstrained.
    """
    scope = _variables(variables, "gcc", distinct=True)
    if isinstance(bounds, Mapping):
        triples = [(value, lo, hi) for value, (lo, hi) in bounds.items()]
    else:
        triples = [tuple(t) for t in bounds]  # type: ignore[misc]
    cleaned: List[Tuple[int, int, int]] = []
    for t in triples:
        if len(t) != 3:
            raise ModelError(f"gcc: expected (value, lo, hi), got {t!r}")
        value = operator.index(t[0])
        lo, hi = _count(t[1], "gcc: lo"), _count(t[2], "gcc: hi")
        if lo > hi:
            raise ModelError(f"gcc: lo ({lo}) is greater than hi ({hi}) for value {value}")
        cleaned.append((value, lo, hi))
    if len({value for value, _, _ in cleaned}) != len(cleaned):
        raise ModelError("gcc: a value appears more than once in bounds")
    return _Gcc(scope, tuple(cleaned))


def regular(
    variables: Iterable[Var],
    transitions: Sequence[Sequence[Optional[int]]],
    initial_state: int,
    accepting_states: Iterable[int],
) -> Constraint:
    """The variables, read in order, form a word accepted by an automaton.

    ``transitions[state][symbol]`` is the state reached from ``state`` on a symbol, or ``None`` if
    there is no transition. The columns are the *sorted* values found in the domains of the
    variables at the time the constraint is added to the model.
    """
    scope = _variables(variables, "regular", distinct=True)
    table = tuple(tuple(None if t is None else operator.index(t) for t in row) for row in transitions)
    states = len(table)
    if states == 0:
        raise ModelError("regular: the transition table is empty")
    initial_state = operator.index(initial_state)
    accepting = _ints(accepting_states, "regular")
    if not 0 <= initial_state < states:
        raise ModelError(f"regular: initial state {initial_state} is not a state of the table")
    for s in accepting:
        if not 0 <= s < states:
            raise ModelError(f"regular: accepting state {s} is not a state of the table")
    for row in table:
        for t in row:
            if t is not None and not 0 <= t < states:
                raise ModelError(f"regular: a transition leads to {t}, which is not a state")
    return _Regular(scope, table, initial_state, accepting)


# --------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------

Shape = Union[int, Tuple[int, ...]]


def _grid(shape: Shape, make: Any) -> Any:
    """Calls ``make()`` once per cell of ``shape`` and nests the results as lists."""
    if isinstance(shape, tuple):
        if not shape:
            raise ModelError("the shape must not be empty")
        head, rest = shape[0], shape[1:]
        return [_grid(rest, make) if rest else make() for _ in range(_count(head, "shape"))]
    return [make() for _ in range(_count(shape, "shape"))]


def _domain(domain: Iterable[int]) -> Tuple[int, ...]:
    values = _ints(domain, "domain")
    if not values:
        raise ModelError("a domain must contain at least one value")
    if len(set(values)) != len(values):
        raise ModelError("a domain must not contain the same value twice")
    return values


class Model:
    """A constraint satisfaction problem: variables with finite integer domains and constraints.

    >>> m = Model()
    >>> grid = m.int_vars((2, 2), domain=range(1, 3))   # a 2 x 2 grid of variables
    >>> m.add(all_different(grid[0]), all_different([row[0] for row in grid]))
    """

    def __init__(self) -> None:
        self._id = next(_model_ids)
        self._domains: List[Tuple[int, ...]] = []  # current domains
        self._created: List[Tuple[int, ...]] = []  # domains at creation, before any `equal`
        # What the model is made of, in the order it was built: ("var", index), ("constraint",
        # Constraint) or ("position", index, position). Replaying it builds the native problem.
        self._ops: List[Tuple[Any, ...]] = []
        self._constraints: List[Constraint] = []
        self._native: Optional[Any] = None

    # --- variables --- #

    def int_var(self, domain: Iterable[int]) -> Var:
        """A variable taking one of the integer values of ``domain``."""
        values = _domain(domain)
        var = Var(len(self._domains), self._id)
        self._domains.append(values)
        self._created.append(values)
        self._ops.append(("var", var.index))
        self._native = None
        return var

    def bool_var(self) -> Var:
        """A variable taking the value 0 or 1."""
        return self.int_var((0, 1))

    def int_vars(self, shape: Shape, domain: Iterable[int]) -> Any:
        """Variables sharing a domain.

        ``shape`` is a number ``n``, giving a list of ``n`` variables, or a tuple, giving nested
        lists: ``(9, 9)`` is a list of 9 rows of 9 variables.
        """
        values = _domain(domain)
        return _grid(shape, lambda: self.int_var(values))

    def bool_vars(self, shape: Shape) -> Any:
        """Boolean variables, shaped like :meth:`int_vars`."""
        return _grid(shape, self.bool_var)

    def set_position(self, var: Var, position: Iterable[int]) -> None:
        """Attaches coordinates to a variable (a cell's row and column, say).

        Learned models can use them as positional information; they have no meaning for the
        constraints.
        """
        self._own(var)
        self._ops.append(("position", var.index, tuple(_count(p, "position") for p in position)))
        self._native = None

    # --- constraints --- #

    def add(self, *constraints: Union[Constraint, Iterable[Constraint]]) -> None:
        """Adds constraints. Each argument is a constraint or an iterable of constraints."""
        flat: List[Constraint] = []
        for c in constraints:
            if isinstance(c, Constraint):
                flat.append(c)
            else:
                try:
                    flat.extend(c)
                except TypeError:
                    raise ModelError(f"cannot add {c!r}: it is not a constraint") from None
        for c in flat:
            if not isinstance(c, Constraint):
                raise ModelError(f"cannot add {c!r}: it is not a constraint")
            for v in c.scope:
                self._own(v)
        # Validate everything before changing anything, so a bad argument leaves no trace.
        for c in flat:
            if isinstance(c, _Equal):
                if c.value not in self._domains[c.x.index]:
                    raise ModelError(f"equal: {c.value} is not in the domain of {c.x}")
        for c in flat:
            self._constraints.append(c)
            self._ops.append(("constraint", c))
            if isinstance(c, _Equal):
                self._domains[c.x.index] = (c.value,)
        self._native = None

    # --- inspection --- #

    @property
    def variables(self) -> List[Var]:
        """All the variables, in the order of an assignment."""
        return [Var(i, self._id) for i in range(len(self._domains))]

    @property
    def constraints(self) -> Tuple[Constraint, ...]:
        """All the constraints, in the order they were added."""
        return tuple(self._constraints)

    @property
    def num_vars(self) -> int:
        return len(self._domains)

    @property
    def num_constraints(self) -> int:
        return len(self._constraints)

    def domain(self, var: Var) -> Tuple[int, ...]:
        """The current domain of a variable."""
        self._own(var)
        return self._domains[var.index]

    # --- checking --- #

    def check(self, assignment: Sequence[int]) -> bool:
        """Whether ``assignment`` (one value per variable) satisfies the model."""
        return self._native_problem().is_solution(self._assignment(assignment))

    def satisfied_fraction(self, assignment: Sequence[int]) -> float:
        """The fraction of the constraints that ``assignment`` satisfies."""
        return self._native_problem().satisfied_fraction(self._assignment(assignment))

    # --- copying and pickling --- #

    def copy(self) -> "Model":
        """An independent copy: later changes to one do not affect the other.

        The variables of the original are valid in the copy, so a constraint built on them can be
        added to either.
        """
        other = Model()
        other._id = self._id
        other._domains = list(self._domains)
        other._created = list(self._created)
        other._ops = list(self._ops)
        other._constraints = list(self._constraints)
        return other

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_native"] = None  # rebuilt on demand
        return state

    def __repr__(self) -> str:
        return f"Model({self.num_vars} variables, {self.num_constraints} constraints)"

    # --- internals --- #

    def _own(self, var: Var) -> None:
        if not isinstance(var, Var):
            raise ModelError(f"expected a variable created by this model, got {var!r}")
        if var._owner != self._id or var.index >= len(self._domains):
            raise ModelError(f"{var} does not belong to this model")

    def _assignment(self, assignment: Sequence[int]) -> List[int]:
        values = _ints(assignment, "assignment")
        if len(values) != self.num_vars:
            raise ModelError(
                f"an assignment needs {self.num_vars} values (one per variable), got {len(values)}"
            )
        return list(values)

    def _native_problem(self) -> Any:
        """The native problem, built on first use and rebuilt after any change."""
        if self._native is None:
            native = _core.PyProblem()
            for op in self._ops:
                kind = op[0]
                if kind == "var":
                    native.add_int_var(list(self._created[op[1]]))
                elif kind == "constraint":
                    op[1]._apply(native)
                else:  # "position"
                    native.set_variable_position(op[1], list(op[2]))
            self._native = native
        return self._native
