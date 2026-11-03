"""Local search.

:func:`local_search` takes a model, or a list of them, and a *search*: it starts from a random
assignment and repeatedly destroys part of it and repairs it, until it satisfies the model or a
limit is reached. Every search runs the same loop; they differ in how they repair. A
:class:`BeliefPropagationSearch` repairs with belief propagation on the diagrams of the
constraints; :class:`NeuralPriorSearch` starts from the predictions of a trained network, and
:class:`NeuralPriorBPSearch` then refines them with belief propagation.

>>> import pyaicad as ac
>>> m = ac.Model()
>>> x = m.int_vars(3, domain=range(3))
>>> m.add(ac.all_different(x))
>>> search = ac.BeliefPropagationSearch(destroy_fraction=0.5)
>>> result = ac.local_search(m, search, ac.Limits(iterations=100), seed=0)
>>> result.status, m.check(result.assignment)
(<Status.FEASIBLE: 'feasible'>, True)

A local search cannot prove that a model has no solution; for that, compile it
(:class:`pyaicad.Compiler`).
"""

from __future__ import annotations

import enum
import math
import os
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence, Union, overload

from . import _core
from ._runtime import require_learning, set_seed
from .learning import Checkpoint
from .model import Model

__all__ = [
    "local_search",
    "Limits",
    "Result",
    "Status",
    "NeuralPriorSearch",
    "NeuralPriorBPSearch",
    "BeliefPropagationSearch",
]


class Status(str, enum.Enum):
    """What a search established."""

    FEASIBLE = "feasible"  #: a solution was found
    UNKNOWN = "unknown"  #: none was found before a limit; a local search never proves infeasibility


@dataclass(frozen=True)
class Limits:
    """When to stop. Whichever limit is reached first stops the search.

    The limits apply to each problem separately: ``Limits(time=10)`` gives every problem of a
    list up to 10 seconds.
    """

    time: Optional[float] = None  #: wall-clock seconds
    iterations: Optional[int] = None  #: iterations of the search

    def __post_init__(self) -> None:
        if self.time is not None and not self.time > 0:
            raise ValueError(f"time must be positive, got {self.time!r}")
        if self.iterations is not None and self.iterations < 1:
            raise ValueError(f"iterations must be at least 1, got {self.iterations!r}")


@dataclass(frozen=True)
class Result:
    """The outcome of solving one model."""

    status: Status
    assignment: Optional[List[int]]  #: one value per variable; ``None`` unless FEASIBLE
    runtime: float  #: seconds, compilation included
    compile_time: float  #: seconds spent compiling the diagrams the search uses
    iterations: Optional[int]  #: iterations of the search
    search: str  #: which search produced it: "neural-prior", "neural-prior-bp" or "bp"

    @property
    def found(self) -> bool:
        """Whether a solution was found."""
        return self.status is Status.FEASIBLE


_DESTROY = {
    "random": _core.PyDestroyKind.Random,
    "worst": _core.PyDestroyKind.Worst,
    "related": _core.PyDestroyKind.Related,
}


def _check_search(destroy: str, destroy_fraction: float, batch_size: Optional[int]) -> None:
    """The checks shared by the local searches."""
    if destroy not in _DESTROY:
        raise ValueError(f"destroy must be one of {sorted(_DESTROY)}, got {destroy!r}")
    if not 0 < destroy_fraction <= 1:
        raise ValueError(f"destroy_fraction must be in (0, 1], got {destroy_fraction!r}")
    if batch_size is not None and batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size!r}")


class _LocalSearch:
    """What the searches share: they repeatedly destroy part of the current assignment and repair
    it, many problems of a :func:`local_search` call running together as batches.

    A subclass is one search architecture, with the settings of that architecture and nothing
    else; it says which native search it is (``_search``) and whether it uses a network
    (``_checkpoint``).
    """

    name: str
    batch_size: Optional[int]

    def with_(self, **changes: Any) -> Any:
        """A copy with some options changed."""
        return replace(self, **changes)  # type: ignore[type-var]

    def _search(self) -> Any:
        raise NotImplementedError

    def _checkpoint(self) -> Optional[Checkpoint]:
        return None

    def _solve(self, models: List[Model], limits: Limits, seed: Optional[int]) -> List[Result]:
        checkpoint = self._checkpoint()
        config = _core.PySolveConfig(
            search=self._search(),
            batch_size=self.batch_size,
            # The search takes whole seconds; a limit is rounded up so it is never cut short.
            time_limit=None if limits.time is None else math.ceil(limits.time),
            iteration_limit=limits.iterations,
            seed=seed,
        )
        solutions = _core.local_search(
            [m._native_problem() for m in models],
            config,
            None if checkpoint is None else str(checkpoint.path),
        )
        return [
            Result(
                status=Status.UNKNOWN if s.solution is None else Status.FEASIBLE,
                assignment=None if s.solution is None else list(s.solution),
                runtime=s.runtime,
                compile_time=s.compilation_runtime,
                iterations=s.iterations,
                search=self.name,
            )
            for s in solutions
        ]


def _load(checkpoint: Union[Checkpoint, str, "os.PathLike[str]"]) -> Checkpoint:
    return checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint.load(checkpoint)


@dataclass(frozen=True)
class NeuralPriorSearch(_LocalSearch):
    """Solves by local search guided by a trained network: at each iteration, part of the current
    assignment is destroyed and repaired from the network's predictions (its prior over the
    values).

    What problems a network accepts depends on the network: see its documentation in the Rust
    crate (``neural::consformer`` for ConsFormer).

    Args:
        checkpoint: The trained network, a :class:`~pyaicad.learning.Checkpoint` or the path of
            one.
        destroy: Which variables are repaired at each iteration: ``"random"`` ones, the ``"worst"``
            ones (those in the most violated constraints) or ``"related"`` ones (sharing
            constraints with a first one).
        destroy_fraction: The fraction of the variables repaired at each iteration.
        stochastic: Sample the values from the network's prediction instead of taking the most
            likely ones.
        temperature: Softens (above 1) or sharpens (below 1) the prediction when ``stochastic``.
        batch_size: How many problems are stepped together on the device; ``None`` puts them all in
            one batch. Batches run one after the other, each with the full limits. Use 1 to solve
            problems one at a time, as a deployment would.
    """

    checkpoint: Union[Checkpoint, str, "os.PathLike[str]"]
    destroy: str = "random"
    destroy_fraction: float = 1.0
    stochastic: bool = False
    temperature: float = 1.0
    batch_size: Optional[int] = None

    name = "neural-prior"

    def __post_init__(self) -> None:
        require_learning("NeuralPriorSearch")
        _check_search(self.destroy, self.destroy_fraction, self.batch_size)

    def _checkpoint(self) -> Optional[Checkpoint]:
        return _load(self.checkpoint)

    def _search(self) -> Any:
        return _core.PySearch.NeuralPrior(
            destroy_kind=_DESTROY[self.destroy],
            destroy_fraction=self.destroy_fraction,
            stochastic=self.stochastic,
            temperature=self.temperature,
        )


@dataclass(frozen=True)
class NeuralPriorBPSearch(_LocalSearch):
    """Solves by local search guided by a trained network whose predictions are the prior of
    belief propagation on the MDDs of the constraints, which chooses the values.

    What problems a network accepts depends on the network: see its documentation in the Rust
    crate (``neural::consformer`` for ConsFormer).

    Args:
        checkpoint: The trained network, a :class:`~pyaicad.learning.Checkpoint` or the path of
            one.
        destroy: Which variables are repaired at each iteration; see :class:`NeuralPriorSearch`.
            ``"related"`` by default, which suits the refinement.
        destroy_fraction: The fraction of the variables repaired at each iteration.
        stochastic: Sample the values from the refined marginals instead of taking the most likely
            ones.
        bp_iterations: Rounds of belief propagation per iteration.
        batch_size: How many problems are stepped together on the device; see
            :class:`NeuralPriorSearch`.
    """

    checkpoint: Union[Checkpoint, str, "os.PathLike[str]"]
    destroy: str = "related"
    destroy_fraction: float = 1.0
    stochastic: bool = False
    bp_iterations: int = 1
    batch_size: Optional[int] = None

    name = "neural-prior-bp"

    def __post_init__(self) -> None:
        require_learning("NeuralPriorBPSearch")
        _check_search(self.destroy, self.destroy_fraction, self.batch_size)
        if self.bp_iterations < 1:
            raise ValueError("bp_iterations must be at least 1")

    def _checkpoint(self) -> Optional[Checkpoint]:
        return _load(self.checkpoint)

    def _search(self) -> Any:
        return _core.PySearch.NeuralPriorBp(
            destroy_kind=_DESTROY[self.destroy],
            destroy_fraction=self.destroy_fraction,
            stochastic=self.stochastic,
            bp_iterations=self.bp_iterations,
        )


@dataclass(frozen=True)
class BeliefPropagationSearch(_LocalSearch):
    """Solves by local search that repairs with belief propagation on the MDDs of the constraints,
    starting from uniform probabilities. It needs no network and runs on the CPU only, and the
    problems of a batch may have different sizes.

    Args:
        destroy: Which variables are repaired at each iteration; see :class:`NeuralPriorSearch`.
        destroy_fraction: The fraction of the variables repaired at each iteration.
        stochastic: Sample the values from the marginals instead of taking the most likely ones.
            Uniform probabilities tie everywhere, so this is what lets the search move.
        bp_iterations: Rounds of belief propagation per iteration.
        batch_size: How many problems are stepped together; see :class:`NeuralPriorSearch`.
    """

    destroy: str = "random"
    destroy_fraction: float = 1.0
    stochastic: bool = True
    bp_iterations: int = 1
    batch_size: Optional[int] = None

    name = "bp"

    def __post_init__(self) -> None:
        _check_search(self.destroy, self.destroy_fraction, self.batch_size)
        if self.bp_iterations < 1:
            raise ValueError("bp_iterations must be at least 1")

    def _search(self) -> Any:
        return _core.PySearch.BeliefPropagation(
            destroy_kind=_DESTROY[self.destroy],
            destroy_fraction=self.destroy_fraction,
            stochastic=self.stochastic,
            bp_iterations=self.bp_iterations,
        )


Search = Union[NeuralPriorSearch, NeuralPriorBPSearch, BeliefPropagationSearch]
_SEARCHES = (NeuralPriorSearch, NeuralPriorBPSearch, BeliefPropagationSearch)


@overload
def local_search(
    problems: Model,
    search: Search,
    limits: Optional[Limits] = ...,
    *,
    seed: Optional[int] = ...,
) -> Result: ...


@overload
def local_search(
    problems: Sequence[Model],
    search: Search,
    limits: Optional[Limits] = ...,
    *,
    seed: Optional[int] = ...,
) -> List[Result]: ...


def local_search(
    problems: Union[Model, Sequence[Model]],
    search: Search,
    limits: Optional[Limits] = None,
    *,
    seed: Optional[int] = None,
) -> Union[Result, List[Result]]:
    """Runs a local search on a model, or on a list of models, and returns a :class:`Result` for
    each.

    Args:
        problems: A model, giving one result, or a list of models, giving a list of results in the
            same order.
        search: How the search repairs: :class:`NeuralPriorSearch`, :class:`NeuralPriorBPSearch`
            or :class:`BeliefPropagationSearch`, with their settings.
        limits: When to stop, per problem. No limit by default: a search that finds no solution
            then runs forever.
        seed: Makes the run reproducible: it seeds the search and every other source of randomness
            (as :func:`pyaicad.set_seed` does). Unset, the run follows the current state of those.
    """
    if not isinstance(search, _SEARCHES):
        raise TypeError(
            "search must be a NeuralPriorSearch, a NeuralPriorBPSearch or a "
            f"BeliefPropagationSearch, got {search!r}"
        )
    limits = Limits() if limits is None else limits
    single = isinstance(problems, Model)
    models = [problems] if isinstance(problems, Model) else list(problems)
    for m in models:
        if not isinstance(m, Model):
            raise TypeError(f"expected a Model, got {m!r}")
    if not models:
        return []
    if seed is not None:
        # The destroy choices and the sampling follow `seed` directly; the network uses the
        # process-wide generators, which this seeds too.
        set_seed(seed)
    results = search._solve(models, limits, seed)
    return results[0] if single else results
