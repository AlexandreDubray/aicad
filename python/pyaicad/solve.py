"""Solving models.

:func:`solve` takes a model, or a list of them, and a *method*: :class:`Compilation` compiles an
MDD and reads a solution off it; :class:`NLS` runs neural local search with a trained network.

>>> import pyaicad as ac
>>> m = ac.Model()
>>> x = m.int_vars(3, domain=range(3))
>>> m.add(ac.all_different(x))
>>> result = ac.solve(m)
>>> result.status, m.check(result.assignment)
(<Status.FEASIBLE: 'feasible'>, True)
"""

from __future__ import annotations

import enum
import math
import os
import time
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Sequence, Union, overload

from . import _core
from ._runtime import set_seed
from .errors import CheckpointError
from .learning import Checkpoint
from .mdd import compile as compile_mdd
from .model import Model

__all__ = ["solve", "Limits", "Result", "Status", "Compilation", "NLS"]


class Status(str, enum.Enum):
    """What a solve established."""

    FEASIBLE = "feasible"  #: a solution was found
    INFEASIBLE = "infeasible"  #: the model was proved to have no solution
    UNKNOWN = "unknown"  #: neither: the method stopped, or cannot decide


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
    compile_time: float  #: seconds spent compiling diagrams (0 when the method compiles none)
    iterations: Optional[int]  #: iterations of the search, ``None`` for methods without any
    method: str

    @property
    def found(self) -> bool:
        """Whether a solution was found."""
        return self.status is Status.FEASIBLE


@dataclass(frozen=True)
class Compilation:
    """Solves by compiling an MDD and reading a solution off it.

    Exact when ``width`` is ``None`` (see :func:`pyaicad.compile` about memory), relaxed otherwise;
    a relaxed diagram can fail to give a solution, in which case the status is UNKNOWN.
    It is complete: an exact diagram decides both feasible and infeasible, and a relaxed one can
    still prove infeasibility.

    Limits are not supported yet: a compilation runs to its end.
    """

    width: Optional[int] = None
    order: Union[str, Sequence[int]] = "auto"
    merge: str = "less_relaxed"

    name = "compilation"

    def _solve(self, models: List[Model], limits: Limits, seed: Optional[int]) -> List[Result]:
        if limits.time is not None or limits.iterations is not None:
            raise ValueError("Compilation does not support limits yet")
        return [self._solve_one(m) for m in models]

    def _solve_one(self, model: Model) -> Result:
        start = time.perf_counter()
        mdd = compile_mdd(model, width=self.width, order=self.order, merge=self.merge)
        compile_time = time.perf_counter() - start
        assignment = mdd.solution()
        if mdd.is_unsat:
            status = Status.INFEASIBLE
        elif assignment is not None and model.check(assignment):
            status = Status.FEASIBLE
        else:
            status, assignment = Status.UNKNOWN, None
        runtime = time.perf_counter() - start
        return Result(status, assignment, runtime, compile_time, None, self.name)


_DESTROY = {
    "random": _core.PyDestroyKind.Random,
    "worst": _core.PyDestroyKind.Worst,
    "related": _core.PyDestroyKind.Related,
}
_PRIOR = {"network": _core.PyPriorKind.Network, "uniform": _core.PyPriorKind.Uniform}
_DECODE = {"logits": _core.PyDecodeKind.Logits, "mdd": _core.PyDecodeKind.MddSampling}
_STATUS = {
    "Satisfiable": Status.FEASIBLE,
    "Unsatisfiable": Status.INFEASIBLE,
    "Unknown": Status.UNKNOWN,
}


def _choice(name: str, value: str, options: Any) -> None:
    if value not in options:
        raise ValueError(f"{name} must be one of {sorted(options)}, got {value!r}")


@dataclass(frozen=True)
class NLS:
    """Solves by neural local search: repeatedly destroy part of the current assignment and
    repair it using the predictions of a trained network.

    Many problems in one :func:`solve` call run together as a batch, which is what makes it fast.
    Every problem of a batch needs the same number of variables.

    Args:
        checkpoint: The trained network (a :class:`~pyaicad.learning.Checkpoint` or the path of
            one). Only ``prior="uniform"`` can go without.
        destroy: Which variables are repaired at each iteration: ``"random"`` ones, the ``"worst"``
            ones (those in the most violated constraints) or ``"related"`` ones (sharing
            constraints with a first one).
        destroy_fraction: The fraction of the variables repaired at each iteration.
        prior: Where the predictions come from: the ``"network"``, or ``"uniform"`` (no network;
            needs ``decode="mdd"``).
        decode: How a prediction becomes values: ``"logits"`` reads the network's output directly,
            ``"mdd"`` first refines it by belief propagation on the constraints' MDDs.
        bp_iterations: Rounds of belief propagation; only for ``decode="mdd"``.
        stochastic: Sample the values from the prediction instead of taking the most likely ones.
        temperature: Softens (above 1) or sharpens (below 1) the prediction when ``stochastic``.
        batch_size: How many problems are on the device at once; ``None`` puts them all.
        parallel_batches: How many batches run concurrently. Raise it only when ``batch_size`` is
            small enough for several batches to fit in memory together.
    """

    checkpoint: Optional[Union[Checkpoint, str, "os.PathLike[str]"]] = None
    destroy: str = "random"
    destroy_fraction: float = 1.0
    prior: str = "network"
    decode: str = "logits"
    bp_iterations: int = 1
    stochastic: bool = False
    temperature: float = 1.0
    batch_size: Optional[int] = None
    parallel_batches: int = 1

    name = "nls"

    def __post_init__(self) -> None:
        _choice("destroy", self.destroy, _DESTROY)
        _choice("prior", self.prior, _PRIOR)
        _choice("decode", self.decode, _DECODE)
        if self.prior == "uniform" and self.decode != "mdd":
            raise ValueError("prior='uniform' needs decode='mdd': raw logits would carry no information")
        if not 0 < self.destroy_fraction <= 1:
            raise ValueError(f"destroy_fraction must be in (0, 1], got {self.destroy_fraction!r}")
        if self.bp_iterations < 1 or self.parallel_batches < 1:
            raise ValueError("bp_iterations and parallel_batches must be at least 1")
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size!r}")

    @staticmethod
    def preset(name: str, checkpoint: Optional[Any] = None, **changes: Any) -> "NLS":
        """A configuration under a name, optionally with changes.

        ``"consformer"`` repairs from the network's logits; ``"consformer-mdd"`` refines them with
        belief propagation on the MDDs and chooses the variables to repair by relatedness.
        """
        presets = {
            "consformer": {},
            "consformer-mdd": {"decode": "mdd", "destroy": "related"},
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(presets)}")
        return NLS(checkpoint, **{**presets[name], **changes})

    def with_(self, **changes: Any) -> "NLS":
        """A copy with some options changed."""
        return replace(self, **changes)

    def _checkpoint(self) -> Optional[Checkpoint]:
        if self.checkpoint is None:
            if self.prior == "network":
                raise CheckpointError("NLS needs a checkpoint when prior='network'")
            return None
        if isinstance(self.checkpoint, Checkpoint):
            return self.checkpoint
        return Checkpoint.load(self.checkpoint)

    def _solve(self, models: List[Model], limits: Limits, seed: Optional[int]) -> List[Result]:
        checkpoint = self._checkpoint()
        config = _core.PySolveConfig(
            network_kind=_core.PyNetworkKind.ConsFormer,
            batch_size=self.batch_size,
            destroy_kind=_DESTROY[self.destroy],
            destroy_fraction=self.destroy_fraction,
            stochastic_decode=self.stochastic,
            temperature=self.temperature,
            decode_kind=_DECODE[self.decode],
            bp_iterations=self.bp_iterations,
            prior_kind=_PRIOR[self.prior],
            max_concurrent_chunks=self.parallel_batches,
            # The search takes whole seconds; a limit is rounded up so it is never cut short.
            time_limit=None if limits.time is None else math.ceil(limits.time),
            iteration_limit=limits.iterations,
            seed=seed,
        )
        solutions = _core.neural_local_search(
            [m._native_problem() for m in models],
            None if checkpoint is None else str(checkpoint.path),
            config,
        )
        return [
            Result(
                status=_STATUS[str(s.status).rsplit(".", 1)[-1]],
                assignment=None if s.solution is None else list(s.solution),
                runtime=s.runtime,
                compile_time=s.compilation_runtime,
                iterations=s.iterations,
                method=self.name,
            )
            for s in solutions
        ]


Method = Union[Compilation, NLS]


@overload
def solve(
    problems: Model,
    method: Optional[Method] = ...,
    limits: Optional[Limits] = ...,
    *,
    seed: Optional[int] = ...,
) -> Result: ...


@overload
def solve(
    problems: Sequence[Model],
    method: Optional[Method] = ...,
    limits: Optional[Limits] = ...,
    *,
    seed: Optional[int] = ...,
) -> List[Result]: ...


def solve(
    problems: Union[Model, Sequence[Model]],
    method: Optional[Method] = None,
    limits: Optional[Limits] = None,
    *,
    seed: Optional[int] = None,
) -> Union[Result, List[Result]]:
    """Solves a model, or a list of models, and returns a :class:`Result` for each.

    Args:
        problems: A model, giving one result, or a list of models, giving a list of results in the
            same order.
        method: How to solve: :class:`Compilation` (the default, exact) or :class:`NLS`.
        limits: When to stop, per problem. No limit by default.
        seed: Makes the run reproducible: it seeds the search and every other source of randomness
            (as :func:`pyaicad.set_seed` does). Unset, the run follows the current state of those.
    """
    method = Compilation() if method is None else method
    if not isinstance(method, (Compilation, NLS)):
        raise TypeError(f"method must be a Compilation or an NLS, got {method!r}")
    limits = Limits() if limits is None else limits
    single = isinstance(problems, Model)
    models = [problems] if isinstance(problems, Model) else list(problems)
    for m in models:
        if not isinstance(m, Model):
            raise TypeError(f"expected a Model, got {m!r}")
    if not models:
        return []
    if seed is not None:
        # The destroy choices follow `seed` directly; sampling and the network use the process-wide
        # generators, which this seeds too, so that the same call gives the same results.
        set_seed(seed)
    results = method._solve(models, limits, seed)
    return results[0] if single else results
