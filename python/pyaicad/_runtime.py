"""Process-wide settings: the random seed, the number of threads and the log level."""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import _core
from .errors import FeatureUnavailableError

__all__ = [
    "HAS_LEARNING",
    "BuildInfo",
    "version",
    "set_seed",
    "set_num_threads",
    "get_num_threads",
    "set_log_level",
    "require_learning",
]

#: Whether this build of pyaicad includes training and the neural searches (the ``learning``
#: feature of the native library, which is heavy to build). Compilation, sampling and the
#: belief-propagation search are always there.
HAS_LEARNING: bool = bool(getattr(_core, "HAS_LEARNING", False))


@dataclass(frozen=True)
class BuildInfo:
    """The build of the library that is running: what was compiled into the native extension.

    The version of the source tree says little about the code that runs, which may be an
    installed copy of another commit. Record this next to any measurement.
    """

    #: The version of the library.
    version: str
    #: The full hash of the commit it was built from, or ``None`` if it could not be known (a
    #: build from a source archive without ``.git``).
    commit: Optional[str]
    #: The Rust compiler, as printed by ``rustc --version``; empty if it could not be run.
    rustc: str
    #: The compilation target, for example ``x86_64-unknown-linux-gnu``.
    target: str
    #: The cargo profile, ``"debug"`` or ``"release"``. A debug build is much slower.
    profile: str
    #: The optimisation level of the profile: ``"0"``, ``"1"``, ``"2"``, ``"3"``, ``"s"`` or ``"z"``.
    opt_level: str
    #: The cargo features that were on, sorted (``"learning"``, ``"python"``).
    features: Tuple[str, ...]

    def __str__(self) -> str:
        commit = self.commit[:12] if self.commit else "unknown"
        return (
            f"{self.version} (commit {commit}, {self.rustc or 'rustc unknown'}, {self.profile} "
            f"opt-level {self.opt_level}, {self.target}, features: [{', '.join(self.features)}])"
        )


def version() -> BuildInfo:
    """Returns the build of the library that is running: its version, the commit it was built
    from, the Rust compiler and the build settings.

    >>> import pyaicad as ac
    >>> ac.version().profile in ("debug", "release")
    True

    Benchmarks are only comparable between commits, so run them from a clean copy of a commit
    and keep this with the results. The same record is the first line of every file written by
    :func:`pyaicad.diagnostics.enable_data_log`.
    """
    info = dict(_core.build_info())
    info["features"] = tuple(info["features"])
    return BuildInfo(**info)


def require_learning(what: str) -> None:
    """Raises :class:`FeatureUnavailableError` naming ``what`` if the build has no ``learning``."""
    if not HAS_LEARNING:
        raise FeatureUnavailableError(
            f"{what} needs the learning part of pyaicad, which this build does not include. "
            "Install the full build, or build from source with `maturin develop --features learning`."
        )


_LEVELS: Dict[str, str] = {
    "off": "set_verbosity_off",
    "error": "set_verbosity_error",
    "warning": "set_verbosity_warning",
    "info": "set_verbosity_info",
    "debug": "set_verbosity_debug",
    "trace": "set_verbosity_trace",
}

_LOGGER = "aicad"


def set_seed(seed: int) -> None:
    """Seeds every source of randomness of the library, so that a run can be reproduced.

    This covers weight initialisation, sampling from a diagram or a network, the choices of the
    search, and the shuffling done during training.
    """
    _core.set_seed(int(seed))


def set_num_threads(n: int) -> None:
    """Sets how many threads the library may use for its parallel work: solving a batch of
    problems, training, and building the datasets for it.

    Unless this is called, the library uses a quarter of the threads available on the machine
    (at least one), so that it does not take over a shared server. Raise it to use more of the
    machine. It can be called at any time, even between two uses of the library; work already
    running finishes with the threads it had.

    Compiling a single diagram always uses one thread. The number of threads is unrelated to
    ``batch_size`` and to what runs on a GPU.

    Raises:
        TypeError: ``n`` is not an integer.
        ValueError: ``n`` is less than 1.
    """
    if isinstance(n, bool):
        raise TypeError(f"n must be an integer, got {n!r}")
    try:
        count = operator.index(n)
    except TypeError:
        raise TypeError(f"n must be an integer, got {n!r}") from None
    if count < 1:
        raise ValueError(f"n must be at least 1, got {count}")
    _core.set_num_threads(count)


def get_num_threads() -> int:
    """The number of threads the library uses for its parallel work: the value given to
    :func:`set_num_threads`, or the default (a quarter of the available threads, at least one)."""
    return _core.get_num_threads()


def set_log_level(level: str) -> None:
    """Sets how much the library logs: ``"off"`` (the default), ``"error"``, ``"warning"``,
    ``"info"``, ``"debug"`` or ``"trace"``.

    The library logs through the standard :mod:`logging` module, under the logger ``"aicad"``.
    If your program has not configured logging, a handler writing to standard error is added to
    that logger so the messages are visible.
    """
    try:
        setter = _LEVELS[level]
    except (KeyError, TypeError):
        raise ValueError(f"level must be one of {list(_LEVELS)}, got {level!r}") from None
    logger = logging.getLogger(_LOGGER)
    if level != "off" and not logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    getattr(_core, setter)()
