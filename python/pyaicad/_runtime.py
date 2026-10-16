"""Process-wide settings: the random seed, the number of threads and the log level."""

from __future__ import annotations

import logging
import operator
from typing import Dict

from . import _core

__all__ = ["set_seed", "set_num_threads", "get_num_threads", "set_log_level"]

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
