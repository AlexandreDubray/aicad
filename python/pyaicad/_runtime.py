"""Process-wide settings: the random seed and the log level."""

from __future__ import annotations

import logging
from typing import Dict

from . import _core

__all__ = ["set_seed", "set_log_level"]

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
