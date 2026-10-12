"""Structured recording of timings, for offline analysis.

This is for experiments and may change between releases. While recording is on, the library
writes one JSON object per line to a file, which ``pandas.read_json(path, lines=True)`` reads.
"""

from __future__ import annotations

import os
from typing import Union

from . import _core

__all__ = ["enable_data_log", "disable_data_log"]


def enable_data_log(path: Union[str, "os.PathLike[str]"]) -> None:
    """Starts recording to ``path``, truncating it first."""
    _core.enable_data_log(os.fspath(path))


def disable_data_log() -> None:
    """Stops recording and closes the file. Call it when done: nothing flushes it at exit."""
    _core.disable_data_log()
