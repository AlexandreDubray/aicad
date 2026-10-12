"""Exceptions raised by pyaicad."""

from __future__ import annotations

__all__ = ["AicadError", "ModelError", "CheckpointError"]


class AicadError(Exception):
    """Base class of the errors raised by pyaicad itself."""


class ModelError(AicadError, ValueError):
    """The model, or one of its variables or constraints, is invalid."""


class CheckpointError(AicadError):
    """A checkpoint is missing, incomplete, or cannot be used for the requested problems."""
