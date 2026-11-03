"""Exceptions raised by pyaicad."""

from __future__ import annotations

__all__ = ["AicadError", "ModelError", "CheckpointError", "FeatureUnavailableError"]


class AicadError(Exception):
    """Base class of the errors raised by pyaicad itself."""


class ModelError(AicadError, ValueError):
    """The model, or one of its variables or constraints, is invalid."""


class CheckpointError(AicadError):
    """A checkpoint is missing, incomplete, or cannot be used for the requested problems."""


class FeatureUnavailableError(AicadError):
    """The feature needs a part of pyaicad that this build does not include.

    Training and the neural searches come with the ``learning`` feature of the native library.
    """
