"""Training the networks that guide the search.

A training set is a list of problems: one problem per training example. :class:`Dataset` splits a
list into training, validation and test sets; :func:`fit` trains a network on it with a named
recipe; the trained network is a :class:`Checkpoint` that :class:`pyaicad.NLS` solves with.

>>> import pyaicad as ac
>>> from pyaicad import learning
>>> data = learning.Dataset.split(problems, seed=0)                  # doctest: +SKIP
>>> ckpt = learning.fit(data, "consformer-mdd", checkpoint_dir="ckpt", epochs=50)  # doctest: +SKIP
>>> results = ac.solve(data.test, ac.NLS.preset("consformer-mdd", ckpt))  # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from . import _core
from ._runtime import set_seed
from .errors import CheckpointError
from .mdd import _merge_heuristic, _ordering
from .model import Model

__all__ = [
    "Dataset",
    "Network",
    "Adam",
    "AdamW",
    "SGD",
    "Optimiser",
    "Training",
    "Recipe",
    "RECIPES",
    "fit",
    "load",
    "Checkpoint",
]

PathLike = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """Training, validation and (optionally) test problems.

    The validation set is used during training to keep the best network; the test set is never
    seen by training and is there for you to evaluate on.
    """

    train: Tuple[Model, ...]
    validation: Tuple[Model, ...]
    test: Tuple[Model, ...] = ()

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.train:
            raise ValueError("a dataset needs at least one training problem")
        if not self.validation:
            raise ValueError("a dataset needs at least one validation problem")

    @classmethod
    def split(
        cls,
        problems: Iterable[Model],
        fractions: Tuple[float, ...] = (0.8, 0.1, 0.1),
        *,
        seed: Optional[int] = None,
    ) -> "Dataset":
        """Splits ``problems`` at random into training, validation and test sets.

        ``fractions`` gives the share of each set (they are normalised to sum to one); with two
        numbers there is no test set. The same ``seed`` gives the same split.
        """
        items = list(problems)
        if len(fractions) not in (2, 3) or any(f < 0 for f in fractions) or sum(fractions) <= 0:
            raise ValueError("fractions must be two or three non-negative numbers")
        random.Random(seed).shuffle(items)
        total = float(sum(fractions))
        n_train = round(len(items) * fractions[0] / total)
        n_val = round(len(items) * fractions[1] / total)
        if len(fractions) == 2:
            n_val = len(items) - n_train
        return cls(
            train=tuple(items[:n_train]),
            validation=tuple(items[n_train : n_train + n_val]),
            test=tuple(items[n_train + n_val :]),
        )

    def __iter__(self):  # pragma: no cover - convenience
        yield from (*self.train, *self.validation, *self.test)

    def __len__(self) -> int:
        return len(self.train) + len(self.validation) + len(self.test)


# --------------------------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------------------------

_DESTROY = {
    "random": _core.PyDestroyKind.Random,
    "worst": _core.PyDestroyKind.Worst,
    "related": _core.PyDestroyKind.Related,
}


@dataclass(frozen=True)
class Network:
    """The architecture of a ConsFormer network.

    ``domain_size`` is the number of values the network can predict: values ``0`` to
    ``domain_size - 1``. ``"auto"`` takes the smallest size covering the problems it is trained
    on, which requires non-negative values.
    """

    domain_size: Union[int, str] = "auto"
    embedding_size: int = 128
    hidden_size: int = 128
    num_heads: int = 1
    expand_size: int = 128
    num_layers: int = 1
    dropout: float = 0.0
    bias: bool = True
    positional_encoding_dimensions: int = 0
    mask_fraction: float = 0.5
    tau: float = 0.1
    learned_attention_mask: bool = False


def _check_optimiser(
    lr: float, grad_clip_norm: Optional[float], grad_clip_value: Optional[float]
) -> None:
    if not lr > 0:
        raise ValueError(f"lr must be positive, got {lr!r}")
    if grad_clip_norm is not None and grad_clip_value is not None:
        raise ValueError("set grad_clip_norm or grad_clip_value, not both")
    for name, value in (("grad_clip_norm", grad_clip_norm), ("grad_clip_value", grad_clip_value)):
        if value is not None and not value > 0:
            raise ValueError(f"{name} must be positive, got {value!r}")


def _check_betas(beta_1: float, beta_2: float, epsilon: float) -> None:
    if not (0 <= beta_1 < 1 and 0 <= beta_2 < 1):
        raise ValueError(f"beta_1 and beta_2 must be in [0, 1), got {beta_1!r} and {beta_2!r}")
    if not epsilon > 0:
        raise ValueError(f"epsilon must be positive, got {epsilon!r}")


@dataclass(frozen=True)
class Adam:
    """The Adam optimiser. ``weight_decay`` is an L2 penalty (``None`` for none).

    ``grad_clip_norm`` clips the L2 norm of each gradient tensor and ``grad_clip_value`` clips each
    element of the gradient; set at most one.
    """

    lr: float = 3e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-5
    amsgrad: bool = False
    weight_decay: Optional[float] = None
    grad_clip_norm: Optional[float] = None
    grad_clip_value: Optional[float] = None

    def __post_init__(self) -> None:
        _check_optimiser(self.lr, self.grad_clip_norm, self.grad_clip_value)
        _check_betas(self.beta_1, self.beta_2, self.epsilon)
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError(f"weight_decay must not be negative, got {self.weight_decay!r}")


@dataclass(frozen=True)
class AdamW:
    """Adam with a decoupled weight decay, which is always applied (``0`` for none).

    ``grad_clip_norm`` clips the L2 norm of each gradient tensor and ``grad_clip_value`` clips each
    element of the gradient; set at most one.
    """

    lr: float = 3e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-5
    amsgrad: bool = False
    weight_decay: float = 1e-4
    grad_clip_norm: Optional[float] = None
    grad_clip_value: Optional[float] = None

    def __post_init__(self) -> None:
        _check_optimiser(self.lr, self.grad_clip_norm, self.grad_clip_value)
        _check_betas(self.beta_1, self.beta_2, self.epsilon)
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must not be negative, got {self.weight_decay!r}")


@dataclass(frozen=True)
class SGD:
    """Stochastic gradient descent, with momentum if ``momentum`` is set.

    ``dampening`` and ``nesterov`` only apply with a momentum. ``weight_decay`` is an L2 penalty
    (``None`` for none). ``grad_clip_norm`` clips the L2 norm of each gradient tensor and
    ``grad_clip_value`` clips each element of the gradient; set at most one.
    """

    lr: float = 1e-2
    momentum: Optional[float] = None
    dampening: float = 0.0
    nesterov: bool = False
    weight_decay: Optional[float] = None
    grad_clip_norm: Optional[float] = None
    grad_clip_value: Optional[float] = None

    def __post_init__(self) -> None:
        _check_optimiser(self.lr, self.grad_clip_norm, self.grad_clip_value)
        if self.momentum is None and (self.nesterov or self.dampening != 0):
            raise ValueError("dampening and nesterov need a momentum")
        if self.momentum is not None and not 0 <= self.momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum!r}")
        if self.weight_decay is not None and self.weight_decay < 0:
            raise ValueError(f"weight_decay must not be negative, got {self.weight_decay!r}")


Optimiser = Union[Adam, AdamW, SGD]
"""An optimiser with its settings: one of :class:`Adam`, :class:`AdamW` and :class:`SGD`."""


@dataclass(frozen=True)
class Training:
    """How the network is optimised.

    ``optimiser`` is the optimiser with its learning rate and settings (Adam by default). The
    network is evaluated on the validation set every ``validation_interval`` epochs, and the best
    one so far is kept; so ``validation_interval`` must not exceed ``epochs``.

    ``initial_weights`` is the checkpoint directory of an earlier run: its weights initialise the
    network before the first epoch, to continue or fine-tune that run. The network settings must
    be the ones the weights were trained with. By default the network starts from scratch.
    """

    optimiser: Optimiser = field(default_factory=Adam)
    epochs: int = 10
    batch_size: int = 32
    validation_interval: int = 10
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 1e-4
    initial_weights: Optional[PathLike] = None
    destroy: str = "random"  # which variables are re-predicted in a training step; "consformer" only

    def __post_init__(self) -> None:
        if not isinstance(self.optimiser, (Adam, AdamW, SGD)):
            raise TypeError(f"optimiser must be Adam, AdamW or SGD, got {self.optimiser!r}")
        if self.destroy not in _DESTROY:
            raise ValueError(f"destroy must be one of {sorted(_DESTROY)}, got {self.destroy!r}")


@dataclass(frozen=True)
class Recipe:
    """A way of training a network: the loss, the architecture and the optimiser settings.

    Use one of the named recipes in :data:`RECIPES`, optionally with changes:

    >>> Recipe.named("consformer-mdd").replace(epochs=50, lr=1e-3).training.epochs
    50
    >>> Recipe.named("consformer").replace(optimiser=SGD(momentum=0.9)).training.optimiser.lr
    0.01

    ``replace`` takes the name of any setting of the network, the training, or the recipe itself,
    and refuses names it does not know.
    """

    name: str
    loss: str  # "classical": hand-written penalty per constraint type; "mdd": exact MDD model count
    network: Network = field(default_factory=Network)
    training: Training = field(default_factory=Training)
    order: Union[str, Tuple[int, ...]] = "auto"  # how constraint MDDs are ordered ("mdd" loss)
    merge: str = "less_relaxed"  # and merged ("mdd" loss); has no effect on an exact compile

    def __post_init__(self) -> None:
        if self.loss not in ("classical", "mdd"):
            raise ValueError(f"loss must be 'classical' or 'mdd', got {self.loss!r}")

    @staticmethod
    def named(name: str) -> "Recipe":
        """One of the recipes of :data:`RECIPES`."""
        try:
            return RECIPES[name]
        except KeyError:
            raise ValueError(f"unknown recipe {name!r}; choose from {sorted(RECIPES)}") from None

    def replace(self, **changes: Any) -> "Recipe":
        """A copy with some settings changed, by name.

        A setting of the optimiser (``lr``, ``momentum``, ...) changes the optimiser the recipe
        has once ``optimiser=`` is applied, if given.
        """
        own = {f.name for f in dataclasses.fields(self)} - {"network", "training"}
        net = {f.name for f in dataclasses.fields(Network)}
        train = {f.name for f in dataclasses.fields(Training)}
        top: Dict[str, Any] = {}
        n: Dict[str, Any] = {}
        t: Dict[str, Any] = {}
        o: Dict[str, Any] = {}
        optimiser = changes.get("optimiser", self.training.optimiser)
        opt = (
            {f.name for f in dataclasses.fields(optimiser)}
            if dataclasses.is_dataclass(optimiser)
            else set()
        )
        for key, value in changes.items():
            if key in own:
                top[key] = value
            elif key in net:
                n[key] = value
            elif key in train:
                t[key] = value
            elif key in opt:
                o[key] = value
            else:
                known = sorted(own | net | train | opt)
                raise TypeError(f"unknown setting {key!r}; the settings are {known}")
        training = dataclasses.replace(self.training, **t)
        if o:
            training = dataclasses.replace(
                training, optimiser=dataclasses.replace(training.optimiser, **o)
            )
        return dataclasses.replace(
            self,
            network=dataclasses.replace(self.network, **n),
            training=training,
            **top,
        )


RECIPES: Dict[str, Recipe] = {
    "consformer": Recipe("consformer", loss="classical"),
    "consformer-mdd": Recipe("consformer-mdd", loss="mdd"),
}
"""The named recipes. ``"consformer"`` trains with a hand-written penalty per constraint type;
``"consformer-mdd"`` trains against the exact weighted model count of each constraint's MDD."""


# --------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    """A trained network on disk: a directory holding ``config.json`` and ``weights.mpk``."""

    path: Path
    config: Dict[str, Any]

    @property
    def domain_size(self) -> int:
        """The number of values the network predicts."""
        return int(self.config["domain_size"])

    @classmethod
    def load(cls, path: PathLike) -> "Checkpoint":
        """Opens the checkpoint in directory ``path``, checking that it is complete."""
        directory = Path(path)
        config_file, weights_file = directory / "config.json", directory / "weights.mpk"
        if not config_file.is_file() or not weights_file.is_file():
            raise CheckpointError(
                f"{directory} is not a checkpoint: it needs config.json and weights.mpk"
            )
        try:
            config = json.loads(config_file.read_text())
        except ValueError as e:
            raise CheckpointError(f"{config_file} is not valid JSON: {e}") from None
        if "domain_size" not in config:
            raise CheckpointError(f"{config_file} has no domain_size")
        return cls(directory, config)

    def __fspath__(self) -> str:
        return str(self.path)


def load(path: PathLike) -> Checkpoint:
    """Opens a checkpoint written by :func:`fit`. See :meth:`Checkpoint.load`."""
    return Checkpoint.load(path)


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------


def _domain_size(models: Iterable[Model]) -> int:
    largest = 0
    for model in models:
        for var in model.variables:
            for value in model.domain(var):
                if value < 0:
                    raise ValueError(
                        "domain_size='auto' needs non-negative values; "
                        "shift the values or give domain_size"
                    )
                largest = max(largest, value)
    return largest + 1


def _native_network(net: Network, domain_size: int) -> Any:
    return _core.PyConsFormerConfig(
        domain_size=domain_size,
        embedding_size=net.embedding_size,
        hidden_size=net.hidden_size,
        num_heads=net.num_heads,
        expand_size=net.expand_size,
        mask_fraction=net.mask_fraction,
        tau=net.tau,
        num_layers=net.num_layers,
        drop_out=net.dropout,
        bias=net.bias,
        positional_encoding_dimensions=net.positional_encoding_dimensions,
        learn_attention_mask=net.learned_attention_mask,
    )


def _native_optimiser(o: Optimiser) -> Any:
    if isinstance(o, Adam):
        return _core.PyOptimiser.Adam(
            lr=o.lr,
            beta_1=o.beta_1,
            beta_2=o.beta_2,
            epsilon=o.epsilon,
            amsgrad=o.amsgrad,
            weight_decay=o.weight_decay,
            grad_clip_norm=o.grad_clip_norm,
            grad_clip_value=o.grad_clip_value,
        )
    if isinstance(o, AdamW):
        return _core.PyOptimiser.AdamW(
            lr=o.lr,
            beta_1=o.beta_1,
            beta_2=o.beta_2,
            epsilon=o.epsilon,
            amsgrad=o.amsgrad,
            weight_decay=o.weight_decay,
            grad_clip_norm=o.grad_clip_norm,
            grad_clip_value=o.grad_clip_value,
        )
    if isinstance(o, SGD):
        return _core.PyOptimiser.Sgd(
            lr=o.lr,
            momentum=o.momentum,
            dampening=o.dampening,
            nesterov=o.nesterov,
            weight_decay=o.weight_decay,
            grad_clip_norm=o.grad_clip_norm,
            grad_clip_value=o.grad_clip_value,
        )
    raise TypeError(f"optimiser must be Adam, AdamW or SGD, got {o!r}")


def _native_training(t: Training) -> Any:
    return _core.PyTrainingConfig(
        optimiser=_native_optimiser(t.optimiser),
        num_epochs=t.epochs,
        batch_size=t.batch_size,
        validation_interval=t.validation_interval,
        early_stopping_patience=t.early_stopping_patience,
        early_stopping_min_delta=t.early_stopping_min_delta,
        destroy_kind=_DESTROY[t.destroy],
        initial_weights=None if t.initial_weights is None else str(Path(t.initial_weights)),
    )


def fit(
    data: Union[Dataset, Sequence[Model]],
    recipe: Union[str, Recipe] = "consformer-mdd",
    *,
    checkpoint_dir: PathLike,
    seed: Optional[int] = None,
    **settings: Any,
) -> Checkpoint:
    """Trains a network and keeps the best one found in ``checkpoint_dir``.

    The network with the lowest loss on the validation set is written
    to ``checkpoint_dir`` during training, so an interrupted run still leaves a usable one.

    Args:
        data: A :class:`Dataset`, or a list of problems, which is split 80/20 into training and
            validation sets (use :meth:`Dataset.split` for control over the split).
        recipe: The name of a recipe in :data:`RECIPES`, or a :class:`Recipe`.
        checkpoint_dir: Where the network is written. Created if missing.
        seed: Seeds the library's randomness (see :func:`pyaicad.set_seed`) before training, and
            the split of a plain list.
        **settings: Changes to the recipe, by name, as in :meth:`Recipe.replace`:
            ``fit(data, "consformer", checkpoint_dir="ckpt", epochs=50, embedding_size=64)``.

    Returns:
        The :class:`Checkpoint` holding the best network.
    """
    recipe = Recipe.named(recipe) if isinstance(recipe, str) else recipe
    if settings:
        recipe = recipe.replace(**settings)
    if not isinstance(data, Dataset):
        data = Dataset.split(data, (0.8, 0.2), seed=seed)

    t = recipe.training
    if t.validation_interval < 1 or t.epochs < 1:
        raise ValueError("epochs and validation_interval must be at least 1")
    if t.validation_interval > t.epochs:
        raise ValueError(
            f"validation_interval ({t.validation_interval}) is larger than epochs ({t.epochs}): "
            "the network would never be validated, so none would be kept"
        )

    domain_size = recipe.network.domain_size
    if domain_size == "auto":
        domain_size = _domain_size(data)
    elif not isinstance(domain_size, int) or domain_size < 1:
        raise ValueError(f"domain_size must be 'auto' or a positive integer, got {domain_size!r}")

    if seed is not None:
        set_seed(seed)

    directory = Path(checkpoint_dir)
    network = _native_network(recipe.network, domain_size)
    training = _native_training(recipe.training)
    train = [m._native_problem() for m in data.train]
    validation = [m._native_problem() for m in data.validation]

    if recipe.loss == "classical":
        _core.train_consformer(train, network, training, str(directory), validation=validation)
    else:
        _core.train_consformer_mdd(
            train,
            network,
            training,
            str(directory),
            pyordering=_ordering(recipe.order, data.train[0]),
            pymerge=_merge_heuristic(recipe.merge),
            validation=validation,
        )
    return Checkpoint.load(directory)
