---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Installation

`aicad` is a Rust crate, and `pyaicad` is the Python package built from it. Install the one you need.

## Python

```
pip install pyaicad
```

`pyaicad` comes in two builds of the same package.

The **slim build** has the modelling, the compiler and the diagrams, and the local search with
belief propagation (`BeliefPropagationSearch`). It has no heavy dependency.

The **full build** adds training and the neural searches (`NeuralPriorSearch`,
`NeuralPriorBPSearch`, `pyaicad.learning`). It embeds the deep-learning framework
[burn](https://burn.dev), which takes a long time to compile.

`pyaicad.HAS_LEARNING` tells which one is installed. Asking the slim build for something that needs
the full one raises `FeatureUnavailableError`, with a message saying so.

```{code-cell} python
import pyaicad as ac

print(ac.__version__, "- learning included:", ac.HAS_LEARNING)
```

## Rust

```
cargo add aicad                       # the compiler, belief propagation, local search
cargo add aicad --features learning   # and the neural networks, their training and the neural searches
```

The `learning` feature pulls in [burn](https://burn.dev), which takes a long time to compile, and is
off by default, as for the Python wheel. The code that needs it is gated with
`#[cfg(feature = "learning")]`, and the Rust guide marks the chapter that needs it.

## Building from source

From a checkout, with [maturin](https://www.maturin.rs):

```
maturin develop                                 # slim build
maturin develop --features python,learning      # full build
```

A `--features` given on the command line replaces the list in `pyproject.toml`, so `python` has to
be named again.
