# aicad

`aicad` compiles constraint satisfaction problems into decision diagrams (MDDs) and searches for
solutions with the help of those diagrams. It is a Rust library, and a Python package, `pyaicad`,
built on it: everything that can be done from Python can be done from Rust. You describe a problem
once, as a model, and then

- **compile** it into a diagram, exact or relaxed, to count, sample, read solutions off it, or
  prove that there is none;
- **search** for solutions with a local search that repairs assignments with belief propagation on
  the diagrams, optionally guided by a trained network;
- **train** such a network from a set of problems (needs the learning feature).

The **Python guide** and the **Rust guide** go through these in the same order, with the same
examples: the Python ones are run when the documentation is built, the Rust ones are compiled and
run by `cargo test`. The [Python API reference](api.md) lists everything of the Python package with
its signature; the Rust API is on docs.rs. People who want to change the library, or to plug their
own constraint, destroy operator, search or network into it, should read *Contributing*.

```{toctree}
:maxdepth: 2
:caption: Getting started

install
```

```{toctree}
:maxdepth: 2
:caption: Python guide

python-guide/modelling
python-guide/compiling
python-guide/relaxing
python-guide/local-search
python-guide/learning
```

```{toctree}
:maxdepth: 2
:caption: Rust guide

rust-guide/modelling
rust-guide/compiling
rust-guide/relaxing
rust-guide/local-search
rust-guide/learning
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

```{toctree}
:maxdepth: 2
:caption: Contributing

contributing/index
```
