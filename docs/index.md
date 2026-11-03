# pyaicad

`pyaicad` compiles constraint satisfaction problems into decision diagrams (MDDs) and searches for
solutions with the help of those diagrams. You describe a problem once, as a model, and then

- **compile** it into a diagram, exact or relaxed, to count, sample, read solutions off it, or
  prove that there is none;
- **search** for solutions with a local search that repairs assignments with belief propagation on
  the diagrams, optionally guided by a trained network;
- **train** such a network from a set of problems (full build only).

The guide goes through these in order, with small examples that are run when the documentation is
built. The [API reference](api.md) lists everything with its signature.

```{toctree}
:maxdepth: 2
:caption: Guide

install
guide/modelling
guide/compiling
guide/relaxing
guide/local-search
guide/learning
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```
