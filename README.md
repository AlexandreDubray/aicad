# AICAD

![](assets/logo.png#gh-light-mode-only)
![](assets/logo_rev.png#gh-dark-mode-only)

MDD compilation for constraint satisfaction with search strategies built on top of it.
Compile a set of constraints into a decision diagram, query it, or solve it, with or without a
trained neural network.

## Python

The main entry point for using the library is the `pyaicad` Python package.

```python
import pyaicad as ac

model = ac.Model()
x = model.int_vars((4, 4), domain=range(4))              # a 4 x 4 grid of variables
model.add(ac.all_different(row) for row in x)             # latin square: rows ...
model.add(ac.all_different(col) for col in zip(*x))       # ... and columns

mdd = ac.Compiler(model).compile()                        # exact diagram; compile(width=8) relaxes
mdd.count()                                               # 576
mdd.solution()                                            # one latin square
```

- **Model**: `Model`, `all_different`, `not_equal`, `equal`, `among`, `at_least`, `sum_equals`,
  `gcc`, `regular`.
- **Compile**: `Compiler(model, order=..., merge=...)` hands out `MDD`s, exact or relaxed, that can be
  counted, sampled, refined, frozen and drawn. This is useful to integrate MDD-based reasoning
  inside other projects.
- **Search**: `local_search(problems, search, limits)` with `BeliefPropagationSearch(...)`, or with
  `NeuralPriorSearch(...)` / `NeuralPriorBPSearch(...)` (full build), gives a `Result` (status,
  assignment, iterations, runtime). A local search finds solutions; it cannot prove that there are none.
- **Learn**: `pyaicad.learning` trains the networks used by the neural searches: `Dataset` (a train/validation split), `fit` with a named
  recipe (`"consformer"`, `"consformer-mdd"`), `load`.

The documentation (`docs/`) has a Python guide and a Rust guide that go through all of this with small examples that are run when it is built:
`pip install -e ".[docs]"` then `sphinx-build docs docs/_build`.

Build the extension with `maturin develop` and run the tests with `pytest`. That is the slim build: the
compiler, sampling and the belief-propagation search. Training and the neural searches need the `learning`
feature (it pulls in burn, which is slow to build): `maturin develop --features python,learning`.
`pyaicad.HAS_LEARNING` tells which one is installed, and the learning calls raise `FeatureUnavailableError`
on the slim build.

## Rust

The Rust crate is the core the Python package is built on: `modelling`, `constraints`, `mdd` and
`sampling`. It is public so that other solvers can embed the MDD compiler, but it is
**experimental** and may change between minor versions. The modules above it (`learning`, `local_search`,
the bindings) are private.
